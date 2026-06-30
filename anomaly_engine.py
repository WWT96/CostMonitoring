from __future__ import annotations

import json as _json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import harness
from data_ingestion import detect_price_column
from storage_service import build_record_keys


_EXPERT_WEIGHT = 80
_TIME_DECAY_FULL_WEIGHT_DAYS = 183
_TIME_DECAY_DECAY_DAYS = 365.0
DEFAULT_DECAY_ALPHA = 1.0
DEFAULT_GAP_K = 4.0
DEFAULT_BASELINE_QUANTILE = 0.5
DGB_KDE_GRID_MIN_POINTS = 100
DGB_KDE_GRID_MAX_POINTS = 300
DGB_KDE_GRID_POINTS_PER_UNIQUE = 2
DGB_PARALLEL_MIN_GROUPS = 8
DGB_PARALLEL_MIN_ROWS = 200000
DGB_PARALLEL_MIN_AVG_GROUP_ROWS = 1000
DGB_PROCESS_POOL_ENV = "COST_MONITOR_ENABLE_PROCESS_POOL"
TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}
ACCEPTED_RING_ROLES = {"主邻居圈", "次邻居圈"}

RAW_RESULT_COLUMNS = [
    "_record_key",
    "物料编码",
    "物料名称",
    "适用车系",
    "工厂",
    "备件简称",
    "实际成本",
    "价格有效于",
    "样本量",
    "预测值",
    "合理下限",
    "合理上限",
    "偏离数值",
    "偏离比例",
    "status",
    "圈层编号",
    "圈层角色",
    "圈层置信度",
    "多圈合理区间",
]

WEIGHTED_RESULT_COLUMNS = [
    *RAW_RESULT_COLUMNS,
    "专家校准",
    "判定依据",
]


def _resolve_target_column(df: pd.DataFrame, target_column: str) -> str:
    if target_column and target_column in df.columns:
        return target_column
    detected_column = detect_price_column(df.columns)
    if detected_column:
        return detected_column
    raise ValueError(f"找不到目标指标列: {target_column}")


def calculate_recency_weight_series(
    date_values: Any,
    decay_alpha: float = DEFAULT_DECAY_ALPHA,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    dates = pd.to_datetime(pd.Series(date_values), errors="coerce")
    anchor_date = pd.Timestamp(reference_date).normalize() if reference_date is not None else pd.Timestamp.today().normalize()
    normalized_dates = dates.dt.normalize()
    age_days = (anchor_date - normalized_dates).dt.days.clip(lower=0)
    decay_days = (age_days - _TIME_DECAY_FULL_WEIGHT_DAYS).clip(lower=0)
    alpha = float(decay_alpha) if pd.notna(decay_alpha) else DEFAULT_DECAY_ALPHA
    if alpha <= 0:
        alpha = DEFAULT_DECAY_ALPHA
    weights = np.exp(-((alpha * decay_days) / _TIME_DECAY_DECAY_DAYS))
    weights = np.where(normalized_dates.isna(), 1.0, weights)
    return pd.Series(weights, index=dates.index, dtype=float)


def _prepare_weighted_inputs(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clean_values = np.asarray(values, dtype=float)
    clean_weights = np.asarray(weights, dtype=float)
    valid_mask = np.isfinite(clean_values) & np.isfinite(clean_weights) & (clean_weights > 0)
    return clean_values[valid_mask], clean_weights[valid_mask]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float, fallback: float) -> float:
    clean_values, clean_weights = _prepare_weighted_inputs(values, weights)
    if clean_values.size == 0:
        return float(fallback)

    order = np.argsort(clean_values)
    sorted_values = clean_values[order]
    sorted_weights = clean_weights[order]
    cumulative_weights = np.cumsum(sorted_weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative_weights[-1])
    cutoff = max(cutoff, 0.0)
    target_idx = int(np.searchsorted(cumulative_weights, cutoff, side="left"))
    target_idx = min(target_idx, sorted_values.size - 1)
    return float(sorted_values[target_idx])


def _weighted_median(values: np.ndarray, weights: np.ndarray, fallback: float) -> float:
    return _weighted_quantile(values, weights, 0.5, fallback)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    clean_values, clean_weights = _prepare_weighted_inputs(values, weights)
    if clean_values.size <= 1:
        return 0.0
    weighted_mean = float(np.average(clean_values, weights=clean_weights))
    variance = float(np.average((clean_values - weighted_mean) ** 2, weights=clean_weights))
    return float(np.sqrt(max(variance, 0.0)))


def _collapse_weighted_values(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_values, clean_weights = _prepare_weighted_inputs(values, weights)
    if clean_values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)

    uniq_vals, inverse = np.unique(clean_values, return_inverse=True)
    uniq_weight_sums = np.bincount(inverse, weights=clean_weights, minlength=uniq_vals.size)
    uniq_counts = np.bincount(inverse, minlength=uniq_vals.size)
    return uniq_vals.astype(float), uniq_weight_sums.astype(float), uniq_counts.astype(int)


def _rename_metric_anomaly_columns(
    result_df: pd.DataFrame,
    *,
    value_label: str,
    baseline_label: str,
    deviation_label: str,
    date_label: str,
) -> pd.DataFrame:
    rename_map = {}
    if value_label != "实际成本":
        rename_map["实际成本"] = value_label
    if baseline_label != "预测值":
        rename_map["预测值"] = baseline_label
    if deviation_label != "偏离数值":
        rename_map["偏离数值"] = deviation_label
    if date_label != "价格有效于":
        rename_map["价格有效于"] = date_label
    if not rename_map:
        return result_df
    return result_df.rename(columns=rename_map)


def detect_dgb_anomalies(
    df: pd.DataFrame,
    target_column: str = "成本金额",
    *,
    decay_alpha: float = DEFAULT_DECAY_ALPHA,
    gap_k: float = DEFAULT_GAP_K,
    baseline_quantile: float = DEFAULT_BASELINE_QUANTILE,
    value_label: str | None = None,
    baseline_label: str = "预测值",
    deviation_label: str = "偏离数值",
    date_label: str = "价格有效于",
) -> pd.DataFrame:
    resolved_target_column = _resolve_target_column(df, target_column)
    result_df = detect_cost_anomalies(
        df,
        resolved_target_column,
        decay_alpha=decay_alpha,
        gap_k=gap_k,
        baseline_quantile=baseline_quantile,
    )
    return _rename_metric_anomaly_columns(
        result_df,
        value_label=value_label or resolved_target_column,
        baseline_label=baseline_label,
        deviation_label=deviation_label,
        date_label=date_label,
    )


def detect_dgb_anomalies_weighted(
    df: pd.DataFrame,
    target_column: str = "成本金额",
    expert_labels_tuple: tuple = (),
    *,
    sigma_multiplier: float = 1.0,
    expert_weight_override: int = 0,
    decay_alpha: float = DEFAULT_DECAY_ALPHA,
    gap_k: float = DEFAULT_GAP_K,
    baseline_quantile: float = DEFAULT_BASELINE_QUANTILE,
    skills_overrides_json: str = "",
    value_label: str | None = None,
    baseline_label: str = "预测值",
    deviation_label: str = "偏离数值",
    date_label: str = "价格有效于",
) -> pd.DataFrame:
    resolved_target_column = _resolve_target_column(df, target_column)
    result_df = detect_cost_anomalies_weighted(
        df,
        resolved_target_column,
        expert_labels_tuple,
        sigma_multiplier=sigma_multiplier,
        expert_weight_override=expert_weight_override,
        decay_alpha=decay_alpha,
        gap_k=gap_k,
        baseline_quantile=baseline_quantile,
        skills_overrides_json=skills_overrides_json,
    )
    return _rename_metric_anomaly_columns(
        result_df,
        value_label=value_label or resolved_target_column,
        baseline_label=baseline_label,
        deviation_label=deviation_label,
        date_label=date_label,
    )


def _normalize_match_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).strip().lower()


def _normalize_short_name_core(value: Any) -> str:
    text = _normalize_match_text(value)
    if not text:
        return ""
    text = re.sub(r"(总成件|总成|组件|模块|套件)$", "", text)
    core = re.sub(r"[前后左右上下内外高中低大小新老轻重]", "", text)
    return core or text


def _common_suffix_length(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    count = 0
    for idx in range(1, max_len + 1):
        if left[-idx] != right[-idx]:
            break
        count += 1
    return count


def _short_name_similarity(left: Any, right: Any) -> Tuple[float, bool]:
    left_text = _normalize_match_text(left)
    right_text = _normalize_match_text(right)
    if not left_text or not right_text:
        return 0.0, False
    if left_text == right_text:
        return 1.0, True

    left_core = _normalize_short_name_core(left_text)
    right_core = _normalize_short_name_core(right_text)
    sequence_ratio = SequenceMatcher(None, left_core, right_core).ratio()
    overlap = 0.0
    union_chars = set(left_core) | set(right_core)
    if union_chars:
        overlap = len(set(left_core) & set(right_core)) / len(union_chars)
    suffix_len = _common_suffix_length(left_core, right_core)
    suffix_ratio = suffix_len / max(len(left_core), len(right_core), 1)
    same_class = (
        left_core == right_core
        or (left_core and left_core in right_core)
        or (right_core and right_core in left_core)
        or suffix_len >= 1
    )
    return max(sequence_ratio, overlap, suffix_ratio), same_class


def _infer_rule_direction(rule_content: Any) -> str:
    text = str(rule_content or "")
    if re.search(r"溢价|偏高|偏贵|高价|高性能|升级|模摊|材料更高|工艺更高", text):
        return "high"
    if re.search(r"偏低|偏便宜|低价|折价|降配|简配|替代料|返利", text):
        return "low"
    return "neutral"


def _build_core_context_lookup() -> Dict[str, Dict[str, str]]:
    core_df, _, _ = harness.execute_action("load_core_cost_records")
    if core_df is None or core_df.empty:
        return {}
    lookup_df = core_df.copy()
    lookup_df["价格有效于"] = pd.to_datetime(lookup_df.get("monitor_date"), errors="coerce")
    lookup_df["实际成本"] = pd.to_numeric(lookup_df.get("成本"), errors="coerce")
    lookup_df["_record_key"] = build_record_keys(lookup_df)
    lookup_df["一级总成供应商代码"] = lookup_df.get("一级总成供应商代码", "").fillna("").astype(str)
    lookup_df["一级总成供应商名称"] = lookup_df.get("一级总成供应商名称", "").fillna("").astype(str)
    lookup_df["适用车系"] = lookup_df.get("适用车系", "").fillna("").astype(str)
    lookup_df["备件简称"] = lookup_df.get("备件简称", "").fillna("").astype(str)
    lookup_df = lookup_df.drop_duplicates(subset=["_record_key"], keep="last")
    return {
        row["_record_key"]: {
            "supplier_code": row.get("一级总成供应商代码", ""),
            "supplier_name": row.get("一级总成供应商名称", ""),
            "vehicle_series": row.get("适用车系", ""),
            "short_name": row.get("备件简称", ""),
        }
        for row in lookup_df.to_dict(orient="records")
    }


def _resolve_anomaly_context(anomaly_record: Any, core_context_lookup: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
    core_context_lookup = core_context_lookup or {}
    if hasattr(anomaly_record, "to_dict"):
        data = anomaly_record.to_dict()
    else:
        data = dict(anomaly_record)

    record_key = str(data.get("_record_key") or "")
    fallback = core_context_lookup.get(record_key, {})
    actual_price = pd.to_numeric(pd.Series([data.get("实际成本")]), errors="coerce").iloc[0]
    baseline_price = pd.to_numeric(pd.Series([data.get("预测值")]), errors="coerce").iloc[0]
    deviation_ratio = pd.to_numeric(pd.Series([data.get("偏离比例")]), errors="coerce").iloc[0]
    if pd.isna(deviation_ratio) and pd.notna(actual_price) and pd.notna(baseline_price) and float(baseline_price) != 0.0:
        deviation_ratio = (float(actual_price) - float(baseline_price)) / float(baseline_price)

    return {
        "record_key": record_key,
        "status": str(data.get("status") or ""),
        "short_name": str(data.get("备件简称") or fallback.get("short_name") or "").strip(),
        "vehicle_series": str(data.get("适用车系") or fallback.get("vehicle_series") or "").strip(),
        "supplier_code": str(
            data.get("供应商代码")
            or data.get("一级总成供应商代码")
            or fallback.get("supplier_code")
            or ""
        ).strip(),
        "supplier_name": str(
            data.get("供应商名称")
            or data.get("一级总成供应商名称")
            or fallback.get("supplier_name")
            or ""
        ).strip(),
        "actual_price": None if pd.isna(actual_price) else float(actual_price),
        "baseline_price": None if pd.isna(baseline_price) else float(baseline_price),
        "deviation_ratio": None if pd.isna(deviation_ratio) else float(deviation_ratio),
    }


def _score_knowledge_rule_match(context: Dict[str, Any], rule_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    short_name_score, same_class = _short_name_similarity(context.get("short_name"), rule_row.get("short_name"))
    if short_name_score <= 0:
        return None

    context_supplier = _normalize_match_text(context.get("supplier_code"))
    context_vehicle = _normalize_match_text(context.get("vehicle_series"))
    rule_supplier = _normalize_match_text(rule_row.get("supplier_code"))
    rule_vehicle = _normalize_match_text(rule_row.get("vehicle_series"))
    rule_confidence = float(pd.to_numeric(pd.Series([rule_row.get("confidence_score")]), errors="coerce").fillna(0.0).iloc[0])
    direction_penalty = 1.0
    rule_direction = _infer_rule_direction(rule_row.get("rule_content"))
    status_text = str(context.get("status") or "")
    if "偏高" in status_text and rule_direction == "low":
        direction_penalty = 0.35
    elif "偏低" in status_text and rule_direction == "high":
        direction_penalty = 0.35

    match_type = ""
    base_score = 0.0
    if context_supplier and rule_supplier and context_supplier == rule_supplier:
        if short_name_score >= 0.999:
            match_type = "供应商精准匹配"
            base_score = 1.0
        elif same_class and short_name_score >= 0.34:
            match_type = "同供应商同类继承"
            base_score = 0.86
    elif context_vehicle and rule_vehicle and context_vehicle == rule_vehicle:
        if short_name_score >= 0.999:
            match_type = "车系精准匹配"
            base_score = 0.82
        elif same_class and short_name_score >= 0.34:
            match_type = "同车系同类继承"
            base_score = 0.72

    if not match_type:
        return None

    total_score = (base_score + short_name_score * 0.08 + rule_confidence * 0.06) * direction_penalty
    return {
        "match_type": match_type,
        "score": total_score,
        "short_name_score": short_name_score,
        "rule_confidence": rule_confidence,
        "rule_id": str(rule_row.get("rule_id") or ""),
        "rule_short_name": str(rule_row.get("short_name") or ""),
        "rule_content": str(rule_row.get("rule_content") or ""),
        "material_code": str(rule_row.get("material_code") or ""),
        "material_name": str(rule_row.get("material_name") or ""),
        "supplier_code": str(rule_row.get("supplier_code") or ""),
        "supplier_name": str(rule_row.get("supplier_name") or ""),
        "vehicle_series": str(rule_row.get("vehicle_series") or ""),
    }


def _format_inferred_reason(context: Dict[str, Any], match_detail: Dict[str, Any]) -> str:
    material_code = str(match_detail.get("material_code") or "").strip()
    material_name = str(match_detail.get("material_name") or "").strip()
    reference_material = material_code
    if material_name:
        reference_material = f"{reference_material}（{material_name}）" if reference_material else material_name
    if not reference_material:
        reference_material = "历史标注物料"

    same_scope_parts = []
    if str(context.get("vehicle_series") or "").strip() and str(match_detail.get("vehicle_series") or "").strip():
        same_scope_parts.append("同车系")
    if str(context.get("supplier_code") or "").strip() and str(match_detail.get("supplier_code") or "").strip():
        same_scope_parts.append("同供应商")
    if not same_scope_parts:
        same_scope_parts.append("同类备件")

    status_text = str(context.get("status") or "")
    direction_text = "偏低" if "偏低" in status_text else "偏高"
    rule_content = str(match_detail.get("rule_content") or "").strip() or "历史专家批注原因相近"

    return (
        f"[系统预测] 根据专家批注历史，本物料与{reference_material}属于{'、'.join(same_scope_parts)}；"
        f"该物料批注因{rule_content}导致成本{direction_text}，建议参考该物料的异常原因综合评估。"
    )


def infer_anomaly_reason(
    anomaly_record: Any,
    knowledge_df: Optional[pd.DataFrame] = None,
    core_context_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    detail = _infer_anomaly_reason_detail(
        anomaly_record,
        knowledge_df=knowledge_df,
        core_context_lookup=core_context_lookup,
    )
    return detail.get("analysis", "")


def _infer_anomaly_reason_detail(
    anomaly_record: Any,
    knowledge_df: Optional[pd.DataFrame] = None,
    core_context_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    empty_result = {
        "analysis": "",
        "rule_id": "",
        "match_type": "",
        "rule_content": "",
        "rule_short_name": "",
        "confidence_score": None,
    }
    if knowledge_df is None:
        knowledge_df = harness.execute_action("load_expert_knowledge_base")
    if knowledge_df is None or knowledge_df.empty:
        return empty_result

    context = _resolve_anomaly_context(anomaly_record, core_context_lookup=core_context_lookup)
    if not context.get("short_name"):
        return empty_result
    status_text = str(context.get("status") or "")
    if not re.search(r"异常偏高|异常偏低|严重异常偏低", status_text):
        return empty_result

    candidates: List[Dict[str, Any]] = []
    for rule_row in knowledge_df.to_dict(orient="records"):
        scored = _score_knowledge_rule_match(context, rule_row)
        if scored:
            candidates.append(scored)
    if not candidates:
        return empty_result

    best = max(candidates, key=lambda item: (item["score"], item["rule_confidence"]))
    if best["score"] < 0.72:
        return empty_result
    return {
        "analysis": _format_inferred_reason(context, best),
        "rule_id": best["rule_id"],
        "match_type": best["match_type"],
        "rule_content": best["rule_content"],
        "rule_short_name": best["rule_short_name"],
        "confidence_score": best["rule_confidence"],
    }


def enrich_anomaly_with_inferred_reasons(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df is None or result_df.empty:
        enriched = pd.DataFrame() if result_df is None else result_df.copy()
    else:
        enriched = result_df.copy()

    public_column = "AI 辅助分析"
    internal_columns = ["_ai_rule_id", "_ai_match_scope", "_ai_rule_short_name", "_ai_reference_content", "_ai_confidence_score"]
    if enriched.empty:
        for column_name in [public_column] + internal_columns:
            enriched[column_name] = pd.Series(dtype="string")
        return enriched

    knowledge_df = harness.execute_action("load_expert_knowledge_base")
    if knowledge_df.empty:
        enriched[public_column] = ""
        for column_name in internal_columns:
            enriched[column_name] = ""
        return enriched

    core_lookup = _build_core_context_lookup()
    analyses: List[str] = []
    rule_ids: List[str] = []
    match_scopes: List[str] = []
    rule_short_names: List[str] = []
    rule_contents: List[str] = []
    confidence_scores: List[Any] = []
    for _, row in enriched.iterrows():
        status_text = str(row.get("status") or "")
        if not re.search(r"异常偏高|异常偏低|严重异常偏低", status_text):
            detail = {
                "analysis": "",
                "rule_id": "",
                "match_type": "",
                "rule_content": "",
                "rule_short_name": "",
                "confidence_score": None,
            }
        else:
            detail = _infer_anomaly_reason_detail(row, knowledge_df=knowledge_df, core_context_lookup=core_lookup)
        analyses.append(detail.get("analysis", ""))
        rule_ids.append(detail.get("rule_id", ""))
        match_scopes.append(detail.get("match_type", ""))
        rule_short_names.append(detail.get("rule_short_name", ""))
        rule_contents.append(detail.get("rule_content", ""))
        confidence_scores.append(detail.get("confidence_score"))

    enriched[public_column] = analyses
    enriched["_ai_rule_id"] = rule_ids
    enriched["_ai_match_scope"] = match_scopes
    enriched["_ai_rule_short_name"] = rule_short_names
    enriched["_ai_reference_content"] = rule_contents
    enriched["_ai_confidence_score"] = confidence_scores
    enriched[public_column] = enriched[public_column].fillna("")
    for column_name in ["_ai_rule_id", "_ai_match_scope", "_ai_rule_short_name", "_ai_reference_content"]:
        enriched[column_name] = enriched[column_name].fillna("")
    return enriched


def _reanchor_cluster_price(
    cluster_samples: np.ndarray,
    cluster_weights: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    fallback: float,
) -> float:
    anchor = _weighted_median(cluster_samples, cluster_weights, fallback) if cluster_samples.size else float(fallback)
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    return float(np.clip(anchor, lower_bound, upper_bound))


def _ensure_baseline_within_bounds(lower_bound: float, baseline: float, upper_bound: float) -> Tuple[float, float, float]:
    lower = float(lower_bound) if pd.notna(lower_bound) else 0.0
    upper = float(upper_bound) if pd.notna(upper_bound) else lower
    center = float(baseline) if pd.notna(baseline) else (lower + upper) / 2.0
    if lower > upper:
        lower, upper = upper, lower

    lower = min(lower, center)
    upper = max(upper, center)
    lower = max(0.0, lower)
    center = float(np.clip(center, lower, upper))
    return lower, center, upper


def _normalize_positive_float(value: Any, default: float) -> float:
    normalized = float(value) if pd.notna(value) else default
    if normalized <= 0:
        return default
    return normalized


def _elbow_index(seq: np.ndarray) -> int:
    if seq.size <= 1:
        return 0
    x = np.arange(seq.size, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    y_min, y_max = float(np.min(seq)), float(np.max(seq))
    if y_max > y_min:
        y_norm = (seq - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(seq, dtype=float)
    line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    dist = y_norm - line
    return int(np.argmax(dist))


def _build_components(values: np.ndarray, break_gap_idx: set[int]) -> List[Tuple[int, int]]:
    if values.size == 0:
        return []

    components: List[Tuple[int, int]] = []
    start = 0
    for idx in range(values.size - 1):
        if idx in break_gap_idx:
            components.append((start, idx))
            start = idx + 1
    components.append((start, values.size - 1))
    return components


def _normalize_baseline_quantile(value: Any) -> float:
    normalized = float(value) if pd.notna(value) else DEFAULT_BASELINE_QUANTILE
    return float(np.clip(normalized, 0.4, 0.6))


def _secondary_ring_min_count(sample_count: int) -> int:
    n = max(0, int(sample_count))
    if n < 50:
        return max(3, int(np.ceil(0.12 * n)))
    if n < 200:
        return max(5, int(np.ceil(0.08 * n)))
    return max(8, int(np.ceil(0.04 * n)))


def _select_primary_component_index(
    kde_peak_component_idx: int,
    comp_raw_counts: np.ndarray,
    comp_density: np.ndarray,
    comp_has_expert_anchor: List[bool],
) -> int:
    if comp_raw_counts.size == 0:
        return int(kde_peak_component_idx)

    total_raw = float(np.sum(comp_raw_counts))
    max_density = float(np.max(comp_density)) if comp_density.size else 0.0
    support_score = comp_raw_counts / max(total_raw, np.finfo(float).eps)
    density_score = comp_density / max(max_density, np.finfo(float).eps) if comp_density.size else np.zeros_like(comp_raw_counts)
    peak_bonus = np.zeros(comp_raw_counts.size, dtype=float)
    if 0 <= int(kde_peak_component_idx) < peak_bonus.size:
        peak_bonus[int(kde_peak_component_idx)] = 1.0
    anchor_bonus = np.asarray([1.0 if has_anchor else 0.0 for has_anchor in comp_has_expert_anchor], dtype=float)
    primary_score = 0.72 * support_score + 0.18 * density_score + 0.06 * anchor_bonus + 0.04 * peak_bonus
    return int(np.argmax(primary_score))


def _ring_confidence(
    *,
    raw_count: float,
    sample_count: int,
    min_count: int,
    density: float,
    max_density: float,
    is_main: bool,
    has_expert_anchor: bool,
) -> float:
    if is_main:
        return 1.0
    support_score = min(1.0, float(raw_count) / max(float(min_count), 1.0))
    density_score = min(1.0, float(density) / max(float(max_density), np.finfo(float).eps))
    prevalence_score = min(1.0, float(raw_count) / max(float(sample_count), 1.0) * 6.0)
    anchor_score = 0.2 if has_expert_anchor else 0.0
    return round(float(min(1.0, 0.45 * support_score + 0.25 * density_score + 0.25 * prevalence_score + anchor_score)), 4)


def _format_ring_intervals(rings: List[Dict[str, Any]]) -> str:
    public_rings = []
    for ring in rings:
        public_rings.append(
            {
                "圈层编号": int(ring["id"]),
                "圈层角色": str(ring["role"]),
                "合理下限": round(float(ring["lower"]), 4),
                "合理上限": round(float(ring["upper"]), 4),
                "预测值": round(float(ring["baseline"]), 4),
                "样本量": int(ring["raw_count"]),
                "加权样本量": round(float(ring["weighted_count"]), 4),
                "圈层置信度": round(float(ring["confidence"]), 4),
                "专家锚点": bool(ring["has_expert_anchor"]),
            }
        )
    return _json.dumps(public_rings, ensure_ascii=False, separators=(",", ":"))


def _coerce_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def parse_accepted_ring_intervals(payload: Any) -> List[Dict[str, Any]]:
    """Parse public ring payload and keep only main/secondary normal intervals."""
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        try:
            raw_rings = _json.loads(text)
        except (TypeError, ValueError):
            return []
    elif isinstance(payload, list):
        raw_rings = payload
    else:
        return []

    intervals: List[Dict[str, Any]] = []
    for raw_ring in raw_rings:
        if not isinstance(raw_ring, dict):
            continue
        role = str(raw_ring.get("圈层角色", raw_ring.get("role", "")) or "").strip()
        if role not in ACCEPTED_RING_ROLES:
            continue
        lower = _coerce_finite_float(raw_ring.get("合理下限", raw_ring.get("lower")))
        upper = _coerce_finite_float(raw_ring.get("合理上限", raw_ring.get("upper")))
        if lower is None or upper is None:
            continue
        if lower > upper:
            lower, upper = upper, lower
        baseline = _coerce_finite_float(raw_ring.get("预测值", raw_ring.get("baseline")))
        if baseline is None:
            baseline = (lower + upper) / 2.0
        baseline = float(np.clip(baseline, lower, upper))
        ring_id = raw_ring.get("圈层编号", raw_ring.get("id", len(intervals) + 1))
        try:
            ring_id = int(ring_id)
        except (TypeError, ValueError):
            ring_id = len(intervals) + 1
        confidence = _coerce_finite_float(raw_ring.get("圈层置信度", raw_ring.get("confidence")))
        intervals.append(
            {
                "圈层编号": ring_id,
                "圈层角色": role,
                "合理下限": float(lower),
                "合理上限": float(upper),
                "预测值": float(baseline),
                "圈层置信度": float(confidence if confidence is not None else np.nan),
            }
        )

    role_order = {"主邻居圈": 0, "次邻居圈": 1}
    return sorted(intervals, key=lambda item: (float(item["合理下限"]), role_order.get(str(item["圈层角色"]), 9)))


def find_accepted_ring_for_value(value: Any, intervals: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    number = _coerce_finite_float(value)
    if number is None:
        return None
    for interval in intervals:
        lower = float(interval["合理下限"])
        upper = float(interval["合理上限"])
        if lower <= number <= upper:
            return interval
    return None


def apply_accepted_ring_union_status(
    result_df: pd.DataFrame,
    *,
    value_col: str = "实际成本",
    baseline_col: str = "预测值",
    deviation_col: str = "偏离数值",
    ratio_col: str = "偏离比例",
) -> pd.DataFrame:
    """Mark samples inside any main/secondary ring interval as normal.

    The helper intentionally does not bridge gaps between accepted intervals.
    """
    if result_df is None or result_df.empty or value_col not in result_df.columns:
        return result_df
    if "多圈合理区间" not in result_df.columns:
        return result_df

    data = result_df.copy()
    for column_name in [baseline_col, "合理下限", "合理上限", deviation_col, ratio_col, "圈层编号", "圈层角色", "圈层置信度"]:
        if column_name not in data.columns:
            data[column_name] = np.nan if column_name != "圈层角色" else ""

    for row_idx, row in data.iterrows():
        intervals = parse_accepted_ring_intervals(row.get("多圈合理区间"))
        ring = find_accepted_ring_for_value(row.get(value_col), intervals)
        if ring is None:
            continue
        value = _coerce_finite_float(row.get(value_col))
        baseline = float(ring["预测值"])
        data.at[row_idx, baseline_col] = baseline
        data.at[row_idx, "合理下限"] = float(ring["合理下限"])
        data.at[row_idx, "合理上限"] = float(ring["合理上限"])
        data.at[row_idx, "圈层编号"] = int(ring["圈层编号"])
        data.at[row_idx, "圈层角色"] = str(ring["圈层角色"])
        confidence = ring.get("圈层置信度")
        if _coerce_finite_float(confidence) is not None:
            data.at[row_idx, "圈层置信度"] = float(confidence)
        if value is not None:
            deviation = float(value) - baseline
            data.at[row_idx, deviation_col] = deviation
            data.at[row_idx, ratio_col] = np.nan if baseline == 0 else deviation / baseline
        data.at[row_idx, "status"] = f"正常（{ring['圈层角色']}）"
    return data


def apply_fixed_cost_skill_intervals(
    result_df: pd.DataFrame,
    intervals_payload: Any,
    *,
    value_col: str = "实际成本",
    baseline_col: str = "预测值",
    deviation_col: str = "偏离数值",
    ratio_col: str = "偏离比例",
    source: str = "",
) -> pd.DataFrame:
    intervals = parse_accepted_ring_intervals(intervals_payload)
    if result_df is None or result_df.empty or value_col not in result_df.columns or not intervals:
        return result_df

    data = result_df.copy()
    public_payload = _json.dumps(intervals, ensure_ascii=False, separators=(",", ":"))
    for column_name in [baseline_col, "合理下限", "合理上限", deviation_col, ratio_col, "圈层编号", "圈层角色", "圈层置信度", "多圈合理区间"]:
        if column_name not in data.columns:
            data[column_name] = np.nan if column_name != "圈层角色" else ""

    min_lower = min(float(interval["合理下限"]) for interval in intervals)
    max_upper = max(float(interval["合理上限"]) for interval in intervals)
    for row_idx, row in data.iterrows():
        value = _coerce_finite_float(row.get(value_col))
        data.at[row_idx, "多圈合理区间"] = public_payload
        if value is None:
            continue

        ring = find_accepted_ring_for_value(value, intervals)
        if ring is None:
            nearest_ring = min(
                intervals,
                key=lambda item: 0.0
                if float(item["合理下限"]) <= value <= float(item["合理上限"])
                else min(abs(value - float(item["合理下限"])), abs(value - float(item["合理上限"]))),
            )
            baseline = float(nearest_ring["预测值"])
            lower = float(nearest_ring["合理下限"])
            upper = float(nearest_ring["合理上限"])
            data.at[row_idx, baseline_col] = baseline
            data.at[row_idx, "合理下限"] = lower
            data.at[row_idx, "合理上限"] = upper
            data.at[row_idx, "圈层编号"] = 0
            data.at[row_idx, "圈层角色"] = "未采纳孤立圈"
            data.at[row_idx, "圈层置信度"] = 0.0
            if value < min_lower or value < baseline:
                data.at[row_idx, "status"] = "异常偏低" if value >= min_lower else "严重异常偏低"
            elif value > max_upper or value > baseline:
                data.at[row_idx, "status"] = "异常偏高"
            else:
                data.at[row_idx, "status"] = "异常偏高"
        else:
            baseline = float(ring["预测值"])
            data.at[row_idx, baseline_col] = baseline
            data.at[row_idx, "合理下限"] = float(ring["合理下限"])
            data.at[row_idx, "合理上限"] = float(ring["合理上限"])
            data.at[row_idx, "圈层编号"] = int(ring["圈层编号"])
            data.at[row_idx, "圈层角色"] = str(ring["圈层角色"])
            confidence = _coerce_finite_float(ring.get("圈层置信度"))
            data.at[row_idx, "圈层置信度"] = 1.0 if confidence is None else confidence
            data.at[row_idx, "status"] = f"正常（{ring['圈层角色']}）"

        deviation = value - float(data.at[row_idx, baseline_col])
        data.at[row_idx, deviation_col] = deviation
        data.at[row_idx, ratio_col] = np.nan if float(data.at[row_idx, baseline_col]) == 0 else deviation / float(data.at[row_idx, baseline_col])

    if source:
        data["判定依据"] = source
    return data


def _finalize_group_ring_status(g: pd.DataFrame, fixed_intervals: List[Dict[str, Any]], group_source: str) -> pd.DataFrame:
    if fixed_intervals:
        return apply_fixed_cost_skill_intervals(g, fixed_intervals, value_col="实际成本", source=group_source)
    return apply_accepted_ring_union_status(g, value_col="实际成本")


def _prepare_detection_input(
    df: pd.DataFrame,
    price_col: str,
    *,
    decay_alpha: float | None = None,
    include_record_keys: bool = False,
) -> pd.DataFrame:
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")

    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=["物料编码", "备件简称", "monitor_date", price_col])

    if "价格有效于" in data.columns and "monitor_date" in data.columns:
        data = data.drop(columns=["价格有效于"])
    data = data.rename(columns={price_col: "实际成本", "monitor_date": "价格有效于"})
    data["样本量"] = data.groupby("备件简称")["物料编码"].transform("size")

    if decay_alpha is not None:
        data["_recency_weight"] = calculate_recency_weight_series(
            data["价格有效于"],
            decay_alpha=decay_alpha,
        ).to_numpy(dtype=float)

    if include_record_keys:
        data["_record_key"] = build_record_keys(data)

    return data


def _should_use_parallel_detection(total_rows: int, group_count: int) -> bool:
    if str(os.environ.get(DGB_PROCESS_POOL_ENV, "") or "").strip().lower() not in TRUE_ENV_VALUES:
        return False
    available_cpus = os.cpu_count() or 1
    if available_cpus <= 1 or group_count < DGB_PARALLEL_MIN_GROUPS or total_rows < DGB_PARALLEL_MIN_ROWS:
        return False
    avg_group_rows = total_rows / max(group_count, 1)
    return avg_group_rows >= DGB_PARALLEL_MIN_AVG_GROUP_ROWS


def _detect_group_anomalies_worker(group_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    try:
        from sklearn.neighbors import KernelDensity, NearestNeighbors
    except Exception as exc:
        raise ImportError("缺少依赖 scikit-learn，请先安装：pip install scikit-learn") from exc

    g = group_df.copy()
    n = len(g)
    vals = g["实际成本"].to_numpy(dtype=float)
    weighted_mode = bool(config.get("weighted", False))
    group_gap_k = _normalize_positive_float(config.get("gap_k", DEFAULT_GAP_K), DEFAULT_GAP_K)
    group_baseline_quantile = _normalize_baseline_quantile(config.get("baseline_quantile", DEFAULT_BASELINE_QUANTILE))
    group_source = str(config.get("group_source", "") or "").strip()
    fixed_intervals = parse_accepted_ring_intervals(config.get("fixed_intervals"))

    if weighted_mode:
        group_sigma = _normalize_positive_float(config.get("sigma_multiplier", 1.0), 1.0)
        group_weight = int(config.get("expert_weight", _EXPERT_WEIGHT))
        group_decay_alpha = _normalize_positive_float(config.get("decay_alpha", DEFAULT_DECAY_ALPHA), DEFAULT_DECAY_ALPHA)
        recency_weights = calculate_recency_weight_series(
            g["价格有效于"],
            decay_alpha=group_decay_alpha,
        ).to_numpy(dtype=float)
        expert_mask_arr = g["_is_expert_normal"].to_numpy(dtype=bool) if "_is_expert_normal" in g.columns else np.zeros(n, dtype=bool)
        expert_vals = vals[expert_mask_arr]
    else:
        group_sigma = 1.0
        group_weight = 0
        recency_weights = g["_recency_weight"].to_numpy(dtype=float)
        expert_mask_arr = np.zeros(n, dtype=bool)
        expert_vals = np.array([], dtype=float)

    if n < 10:
        q_low = float(g["实际成本"].quantile(0.05))
        q_high = float(g["实际成本"].quantile(0.95))
        baseline = _weighted_quantile(vals, recency_weights, group_baseline_quantile, float(g["实际成本"].median()))
        if weighted_mode and expert_vals.size > 0:
            expert_center = float(np.median(expert_vals))
            baseline = (baseline + expert_center * group_weight) / (1 + group_weight)
            q_low = min(q_low, float(np.min(expert_vals)))
            q_high = max(q_high, float(np.max(expert_vals)))

        lower_bound, baseline, upper_bound = _ensure_baseline_within_bounds(max(0.0, q_low), baseline, q_high)

        deviation_amounts = vals - baseline
        deviation_ratios = np.full(n, np.nan, dtype=float) if baseline == 0 else deviation_amounts / baseline
        status_values = np.full(n, "正常（小样本数据）", dtype=object)
        status_values[vals > upper_bound] = "异常偏高（小样本数据）"
        status_values[vals < lower_bound] = "异常偏低（小样本数据）"

        g["预测值"] = baseline
        g["合理下限"] = lower_bound
        g["合理上限"] = upper_bound
        g["偏离数值"] = deviation_amounts
        g["偏离比例"] = deviation_ratios
        g["status"] = status_values
        ring_payload = _format_ring_intervals(
            [
                {
                    "id": 1,
                    "role": "主邻居圈",
                    "lower": lower_bound,
                    "upper": upper_bound,
                    "baseline": baseline,
                    "raw_count": n,
                    "weighted_count": float(np.sum(recency_weights)),
                    "confidence": 1.0,
                    "has_expert_anchor": bool(expert_vals.size > 0),
                }
            ]
        )
        g["圈层编号"] = 1
        g["圈层角色"] = "主邻居圈"
        g["圈层置信度"] = 1.0
        g["多圈合理区间"] = ring_payload
        if weighted_mode:
            g["专家校准"] = np.where(expert_mask_arr, "✅", "")
            g["判定依据"] = group_source
        return _finalize_group_ring_status(g, fixed_intervals, group_source)

    effective_weights = recency_weights.copy()
    if weighted_mode and expert_mask_arr.any():
        effective_weights[expert_mask_arr] = effective_weights[expert_mask_arr] + float(group_weight)

    uniq_vals, uniq_weights, uniq_raw_counts = _collapse_weighted_values(vals, effective_weights)
    if uniq_vals.size == 1:
        single_val = float(uniq_vals[0])
        g["预测值"] = single_val
        g["合理下限"] = max(0.0, single_val)
        g["合理上限"] = single_val
        g["偏离数值"] = 0.0
        g["偏离比例"] = 0.0
        g["status"] = "正常"
        g["圈层编号"] = 1
        g["圈层角色"] = "主邻居圈"
        g["圈层置信度"] = 1.0
        g["多圈合理区间"] = _format_ring_intervals(
            [
                {
                    "id": 1,
                    "role": "主邻居圈",
                    "lower": max(0.0, single_val),
                    "upper": single_val,
                    "baseline": single_val,
                    "raw_count": n,
                    "weighted_count": float(np.sum(effective_weights)),
                    "confidence": 1.0,
                    "has_expert_anchor": bool(expert_vals.size > 0),
                }
            ]
        )
        if weighted_mode:
            g["专家校准"] = np.where(expert_mask_arr, "✅", "")
            g["判定依据"] = group_source
        return _finalize_group_ring_status(g, fixed_intervals, group_source)

    std_val = _weighted_std(vals, effective_weights)
    weighted_q75 = _weighted_quantile(vals, effective_weights, 0.75, float(np.percentile(vals, 75)))
    weighted_q25 = _weighted_quantile(vals, effective_weights, 0.25, float(np.percentile(vals, 25)))
    iqr_val = float(weighted_q75 - weighted_q25)
    spread = min(value for value in [std_val, iqr_val] if value > 0) if (std_val > 0 and iqr_val > 0) else max(std_val, iqr_val)
    if spread <= 0:
        if weighted_mode:
            unique_weighted = np.sort(np.unique(vals))
            spread = float(np.mean(np.abs(np.diff(unique_weighted)))) if unique_weighted.size > 1 else 1.0
        else:
            spread = float(np.mean(np.abs(np.diff(uniq_vals)))) if uniq_vals.size > 1 else 1.0

    bandwidth = spread / np.sqrt(max(1.0, float(np.sum(effective_weights))))
    if weighted_mode:
        bandwidth *= group_sigma
    if bandwidth <= 0:
        bandwidth = 1.0

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    density_weights = recency_weights if weighted_mode else effective_weights
    kde.fit(vals.reshape(-1, 1), sample_weight=density_weights)
    grid_size = int(
        min(
            DGB_KDE_GRID_MAX_POINTS,
            max(DGB_KDE_GRID_MIN_POINTS, uniq_vals.size * DGB_KDE_GRID_POINTS_PER_UNIQUE),
        )
    )
    grid = np.linspace(float(np.min(vals)), float(np.max(vals)), grid_size)
    density = np.exp(kde.score_samples(grid.reshape(-1, 1)))
    peak_price = float(grid[int(np.argmax(density))])

    peak_idx = int(np.argmin(np.abs(uniq_vals - peak_price)))
    k = int(np.sqrt(uniq_vals.size))
    nn_count = min(uniq_vals.size, max(2, k + 1))
    nbrs = NearestNeighbors(n_neighbors=nn_count)
    nbrs.fit(uniq_vals.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(uniq_vals.reshape(-1, 1))
    local_mean = distances[:, 1:].mean(axis=1) if nn_count > 1 else np.ones(uniq_vals.size)
    local_mean = np.where(local_mean == 0, np.nanmedian(local_mean[local_mean > 0]) if np.any(local_mean > 0) else 1.0, local_mean)

    gaps = np.diff(uniq_vals)
    local_scale = (local_mean[:-1] + local_mean[1:]) / 2.0
    local_scale = np.where(local_scale == 0, np.nanmedian(local_scale[local_scale > 0]) if np.any(local_scale > 0) else 1.0, local_scale)
    norm_gaps = gaps / local_scale

    right_seq = norm_gaps[peak_idx:] if peak_idx < norm_gaps.size else np.array([])
    left_seq = norm_gaps[:peak_idx][::-1] if peak_idx > 0 else np.array([])
    right_elbow = _elbow_index(right_seq) if right_seq.size else 0
    left_elbow = _elbow_index(left_seq) if left_seq.size else 0

    threshold_candidates = []
    if right_seq.size:
        threshold_candidates.append(float(right_seq[right_elbow]))
    if left_seq.size:
        threshold_candidates.append(float(left_seq[left_elbow]))
    base_threshold = float(np.median(threshold_candidates)) if threshold_candidates else float(np.median(norm_gaps))

    split_threshold = max(2.5, float(base_threshold) * (group_gap_k / DEFAULT_GAP_K))
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.size:
        median_gap = float(np.median(positive_gaps))
        q75_gap = float(np.percentile(positive_gaps, 75))
        absolute_gap_threshold = max(median_gap * group_gap_k, q75_gap * 2.0)
        dominant_gap_threshold = max(median_gap * group_gap_k, q75_gap * 1.5)
        absolute_break_idx = set(
            np.where((gaps >= absolute_gap_threshold) | (gaps >= dominant_gap_threshold))[0].tolist()
        )
    else:
        absolute_break_idx = set()
    break_idx = set(np.where(norm_gaps >= split_threshold)[0].tolist()) | absolute_break_idx
    components = _build_components(uniq_vals, break_idx)
    comp_counts: List[float] = []
    comp_raw_counts: List[float] = []
    comp_density: List[float] = []
    comp_has_expert_anchor: List[bool] = []
    for start, end in components:
        count = float(np.sum(uniq_weights[start : end + 1]))
        raw_count = float(np.sum(uniq_raw_counts[start : end + 1]))
        span = float(max(uniq_vals[end] - uniq_vals[start], np.finfo(float).eps))
        comp_counts.append(count)
        comp_raw_counts.append(raw_count)
        comp_density.append(count / span)
        if weighted_mode and expert_mask_arr.any():
            comp_mask = (vals >= float(uniq_vals[start])) & (vals <= float(uniq_vals[end]))
            comp_has_expert_anchor.append(bool(np.any(expert_mask_arr & comp_mask)))
        else:
            comp_has_expert_anchor.append(False)

    comp_counts_arr = np.asarray(comp_counts, dtype=float)
    comp_raw_counts_arr = np.asarray(comp_raw_counts, dtype=float)
    comp_density_arr = np.asarray(comp_density, dtype=float)
    kde_peak_comp_idx = next(i for i, (start, end) in enumerate(components) if start <= peak_idx <= end)

    global_dispersion = _weighted_std(vals, effective_weights)
    if global_dispersion <= 0:
        global_dispersion = float(np.mean(np.abs(gaps))) if gaps.size else 1.0
    gap_ratio = gaps / global_dispersion if global_dispersion > 0 else gaps
    density_threshold = _weighted_median(
        comp_density_arr,
        comp_counts_arr,
        float(np.median(comp_density_arr)) if comp_density_arr.size else 0.0,
    ) if comp_density_arr.size else 0.0

    min_secondary_count = _secondary_ring_min_count(n)
    peak_comp_idx = _select_primary_component_index(
        kde_peak_comp_idx,
        comp_raw_counts_arr,
        comp_density_arr,
        comp_has_expert_anchor,
    )
    if comp_raw_counts_arr.size and comp_raw_counts_arr[peak_comp_idx] < min_secondary_count:
        peak_comp_idx = int(np.argmax(comp_raw_counts_arr))
    max_density = float(np.max(comp_density_arr)) if comp_density_arr.size else 1.0
    min_secondary_density = float(density_threshold) * 0.25 if density_threshold > 0 else 0.0
    accepted_component_indexes: set[int] = {peak_comp_idx}
    for comp_idx, raw_count in enumerate(comp_raw_counts_arr):
        if comp_idx == peak_comp_idx:
            continue
        has_anchor = bool(comp_has_expert_anchor[comp_idx])
        enough_support = float(raw_count) >= float(min_secondary_count)
        enough_density = float(comp_density[comp_idx]) >= min_secondary_density
        if has_anchor or (enough_support and enough_density):
            accepted_component_indexes.add(comp_idx)

    rings: List[Dict[str, Any]] = []
    for comp_idx in sorted(accepted_component_indexes, key=lambda idx: float(uniq_vals[components[idx][0]])):
        start, end = components[comp_idx]
        component_lower_bound = max(0.0, float(uniq_vals[start]))
        component_upper_bound = float(uniq_vals[end])
        cluster_mask = (vals >= component_lower_bound) & (vals <= component_upper_bound)
        cluster_samples = vals[cluster_mask]
        cluster_weights = effective_weights[cluster_mask]
        fallback = float(np.median(cluster_samples)) if cluster_samples.size else float(uniq_vals[start])
        ring_baseline = _weighted_quantile(cluster_samples, cluster_weights, group_baseline_quantile, fallback)
        ring_baseline = float(np.clip(ring_baseline, component_lower_bound, component_upper_bound))
        is_main_ring = comp_idx == peak_comp_idx
        role = "主邻居圈" if is_main_ring else "次邻居圈"
        confidence = _ring_confidence(
            raw_count=float(comp_raw_counts_arr[comp_idx]),
            sample_count=n,
            min_count=min_secondary_count,
            density=float(comp_density[comp_idx]),
            max_density=max_density,
            is_main=is_main_ring,
            has_expert_anchor=bool(comp_has_expert_anchor[comp_idx]),
        )
        rings.append(
            {
                "id": len(rings) + 1,
                "component_idx": comp_idx,
                "role": role,
                "lower": component_lower_bound,
                "upper": component_upper_bound,
                "baseline": ring_baseline,
                "raw_count": int(comp_raw_counts_arr[comp_idx]),
                "weighted_count": float(comp_counts_arr[comp_idx]),
                "confidence": confidence,
                "has_expert_anchor": bool(comp_has_expert_anchor[comp_idx]),
            }
        )

    main_ring = next((ring for ring in rings if ring["role"] == "主邻居圈"), rings[0])
    ring_by_component = {int(ring["component_idx"]): ring for ring in rings}
    ring_payload = _format_ring_intervals(rings)
    comp_index_arr = np.zeros(uniq_vals.size, dtype=int)
    for comp_id, (start, end) in enumerate(components):
        comp_index_arr[start : end + 1] = comp_id

    value_indexes = np.searchsorted(uniq_vals, vals)
    value_indexes = np.clip(value_indexes, 0, uniq_vals.size - 1)
    comp_ids = comp_index_arr[value_indexes]
    small_comp_threshold = float(np.median(comp_raw_counts_arr)) if comp_raw_counts_arr.size else 0.0

    baseline_values = np.empty(n, dtype=float)
    lower_values = np.empty(n, dtype=float)
    upper_values = np.empty(n, dtype=float)
    confidence_values = np.empty(n, dtype=float)
    ring_id_values = np.zeros(n, dtype=int)
    ring_role_values = np.full(n, "未采纳孤立圈", dtype=object)
    status_values = np.full(n, "异常偏高", dtype=object)

    min_accepted_lower = min(float(ring["lower"]) for ring in rings)
    max_accepted_upper = max(float(ring["upper"]) for ring in rings)

    for row_idx, value in enumerate(vals):
        comp_id = int(comp_ids[row_idx])
        ring = ring_by_component.get(comp_id)
        if ring is not None:
            baseline = float(ring["baseline"])
            baseline_values[row_idx] = baseline
            lower_values[row_idx] = float(ring["lower"])
            upper_values[row_idx] = float(ring["upper"])
            confidence_values[row_idx] = float(ring["confidence"])
            ring_id_values[row_idx] = int(ring["id"])
            ring_role_values[row_idx] = str(ring["role"])
            status_values[row_idx] = "正常（主邻居圈）" if ring["role"] == "主邻居圈" else "正常（次邻居圈）"
            continue

        nearest_ring = min(
            rings,
            key=lambda item: 0.0
            if float(item["lower"]) <= float(value) <= float(item["upper"])
            else min(abs(float(value) - float(item["lower"])), abs(float(value) - float(item["upper"]))),
        )
        baseline = float(nearest_ring["baseline"])
        baseline_values[row_idx] = baseline
        lower_values[row_idx] = float(nearest_ring["lower"])
        upper_values[row_idx] = float(nearest_ring["upper"])
        confidence_values[row_idx] = 0.0
        if float(value) < min_accepted_lower and comp_raw_counts_arr[comp_id] <= small_comp_threshold:
            status_values[row_idx] = "严重异常偏低"
        elif float(value) < float(nearest_ring["lower"]):
            status_values[row_idx] = "异常偏低"
        elif float(value) > float(nearest_ring["upper"]):
            status_values[row_idx] = "异常偏高"
        elif float(value) < float(main_ring["baseline"]):
            status_values[row_idx] = "异常偏低"
        else:
            status_values[row_idx] = "异常偏高"

    deviation_amounts = vals - baseline_values
    deviation_ratios = np.divide(
        deviation_amounts,
        baseline_values,
        out=np.full(n, np.nan, dtype=float),
        where=baseline_values != 0,
    )

    g["预测值"] = baseline_values
    g["合理下限"] = lower_values
    g["合理上限"] = upper_values
    g["偏离数值"] = deviation_amounts
    g["偏离比例"] = deviation_ratios
    g["status"] = status_values
    g["圈层编号"] = ring_id_values
    g["圈层角色"] = ring_role_values
    g["圈层置信度"] = confidence_values
    g["多圈合理区间"] = ring_payload
    if weighted_mode:
        g["专家校准"] = np.where(expert_mask_arr, "✅", "")
        g["判定依据"] = group_source
    return _finalize_group_ring_status(g, fixed_intervals, group_source)


def _run_group_detection_tasks(
    group_tasks: List[Tuple[pd.DataFrame, Dict[str, Any]]],
    total_rows: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    total_groups = len(group_tasks)
    max_workers = min(os.cpu_count() or 1, total_groups) if total_groups else 1
    use_parallel = _should_use_parallel_detection(total_rows, total_groups)
    results: List[pd.DataFrame] = []
    processed_rows = 0

    if progress_callback is not None:
        progress_callback(0, max(total_rows, 1), "准备异常检测")

    if use_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_rows = {
                executor.submit(_detect_group_anomalies_worker, group_df, config): len(group_df)
                for group_df, config in group_tasks
            }
            for completed_groups, future in enumerate(as_completed(future_to_rows), start=1):
                results.append(future.result())
                processed_rows += future_to_rows[future]
                if progress_callback is not None:
                    progress_callback(
                        processed_rows,
                        max(total_rows, 1),
                        f"并行计算中（{completed_groups}/{total_groups} 组）",
                    )
    else:
        for idx, (group_df, config) in enumerate(group_tasks, start=1):
            results.append(_detect_group_anomalies_worker(group_df, config))
            processed_rows += len(group_df)
            if progress_callback is not None:
                progress_callback(
                    processed_rows,
                    max(total_rows, 1),
                    f"计算中（{idx}/{total_groups} 组）",
                )

    if progress_callback is not None:
        progress_callback(max(total_rows, 1), max(total_rows, 1), "计算完成")

    return results, {
        "parallel": use_parallel,
        "workers": max_workers,
        "groups": total_groups,
    }


def detect_cost_anomalies(
    df: pd.DataFrame,
    price_col: str,
    decay_alpha: float = DEFAULT_DECAY_ALPHA,
    gap_k: float = DEFAULT_GAP_K,
    baseline_quantile: float = DEFAULT_BASELINE_QUANTILE,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=RAW_RESULT_COLUMNS)

    if price_col not in df.columns:
        raise ValueError(f"找不到价格列: {price_col}")

    try:
        from sklearn.neighbors import KernelDensity, NearestNeighbors
    except Exception as exc:
        raise ImportError("缺少依赖 scikit-learn，请先安装：pip install scikit-learn") from exc

    total_started_at = time.perf_counter()
    preprocess_started_at = time.perf_counter()
    normalized_decay_alpha = _normalize_positive_float(decay_alpha, DEFAULT_DECAY_ALPHA)
    normalized_gap_k = _normalize_positive_float(gap_k, DEFAULT_GAP_K)
    normalized_baseline_quantile = _normalize_baseline_quantile(baseline_quantile)
    data = _prepare_detection_input(df, price_col, decay_alpha=normalized_decay_alpha)
    preprocess_seconds = time.perf_counter() - preprocess_started_at

    group_tasks = [
        (group.copy(), {"weighted": False, "gap_k": normalized_gap_k, "baseline_quantile": normalized_baseline_quantile})
        for _, group in data.groupby("备件简称", sort=False)
    ]

    compute_started_at = time.perf_counter()
    results, execution_meta = _run_group_detection_tasks(group_tasks, len(data), progress_callback=progress_callback)
    compute_seconds = time.perf_counter() - compute_started_at

    finalize_started_at = time.perf_counter()
    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if result_df.empty:
        return result_df

    result_df["_record_key"] = build_record_keys(result_df)
    columns = [column for column in RAW_RESULT_COLUMNS if column in result_df.columns]
    result_df = result_df[columns].sort_values(["备件简称", "物料编码"]).reset_index(drop=True)
    finalize_seconds = time.perf_counter() - finalize_started_at
    total_seconds = time.perf_counter() - total_started_at
    print(
        f"[performance][计算阶段][raw] 记录数={len(data)} 分组数={execution_meta['groups']} "
        f"并行={'是' if execution_meta['parallel'] else '否'} 进程数={execution_meta['workers']} "
        f"预处理耗时={preprocess_seconds:.3f}s 分组计算耗时={compute_seconds:.3f}s "
        f"汇总耗时={finalize_seconds:.3f}s 总耗时={total_seconds:.3f}s"
    )
    harness.execute_action("save_cost_anomaly_results", result_df=result_df, result_mode="raw")
    return result_df


def detect_cost_anomalies_weighted(
    df: pd.DataFrame,
    price_col: str,
    expert_labels_tuple: tuple,
    sigma_multiplier: float = 1.0,
    expert_weight_override: int = 0,
    decay_alpha: float = DEFAULT_DECAY_ALPHA,
    gap_k: float = DEFAULT_GAP_K,
    baseline_quantile: float = DEFAULT_BASELINE_QUANTILE,
    skills_overrides_json: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    expert_labels: Dict[str, str] = dict(expert_labels_tuple)
    expert_weight = expert_weight_override if expert_weight_override > 0 else _EXPERT_WEIGHT

    skills_index: Dict[str, dict] = {}
    if skills_overrides_json:
        try:
            skills_index = _json.loads(skills_overrides_json)
        except Exception:
            skills_index = {}

    if df.empty:
        return pd.DataFrame(columns=WEIGHTED_RESULT_COLUMNS)

    if price_col not in df.columns:
        raise ValueError(f"找不到价格列: {price_col}")

    try:
        from sklearn.neighbors import KernelDensity, NearestNeighbors
    except Exception as exc:
        raise ImportError("缺少依赖 scikit-learn，请先安装：pip install scikit-learn") from exc

    total_started_at = time.perf_counter()
    preprocess_started_at = time.perf_counter()
    normalized_baseline_quantile = _normalize_baseline_quantile(baseline_quantile)
    data = _prepare_detection_input(df, price_col, include_record_keys=True)
    normal_keys = {key for key, value in expert_labels.items() if value == "正常"}
    data["_is_expert_normal"] = data["_record_key"].isin(normal_keys)
    preprocess_seconds = time.perf_counter() - preprocess_started_at

    group_tasks: List[Tuple[pd.DataFrame, Dict[str, Any]]] = []
    for short_name, group in data.groupby("备件简称", sort=False):
        group_skill = skills_index.get(str(short_name))
        if group_skill:
            group_config = {
                "weighted": True,
                "sigma_multiplier": float(group_skill.get("sigma", sigma_multiplier)),
                "expert_weight": int(group_skill.get("weight", expert_weight)),
                "decay_alpha": float(group_skill.get("decay_alpha", decay_alpha)),
                "gap_k": float(group_skill.get("gap_k", gap_k)),
                "baseline_quantile": float(group_skill.get("baseline_quantile", normalized_baseline_quantile)),
                "fixed_intervals": group_skill.get("fixed_intervals", []),
                "group_source": "专家经验报告Excel" if group_skill.get("fixed_intervals") else "技能书校验",
            }
        else:
            group_config = {
                "weighted": True,
                "sigma_multiplier": sigma_multiplier,
                "expert_weight": expert_weight,
                "decay_alpha": decay_alpha,
                "gap_k": gap_k,
                "baseline_quantile": normalized_baseline_quantile,
                "group_source": "默认算法",
            }
        group_tasks.append((group.copy(), group_config))

    compute_started_at = time.perf_counter()
    results, execution_meta = _run_group_detection_tasks(group_tasks, len(data), progress_callback=progress_callback)
    compute_seconds = time.perf_counter() - compute_started_at

    finalize_started_at = time.perf_counter()
    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if result_df.empty:
        return result_df

    columns = [column for column in WEIGHTED_RESULT_COLUMNS if column in result_df.columns]
    result_df = result_df[columns].sort_values(["备件简称", "物料编码"]).reset_index(drop=True)
    finalize_seconds = time.perf_counter() - finalize_started_at
    total_seconds = time.perf_counter() - total_started_at
    print(
        f"[performance][计算阶段][weighted] 记录数={len(data)} 分组数={execution_meta['groups']} "
        f"并行={'是' if execution_meta['parallel'] else '否'} 进程数={execution_meta['workers']} "
        f"预处理耗时={preprocess_seconds:.3f}s 分组计算耗时={compute_seconds:.3f}s "
        f"汇总耗时={finalize_seconds:.3f}s 总耗时={total_seconds:.3f}s"
    )
    harness.execute_action("save_cost_anomaly_results", result_df=result_df, result_mode="weighted")
    return result_df
