"""Skills 技能引擎 & AutoResearch 棘轮迭代。

提供：
1. Skills 技能书 — 每个备件简称的算法参数、分布特征、合理区间边界 (JSON / Markdown)
2. AutoResearch — 严格棘轮循环自动调参（σ × 偏置权重 × 时序衰减 × 断层灵敏度）
3. 深度审计报表 — 物料级原始结论 vs 优化结论对照
"""

import html
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from anomaly_engine import (
    DEFAULT_BASELINE_QUANTILE,
    DEFAULT_DECAY_ALPHA,
    DEFAULT_GAP_K,
    _EXPERT_WEIGHT,
    detect_cost_anomalies_weighted,
)
from local_logging import log_event
from skills_excel_export import flatten_skills_for_excel, skills_to_excel_bytes as _skills_to_excel_bytes


AUTORESEARCH_PARAM_GRID = {
    "sigma": np.round(np.linspace(0.1, 5.0, 25), 4),
    "expert_weight": np.arange(40, 321, 20, dtype=int),
    "decay_alpha": np.round(np.linspace(0.1, 2.0, 20), 4),
    "gap_k": np.round(np.linspace(2.0, 10.0, 17), 4),
    "baseline_quantile": np.round(np.linspace(0.4, 0.6, 9), 4),
}
AUTORESEARCH_MAX_RUNTIME_SECONDS = 30.0


# ── 1. Skills 技能书 ──────────────────────────────────────────────────────


SKILLS_REPORT_STYLE = """
<style>
.skills-report-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  width: 100%;
  margin: 12px 0 18px;
}
.skills-report-panel {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
  padding: 12px 14px;
  min-width: 0;
}
.skills-report-panel h3 {
  margin: 0 0 10px;
  font-size: 1.05rem;
}
.skills-report-panel table {
  width: 100%;
  border-collapse: collapse;
  table-layout: auto;
}
.skills-report-panel th,
.skills-report-panel td {
  border: 1px solid #e5e7eb;
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
  word-break: break-word;
}
.skills-report-panel th {
  background: #f8fafc;
  font-weight: 700;
}
.skills-report-panel ul {
  margin: 0;
  padding-left: 1.15rem;
}
@media (max-width: 900px) {
  .skills-report-grid {
    grid-template-columns: 1fr;
  }
}
</style>
""".strip()


def _format_report_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        return f"{float(value):,.4f}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    return str(value)


def _safe_report_text(value: object) -> str:
    return html.escape(_format_report_value(value), quote=True)


def _html_table(headers: List[str], rows: List[List[object]]) -> str:
    header_html = "".join(f"<th>{_safe_report_text(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append(
            "<tr>"
            + "".join(f"<td>{_safe_report_text(value)}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def _html_panel(title: str, body_html: str) -> str:
    return f"<section class='skills-report-panel'><h3>{_safe_report_text(title)}</h3>{body_html}</section>"


def _html_grid(*panels: str) -> str:
    return "<div class='skills-report-grid'>" + "".join(panels) + "</div>"


def _semantic_report_panel(semantic_count: object, semantic_modes: List[str], semantic_refs: List[str]) -> str:
    mode_text = " / ".join(semantic_modes) if semantic_modes else "无"
    ref_items = "".join(f"<li>{_safe_report_text(ref)}</li>" for ref in semantic_refs) if semantic_refs else "<li>无</li>"
    body = (
        "<ul>"
        f"<li><strong>引用规律数</strong>: {_safe_report_text(semantic_count)}</li>"
        f"<li><strong>主要匹配方式</strong>: {_safe_report_text(mode_text)}</li>"
        f"<li><strong>参考文本规律</strong>: <ul>{ref_items}</ul></li>"
        "</ul>"
    )
    return _html_panel("语义校准报告", body)


def _normalize_semantic_report(report: Optional[dict] = None) -> Dict[str, object]:
    report = report or {}

    raw_modes = report.get("主要匹配方式", [])
    if not isinstance(raw_modes, list):
        raw_modes = [raw_modes] if raw_modes else []
    modes = [str(value).strip() for value in raw_modes if str(value).strip()]

    raw_refs = report.get("参考文本规律", [])
    if not isinstance(raw_refs, list):
        raw_refs = [raw_refs] if raw_refs else []
    refs = [str(value).strip() for value in raw_refs if str(value).strip()]

    try:
        count = int(report.get("引用规律数", len(set(refs)) if refs else 0))
    except (TypeError, ValueError):
        count = len(set(refs)) if refs else 0

    return {
        "引用规律数": max(count, len(set(refs)) if refs else 0),
        "主要匹配方式": list(dict.fromkeys(modes)),
        "参考文本规律": list(dict.fromkeys(refs)),
    }


def _build_parameter_semantic_notes(decay_alpha: float, gap_k: float) -> List[str]:
    notes: List[str] = []
    if decay_alpha >= 1.2:
        notes.append("该备件受近期市场波动影响剧烈，算法已自动切换为短时记忆模式。")
    elif decay_alpha <= 0.4:
        notes.append("该备件历史价格惯性较强，算法保留了更长的稳定记忆窗口。")

    if gap_k <= 3.5:
        notes.append("该类备件定价规律极度统一，算法已收紧断层检测阈值。")
    elif gap_k >= 7.5:
        notes.append("该类备件价格结构更分散，算法适度放宽断层切割阈值。")

    if not notes:
        notes.append("该备件当前保持稳健模式，时序记忆与圈层切割阈值均处于均衡区间。")
    return notes


def _sample_autoresearch_candidates(rng: np.random.RandomState, n_iterations: int) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    seen = set()
    attempts = 0
    max_attempts = max(20, n_iterations * 20)

    while len(candidates) < n_iterations and attempts < max_attempts:
        attempts += 1
        candidate = {
            "sigma": float(rng.choice(AUTORESEARCH_PARAM_GRID["sigma"])),
            "weight": int(rng.choice(AUTORESEARCH_PARAM_GRID["expert_weight"])),
            "decay_alpha": float(rng.choice(AUTORESEARCH_PARAM_GRID["decay_alpha"])),
            "gap_k": float(rng.choice(AUTORESEARCH_PARAM_GRID["gap_k"])),
            "baseline_quantile": float(rng.choice(AUTORESEARCH_PARAM_GRID["baseline_quantile"])),
        }
        key = (
            round(candidate["sigma"], 4),
            int(candidate["weight"]),
            round(candidate["decay_alpha"], 4),
            round(candidate["gap_k"], 4),
            round(candidate["baseline_quantile"], 4),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return candidates


def _is_trial_better(
    trial_score: float,
    trial_conflicts: int,
    trial_decay_alpha: float,
    trial_ring_count: int,
    trial_interval_width: float,
    best_score: float,
    best_conflicts: int,
    best_decay_alpha: float,
    best_ring_count: int,
    best_interval_width: float,
) -> bool:
    epsilon = 1e-9
    if trial_score > best_score + epsilon:
        return True
    if abs(trial_score - best_score) <= epsilon:
        if trial_conflicts < best_conflicts:
            return True
        if trial_conflicts == best_conflicts and trial_ring_count < best_ring_count:
            return True
        if trial_conflicts == best_conflicts and trial_ring_count == best_ring_count and trial_interval_width < best_interval_width - epsilon:
            return True
        if (
            trial_conflicts == best_conflicts
            and trial_ring_count == best_ring_count
            and abs(trial_interval_width - best_interval_width) <= epsilon
            and trial_decay_alpha < best_decay_alpha - epsilon
        ):
            return True
    return False


def _result_ring_complexity(result_df: pd.DataFrame) -> tuple[int, float]:
    if result_df is None or result_df.empty or "圈层角色" not in result_df.columns:
        return 1, 0.0
    normal_rings = result_df[result_df["圈层角色"].astype(str).isin(["主邻居圈", "次邻居圈"])]
    if normal_rings.empty:
        return 0, 0.0
    ring_count = int(normal_rings[["备件简称", "圈层编号"]].drop_duplicates().shape[0]) if "圈层编号" in normal_rings.columns else 1
    interval_df = normal_rings.drop_duplicates(subset=["备件简称", "圈层编号", "合理下限", "合理上限"])
    interval_width = float((pd.to_numeric(interval_df["合理上限"], errors="coerce") - pd.to_numeric(interval_df["合理下限"], errors="coerce")).clip(lower=0).sum())
    return ring_count, interval_width


def extract_skills(
    anomaly_df: pd.DataFrame,
    expert_labels: Dict[str, str],
    sigma_multiplier: float = 1.0,
    expert_weight: int = 80,
    decay_alpha: float = DEFAULT_DECAY_ALPHA,
    gap_k: float = DEFAULT_GAP_K,
    baseline_quantile: float = DEFAULT_BASELINE_QUANTILE,
) -> List[dict]:
    """为每个备件简称提取 Skills 条目。

    Parameters
    ----------
    anomaly_df : 异常检测结果 DataFrame（含 _record_key / 预测值 / 合理上下限 等列）。
    expert_labels : {record_key: label} 专家标注字典。
    sigma_multiplier : 当前使用的 σ 缩放系数。
    expert_weight : 当前使用的偏置权重倍数。
    """
    has_rk = "_record_key" in anomaly_df.columns
    skills: List[dict] = []
    normalized_decay_alpha = round(float(decay_alpha), 4)
    normalized_gap_k = round(float(gap_k), 4)
    normalized_baseline_quantile = round(float(baseline_quantile), 4)
    parameter_notes = _build_parameter_semantic_notes(normalized_decay_alpha, normalized_gap_k)

    for short_name, group in anomaly_df.groupby("备件简称", sort=True):
        costs = group["实际成本"].to_numpy(dtype=float)
        ring_intervals = []
        if "多圈合理区间" in group.columns:
            for payload_text in group["多圈合理区间"].dropna().astype(str):
                if not payload_text.strip():
                    continue
                try:
                    parsed = json.loads(payload_text)
                    if isinstance(parsed, list):
                        ring_intervals = parsed
                        break
                except Exception:
                    continue
        main_interval = next(
            (item for item in ring_intervals if str(item.get("圈层角色", "")) == "主邻居圈"),
            ring_intervals[0] if ring_intervals else {},
        )
        if main_interval:
            main_baseline = main_interval.get("预测值")
            main_lower = main_interval.get("合理下限")
            main_upper = main_interval.get("合理上限")
            main_ring_id = main_interval.get("圈层编号")
        else:
            main_rows = group[group.get("圈层角色", pd.Series(index=group.index, dtype=object)).astype(str) == "主邻居圈"] if "圈层角色" in group.columns else group
            if main_rows.empty:
                main_rows = group
            main_baseline = main_rows["预测值"].median()
            main_lower = main_rows["合理下限"].median()
            main_upper = main_rows["合理上限"].median()
            main_ring_id = 1

        # 本组专家标注数
        if has_rk:
            rkeys = set(group["_record_key"].values)
            expert_count = sum(1 for k, v in expert_labels.items() if v == "正常" and k in rkeys)
        else:
            expert_count = 0

        dist: dict = {
            "样本量": int(len(costs)),
            "均值": round(float(np.mean(costs)), 4),
            "标准差": round(float(np.std(costs)), 4),
            "中位数": round(float(np.median(costs)), 4),
            "最小值": round(float(np.min(costs)), 4),
            "最大值": round(float(np.max(costs)), 4),
        }
        if len(costs) > 2:
            dist["偏度"] = round(float(pd.Series(costs).skew()), 4)

        skill = {
            "备件简称": str(short_name),
            "适用算法": "DGB-MultiRing KDE+KNN+Elbow 密度连接异常检测",
            "数据结构分布描述": dist,
            "当前σ参数": round(sigma_multiplier, 4),
            "偏置权重": expert_weight,
            "时序敏感度 (Decay Alpha)": normalized_decay_alpha,
            "圈子严格度 (Gap K)": normalized_gap_k,
            "Baseline Quantile": normalized_baseline_quantile,
            "参数语义说明": list(parameter_notes),
            "本组专家标注数": expert_count,
            "成本合理区间边界": {
                "预测值": round(float(main_baseline), 4),
                "合理下限": round(float(main_lower), 4),
                "合理上限": round(float(main_upper), 4),
            },
            "多邻居圈合理区间": ring_intervals,
            "主邻居圈编号": int(main_ring_id) if pd.notna(main_ring_id) else 1,
            "次邻居圈数量": sum(1 for item in ring_intervals if str(item.get("圈层角色", "")) == "次邻居圈"),
            "异常统计": {
                "正常": int(group["status"].astype(str).str.contains("正常").sum()),
                "异常偏高": int(group["status"].astype(str).str.contains("异常偏高").sum()),
                "异常偏低": int(
                    group["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum()
                ),
            },
        }

        ai_analyses = []
        if "AI 辅助分析" in group.columns:
            ai_analyses = [
                text.strip()
                for text in group["AI 辅助分析"].fillna("").astype(str).tolist()
                if text and text.strip()
            ]
        ai_match_modes = []
        if "_ai_match_scope" in group.columns:
            ai_match_modes = [
                text.strip()
                for text in group["_ai_match_scope"].fillna("").astype(str).tolist()
                if text and text.strip()
            ]
        ai_rule_ids = []
        if "_ai_rule_id" in group.columns:
            ai_rule_ids = [
                text.strip()
                for text in group["_ai_rule_id"].fillna("").astype(str).tolist()
                if text and text.strip()
            ]
        unique_analyses = list(dict.fromkeys(ai_analyses))[:3]
        unique_modes = list(dict.fromkeys(ai_match_modes))[:3]
        skill["语义校准报告"] = _normalize_semantic_report(
            {
                "引用规律数": len(set(ai_rule_ids)) if ai_rule_ids else len(unique_analyses),
                "主要匹配方式": unique_modes,
                "参考文本规律": unique_analyses,
            }
        )

        # ── 经验对齐率：专家标注"正常"的记录中，实际落在合理区间内的比例 ──
        if has_rk and expert_count > 0:
            row_map = {
                row["_record_key"]: row
                for row in group.to_dict(orient="records")
                if row.get("_record_key") is not None
            }
            aligned = sum(
                1 for k, v in expert_labels.items()
                if v == "正常" and k in row_map and "正常" in str(row_map[k].get("status", ""))
            )
            skill["经验对齐率"] = round(aligned / expert_count, 4)
        else:
            skill["经验对齐率"] = "N/A"

        skills.append(skill)

    return skills


def skills_to_json_bytes(skills: List[dict]) -> bytes:
    """导出 Skills 为可下载的 JSON 字节。"""
    payload = {
        "version": "1.1",
        "generated_at": datetime.now().isoformat(),
        "skills_count": len(skills),
        "skills": skills,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def skills_to_excel_table(skills: List[dict]) -> pd.DataFrame:
    return flatten_skills_for_excel(
        skills,
        interval_key="成本合理区间边界",
        distribution_key="数据结构分布描述",
    )


def skills_to_excel_bytes(skills: List[dict]) -> bytes:
    return _skills_to_excel_bytes(
        skills,
        interval_key="成本合理区间边界",
        distribution_key="数据结构分布描述",
        sheet_name="成本区间Skills",
    )


def _coerce_export_datetime(value: Any | None = None) -> datetime:
    if value is None:
        return datetime.now()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return datetime.now()
    return parsed.to_pydatetime()


def _normalize_export_directory(export_path: Any) -> Path | None:
    path_text = str(export_path or "").strip()
    if not path_text:
        return None
    export_dir = Path(path_text).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _write_cost_skills_excel_file(
    skills: Sequence[dict],
    export_path: Any,
    *,
    artifact_label: str,
    generated_at: Any | None = None,
    force_new: bool = False,
) -> str:
    export_dir = _normalize_export_directory(export_path)
    if export_dir is None:
        return ""
    payload = skills_to_excel_bytes(list(skills))
    digest = hashlib.sha1(payload).hexdigest()[:10]
    safe_label = str(artifact_label or "全量").strip() or "全量"
    if not force_new:
        existing_files = sorted(export_dir.glob(f"成本区间Skills_{safe_label}_*_{digest}.xlsx"))
        if existing_files:
            return str(existing_files[-1])
    timestamp_text = _coerce_export_datetime(generated_at).strftime("%Y%m%d_%H%M%S")
    target_path = export_dir / f"成本区间Skills_{safe_label}_{timestamp_text}_{digest}.xlsx"
    target_path.write_bytes(payload)
    return str(target_path)


def export_cost_skills_excel_artifacts(
    skills: Sequence[dict],
    *,
    model_export_path: Any,
    expert_report_export_path: Any = "",
    generated_at: Any | None = None,
    force_new: bool = False,
) -> Dict[str, str]:
    """Persist cost Skills Excel artifacts to the configured local export folders."""
    model_path = _write_cost_skills_excel_file(
        skills,
        model_export_path,
        artifact_label="全量",
        generated_at=generated_at,
        force_new=force_new,
    )
    report_path = _write_cost_skills_excel_file(
        skills,
        expert_report_export_path,
        artifact_label="优化后",
        generated_at=generated_at,
        force_new=force_new,
    )
    return {
        "model_export_path": model_path,
        "expert_report_export_path": report_path,
    }


def _cell_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _cell_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _cell_int(value: Any, default: int = 0) -> int:
    number = _cell_float(value)
    if number is None:
        return default
    return int(number)


def _row_to_cost_skill(row: Dict[str, Any]) -> dict:
    skill = {
        "备件简称": _cell_text(row.get("备件简称")),
        "适用算法": _cell_text(row.get("适用算法"), "DGB-MultiRing KDE+KNN+Elbow 密度连接异常检测"),
        "当前σ参数": _cell_float(row.get("当前σ参数"), 1.0),
        "偏置权重": _cell_int(row.get("偏置权重"), 80),
        "时序敏感度 (Decay Alpha)": _cell_float(row.get("时序敏感度 (Decay Alpha)"), DEFAULT_DECAY_ALPHA),
        "圈子严格度 (Gap K)": _cell_float(row.get("圈子严格度 (Gap K)"), DEFAULT_GAP_K),
        "Baseline Quantile": _cell_float(row.get("Baseline Quantile"), DEFAULT_BASELINE_QUANTILE),
        "本组专家标注数": _cell_int(row.get("本组专家标注数"), 0),
        "经验对齐率": row.get("经验对齐率", "N/A"),
        "成本合理区间边界": {
            "预测值": _cell_float(row.get("合理区间_预测值"), 0.0),
            "合理下限": _cell_float(row.get("合理区间_合理下限"), 0.0),
            "合理上限": _cell_float(row.get("合理区间_合理上限"), 0.0),
        },
        "数据结构分布描述": {},
        "异常统计": {},
        "语义校准报告": {
            "引用规律数": _cell_int(row.get("语义_引用规律数"), 0),
            "主要匹配方式": [value for value in _cell_text(row.get("语义_主要匹配方式")).split(" / ") if value],
            "参考文本规律": [value for value in _cell_text(row.get("语义_参考文本规律")).split(" / ") if value],
        },
    }
    rings: List[dict] = []
    ring_idx = 1
    while any(key.startswith(f"邻居圈{ring_idx}_") for key in row):
        lower = _cell_float(row.get(f"邻居圈{ring_idx}_合理下限"))
        upper = _cell_float(row.get(f"邻居圈{ring_idx}_合理上限"))
        if lower is not None and upper is not None:
            rings.append(
                {
                    "圈层编号": _cell_int(row.get(f"邻居圈{ring_idx}_编号"), ring_idx),
                    "圈层角色": _cell_text(
                        row.get(f"邻居圈{ring_idx}_角色"),
                        "主邻居圈" if ring_idx == 1 else "次邻居圈",
                    ),
                    "合理下限": lower,
                    "预测值": _cell_float(row.get(f"邻居圈{ring_idx}_预测值"), (lower + upper) / 2.0),
                    "合理上限": upper,
                    "样本量": _cell_int(row.get(f"邻居圈{ring_idx}_样本量"), 0),
                    "加权样本量": _cell_float(row.get(f"邻居圈{ring_idx}_加权样本量"), 0.0),
                    "圈层置信度": _cell_float(row.get(f"邻居圈{ring_idx}_置信度"), 1.0),
                    "专家锚点": bool(row.get(f"邻居圈{ring_idx}_专家锚点", False)),
                }
            )
        ring_idx += 1
    if not rings:
        bounds = skill["成本合理区间边界"]
        lower = _cell_float(bounds.get("合理下限"))
        upper = _cell_float(bounds.get("合理上限"))
        if lower is not None and upper is not None:
            rings.append(
                {
                    "圈层编号": 1,
                    "圈层角色": "主邻居圈",
                    "合理下限": lower,
                    "预测值": _cell_float(bounds.get("预测值"), (lower + upper) / 2.0),
                    "合理上限": upper,
                    "圈层置信度": 1.0,
                }
            )
    skill["多邻居圈合理区间"] = rings
    return skill


def load_latest_cost_skills_excel(export_path: Any) -> Optional[Dict[str, Any]]:
    export_dir = Path(str(export_path or "").strip()).expanduser() if str(export_path or "").strip() else None
    if export_dir is None or not export_dir.exists() or not export_dir.is_dir():
        return None
    candidates = [
        path
        for path in export_dir.glob("*.xlsx")
        if path.is_file() and not path.name.startswith("~$")
    ]
    if not candidates:
        return None
    latest_path = sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]
    table_df = pd.read_excel(latest_path)
    if table_df.empty or "备件简称" not in table_df.columns:
        return None
    skills = [
        skill
        for skill in (_row_to_cost_skill(row) for row in table_df.to_dict(orient="records"))
        if _cell_text(skill.get("备件简称"))
    ]
    if not skills:
        return None
    return {
        "snapshot_id": latest_path.stem,
        "module_type": "cost",
        "skill_domain": "cost",
        "version": "excel",
        "saved_at": datetime.fromtimestamp(latest_path.stat().st_mtime).isoformat(),
        "source_path": str(latest_path),
        "skills": skills,
        "index": {str(skill.get("备件简称", "")): skill for skill in skills},
    }


def build_cost_skill_overrides_json(skills_data: Optional[Dict[str, Any] | Sequence[dict]]) -> str:
    if not skills_data:
        return ""
    skills = skills_data.get("skills", []) if isinstance(skills_data, dict) else skills_data
    overrides: Dict[str, Dict[str, Any]] = {}
    for skill in skills:
        short_name = _cell_text(skill.get("备件简称"))
        if not short_name:
            continue
        override = {
            "sigma": _cell_float(skill.get("当前σ参数"), 1.0),
            "weight": _cell_int(skill.get("偏置权重"), 80),
            "decay_alpha": _cell_float(skill.get("时序敏感度 (Decay Alpha)"), DEFAULT_DECAY_ALPHA),
            "gap_k": _cell_float(skill.get("圈子严格度 (Gap K)"), DEFAULT_GAP_K),
            "baseline_quantile": _cell_float(skill.get("Baseline Quantile"), DEFAULT_BASELINE_QUANTILE),
        }
        fixed_intervals = skill.get("多邻居圈合理区间")
        if isinstance(fixed_intervals, list) and fixed_intervals:
            override["fixed_intervals"] = fixed_intervals
        overrides[short_name] = override
    return json.dumps(overrides, ensure_ascii=False)


def skills_to_markdown(skills: List[dict]) -> str:
    """将 Skills 转为人类可读的 Markdown 技能书报告。"""
    lines = [
        "# Skills 技能书报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**备件简称总数**: {len(skills)}",
        "",
        SKILLS_REPORT_STYLE,
        "",
        "---",
        "",
    ]

    for i, sk in enumerate(skills, 1):
        lines.append(f"## {i}. {_safe_report_text(sk['备件简称'])}")
        lines.append("")
        lines.append(f"- **适用算法**: {_safe_report_text(sk['适用算法'])}")
        lines.append(f"- **当前 σ 参数**: {_safe_report_text(sk['当前σ参数'])}")
        lines.append(f"- **偏置权重**: {_safe_report_text(sk['偏置权重'])}×")
        lines.append(f"- **时序敏感度 (Decay Alpha)**: {_safe_report_text(sk.get('时序敏感度 (Decay Alpha)', DEFAULT_DECAY_ALPHA))}")
        lines.append(f"- **圈子严格度 (Gap K)**: {_safe_report_text(sk.get('圈子严格度 (Gap K)', DEFAULT_GAP_K))}")
        lines.append(f"- **Baseline Quantile**: {_safe_report_text(sk.get('Baseline Quantile', DEFAULT_BASELINE_QUANTILE))}")
        lines.append(f"- **本组专家标注数**: {_safe_report_text(sk['本组专家标注数'])}")
        align = sk.get("经验对齐率", "N/A")
        if isinstance(align, float):
            lines.append(f"- **经验对齐率**: {align:.2%}")
        else:
            lines.append(f"- **经验对齐率**: {_safe_report_text(align)}")
        param_notes = [_safe_report_text(note) for note in sk.get("参数语义说明", []) if str(note).strip()]
        if param_notes:
            lines.append("- **参数语义说明**:")
            for note in param_notes:
                lines.append(f"  - {note}")
        semantic_report = _normalize_semantic_report(sk.get("语义校准报告", {}))
        semantic_refs = semantic_report.get("参考文本规律", [])
        semantic_modes = semantic_report.get("主要匹配方式", [])
        semantic_count = semantic_report.get("引用规律数", 0)
        lines.append("")

        bounds = sk["成本合理区间边界"]
        dist = sk["数据结构分布描述"]
        stats = sk["异常统计"]
        bounds_panel = _html_panel(
            "成本合理区间",
            _html_table(
                ["指标", "数值"],
                [
                    ["预测值（基准合理价）", bounds["预测值"]],
                    ["合理下限", bounds["合理下限"]],
                    ["合理上限", bounds["合理上限"]],
                ],
            ),
        )
        distribution_panel = _html_panel(
            "数据分布特征",
            _html_table(["统计量", "数值"], [[key, value] for key, value in dist.items()]),
        )
        lines.append(_html_grid(bounds_panel, distribution_panel))
        lines.append("")

        ring_intervals = sk.get("多邻居圈合理区间", [])
        if isinstance(ring_intervals, list) and ring_intervals:
            lines.append("### 多邻居圈合理区间")
            lines.append("")
            lines.append("| 圈层 | 角色 | 合理下限 | 基准价 | 合理上限 | 样本量 | 置信度 |")
            lines.append("|------|------|----------|--------|----------|--------|--------|")
            for ring in ring_intervals:
                lines.append(
                    "| {id} | {role} | {lower:,.4f} | {base:,.4f} | {upper:,.4f} | {count} | {confidence:.4f} |".format(
                        id=_safe_report_text(ring.get("圈层编号", "")),
                        role=_safe_report_text(ring.get("圈层角色", "")),
                        lower=float(ring.get("合理下限", 0) or 0),
                        base=float(ring.get("预测值", 0) or 0),
                        upper=float(ring.get("合理上限", 0) or 0),
                        count=int(ring.get("样本量", 0) or 0),
                        confidence=float(ring.get("圈层置信度", 0) or 0),
                    )
                )
            lines.append("")

        stats_panel = _html_panel(
            "异常统计",
            _html_table(["分类", "数量"], [[key, value] for key, value in stats.items()]),
        )
        lines.append(_html_grid(stats_panel, _semantic_report_panel(semantic_count, semantic_modes, semantic_refs)))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── 2. AutoResearch 棘轮迭代 ──────────────────────────────────────────────


def _score_params(
    result_df: pd.DataFrame,
    expert_labels: Dict[str, str],
) -> tuple:
    """准确率得分 + 冲突数。

    返回 (score, conflict_count, total_expert)。
    score = 自然落入合理区间的专家标注记录比例。
    conflict_count = 未落入合理区间的专家标注记录数（冲突数）。
    """
    normal_keys = {k for k, v in expert_labels.items() if v == "正常"}
    if not normal_keys:
        return 1.0, 0, 0

    total = 0
    correct = 0
    for key in normal_keys:
        rows = result_df[result_df["_record_key"] == key]
        if rows.empty:
            continue
        total += 1
        r = rows.iloc[0]
        if r["合理下限"] <= r["实际成本"] <= r["合理上限"]:
            correct += 1

    score = correct / total if total > 0 else 0.0
    conflict = total - correct
    return score, conflict, total


def _summarize_expert_short_names(result_df: pd.DataFrame, expert_labels: Dict[str, str]) -> str:
    if result_df is None or result_df.empty or not expert_labels:
        return ""
    if "_record_key" not in result_df.columns or "备件简称" not in result_df.columns:
        return ""

    matched = result_df[result_df["_record_key"].astype(str).isin({str(key) for key in expert_labels.keys()})]
    record_to_short_name = {
        str(row["_record_key"]): str(row["备件简称"]).strip()
        for _, row in matched[["_record_key", "备件简称"]].drop_duplicates(subset=["_record_key"]).iterrows()
    }
    short_names = [
        record_to_short_name.get(str(key), "")
        for key in expert_labels.keys()
    ]
    short_names = [value for value in short_names if value and value.lower() != "nan"]
    return "、".join(dict.fromkeys(short_names))


def run_auto_research(
    df: pd.DataFrame,
    price_col: str,
    expert_labels: Dict[str, str],
    n_iterations: int = 10,
    max_runtime_seconds: float = AUTORESEARCH_MAX_RUNTIME_SECONDS,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """棘轮循环自动调参。

    Parameters
    ----------
    df : 全量原始数据（含 price_col, monitor_date）。
    price_col : 价格列名。
    expert_labels : {record_key: label} 专家标注。
    n_iterations : 迭代轮数（5 / 10 / 20）。
    progress_callback : ``(current, total, best_score, trial_score, trial_params, best_params)`` 回调。

    Returns
    -------
    dict
        best_sigma, best_weight, best_decay_alpha, best_gap_k, best_score, history, result_df
    """
    labels_tuple = tuple(sorted(expert_labels.items()))
    started_at = time.perf_counter()

    log_event(
        "autoresearch",
        "start",
        "Started AutoResearch run",
        iteration_budget=int(n_iterations),
        expert_label_count=len(expert_labels),
        price_col=str(price_col),
        max_runtime_seconds=float(max_runtime_seconds),
    )

    if not expert_labels:
        baseline_df = detect_cost_anomalies_weighted(
            df,
            price_col,
            (),
            decay_alpha=DEFAULT_DECAY_ALPHA,
            gap_k=DEFAULT_GAP_K,
            baseline_quantile=DEFAULT_BASELINE_QUANTILE,
        )
        log_event(
            "autoresearch",
            "complete",
            "AutoResearch finished immediately because no expert labels were available",
            iteration_budget=int(n_iterations),
            best_sigma=1.0,
            best_weight=int(_EXPERT_WEIGHT),
            best_decay_alpha=float(DEFAULT_DECAY_ALPHA),
            best_gap_k=float(DEFAULT_GAP_K),
            best_baseline_quantile=float(DEFAULT_BASELINE_QUANTILE),
            best_score=1.0,
            best_conflicts=0,
        )
        return {
            "best_sigma": 1.0,
            "best_weight": _EXPERT_WEIGHT,
            "best_decay_alpha": round(float(DEFAULT_DECAY_ALPHA), 4),
            "best_gap_k": round(float(DEFAULT_GAP_K), 4),
            "best_baseline_quantile": round(float(DEFAULT_BASELINE_QUANTILE), 4),
            "best_score": 1.0,
            "best_conflicts": 0,
            "total_expert": 0,
            "history": [],
            "result_df": baseline_df,
            "search_strategy": "randomized_param_grid",
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }

    rng = np.random.RandomState(42)

    # ── 初始基线 ──
    best_sigma = 1.0
    best_weight = _EXPERT_WEIGHT
    best_decay_alpha = float(DEFAULT_DECAY_ALPHA)
    best_gap_k = float(DEFAULT_GAP_K)
    best_baseline_quantile = float(DEFAULT_BASELINE_QUANTILE)
    best_df = detect_cost_anomalies_weighted(
        df, price_col, labels_tuple,
        sigma_multiplier=best_sigma,
        expert_weight_override=best_weight,
        decay_alpha=best_decay_alpha,
        gap_k=best_gap_k,
        baseline_quantile=best_baseline_quantile,
    )
    best_score, best_conflicts, total_expert = _score_params(best_df, expert_labels)
    best_ring_count, best_interval_width = _result_ring_complexity(best_df)
    expert_short_name_scope = _summarize_expert_short_names(best_df, expert_labels)

    prev_best = {
        "sigma": best_sigma,
        "weight": best_weight,
        "decay_alpha": best_decay_alpha,
        "gap_k": best_gap_k,
        "baseline_quantile": best_baseline_quantile,
        "score": best_score,
        "conflicts": best_conflicts,
        "ring_count": best_ring_count,
        "interval_width": best_interval_width,
    }
    rollback_used = False
    candidates = _sample_autoresearch_candidates(rng, n_iterations)

    history = [
        {
            "迭代": 0,
            "备件简称": expert_short_name_scope,
            "σ系数": best_sigma,
            "偏置权重": best_weight,
            "时序衰减系数": round(best_decay_alpha, 4),
            "断层倍数": round(best_gap_k, 4),
            "基准分位点": round(best_baseline_quantile, 4),
            "得分": round(best_score, 4),
            "冲突数": best_conflicts,
            "采纳圈数": best_ring_count,
            "是否采纳": "✅",
            "备注": "初始基线",
        }
    ]

    for i, candidate in enumerate(candidates):
        if (time.perf_counter() - started_at) >= max_runtime_seconds:
            history.append(
                {
                    "迭代": i + 1,
                    "备件简称": expert_short_name_scope,
                    "σ系数": None,
                    "偏置权重": None,
                    "时序衰减系数": None,
                    "断层倍数": None,
                    "基准分位点": None,
                    "得分": None,
                    "冲突数": None,
                    "采纳圈数": None,
                    "是否采纳": "⏹️",
                    "备注": f"达到 {max_runtime_seconds:.0f} 秒预算，提前结束随机搜索",
                }
            )
            break

        trial_sigma = round(float(candidate["sigma"]), 4)
        trial_weight = int(candidate["weight"])
        trial_decay_alpha = round(float(candidate["decay_alpha"]), 4)
        trial_gap_k = round(float(candidate["gap_k"]), 4)
        trial_baseline_quantile = round(float(candidate["baseline_quantile"]), 4)
        trial_params = {
            "sigma": trial_sigma,
            "weight": trial_weight,
            "decay_alpha": trial_decay_alpha,
            "gap_k": trial_gap_k,
            "baseline_quantile": trial_baseline_quantile,
        }

        try:
            trial_df = detect_cost_anomalies_weighted(
                df, price_col, labels_tuple,
                sigma_multiplier=trial_sigma,
                expert_weight_override=trial_weight,
                decay_alpha=trial_decay_alpha,
                gap_k=trial_gap_k,
                baseline_quantile=trial_baseline_quantile,
            )
            trial_score, trial_conflicts, _ = _score_params(trial_df, expert_labels)
            trial_ring_count, trial_interval_width = _result_ring_complexity(trial_df)
        except Exception as exc:
            note = f"计算错误: {exc}"
            if not rollback_used:
                best_sigma = prev_best["sigma"]
                best_weight = prev_best["weight"]
                best_decay_alpha = prev_best["decay_alpha"]
                best_gap_k = prev_best["gap_k"]
                best_baseline_quantile = prev_best["baseline_quantile"]
                best_score = prev_best["score"]
                best_conflicts = prev_best["conflicts"]
                best_ring_count = prev_best["ring_count"]
                best_interval_width = prev_best["interval_width"]
                note += "（已回滚一个版本）"
                rollback_used = True
            history.append({
                "迭代": i + 1,
                "备件简称": expert_short_name_scope,
                "σ系数": trial_sigma,
                "偏置权重": trial_weight,
                "时序衰减系数": trial_decay_alpha,
                "断层倍数": trial_gap_k,
                "基准分位点": trial_baseline_quantile,
                "得分": None,
                "冲突数": None,
                "采纳圈数": None,
                "是否采纳": "❌",
                "备注": note,
            })
            log_event(
                "autoresearch",
                "iteration_error",
                "AutoResearch iteration failed",
                iteration=i + 1,
                sigma=trial_sigma,
                weight=trial_weight,
                decay_alpha=trial_decay_alpha,
                gap_k=trial_gap_k,
                baseline_quantile=trial_baseline_quantile,
                error=str(exc),
                rollback_used=rollback_used,
            )
            if progress_callback:
                progress_callback(
                    i + 1,
                    len(candidates),
                    best_score,
                    0.0,
                    trial_params,
                    {
                        "sigma": best_sigma,
                        "weight": best_weight,
                        "decay_alpha": best_decay_alpha,
                        "gap_k": best_gap_k,
                        "baseline_quantile": best_baseline_quantile,
                    },
                )
            continue

        accepted = _is_trial_better(
            trial_score,
            trial_conflicts,
            trial_decay_alpha,
            trial_ring_count,
            trial_interval_width,
            best_score,
            best_conflicts,
            best_decay_alpha,
            best_ring_count,
            best_interval_width,
        )
        if accepted:
            prev_best = {
                "sigma": best_sigma,
                "weight": best_weight,
                "decay_alpha": best_decay_alpha,
                "gap_k": best_gap_k,
                "baseline_quantile": best_baseline_quantile,
                "score": best_score,
                "conflicts": best_conflicts,
                "ring_count": best_ring_count,
                "interval_width": best_interval_width,
            }
            best_sigma = trial_sigma
            best_weight = trial_weight
            best_decay_alpha = trial_decay_alpha
            best_gap_k = trial_gap_k
            best_baseline_quantile = trial_baseline_quantile
            best_score = trial_score
            best_conflicts = trial_conflicts
            best_ring_count = trial_ring_count
            best_interval_width = trial_interval_width
            best_df = trial_df
            note = (
                f"得分 {prev_best['score']:.2%}→{best_score:.2%}；"
                f"α {prev_best['decay_alpha']:.4f}→{best_decay_alpha:.4f}；"
                f"GapK {prev_best['gap_k']:.4f}→{best_gap_k:.4f}；"
                f"Q {prev_best['baseline_quantile']:.4f}→{best_baseline_quantile:.4f}"
            )
        else:
            note = (
                f"得分 {trial_score:.2%} 未优于 {best_score:.2%}；"
                f"α={trial_decay_alpha:.4f}，GapK={trial_gap_k:.4f}，Q={trial_baseline_quantile:.4f}"
            )

        history.append({
            "迭代": i + 1,
            "备件简称": expert_short_name_scope,
            "σ系数": trial_sigma,
            "偏置权重": trial_weight,
            "时序衰减系数": trial_decay_alpha,
            "断层倍数": trial_gap_k,
            "基准分位点": trial_baseline_quantile,
            "得分": round(trial_score, 4),
            "冲突数": trial_conflicts,
            "采纳圈数": trial_ring_count,
            "是否采纳": "✅" if accepted else "❌",
            "备注": note,
        })

        log_event(
            "autoresearch",
            "iteration",
            "Completed an AutoResearch iteration",
            iteration=i + 1,
            sigma=trial_sigma,
            weight=trial_weight,
            decay_alpha=trial_decay_alpha,
            gap_k=trial_gap_k,
            baseline_quantile=trial_baseline_quantile,
            trial_score=round(float(trial_score), 4),
            trial_conflicts=int(trial_conflicts),
            trial_ring_count=int(trial_ring_count),
            accepted=accepted,
            best_sigma=round(float(best_sigma), 4),
            best_weight=int(best_weight),
            best_decay_alpha=round(float(best_decay_alpha), 4),
            best_gap_k=round(float(best_gap_k), 4),
            best_baseline_quantile=round(float(best_baseline_quantile), 4),
            best_score=round(float(best_score), 4),
            best_conflicts=int(best_conflicts),
            best_ring_count=int(best_ring_count),
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
        )

        if progress_callback:
            progress_callback(
                i + 1,
                len(candidates),
                best_score,
                trial_score,
                trial_params,
                {
                    "sigma": best_sigma,
                    "weight": best_weight,
                    "decay_alpha": best_decay_alpha,
                    "gap_k": best_gap_k,
                    "baseline_quantile": best_baseline_quantile,
                },
            )

        if best_score >= 1.0 and best_conflicts == 0:
            break

    result = {
        "best_sigma": round(best_sigma, 4),
        "best_weight": best_weight,
        "best_decay_alpha": round(best_decay_alpha, 4),
        "best_gap_k": round(best_gap_k, 4),
        "best_baseline_quantile": round(best_baseline_quantile, 4),
        "best_score": round(best_score, 4),
        "best_conflicts": best_conflicts,
        "best_ring_count": best_ring_count,
        "total_expert": total_expert,
        "history": history,
        "result_df": best_df,
        "search_strategy": "randomized_param_grid",
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    log_event(
        "autoresearch",
        "complete",
        "Completed AutoResearch run",
        iteration_budget=int(n_iterations),
        executed_iterations=max(len(history) - 1, 0),
        best_sigma=result["best_sigma"],
        best_weight=result["best_weight"],
        best_decay_alpha=result["best_decay_alpha"],
        best_gap_k=result["best_gap_k"],
        best_baseline_quantile=result["best_baseline_quantile"],
        best_score=result["best_score"],
        best_conflicts=result["best_conflicts"],
        best_ring_count=result["best_ring_count"],
        total_expert=result["total_expert"],
        elapsed_seconds=result["elapsed_seconds"],
        search_strategy=result["search_strategy"],
    )
    return result


# ── 3. 深度审计报表 ──────────────────────────────────────────────────────


def generate_audit_report(
    original_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    expert_labels: Dict[str, str],
) -> pd.DataFrame:
    """生成本轮标注记录的深度审计报表。

    字段：物料编码、物料名称、备件简称、成本数值、原始结论、专家反馈、最终优化结论
    """
    labeled_keys = {str(key) for key in (expert_labels or {}).keys()}
    source_df = original_df.copy()
    if labeled_keys and "_record_key" in source_df.columns:
        source_df = source_df[source_df["_record_key"].astype(str).isin(labeled_keys)].copy()
    elif expert_labels:
        source_df = source_df.iloc[0:0].copy()

    report = source_df[["_record_key", "物料编码"]].copy()

    report["物料名称"] = (
        source_df["物料名称"].values
        if "物料名称" in source_df.columns
        else "未知"
    )
    report["备件简称"] = source_df["备件简称"].values
    report["成本数值"] = source_df["实际成本"].values
    report["原始结论"] = source_df["status"].values

    report["专家反馈"] = report["_record_key"].map(
        lambda k: expert_labels.get(k, "未标注")
    )

    opt_map = dict(zip(optimized_df["_record_key"], optimized_df["status"]))
    report["最终优化结论"] = report["_record_key"].map(
        lambda k: opt_map.get(k, "—")
    )

    if "AI 辅助分析" in optimized_df.columns:
        ai_map = dict(zip(optimized_df["_record_key"], optimized_df["AI 辅助分析"].fillna("")))
        report["AI辅助分析"] = report["_record_key"].map(lambda k: ai_map.get(k, ""))

    report = report.drop(columns=["_record_key"])
    return report
