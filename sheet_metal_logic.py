from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from anomaly_engine import _EXPERT_WEIGHT, detect_dgb_anomalies, detect_dgb_anomalies_weighted
from local_logging import log_event
from skills_excel_export import flatten_skills_for_excel, skills_to_excel_bytes as _skills_to_excel_bytes
from storage_service import split_metric_record_key


_STEEL_DENSITY = 7.6
_EXCEL_PATTERNS = ("*.xlsx", "*.xls", "*.xlsm")
_STEEL_MARKET_CATEGORIES = ("热轧板卷", "冷轧板", "镀锌板", "中厚板")
_STEEL_MARKET_URLS = {
    "热轧板卷": "https://rejuan.100ppi.com/",
    "冷轧板": "https://lyb.100ppi.com/",
    "镀锌板": "https://dxb.100ppi.com/",
    "中厚板": "https://zhb.100ppi.com/",
}
SHEET_METAL_NON_MATERIAL_OUTPUT_COLUMNS = [
    "物料编码",
    "物料名称",
    "备件简称",
    "样本数",
    "材料锚点",
    "材料时令价格",
    "成本",
    "重量",
    "白痴指数",
    "非材料成本系数",
]
_NON_MATERIAL_EXCLUDE_KEYS = [
    "not_reasonable",
    "cost_missing",
    "weight_missing",
    "weight_invalid",
    "steel_anchor_missing",
    "material_cost_invalid",
    "short_name_missing",
]
_SHEET_METAL_COST_WEIGHT_FIELDS = ["产品成本", "出厂单价", "净重", "包装后重量"]
_COLUMN_ALIASES = {
    "车型": ["车型"],
    "物料编码": ["物料编码", "物料号", "料号", "零件号", "零件编码"],
    "物料描述": ["物料描述", "物料名称", "零件名称", "品名描述", "名称"],
    "产品成本": ["产品成本", "成本", "产品单价"],
    "备件简称": ["备件简称", "简称", "零件简称"],
    "车系": ["车系", "适用车系"],
    "车型梯度": ["车型梯度", "梯度"],
    "工厂": ["工厂", "工厂名称", "工厂代码"],
    "出厂单价": ["出厂单价", "出厂价", "采购单价", "单价", "价格"],
    "包装费": ["包装费"],
    "净重": ["净重", "净重(g)", "重量", "重量(g)"],
    "包装后重量": ["包装后重量", "包装后重量(g)", "毛重", "毛重(g)"],
    "白痴指数": ["白痴指数", "钣金件白痴指数"],
}


def _pick_first_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(column).strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _apply_alias_renames(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]
    rename_map: Dict[str, str] = {}
    for target_name, aliases in _COLUMN_ALIASES.items():
        if target_name in data.columns:
            continue
        matched = _pick_first_column(data.columns, aliases)
        if matched and matched != target_name:
            rename_map[matched] = target_name
    if rename_map:
        data = data.rename(columns=rename_map)
    return data


def _clean_text_series(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    normalized = normalized.map(lambda value: np.nan if pd.isna(value) else str(value).strip())
    return normalized.replace({"": np.nan, "nan": np.nan, "None": np.nan})


def _first_nonempty_text_series(data: pd.DataFrame, *column_names: str, default: str | None = None) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype="object")
    for column_name in column_names:
        if column_name not in data.columns:
            continue
        candidate = _clean_text_series(data[column_name])
        result = result.where(result.notna(), candidate)
    if default is not None:
        result = result.fillna(default)
    return result


def _compute_sheet_metal_index(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "出厂单价" not in result.columns:
        result["出厂单价"] = np.nan
    if "净重" not in result.columns:
        result["净重"] = np.nan
    if "白痴指数" not in result.columns:
        result["白痴指数"] = np.nan

    result["出厂单价"] = pd.to_numeric(result["出厂单价"], errors="coerce")
    result["净重"] = pd.to_numeric(result["净重"], errors="coerce")
    result["白痴指数"] = pd.to_numeric(result["白痴指数"], errors="coerce")

    denominator = (result["净重"] / 1000.0) * _STEEL_DENSITY
    computed_index = result["出厂单价"] / denominator.replace(0, pd.NA)
    fill_mask = result["白痴指数"].isna() & computed_index.notna()
    result.loc[fill_mask, "白痴指数"] = computed_index.loc[fill_mask]
    return result


def _empty_non_material_result(summary: dict[str, int] | None = None) -> pd.DataFrame:
    result = pd.DataFrame(columns=SHEET_METAL_NON_MATERIAL_OUTPUT_COLUMNS)
    base_summary = {key: 0 for key in _NON_MATERIAL_EXCLUDE_KEYS}
    if summary:
        for key, value in summary.items():
            base_summary[str(key)] = int(value or 0)
    result.attrs["excluded_summary"] = base_summary
    return result


def _coerce_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    try:
        number = float(number)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _coerce_numeric_series(data: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column_name], errors="coerce")


def _first_numeric_series(data: pd.DataFrame, *column_names: str) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype="float64")
    for column_name in column_names:
        candidate = _coerce_numeric_series(data, column_name)
        result = result.where(result.notna(), candidate)
    return result


def _parse_price_quotes_from_html(html: str) -> list[float]:
    text = re.sub(r"\s+", " ", str(html or ""))
    values: list[float] = []
    occupied_spans: list[tuple[int, int]] = []

    def append_match(match: re.Match[str]) -> None:
        number = _coerce_float(match.group(1))
        if number is not None and 1000 <= number <= 20000:
            values.append(float(number))
            occupied_spans.append(match.span())

    for match in re.finditer(r"(?:参考价(?:格)?|均价|报价|价格)\s*(?:为|是|:|：)?\s*(\d{3,6}(?:\.\d+)?)", text):
        append_match(match)

    for match in re.finditer(r"(?<!\d)(\d{3,6}(?:\.\d+)?)\s*(?:元\s*/?\s*吨|元/吨)", text):
        if any(match.start() < span_end and span_start < match.end() for span_start, span_end in occupied_spans):
            continue
        append_match(match)
    return values


def _extract_100ppi_security_cookie(html: str) -> str | None:
    if "HW_CHECK" not in str(html or ""):
        return None
    match = re.search(r"var\s+_0x2\s*=\s*['\"]([^'\"]+)['\"]", str(html or ""))
    if not match:
        return None
    return match.group(1)


def _fetch_steel_market_html(source_url: str, *, session: Any, timeout: float) -> str:
    response = session.get(source_url, timeout=timeout)
    response.raise_for_status()
    html = response.text
    security_cookie = _extract_100ppi_security_cookie(html)
    if security_cookie:
        session.cookies.set("HW_CHECK", security_cookie, domain=".100ppi.com", path="/")
        response = session.get(source_url, timeout=timeout)
        response.raise_for_status()
        html = response.text
    return html


def _normalize_quote_values(raw_quotes: Any) -> list[float]:
    if raw_quotes is None:
        return []
    if isinstance(raw_quotes, (str, bytes)):
        raw_iterable = [raw_quotes]
    elif isinstance(raw_quotes, Iterable):
        raw_iterable = list(raw_quotes)
    else:
        raw_iterable = [raw_quotes]

    values: list[float] = []
    for item in raw_iterable:
        number = _coerce_float(item)
        if number is not None and number > 0:
            values.append(float(number))
    return values


def _normalize_steel_market_anchor(steel_anchor: dict | None) -> dict:
    payload = dict(steel_anchor or {})
    raw_categories = payload.get("categories") or payload.get("钢材大类") or []
    normalized_categories: list[dict[str, Any]] = []

    if isinstance(raw_categories, dict):
        category_items = [{"category": key, "quotes": value} for key, value in raw_categories.items()]
    else:
        category_items = list(raw_categories or [])

    for item in category_items:
        if not isinstance(item, dict):
            continue
        category_name = str(item.get("category") or item.get("name") or item.get("大类") or item.get("钢材大类") or "").strip()
        quotes = _normalize_quote_values(
            item.get("quotes")
            or item.get("报价")
            or item.get("价格列表")
            or item.get("均价")
            or item.get("average")
            or item.get("price")
            or item.get("价格")
        )
        if not category_name or not quotes:
            continue
        category_average = float(np.mean(quotes))
        normalized_categories.append(
            {
                "category": category_name,
                "quotes": quotes,
                "average": round(category_average, 4),
                "source": item.get("source") or item.get("来源") or payload.get("source") or "",
                "date": item.get("date") or item.get("日期") or payload.get("date") or "",
            }
        )

    explicit_average = _coerce_float(
        payload.get("average_price_per_ton")
        or payload.get("材料时令价格")
        or payload.get("均价")
        or payload.get("price_per_ton")
    )
    category_average_values = [float(item["average"]) for item in normalized_categories]
    average_price = float(np.mean(category_average_values)) if category_average_values else explicit_average
    anchor_categories = [item["category"] for item in normalized_categories] or list(_STEEL_MARKET_CATEGORIES)

    normalized = {
        "categories": normalized_categories,
        "source": str(payload.get("source") or payload.get("来源") or "").strip(),
        "date": str(payload.get("date") or payload.get("日期") or "").strip(),
        "average_price_per_ton": round(float(average_price), 4) if average_price is not None and average_price > 0 else None,
        "anchor_label": payload.get("anchor_label") or payload.get("材料锚点") or f"钢材均价锚点：{'/'.join(anchor_categories)}",
    }
    return normalized


def load_sheet_metal_steel_market_anchor(
    manual_prices: dict[str, Any] | None = None,
    *,
    as_of_date: datetime | pd.Timestamp | str | None = None,
    timeout: float = 5.0,
    session_factory: Callable[[], Any] | None = None,
    retries: int = 2,
) -> dict:
    """Load the steel market anchor from public pages, with manual prices as the stable fallback."""
    if manual_prices:
        categories = [{"category": name, "quotes": values, "source": "manual"} for name, values in manual_prices.items()]
        return _normalize_steel_market_anchor(
            {
                "categories": categories,
                "source": "manual",
                "date": str(pd.Timestamp(as_of_date or datetime.now()).date()),
            }
        )

    categories: list[dict[str, Any]] = []
    try:
        import requests
    except Exception as exc:
        log_event("sheet_metal_steel_anchor", "fetch_failed", "Failed to import requests for public steel market anchor", error=str(exc))
        requests = None  # type: ignore[assignment]

    if requests is not None:
        session = session_factory() if session_factory else requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        attempt_count = max(1, int(retries or 1))
        for category_name, source_url in _STEEL_MARKET_URLS.items():
            quotes: list[float] = []
            last_error: Exception | None = None
            for _ in range(attempt_count):
                try:
                    html = _fetch_steel_market_html(source_url, session=session, timeout=timeout)
                    quotes = _parse_price_quotes_from_html(html)
                    if quotes:
                        break
                except Exception as exc:
                    last_error = exc
            if not quotes and last_error is not None:
                log_event(
                    "sheet_metal_steel_anchor",
                    "category_fetch_failed",
                    "Failed to fetch public steel market category",
                    steel_category=category_name,
                    source=source_url,
                    error=str(last_error),
                )
            if quotes:
                categories.append(
                    {
                        "category": category_name,
                        "quotes": quotes,
                        "source": source_url,
                        "date": str(pd.Timestamp(as_of_date or datetime.now()).date()),
                    }
                )

    source = "public_web" if categories else "manual_required"
    return _normalize_steel_market_anchor(
        {
            "categories": categories,
            "source": source,
            "date": str(pd.Timestamp(as_of_date or datetime.now()).date()),
        }
    )


def build_reasonable_sheet_metal_samples(review_df: pd.DataFrame) -> pd.DataFrame:
    if review_df is None or review_df.empty:
        return _empty_non_material_result()

    data = review_df.copy()
    if "status" not in data.columns:
        data["status"] = ""
    for column_name in ["白痴指数", "合理下限", "合理上限"]:
        data[column_name] = _coerce_numeric_series(data, column_name)

    status_normal = data["status"].astype(str).str.contains("正常", na=False)
    in_bounds = data["白痴指数"].notna() & data["合理下限"].notna() & data["合理上限"].notna()
    in_bounds &= data["白痴指数"].between(data["合理下限"], data["合理上限"], inclusive="both")
    reasonable_mask = status_normal | in_bounds

    samples = data[reasonable_mask].copy()
    samples.attrs["excluded_summary"] = {"not_reasonable": int((~reasonable_mask).sum())}
    return samples


def calculate_non_material_coefficients(samples_df: pd.DataFrame, steel_anchor: dict | None) -> pd.DataFrame:
    if samples_df is None or samples_df.empty:
        summary = dict(getattr(samples_df, "attrs", {}).get("excluded_summary", {}) if samples_df is not None else {})
        return _empty_non_material_result(summary)

    data = samples_df.copy()
    summary = {key: 0 for key in _NON_MATERIAL_EXCLUDE_KEYS}
    summary.update({str(key): int(value or 0) for key, value in (samples_df.attrs.get("excluded_summary") or {}).items()})
    normalized_anchor = _normalize_steel_market_anchor(steel_anchor)
    material_price_per_ton = _coerce_float(normalized_anchor.get("average_price_per_ton"))
    if material_price_per_ton is None or material_price_per_ton <= 0:
        summary["steel_anchor_missing"] += int(len(data))
        return _empty_non_material_result(summary)

    if "物料名称" not in data.columns and "物料描述" in data.columns:
        data["物料名称"] = data["物料描述"]
    for column_name in ["物料编码", "物料名称", "备件简称"]:
        if column_name not in data.columns:
            data[column_name] = ""

    data["成本"] = _first_numeric_series(data, "出厂单价", "产品成本")
    data["重量"] = _first_numeric_series(data, "净重", "包装后重量")
    data["白痴指数"] = _coerce_numeric_series(data, "白痴指数")
    short_names = _clean_text_series(data["备件简称"])

    cost_missing = data["成本"].isna()
    weight_missing = data["重量"].isna()
    weight_invalid = data["重量"].notna() & data["重量"].le(0)
    short_name_missing = short_names.isna()

    material_price_per_kg = float(material_price_per_ton) / 1000.0
    material_cost = (data["重量"] / 1000.0) * material_price_per_kg
    material_cost_invalid = material_cost.isna() | material_cost.le(0)

    invalid_mask = cost_missing | weight_missing | weight_invalid | short_name_missing | material_cost_invalid
    summary["cost_missing"] += int(cost_missing.sum())
    summary["weight_missing"] += int(weight_missing.sum())
    summary["weight_invalid"] += int(weight_invalid.sum())
    summary["short_name_missing"] += int(short_name_missing.sum())
    summary["material_cost_invalid"] += int((material_cost_invalid & ~weight_missing & ~weight_invalid).sum())

    valid_data = data[~invalid_mask].copy()
    if valid_data.empty:
        return _empty_non_material_result(summary)

    valid_material_cost = material_cost.loc[valid_data.index].astype(float)
    valid_data["备件简称"] = short_names.loc[valid_data.index].astype(str)
    valid_data["_单行非材料成本系数"] = (valid_data["成本"].astype(float) / valid_material_cost) - 1.0
    group_coefficients = valid_data.groupby("备件简称")["_单行非材料成本系数"].transform("mean")
    group_sample_counts = valid_data.groupby("备件简称")["备件简称"].transform("size")

    result = pd.DataFrame(
        {
            "物料编码": valid_data["物料编码"].astype(str),
            "物料名称": _first_nonempty_text_series(valid_data, "物料名称", "物料描述", default=""),
            "备件简称": valid_data["备件简称"].astype(str),
            "样本数": group_sample_counts.astype(int),
            "材料锚点": str(normalized_anchor.get("anchor_label") or f"钢材均价锚点：{'/'.join(_STEEL_MARKET_CATEGORIES)}"),
            "材料时令价格": round(float(material_price_per_ton), 4),
            "成本": valid_data["成本"].astype(float).round(4),
            "重量": valid_data["重量"].astype(float).round(4),
            "白痴指数": valid_data["白痴指数"].astype(float).round(4),
            "非材料成本系数": group_coefficients.astype(float).round(6),
        }
    )
    result = result[SHEET_METAL_NON_MATERIAL_OUTPUT_COLUMNS].reset_index(drop=True)
    result.attrs["excluded_summary"] = summary
    result.attrs["steel_anchor"] = normalized_anchor
    return result


def _append_sheet_metal_cost_weight_fields(result_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    if result_df is None or result_df.empty or source_df is None or source_df.empty:
        return result_df
    if "物料编码" not in result_df.columns or "物料编码" not in source_df.columns:
        return result_df

    source_columns = ["物料编码"] + [column for column in _SHEET_METAL_COST_WEIGHT_FIELDS if column in source_df.columns]
    if len(source_columns) <= 1:
        return result_df

    source_lookup = source_df[source_columns].copy()
    source_lookup["物料编码"] = source_lookup["物料编码"].astype(str)
    source_lookup = source_lookup.drop_duplicates(subset=["物料编码"], keep="last")

    data = result_df.copy()
    data["物料编码"] = data["物料编码"].astype(str)
    data = data.merge(source_lookup, on="物料编码", how="left", suffixes=("", "_源数据"))
    for column_name in _SHEET_METAL_COST_WEIGHT_FIELDS:
        source_column = f"{column_name}_源数据"
        if column_name not in data.columns:
            data[column_name] = np.nan
        if source_column in data.columns:
            data[column_name] = data[column_name].where(data[column_name].notna(), data[source_column])
            data = data.drop(columns=[source_column])
    return data


def _normalize_sheet_frame(
    raw_df: pd.DataFrame,
    *,
    source_file: str,
    source_sheet: str,
    snapshot_time: pd.Timestamp,
) -> pd.DataFrame:
    data = _apply_alias_renames(raw_df)
    if "物料编码" not in data.columns:
        return pd.DataFrame()

    for column_name in [
        "车型",
        "车系",
        "车型梯度",
        "物料描述",
        "产品成本",
        "出厂单价",
        "包装费",
        "净重",
        "包装后重量",
        "白痴指数",
        "备件简称",
        "工厂",
    ]:
        if column_name not in data.columns:
            data[column_name] = np.nan

    data = _compute_sheet_metal_index(data)

    data["物料编码"] = _clean_text_series(data["物料编码"]).fillna("")
    data = data[data["物料编码"] != ""].copy()
    if data.empty:
        return pd.DataFrame()

    data["物料描述"] = _first_nonempty_text_series(data, "物料描述").fillna(data["物料编码"])
    data["物料名称"] = data["物料描述"]
    data["备件简称"] = _first_nonempty_text_series(data, "备件简称").fillna(data["物料描述"])
    data["车型"] = _first_nonempty_text_series(data, "车型")
    data["车系"] = _first_nonempty_text_series(data, "车系").fillna(data["车型"])
    data["适用车系"] = _first_nonempty_text_series(data, "车系", "车型", default="未识别车系")
    data["车型梯度"] = _first_nonempty_text_series(data, "车型梯度")
    data["工厂"] = _first_nonempty_text_series(data, "工厂", default="钣金")

    for numeric_column in ["产品成本", "出厂单价", "包装费", "净重", "包装后重量", "白痴指数"]:
        data[numeric_column] = pd.to_numeric(data[numeric_column], errors="coerce")

    data["白痴指数"] = pd.to_numeric(data["白痴指数"], errors="coerce")
    data = data.dropna(subset=["白痴指数"])
    if data.empty:
        return pd.DataFrame()

    data["monitor_date"] = pd.to_datetime(snapshot_time, errors="coerce")
    data["静态快照时间"] = data["monitor_date"]
    data["数据来源文件"] = source_file
    data["数据来源工作表"] = source_sheet
    return data[
        [
            "车型",
            "车系",
            "车型梯度",
            "物料编码",
            "物料描述",
            "物料名称",
            "备件简称",
            "产品成本",
            "包装费",
            "包装后重量",
            "适用车系",
            "工厂",
            "出厂单价",
            "净重",
            "白痴指数",
            "monitor_date",
            "静态快照时间",
            "数据来源文件",
            "数据来源工作表",
        ]
    ].copy()


def load_sheet_metal_base_data(folder_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    normalized_path = str(folder_path or "").strip()
    if not normalized_path:
        return None, "请先配置钣金件基础数据路径"
    if not os.path.isdir(normalized_path):
        return None, f"钣金件基础数据路径不存在: {normalized_path}"

    excel_files: List[str] = []
    for pattern in _EXCEL_PATTERNS:
        excel_files.extend(glob.glob(os.path.join(normalized_path, pattern)))
    excel_files = sorted(set(excel_files))
    if not excel_files:
        return None, "钣金件基础数据路径下没有找到 Excel 文件"

    frames: List[pd.DataFrame] = []
    for file_path in excel_files:
        snapshot_time = pd.Timestamp(datetime.fromtimestamp(Path(file_path).stat().st_mtime))
        try:
            workbook = pd.read_excel(file_path, sheet_name=None)
        except Exception:
            continue
        for sheet_name, raw_df in workbook.items():
            if raw_df is None or raw_df.empty:
                continue
            normalized = _normalize_sheet_frame(
                raw_df,
                source_file=os.path.basename(file_path),
                source_sheet=str(sheet_name),
                snapshot_time=snapshot_time,
            )
            if not normalized.empty:
                frames.append(normalized)

    if not frames:
        return None, "没有成功读取到有效的钣金件基础数据"

    merged = pd.concat(frames, ignore_index=True)
    merged["_row_order"] = np.arange(len(merged))
    merged = merged.drop_duplicates(subset=["物料编码"], keep="last")
    merged = merged.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
    return merged, None


def detect_sheet_metal_anomalies(
    base_df: pd.DataFrame,
    *,
    expert_labels: Dict[str, str] | None = None,
    optimized: bool = False,
    sigma_multiplier: float = 1.0,
    expert_weight_override: int = 0,
    skills_overrides_json: str = "",
) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return pd.DataFrame(
            columns=[
                "_record_key",
                "车型",
                "车系",
                "车型梯度",
                "物料编码",
                "物料描述",
                "物料名称",
                "适用车系",
                "工厂",
                "备件简称",
                "产品成本",
                "出厂单价",
                "包装费",
                "净重",
                "包装后重量",
                "白痴指数",
                "静态快照时间",
                "样本量",
                "基准指数",
                "合理下限",
                "合理上限",
                "偏离指数",
                "偏离比例",
                "status",
            ]
        )

    labels_tuple = tuple(sorted((expert_labels or {}).items()))
    use_skills_overrides = bool(str(skills_overrides_json or "").strip())
    if (optimized and expert_labels) or use_skills_overrides:
        result_df = detect_dgb_anomalies_weighted(
            base_df,
            target_column="白痴指数",
            expert_labels_tuple=labels_tuple,
            sigma_multiplier=sigma_multiplier,
            expert_weight_override=expert_weight_override,
            skills_overrides_json=skills_overrides_json,
            value_label="白痴指数",
            baseline_label="基准指数",
            deviation_label="偏离指数",
            date_label="静态快照时间",
        )
    else:
        result_df = detect_dgb_anomalies(
            base_df,
            target_column="白痴指数",
            value_label="白痴指数",
            baseline_label="基准指数",
            deviation_label="偏离指数",
            date_label="静态快照时间",
        )

    if "静态快照时间" in result_df.columns:
        result_df["静态快照时间"] = pd.to_datetime(result_df["静态快照时间"], errors="coerce")
    if "_record_key" in result_df.columns:
        result_df["_record_key"] = result_df["_record_key"].astype(str)
    result_df = _append_sheet_metal_cost_weight_fields(result_df, base_df)
    if "专家校准" not in result_df.columns:
        result_df["专家校准"] = ""
    if "判定依据" not in result_df.columns:
        result_df["判定依据"] = ""
    result_df["判定依据"] = result_df["判定依据"].replace({"技能书校验": "[技能书校验]"}).fillna("")
    return result_df


def build_sheet_metal_skills_overrides_json(skills_data: Optional[Dict]) -> str:
    if not isinstance(skills_data, dict):
        return ""

    overrides = {}
    for skill in skills_data.get("skills", []) or []:
        short_name = str(skill.get("备件简称", "") or "").strip()
        if not short_name:
            continue

        override = {}
        sigma_value = pd.to_numeric(skill.get("当前σ参数"), errors="coerce")
        weight_value = pd.to_numeric(skill.get("偏置权重"), errors="coerce")
        decay_alpha_value = pd.to_numeric(skill.get("时序敏感度 (Decay Alpha)"), errors="coerce")
        gap_k_value = pd.to_numeric(skill.get("圈子严格度 (Gap K)"), errors="coerce")
        if pd.notna(sigma_value):
            override["sigma"] = float(sigma_value)
        if pd.notna(weight_value):
            override["weight"] = int(weight_value)
        if pd.notna(decay_alpha_value):
            override["decay_alpha"] = float(decay_alpha_value)
        if pd.notna(gap_k_value):
            override["gap_k"] = float(gap_k_value)
        if override:
            overrides[short_name] = override

    return json.dumps(overrides, ensure_ascii=False) if overrides else ""


def score_sheet_metal_alignment(
    result_df: pd.DataFrame,
    expert_labels: Dict[str, str],
) -> Tuple[float, int, int]:
    normal_keys = {key for key, label in expert_labels.items() if label == "正常"}
    if not normal_keys:
        return 0.0, 0, 0

    total = 0
    correct = 0
    for key in normal_keys:
        rows = result_df[result_df["_record_key"] == key]
        if rows.empty:
            continue
        row = rows.iloc[0]
        index_value = pd.to_numeric(pd.Series([row.get("白痴指数")]), errors="coerce").iloc[0]
        lower = pd.to_numeric(pd.Series([row.get("合理下限")]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([row.get("合理上限")]), errors="coerce").iloc[0]
        if pd.isna(index_value) or pd.isna(lower) or pd.isna(upper):
            continue
        total += 1
        if float(lower) <= float(index_value) <= float(upper):
            correct += 1

    if total <= 0:
        return 0.0, 0, 0
    score = correct / total
    conflicts = total - correct
    return round(score, 4), conflicts, total


def run_sheet_metal_auto_research(
    base_df: pd.DataFrame,
    expert_labels: Dict[str, str],
    n_iterations: int = 10,
    progress_callback: Optional[Callable[[int, int, float, float, float, int], None]] = None,
) -> dict:
    labels = dict(expert_labels or {})
    log_event_payload = {
        "iteration_budget": int(n_iterations),
        "expert_label_count": len(labels),
    }
    log_event("sheet_metal_autoresearch", "start", "Started sheet metal AutoResearch run", **log_event_payload)

    if not labels:
        baseline_df = detect_sheet_metal_anomalies(base_df, expert_labels={}, optimized=False)
        log_event(
            "sheet_metal_autoresearch",
            "complete",
            "Sheet metal AutoResearch finished immediately because no expert labels were available",
            **log_event_payload,
            best_sigma=1.0,
            best_weight=int(_EXPERT_WEIGHT),
            best_score=1.0,
            best_conflicts=0,
        )
        return {
            "best_sigma": 1.0,
            "best_weight": int(_EXPERT_WEIGHT),
            "best_score": 1.0,
            "best_conflicts": 0,
            "total_expert": 0,
            "history": [],
            "result_df": baseline_df,
        }

    rng = np.random.RandomState(42)
    best_sigma = 1.0
    best_weight = int(_EXPERT_WEIGHT)
    best_df = detect_sheet_metal_anomalies(
        base_df,
        expert_labels=labels,
        optimized=True,
        sigma_multiplier=best_sigma,
        expert_weight_override=best_weight,
    )
    best_score, best_conflicts, total_expert = score_sheet_metal_alignment(best_df, labels)
    prev_best = {"sigma": best_sigma, "weight": best_weight, "score": best_score, "conflicts": best_conflicts}
    rollback_used = False

    history = [
        {
            "迭代": 0,
            "σ系数": best_sigma,
            "偏置权重": best_weight,
            "得分": round(best_score, 4),
            "冲突数": best_conflicts,
            "是否采纳": "✅",
            "备注": "初始基线",
        }
    ]

    for index in range(n_iterations):
        trial_sigma = best_sigma * (1 + rng.uniform(-0.3, 0.4))
        trial_weight = best_weight + int(rng.randint(-40, 80))
        trial_sigma = max(0.1, min(5.0, round(trial_sigma, 4)))
        trial_weight = max(1, min(500, trial_weight))

        try:
            trial_df = detect_sheet_metal_anomalies(
                base_df,
                expert_labels=labels,
                optimized=True,
                sigma_multiplier=trial_sigma,
                expert_weight_override=trial_weight,
            )
            trial_score, trial_conflicts, _ = score_sheet_metal_alignment(trial_df, labels)
        except Exception as exc:
            note = f"计算错误: {exc}"
            if not rollback_used:
                best_sigma = prev_best["sigma"]
                best_weight = prev_best["weight"]
                best_score = prev_best["score"]
                best_conflicts = prev_best["conflicts"]
                note += "（已回滚一个版本）"
                rollback_used = True
            history.append(
                {
                    "迭代": index + 1,
                    "σ系数": trial_sigma,
                    "偏置权重": trial_weight,
                    "得分": None,
                    "冲突数": None,
                    "是否采纳": "❌",
                    "备注": note,
                }
            )
            log_event(
                "sheet_metal_autoresearch",
                "iteration_error",
                "Sheet metal AutoResearch iteration failed",
                iteration=index + 1,
                sigma=trial_sigma,
                weight=trial_weight,
                error=str(exc),
                rollback_used=rollback_used,
            )
            if progress_callback:
                progress_callback(index + 1, n_iterations, best_score, 0.0, best_sigma, best_weight)
            continue

        accepted = (trial_conflicts < best_conflicts) or (
            trial_conflicts == best_conflicts and trial_score > best_score
        )
        if accepted:
            prev_best = {"sigma": best_sigma, "weight": best_weight, "score": best_score, "conflicts": best_conflicts}
            best_sigma = trial_sigma
            best_weight = trial_weight
            best_score = trial_score
            best_conflicts = trial_conflicts
            best_df = trial_df
            note = f"冲突 {prev_best['conflicts']}→{best_conflicts} 得分 {prev_best['score']:.2%}→{best_score:.2%}"
        else:
            note = f"冲突 {trial_conflicts}≥{best_conflicts} 得分 {trial_score:.2%}≤{best_score:.2%}"

        history.append(
            {
                "迭代": index + 1,
                "σ系数": trial_sigma,
                "偏置权重": trial_weight,
                "得分": round(trial_score, 4),
                "冲突数": trial_conflicts,
                "是否采纳": "✅" if accepted else "❌",
                "备注": note,
            }
        )
        log_event(
            "sheet_metal_autoresearch",
            "iteration",
            "Completed a sheet metal AutoResearch iteration",
            iteration=index + 1,
            sigma=trial_sigma,
            weight=trial_weight,
            trial_score=round(float(trial_score), 4),
            trial_conflicts=int(trial_conflicts),
            accepted=accepted,
            best_sigma=round(float(best_sigma), 4),
            best_weight=int(best_weight),
            best_score=round(float(best_score), 4),
            best_conflicts=int(best_conflicts),
        )

        if progress_callback:
            progress_callback(index + 1, n_iterations, best_score, trial_score, best_sigma, best_weight)

        if best_score >= 1.0 and best_conflicts == 0:
            break

    result = {
        "best_sigma": round(best_sigma, 4),
        "best_weight": int(best_weight),
        "best_score": round(best_score, 4),
        "best_conflicts": int(best_conflicts),
        "total_expert": int(total_expert),
        "history": history,
        "result_df": best_df,
    }
    log_event(
        "sheet_metal_autoresearch",
        "complete",
        "Completed sheet metal AutoResearch run",
        iteration_budget=int(n_iterations),
        executed_iterations=max(len(history) - 1, 0),
        best_sigma=result["best_sigma"],
        best_weight=result["best_weight"],
        best_score=result["best_score"],
        best_conflicts=result["best_conflicts"],
        total_expert=result["total_expert"],
    )
    return result


def build_sheet_metal_audit_report(
    original_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    expert_labels: Dict[str, str] | None = None,
) -> pd.DataFrame:
    columns = [
        "_record_key",
        "物料编码",
        "物料描述",
        "备件简称",
        "工厂",
        "白痴指数",
        "基准指数",
        "合理下限",
        "合理上限",
        "status",
        "判定依据",
        "专家校准",
    ]
    if original_df is None or original_df.empty or optimized_df is None or optimized_df.empty:
        return pd.DataFrame(columns=["物料编码", "物料描述", "备件简称", "原始结论", "优化后结论", "结论变化", "判定依据"])

    if expert_labels and "_record_key" in original_df.columns:
        labeled_keys = {str(key) for key in expert_labels.keys()}
        original_df = original_df[original_df["_record_key"].astype(str).isin(labeled_keys)].copy()
        optimized_df = optimized_df[optimized_df["_record_key"].astype(str).isin(labeled_keys)].copy()
        if original_df.empty:
            return pd.DataFrame(columns=["物料编码", "物料描述", "备件简称", "原始结论", "优化后结论", "结论变化", "判定依据"])

    original_slice = original_df[[column for column in columns if column in original_df.columns]].copy()
    optimized_slice = optimized_df[[column for column in columns if column in optimized_df.columns]].copy()
    original_slice = original_slice.rename(
        columns={
            "status": "原始结论",
            "基准指数": "原始基准指数",
            "合理下限": "原始合理下限",
            "合理上限": "原始合理上限",
        }
    )
    optimized_slice = optimized_slice.rename(
        columns={
            "status": "优化后结论",
            "基准指数": "优化后基准指数",
            "合理下限": "优化后合理下限",
            "合理上限": "优化后合理上限",
        }
    )

    audit_df = original_slice.merge(
        optimized_slice,
        on="_record_key",
        how="outer",
        suffixes=("_orig", "_opt"),
    )
    for column_name in ["物料编码", "物料描述", "备件简称", "工厂", "白痴指数"]:
        orig_name = f"{column_name}_orig"
        opt_name = f"{column_name}_opt"
        if orig_name in audit_df.columns or opt_name in audit_df.columns:
            audit_df[column_name] = audit_df.get(orig_name, pd.Series(dtype="object")).combine_first(
                audit_df.get(opt_name, pd.Series(dtype="object"))
            )

    audit_df["原始结论"] = audit_df.get("原始结论", pd.Series(dtype="object")).fillna("未生成")
    audit_df["优化后结论"] = audit_df.get("优化后结论", pd.Series(dtype="object")).fillna("未生成")
    audit_df["结论变化"] = np.where(audit_df["原始结论"] == audit_df["优化后结论"], "", "已调整")
    audit_df["判定依据"] = audit_df.get("判定依据", pd.Series(dtype="object")).fillna("")
    audit_df["专家校准"] = audit_df.get("专家校准", pd.Series(dtype="object")).fillna("")

    ordered_columns = [
        "物料编码",
        "物料描述",
        "备件简称",
        "工厂",
        "白痴指数",
        "原始结论",
        "优化后结论",
        "结论变化",
        "原始基准指数",
        "原始合理下限",
        "原始合理上限",
        "优化后基准指数",
        "优化后合理下限",
        "优化后合理上限",
        "判定依据",
        "专家校准",
    ]
    ordered_columns = [column for column in ordered_columns if column in audit_df.columns]
    audit_df = audit_df[ordered_columns].sort_values(["结论变化", "备件简称", "物料编码"], ascending=[False, True, True]).reset_index(drop=True)
    return audit_df


def build_sheet_metal_calibration_management_df(
    label_details: Dict[str, Dict[str, str]],
    *,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    display_columns = [
        "物料编码",
        "物料名称",
        "备件简称",
        "工厂",
        "白痴指数",
        "当前标注",
        "标注备注",
        "撤回标注",
    ]
    rows = []
    for record_key, payload in label_details.items():
        parsed = split_metric_record_key(record_key, value_label="白痴指数", date_label="静态快照时间")
        rows.append(
            {
                "record_key": parsed["record_key"],
                "物料编码": parsed.get("物料编码", ""),
                "工厂": parsed.get("工厂", ""),
                "白痴指数": parsed.get("白痴指数", np.nan),
                "当前标注": payload.get("label", ""),
                "标注备注": payload.get("remark", ""),
                "撤回标注": False,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["record_key"] + display_columns)

    result_df = pd.DataFrame(rows)
    if source_df is None or source_df.empty:
        result_df["物料名称"] = ""
        result_df["备件简称"] = ""
        return result_df[["record_key"] + display_columns].copy()

    lookup_df = source_df.copy()
    lookup_df["物料编码"] = lookup_df["物料编码"].fillna("").astype(str)
    if "物料名称" not in lookup_df.columns and "物料描述" in lookup_df.columns:
        lookup_df["物料名称"] = lookup_df["物料描述"]
    for column_name in ["物料名称", "备件简称", "工厂"]:
        if column_name not in lookup_df.columns:
            lookup_df[column_name] = ""
        lookup_df[column_name] = lookup_df[column_name].fillna("").astype(str)
    lookup_df = lookup_df.drop_duplicates(subset=["物料编码"], keep="last")

    result_df = result_df.merge(
        lookup_df[["物料编码", "物料名称", "备件简称", "工厂"]],
        on="物料编码",
        how="left",
        suffixes=("", "_source"),
    )
    result_df["物料名称"] = result_df["物料名称"].fillna("")
    result_df["备件简称"] = result_df["备件简称"].fillna("")
    result_df["工厂"] = result_df["工厂_source"].fillna(result_df["工厂"]).fillna("")
    result_df = result_df.drop(columns=["工厂_source"], errors="ignore")
    return result_df[["record_key"] + display_columns].copy()


def extract_sheet_metal_skills(
    review_df: pd.DataFrame,
    expert_labels: Dict[str, str],
    sigma_multiplier: float = 1.0,
    expert_weight: int = 80,
) -> List[dict]:
    if review_df is None or review_df.empty:
        return []

    has_record_key = "_record_key" in review_df.columns
    skills: List[dict] = []
    for short_name, group in review_df.groupby("备件简称", sort=True):
        index_values = pd.to_numeric(group["白痴指数"], errors="coerce").dropna().to_numpy(dtype=float)
        if index_values.size == 0:
            continue

        if has_record_key:
            record_keys = set(group["_record_key"].astype(str).tolist())
            expert_count = sum(1 for key, value in expert_labels.items() if value == "正常" and key in record_keys)
        else:
            expert_count = 0

        dist = {
            "样本量": int(len(index_values)),
            "均值": round(float(np.mean(index_values)), 4),
            "标准差": round(float(np.std(index_values)), 4),
            "中位数": round(float(np.median(index_values)), 4),
            "最小值": round(float(np.min(index_values)), 4),
            "最大值": round(float(np.max(index_values)), 4),
        }
        if len(index_values) > 2:
            dist["偏度"] = round(float(pd.Series(index_values).skew()), 4)

        skill = {
            "备件简称": str(short_name),
            "适用算法": "KDE+KNN+Elbow 密度连接异常检测",
            "白痴指数分布描述": dist,
            "当前σ参数": round(sigma_multiplier, 4),
            "偏置权重": expert_weight,
            "本组专家标注数": expert_count,
            "白痴指数合理区间": {
                "基准指数": round(float(group["基准指数"].iloc[0]), 4),
                "合理下限": round(float(group["合理下限"].iloc[0]), 4),
                "合理上限": round(float(group["合理上限"].iloc[0]), 4),
            },
            "异常统计": {
                "正常": int(group["status"].astype(str).str.contains("正常").sum()),
                "异常偏高": int(group["status"].astype(str).str.contains("异常偏高").sum()),
                "异常偏低": int(group["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum()),
            },
        }
        if "多圈合理区间" in group.columns:
            ring_payload = str(group["多圈合理区间"].dropna().astype(str).iloc[0]) if not group["多圈合理区间"].dropna().empty else ""
            if ring_payload:
                try:
                    skill["多邻居圈合理区间"] = json.loads(ring_payload)
                except (TypeError, ValueError):
                    skill["多邻居圈合理区间"] = []
            else:
                skill["多邻居圈合理区间"] = []
        else:
            skill["多邻居圈合理区间"] = []

        if has_record_key and expert_count > 0:
            status_map = dict(zip(group["_record_key"].astype(str), group["status"].astype(str)))
            aligned = sum(
                1
                for key, value in expert_labels.items()
                if value == "正常" and "正常" in status_map.get(str(key), "")
            )
            skill["经验对齐率"] = round(aligned / expert_count, 4)
        else:
            skill["经验对齐率"] = "N/A"

        skills.append(skill)

    return skills


def sheet_metal_skills_to_json_bytes(skills: List[dict]) -> bytes:
    payload = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "skills_count": len(skills),
        "skills": skills,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def sheet_metal_skills_to_markdown(skills: List[dict]) -> str:
    lines = [
        "# 钣金件指数 Skills 报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**备件简称总数**: {len(skills)}",
        "",
        "---",
        "",
    ]

    for index, skill in enumerate(skills, start=1):
        lines.append(f"## {index}. {skill['备件简称']}")
        lines.append("")
        lines.append(f"- **适用算法**: {skill['适用算法']}")
        lines.append(f"- **当前 σ 参数**: {skill['当前σ参数']}")
        lines.append(f"- **偏置权重**: {skill['偏置权重']}×")
        lines.append(f"- **本组专家标注数**: {skill['本组专家标注数']}")
        align_rate = skill.get("经验对齐率", "N/A")
        if isinstance(align_rate, float):
            lines.append(f"- **经验对齐率**: {align_rate:.2%}")
        else:
            lines.append(f"- **经验对齐率**: {align_rate}")
        lines.append("")

        bounds = skill["白痴指数合理区间"]
        lines.append("### 白痴指数合理区间")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 基准指数 | {bounds['基准指数']:,.4f} |")
        lines.append(f"| 合理下限 | {bounds['合理下限']:,.4f} |")
        lines.append(f"| 合理上限 | {bounds['合理上限']:,.4f} |")
        lines.append("")

        dist = skill["白痴指数分布描述"]
        lines.append("### 指数分布特征")
        lines.append("")
        lines.append("| 统计量 | 数值 |")
        lines.append("|--------|------|")
        for key, value in dist.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:,.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

        stats = skill["异常统计"]
        lines.append("### 异常统计")
        lines.append("")
        lines.append("| 分类 | 数量 |")
        lines.append("|------|------|")
        for key, value in stats.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def sheet_metal_skills_to_table(skills: List[dict]) -> pd.DataFrame:
    return flatten_skills_for_excel(
        skills,
        interval_key="白痴指数合理区间",
        distribution_key="白痴指数分布描述",
    )


def sheet_metal_skills_to_excel_bytes(skills: List[dict]) -> bytes:
    return _skills_to_excel_bytes(
        skills,
        interval_key="白痴指数合理区间",
        distribution_key="白痴指数分布描述",
        sheet_name="钣金指数Skills",
    )


