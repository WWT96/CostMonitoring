from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from storage_service import (
    CORE_COST_RECORDS_TABLE,
    _CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS,
    _CORE_COST_RECORDS_UPSERT_UPDATE_COLUMNS,
    _CORE_RECORD_EXPORT_COLUMNS,
    _ensure_core_cost_records_business_key_index,
    _insert_rows,
    _rows_from_dataframe,
    _upsert_rows,
    require_db_engine,
)


BASE_COLS = ["物料编码", "物料名称", "适用车系", "备件简称", "工厂"]
PRICE_COL_CANDIDATES = ["价格", "成本", "单价", "Price", "Cost", "含税价", "未税价"]
SUPPORTED_IMPORT_EXTENSIONS = {".xlsx", ".xls", ".csv"}

FIELD_MAP: Dict[str, str] = {
    "partId": "物料编码",
    "materialCode": "物料编码",
    "materialId": "物料编码",
    "part_id": "物料编码",
    "material_code": "物料编码",
    "partName": "物料名称",
    "materialName": "物料名称",
    "part_name": "物料名称",
    "material_name": "物料名称",
    "vehicleSeries": "适用车系",
    "vehicle_series": "适用车系",
    "carModel": "适用车系",
    "car_model": "适用车系",
    "shortName": "备件简称",
    "short_name": "备件简称",
    "partAlias": "备件简称",
    "part_alias": "备件简称",
    "factory": "工厂",
    "plant": "工厂",
    "plantCode": "工厂",
    "plant_code": "工厂",
    "price": "价格",
    "cost": "成本",
    "unitPrice": "单价",
    "unit_price": "单价",
    "validDate": "价格有效于",
    "valid_date": "价格有效于",
    "effectiveDate": "价格有效于",
    "effective_date": "价格有效于",
    "priceDate": "价格有效于",
    "price_date": "价格有效于",
    "priceValidFrom": "价格有效期于",
    "price_valid_from": "价格有效期于",
    "priceValidTo": "价格有效期至",
    "price_valid_to": "价格有效期至",
    "supplierName": "供应商名称",
    "supplier_name": "供应商名称",
    "supplierCode": "供应商代码",
    "supplier_code": "供应商代码",
    "firstLevelAssyPartNo": "一级总成料号",
    "first_level_assy_part_no": "一级总成料号",
    "assyPartNo": "一级总成料号",
    "assy_part_no": "一级总成料号",
    "firstLevelAssyDesc": "一级总成品名描述",
    "first_level_assy_desc": "一级总成品名描述",
    "assyDesc": "一级总成品名描述",
    "assy_desc": "一级总成品名描述",
    "firstLevelAssySupplierName": "一级总成供应商名称",
    "first_level_assy_supplier_name": "一级总成供应商名称",
    "assySupplierName": "一级总成供应商名称",
    "assy_supplier_name": "一级总成供应商名称",
    "firstLevelAssySupplierCode": "一级总成供应商代码",
    "first_level_assy_supplier_code": "一级总成供应商代码",
    "assySupplierCode": "一级总成供应商代码",
    "assy_supplier_code": "一级总成供应商代码",
    "firstLevelAssyCost": "一级总成成本",
    "first_level_assy_cost": "一级总成成本",
    "assyCost": "一级总成成本",
    "assy_cost": "一级总成成本",
}


PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
PARQUET_CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
PARQUET_CACHE_VERSION = 1
BUILTIN_IMPORT_TEMPLATE_COLUMNS: Dict[str, List[str]] = {
    "cost": [
        "物料编码",
        "物料名称",
        "适用车系",
        "备件简称",
        "供应商名称",
        "供应商代码",
        "工厂",
        "价格",
        "价格有效期于",
        "价格有效期至",
    ],
    "assembly": ["层级1编码", "层级1名称", "层级2编码", "层级2名称"],
    "sheet_metal": [
        "车型",
        "物料编码",
        "物料描述",
        "产品成本",
        "出厂单价",
        "包装费",
        "净重",
        "包装后重量",
        "白痴指数",
        "备件简称",
        "车系",
        "车型梯度",
    ],
}
BUILTIN_IMPORT_TEMPLATE_NAMES: Dict[str, str] = {
    "cost": "原始成本数据模板",
    "assembly": "拆分件层级关系模板",
    "sheet_metal": "钣金基础数据模板",
}


@dataclass
class FolderLoadReport:
    dataframe: Optional[pd.DataFrame] = None
    price_col: Optional[str] = None
    error_message: Optional[str] = None
    scanned_files: list[str] = field(default_factory=list)
    loaded_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


def build_builtin_template_dataframe(template_name: str) -> pd.DataFrame:
    normalized_name = str(template_name or "").strip().lower()
    if normalized_name not in BUILTIN_IMPORT_TEMPLATE_COLUMNS:
        supported = "、".join(sorted(BUILTIN_IMPORT_TEMPLATE_COLUMNS))
        raise ValueError(f"不支持的内置模板: {template_name}，可选值: {supported}")
    return pd.DataFrame(columns=BUILTIN_IMPORT_TEMPLATE_COLUMNS[normalized_name])


def build_builtin_template_excel_bytes(template_name: str) -> bytes:
    normalized_name = str(template_name or "").strip().lower()
    template_df = build_builtin_template_dataframe(normalized_name)
    sheet_name = BUILTIN_IMPORT_TEMPLATE_NAMES.get(normalized_name, "导入模板")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def _build_file_cache_paths(source_path: str) -> tuple[Path, Path]:
    source_file = Path(source_path)
    parent_hash = hashlib.sha1(str(source_file.resolve().parent).encode("utf-8")).hexdigest()[:10]
    source_hash = hashlib.sha1(str(source_file.resolve()).encode("utf-8")).hexdigest()[:10]
    cache_dir = PARQUET_CACHE_ROOT / parent_hash / source_hash
    return cache_dir / f"{source_file.stem}.parquet", cache_dir / f"{source_file.stem}.meta.json"


def _load_processed_file_from_disk(
    source_path: str,
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str], dict[str, Any]]:
    source_file = Path(source_path)
    source_mtime_ns = source_file.stat().st_mtime_ns
    parquet_path, meta_path = _build_file_cache_paths(str(source_file))
    stats: dict[str, Any] = {
        "source": str(source_file),
        "cache_hit": False,
        "duration_seconds": 0.0,
    }
    started_at = time.perf_counter()

    # 仅当源文件修改时间未变化时直接复用处理后的 Parquet，避免重复解析 Excel。
    if parquet_path.exists() and meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                str(meta_payload.get("source_path", "")).lower() == str(source_file.resolve()).lower()
                and int(meta_payload.get("source_mtime_ns", -1)) == int(source_mtime_ns)
                and int(meta_payload.get("cache_version", -1)) == PARQUET_CACHE_VERSION
            ):
                cached_df = pd.read_parquet(parquet_path, engine="pyarrow")
                stats["cache_hit"] = True
                stats["duration_seconds"] = time.perf_counter() - started_at
                return cached_df, meta_payload.get("price_col"), None, stats
        except Exception:
            pass

    try:
        if source_file.suffix.lower() == ".csv":
            try:
                raw_df = pd.read_csv(source_file, encoding="utf-8")
            except UnicodeDecodeError:
                raw_df = pd.read_csv(source_file, encoding="gbk")
        else:
            raw_df = pd.read_excel(source_file)

        rename_map = {
            src: dst
            for src, dst in FIELD_MAP.items()
            if src in raw_df.columns and dst not in raw_df.columns
        }
        if rename_map:
            raw_df = raw_df.rename(columns=rename_map)

        processed_df, detected_price_col, error_msg = process_dataframe(raw_df)
        if processed_df is not None and error_msg is None:
            try:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                processed_df.to_parquet(parquet_path, engine="pyarrow", index=False)
                meta_path.write_text(
                    json.dumps(
                        {
                            "source_path": str(source_file.resolve()),
                            "source_mtime_ns": int(source_mtime_ns),
                            "price_col": detected_price_col,
                            "cache_version": PARQUET_CACHE_VERSION,
                            "cached_at": datetime.now().isoformat(timespec="seconds"),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as cache_exc:
                print(f"[performance][读取阶段] Parquet 缓存写入失败: {source_file.name} -> {cache_exc}")

        stats["duration_seconds"] = time.perf_counter() - started_at
        return processed_df, detected_price_col, error_msg, stats
    except Exception as exc:
        stats["duration_seconds"] = time.perf_counter() - started_at
        return None, None, str(exc), stats


def scan_import_files(folder_path: str) -> list[str]:
    base_path = Path(str(folder_path or "").strip())
    if not base_path.exists() or not base_path.is_dir():
        return []
    files = [
        str(file_path)
        for file_path in base_path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_IMPORT_EXTENSIONS
        and not file_path.name.startswith("~$")
    ]
    return sorted(files, key=lambda value: value.lower())


def extract_date(val):
    if pd.isna(val):
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(val))
    if match:
        return match.group(1)

    parsed = pd.to_datetime(str(val).strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def normalize_vehicle_name(name: str) -> str:
    if name is None:
        return ""
    return re.sub(r"\s+", "", str(name)).lower()


_VEHICLE_CHINESE_RE = re.compile(r"[\u3400-\u9fff]+")


def _vehicle_chinese_key(name: Any) -> str:
    if name is None:
        return ""
    return "".join(_VEHICLE_CHINESE_RE.findall(str(name)))


def detect_price_column(columns: Iterable[str]) -> Optional[str]:
    column_list = list(columns)
    for col in PRICE_COL_CANDIDATES:
        if col in column_list:
            return col
    for col in column_list:
        if "价格" in str(col) or "成本" in str(col):
            return col
    return None


def parse_vehicle_rank_config(text: str) -> List[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


_VEHICLE_SERIES_SPLIT_RE = re.compile(r"[、,，/／;；|｜\r\n]+")


def extract_first_vehicle_series_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    for part in _VEHICLE_SERIES_SPLIT_RE.split(text):
        cleaned = str(part or "").strip()
        if cleaned:
            return cleaned
    return ""


def extract_vehicle_rank_candidates(df: pd.DataFrame) -> List[str]:
    if df.empty or "适用车系" not in df.columns:
        return []
    seen: set[str] = set()
    candidates: list[str] = []
    for value in df["适用车系"].tolist():
        vehicle_name = extract_first_vehicle_series_name(value)
        normalized = normalize_vehicle_name(vehicle_name)
        if vehicle_name and normalized not in seen:
            seen.add(normalized)
            candidates.append(vehicle_name)
    return candidates


def _prepare_vehicle_market_price_data(market_prices: pd.DataFrame | None) -> pd.DataFrame:
    if market_prices is None or market_prices.empty:
        return pd.DataFrame()
    if "vehicle_series" not in market_prices.columns or "market_price" not in market_prices.columns:
        return pd.DataFrame()

    data = market_prices.copy()
    for column_name in [
        "vehicle_series",
        "market_price",
        "variant_name",
        "source_url",
        "status",
        "fetched_at",
        "failure_reason",
        "raw_response_json",
    ]:
        if column_name not in data.columns:
            data[column_name] = None

    data["vehicle_series"] = data["vehicle_series"].fillna("").astype(str).str.strip()
    data["_vehicle_norm"] = data["vehicle_series"].map(normalize_vehicle_name)
    data["_vehicle_chinese_key"] = data["vehicle_series"].map(_vehicle_chinese_key)
    data["_market_price_numeric"] = pd.to_numeric(data["market_price"], errors="coerce")
    data["_has_market_price"] = data["_market_price_numeric"].notna()
    data["_fetched_at_ts"] = pd.to_datetime(data["fetched_at"], errors="coerce")
    return data[data["_vehicle_norm"].ne("")].copy()


def _best_vehicle_market_price_row(data: pd.DataFrame) -> pd.Series | None:
    if data is None or data.empty:
        return None
    ranked = data.sort_values(
        by=["_has_market_price", "_fetched_at_ts", "_market_price_numeric", "vehicle_series"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    return ranked.iloc[0].copy()


def _localize_vehicle_market_price_row(row: pd.Series, local_vehicle: str, *, annotate_fallback: bool) -> dict[str, Any]:
    localized = row.to_dict()
    source_vehicle = str(localized.get("vehicle_series") or "").strip()
    localized["vehicle_series"] = local_vehicle
    localized["_vehicle_norm"] = normalize_vehicle_name(local_vehicle)
    localized["_vehicle_chinese_key"] = _vehicle_chinese_key(local_vehicle)
    if annotate_fallback and source_vehicle and normalize_vehicle_name(source_vehicle) != normalize_vehicle_name(local_vehicle):
        variant_name = str(localized.get("variant_name") or "").strip()
        localized["variant_name"] = f"参考{source_vehicle}：{variant_name}" if variant_name else f"参考{source_vehicle}估算"
    return localized


def _blank_vehicle_market_price_row(local_vehicle: str) -> dict[str, Any]:
    return {
        "vehicle_series": local_vehicle,
        "market_price": None,
        "variant_name": "",
        "source_url": "",
        "status": "待确认",
        "fetched_at": None,
        "failure_reason": "未取得本地车系对应的估算价格",
        "raw_response_json": "",
        "_vehicle_norm": normalize_vehicle_name(local_vehicle),
        "_vehicle_chinese_key": _vehicle_chinese_key(local_vehicle),
        "_market_price_numeric": pd.NA,
        "_has_market_price": False,
        "_fetched_at_ts": pd.NaT,
    }


def _local_vehicle_market_price_data(
    market_prices: pd.DataFrame | None,
    vehicle_candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    data = _prepare_vehicle_market_price_data(market_prices)
    if data.empty or not vehicle_candidates:
        return data

    local_vehicles: list[str] = []
    seen: set[str] = set()
    for vehicle_name in vehicle_candidates:
        vehicle_text = str(vehicle_name or "").strip()
        normalized = normalize_vehicle_name(vehicle_text)
        if not vehicle_text or not normalized or normalized in seen:
            continue
        local_vehicles.append(vehicle_text)
        seen.add(normalized)

    priced_data = data[data["_has_market_price"] & data["_vehicle_chinese_key"].ne("")].copy()
    rows: list[dict[str, Any]] = []
    for vehicle_name in local_vehicles:
        normalized = normalize_vehicle_name(vehicle_name)
        exact_rows = data[data["_vehicle_norm"].eq(normalized)]
        exact_priced_row = _best_vehicle_market_price_row(exact_rows[exact_rows["_has_market_price"]])
        if exact_priced_row is not None:
            rows.append(_localize_vehicle_market_price_row(exact_priced_row, vehicle_name, annotate_fallback=False))
            continue

        chinese_key = _vehicle_chinese_key(vehicle_name)
        fallback_rows = priced_data[
            priced_data["_vehicle_chinese_key"].eq(chinese_key) & priced_data["_vehicle_norm"].ne(normalized)
        ]
        fallback_row = _best_vehicle_market_price_row(fallback_rows)
        if chinese_key and fallback_row is not None:
            rows.append(_localize_vehicle_market_price_row(fallback_row, vehicle_name, annotate_fallback=True))
            continue

        exact_pending_row = _best_vehicle_market_price_row(exact_rows)
        if exact_pending_row is not None:
            rows.append(_localize_vehicle_market_price_row(exact_pending_row, vehicle_name, annotate_fallback=False))
        else:
            rows.append(_blank_vehicle_market_price_row(vehicle_name))

    result = pd.DataFrame(rows)
    if not result.empty:
        result["_market_price_numeric"] = pd.to_numeric(result["_market_price_numeric"], errors="coerce")
        result["_has_market_price"] = result["_market_price_numeric"].notna()
        result["_fetched_at_ts"] = pd.to_datetime(result["_fetched_at_ts"], errors="coerce")
    return result


def _market_price_map(
    market_prices: pd.DataFrame | None,
    vehicle_candidates: Sequence[str] | None = None,
) -> dict[str, float]:
    price_df = _local_vehicle_market_price_data(market_prices, vehicle_candidates)
    if price_df.empty:
        return {}
    price_df = price_df[price_df["_has_market_price"]].copy()
    result: dict[str, float] = {}
    for vehicle, price in price_df[["vehicle_series", "_market_price_numeric"]].itertuples(index=False, name=None):
        normalized = normalize_vehicle_name(str(vehicle))
        if normalized:
            result[normalized] = float(price)
    return result


def build_default_vehicle_rank(
    df: pd.DataFrame,
    market_prices: pd.DataFrame | None = None,
    price_col: str | None = None,
) -> List[str]:
    candidates = extract_vehicle_rank_candidates(df)
    if not candidates:
        return []

    market_lookup = _market_price_map(market_prices, candidates)
    if market_lookup:
        return sorted(
            candidates,
            key=lambda name: (
                0 if normalize_vehicle_name(name) in market_lookup else 1,
                -market_lookup.get(normalize_vehicle_name(name), float("-inf")),
                name,
            ),
        )

    resolved_price_col = price_col or detect_price_column(df.columns)
    if not resolved_price_col or resolved_price_col not in df.columns:
        return sorted(candidates)

    data = df.copy()
    data["_vehicle_first"] = data["适用车系"].apply(extract_first_vehicle_series_name)
    data[resolved_price_col] = pd.to_numeric(data[resolved_price_col], errors="coerce")
    if "monitor_date" in data.columns:
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")
        data = data.sort_values("monitor_date")
    latest_price = (
        data.dropna(subset=[resolved_price_col])
        .drop_duplicates(subset=["_vehicle_first"], keep="last")
        .set_index("_vehicle_first")[resolved_price_col]
        .to_dict()
    )
    return sorted(candidates, key=lambda name: (-float(latest_price.get(name, float("-inf"))), name))


def _extract_market_price_failure_reason(row: pd.Series) -> str:
    reason = str(row.get("failure_reason") or "").strip()
    if reason:
        return reason

    raw_payload = str(row.get("raw_response_json") or "").strip()
    if not raw_payload:
        return ""
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError:
        return ""
    if isinstance(parsed, dict):
        return str(parsed.get("failure_reason") or parsed.get("失败原因") or "").strip()
    return ""


def build_vehicle_market_price_display_df(
    market_prices: pd.DataFrame | None,
    rank_order: List[str] | None = None,
    vehicle_candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    display_columns = [
        "梯度排名",
        "车系",
        "次顶配车型",
        "估算价格（元）",
    ]
    if market_prices is None or market_prices.empty:
        return pd.DataFrame(columns=display_columns)

    data = _local_vehicle_market_price_data(market_prices, vehicle_candidates)
    if data.empty:
        return pd.DataFrame(columns=display_columns)

    rank_lookup = {
        normalize_vehicle_name(vehicle_name): index
        for index, vehicle_name in enumerate(rank_order or [], start=1)
        if normalize_vehicle_name(str(vehicle_name))
    }
    if rank_lookup:
        data["_manual_rank"] = data["vehicle_series"].map(lambda value: rank_lookup.get(normalize_vehicle_name(value)))
        data = data.sort_values(
            by=["_manual_rank", "_has_market_price", "_market_price_numeric", "vehicle_series"],
            ascending=[True, False, False, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        data = data.sort_values(
            by=["_has_market_price", "_market_price_numeric", "vehicle_series"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    display_df = pd.DataFrame(
        {
            "梯度排名": pd.Series(range(1, len(data) + 1), dtype="Int64"),
            "车系": data["vehicle_series"],
            "次顶配车型": data["variant_name"].fillna("").astype(str).str.strip(),
            "估算价格（元）": data["_market_price_numeric"].round(0).astype("Int64"),
        }
    )
    display_df.loc[display_df["估算价格（元）"].isna(), "估算价格（元）"] = None
    return display_df[display_columns]


def extract_missing_vehicle_market_price_series(display_df: pd.DataFrame | None) -> List[str]:
    if display_df is None or display_df.empty or "车系" not in display_df.columns or "估算价格（元）" not in display_df.columns:
        return []

    data = display_df.copy()
    data["车系"] = data["车系"].fillna("").astype(str).str.strip()
    data["_market_price_numeric"] = pd.to_numeric(data["估算价格（元）"], errors="coerce")
    missing_rows = data[data["车系"].ne("") & data["_market_price_numeric"].isna()]

    missing_vehicle_series: list[str] = []
    seen: set[str] = set()
    for vehicle_name in missing_rows["车系"].tolist():
        normalized = normalize_vehicle_name(vehicle_name)
        if not normalized or normalized in seen:
            continue
        missing_vehicle_series.append(vehicle_name)
        seen.add(normalized)
    return missing_vehicle_series


def build_manual_vehicle_rank_from_display(display_df: pd.DataFrame | None) -> List[str]:
    if display_df is None or display_df.empty:
        return []
    if "梯度排名" not in display_df.columns or "车系" not in display_df.columns:
        return []

    data = display_df.copy()
    data["_rank_numeric"] = pd.to_numeric(data["梯度排名"], errors="coerce")
    data["_original_order"] = range(len(data))
    data["车系"] = data["车系"].fillna("").astype(str).str.strip()
    data = data[data["车系"].ne("")].copy()
    if data.empty:
        return []

    data = data.sort_values(
        by=["_rank_numeric", "_original_order", "车系"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    )
    ranked: list[str] = []
    seen: set[str] = set()
    for vehicle_name in data["车系"].tolist():
        normalized = normalize_vehicle_name(vehicle_name)
        if not normalized or normalized in seen:
            continue
        ranked.append(vehicle_name)
        seen.add(normalized)
    return ranked


def build_manual_vehicle_market_price_rows_from_display(display_df: pd.DataFrame | None) -> List[Dict[str, Any]]:
    if display_df is None or display_df.empty or "车系" not in display_df.columns:
        return []

    data = display_df.copy()
    for column_name in ["次顶配车型", "估算价格（元）"]:
        if column_name not in data.columns:
            data[column_name] = None

    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now().isoformat(timespec="seconds")
    seen: set[str] = set()
    for _, row in data.iterrows():
        vehicle_series = str(row.get("车系") or "").strip()
        normalized = normalize_vehicle_name(vehicle_series)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        market_price = pd.to_numeric(pd.Series([row.get("估算价格（元）")]), errors="coerce").iloc[0]
        has_price = pd.notna(market_price)
        rows.append(
            {
                "vehicle_series": vehicle_series,
                "market_price": float(market_price) if has_price else None,
                "variant_name": str(row.get("次顶配车型") or "").strip(),
                "source_url": "",
                "source_domain": "",
                "status": "人工修正" if has_price else "待确认",
                "fetched_at": fetched_at,
                "failure_reason": "" if has_price else "人工未填写估算价格",
                "raw_response_json": {
                    "source": "manual_vehicle_rank_edit",
                    "rank_order": row.get("梯度排名"),
                    "market_price": float(market_price) if has_price else None,
                },
            }
        )
    return rows


def build_vehicle_candidate_display_df(vehicle_candidates: Sequence[str] | None) -> pd.DataFrame:
    rows = [
        {"序号": index, "车系": str(vehicle_name).strip()}
        for index, vehicle_name in enumerate(vehicle_candidates or [], start=1)
        if str(vehicle_name).strip()
    ]
    return pd.DataFrame(rows, columns=["序号", "车系"])


def process_records_from_json(
    records: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    if not records:
        return None, None, "输入记录列表为空"

    df = pd.DataFrame(records)
    rename_map = {
        src: dst
        for src, dst in FIELD_MAP.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    return process_dataframe(df)


def process_dataframe(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    date_col = None

    preferred_date_columns = ["价格有效期于", "价格有效于", "有效期起", "生效日期"]
    possible_date_cols = [col for col in preferred_date_columns if col in df.columns]
    if not possible_date_cols:
        possible_date_cols = [col for col in df.columns if "价格有效于" in str(col) or "价格有效期于" in str(col)]
    if not possible_date_cols:
        possible_date_cols = [col for col in df.columns if "有效" in str(col)]
    if not possible_date_cols:
        possible_date_cols = [col for col in df.columns if "日期" in str(col) or "时间" in str(col)]
    if possible_date_cols:
        date_col = possible_date_cols[0]

    price_col = detect_price_column(df.columns)

    if not date_col or not price_col or "物料编码" not in df.columns:
        return None, None, "缺少必要列"

    result = df.copy()
    if "工厂" in result.columns:
        result["工厂"] = result["工厂"].fillna("总装")
    else:
        result["工厂"] = "总装"

    result["monitor_date"] = result[date_col].apply(extract_date)
    result = result.dropna(subset=["monitor_date"])
    result["monitor_date"] = pd.to_datetime(result["monitor_date"])

    result[price_col] = pd.to_numeric(result[price_col], errors="coerce")
    result = result.dropna(subset=[price_col])

    if "物料名称" not in result.columns:
        result["物料名称"] = "未知"
    if "适用车系" not in result.columns:
        result["适用车系"] = "未知"
    if "备件简称" not in result.columns:
        result["备件简称"] = "未知"
    if "供应商名称" in result.columns and "一级总成供应商名称" not in result.columns:
        result["一级总成供应商名称"] = result["供应商名称"]
    if "供应商代码" in result.columns and "一级总成供应商代码" not in result.columns:
        result["一级总成供应商代码"] = result["供应商代码"]

    if "一级总成成本" in result.columns:
        result["一级总成成本"] = pd.to_numeric(result["一级总成成本"], errors="coerce")
    for assy_col in ["一级总成料号", "一级总成品名描述", "一级总成供应商名称", "一级总成供应商代码"]:
        if assy_col in result.columns:
            result[assy_col] = result[assy_col].astype(str).replace({"nan": "", "None": ""})

    return result, price_col, None


def _prepare_core_cost_records(
    df: pd.DataFrame,
    price_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return pd.DataFrame(columns=_CORE_RECORD_EXPORT_COLUMNS + ["created_at"]), price_col or "成本"

    data = df.copy().reset_index(drop=True)
    resolved_price_col = price_col if price_col in data.columns else None
    if not resolved_price_col:
        resolved_price_col = detect_price_column(data.columns)
    if not resolved_price_col and "成本金额" in data.columns:
        resolved_price_col = "成本金额"
    if not resolved_price_col:
        raise ValueError("无法识别核心成本数据的价格列")

    if "monitor_date" not in data.columns:
        if "价格有效于" in data.columns:
            data["monitor_date"] = pd.to_datetime(data["价格有效于"], errors="coerce")
        else:
            raise ValueError("核心成本数据缺少 monitor_date 字段")
    else:
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")

    if "工厂" in data.columns:
        data["工厂"] = data["工厂"].fillna("总装")
    else:
        data["工厂"] = "总装"

    for col_name, default_value in {
        "物料名称": "未知",
        "适用车系": "未知",
        "备件简称": "未知",
        "供应商名称": None,
        "供应商代码": None,
        "价格有效期至": None,
        "一级总成料号": None,
        "一级总成品名描述": None,
        "一级总成供应商名称": None,
        "一级总成供应商代码": None,
        "一级总成成本": None,
    }.items():
        if col_name not in data.columns:
            data[col_name] = default_value
    data["一级总成供应商名称"] = data["一级总成供应商名称"].fillna(data["供应商名称"])
    data["一级总成供应商代码"] = data["一级总成供应商代码"].fillna(data["供应商代码"])

    data[resolved_price_col] = pd.to_numeric(data[resolved_price_col], errors="coerce")
    if "一级总成成本" in data.columns:
        data["一级总成成本"] = pd.to_numeric(data["一级总成成本"], errors="coerce")
    data["价格有效期至"] = pd.to_datetime(data["价格有效期至"], errors="coerce")

    prepared = pd.DataFrame(
        {
            "material_code": data["物料编码"].astype(str),
            "material_name": data["物料名称"],
            "vehicle_series": data["适用车系"],
            "short_name": data["备件简称"],
            "factory": data["工厂"],
            "cost_amount": data[resolved_price_col],
            "monitor_date": data["monitor_date"],
            "supplier_name": data["供应商名称"],
            "supplier_code": data["供应商代码"],
            "price_valid_to": data["价格有效期至"],
            "assy_part_no": data["一级总成料号"],
            "assy_desc": data["一级总成品名描述"],
            "assy_supplier_name": data["一级总成供应商名称"],
            "assy_supplier_code": data["一级总成供应商代码"],
            "assy_cost": data["一级总成成本"],
            "created_at": datetime.now(),
        }
    )
    prepared = prepared.dropna(subset=["material_code", "factory", "monitor_date", "cost_amount"])
    prepared["source_row_hash"] = _build_core_cost_source_row_hashes(prepared)
    return prepared, resolved_price_col


def _normalize_core_hash_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value).strip()


def _build_core_cost_source_row_hashes(prepared: pd.DataFrame) -> List[str]:
    hash_columns = [
        "material_code",
        "material_name",
        "vehicle_series",
        "short_name",
        "factory",
        "cost_amount",
        "monitor_date",
        "supplier_name",
        "supplier_code",
        "price_valid_to",
        "assy_part_no",
        "assy_desc",
        "assy_supplier_name",
        "assy_supplier_code",
        "assy_cost",
    ]
    seen: dict[str, int] = {}
    hashes: list[str] = []
    for row in prepared[hash_columns].to_dict(orient="records"):
        payload = json.dumps(
            {key: _normalize_core_hash_value(value) for key, value in row.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        occurrence = seen.get(payload, 0)
        seen[payload] = occurrence + 1
        hashes.append(hashlib.sha1(f"{payload}|{occurrence}".encode("utf-8")).hexdigest())
    return hashes


def persist_core_cost_records(
    df: pd.DataFrame,
    price_col: Optional[str] = None,
    mode: str = "incremental",
) -> int:
    prepared, _ = _prepare_core_cost_records(df, price_col)
    rows = _rows_from_dataframe(prepared)
    engine = require_db_engine()
    _ensure_core_cost_records_business_key_index()

    if mode not in {"full", "incremental"}:
        raise ValueError(f"不支持的持久化模式: {mode}")

    with Session(engine) as session:
        with session.begin():
            if mode == "full":
                session.execute(delete(CORE_COST_RECORDS_TABLE))
                _insert_rows(CORE_COST_RECORDS_TABLE, rows, session=session)
            else:
                _upsert_rows(
                    CORE_COST_RECORDS_TABLE,
                    rows,
                    conflict_columns=_CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS,
                    update_columns=_CORE_COST_RECORDS_UPSERT_UPDATE_COLUMNS,
                    session=session,
                )
    return len(rows)


def load_core_cost_records() -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    engine = require_db_engine()
    query = select(*(CORE_COST_RECORDS_TABLE.c[column] for column in _CORE_RECORD_EXPORT_COLUMNS))
    df = pd.read_sql(query, engine)
    if df.empty:
        return None, None, "本地数据库中暂无核心成本数据"

    df["monitor_date"] = pd.to_datetime(df["monitor_date"], errors="coerce")
    if "price_valid_to" in df.columns:
        df["price_valid_to"] = pd.to_datetime(df["price_valid_to"], errors="coerce")
    df = df.rename(
        columns={
            "material_code": "物料编码",
            "material_name": "物料名称",
            "vehicle_series": "适用车系",
            "short_name": "备件简称",
            "factory": "工厂",
            "supplier_name": "供应商名称",
            "supplier_code": "供应商代码",
            "price_valid_to": "价格有效期至",
            "assy_part_no": "一级总成料号",
            "assy_desc": "一级总成品名描述",
            "assy_supplier_name": "一级总成供应商名称",
            "assy_supplier_code": "一级总成供应商代码",
            "assy_cost": "一级总成成本",
        }
    )
    df["成本"] = pd.to_numeric(df.pop("cost_amount"), errors="coerce")
    return df, "成本", None


def get_core_cost_records_refresh_token() -> float:
    engine = require_db_engine()
    with engine.connect() as conn:
        latest = conn.execute(
            select(func.max(CORE_COST_RECORDS_TABLE.c.created_at))
        ).scalar_one_or_none()
    if latest is None:
        return 0.0
    return pd.Timestamp(latest).timestamp()


def get_core_cost_records_status() -> Dict[str, Any]:
    engine = require_db_engine()
    with engine.connect() as conn:
        row_count = int(conn.execute(select(func.count()).select_from(CORE_COST_RECORDS_TABLE)).scalar_one())
        latest = conn.execute(select(func.max(CORE_COST_RECORDS_TABLE.c.created_at))).scalar_one_or_none()
    return {
        "row_count": row_count,
        "updated_at": pd.Timestamp(latest).to_pydatetime() if latest is not None else None,
        "price_col": "成本" if row_count > 0 else None,
    }


def _format_uploaded_file_failure(file_name: str, reason: Any) -> str:
    safe_name = os.path.basename(str(file_name or "未知文件").strip()) or "未知文件"
    safe_reason = re.sub(r"\s+", " ", str(reason or "未能解析").strip())
    if len(safe_reason) > 160:
        safe_reason = safe_reason[:157] + "..."
    return f"{safe_name}: {safe_reason}"


def load_data_from_uploaded_files(
    uploaded_files: List[Any],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], List[str]]:
    if not uploaded_files:
        return None, None, "未提供任何上传文件", []

    df_list: List[pd.DataFrame] = []
    price_col_detected: Optional[str] = None
    failed_files: List[str] = []

    for uploaded_file in uploaded_files:
        display_name = str(getattr(uploaded_file, "name", "未知文件") or "未知文件")
        try:
            name: str = display_name.lower()
            raw_bytes = uploaded_file.getvalue()
            if name.endswith(".csv"):
                try:
                    raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8")
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="gbk")
            else:
                raw_df = pd.read_excel(io.BytesIO(raw_bytes))

            rename_map = {
                src: dst
                for src, dst in FIELD_MAP.items()
                if src in raw_df.columns and dst not in raw_df.columns
            }
            if rename_map:
                raw_df = raw_df.rename(columns=rename_map)

            processed_df, detected_price_col, error_msg = process_dataframe(raw_df)
            if processed_df is not None:
                df_list.append(processed_df)
                if not price_col_detected:
                    price_col_detected = detected_price_col
            else:
                failed_files.append(_format_uploaded_file_failure(display_name, error_msg or "未识别到有效数据"))
        except Exception as exc:
            failed_files.append(_format_uploaded_file_failure(display_name, exc))

    if not df_list:
        detail = ""
        if failed_files:
            detail = "；失败文件：" + "；".join(failed_files[:5])
            if len(failed_files) > 5:
                detail += f"；另有 {len(failed_files) - 5} 个文件"
        return None, None, f"所有上传文件均未能成功解析，请检查文件格式和必要列名{detail}", failed_files

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df, price_col_detected, None, failed_files


def load_data_from_folder_with_report(folder_path: str) -> FolderLoadReport:
    if not os.path.isdir(folder_path):
        return FolderLoadReport(error_message=f"路径不存在: {folder_path}")

    all_files = scan_import_files(folder_path)
    if not all_files:
        return FolderLoadReport(error_message="路径下没有找到 Excel 或 CSV 文件")

    load_started_at = time.perf_counter()
    df_list: List[pd.DataFrame] = []
    price_col_detected = None
    cache_hits = 0
    cache_misses = 0
    failed_files: List[str] = []
    loaded_files: List[str] = []
    for filename in all_files:
        processed_df, detected_price_col, error_msg, stats = _load_processed_file_from_disk(filename)
        if stats.get("cache_hit"):
            cache_hits += 1
        else:
            cache_misses += 1

        if processed_df is not None:
            df_list.append(processed_df)
            loaded_files.append(os.path.basename(filename))
            if not price_col_detected:
                price_col_detected = detected_price_col
            continue

        failure_text = os.path.basename(filename)
        if error_msg:
            failure_text = f"{failure_text}: {error_msg}"
        failed_files.append(failure_text)
        if error_msg:
            print(f"[performance][读取阶段] 跳过文件 {os.path.basename(filename)}: {error_msg}")

    if not df_list:
        return FolderLoadReport(
            error_message="没有成功读取到有效数据",
            scanned_files=[os.path.basename(file_path) for file_path in all_files],
            failed_files=failed_files,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    final_df = pd.concat(df_list, ignore_index=True)
    total_seconds = time.perf_counter() - load_started_at
    print(
        "[performance][读取阶段] "
        f"文件数={len(all_files)} 命中缓存={cache_hits} 重建缓存={cache_misses} "
        f"失败文件={len(failed_files)} 输出行数={len(final_df)} 总耗时={total_seconds:.3f}s"
    )
    return FolderLoadReport(
        dataframe=final_df,
        price_col=price_col_detected,
        scanned_files=[os.path.basename(file_path) for file_path in all_files],
        loaded_files=loaded_files,
        failed_files=failed_files,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def load_data_from_folder(
    folder_path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    report = load_data_from_folder_with_report(folder_path)
    return report.dataframe, report.price_col, report.error_message


def generate_pivot_report(df: pd.DataFrame, price_col: str, max_changes: int = 9) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BASE_COLS)

    data = df.copy()
    data = data.dropna(subset=["物料编码", price_col])
    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"])

    for col in ["物料名称", "适用车系", "备件简称"]:
        if col not in data.columns:
            data[col] = "N/A"
    if "工厂" not in data.columns:
        data["工厂"] = "总装"

    data = data.sort_values("monitor_date")
    data = data.drop_duplicates(subset=["物料编码", "工厂", "monitor_date"], keep="last")
    data = data.sort_values(["物料编码", "工厂", "monitor_date"])

    data["change_seq"] = data.groupby(["物料编码", "工厂"]).cumcount() + 1
    data = data[data["change_seq"] <= max_changes]
    data["col_name"] = "价格变动" + data["change_seq"].astype(str)

    pivot_df = data.pivot_table(
        index=BASE_COLS,
        columns="col_name",
        values=price_col,
        aggfunc="first",
    ).reset_index()

    price_cols = [f"价格变动{i}" for i in range(1, max_changes + 1) if f"价格变动{i}" in pivot_df.columns]
    result_df = pivot_df[BASE_COLS + price_cols].copy()
    result_df = result_df.sort_values(["物料编码", "工厂"])
    return result_df


def _price_change_columns(df: pd.DataFrame) -> list[str]:
    columns = [column_name for column_name in df.columns if str(column_name).startswith("价格变动")]
    return sorted(
        columns,
        key=lambda value: int(re.search(r"\d+", str(value)).group()) if re.search(r"\d+", str(value)) else 0,
    )


def _latest_cost_change_metrics(report_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["上期成本", "最新成本", "最新变动额", "最新变动比例", "上期成本列", "最新成本列"]
    if report_df is None or report_df.empty:
        return pd.DataFrame(columns=metric_columns, index=getattr(report_df, "index", None))

    price_cols = _price_change_columns(report_df)
    if len(price_cols) < 2:
        return pd.DataFrame(
            {
                "上期成本": pd.Series([np.nan] * len(report_df), index=report_df.index),
                "最新成本": pd.Series([np.nan] * len(report_df), index=report_df.index),
                "最新变动额": pd.Series([np.nan] * len(report_df), index=report_df.index),
                "最新变动比例": pd.Series([np.nan] * len(report_df), index=report_df.index),
                "上期成本列": pd.Series([""] * len(report_df), index=report_df.index),
                "最新成本列": pd.Series([""] * len(report_df), index=report_df.index),
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for _, row in report_df.iterrows():
        numeric_values: list[tuple[str, float]] = []
        for column_name in price_cols:
            numeric_value = pd.to_numeric(pd.Series([row.get(column_name)]), errors="coerce").iloc[0]
            if pd.notna(numeric_value):
                numeric_values.append((column_name, float(numeric_value)))
        if len(numeric_values) < 2:
            metric_rows.append(
                {
                    "上期成本": np.nan,
                    "最新成本": np.nan,
                    "最新变动额": np.nan,
                    "最新变动比例": np.nan,
                    "上期成本列": "",
                    "最新成本列": "",
                }
            )
            continue
        previous_col, previous_value = numeric_values[-2]
        latest_col, latest_value = numeric_values[-1]
        delta = latest_value - previous_value
        metric_rows.append(
            {
                "上期成本": previous_value,
                "最新成本": latest_value,
                "最新变动额": delta,
                "最新变动比例": delta / previous_value if previous_value else np.nan,
                "上期成本列": previous_col,
                "最新成本列": latest_col,
            }
        )
    return pd.DataFrame(metric_rows, index=report_df.index, columns=metric_columns)


def prioritize_latest_cost_increases(report_df: pd.DataFrame | None) -> pd.DataFrame:
    if report_df is None or report_df.empty:
        return pd.DataFrame() if report_df is None else report_df.copy()

    result = report_df.copy()
    metrics = _latest_cost_change_metrics(result)
    result["_latest_cost_increase"] = pd.to_numeric(metrics["最新变动额"], errors="coerce").gt(0)
    result["_latest_cost_delta"] = pd.to_numeric(metrics["最新变动额"], errors="coerce")
    fallback_sort_columns = [column_name for column_name in ["物料编码", "工厂", "备件简称", "适用车系"] if column_name in result.columns]
    result = result.sort_values(
        by=["_latest_cost_increase", "_latest_cost_delta", *fallback_sort_columns],
        ascending=[False, False, *([True] * len(fallback_sort_columns))],
        na_position="last",
        kind="mergesort",
    )
    return result.drop(columns=["_latest_cost_increase", "_latest_cost_delta"], errors="ignore").reset_index(drop=True)


def filter_latest_cost_increase_rows(report_df: pd.DataFrame | None) -> pd.DataFrame:
    if report_df is None or report_df.empty:
        return pd.DataFrame(columns=[*BASE_COLS, "上期成本", "最新成本", "最新变动额", "最新变动比例", "上期成本列", "最新成本列"])

    metrics = _latest_cost_change_metrics(report_df)
    increase_mask = pd.to_numeric(metrics["最新变动额"], errors="coerce").gt(0)
    if not bool(increase_mask.any()):
        return pd.DataFrame(columns=[*BASE_COLS, "上期成本", "最新成本", "最新变动额", "最新变动比例", "上期成本列", "最新成本列"])

    base_columns = [column_name for column_name in BASE_COLS if column_name in report_df.columns]
    price_columns = _price_change_columns(report_df)
    export_df = pd.concat([report_df.loc[increase_mask, base_columns + price_columns].copy(), metrics.loc[increase_mask].copy()], axis=1)
    export_columns = [
        *base_columns,
        "上期成本",
        "最新成本",
        "最新变动额",
        "最新变动比例",
        "上期成本列",
        "最新成本列",
        *price_columns,
    ]
    export_df = export_df[export_columns].sort_values(
        by=["最新变动额", *[column_name for column_name in ["物料编码", "工厂"] if column_name in export_df.columns]],
        ascending=[False, *([True] * len([column_name for column_name in ["物料编码", "工厂"] if column_name in export_df.columns]))],
        na_position="last",
        kind="mergesort",
    )
    return export_df.reset_index(drop=True)


def generate_trend_report(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    pivot_df = generate_pivot_report(df, price_col)
    if pivot_df.empty:
        return pivot_df

    trend_df = pivot_df.copy()
    price_cols = [col for col in trend_df.columns if col.startswith("价格变动")]
    price_cols.sort(key=lambda value: int(re.search(r"\d+", value).group()))

    trend_cols = []
    for idx in range(len(price_cols) - 1):
        curr_col = price_cols[idx]
        next_col = price_cols[idx + 1]
        trend_col = f"变动趋势{idx + 1}"
        trend_df[curr_col] = pd.to_numeric(trend_df[curr_col], errors="coerce")
        trend_df[next_col] = pd.to_numeric(trend_df[next_col], errors="coerce")
        trend_df[trend_col] = trend_df[next_col] - trend_df[curr_col]
        trend_cols.append(trend_col)

    return trend_df[BASE_COLS + trend_cols].copy()


def get_material_metrics(item_df: pd.DataFrame, price_col: str) -> dict:
    data = item_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    if "工厂" not in data.columns:
        data["工厂"] = "总装"
    data["工厂"] = data["工厂"].fillna("总装").astype(str).str.strip()
    data = data.dropna(subset=["monitor_date", price_col])

    latest_date = data["monitor_date"].max()
    latest_rows = data[data["monitor_date"].eq(latest_date)].copy()
    latest_record = latest_rows.sort_values("工厂", kind="stable").iloc[0]

    latest_price = float(latest_record[price_col])
    latest_factory = str(latest_record.get("工厂", "") or "未知工厂")
    min_price = float(data[price_col].min())
    max_price = float(data[price_col].max())

    x990_rows = data[data["工厂"].astype(str).str.upper().eq("X990")].copy()
    final_assembly_rows = data[data["工厂"].astype(str).eq("总装")].copy()
    freight_factor = None
    if not x990_rows.empty and not final_assembly_rows.empty:
        latest_x990_record = x990_rows.sort_values(["monitor_date", "工厂"], kind="stable").iloc[-1]
        latest_assembly_record = final_assembly_rows.sort_values(["monitor_date", "工厂"], kind="stable").iloc[-1]
        x990_price = float(latest_x990_record[price_col])
        assembly_price = float(latest_assembly_record[price_col])
        if assembly_price != 0:
            freight_factor = x990_price / assembly_price

    prior_year_end = pd.Timestamp(year=int(latest_date.year) - 1, month=12, day=31)
    history_df = data[data["monitor_date"].le(prior_year_end)].copy()
    same_factory_history = history_df[history_df["工厂"].astype(str).eq(latest_factory)].copy()
    reference_df = same_factory_history if not same_factory_history.empty else history_df
    reference_price = None
    reference_factory = ""
    reference_date = pd.NaT
    cost_drop_amount = None
    if not reference_df.empty:
        reference_record = reference_df.sort_values(["monitor_date", "工厂"], kind="stable").iloc[-1]
        reference_price = float(reference_record[price_col])
        reference_factory = str(reference_record.get("工厂", "") or "")
        reference_date = pd.Timestamp(reference_record["monitor_date"])
        cost_drop_amount = latest_price - reference_price

    return {
        "latest_price": latest_price,
        "latest_factory": latest_factory,
        "latest_date": pd.Timestamp(latest_date),
        "min_price": min_price,
        "max_price": max_price,
        "freight_factor": freight_factor,
        "cost_drop_amount": cost_drop_amount,
        "cost_drop_factory": latest_factory,
        "cost_drop_reference_price": reference_price,
        "cost_drop_reference_factory": reference_factory,
        "cost_drop_reference_date": reference_date,
        "cost_drop_reference_year_end": prior_year_end,
    }


def filter_report_df(report_df: pd.DataFrame, search_code: str, search_name: str) -> pd.DataFrame:
    filtered = report_df.copy()
    if search_code:
        codes = [code.strip() for code in search_code.split() if code.strip()]
        if codes:
            filtered = filtered[filtered["物料编码"].astype(str).isin(codes)]
    if search_name:
        filtered = filtered[
            filtered["备件简称"].astype(str).str.contains(search_name, case=False, na=False)
        ]
    return filtered


def paginate_by_material(df: pd.DataFrame, page_number: int, page_size: int = 50) -> dict:
    unique_items = df["物料编码"].astype(str).unique()
    total_items = len(unique_items)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    page_number = min(max(1, page_number), total_pages)

    start = (page_number - 1) * page_size
    end = min(start + page_size, total_items)
    current_items = unique_items[start:end]
    page_df = df[df["物料编码"].astype(str).isin(current_items)].copy()
    page_df = page_df.sort_values(["物料编码", "工厂"])

    return {
        "page_df": page_df,
        "page_number": page_number,
        "total_pages": total_pages,
        "total_items": total_items,
        "start_idx": start,
        "end_idx": end,
    }


def get_vehicle_gradient_comparison(
    df: pd.DataFrame, price_col: str, part_name: str, vehicle_rank: List[str]
) -> pd.DataFrame:
    output_columns = ["梯度排名", "梯度偏差异常", "适用车系", "备件简称", "最新成本", "最新成本有效期"]
    part_df = df[df["备件简称"].astype(str) == str(part_name)].copy()
    if part_df.empty:
        return pd.DataFrame(columns=output_columns)

    part_df["适用车系"] = part_df["适用车系"].apply(extract_first_vehicle_series_name)
    part_df = part_df[part_df["适用车系"].astype(str).str.strip().ne("")].copy()
    part_df[price_col] = pd.to_numeric(part_df[price_col], errors="coerce")
    part_df = part_df.dropna(subset=[price_col])
    if part_df.empty:
        return pd.DataFrame(columns=output_columns)

    part_df = part_df.sort_values("monitor_date", ascending=False)
    latest_df = part_df.drop_duplicates(subset=["适用车系"], keep="first").copy()
    latest_df = latest_df.sort_values([price_col, "适用车系"], ascending=[False, True]).reset_index(drop=True)
    latest_df["_cost_rank"] = np.arange(1, len(latest_df) + 1)

    if vehicle_rank:
        rank_map = {normalize_vehicle_name(name): idx for idx, name in enumerate(vehicle_rank)}
        latest_df["_rank_score"] = latest_df["适用车系"].apply(
            lambda value: rank_map.get(normalize_vehicle_name(value), 9999)
        )
        latest_df["梯度排名"] = latest_df["_rank_score"].apply(lambda value: int(value + 1) if value < 9999 else pd.NA)
    else:
        latest_df["_rank_score"] = 9999
        latest_df["梯度排名"] = pd.NA

    numeric_rank = pd.to_numeric(latest_df["梯度排名"], errors="coerce")
    latest_df["梯度偏差率"] = np.where(
        numeric_rank.gt(0),
        (latest_df["_cost_rank"] - numeric_rank).abs() / numeric_rank,
        np.nan,
    )
    latest_df["梯度偏差异常"] = pd.to_numeric(latest_df["梯度偏差率"], errors="coerce").gt(0.25)

    latest_df["_display_rank_sort"] = numeric_rank.fillna(999999)
    latest_df = latest_df.sort_values(
        ["_display_rank_sort", "适用车系"],
        ascending=[True, True],
        kind="mergesort",
    )

    output = latest_df[["梯度排名", "梯度偏差异常", "适用车系", "备件简称", price_col, "monitor_date"]].copy()
    output.columns = output_columns
    output["最新成本有效期"] = pd.to_datetime(output["最新成本有效期"]).dt.strftime("%Y-%m-%d")
    return output.reset_index(drop=True)


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def analyze_subpart_costs(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    result_cols = [
        "一级总成料号", "一级总成品名描述", "一级总成供应商名称",
        "一级总成供应商代码", "一级总成成本",
        "子零件数量", "子零件加权总和", "测算总成成本", "测算比值", "结论状态",
    ]

    if df.empty or "一级总成料号" not in df.columns:
        return pd.DataFrame(columns=result_cols)

    data = df.copy()
    if "一级总成成本" not in data.columns:
        return pd.DataFrame(columns=result_cols)

    data["一级总成料号"] = data["一级总成料号"].astype(str).str.strip()
    data = data[
        (data["一级总成料号"] != "")
        & (data["一级总成料号"] != "nan")
        & (data["一级总成料号"] != "None")
    ].copy()

    if data.empty:
        return pd.DataFrame(columns=result_cols)

    data["一级总成成本"] = pd.to_numeric(data["一级总成成本"], errors="coerce")
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")

    results = []
    for assy_no, group in data.groupby("一级总成料号", sort=False):
        latest_parts = (
            group.sort_values("monitor_date", ascending=False)
            .drop_duplicates(subset=["物料编码"], keep="first")
        )

        subpart_sum = latest_parts[price_col].sum()
        estimated_cost = subpart_sum * 1.2

        first_row = group.iloc[0]
        assy_cost = first_row.get("一级总成成本", np.nan)

        if pd.isna(assy_cost) or assy_cost == 0:
            ratio = 0.0
            status = "正常"
        else:
            ratio = round(estimated_cost / assy_cost, 4)
            status = "异常" if ratio > 1.2 else "正常"

        results.append({
            "一级总成料号": assy_no,
            "一级总成品名描述": first_row.get("一级总成品名描述", ""),
            "一级总成供应商名称": first_row.get("一级总成供应商名称", ""),
            "一级总成供应商代码": first_row.get("一级总成供应商代码", ""),
            "一级总成成本": assy_cost,
            "子零件数量": len(latest_parts),
            "子零件加权总和": round(subpart_sum, 2),
            "测算总成成本": round(estimated_cost, 2),
            "测算比值": ratio,
            "结论状态": status,
        })

    if not results:
        return pd.DataFrame(columns=result_cols)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("一级总成料号").reset_index(drop=True)
    return result_df


def get_subpart_detail(df: pd.DataFrame, price_col: str, assy_no: str) -> pd.DataFrame:
    if df.empty or "一级总成料号" not in df.columns:
        return pd.DataFrame()

    data = df.copy()
    data["一级总成料号"] = data["一级总成料号"].astype(str).str.strip()
    group = data[data["一级总成料号"] == str(assy_no).strip()]

    if group.empty:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(group["monitor_date"]):
        group = group.copy()
        group["monitor_date"] = pd.to_datetime(group["monitor_date"], errors="coerce")

    latest_parts = (
        group.sort_values("monitor_date", ascending=False)
        .drop_duplicates(subset=["物料编码"], keep="first")
    )

    detail_cols = ["物料编码", "物料名称", "备件简称", "工厂", price_col, "monitor_date"]
    available = [column for column in detail_cols if column in latest_parts.columns]
    result = latest_parts[available].copy()
    if "monitor_date" in result.columns:
        result = result.rename(columns={"monitor_date": "价格有效于"})
    if price_col in result.columns and price_col != "子零件成本":
        result = result.rename(columns={price_col: "子零件成本"})
    return result.sort_values("物料编码").reset_index(drop=True)
