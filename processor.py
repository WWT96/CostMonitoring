import glob
import html
import io
import json as _json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from config import settings


BASE_COLS = ["物料编码", "物料名称", "适用车系", "备件简称", "工厂"]
PRICE_COL_CANDIDATES = ["价格", "成本", "单价", "Price", "Cost", "含税价", "未税价"]

# 企业系统字段名 → 本系统中文标准列名的映射表。
# Java / 英文系统推送数据时，process_records_from_json 会自动按此表重命名字段。
FIELD_MAP: Dict[str, str] = {
    # 物料编码
    "partId": "物料编码",
    "materialCode": "物料编码",
    "materialId": "物料编码",
    "part_id": "物料编码",
    "material_code": "物料编码",
    # 物料名称
    "partName": "物料名称",
    "materialName": "物料名称",
    "part_name": "物料名称",
    "material_name": "物料名称",
    # 适用车系
    "vehicleSeries": "适用车系",
    "vehicle_series": "适用车系",
    "carModel": "适用车系",
    "car_model": "适用车系",
    # 备件简称
    "shortName": "备件简称",
    "short_name": "备件简称",
    "partAlias": "备件简称",
    "part_alias": "备件简称",
    # 工厂
    "factory": "工厂",
    "plant": "工厂",
    "plantCode": "工厂",
    "plant_code": "工厂",
    # 价格 / 成本
    "price": "价格",
    "cost": "成本",
    "unitPrice": "单价",
    "unit_price": "单价",
    # 日期
    "validDate": "价格有效于",
    "valid_date": "价格有效于",
    "effectiveDate": "价格有效于",
    "effective_date": "价格有效于",
    "priceDate": "价格有效于",
    "price_date": "价格有效于",
    # 一级总成
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

PROJECT_ROOT = Path(__file__).resolve().parent
_LEGACY_FEEDBACK_PATH = PROJECT_ROOT / "user_feedback.csv"
_LEGACY_SKILLS_PATH = PROJECT_ROOT / "skills_active.json"
_TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

DB_METADATA = MetaData()

EXPERT_FEEDBACK_TABLE = Table(
    "expert_feedback",
    DB_METADATA,
    Column("record_key", String(160), primary_key=True),
    Column("label", String(32), nullable=False),
    Column("labeled_at", DateTime, nullable=False),
)

CORE_COST_RECORDS_TABLE = Table(
    "core_cost_records",
    DB_METADATA,
    Column("cost_record_id", Integer, primary_key=True, autoincrement=True),
    Column("material_code", String(64), nullable=False),
    Column("cost_amount", Float, nullable=False),
    Column("monitor_date", DateTime, nullable=False),
    Column("factory", String(64), nullable=False),
    Column("material_name", String(255)),
    Column("vehicle_series", String(128)),
    Column("short_name", String(128)),
    Column("assy_part_no", String(64)),
    Column("assy_desc", String(255)),
    Column("assy_supplier_name", String(255)),
    Column("assy_supplier_code", String(64)),
    Column("assy_cost", Float),
    Column("created_at", DateTime, nullable=False),
)

COST_ANOMALY_RESULTS_TABLE = Table(
    "cost_anomaly_results",
    DB_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("_record_key", Text, unique=True),
    Column("material_code", Text, nullable=False),
    Column("material_name", Text),
    Column("vehicle_series", Text),
    Column("factory", Text),
    Column("short_name", Text),
    Column("actual_cost", Numeric(18, 4)),
    Column("effective_date", DateTime),
    Column("sample_count", Integer),
    Column("baseline_price", Numeric(18, 4)),
    Column("lower_bound", Numeric(18, 4)),
    Column("upper_bound", Numeric(18, 4)),
    Column("deviation_amount", Numeric(18, 4)),
    Column("deviation_ratio", Numeric(18, 10)),
    Column("status", Text),
    Column("expert_adjusted", Text),
    Column("decision_basis", Text),
    Column("result_mode", Text),
    Column("computed_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

SKILLS_SNAPSHOT_TABLE = Table(
    "skills_snapshot",
    DB_METADATA,
    Column("snapshot_id", String(36), primary_key=True),
    Column("version", String(16), nullable=False),
    Column("saved_at", DateTime, nullable=False),
    Column("global_sigma", Float, nullable=False),
    Column("global_weight", Integer, nullable=False),
)

SKILLS_ITEMS_TABLE = Table(
    "skills_items",
    DB_METADATA,
    Column("item_id", Integer, primary_key=True, autoincrement=True),
    Column("snapshot_id", Integer, nullable=False),
    Column("snapshot_ref", String(36)),
    Column("short_name", String(128), nullable=False),
    Column("algorithm_type", String(255)),
    Column("sigma_param", Float),
    Column("expert_weight", Integer),
    Column("alignment_rate", Float),
    Column("lower_bound", Float),
    Column("upper_bound", Float),
    Column("base_price", Float),
    Column("payload_json", Text, nullable=False),
)

SKILLS_PAYLOAD_ITEMS_TABLE = Table(
    "skills_items_payload",
    DB_METADATA,
    Column("item_id", Integer, primary_key=True, autoincrement=True),
    Column("snapshot_id", String(36), nullable=False),
    Column("short_name", String(128), nullable=False),
    Column("algorithm_type", String(255)),
    Column("sigma_param", Float),
    Column("expert_weight", Integer),
    Column("alignment_rate", Text),
    Column("lower_bound", Float),
    Column("upper_bound", Float),
    Column("base_price", Float),
    Column("payload_json", Text, nullable=False),
)

_CORE_RECORD_EXPORT_COLUMNS = [
    "material_code",
    "material_name",
    "vehicle_series",
    "short_name",
    "factory",
    "cost_amount",
    "monitor_date",
    "assy_part_no",
    "assy_desc",
    "assy_supplier_name",
    "assy_supplier_code",
    "assy_cost",
]

_ANOMALY_EXPORT_COLUMNS = [
    "result_mode",
    "_record_key",
    "material_code",
    "material_name",
    "vehicle_series",
    "factory",
    "short_name",
    "actual_cost",
    "effective_date",
    "sample_count",
    "baseline_price",
    "lower_bound",
    "upper_bound",
    "deviation_amount",
    "deviation_ratio",
    "status",
    "expert_adjusted",
    "decision_basis",
]

_ANOMALY_RESULT_COLUMN_MAPPING = {
    "_record_key": "_record_key",
    "物料编码": "material_code",
    "物料名称": "material_name",
    "适用车系": "vehicle_series",
    "工厂": "factory",
    "备件简称": "short_name",
    "实际成本": "actual_cost",
    "价格有效于": "effective_date",
    "样本量": "sample_count",
    "预测值": "baseline_price",
    "合理下限": "lower_bound",
    "合理上限": "upper_bound",
    "偏离数值": "deviation_amount",
    "偏离比例": "deviation_ratio",
    "status": "status",
    "专家校准": "expert_adjusted",
    "判定依据": "decision_basis",
    "result_mode": "result_mode",
    "computed_at": "computed_at",
}

_ANOMALY_RESULT_DB_COLUMNS = list(_ANOMALY_RESULT_COLUMN_MAPPING.values())
_ANOMALY_RESULT_TABLE_COLUMNS = ["id", *_ANOMALY_RESULT_DB_COLUMNS]

_ANOMALY_NUMERIC_COLUMNS = [
    "actual_cost",
    "sample_count",
    "baseline_price",
    "lower_bound",
    "upper_bound",
    "deviation_amount",
    "deviation_ratio",
]

_ANOMALY_REQUIRED_COLUMNS = [
    "_record_key",
    "material_code",
    "actual_cost",
]

_COST_ANOMALY_RESULTS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cost_anomaly_results (
    id SERIAL PRIMARY KEY,
    _record_key TEXT UNIQUE,
    material_code TEXT NOT NULL,
    material_name TEXT,
    vehicle_series TEXT,
    factory TEXT,
    short_name TEXT,
    actual_cost DECIMAL(18,4),
    effective_date TIMESTAMP,
    sample_count INTEGER,
    baseline_price DECIMAL(18,4),
    lower_bound DECIMAL(18,4),
    upper_bound DECIMAL(18,4),
    deviation_amount DECIMAL(18,4),
    deviation_ratio DECIMAL(18,10),
    status TEXT,
    expert_adjusted TEXT,
    decision_basis TEXT,
    result_mode TEXT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_COST_ANOMALY_RESULTS_CREATE_SQL_SQLITE = """
CREATE TABLE IF NOT EXISTS cost_anomaly_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    _record_key TEXT UNIQUE,
    material_code TEXT NOT NULL,
    material_name TEXT,
    vehicle_series TEXT,
    factory TEXT,
    short_name TEXT,
    actual_cost DECIMAL(18,4),
    effective_date TIMESTAMP,
    sample_count INTEGER,
    baseline_price DECIMAL(18,4),
    lower_bound DECIMAL(18,4),
    upper_bound DECIMAL(18,4),
    deviation_amount DECIMAL(18,4),
    deviation_ratio DECIMAL(18,10),
    status TEXT,
    expert_adjusted TEXT,
    decision_basis TEXT,
    result_mode TEXT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_DB_INIT_ERROR: Optional[Exception] = None
_COST_ANOMALY_RESULTS_RESET_DONE = False


def _build_db_engine() -> Optional[Engine]:
    global _DB_INIT_ERROR

    if not settings.db_url:
        _DB_INIT_ERROR = RuntimeError("DB_URL 未配置")
        return None

    try:
        return create_engine(settings.db_url, future=True, pool_pre_ping=True)
    except Exception as exc:
        _DB_INIT_ERROR = exc
        return None


DB_ENGINE = _build_db_engine()


def require_db_engine() -> Engine:
    if DB_ENGINE is None:
        if _DB_INIT_ERROR is not None:
            raise RuntimeError(f"数据库不可用: {_DB_INIT_ERROR}")
        raise RuntimeError("数据库不可用: 未配置 DB_URL")
    return DB_ENGINE


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NaT:
        return None
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return None
    return value


def _rows_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append({key: _normalize_scalar(value) for key, value in row.items()})
    return rows


def _chunk_rows(rows: Sequence[Dict[str, Any]], chunk_size: int = 500) -> Iterator[Sequence[Dict[str, Any]]]:
    for idx in range(0, len(rows), chunk_size):
        yield rows[idx : idx + chunk_size]


def _upsert_rows(
    table: Table,
    rows: Sequence[Dict[str, Any]],
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
) -> None:
    if not rows:
        return

    engine = require_db_engine()
    with engine.begin() as conn:
        for batch in _chunk_rows(rows):
            stmt = pg_insert(table).values(list(batch))
            update_map = {column: stmt.excluded[column] for column in update_columns}
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[table.c[column] for column in conflict_columns],
                    set_=update_map,
                )
            )


def _insert_rows(table: Table, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return

    engine = require_db_engine()
    with engine.begin() as conn:
        for batch in _chunk_rows(rows):
            conn.execute(table.insert(), list(batch))


def _count_table_rows(table: Table) -> int:
    engine = require_db_engine()
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def extract_date(val):
    if pd.isna(val):
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(val))
    return match.group(1) if match else None


def normalize_vehicle_name(name: str) -> str:
    if name is None:
        return ""
    return re.sub(r"\s+", "", str(name)).lower()


def detect_price_column(columns: Iterable[str]) -> Optional[str]:
    column_list = list(columns)
    for col in PRICE_COL_CANDIDATES:
        if col in column_list:
            return col
    for col in column_list:
        if "价格" in str(col) or "成本" in str(col):
            return col
    return None


def escape_html_text(value) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value), quote=True)


def parse_vehicle_rank_config(text: str) -> List[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def process_records_from_json(
    records: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """接收从 JSON 解析而来的记录列表，应用 FIELD_MAP 字段映射后交由 process_dataframe 处理。

    Args:
        records: 键值对列表，键可以是中文标准列名或 FIELD_MAP 中定义的英文/Java 映射名。

    Returns:
        (DataFrame, price_col, error_message) 三元组，出错时前两项为 None。
    """
    if not records:
        return None, None, "输入记录列表为空"

    df = pd.DataFrame(records)

    # 仅重命名在 FIELD_MAP 中存在、且目标列名还不在 df 中的列，避免覆盖已有中文列
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

    possible_date_cols = [col for col in df.columns if "价格有效于" in str(col)]
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

    # 一级总成字段清洗
    if "一级总成成本" in result.columns:
        result["一级总成成本"] = pd.to_numeric(result["一级总成成本"], errors="coerce")
    for _assy_col in ["一级总成料号", "一级总成品名描述", "一级总成供应商名称", "一级总成供应商代码"]:
        if _assy_col in result.columns:
            result[_assy_col] = result[_assy_col].astype(str).replace({"nan": "", "None": ""})

    return result, price_col, None


def _prepare_core_cost_records(
    df: pd.DataFrame,
    price_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return pd.DataFrame(columns=_CORE_RECORD_EXPORT_COLUMNS + ["created_at"]), price_col or "成本"

    data = df.copy()
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
        "一级总成料号": None,
        "一级总成品名描述": None,
        "一级总成供应商名称": None,
        "一级总成供应商代码": None,
        "一级总成成本": None,
    }.items():
        if col_name not in data.columns:
            data[col_name] = default_value

    data[resolved_price_col] = pd.to_numeric(data[resolved_price_col], errors="coerce")
    if "一级总成成本" in data.columns:
        data["一级总成成本"] = pd.to_numeric(data["一级总成成本"], errors="coerce")

    prepared = pd.DataFrame(
        {
            "material_code": data["物料编码"].astype(str),
            "material_name": data["物料名称"],
            "vehicle_series": data["适用车系"],
            "short_name": data["备件简称"],
            "factory": data["工厂"],
            "cost_amount": data[resolved_price_col],
            "monitor_date": data["monitor_date"],
            "assy_part_no": data["一级总成料号"],
            "assy_desc": data["一级总成品名描述"],
            "assy_supplier_name": data["一级总成供应商名称"],
            "assy_supplier_code": data["一级总成供应商代码"],
            "assy_cost": data["一级总成成本"],
            "created_at": datetime.now(),
        }
    )
    prepared = prepared.dropna(subset=["material_code", "factory", "monitor_date", "cost_amount"])
    prepared = prepared.drop_duplicates(
        subset=["material_code", "factory", "monitor_date", "cost_amount"],
        keep="last",
    ).reset_index(drop=True)
    return prepared, resolved_price_col


def persist_core_cost_records(
    df: pd.DataFrame,
    price_col: Optional[str] = None,
    mode: str = "incremental",
) -> int:
    prepared, _ = _prepare_core_cost_records(df, price_col)
    rows = _rows_from_dataframe(prepared)
    engine = require_db_engine()

    if mode not in {"full", "incremental"}:
        raise ValueError(f"不支持的持久化模式: {mode}")

    with engine.begin() as conn:
        conn.execute(delete(CORE_COST_RECORDS_TABLE))
    _insert_rows(CORE_COST_RECORDS_TABLE, rows)
    return len(rows)


def load_core_cost_records() -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    engine = require_db_engine()
    query = select(*(CORE_COST_RECORDS_TABLE.c[column] for column in _CORE_RECORD_EXPORT_COLUMNS))
    df = pd.read_sql(query, engine)
    if df.empty:
        return None, None, "云端数据库中暂无核心成本数据"

    df["monitor_date"] = pd.to_datetime(df["monitor_date"], errors="coerce")
    df = df.rename(
        columns={
            "material_code": "物料编码",
            "material_name": "物料名称",
            "vehicle_series": "适用车系",
            "short_name": "备件简称",
            "factory": "工厂",
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


def _prepare_anomaly_results(result_df: pd.DataFrame, result_mode: str) -> pd.DataFrame:
    target_columns = _ANOMALY_RESULT_DB_COLUMNS
    if result_df is None or result_df.empty:
        return pd.DataFrame(columns=target_columns)

    data = result_df.copy()
    optional_defaults = {
        "物料名称": None,
        "适用车系": None,
        "工厂": None,
        "专家校准": None,
        "判定依据": None,
        "样本量": 0,
        "status": "正常",
    }
    for source_column, default_value in optional_defaults.items():
        target_column = _ANOMALY_RESULT_COLUMN_MAPPING[source_column]
        if source_column not in data.columns and target_column not in data.columns:
            data[source_column] = default_value

    data["result_mode"] = result_mode
    data["computed_at"] = datetime.now()
    mapping = dict(_ANOMALY_RESULT_COLUMN_MAPPING)
    prepared = data.rename(columns=mapping)

    for column_name in target_columns:
        if column_name not in prepared.columns:
            prepared[column_name] = None

    # 严格只保留数据库表允许的英文列，避免旧中文列或临时列进入 to_sql。
    prepared = prepared[target_columns]

    for column_name in _ANOMALY_NUMERIC_COLUMNS:
        if column_name not in prepared.columns:
            continue
        prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")

    if "sample_count" in prepared.columns:
        prepared["sample_count"] = prepared["sample_count"].fillna(0).astype(int)
    if "effective_date" in prepared.columns:
        prepared["effective_date"] = pd.to_datetime(prepared["effective_date"], errors="coerce")

    for column_name in ["result_mode", "_record_key", "material_code", "short_name", "status"]:
        if column_name not in prepared.columns:
            continue
        prepared[column_name] = prepared[column_name].astype("string").str.strip()
        prepared.loc[prepared[column_name] == "", column_name] = pd.NA

    prepared["_record_key"] = prepared["_record_key"].apply(
        lambda value: _to_storage_record_key(value, result_mode) if pd.notna(value) else value
    )
    prepared = prepared.drop_duplicates(subset=["_record_key"], keep="last")
    prepared = prepared.dropna(subset=["material_code"])
    prepared = prepared.dropna(subset=_ANOMALY_REQUIRED_COLUMNS)
    return prepared.reset_index(drop=True)


def save_cost_anomaly_results(result_df: pd.DataFrame, result_mode: str = "raw") -> int:
    _ensure_cost_anomaly_results_table()
    df = _prepare_anomaly_results(result_df, result_mode)
    if df.empty:
        print("[cost_anomaly_results] 无可写入记录，清洗后结果为空")
        return 0

    engine = require_db_engine()
    try:
        print(f"即将写入数据库的列: {df.columns.tolist()}")
        print("[cost_anomaly_results] to_sql 前 head(1):")
        print(df.head(1).to_string(index=False))
        with engine.begin() as conn:
            conn.execute(
                delete(COST_ANOMALY_RESULTS_TABLE).where(
                    COST_ANOMALY_RESULTS_TABLE.c.result_mode == result_mode
                )
            )
            df.to_sql(
                COST_ANOMALY_RESULTS_TABLE.name,
                conn,
                if_exists="append",
                index=False,
            )
        print("[cost_anomaly_results] to_sql 后 columns:")
        print(df.columns.tolist())
        print("[cost_anomaly_results] to_sql 后 head(1):")
        print(df.head(1).to_string(index=False))
    except Exception as exc:
        print(f"[cost_anomaly_results] 写入失败: {exc}")
        print("[cost_anomaly_results] 原始 result_df.columns 预览:")
        print(pd.DataFrame({"column_name": list(result_df.columns)}).head(20).to_string(index=False))
        print("[cost_anomaly_results] 待写入列名:")
        print(df.columns.tolist())
        print("[cost_anomaly_results] 待写入前5行:")
        print(df.head().to_string(index=False))
        raise
    return len(df)


def load_cost_anomaly_results(result_mode: str = "raw") -> pd.DataFrame:
    _ensure_cost_anomaly_results_table()
    engine = require_db_engine()
    query = (
        select(*(COST_ANOMALY_RESULTS_TABLE.c[column] for column in _ANOMALY_EXPORT_COLUMNS))
        .where(COST_ANOMALY_RESULTS_TABLE.c.result_mode == result_mode)
    )
    df = pd.read_sql(query, engine)
    if df.empty:
        return df

    if "effective_date" in df.columns:
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    if "_record_key" in df.columns:
        df["_record_key"] = df["_record_key"].apply(
            lambda value: _from_storage_record_key(value, result_mode)
        )
    return (
        df.rename(
            columns={
                "material_code": "物料编码",
                "material_name": "物料名称",
                "vehicle_series": "适用车系",
                "factory": "工厂",
                "short_name": "备件简称",
                "actual_cost": "实际成本",
                "effective_date": "价格有效于",
                "sample_count": "样本量",
                "baseline_price": "预测值",
                "lower_bound": "合理下限",
                "upper_bound": "合理上限",
                "deviation_amount": "偏离数值",
                "deviation_ratio": "偏离比例",
                "expert_adjusted": "专家校准",
                "decision_basis": "判定依据",
            }
        )
        .drop(columns=["result_mode"], errors="ignore")
    )


def load_data_from_uploaded_files(
    uploaded_files: List[Any],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """处理来自 st.file_uploader 的上传文件列表，返回与 load_data_from_folder 相同格式的三元组。

    Parameters
    ----------
    uploaded_files : list
        Streamlit ``UploadedFile`` 对象的列表（支持 .xlsx / .xls / .csv）。

    Returns
    -------
    (DataFrame, price_col, error_message)  出错时前两项为 None。
    """
    if not uploaded_files:
        return None, None, "未提供任何上传文件"

    df_list: List[pd.DataFrame] = []
    price_col_detected: Optional[str] = None

    for uploaded_file in uploaded_files:
        try:
            name: str = uploaded_file.name.lower()
            raw_bytes = uploaded_file.getvalue()
            if name.endswith(".csv"):
                try:
                    raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8")
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="gbk")
            else:
                raw_df = pd.read_excel(io.BytesIO(raw_bytes))

            # 将 FIELD_MAP 中定义的企业系统字段名映射为中文标准列名
            rename_map = {
                src: dst
                for src, dst in FIELD_MAP.items()
                if src in raw_df.columns and dst not in raw_df.columns
            }
            if rename_map:
                raw_df = raw_df.rename(columns=rename_map)

            processed_df, detected_price_col, _ = process_dataframe(raw_df)
            if processed_df is not None:
                df_list.append(processed_df)
                if not price_col_detected:
                    price_col_detected = detected_price_col
        except Exception:
            continue

    if not df_list:
        return None, None, "所有上传文件均未能成功解析，请检查文件格式和必要列名"

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df, price_col_detected, None


def load_data_from_folder(
    folder_path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    if not os.path.isdir(folder_path):
        return None, None, f"路径不存在: {folder_path}"

    all_files = (
        glob.glob(os.path.join(folder_path, "*.xlsx"))
        + glob.glob(os.path.join(folder_path, "*.xls"))
        + glob.glob(os.path.join(folder_path, "*.csv"))
    )
    if not all_files:
        return None, None, "路径下没有找到 Excel 或 CSV 文件"

    df_list = []
    price_col_detected = None
    for filename in all_files:
        try:
            if filename.endswith(".csv"):
                try:
                    raw_df = pd.read_csv(filename, encoding="utf-8")
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(filename, encoding="gbk")
            else:
                raw_df = pd.read_excel(filename)

            rename_map = {
                src: dst
                for src, dst in FIELD_MAP.items()
                if src in raw_df.columns and dst not in raw_df.columns
            }
            if rename_map:
                raw_df = raw_df.rename(columns=rename_map)

            processed_df, detected_price_col, _ = process_dataframe(raw_df)
            if processed_df is not None:
                df_list.append(processed_df)
                if not price_col_detected:
                    price_col_detected = detected_price_col
        except Exception:
            continue

    if not df_list:
        return None, None, "没有成功读取到有效数据"

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df, price_col_detected, None


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


def generate_trend_report(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    pivot_df = generate_pivot_report(df, price_col)
    if pivot_df.empty:
        return pivot_df

    trend_df = pivot_df.copy()
    price_cols = [col for col in trend_df.columns if col.startswith("价格变动")]
    price_cols.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    trend_cols = []
    for i in range(len(price_cols) - 1):
        curr_col = price_cols[i]
        next_col = price_cols[i + 1]
        trend_col = f"变动趋势{i + 1}"
        trend_df[curr_col] = pd.to_numeric(trend_df[curr_col], errors="coerce")
        trend_df[next_col] = pd.to_numeric(trend_df[next_col], errors="coerce")
        trend_df[trend_col] = trend_df[next_col] - trend_df[curr_col]
        trend_cols.append(trend_col)

    return trend_df[BASE_COLS + trend_cols].copy()


def get_material_metrics(item_df: pd.DataFrame, price_col: str) -> dict:
    latest_record = item_df.loc[item_df["monitor_date"].idxmax()]
    earliest_record = item_df.loc[item_df["monitor_date"].idxmin()]

    latest_price = latest_record[price_col]
    earliest_price = earliest_record[price_col]
    min_price = item_df[price_col].min()
    max_price = item_df[price_col].max()

    max_change_pct = (max_price - min_price) / max_price if max_price else 0
    cum_change_pct = (latest_price - earliest_price) / earliest_price if earliest_price else 0

    return {
        "latest_price": latest_price,
        "earliest_price": earliest_price,
        "min_price": min_price,
        "max_price": max_price,
        "max_change_pct": max_change_pct,
        "cum_change_pct": cum_change_pct,
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


def render_merged_html_table(df: pd.DataFrame, value_cols: List[str], is_trend_mode: bool = False) -> str:
    if df.empty:
        return "<div style='text-align:center; padding: 20px;'>暂无数据</div>"

    all_cols = BASE_COLS + value_cols
    html_parts = [
        """
        <div style="width: 100%; overflow-x: auto; margin-bottom: 20px;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 14px; border: 1px solid #dee2e6;">
                <thead>
                    <tr style="background-color: #f8f9fa; color: #495057;">
        """
    ]

    for col in all_cols:
        html_parts.append(
            f'<th style="padding: 12px; border: 1px solid #dee2e6; text-align: center !important; vertical-align: middle !important; font-weight: 600; white-space: nowrap;">{escape_html_text(col)}</th>'
        )

    html_parts.append("</tr></thead><tbody>")

    grouped = df.sort_values(["物料编码", "工厂"]).groupby("物料编码", sort=False)
    for _, group in grouped:
        row_count = len(group)
        first_row = True
        for _, row in group.iterrows():
            html_parts.append(
                '<tr style="background-color: #ffffff; transition: background-color 0.1s ease;" onmouseover="this.style.backgroundColor=\'#f1f3f5\'" onmouseout="this.style.backgroundColor=\'#ffffff\'">'
            )
            for col_idx, col_name in enumerate(all_cols):
                val = row.get(col_name, "")
                if col_idx < 4:
                    if first_row:
                        val_text = escape_html_text(val)
                        html_parts.append(
                            f'<td rowspan="{row_count}" style="padding: 10px; border: 1px solid #dee2e6; text-align: center !important; vertical-align: middle !important; background-color: #fff;">{val_text}</td>'
                        )
                    continue

                cell_style = (
                    "padding: 8px; border: 1px solid #dee2e6; text-align: center !important; vertical-align: middle !important;"
                )
                display_val = ""

                if pd.isna(val) or val == "":
                    display_val = ""
                elif isinstance(val, (int, float)):
                    display_val = f"{val:,.2f}"
                    if is_trend_mode and col_name in value_cols:
                        if val > 0:
                            cell_style += " color: #e74c3c; font-weight: bold;"
                            display_val = f"▲ {display_val}"
                        elif val < 0:
                            cell_style += " color: #27ae60; font-weight: bold;"
                            display_val = f"▼ {display_val}"
                        else:
                            cell_style += " color: #2c3e50;"
                            display_val = "-"
                else:
                    display_val = str(val)

                html_parts.append(f'<td style="{cell_style}">{escape_html_text(display_val)}</td>')
            html_parts.append("</tr>")
            first_row = False

    html_parts.append("</tbody></table></div>")
    return "".join(html_parts)


def get_vehicle_gradient_comparison(
    df: pd.DataFrame, price_col: str, part_name: str, vehicle_rank: List[str]
) -> pd.DataFrame:
    part_df = df[df["备件简称"].astype(str) == str(part_name)].copy()
    if part_df.empty:
        return pd.DataFrame(
            columns=["梯度排名", "适用车系", "备件简称", "最新成本", "最新成本有效期"]
        )

    part_df = part_df.sort_values("monitor_date", ascending=False)
    latest_df = part_df.drop_duplicates(subset=["适用车系"], keep="first").copy()

    if vehicle_rank:
        rank_map = {normalize_vehicle_name(name): idx for idx, name in enumerate(vehicle_rank)}
        latest_df["rank_score"] = latest_df["适用车系"].apply(
            lambda x: rank_map.get(normalize_vehicle_name(x), 9999)
        )
        latest_df = latest_df.sort_values(["rank_score", "适用车系"])
        latest_df["梯度排名"] = latest_df["rank_score"].apply(lambda x: str(x + 1) if x < 9999 else "-")
    else:
        latest_df = latest_df.sort_values("适用车系")
        latest_df["梯度排名"] = "-"

    output = latest_df[["梯度排名", "适用车系", "备件简称", price_col, "monitor_date"]].copy()
    output.columns = ["梯度排名", "适用车系", "备件简称", "最新成本", "最新成本有效期"]
    output["最新成本有效期"] = pd.to_datetime(output["最新成本有效期"]).dt.strftime("%Y-%m-%d")
    return output


def render_center_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div style='text-align:center; padding: 20px;'>暂无数据</div>"

    html = [
        """
        <div style="width: 100%; overflow-x: auto; margin-bottom: 20px;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 14px; border: 1px solid #dee2e6;">
                <thead>
                    <tr style="background-color: #f8f9fa; color: #495057;">
        """
    ]
    for col in df.columns:
        html.append(
            f'<th style="padding: 12px; border: 1px solid #dee2e6; text-align: center !important; font-weight: 600;">{escape_html_text(col)}</th>'
        )
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        html.append(
            '<tr style="background-color: #ffffff; transition: background-color 0.1s ease;" onmouseover="this.style.backgroundColor=\'#f1f3f5\'" onmouseout="this.style.backgroundColor=\'#ffffff\'">'
        )
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                show_val = ""
            elif col == "最新成本" and isinstance(val, (int, float)):
                show_val = f"{val:,.2f}"
            else:
                show_val = str(val)
            html.append(
                f'<td style="padding: 10px; border: 1px solid #dee2e6; text-align: center !important;">{escape_html_text(show_val)}</td>'
            )
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


# ---------------------------------------------------------------------------
# 拆分件成本监控
# ---------------------------------------------------------------------------

def analyze_subpart_costs(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """分析一级总成与子零件的联动成本。

    逻辑：
    1. 仅提取 一级总成料号 不为空的行。
    2. 以 一级总成料号 为聚合维度，对每个总成下的每一个物料编码（子零件）
       仅取最新日期的成本记录，然后去重合并。
    3. 子零件加权总和 = Σ(各子零件最新成本)
       测算总成成本 = 子零件加权总和 * 1.2
       测算比值 = 测算总成成本 / 一级总成成本
    4. 若 测算比值 <= 1.2 → 正常（绿色）
       若 测算比值 > 1.2 → 异常（红色）

    Returns
    -------
    pd.DataFrame
        每行一个总成，包含子零件汇总和判定结果。
    """
    RESULT_COLS = [
        "一级总成料号", "一级总成品名描述", "一级总成供应商名称",
        "一级总成供应商代码", "一级总成成本",
        "子零件数量", "子零件加权总和", "测算总成成本", "测算比值", "结论状态",
    ]

    if df.empty or "一级总成料号" not in df.columns:
        return pd.DataFrame(columns=RESULT_COLS)

    data = df.copy()

    # 确保关键列存在
    if "一级总成成本" not in data.columns:
        return pd.DataFrame(columns=RESULT_COLS)

    # 过滤：仅保留一级总成料号不为空的行
    data["一级总成料号"] = data["一级总成料号"].astype(str).str.strip()
    data = data[
        (data["一级总成料号"] != "")
        & (data["一级总成料号"] != "nan")
        & (data["一级总成料号"] != "None")
    ].copy()

    if data.empty:
        return pd.DataFrame(columns=RESULT_COLS)

    data["一级总成成本"] = pd.to_numeric(data["一级总成成本"], errors="coerce")
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")

    results = []
    for assy_no, group in data.groupby("一级总成料号", sort=False):
        # 对每个子零件（物料编码）取最新日期的成本
        latest_parts = (
            group.sort_values("monitor_date", ascending=False)
            .drop_duplicates(subset=["物料编码"], keep="first")
        )

        subpart_sum = latest_parts[price_col].sum()
        estimated_cost = subpart_sum * 1.2

        # 取总成信息（取第一行的元数据）
        first_row = group.iloc[0]
        assy_cost = first_row.get("一级总成成本", np.nan)

        # 计算测算比值，防止除零
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
        return pd.DataFrame(columns=RESULT_COLS)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("一级总成料号").reset_index(drop=True)
    return result_df


def get_subpart_detail(df: pd.DataFrame, price_col: str, assy_no: str) -> pd.DataFrame:
    """返回指定总成下所有子零件的最新成本明细。"""
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
    available = [c for c in detail_cols if c in latest_parts.columns]
    result = latest_parts[available].copy()
    if "monitor_date" in result.columns:
        result = result.rename(columns={"monitor_date": "价格有效于"})
    if price_col in result.columns and price_col != "子零件成本":
        result = result.rename(columns={price_col: "子零件成本"})
    return result.sort_values("物料编码").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 唯一主键 & 专家标注管理
# ---------------------------------------------------------------------------

def make_record_key(row) -> str:
    """生成唯一记录主键：物料编码_工厂_日期_成本。

    用于标注持久化时唯一确定每条历史记录。
    """
    code = str(row.get("物料编码", ""))
    factory = str(row.get("工厂", ""))
    date_val = row.get("价格有效于", "")
    if hasattr(date_val, "strftime"):
        date_str = date_val.strftime("%Y-%m-%d")
    else:
        date_str = str(date_val)[:10]
    cost = row.get("实际成本", "")
    if isinstance(cost, float):
        cost = f"{cost:.4f}"
    return f"{code}_{factory}_{date_str}_{cost}"


class LabelManager:
    """管理用户对异常检测结果的手动标注，持久化到 expert_feedback 表。"""

    _COLUMNS = ["record_key", "label", "labeled_at"]

    def __init__(self, table_name: str = "expert_feedback"):
        self._table_name = table_name

    # --- 读取 ---

    def get_labels(self) -> Dict[str, str]:
        """返回 ``{record_key: label}`` 字典。数据库未就绪时返回空字典。"""
        if DB_ENGINE is None:
            return {}
        try:
            query = select(EXPERT_FEEDBACK_TABLE.c.record_key, EXPERT_FEEDBACK_TABLE.c.label)
            df = pd.read_sql(query, require_db_engine())
            if df.empty:
                return {}
            return dict(zip(df["record_key"].astype(str), df["label"].astype(str)))
        except Exception:
            return {}

    def count(self) -> int:
        """已标注记录总数。"""
        if DB_ENGINE is None:
            return 0
        try:
            return _count_table_rows(EXPERT_FEEDBACK_TABLE)
        except Exception:
            return 0

    # --- 写入 ---

    def save_label(self, key: str, status: str) -> None:
        """保存或更新单条标注记录。"""
        now = datetime.now()
        _upsert_rows(
            EXPERT_FEEDBACK_TABLE,
            [{"record_key": key, "label": status, "labeled_at": now}],
            conflict_columns=["record_key"],
            update_columns=["label", "labeled_at"],
        )

    def save_labels_batch(self, updates: Dict[str, str]) -> None:
        """批量保存标注记录。"""
        if not updates:
            return
        now = datetime.now()
        rows = [
            {"record_key": record_key, "label": label, "labeled_at": now}
            for record_key, label in updates.items()
        ]
        _upsert_rows(
            EXPERT_FEEDBACK_TABLE,
            rows,
            conflict_columns=["record_key"],
            update_columns=["label", "labeled_at"],
        )

    def delete_labels(self, keys_to_remove) -> int:
        """删除指定 record_key 的标注，返回实际删除数量。"""
        keys = [str(key) for key in keys_to_remove if str(key)]
        if not keys:
            return 0
        with require_db_engine().begin() as conn:
            result = conn.execute(
                delete(EXPERT_FEEDBACK_TABLE).where(EXPERT_FEEDBACK_TABLE.c.record_key.in_(keys))
            )
        return int(result.rowcount or 0)

    def clear_all(self) -> None:
        """清空所有标注。"""
        with require_db_engine().begin() as conn:
            conn.execute(delete(EXPERT_FEEDBACK_TABLE))

    def file_row_count(self) -> int:
        """返回 expert_feedback 表中的数据行数，用于快速一致性校验。"""
        return self.count()


def get_latest_feedback() -> Dict[str, str]:
    """单源读取：每次都从 expert_feedback 表获取最新专家标注，不使用缓存。"""
    return label_manager.get_labels()


def _reanchor_cluster_price(
    cluster_samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    fallback: float,
) -> float:
    """在已选定的邻居圈内重算基准价，并强制约束在边界内。"""
    anchor = float(np.median(cluster_samples)) if cluster_samples.size else float(fallback)
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    return float(np.clip(anchor, lower_bound, upper_bound))


# 模块级单例，供 app.py 直接使用
label_manager = LabelManager()


@st.cache_data
def detect_cost_anomalies(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    基于密度连接（Density-Linked）的全量历史异常检测。
    - KDE 寻找主密度峰值
    - KNN 局部间距 + Elbow 识别自然断层
    - 连接主群体并保护梯度定价区间
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    if price_col not in df.columns:
        raise ValueError(f"找不到价格列: {price_col}")

    # 依赖检查
    try:
        from sklearn.neighbors import KernelDensity, NearestNeighbors
    except Exception as e:
        raise ImportError("缺少依赖 scikit-learn，请先安装：pip install scikit-learn") from e

    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")

    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=["物料编码", "备件简称", "monitor_date", price_col])

    # 避免 rename 后出现重复的"价格有效于"列（原始日期列和 monitor_date 同名）
    if "价格有效于" in data.columns and "monitor_date" in data.columns:
        data = data.drop(columns=["价格有效于"])
    data = data.rename(columns={price_col: "实际成本", "monitor_date": "价格有效于"})
    data["样本量"] = data.groupby("备件简称")["物料编码"].transform("size")

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

    def _build_components(values: np.ndarray, break_gap_idx: set):
        comps = []
        s = 0
        for i in range(values.size - 1):
            if i in break_gap_idx:
                comps.append((s, i))
                s = i + 1
        comps.append((s, values.size - 1))
        return comps

    results = []
    for short_name, group in data.groupby("备件简称", sort=False):
        g = group.copy()
        n = len(g)

        # 小样本直接统计分位数回退
        if n < 10:
            q_low = float(g["实际成本"].quantile(0.05))
            q_high = float(g["实际成本"].quantile(0.95))
            baseline = float(g["实际成本"].median())
            g["预测值"] = baseline
            g["合理下限"] = max(0.0, q_low)
            g["合理上限"] = q_high
            g["偏离数值"] = g["实际成本"] - g["预测值"]
            g["偏离比例"] = g["偏离数值"] / g["预测值"].replace(0, pd.NA)
            g["status"] = "正常（小样本数据）"
            g.loc[g["实际成本"] > g["合理上限"], "status"] = "异常偏高（小样本数据）"
            g.loc[g["实际成本"] < g["合理下限"], "status"] = "异常偏低（小样本数据）"
            results.append(g)
            continue

        vals = g["实际成本"].to_numpy(dtype=float)
        uniq_vals, uniq_counts = np.unique(vals, return_counts=True)
        if uniq_vals.size == 1:
            single_val = float(uniq_vals[0])
            g["预测值"] = single_val
            g["合理下限"] = max(0.0, single_val)
            g["合理上限"] = single_val
            g["偏离数值"] = 0.0
            g["偏离比例"] = 0.0
            g["status"] = "正常"
            results.append(g)
            continue

        # Step 1: KDE 寻峰
        std_val = float(np.std(vals))
        iqr_val = float(np.subtract(*np.percentile(vals, [75, 25])))
        spread = min(x for x in [std_val, iqr_val] if x > 0) if (std_val > 0 and iqr_val > 0) else max(std_val, iqr_val)
        if spread <= 0:
            spread = float(np.mean(np.abs(np.diff(uniq_vals)))) if uniq_vals.size > 1 else 1.0
        bandwidth = spread / np.sqrt(max(1, n))
        if bandwidth <= 0:
            bandwidth = 1.0

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(vals.reshape(-1, 1))
        grid_size = int(min(600, max(200, uniq_vals.size * 4)))
        grid = np.linspace(float(np.min(vals)), float(np.max(vals)), grid_size)
        density = np.exp(kde.score_samples(grid.reshape(-1, 1)))
        peak_price = float(grid[int(np.argmax(density))])

        # Step 2: 动态步长扫描（局部 KNN 间距 + Elbow）
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

        right_cut_gap = peak_idx + right_elbow if right_seq.size else norm_gaps.size
        left_cut_gap = peak_idx - 1 - left_elbow if left_seq.size else -1

        # Step 3: 领土扩张 + 梯度保护
        # 先按主方向阈值构造天然断层
        threshold_candidates = []
        if right_seq.size:
            threshold_candidates.append(float(right_seq[right_elbow]))
        if left_seq.size:
            threshold_candidates.append(float(left_seq[left_elbow]))
        base_threshold = float(np.median(threshold_candidates)) if threshold_candidates else float(np.median(norm_gaps))

        break_idx = set(np.where(norm_gaps > base_threshold)[0].tolist())
        components = _build_components(uniq_vals, break_idx)
        comp_counts = []
        comp_density = []
        for s, e in components:
            c = int(np.sum(uniq_counts[s : e + 1]))
            span = float(max(uniq_vals[e] - uniq_vals[s], np.finfo(float).eps))
            comp_counts.append(c)
            comp_density.append(c / span)

        peak_comp_idx = next(i for i, (s, e) in enumerate(components) if s <= peak_idx <= e)
        merged_left, merged_right = components[peak_comp_idx]

        global_dispersion = float(np.std(vals))
        if global_dispersion <= 0:
            global_dispersion = float(np.mean(np.abs(gaps))) if gaps.size else 1.0
        gap_ratio = gaps / global_dispersion if global_dispersion > 0 else gaps
        gap_ratio_threshold = float(np.median(gap_ratio)) if gap_ratio.size else 0.0
        density_threshold = float(np.median(comp_density)) if comp_density else 0.0

        # 左右邻接扩张：仅在“缺口相对全局离散度不突兀且两侧有密度”时合并
        current_idx = peak_comp_idx
        while current_idx - 1 >= 0:
            prev_s, prev_e = components[current_idx - 1]
            curr_s, curr_e = components[current_idx]
            boundary_gap_idx = prev_e
            cond_gap = boundary_gap_idx < gap_ratio.size and gap_ratio[boundary_gap_idx] <= gap_ratio_threshold
            cond_density = comp_density[current_idx - 1] >= density_threshold and comp_density[current_idx] >= density_threshold
            if cond_gap and cond_density:
                merged_left = prev_s
                current_idx -= 1
            else:
                break

        current_idx = peak_comp_idx
        while current_idx + 1 < len(components):
            curr_s, curr_e = components[current_idx]
            next_s, next_e = components[current_idx + 1]
            boundary_gap_idx = curr_e
            cond_gap = boundary_gap_idx < gap_ratio.size and gap_ratio[boundary_gap_idx] <= gap_ratio_threshold
            cond_density = comp_density[current_idx + 1] >= density_threshold and comp_density[current_idx] >= density_threshold
            if cond_gap and cond_density:
                merged_right = next_e
                current_idx += 1
            else:
                break

        component_lower_bound = float(uniq_vals[merged_left])
        component_upper_bound = float(uniq_vals[merged_right])
        cluster_samples = vals[
            (vals >= component_lower_bound) & (vals <= component_upper_bound)
        ]

        lower_bound = max(0.0, component_lower_bound)
        upper_bound = component_upper_bound
        peak_price = _reanchor_cluster_price(
            cluster_samples,
            lower_bound,
            upper_bound,
            peak_price,
        )

        # 记录每个价格落入的组件，便于识别“孤立小群”
        comp_index_arr = np.zeros(uniq_vals.size, dtype=int)
        for comp_id, (s, e) in enumerate(components):
            comp_index_arr[s : e + 1] = comp_id
        value_to_comp = {float(v): int(comp_index_arr[i]) for i, v in enumerate(uniq_vals)}
        comp_count_series = pd.Series(comp_counts)
        small_comp_threshold = float(comp_count_series.median()) if not comp_count_series.empty else 0.0

        peak_price = max(lower_bound, min(peak_price, upper_bound))

        g["预测值"] = peak_price
        g["合理下限"] = lower_bound
        g["合理上限"] = upper_bound
        g["偏离数值"] = g["实际成本"] - g["预测值"]
        g["偏离比例"] = g["偏离数值"] / g["预测值"].replace(0, pd.NA)

        g["status"] = "正常"
        g.loc[g["实际成本"] > g["合理上限"], "status"] = "异常偏高"
        g.loc[g["实际成本"] < g["合理下限"], "status"] = "异常偏低"

        # 孤立小群且低于合理下限 -> 严重异常偏低
        def _is_isolated_low(v):
            comp_id = value_to_comp.get(float(v), -1)
            if comp_id < 0 or comp_id >= len(comp_counts):
                return False
            return (comp_counts[comp_id] <= small_comp_threshold) and (v < lower_bound)

        isolated_mask = g["实际成本"].apply(_is_isolated_low)
        g.loc[isolated_mask, "status"] = "严重异常偏低"

        results.append(g)

    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if result_df.empty:
        return result_df

    # 为每条记录生成唯一主键，供标注时精确定位
    result_df["_record_key"] = result_df.apply(make_record_key, axis=1)

    cols = [
        "_record_key",
        "物料编码",
        "物料名称" if "物料名称" in result_df.columns else None,
        "适用车系" if "适用车系" in result_df.columns else None,
        "工厂" if "工厂" in result_df.columns else None,
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
    ]
    cols = [c for c in cols if c is not None]
    result_df = result_df[cols].sort_values(["备件简称", "物料编码"]).reset_index(drop=True)
    save_cost_anomaly_results(result_df, result_mode="raw")
    return result_df


# ---------------------------------------------------------------------------
# Skills 技能书持久化（闭环自学习）
# ---------------------------------------------------------------------------


def save_skills(skills: list, sigma: float = 1.0, weight: int = 80) -> str:
    """将 Skills 列表持久化到 skills_snapshot / skills_items 表。"""
    snapshot_id = str(uuid4())
    saved_at = datetime.now()
    _insert_rows(
        SKILLS_SNAPSHOT_TABLE,
        [
            {
                "snapshot_id": snapshot_id,
                "version": "1.0",
                "saved_at": saved_at,
                "global_sigma": round(float(sigma), 4),
                "global_weight": int(weight),
            }
        ],
    )

    rows = []
    for skill in skills:
        bounds = skill.get("成本合理区间边界", {}) or {}
        alignment = skill.get("经验对齐率")
        alignment_rate = float(alignment) if isinstance(alignment, (int, float)) else None
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "short_name": str(skill.get("备件简称", "")),
                "algorithm_type": str(skill.get("适用算法", "")),
                "sigma_param": float(skill.get("当前σ参数", sigma)) if skill.get("当前σ参数") is not None else None,
                "expert_weight": int(skill.get("偏置权重", weight)) if skill.get("偏置权重") is not None else None,
                "alignment_rate": alignment_rate,
                "lower_bound": float(bounds.get("合理下限")) if bounds.get("合理下限") is not None else None,
                "upper_bound": float(bounds.get("合理上限")) if bounds.get("合理上限") is not None else None,
                "base_price": float(bounds.get("预测值")) if bounds.get("预测值") is not None else None,
                "payload_json": _json.dumps(skill, ensure_ascii=False),
            }
        )
    _insert_rows(SKILLS_PAYLOAD_ITEMS_TABLE, rows)
    return "skills_snapshot / skills_items"


def load_skills() -> Optional[Dict]:
    """加载最近一次 Skills 快照。"""
    if DB_ENGINE is None:
        return None
    try:
        engine = require_db_engine()
        with engine.connect() as conn:
            snapshot = conn.execute(
                select(SKILLS_SNAPSHOT_TABLE)
                .order_by(SKILLS_SNAPSHOT_TABLE.c.saved_at.desc())
                .limit(1)
            ).mappings().first()
        if snapshot is None:
            return None

        items_query = (
            select(
                SKILLS_PAYLOAD_ITEMS_TABLE.c.payload_json,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.short_name,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.algorithm_type,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.sigma_param,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.expert_weight,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.alignment_rate,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.lower_bound,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.upper_bound,
                SKILLS_PAYLOAD_ITEMS_TABLE.c.base_price,
            )
            .where(SKILLS_PAYLOAD_ITEMS_TABLE.c.snapshot_id == snapshot["snapshot_id"])
            .order_by(SKILLS_PAYLOAD_ITEMS_TABLE.c.short_name)
        )
        items_df = pd.read_sql(items_query, engine)
        skills_list = []
        for row in items_df.to_dict(orient="records"):
            payload_json = row.get("payload_json")
            if payload_json:
                skills_list.append(_json.loads(payload_json))
                continue

            skills_list.append(
                {
                    "备件简称": row.get("short_name") or "",
                    "适用算法": row.get("algorithm_type") or "KDE+KNN+Elbow 密度连接异常检测",
                    "当前σ参数": row.get("sigma_param") if row.get("sigma_param") is not None else 1.0,
                    "偏置权重": row.get("expert_weight") if row.get("expert_weight") is not None else 80,
                    "本组专家标注数": 0,
                    "经验对齐率": row.get("alignment_rate") if row.get("alignment_rate") is not None else "N/A",
                    "数据结构分布描述": {},
                    "成本合理区间边界": {
                        "预测值": row.get("base_price") if row.get("base_price") is not None else 0.0,
                        "合理下限": row.get("lower_bound") if row.get("lower_bound") is not None else 0.0,
                        "合理上限": row.get("upper_bound") if row.get("upper_bound") is not None else 0.0,
                    },
                    "异常统计": {},
                }
            )
        index = {}
        for sk in skills_list:
            name = sk.get("备件简称")
            if name:
                index[str(name)] = sk
        return {
            "snapshot_id": snapshot["snapshot_id"],
            "version": snapshot["version"],
            "saved_at": snapshot["saved_at"].isoformat() if snapshot["saved_at"] else None,
            "global_sigma": snapshot["global_sigma"],
            "global_weight": snapshot["global_weight"],
            "skills": skills_list,
            "index": index,
        }
    except Exception:
        return None


def has_skills_snapshot() -> bool:
    if DB_ENGINE is None:
        return False
    try:
        return _count_table_rows(SKILLS_SNAPSHOT_TABLE) > 0
    except Exception:
        return False


def _ensure_database_columns() -> None:
    engine = require_db_engine()
    ddl_statements = [
        "ALTER TABLE skills_items ADD COLUMN IF NOT EXISTS payload_json TEXT",
        "ALTER TABLE skills_items ADD COLUMN IF NOT EXISTS snapshot_ref VARCHAR(36)",
    ]
    with engine.begin() as conn:
        for statement in ddl_statements:
            conn.execute(text(statement))


def _cost_anomaly_results_create_sql_for_engine(engine: Engine) -> str:
    if engine.dialect.name == "sqlite":
        return _COST_ANOMALY_RESULTS_CREATE_SQL_SQLITE
    return _COST_ANOMALY_RESULTS_CREATE_SQL


def _to_storage_record_key(record_key: Any, result_mode: str) -> str:
    return f"{result_mode}::{record_key}"


def _from_storage_record_key(record_key: Any, result_mode: str) -> Any:
    if record_key is None:
        return None
    prefix = f"{result_mode}::"
    text_key = str(record_key)
    if text_key.startswith(prefix):
        return text_key[len(prefix):]
    return text_key


def _get_cost_anomaly_results_columns(engine: Engine) -> List[str]:
    inspector = inspect(engine)
    if not inspector.has_table(COST_ANOMALY_RESULTS_TABLE.name):
        return []
    return [column["name"] for column in inspector.get_columns(COST_ANOMALY_RESULTS_TABLE.name)]


def _cost_anomaly_results_schema_needs_reset(actual_columns: Sequence[str]) -> bool:
    return list(actual_columns) != _ANOMALY_RESULT_TABLE_COLUMNS


def _reset_cost_anomaly_results_table() -> None:
    engine = require_db_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS cost_anomaly_results"))
        conn.execute(text(_cost_anomaly_results_create_sql_for_engine(engine)))
    print("[cost_anomaly_results] 已按纯英文字段重置表结构")


def _ensure_cost_anomaly_results_table() -> None:
    global _COST_ANOMALY_RESULTS_RESET_DONE

    if _COST_ANOMALY_RESULTS_RESET_DONE:
        return

    engine = require_db_engine()
    actual_columns = _get_cost_anomaly_results_columns(engine)
    reset_requested = settings.reset_cost_anomaly_results_on_start

    if not actual_columns:
        with engine.begin() as conn:
            conn.execute(text(_cost_anomaly_results_create_sql_for_engine(engine)))
        print("[cost_anomaly_results] 已按纯英文字段创建表结构")
    elif reset_requested or _cost_anomaly_results_schema_needs_reset(actual_columns):
        print(f"[cost_anomaly_results] 检测到旧表结构，准备重建。现有列: {actual_columns}")
        _reset_cost_anomaly_results_table()

    _COST_ANOMALY_RESULTS_RESET_DONE = True


def _get_latest_skills_snapshot_id() -> Optional[str]:
    engine = require_db_engine()
    with engine.connect() as conn:
        return conn.execute(
            select(SKILLS_SNAPSHOT_TABLE.c.snapshot_id)
            .order_by(SKILLS_SNAPSHOT_TABLE.c.saved_at.desc())
            .limit(1)
        ).scalar_one_or_none()


def _latest_skills_snapshot_has_items() -> bool:
    snapshot_id = _get_latest_skills_snapshot_id()
    if not snapshot_id:
        return False

    engine = require_db_engine()
    with engine.connect() as conn:
        item_count = conn.execute(
            select(func.count())
            .select_from(SKILLS_PAYLOAD_ITEMS_TABLE)
            .where(SKILLS_PAYLOAD_ITEMS_TABLE.c.snapshot_id == snapshot_id)
        ).scalar_one()
    return int(item_count) > 0


def _import_feedback_from_legacy_csv() -> None:
    if not _LEGACY_FEEDBACK_PATH.exists() or _count_table_rows(EXPERT_FEEDBACK_TABLE) > 0:
        return

    try:
        legacy_df = pd.read_csv(_LEGACY_FEEDBACK_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        legacy_df = pd.read_csv(_LEGACY_FEEDBACK_PATH, encoding="gbk")
    except Exception:
        return

    if "record_key" not in legacy_df.columns or "label" not in legacy_df.columns:
        return

    if "labeled_at" not in legacy_df.columns:
        legacy_df["labeled_at"] = datetime.now()
    legacy_df["labeled_at"] = pd.to_datetime(legacy_df["labeled_at"], errors="coerce")
    legacy_df["labeled_at"] = legacy_df["labeled_at"].fillna(pd.Timestamp(datetime.now()))
    rows = _rows_from_dataframe(legacy_df[["record_key", "label", "labeled_at"]].drop_duplicates("record_key", keep="last"))
    _upsert_rows(
        EXPERT_FEEDBACK_TABLE,
        rows,
        conflict_columns=["record_key"],
        update_columns=["label", "labeled_at"],
    )


def _import_skills_from_legacy_json() -> None:
    if not _LEGACY_SKILLS_PATH.exists():
        return

    has_snapshot = _count_table_rows(SKILLS_SNAPSHOT_TABLE) > 0
    if has_snapshot and _latest_skills_snapshot_has_items():
        return

    if has_snapshot:
        latest_snapshot_id = _get_latest_skills_snapshot_id()
        if latest_snapshot_id:
            with require_db_engine().begin() as conn:
                conn.execute(delete(SKILLS_PAYLOAD_ITEMS_TABLE))
                conn.execute(
                    delete(SKILLS_SNAPSHOT_TABLE).where(
                        SKILLS_SNAPSHOT_TABLE.c.snapshot_id == latest_snapshot_id
                    )
                )

    try:
        with open(_LEGACY_SKILLS_PATH, encoding="utf-8") as file_obj:
            payload = _json.load(file_obj)
    except Exception:
        return

    skills = payload.get("skills")
    if not isinstance(skills, list):
        return

    save_skills(
        skills,
        sigma=float(payload.get("global_sigma", 1.0)),
        weight=int(payload.get("global_weight", 80)),
    )


def _import_core_records_from_legacy_source() -> None:
    if _count_table_rows(CORE_COST_RECORDS_TABLE) > 0:
        return

    candidate_paths = [_resolve_project_path(settings.api_data_cache_path), _TEST_DATA_PATH]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue

        try:
            if candidate.suffix.lower() == ".parquet":
                legacy_df = pd.read_parquet(candidate)
                legacy_price_col = detect_price_column(legacy_df.columns)
                if "monitor_date" not in legacy_df.columns and "价格有效于" in legacy_df.columns:
                    legacy_df["monitor_date"] = pd.to_datetime(legacy_df["价格有效于"], errors="coerce")
                persist_core_cost_records(legacy_df, price_col=legacy_price_col, mode="full")
                return

            if candidate.suffix.lower() == ".csv":
                try:
                    raw_df = pd.read_csv(candidate, encoding="utf-8")
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(candidate, encoding="gbk")
                processed_df, legacy_price_col, error_msg = process_dataframe(raw_df)
                if processed_df is not None and not error_msg:
                    persist_core_cost_records(processed_df, price_col=legacy_price_col, mode="full")
                    return
        except Exception:
            continue


def initialize_supabase_storage() -> None:
    global _DB_INIT_ERROR

    if DB_ENGINE is None:
        return

    try:
        engine = require_db_engine()
        _ensure_cost_anomaly_results_table()
        DB_METADATA.create_all(engine)
        _ensure_database_columns()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Supabase Connection Verified")
        _import_feedback_from_legacy_csv()
        _import_skills_from_legacy_json()
        _import_core_records_from_legacy_source()
        _DB_INIT_ERROR = None
    except Exception as exc:
        _DB_INIT_ERROR = exc


# ---------------------------------------------------------------------------
# 策略 B：加权自学习异常检测
# ---------------------------------------------------------------------------
_EXPERT_WEIGHT = 80  # 专家标注样本在 KDE 拟合中的复制倍数

@st.cache_data
def detect_cost_anomalies_weighted(
    df: pd.DataFrame,
    price_col: str,
    expert_labels_tuple: tuple,
    sigma_multiplier: float = 1.0,
    expert_weight_override: int = 0,
    skills_overrides_json: str = "",
) -> pd.DataFrame:
    """基于策略 B（样本加权）的异常检测。

    与 ``detect_cost_anomalies`` 共享同一套 KDE+KNN+Elbow 算法内核，
    但在 KDE 拟合阶段，将专家标注为"正常"的样本复制高倍数，
    以极高权重强迫密度峰值（中轴线 / 预测值）向专家标注点偏移，
    同时自然扩展合理区间边界，使之不再被判定为异常。

    Parameters
    ----------
    df : pd.DataFrame
        含 ``price_col`` 和 ``monitor_date`` 的全量历史数据。
    price_col : str
        价格列名。
    expert_labels_tuple : tuple
        ``((record_key, label), ...)`` — 由 ``tuple(label_manager.get_labels().items())``
        传入以便 Streamlit ``cache_data`` 可正确缓存。
    sigma_multiplier : float
        KDE 带宽缩放系数，>1 放宽边界，<1 收紧边界。默认 1.0。
    expert_weight_override : int
        覆盖默认 ``_EXPERT_WEIGHT``。0 表示使用默认值 (80)。
    skills_overrides_json : str
        JSON 字符串，格式为 ``{"备件简称": {"sigma": float, "weight": int}, ...}``。
        若提供，对匹配的备件简称使用技能书中的个性化参数。空字符串表示不使用。

    Returns
    -------
    pd.DataFrame
        与 ``detect_cost_anomalies`` 输出相同的列结构，额外增加 ``专家校准`` 和 ``判定依据`` 列。
    """
    expert_labels: Dict[str, str] = dict(expert_labels_tuple)
    ew = expert_weight_override if expert_weight_override > 0 else _EXPERT_WEIGHT

    # 解析 Skills 覆盖参数
    _skills_idx: Dict[str, dict] = {}
    if skills_overrides_json:
        try:
            _skills_idx = _json.loads(skills_overrides_json)
        except Exception:
            _skills_idx = {}

    if df.empty:
        return pd.DataFrame(
            columns=[
                "_record_key", "物料编码", "物料名称", "适用车系", "工厂",
                "备件简称", "实际成本", "价格有效于", "样本量",
                "预测值", "合理下限", "合理上限",
                "偏离数值", "偏离比例", "status", "专家校准", "判定依据",
            ]
        )

    if price_col not in df.columns:
        raise ValueError(f"找不到价格列: {price_col}")

    try:
        from sklearn.neighbors import KernelDensity, NearestNeighbors
    except Exception as e:
        raise ImportError("缺少依赖 scikit-learn，请先安装：pip install scikit-learn") from e

    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["monitor_date"]):
        data["monitor_date"] = pd.to_datetime(data["monitor_date"], errors="coerce")

    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=["物料编码", "备件简称", "monitor_date", price_col])

    if "价格有效于" in data.columns and "monitor_date" in data.columns:
        data = data.drop(columns=["价格有效于"])
    data = data.rename(columns={price_col: "实际成本", "monitor_date": "价格有效于"})
    data["样本量"] = data.groupby("备件简称")["物料编码"].transform("size")

    # 预生成 _record_key 以便匹配标注
    data["_record_key"] = data.apply(make_record_key, axis=1)

    # 筛出标注为"正常"的 key 集合
    normal_keys = {k for k, v in expert_labels.items() if v == "正常"}

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

    def _build_components(values: np.ndarray, break_gap_idx: set):
        comps = []
        s = 0
        for i in range(values.size - 1):
            if i in break_gap_idx:
                comps.append((s, i))
                s = i + 1
        comps.append((s, values.size - 1))
        return comps

    results = []

    for short_name, group in data.groupby("备件简称", sort=False):
        g = group.copy()
        n = len(g)

        # ── Skills 闭环：查找本组个性化参数 ──
        _sn = str(short_name)
        _sk = _skills_idx.get(_sn)
        if _sk:
            grp_sigma = float(_sk.get("sigma", sigma_multiplier))
            grp_ew = int(_sk.get("weight", ew))
            grp_source = "技能书校验"
        else:
            grp_sigma = sigma_multiplier
            grp_ew = ew
            grp_source = "默认算法"

        # 识别本组内被专家标注为"正常"的成本值
        expert_mask = g["_record_key"].isin(normal_keys)
        expert_vals = g.loc[expert_mask, "实际成本"].to_numpy(dtype=float)

        if n < 10:
            q_low = float(g["实际成本"].quantile(0.05))
            q_high = float(g["实际成本"].quantile(0.95))
            baseline = float(g["实际成本"].median())
            # 加权：如果有专家标注值，将 baseline 向它们偏移
            if expert_vals.size > 0:
                expert_center = float(np.median(expert_vals))
                baseline = (baseline + expert_center * grp_ew) / (1 + grp_ew)
                q_low = min(q_low, float(np.min(expert_vals)))
                q_high = max(q_high, float(np.max(expert_vals)))
            g["预测值"] = baseline
            g["合理下限"] = max(0.0, q_low)
            g["合理上限"] = q_high
            g["偏离数值"] = g["实际成本"] - g["预测值"]
            g["偏离比例"] = g["偏离数值"] / g["预测值"].replace(0, pd.NA)
            g["status"] = "正常（小样本数据）"
            g.loc[g["实际成本"] > g["合理上限"], "status"] = "异常偏高（小样本数据）"
            g.loc[g["实际成本"] < g["合理下限"], "status"] = "异常偏低（小样本数据）"
            g["专家校准"] = ""
            g.loc[expert_mask, "专家校准"] = "✅"
            g["判定依据"] = grp_source
            results.append(g)
            continue

        vals = g["实际成本"].to_numpy(dtype=float)

        # ★ 策略 B 核心：将专家标注"正常"的值复制 grp_ew 倍
        if expert_vals.size > 0:
            weighted_vals = np.concatenate([vals] + [expert_vals] * grp_ew)
        else:
            weighted_vals = vals

        uniq_vals, uniq_counts = np.unique(vals, return_counts=True)
        if uniq_vals.size == 1:
            single_val = float(uniq_vals[0])
            g["预测值"] = single_val
            g["合理下限"] = max(0.0, single_val)
            g["合理上限"] = single_val
            g["偏离数值"] = 0.0
            g["偏离比例"] = 0.0
            g["status"] = "正常"
            g["专家校准"] = ""
            g.loc[expert_mask, "专家校准"] = "✅"
            g["判定依据"] = grp_source
            results.append(g)
            continue

        # Step 1: KDE 寻峰 —— 使用加权后的数据拟合
        std_val = float(np.std(weighted_vals))
        iqr_val = float(np.subtract(*np.percentile(weighted_vals, [75, 25])))
        spread = min(x for x in [std_val, iqr_val] if x > 0) if (std_val > 0 and iqr_val > 0) else max(std_val, iqr_val)
        if spread <= 0:
            spread = float(np.mean(np.abs(np.diff(np.sort(np.unique(weighted_vals)))))) if np.unique(weighted_vals).size > 1 else 1.0
        bandwidth = (spread / np.sqrt(max(1, len(weighted_vals)))) * grp_sigma
        if bandwidth <= 0:
            bandwidth = 1.0

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(weighted_vals.reshape(-1, 1))
        grid_size = int(min(600, max(200, uniq_vals.size * 4)))
        grid = np.linspace(float(np.min(vals)), float(np.max(vals)), grid_size)
        density = np.exp(kde.score_samples(grid.reshape(-1, 1)))
        peak_price = float(grid[int(np.argmax(density))])

        # Step 2: KNN 间距仍基于原始 uniq_vals（不膨胀）
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

        break_idx = set(np.where(norm_gaps > base_threshold)[0].tolist())
        components = _build_components(uniq_vals, break_idx)
        comp_counts = []
        comp_density = []
        for s, e in components:
            c = int(np.sum(uniq_counts[s : e + 1]))
            span = float(max(uniq_vals[e] - uniq_vals[s], np.finfo(float).eps))
            comp_counts.append(c)
            comp_density.append(c / span)

        peak_comp_idx = next(i for i, (s, e) in enumerate(components) if s <= peak_idx <= e)
        merged_left, merged_right = components[peak_comp_idx]

        global_dispersion = float(np.std(vals))
        if global_dispersion <= 0:
            global_dispersion = float(np.mean(np.abs(gaps))) if gaps.size else 1.0
        gap_ratio = gaps / global_dispersion if global_dispersion > 0 else gaps
        gap_ratio_threshold = float(np.median(gap_ratio)) if gap_ratio.size else 0.0
        density_threshold = float(np.median(comp_density)) if comp_density else 0.0

        current_idx = peak_comp_idx
        while current_idx - 1 >= 0:
            prev_s, prev_e = components[current_idx - 1]
            curr_s, curr_e = components[current_idx]
            boundary_gap_idx = prev_e
            cond_gap = boundary_gap_idx < gap_ratio.size and gap_ratio[boundary_gap_idx] <= gap_ratio_threshold
            cond_density = comp_density[current_idx - 1] >= density_threshold and comp_density[current_idx] >= density_threshold
            if cond_gap and cond_density:
                merged_left = prev_s
                current_idx -= 1
            else:
                break

        current_idx = peak_comp_idx
        while current_idx + 1 < len(components):
            curr_s, curr_e = components[current_idx]
            next_s, next_e = components[current_idx + 1]
            boundary_gap_idx = curr_e
            cond_gap = boundary_gap_idx < gap_ratio.size and gap_ratio[boundary_gap_idx] <= gap_ratio_threshold
            cond_density = comp_density[current_idx + 1] >= density_threshold and comp_density[current_idx] >= density_threshold
            if cond_gap and cond_density:
                merged_right = next_e
                current_idx += 1
            else:
                break

        component_lower_bound = float(uniq_vals[merged_left])
        component_upper_bound = float(uniq_vals[merged_right])
        cluster_samples = vals[
            (vals >= component_lower_bound) & (vals <= component_upper_bound)
        ]

        lower_bound = max(0.0, component_lower_bound)
        upper_bound = component_upper_bound

        # ★ 策略 B 增强：将专家标注"正常"的值强制纳入合理区间
        if expert_vals.size > 0:
            lower_bound = min(lower_bound, float(np.min(expert_vals)))
            upper_bound = max(upper_bound, float(np.max(expert_vals)))

        peak_price = _reanchor_cluster_price(
            cluster_samples,
            lower_bound,
            upper_bound,
            peak_price,
        )

        comp_index_arr = np.zeros(uniq_vals.size, dtype=int)
        for comp_id, (s, e) in enumerate(components):
            comp_index_arr[s : e + 1] = comp_id
        value_to_comp = {float(v): int(comp_index_arr[i]) for i, v in enumerate(uniq_vals)}
        comp_count_series = pd.Series(comp_counts)
        small_comp_threshold = float(comp_count_series.median()) if not comp_count_series.empty else 0.0

        peak_price = max(lower_bound, min(peak_price, upper_bound))

        g["预测值"] = peak_price
        g["合理下限"] = lower_bound
        g["合理上限"] = upper_bound
        g["偏离数值"] = g["实际成本"] - g["预测值"]
        g["偏离比例"] = g["偏离数值"] / g["预测值"].replace(0, pd.NA)

        g["status"] = "正常"
        g.loc[g["实际成本"] > g["合理上限"], "status"] = "异常偏高"
        g.loc[g["实际成本"] < g["合理下限"], "status"] = "异常偏低"

        def _is_isolated_low(v):
            comp_id = value_to_comp.get(float(v), -1)
            if comp_id < 0 or comp_id >= len(comp_counts):
                return False
            return (comp_counts[comp_id] <= small_comp_threshold) and (v < lower_bound)

        isolated_mask = g["实际成本"].apply(_is_isolated_low)
        g.loc[isolated_mask, "status"] = "严重异常偏低"

        # 专家标注为"正常"的记录强制覆盖为"正常"
        g.loc[expert_mask, "status"] = "正常"
        g["专家校准"] = ""
        g.loc[expert_mask, "专家校准"] = "✅"
        g["判定依据"] = grp_source

        results.append(g)

    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if result_df.empty:
        return result_df

    cols = [
        "_record_key",
        "物料编码",
        "物料名称" if "物料名称" in result_df.columns else None,
        "适用车系" if "适用车系" in result_df.columns else None,
        "工厂" if "工厂" in result_df.columns else None,
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
        "专家校准",
        "判定依据",
    ]
    cols = [c for c in cols if c is not None]
    result_df = result_df[cols].sort_values(["备件简称", "物料编码"]).reset_index(drop=True)
    save_cost_anomaly_results(result_df, result_mode="weighted")
    return result_df


initialize_supabase_storage()
