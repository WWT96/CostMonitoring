import glob
import io
import json as _json
import os
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
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
    Index,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

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
    Column("remark", Text),
    Column("labeled_at", DateTime, nullable=False),
)

EXPERT_KNOWLEDGE_BASE_TABLE = Table(
    "expert_knowledge_base",
    DB_METADATA,
    Column("rule_id", String(64), primary_key=True),
    Column("short_name", String(128), nullable=False),
    Column("supplier_code", String(64)),
    Column("vehicle_series", String(255)),
    Column("rule_content", Text, nullable=False),
    Column("confidence_score", Float),
    Column("updated_at", DateTime, nullable=False),
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

Index(
    "ux_core_cost_records_business_key",
    CORE_COST_RECORDS_TABLE.c.material_code,
    CORE_COST_RECORDS_TABLE.c.factory,
    CORE_COST_RECORDS_TABLE.c.monitor_date,
    unique=True,
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

SKILLS_SNAPSHOTS_TABLE = Table(
    "skills_snapshots",
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
    Column("snapshot_id", String(36), ForeignKey("skills_snapshots.snapshot_id", ondelete="CASCADE"), nullable=False),
    Column("short_name", String(128), nullable=False),
    Column("algorithm_type", String(255)),
    Column("sigma_param", Float),
    Column("expert_weight", Integer),
    Column("alignment_rate", Float),
    Column("lower_bound", Float),
    Column("upper_bound", Float),
    Column("base_price", Float),
    Column("skill_payload_json", Text),
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

_CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS = ["material_code", "factory", "monitor_date"]
_CORE_COST_RECORDS_UPSERT_UPDATE_COLUMNS = [
    column_name
    for column_name in [*_CORE_RECORD_EXPORT_COLUMNS, "created_at"]
    if column_name not in _CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS
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
_SKILLS_STORAGE_RESET_DONE = False
_LEGACY_SKILLS_SNAPSHOT_TABLE_NAME = "skills_snapshot"
_LEGACY_SKILLS_ITEMS_PAYLOAD_TABLE_NAME = "skills_items_payload"


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


def _build_upsert_statement(
    conn,
    table: Table,
    rows: Sequence[Dict[str, Any]],
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
):
    insert_factory = sqlite_insert if conn.dialect.name == "sqlite" else pg_insert
    stmt = insert_factory(table).values(list(rows))
    update_map = {column: stmt.excluded[column] for column in update_columns}
    return stmt.on_conflict_do_update(
        index_elements=[table.c[column] for column in conflict_columns],
        set_=update_map,
    )


def _upsert_rows(
    table: Table,
    rows: Sequence[Dict[str, Any]],
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
    session: Optional[Session] = None,
) -> None:
    if not rows:
        return

    if session is None:
        with Session(require_db_engine()) as managed_session:
            with managed_session.begin():
                _upsert_rows(
                    table,
                    rows,
                    conflict_columns=conflict_columns,
                    update_columns=update_columns,
                    session=managed_session,
                )
        return

    conn = session.connection()
    for batch in _chunk_rows(rows):
        session.execute(
            _build_upsert_statement(
                conn,
                table,
                batch,
                conflict_columns=conflict_columns,
                update_columns=update_columns,
            )
        )


def _insert_rows(table: Table, rows: Sequence[Dict[str, Any]], session: Optional[Session] = None) -> None:
    if not rows:
        return

    if session is None:
        with Session(require_db_engine()) as managed_session:
            with managed_session.begin():
                _insert_rows(table, rows, session=managed_session)
        return

    for batch in _chunk_rows(rows):
        session.execute(table.insert(), list(batch))


def _count_table_rows(table: Table) -> int:
    engine = require_db_engine()
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def _core_cost_records_has_business_key_index(engine: Engine) -> bool:
    inspector = inspect(engine)
    expected_columns = set(_CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS)

    for index_meta in inspector.get_indexes(CORE_COST_RECORDS_TABLE.name):
        column_names = set(index_meta.get("column_names") or [])
        if index_meta.get("unique") and column_names == expected_columns:
            return True

    for constraint_meta in inspector.get_unique_constraints(CORE_COST_RECORDS_TABLE.name):
        column_names = set(constraint_meta.get("column_names") or [])
        if column_names == expected_columns:
            return True

    return False


def _dedupe_core_cost_records_table(session: Session) -> None:
    rows_df = pd.read_sql(
        select(
            CORE_COST_RECORDS_TABLE.c.cost_record_id,
            CORE_COST_RECORDS_TABLE.c.material_code,
            CORE_COST_RECORDS_TABLE.c.factory,
            CORE_COST_RECORDS_TABLE.c.monitor_date,
            CORE_COST_RECORDS_TABLE.c.created_at,
        ).order_by(
            CORE_COST_RECORDS_TABLE.c.created_at.desc(),
            CORE_COST_RECORDS_TABLE.c.cost_record_id.desc(),
        ),
        session.connection(),
    )
    if rows_df.empty:
        return

    duplicate_ids = rows_df[
        rows_df.duplicated(subset=_CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS, keep="first")
    ]["cost_record_id"].tolist()
    if duplicate_ids:
        session.execute(
            delete(CORE_COST_RECORDS_TABLE).where(
                CORE_COST_RECORDS_TABLE.c.cost_record_id.in_(duplicate_ids)
            )
        )


def _ensure_core_cost_records_business_key_index() -> None:
    engine = require_db_engine()
    CORE_COST_RECORDS_TABLE.create(engine, checkfirst=True)

    if _core_cost_records_has_business_key_index(engine):
        return

    with Session(engine) as session:
        with session.begin():
            _dedupe_core_cost_records_table(session)
            session.execute(
                text(
                    'CREATE UNIQUE INDEX IF NOT EXISTS "ux_core_cost_records_business_key" '
                    'ON "core_cost_records" (material_code, factory, monitor_date)'
                )
            )


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
        subset=_CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS,
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


def split_record_key(record_key: Any) -> Dict[str, Any]:
    text_key = str(record_key or "").strip()
    parts = text_key.split("_", 3)
    while len(parts) < 4:
        parts.append("")

    material_code, factory, monitor_date_text, price_text = [part.strip() for part in parts[:4]]
    monitor_date = pd.to_datetime(monitor_date_text, errors="coerce")
    price_value = pd.to_numeric(pd.Series([price_text]), errors="coerce").iloc[0]

    return {
        "record_key": text_key,
        "物料编码": material_code,
        "工厂": factory,
        "价格有效期于": monitor_date.strftime("%Y-%m-%d") if pd.notna(monitor_date) else monitor_date_text,
        "价格": float(price_value) if pd.notna(price_value) else np.nan,
        "_join_date_key": monitor_date.strftime("%Y-%m-%d") if pd.notna(monitor_date) else monitor_date_text,
        "_join_price_key": f"{float(price_value):.4f}" if pd.notna(price_value) else price_text,
    }


class LabelManager:
    """管理用户对异常检测结果的手动标注，持久化到 expert_feedback 表。"""

    _COLUMNS = ["record_key", "label", "remark", "labeled_at"]

    def __init__(self, table_name: str = "expert_feedback"):
        self._table_name = table_name

    # --- 读取 ---

    def get_labels(self) -> Dict[str, Dict[str, str]]:
        """返回 ``{record_key: {label, remark}}`` 字典。数据库未就绪时返回空字典。"""
        if DB_ENGINE is None:
            return {}
        try:
            query = select(
                EXPERT_FEEDBACK_TABLE.c.record_key,
                EXPERT_FEEDBACK_TABLE.c.label,
                EXPERT_FEEDBACK_TABLE.c.remark,
            )
            df = pd.read_sql(query, require_db_engine())
            if df.empty:
                return {}
            df["record_key"] = df["record_key"].astype(str)
            df["label"] = df["label"].astype(str)
            df["remark"] = df["remark"].fillna("").astype(str)
            return {
                row["record_key"]: {
                    "label": row["label"],
                    "remark": row["remark"],
                }
                for row in df.to_dict(orient="records")
            }
        except Exception:
            return {}

    def get_label_statuses(self) -> Dict[str, str]:
        """返回 ``{record_key: label}`` 字典，供算法和筛选逻辑使用。"""
        return {
            record_key: payload.get("label", "")
            for record_key, payload in self.get_labels().items()
        }

    def get_label_remarks(self) -> Dict[str, str]:
        """返回 ``{record_key: remark}`` 字典，供界面展示与导出使用。"""
        return {
            record_key: payload.get("remark", "")
            for record_key, payload in self.get_labels().items()
        }

    def get_label_records(self) -> pd.DataFrame:
        """返回包含 record_key / label / remark / labeled_at 的明细表。"""
        if DB_ENGINE is None:
            return pd.DataFrame(columns=["record_key", "label", "remark", "labeled_at"])
        try:
            query = select(
                EXPERT_FEEDBACK_TABLE.c.record_key,
                EXPERT_FEEDBACK_TABLE.c.label,
                EXPERT_FEEDBACK_TABLE.c.remark,
                EXPERT_FEEDBACK_TABLE.c.labeled_at,
            )
            df = pd.read_sql(query, require_db_engine())
            if df.empty:
                return pd.DataFrame(columns=["record_key", "label", "remark", "labeled_at"])
            df["record_key"] = df["record_key"].astype(str)
            df["label"] = df["label"].astype(str)
            df["remark"] = df["remark"].fillna("").astype(str)
            df["labeled_at"] = pd.to_datetime(df["labeled_at"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=["record_key", "label", "remark", "labeled_at"])

    def count(self) -> int:
        """已标注记录总数。"""
        if DB_ENGINE is None:
            return 0
        try:
            return _count_table_rows(EXPERT_FEEDBACK_TABLE)
        except Exception:
            return 0

    # --- 写入 ---

    def save_label(self, key: str, status: str, remark: str = "") -> None:
        """保存或更新单条标注记录。"""
        now = datetime.now()
        _upsert_rows(
            EXPERT_FEEDBACK_TABLE,
            [{"record_key": key, "label": status, "remark": str(remark or "").strip(), "labeled_at": now}],
            conflict_columns=["record_key"],
            update_columns=["label", "remark", "labeled_at"],
        )

    def save_labels_batch(self, updates: Dict[str, Any]) -> None:
        """批量保存标注记录。"""
        if not updates:
            return
        now = datetime.now()
        rows = []
        for record_key, payload in updates.items():
            if isinstance(payload, dict):
                label = payload.get("label")
                remark = payload.get("remark", "")
            else:
                label = payload
                remark = ""
            if label is None or str(label).strip() == "":
                continue
            rows.append(
                {
                    "record_key": record_key,
                    "label": str(label).strip(),
                    "remark": str(remark or "").strip(),
                    "labeled_at": now,
                }
            )
        _upsert_rows(
            EXPERT_FEEDBACK_TABLE,
            rows,
            conflict_columns=["record_key"],
            update_columns=["label", "remark", "labeled_at"],
        )

    def replace_all(self, final_labels_df: pd.DataFrame) -> None:
        """以当前最终标注集为准，同步数据库中的 expert_feedback 记录。"""
        if isinstance(final_labels_df, dict):
            rows = []
            for key, payload in final_labels_df.items():
                if isinstance(payload, dict):
                    rows.append(
                        {
                            "record_key": key,
                            "label": payload.get("label"),
                            "remark": payload.get("remark", ""),
                        }
                    )
                else:
                    rows.append({"record_key": key, "label": payload, "remark": ""})
            final_df = pd.DataFrame(rows)
        elif final_labels_df is None:
            final_df = pd.DataFrame(columns=["record_key", "label", "remark"])
        else:
            final_df = final_labels_df.copy()

        final_df = final_df.rename(
            columns={
                "_record_key": "record_key",
                "记录主键": "record_key",
                "当前标注": "label",
                "标注备注": "remark",
                "专家备注": "remark",
                "备注": "remark",
                "status": "label",
            }
        )
        if "record_key" not in final_df.columns:
            final_df["record_key"] = pd.Series(dtype="string")
        if "label" not in final_df.columns:
            final_df["label"] = pd.Series(dtype="string")
        if "remark" not in final_df.columns:
            final_df["remark"] = pd.Series(dtype="string")

        final_df = final_df[["record_key", "label", "remark"]].copy()
        final_df["record_key"] = final_df["record_key"].astype("string").str.strip()
        final_df["label"] = final_df["label"].astype("string").str.strip()
        final_df["remark"] = final_df["remark"].fillna("").astype("string").str.strip()
        final_df = final_df.replace({"record_key": {"": pd.NA}, "label": {"": pd.NA}})
        final_df = final_df.dropna(subset=["record_key", "label"])
        final_df = final_df.drop_duplicates(subset=["record_key"], keep="last")
        final_df["labeled_at"] = datetime.now()

        engine = require_db_engine()
        with engine.begin() as conn:
            existing_df = pd.read_sql(
                select(
                    EXPERT_FEEDBACK_TABLE.c.record_key,
                    EXPERT_FEEDBACK_TABLE.c.label,
                    EXPERT_FEEDBACK_TABLE.c.remark,
                ),
                conn,
            )
            existing_keys = set(existing_df["record_key"].astype(str)) if not existing_df.empty else set()
            final_keys = set(final_df["record_key"].astype(str))

            keys_to_delete = sorted(existing_keys - final_keys)
            if keys_to_delete:
                conn.execute(
                    delete(EXPERT_FEEDBACK_TABLE).where(EXPERT_FEEDBACK_TABLE.c.record_key.in_(keys_to_delete))
                )

            rows = _rows_from_dataframe(final_df[self._COLUMNS])
            for batch in _chunk_rows(rows):
                conn.execute(
                    _build_upsert_statement(
                        conn,
                        EXPERT_FEEDBACK_TABLE,
                        batch,
                        conflict_columns=["record_key"],
                        update_columns=["label", "remark", "labeled_at"],
                    )
                )

    def _flush(self, final_labels_df: pd.DataFrame) -> None:
        """兼容旧调用方；请改用公开的 replace_all。"""
        self.replace_all(final_labels_df)

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
    """单源读取：每次都从 expert_feedback 表获取最新专家标注状态，不使用缓存。"""
    return label_manager.get_label_statuses()


def get_latest_feedback_details() -> Dict[str, Dict[str, str]]:
    """单源读取：返回 expert_feedback 中的标注状态与备注。"""
    return label_manager.get_labels()


def load_expert_knowledge_base() -> pd.DataFrame:
    """读取专家经验知识库。"""
    _ensure_expert_knowledge_base_columns()
    engine = require_db_engine()
    query = select(EXPERT_KNOWLEDGE_BASE_TABLE).order_by(EXPERT_KNOWLEDGE_BASE_TABLE.c.updated_at.desc())
    df = pd.read_sql(query, engine)
    if df.empty:
        return df
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    for column_name in ["rule_id", "short_name", "supplier_code", "vehicle_series", "rule_content"]:
        if column_name in df.columns:
            df[column_name] = df[column_name].fillna("").astype(str)
    return df


def get_expert_knowledge_last_updated_at() -> Optional[pd.Timestamp]:
    """返回知识库最后更新时间。"""
    _ensure_expert_knowledge_base_columns()
    engine = require_db_engine()
    with engine.connect() as conn:
        latest = conn.execute(select(func.max(EXPERT_KNOWLEDGE_BASE_TABLE.c.updated_at))).scalar_one_or_none()
    if latest is None:
        return None
    return pd.Timestamp(latest)


def save_expert_knowledge_rules(rules: Sequence[Dict[str, Any]]) -> int:
    """批量写入专家经验知识库。"""
    if not rules:
        return 0
    _ensure_expert_knowledge_base_columns()
    rows = _rows_from_dataframe(pd.DataFrame(rules))
    _upsert_rows(
        EXPERT_KNOWLEDGE_BASE_TABLE,
        rows,
        conflict_columns=["rule_id"],
        update_columns=[
            "short_name",
            "supplier_code",
            "vehicle_series",
            "rule_content",
            "confidence_score",
            "updated_at",
        ],
    )
    return len(rows)


def delete_expert_knowledge_rules(rule_ids: Sequence[str]) -> int:
    """删除指定 rule_id 的知识库规则。"""
    keys = [str(rule_id).strip() for rule_id in rule_ids if str(rule_id).strip()]
    if not keys:
        return 0
    _ensure_expert_knowledge_base_columns()
    with require_db_engine().begin() as conn:
        result = conn.execute(
            delete(EXPERT_KNOWLEDGE_BASE_TABLE).where(EXPERT_KNOWLEDGE_BASE_TABLE.c.rule_id.in_(keys))
        )
    return int(result.rowcount or 0)


def get_expert_knowledge_refresh_token() -> float:
    """返回知识库刷新令牌，用于缓存失效。"""
    latest = get_expert_knowledge_last_updated_at()
    if latest is None:
        return 0.0
    return float(pd.Timestamp(latest).timestamp())


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
    core_df, _, _ = load_core_cost_records()
    if core_df is None or core_df.empty:
        return {}
    lookup_df = core_df.copy()
    lookup_df["价格有效于"] = pd.to_datetime(lookup_df.get("monitor_date"), errors="coerce")
    lookup_df["实际成本"] = pd.to_numeric(lookup_df.get("成本"), errors="coerce")
    lookup_df["_record_key"] = lookup_df.apply(make_record_key, axis=1)
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
        "supplier_code": str(rule_row.get("supplier_code") or ""),
        "vehicle_series": str(rule_row.get("vehicle_series") or ""),
    }


def _format_inferred_reason(context: Dict[str, Any], match_detail: Dict[str, Any]) -> str:
    deviation_ratio = context.get("deviation_ratio")
    if deviation_ratio is None:
        deviation_text = ""
    elif deviation_ratio >= 0:
        deviation_text = f"当前价格较基准高 {abs(float(deviation_ratio)):.1%}，"
    else:
        deviation_text = f"当前价格较基准低 {abs(float(deviation_ratio)):.1%}，"

    if match_detail["match_type"] == "供应商精准匹配":
        prefix = "命中同供应商历史规律"
    elif match_detail["match_type"] == "车系精准匹配":
        prefix = "命中同车系历史规律"
    elif match_detail["match_type"] == "同供应商同类继承":
        prefix = f"命中同供应商同类备件经验（{match_detail['rule_short_name']} → {context.get('short_name', '')}）"
    else:
        prefix = f"命中同车系同类备件经验（{match_detail['rule_short_name']} → {context.get('short_name', '')}）"

    guidance = "建议优先核查该规律是否仍然成立。"
    if "同类" in match_detail["match_type"]:
        guidance = "建议优先核查是否延续了同类备件中的材料、工艺或模摊原因。"

    return f"[系统预测] {deviation_text}{prefix}：{match_detail['rule_content']} {guidance}".strip()


def infer_anomaly_reason(
    anomaly_record: Any,
    knowledge_df: Optional[pd.DataFrame] = None,
    core_context_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """根据专家经验知识库推理当前异常记录的可能原因。"""
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
        knowledge_df = load_expert_knowledge_base()
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
    """为异常结果附加 AI 辅助分析与推理元数据。"""
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

    knowledge_df = load_expert_knowledge_base()
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


class CostMonitoringService:
    """前端与接口层使用的统一服务边界。"""

    def initialize_storage(self) -> None:
        initialize_supabase_storage()

    def sync_core_cost_records(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None,
        mode: str = "incremental",
    ) -> int:
        return persist_core_cost_records(df, price_col=price_col, mode=mode)

    def load_core_cost_records(self) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        return load_core_cost_records()

    def get_core_cost_records_status(self) -> Dict[str, Any]:
        return get_core_cost_records_status()

    def get_feedback_details(self) -> Dict[str, Dict[str, str]]:
        return label_manager.get_labels()

    def get_feedback_statuses(self) -> Dict[str, str]:
        return label_manager.get_label_statuses()

    def replace_feedback(self, final_labels_df: pd.DataFrame) -> None:
        label_manager.replace_all(final_labels_df)

    def delete_feedback(self, keys_to_remove) -> int:
        return label_manager.delete_labels(keys_to_remove)

    def clear_feedback(self) -> None:
        label_manager.clear_all()

    def get_feedback_row_count(self) -> int:
        return label_manager.file_row_count()

    def load_skills_snapshot(self) -> Optional[Dict]:
        return load_skills()

    def save_skills_snapshot(self, skills: list, sigma: float = 1.0, weight: int = 80) -> str:
        return save_skills(skills, sigma=sigma, weight=weight)

    def has_skills_snapshot(self) -> bool:
        return has_skills_snapshot()


service = CostMonitoringService()


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
    """将 Skills 列表持久化到 skills_snapshots / skills_items 表。"""
    _ensure_skills_storage_tables()
    snapshot_id = str(uuid4())
    saved_at = datetime.now()
    snapshot_rows = [
        {
            "snapshot_id": snapshot_id,
            "version": "1.0",
            "saved_at": saved_at,
            "global_sigma": round(float(sigma), 4),
            "global_weight": int(weight),
        }
    ]

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
                "skill_payload_json": _json.dumps(skill, ensure_ascii=False, default=str),
            }
        )

    with Session(require_db_engine()) as session:
        with session.begin():
            _insert_rows(SKILLS_SNAPSHOTS_TABLE, snapshot_rows, session=session)
            _insert_rows(SKILLS_ITEMS_TABLE, rows, session=session)
    return "skills_snapshots / skills_items"


def load_skills() -> Optional[Dict]:
    """加载最近一次 Skills 快照。"""
    if DB_ENGINE is None:
        return None
    _ensure_skills_storage_tables()
    try:
        engine = require_db_engine()
        with engine.connect() as conn:
            snapshot = conn.execute(
                select(SKILLS_SNAPSHOTS_TABLE)
                .order_by(SKILLS_SNAPSHOTS_TABLE.c.saved_at.desc())
                .limit(1)
            ).mappings().first()
        if snapshot is None:
            return None

        items_query = (
            select(
                SKILLS_ITEMS_TABLE.c.short_name,
                SKILLS_ITEMS_TABLE.c.algorithm_type,
                SKILLS_ITEMS_TABLE.c.sigma_param,
                SKILLS_ITEMS_TABLE.c.expert_weight,
                SKILLS_ITEMS_TABLE.c.alignment_rate,
                SKILLS_ITEMS_TABLE.c.lower_bound,
                SKILLS_ITEMS_TABLE.c.upper_bound,
                SKILLS_ITEMS_TABLE.c.base_price,
                SKILLS_ITEMS_TABLE.c.skill_payload_json,
            )
            .where(SKILLS_ITEMS_TABLE.c.snapshot_id == snapshot["snapshot_id"])
            .order_by(SKILLS_ITEMS_TABLE.c.short_name)
        )
        items_df = pd.read_sql(items_query, engine)
        skills_list = []
        for row in items_df.to_dict(orient="records"):
            payload = {}
            payload_text = row.get("skill_payload_json")
            if isinstance(payload_text, str) and payload_text.strip():
                try:
                    parsed = _json.loads(payload_text)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}

            skill = dict(payload)
            skill["备件简称"] = skill.get("备件简称") or row.get("short_name") or ""
            skill["适用算法"] = skill.get("适用算法") or row.get("algorithm_type") or "KDE+KNN+Elbow 密度连接异常检测"
            skill["当前σ参数"] = row.get("sigma_param") if row.get("sigma_param") is not None else skill.get("当前σ参数", 1.0)
            skill["偏置权重"] = row.get("expert_weight") if row.get("expert_weight") is not None else skill.get("偏置权重", 80)
            skill["本组专家标注数"] = skill.get("本组专家标注数", 0)
            skill["经验对齐率"] = row.get("alignment_rate") if row.get("alignment_rate") is not None else skill.get("经验对齐率", "N/A")
            if not isinstance(skill.get("数据结构分布描述"), dict):
                skill["数据结构分布描述"] = {}
            if not isinstance(skill.get("异常统计"), dict):
                skill["异常统计"] = {}

            bounds = skill.get("成本合理区间边界")
            if not isinstance(bounds, dict):
                bounds = {}
            skill["成本合理区间边界"] = {
                "预测值": row.get("base_price") if row.get("base_price") is not None else bounds.get("预测值", 0.0),
                "合理下限": row.get("lower_bound") if row.get("lower_bound") is not None else bounds.get("合理下限", 0.0),
                "合理上限": row.get("upper_bound") if row.get("upper_bound") is not None else bounds.get("合理上限", 0.0),
            }

            semantic_report = skill.get("语义校准报告")
            if not isinstance(semantic_report, dict):
                semantic_report = {}
            modes = semantic_report.get("主要匹配方式", [])
            if not isinstance(modes, list):
                modes = [modes] if modes else []
            refs = semantic_report.get("参考文本规律", [])
            if not isinstance(refs, list):
                refs = [refs] if refs else []
            skill["语义校准报告"] = {
                "引用规律数": int(semantic_report.get("引用规律数", len([value for value in refs if str(value).strip()])) or 0),
                "主要匹配方式": [str(value).strip() for value in modes if str(value).strip()],
                "参考文本规律": [str(value).strip() for value in refs if str(value).strip()],
            }

            skills_list.append(skill)
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
    _ensure_skills_storage_tables()
    try:
        return _count_table_rows(SKILLS_SNAPSHOTS_TABLE) > 0
    except Exception:
        return False


def _ensure_database_columns() -> None:
    _ensure_core_cost_records_business_key_index()
    _ensure_expert_feedback_columns()
    _ensure_expert_knowledge_base_columns()
    _ensure_skills_storage_tables()


def _ensure_expert_feedback_columns() -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, EXPERT_FEEDBACK_TABLE.name)
    if not actual_columns:
        EXPERT_FEEDBACK_TABLE.create(engine, checkfirst=True)
        return
    if "remark" not in actual_columns:
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE expert_feedback ADD COLUMN IF NOT EXISTS remark TEXT"))
            else:
                conn.execute(text("ALTER TABLE expert_feedback ADD COLUMN remark TEXT"))


def _ensure_expert_knowledge_base_columns() -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, EXPERT_KNOWLEDGE_BASE_TABLE.name)
    if not actual_columns:
        EXPERT_KNOWLEDGE_BASE_TABLE.create(engine, checkfirst=True)
        return

    expected_columns = {
        "rule_id": "VARCHAR(64)",
        "short_name": "VARCHAR(128)",
        "supplier_code": "VARCHAR(64)",
        "vehicle_series": "VARCHAR(255)",
        "rule_content": "TEXT",
        "confidence_score": "FLOAT",
        "updated_at": "TIMESTAMP",
    }
    missing_columns = [column_name for column_name in expected_columns if column_name not in actual_columns]
    if not missing_columns:
        return

    for column_name in missing_columns:
        ddl = expected_columns[column_name]
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(
                    text(
                        f"ALTER TABLE expert_knowledge_base ADD COLUMN IF NOT EXISTS {column_name} {ddl}"
                    )
                )
            else:
                conn.execute(text(f"ALTER TABLE expert_knowledge_base ADD COLUMN {column_name} {ddl}"))


def _ensure_skills_items_columns() -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, SKILLS_ITEMS_TABLE.name)
    if not actual_columns:
        SKILLS_ITEMS_TABLE.create(engine, checkfirst=True)
        return

    expected_columns = {
        "skill_payload_json": "TEXT",
    }
    missing_columns = [column_name for column_name in expected_columns if column_name not in actual_columns]
    if not missing_columns:
        return

    for column_name in missing_columns:
        ddl = expected_columns[column_name]
        with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                conn.execute(
                    text(
                        f"ALTER TABLE {SKILLS_ITEMS_TABLE.name} ADD COLUMN IF NOT EXISTS {column_name} {ddl}"
                    )
                )
            else:
                conn.execute(text(f"ALTER TABLE {SKILLS_ITEMS_TABLE.name} ADD COLUMN {column_name} {ddl}"))


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
    _ensure_skills_storage_tables()
    engine = require_db_engine()
    with engine.connect() as conn:
        return conn.execute(
            select(SKILLS_SNAPSHOTS_TABLE.c.snapshot_id)
            .order_by(SKILLS_SNAPSHOTS_TABLE.c.saved_at.desc())
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
            .select_from(SKILLS_ITEMS_TABLE)
            .where(SKILLS_ITEMS_TABLE.c.snapshot_id == snapshot_id)
        ).scalar_one()
    return int(item_count) > 0


def _get_table_columns(engine: Engine, table_name: str) -> List[str]:
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return []
    return [column["name"] for column in inspector.get_columns(table_name)]


def _drop_table_by_name(engine: Engine, table_name: str, *, cascade: bool = False) -> None:
    suffix = " CASCADE" if cascade and engine.dialect.name == "postgresql" else ""
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"{suffix}'))


def _reset_table(table: Table, *, recreate: bool = True, cascade: bool = False) -> None:
    engine = require_db_engine()
    if cascade and engine.dialect.name == "postgresql":
        _drop_table_by_name(engine, table.name, cascade=True)
    else:
        table.drop(engine, checkfirst=True)
    if recreate:
        table.create(engine, checkfirst=True)


def _reset_skills_storage_tables(engine: Engine) -> None:
    _reset_table(SKILLS_ITEMS_TABLE, recreate=False)
    _reset_table(SKILLS_SNAPSHOTS_TABLE, recreate=False, cascade=True)
    SKILLS_SNAPSHOTS_TABLE.create(engine, checkfirst=True)
    SKILLS_ITEMS_TABLE.create(engine, checkfirst=True)


def _skills_storage_has_data(engine: Engine) -> bool:
    if _count_table_rows(SKILLS_SNAPSHOTS_TABLE) <= 0:
        return False
    with engine.connect() as conn:
        item_count = conn.execute(select(func.count()).select_from(SKILLS_ITEMS_TABLE)).scalar_one()
    return int(item_count) > 0


def _migrate_legacy_skills_storage(engine: Engine) -> bool:
    inspector = inspect(engine)
    if _count_table_rows(SKILLS_SNAPSHOTS_TABLE) > 0:
        return False
    if not inspector.has_table(_LEGACY_SKILLS_SNAPSHOT_TABLE_NAME) or not inspector.has_table(_LEGACY_SKILLS_ITEMS_PAYLOAD_TABLE_NAME):
        return False

    legacy_snapshot_df = pd.read_sql(
        text(
            """
            SELECT snapshot_id, version, saved_at, global_sigma, global_weight
            FROM skills_snapshot
            ORDER BY saved_at DESC
            LIMIT 1
            """
        ),
        engine,
    )
    if legacy_snapshot_df.empty:
        return False

    legacy_snapshot = legacy_snapshot_df.iloc[0].to_dict()
    legacy_items_df = pd.read_sql(
        text(
            """
            SELECT short_name, algorithm_type, sigma_param, expert_weight,
                   alignment_rate, lower_bound, upper_bound, base_price
            FROM skills_items_payload
            WHERE snapshot_id = :snapshot_id
            ORDER BY short_name
            """
        ),
        engine,
        params={"snapshot_id": str(legacy_snapshot["snapshot_id"])},
    )
    if legacy_items_df.empty:
        return False

    legacy_saved_at = pd.to_datetime(legacy_snapshot.get("saved_at"), errors="coerce")
    if pd.isna(legacy_saved_at):
        legacy_saved_at = datetime.now()

    snapshot_row = _rows_from_dataframe(
        pd.DataFrame(
            [
                {
                    "snapshot_id": str(legacy_snapshot["snapshot_id"]),
                    "version": str(legacy_snapshot.get("version") or "1.0"),
                    "saved_at": legacy_saved_at,
                    "global_sigma": float(legacy_snapshot.get("global_sigma") or 1.0),
                    "global_weight": int(legacy_snapshot.get("global_weight") or 80),
                }
            ]
        )
    )

    normalized_items = legacy_items_df.copy()
    normalized_items["snapshot_id"] = str(legacy_snapshot["snapshot_id"])
    normalized_items["short_name"] = normalized_items["short_name"].astype(str)
    normalized_items["algorithm_type"] = normalized_items["algorithm_type"].fillna("").astype(str)
    normalized_items["skill_payload_json"] = None
    for numeric_column in [
        "sigma_param",
        "expert_weight",
        "alignment_rate",
        "lower_bound",
        "upper_bound",
        "base_price",
    ]:
        normalized_items[numeric_column] = pd.to_numeric(normalized_items[numeric_column], errors="coerce")

    item_rows = _rows_from_dataframe(
        normalized_items[
            [
                "snapshot_id",
                "short_name",
                "algorithm_type",
                "sigma_param",
                "expert_weight",
                "alignment_rate",
                "lower_bound",
                "upper_bound",
                "base_price",
                "skill_payload_json",
            ]
        ]
    )

    with engine.begin() as conn:
        conn.execute(SKILLS_SNAPSHOTS_TABLE.insert(), snapshot_row)
        conn.execute(SKILLS_ITEMS_TABLE.insert(), item_rows)
    print("[skills] 已将 legacy skills_snapshot / skills_items_payload 迁移到新结构")
    return True


def _drop_legacy_skills_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table(_LEGACY_SKILLS_SNAPSHOT_TABLE_NAME) and not inspector.has_table(_LEGACY_SKILLS_ITEMS_PAYLOAD_TABLE_NAME):
        return
    if not _skills_storage_has_data(engine):
        return

    _drop_table_by_name(engine, _LEGACY_SKILLS_ITEMS_PAYLOAD_TABLE_NAME, cascade=True)
    _drop_table_by_name(engine, _LEGACY_SKILLS_SNAPSHOT_TABLE_NAME, cascade=True)
    print("[skills] 已清理 legacy skills_snapshot / skills_items_payload 表")


def _ensure_skills_storage_tables() -> None:
    global _SKILLS_STORAGE_RESET_DONE

    if _SKILLS_STORAGE_RESET_DONE or DB_ENGINE is None:
        return

    engine = require_db_engine()
    actual_snapshots_columns = _get_table_columns(engine, SKILLS_SNAPSHOTS_TABLE.name)
    actual_items_columns = _get_table_columns(engine, SKILLS_ITEMS_TABLE.name)
    expected_snapshots_columns = [column.name for column in SKILLS_SNAPSHOTS_TABLE.columns]
    expected_items_columns = [column.name for column in SKILLS_ITEMS_TABLE.columns]

    if actual_items_columns and "skill_payload_json" not in actual_items_columns and set(actual_items_columns).issubset(set(expected_items_columns)):
        _ensure_skills_items_columns()
        actual_items_columns = _get_table_columns(engine, SKILLS_ITEMS_TABLE.name)

    snapshots_mismatch = bool(actual_snapshots_columns) and actual_snapshots_columns != expected_snapshots_columns
    items_mismatch = bool(actual_items_columns) and actual_items_columns != expected_items_columns

    if snapshots_mismatch or items_mismatch:
        if snapshots_mismatch:
            print(f"[skills] 检测到旧表结构，重建 {SKILLS_SNAPSHOTS_TABLE.name}: {actual_snapshots_columns}")
        if items_mismatch:
            print(f"[skills] 检测到旧表结构，重建 {SKILLS_ITEMS_TABLE.name}: {actual_items_columns}")
        _reset_skills_storage_tables(engine)
    else:
        if not actual_snapshots_columns:
            SKILLS_SNAPSHOTS_TABLE.create(engine, checkfirst=True)
        if not actual_items_columns:
            SKILLS_ITEMS_TABLE.create(engine, checkfirst=True)

    _migrate_legacy_skills_storage(engine)
    _drop_legacy_skills_tables(engine)

    _SKILLS_STORAGE_RESET_DONE = True


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

    if "remark" not in legacy_df.columns:
        legacy_df["remark"] = ""
    if "labeled_at" not in legacy_df.columns:
        legacy_df["labeled_at"] = datetime.now()
    legacy_df["remark"] = legacy_df["remark"].fillna("").astype(str)
    legacy_df["labeled_at"] = pd.to_datetime(legacy_df["labeled_at"], errors="coerce")
    legacy_df["labeled_at"] = legacy_df["labeled_at"].fillna(pd.Timestamp(datetime.now()))
    rows = _rows_from_dataframe(
        legacy_df[["record_key", "label", "remark", "labeled_at"]].drop_duplicates("record_key", keep="last")
    )
    _upsert_rows(
        EXPERT_FEEDBACK_TABLE,
        rows,
        conflict_columns=["record_key"],
        update_columns=["label", "remark", "labeled_at"],
    )


def _import_skills_from_legacy_json() -> None:
    if not _LEGACY_SKILLS_PATH.exists():
        return

    _ensure_skills_storage_tables()
    has_snapshot = _count_table_rows(SKILLS_SNAPSHOTS_TABLE) > 0
    if has_snapshot and _latest_skills_snapshot_has_items():
        return

    if has_snapshot:
        latest_snapshot_id = _get_latest_skills_snapshot_id()
        if latest_snapshot_id:
            with require_db_engine().begin() as conn:
                conn.execute(delete(SKILLS_ITEMS_TABLE))
                conn.execute(
                    delete(SKILLS_SNAPSHOTS_TABLE).where(
                        SKILLS_SNAPSHOTS_TABLE.c.snapshot_id == latest_snapshot_id
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
        _drop_legacy_skills_tables(engine)
        _import_core_records_from_legacy_source()
        _DB_INIT_ERROR = None
    except Exception as exc:
        _DB_INIT_ERROR = exc


# ---------------------------------------------------------------------------
# 策略 B：加权自学习异常检测
# ---------------------------------------------------------------------------
_EXPERT_WEIGHT = 80  # 专家标注样本在 KDE 拟合中的复制倍数

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
        ``((record_key, label), ...)`` — 由 ``tuple(get_latest_feedback().items())``
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


