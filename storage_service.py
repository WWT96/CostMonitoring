from __future__ import annotations

import base64
import hashlib
import json as _json
import os
import time
import zlib
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
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
    event,
    func,
    inspect,
    Index,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from config import settings
import harness
from local_logging import log_event


PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))

DB_METADATA = MetaData()

EXPERT_FEEDBACK_TABLE = Table(
    "expert_feedback",
    DB_METADATA,
    Column("record_key", String(160), primary_key=True),
    Column("label", String(32), nullable=False),
    Column("remark", Text),
    Column("labeled_at", DateTime, nullable=False),
)

SHEET_METAL_FEEDBACK_TABLE = Table(
    "sheet_metal_feedback",
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
    Column("material_code", String(64)),
    Column("material_name", String(255)),
    Column("supplier_code", String(64)),
    Column("supplier_name", String(255)),
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
    Column("supplier_name", String(255)),
    Column("supplier_code", String(64)),
    Column("price_valid_to", DateTime),
    Column("assy_part_no", String(64)),
    Column("assy_desc", String(255)),
    Column("assy_supplier_name", String(255)),
    Column("assy_supplier_code", String(64)),
    Column("assy_cost", Float),
    Column("source_row_hash", String(64)),
    Column("created_at", DateTime, nullable=False),
)

Index(
    "ux_core_cost_records_business_key",
    CORE_COST_RECORDS_TABLE.c.source_row_hash,
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
    Column("ring_id", Integer),
    Column("ring_role", Text),
    Column("ring_confidence", Float),
    Column("ring_intervals_json", Text),
    Column("expert_adjusted", Text),
    Column("decision_basis", Text),
    Column("result_mode", Text),
    Column("computed_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

COST_ANOMALY_RESULT_RUNS_TABLE = Table(
    "cost_anomaly_result_runs",
    DB_METADATA,
    Column("result_mode", Text, primary_key=True),
    Column("source_signature", Text, nullable=False),
    Column("options_signature", Text, nullable=False),
    Column("row_count", Integer, nullable=False),
    Column("computed_at", DateTime, nullable=False),
)

SKILL_DOMAIN_COST = "cost"
SKILL_DOMAIN_SHEET_METAL = "sheet_metal"

SKILLS_SNAPSHOTS_TABLE = Table(
    "skills_snapshots",
    DB_METADATA,
    Column("snapshot_id", String(36), primary_key=True),
    Column("module_type", String(32), nullable=False, server_default=text("'cost'")),
    Column("skill_domain", String(32), nullable=False, server_default=text("'cost'")),
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
    Column("module_type", String(32), nullable=False, server_default=text("'cost'")),
    Column("skill_domain", String(32), nullable=False, server_default=text("'cost'")),
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

VEHICLE_RANK_CONFIG_TABLE = Table(
    "vehicle_rank_config",
    DB_METADATA,
    Column("vehicle_series", String(128), primary_key=True),
    Column("rank_order", Integer, nullable=False),
    Column("source", String(64)),
    Column("updated_at", DateTime, nullable=False),
)

VEHICLE_MARKET_PRICES_TABLE = Table(
    "vehicle_market_prices",
    DB_METADATA,
    Column("vehicle_series", String(128), primary_key=True),
    Column("market_price", Float),
    Column("variant_name", String(255)),
    Column("source_url", Text),
    Column("source_domain", String(128)),
    Column("status", String(32), nullable=False),
    Column("fetched_at", DateTime, nullable=False),
    Column("failure_reason", Text),
    Column("raw_response_json", Text),
)

_CORE_RECORD_EXPORT_COLUMNS = [
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
    "source_row_hash",
]

_CORE_COST_RECORDS_BUSINESS_KEY_COLUMNS = ["source_row_hash"]
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
    "ring_id",
    "ring_role",
    "ring_confidence",
    "ring_intervals_json",
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
    "圈层编号": "ring_id",
    "圈层角色": "ring_role",
    "圈层置信度": "ring_confidence",
    "多圈合理区间": "ring_intervals_json",
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
    "ring_id",
    "ring_confidence",
]

_ANOMALY_REQUIRED_COLUMNS = [
    "_record_key",
    "material_code",
    "actual_cost",
]

_COST_ANOMALY_RESULTS_CREATE_SQL = """
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
    ring_id INTEGER,
    ring_role TEXT,
    ring_confidence FLOAT,
    ring_intervals_json TEXT,
    expert_adjusted TEXT,
    decision_basis TEXT,
    result_mode TEXT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_DB_INIT_ERROR: Optional[Exception] = None
_COST_ANOMALY_RESULTS_RESET_DONE = False
_SKILLS_STORAGE_RESET_DONE = False
SQLITE_BUSY_TIMEOUT_SECONDS = 30
SQLITE_CONNECTION_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=30000",
)
_LEGACY_SKILLS_SNAPSHOT_TABLE_NAME = "skills_snapshot"
_LEGACY_SKILLS_ITEMS_PAYLOAD_TABLE_NAME = "skills_items_payload"
RECORD_KEY_VERSION = "rk2"
RECORD_KEY_DIGEST_LENGTH = 16
RECORD_KEY_NUMERIC_SCALE = Decimal("0.0000000001")


def _build_sqlite_connect_args() -> Dict[str, Any]:
    return {
        "check_same_thread": False,
        "timeout": SQLITE_BUSY_TIMEOUT_SECONDS,
    }


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        for pragma_sql in SQLITE_CONNECTION_PRAGMAS:
            cursor.execute(pragma_sql)
    finally:
        cursor.close()


def _build_db_engine() -> Optional[Engine]:
    global _DB_INIT_ERROR

    try:
        harness.authorize_db_operation("connect", "storage_service._build_db_engine")
        engine_kwargs: Dict[str, Any] = {
            "future": True,
            "pool_pre_ping": True,
        }
        if settings.db_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = _build_sqlite_connect_args()
        engine = create_engine(settings.db_url, **engine_kwargs)
        if settings.db_url.startswith("sqlite"):
            event.listen(engine, "connect", _configure_sqlite_connection)
        return engine
    except Exception as exc:
        _DB_INIT_ERROR = exc
        return None


DB_ENGINE = _build_db_engine()


def require_db_engine() -> Engine:
    harness.authorize_db_operation("connect", "storage_service.require_db_engine")
    if DB_ENGINE is None:
        if _DB_INIT_ERROR is not None:
            raise RuntimeError(f"本地 SQLite 数据库不可用: {_DB_INIT_ERROR}")
        raise RuntimeError("本地 SQLite 数据库不可用")
    return DB_ENGINE


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
    stmt = sqlite_insert(table).values(list(rows))
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


def _ensure_core_cost_records_columns(engine: Engine) -> None:
    actual_columns = _get_table_columns(engine, CORE_COST_RECORDS_TABLE.name)
    if not actual_columns:
        CORE_COST_RECORDS_TABLE.create(engine, checkfirst=True)
        return

    expected_columns = {
        "supplier_name": "VARCHAR(255)",
        "supplier_code": "VARCHAR(64)",
        "price_valid_to": "TIMESTAMP",
        "source_row_hash": "VARCHAR(64)",
    }
    missing_columns = [column_name for column_name in expected_columns if column_name not in actual_columns]
    if missing_columns:
        with engine.begin() as conn:
            for column_name in missing_columns:
                conn.execute(
                    text(
                        f'ALTER TABLE "{CORE_COST_RECORDS_TABLE.name}" '
                        f'ADD COLUMN {column_name} {expected_columns[column_name]}'
                    )
                )

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f'SELECT cost_record_id, material_code, factory, monitor_date, cost_amount, '
                f'material_name, vehicle_series, short_name '
                f'FROM "{CORE_COST_RECORDS_TABLE.name}" '
                f"WHERE source_row_hash IS NULL OR source_row_hash = ''"
            )
        ).mappings().all()
        for row in rows:
            payload = _json.dumps(
                {
                    "legacy_id": row["cost_record_id"],
                    "material_code": row["material_code"],
                    "factory": row["factory"],
                    "monitor_date": str(row["monitor_date"]),
                    "cost_amount": row["cost_amount"],
                    "material_name": row["material_name"],
                    "vehicle_series": row["vehicle_series"],
                    "short_name": row["short_name"],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            source_row_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            conn.execute(
                text(
                    f'UPDATE "{CORE_COST_RECORDS_TABLE.name}" '
                    f"SET source_row_hash = :source_row_hash WHERE cost_record_id = :cost_record_id"
                ),
                {"source_row_hash": source_row_hash, "cost_record_id": row["cost_record_id"]},
            )


def _dedupe_core_cost_records_table(session: Session) -> None:
    rows_df = pd.read_sql(
        select(
            CORE_COST_RECORDS_TABLE.c.cost_record_id,
            CORE_COST_RECORDS_TABLE.c.source_row_hash,
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
    _ensure_core_cost_records_columns(engine)

    if _core_cost_records_has_business_key_index(engine):
        return

    with Session(engine) as session:
        with session.begin():
            _dedupe_core_cost_records_table(session)
            session.execute(
                text(
                    'DROP INDEX IF EXISTS "ux_core_cost_records_business_key"'
                )
            )
            session.execute(
                text(
                    'CREATE UNIQUE INDEX IF NOT EXISTS "ux_core_cost_records_business_key" '
                    'ON "core_cost_records" (source_row_hash)'
                )
            )


def _normalize_record_key_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_record_key_datetime(value: Any) -> tuple[str, str]:
    timestamp = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(timestamp):
        raw_text = _normalize_record_key_text(value)
        return raw_text, raw_text[:10]

    normalized = pd.Timestamp(timestamp)
    if normalized.tzinfo is not None:
        normalized = normalized.tz_convert("UTC").tz_localize(None)

    normalized = normalized.normalize()

    return normalized.isoformat(), normalized.strftime("%Y-%m-%d")


def _normalize_record_key_numeric(value: Any) -> tuple[str, Any]:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return _normalize_record_key_text(value), np.nan

    decimal_value = Decimal(str(numeric_value)).quantize(
        RECORD_KEY_NUMERIC_SCALE,
        rounding=ROUND_HALF_UP,
    )
    normalized_text = format(decimal_value, "f").rstrip("0").rstrip(".") or "0"
    return normalized_text, float(decimal_value)


def _encode_record_key_payload(payload: Dict[str, str]) -> str:
    payload_json = _json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _encode_record_key_payload_json(payload_json)


def _encode_record_key_payload_json(payload_json: str | bytes) -> str:
    if isinstance(payload_json, str):
        payload_bytes = payload_json.encode("utf-8")
    else:
        payload_bytes = payload_json
    compressed = zlib.compress(payload_bytes, level=9)
    return base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")


def _build_stable_record_key(payload: Dict[str, str]) -> str:
    payload_json = _json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:RECORD_KEY_DIGEST_LENGTH]
    return f"{RECORD_KEY_VERSION}:{digest}:{_encode_record_key_payload_json(payload_json)}"


def _build_stable_record_key_from_parts(material_code: str, factory: str, date_iso_text: str, metric_text: str) -> str:
    if not any([material_code, factory, date_iso_text, metric_text]):
        return ""
    payload = {"m": material_code, "f": factory, "d": date_iso_text, "v": metric_text}
    return _build_stable_record_key(payload)


def _build_stable_record_keys_from_parts(
    material_values: Sequence[str],
    factory_values: Sequence[str],
    date_iso_texts: Sequence[str],
    metric_texts: Sequence[str],
) -> List[str]:
    cache: Dict[tuple[str, str, str, str], str] = {}
    record_keys: List[str] = []
    for material_code, factory, date_iso_text, metric_text in zip(
        material_values,
        factory_values,
        date_iso_texts,
        metric_texts,
    ):
        key_tuple = (str(material_code or ""), str(factory or ""), str(date_iso_text or ""), str(metric_text or ""))
        cached_key = cache.get(key_tuple)
        if cached_key is None:
            cached_key = _build_stable_record_key_from_parts(*key_tuple)
            cache[key_tuple] = cached_key
        record_keys.append(cached_key)
    return record_keys


def _decode_record_key_payload(payload_token: str) -> Optional[Dict[str, str]]:
    if not payload_token:
        return None
    padded = payload_token + ("=" * (-len(payload_token) % 4))
    try:
        payload_bytes = base64.urlsafe_b64decode(padded)
        payload_json = zlib.decompress(payload_bytes).decode("utf-8")
        payload = _json.loads(payload_json)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    return {
        "m": _normalize_record_key_text(payload.get("m")),
        "f": _normalize_record_key_text(payload.get("f")),
        "d": _normalize_record_key_text(payload.get("d")),
        "v": _normalize_record_key_text(payload.get("v")),
    }


def _build_legacy_record_key(
    material_code: str,
    factory: str,
    date_text: str,
    value_text: str,
) -> str:
    if not any([material_code, factory, date_text, value_text]):
        return ""

    numeric_value = pd.to_numeric(pd.Series([value_text]), errors="coerce").iloc[0]
    legacy_value = value_text
    if pd.notna(numeric_value):
        legacy_value = f"{float(numeric_value):.4f}"
    return f"{material_code}_{factory}_{date_text}_{legacy_value}"


def _build_record_key_metadata(
    material_code: Any,
    factory: Any,
    date_value: Any,
    metric_value: Any,
    *,
    fallback_key: str = "",
) -> Dict[str, Any]:
    material_code_text = _normalize_record_key_text(material_code)
    factory_text = _normalize_record_key_text(factory)
    date_iso_text, date_day_text = _normalize_record_key_datetime(date_value)
    value_text, numeric_value = _normalize_record_key_numeric(metric_value)

    if not any([material_code_text, factory_text, date_iso_text, value_text]):
        return {
            "record_key": _normalize_record_key_text(fallback_key),
            "legacy_record_key": _normalize_record_key_text(fallback_key),
            "物料编码": material_code_text,
            "工厂": factory_text,
            "价格有效期于": date_day_text,
            "价格": float(numeric_value) if pd.notna(numeric_value) else np.nan,
            "_join_date_key": date_day_text,
            "_join_price_key": value_text,
            "_record_date_iso": date_iso_text,
            "_record_value_text": value_text,
        }

    payload = {
        "m": material_code_text,
        "f": factory_text,
        "d": date_iso_text,
        "v": value_text,
    }
    join_price_key = value_text
    if pd.notna(numeric_value):
        join_price_key = f"{float(numeric_value):.4f}"

    return {
        "record_key": _build_stable_record_key(payload),
        "legacy_record_key": _build_legacy_record_key(
            material_code_text,
            factory_text,
            date_day_text,
            value_text,
        ),
        "物料编码": material_code_text,
        "工厂": factory_text,
        "价格有效期于": date_day_text,
        "价格": float(numeric_value) if pd.notna(numeric_value) else np.nan,
        "_join_date_key": date_day_text,
        "_join_price_key": join_price_key,
        "_record_date_iso": date_iso_text,
        "_record_value_text": value_text,
    }


def _split_legacy_record_key(record_key: str) -> Optional[Dict[str, str]]:
    if not record_key or record_key.startswith(f"{RECORD_KEY_VERSION}:"):
        return None
    parts = record_key.split("_", 3)
    if len(parts) != 4:
        return None
    material_code, factory, monitor_date_text, price_text = [part.strip() for part in parts]
    return {
        "material_code": material_code,
        "factory": factory,
        "monitor_date_text": monitor_date_text,
        "price_text": price_text,
    }


def _decode_stable_record_key(record_key: str) -> Optional[Dict[str, Any]]:
    if not record_key.startswith(f"{RECORD_KEY_VERSION}:"):
        return None
    parts = record_key.split(":", 2)
    if len(parts) != 3:
        return None

    payload = _decode_record_key_payload(parts[2])
    if payload is None:
        return None

    metadata = _build_record_key_metadata(
        payload.get("m", ""),
        payload.get("f", ""),
        payload.get("d", ""),
        payload.get("v", ""),
        fallback_key=record_key,
    )
    if metadata["record_key"].split(":", 2)[1] != parts[1]:
        return None
    return metadata


def strip_result_mode_record_key_prefix(record_key: Any) -> str:
    text_key = _normalize_record_key_text(record_key)
    if not text_key or "::" not in text_key:
        return text_key
    prefix, remainder = text_key.split("::", 1)
    if prefix in {"raw", "weighted"} and remainder:
        return remainder
    return text_key


def canonicalize_record_key(record_key: Any) -> str:
    text_key = _normalize_record_key_text(record_key)
    if not text_key:
        return ""
    text_key = strip_result_mode_record_key_prefix(text_key)

    decoded = _decode_stable_record_key(text_key)
    if decoded is not None:
        return decoded["record_key"]

    legacy_parts = _split_legacy_record_key(text_key)
    if legacy_parts is None:
        return text_key

    return _build_record_key_metadata(
        legacy_parts["material_code"],
        legacy_parts["factory"],
        legacy_parts["monitor_date_text"],
        legacy_parts["price_text"],
        fallback_key=text_key,
    )["record_key"]


def get_record_key_aliases(record_key: Any) -> set[str]:
    text_key = _normalize_record_key_text(record_key)
    text_key = strip_result_mode_record_key_prefix(text_key)
    aliases = {text_key} if text_key else set()

    decoded = _decode_stable_record_key(text_key) if text_key else None
    if decoded is None and text_key:
        legacy_parts = _split_legacy_record_key(text_key)
        if legacy_parts is not None:
            decoded = _build_record_key_metadata(
                legacy_parts["material_code"],
                legacy_parts["factory"],
                legacy_parts["monitor_date_text"],
                legacy_parts["price_text"],
                fallback_key=text_key,
            )

    if decoded is not None:
        aliases.add(decoded["record_key"])
        if decoded.get("legacy_record_key"):
            aliases.add(decoded["legacy_record_key"])

    return {alias for alias in aliases if alias}


def make_record_key(row) -> str:
    return _build_record_key_metadata(
        row.get("物料编码", ""),
        row.get("工厂", ""),
        row.get("价格有效于", ""),
        row.get("实际成本", ""),
    )["record_key"]


def build_record_keys(
    df: pd.DataFrame,
    *,
    material_column: str = "物料编码",
    factory_column: str = "工厂",
    date_column: str = "价格有效于",
    value_column: str = "实际成本",
) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(index=getattr(df, "index", None), dtype="string")

    def _column_or_default(column_name: str, default_value: Any) -> pd.Series:
        if column_name in df.columns:
            return df[column_name]
        return pd.Series([default_value] * len(df), index=df.index)

    material_values = _column_or_default(material_column, "").map(_normalize_record_key_text).to_numpy(dtype=object)
    factory_values = _column_or_default(factory_column, "").map(_normalize_record_key_text).to_numpy(dtype=object)

    raw_date_series = _column_or_default(date_column, "")
    raw_date_texts = raw_date_series.map(_normalize_record_key_text)
    parsed_dates = pd.to_datetime(raw_date_series, errors="coerce")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert("UTC").dt.tz_localize(None)
    normalized_dates = parsed_dates.dt.normalize()
    date_iso_texts = normalized_dates.dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_iso_texts = date_iso_texts.where(normalized_dates.notna(), raw_date_texts).to_numpy(dtype=object)

    raw_metric_series = _column_or_default(value_column, "")
    raw_metric_texts = raw_metric_series.map(_normalize_record_key_text).tolist()
    numeric_values = pd.to_numeric(raw_metric_series, errors="coerce").tolist()
    metric_texts: List[str] = []
    for raw_text, numeric_value in zip(raw_metric_texts, numeric_values):
        if pd.isna(numeric_value):
            metric_texts.append(raw_text)
            continue
        decimal_value = Decimal(str(numeric_value)).quantize(
            RECORD_KEY_NUMERIC_SCALE,
            rounding=ROUND_HALF_UP,
        )
        metric_texts.append(format(decimal_value, "f").rstrip("0").rstrip(".") or "0")

    # 这里先做列级解析，再逐行组装 payload，避免每行重复创建 pandas Series。
    record_keys = _build_stable_record_keys_from_parts(
        material_values,
        factory_values,
        date_iso_texts,
        metric_texts,
    )
    return pd.Series(record_keys, index=df.index, dtype="string")


def make_metric_record_key(
    row,
    *,
    value_column: str = "实际成本",
    date_column: str = "价格有效于",
) -> str:
    return _build_record_key_metadata(
        row.get("物料编码", ""),
        row.get("工厂", ""),
        row.get(date_column, ""),
        row.get(value_column, ""),
    )["record_key"]


def split_record_key(record_key: Any) -> Dict[str, Any]:
    text_key = str(record_key or "").strip()
    text_key = strip_result_mode_record_key_prefix(text_key)
    decoded = _decode_stable_record_key(text_key)
    if decoded is None:
        legacy_parts = _split_legacy_record_key(text_key)
        if legacy_parts is not None:
            decoded = _build_record_key_metadata(
                legacy_parts["material_code"],
                legacy_parts["factory"],
                legacy_parts["monitor_date_text"],
                legacy_parts["price_text"],
                fallback_key=text_key,
            )

    if decoded is None:
        return {
            "record_key": text_key,
            "物料编码": "",
            "工厂": "",
            "价格有效期于": "",
            "价格": np.nan,
            "_join_date_key": "",
            "_join_price_key": "",
        }

    return {
        "record_key": decoded["record_key"],
        "物料编码": decoded["物料编码"],
        "工厂": decoded["工厂"],
        "价格有效期于": decoded["价格有效期于"],
        "价格": decoded["价格"],
        "_join_date_key": decoded["_join_date_key"],
        "_join_price_key": decoded["_join_price_key"],
    }


def split_metric_record_key(
    record_key: Any,
    *,
    value_label: str = "实际成本",
    date_label: str = "价格有效于",
) -> Dict[str, Any]:
    parsed = split_record_key(record_key)
    parsed[date_label] = parsed.pop("价格有效期于")
    parsed[value_label] = parsed.pop("价格")
    return parsed


def _resolve_feedback_table(table_name: str) -> Table:
    table_map = {
        EXPERT_FEEDBACK_TABLE.name: EXPERT_FEEDBACK_TABLE,
        SHEET_METAL_FEEDBACK_TABLE.name: SHEET_METAL_FEEDBACK_TABLE,
    }
    if table_name not in table_map:
        raise ValueError(f"不支持的标注表: {table_name}")
    return table_map[table_name]


def _normalize_feedback_rows_for_validation(final_labels_df: Any) -> pd.DataFrame:
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
        data = pd.DataFrame(rows)
    elif final_labels_df is None:
        data = pd.DataFrame(columns=["record_key", "label", "remark"])
    else:
        data = final_labels_df.copy()

    data = data.rename(
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
    for column_name in ["record_key", "label", "remark"]:
        if column_name not in data.columns:
            data[column_name] = pd.Series(dtype="string")
    data = data[["record_key", "label", "remark"]].copy()
    data["record_key"] = data["record_key"].apply(canonicalize_record_key).astype("string").str.strip()
    data["label"] = data["label"].fillna("").astype("string").str.strip()
    data["remark"] = data["remark"].fillna("").astype("string").str.strip()
    return data


def find_feedback_rows_missing_required_remarks(final_labels_df: Any) -> List[str]:
    """Return record keys that are marked normal without an expert remark."""
    data = _normalize_feedback_rows_for_validation(final_labels_df)
    if data.empty:
        return []
    normal_mask = data["label"].astype(str).str.startswith("正常", na=False)
    blank_remark_mask = data["remark"].astype(str).str.strip().eq("")
    missing = data.loc[normal_mask & blank_remark_mask, "record_key"]
    return [str(key) for key in missing.tolist() if str(key).strip()]


def _raise_if_feedback_remarks_missing(final_labels_df: Any) -> None:
    missing_keys = find_feedback_rows_missing_required_remarks(final_labels_df)
    if missing_keys:
        preview = "、".join(missing_keys[:5])
        suffix = "" if len(missing_keys) <= 5 else f" 等 {len(missing_keys)} 条"
        raise ValueError(f"标注为正常的记录必须填写批注原因：{preview}{suffix}")


class LabelManager:
    _COLUMNS = ["record_key", "label", "remark", "labeled_at"]

    def __init__(self, table_name: str = "expert_feedback"):
        self._table_name = table_name
        self._table = _resolve_feedback_table(table_name)

    def get_labels(self) -> Dict[str, Dict[str, str]]:
        if DB_ENGINE is None:
            return {}
        try:
            query = select(
                self._table.c.record_key,
                self._table.c.label,
                self._table.c.remark,
                self._table.c.labeled_at,
            )
            df = pd.read_sql(query, require_db_engine())
            if df.empty:
                return {}
            df["record_key"] = df["record_key"].apply(canonicalize_record_key)
            df["label"] = df["label"].astype(str)
            df["remark"] = df["remark"].fillna("").astype(str)
            df["labeled_at"] = pd.to_datetime(df["labeled_at"], errors="coerce")
            df = df.sort_values("labeled_at", na_position="last")
            df = df[df["record_key"].astype(str).str.strip() != ""]
            df = df.drop_duplicates(subset=["record_key"], keep="last")
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
        return {
            record_key: payload.get("label", "")
            for record_key, payload in self.get_labels().items()
        }

    def get_label_remarks(self) -> Dict[str, str]:
        return {
            record_key: payload.get("remark", "")
            for record_key, payload in self.get_labels().items()
        }

    def get_label_records(self) -> pd.DataFrame:
        if DB_ENGINE is None:
            return pd.DataFrame(columns=["record_key", "label", "remark", "labeled_at"])
        try:
            query = select(
                self._table.c.record_key,
                self._table.c.label,
                self._table.c.remark,
                self._table.c.labeled_at,
            )
            df = pd.read_sql(query, require_db_engine())
            if df.empty:
                return pd.DataFrame(columns=["record_key", "label", "remark", "labeled_at"])
            df["record_key"] = df["record_key"].apply(canonicalize_record_key)
            df["label"] = df["label"].astype(str)
            df["remark"] = df["remark"].fillna("").astype(str)
            df["labeled_at"] = pd.to_datetime(df["labeled_at"], errors="coerce")
            df = df.sort_values("labeled_at", na_position="last")
            df = df[df["record_key"].astype(str).str.strip() != ""]
            df = df.drop_duplicates(subset=["record_key"], keep="last")
            return df
        except Exception:
            return pd.DataFrame(columns=["record_key", "label", "remark", "labeled_at"])

    def count(self) -> int:
        if DB_ENGINE is None:
            return 0
        try:
            return _count_table_rows(self._table)
        except Exception:
            return 0

    def save_label(self, key: str, status: str, remark: str = "") -> None:
        _raise_if_feedback_remarks_missing(
            pd.DataFrame([{"record_key": key, "label": status, "remark": remark}])
        )
        now = datetime.now()
        canonical_key = canonicalize_record_key(key)
        alias_keys = sorted(alias for alias in get_record_key_aliases(key) if alias != canonical_key)
        with Session(require_db_engine()) as session:
            with session.begin():
                if alias_keys:
                    session.execute(delete(self._table).where(self._table.c.record_key.in_(alias_keys)))
                _upsert_rows(
                    self._table,
                    [{"record_key": canonical_key, "label": status, "remark": str(remark or "").strip(), "labeled_at": now}],
                    conflict_columns=["record_key"],
                    update_columns=["label", "remark", "labeled_at"],
                    session=session,
                )
        log_event(
            "feedback",
            "save_label",
            "Saved a single expert feedback label",
            record_key=canonical_key,
            label=str(status),
            has_remark=bool(str(remark or "").strip()),
        )

    def save_labels_batch(self, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        now = datetime.now()
        rows_by_key: Dict[str, Dict[str, Any]] = {}
        alias_keys_to_delete: set[str] = set()
        for record_key, payload in updates.items():
            if isinstance(payload, dict):
                label = payload.get("label")
                remark = payload.get("remark", "")
            else:
                label = payload
                remark = ""
            if label is None or str(label).strip() == "":
                continue
            canonical_key = canonicalize_record_key(record_key)
            if not canonical_key:
                continue
            alias_keys_to_delete.update(
                alias for alias in get_record_key_aliases(record_key) if alias != canonical_key
            )
            rows_by_key[canonical_key] = {
                "record_key": canonical_key,
                "label": str(label).strip(),
                "remark": str(remark or "").strip(),
                "labeled_at": now,
            }
        rows = list(rows_by_key.values())
        _raise_if_feedback_remarks_missing(pd.DataFrame(rows))
        with Session(require_db_engine()) as session:
            with session.begin():
                if alias_keys_to_delete:
                    session.execute(
                        delete(self._table).where(self._table.c.record_key.in_(sorted(alias_keys_to_delete)))
                    )
                _upsert_rows(
                    self._table,
                    rows,
                    conflict_columns=["record_key"],
                    update_columns=["label", "remark", "labeled_at"],
                    session=session,
                )
        if rows:
            log_event(
                "feedback",
                "save_labels_batch",
                "Saved expert feedback labels in batch",
                row_count=len(rows),
            )

    def replace_all(self, final_labels_df: pd.DataFrame) -> None:
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
        final_df["record_key"] = final_df["record_key"].apply(canonicalize_record_key).astype("string").str.strip()
        final_df["label"] = final_df["label"].astype("string").str.strip()
        final_df["remark"] = final_df["remark"].fillna("").astype("string").str.strip()
        final_df = final_df.replace({"record_key": {"": pd.NA}, "label": {"": pd.NA}})
        final_df = final_df.dropna(subset=["record_key", "label"])
        final_df = final_df.drop_duplicates(subset=["record_key"], keep="last")
        _raise_if_feedback_remarks_missing(final_df)
        final_df["labeled_at"] = datetime.now()

        previous_count = self.count()
        rows = _rows_from_dataframe(final_df[self._COLUMNS])
        with Session(require_db_engine()) as session:
            with session.begin():
                session.execute(delete(self._table))
                _insert_rows(self._table, rows, session=session)
        log_event(
            "feedback",
            "replace_all",
            "Replaced expert feedback snapshot",
            final_count=len(final_df),
            previous_count=previous_count,
        )

    def _flush(self, final_labels_df: pd.DataFrame) -> None:
        self.replace_all(final_labels_df)

    def delete_labels(self, keys_to_remove) -> int:
        keys: set[str] = set()
        for key in keys_to_remove:
            keys.update(get_record_key_aliases(key))
        keys = {key for key in keys if key}
        if not keys:
            return 0
        with require_db_engine().begin() as conn:
            result = conn.execute(
                delete(self._table).where(self._table.c.record_key.in_(sorted(keys)))
            )
        deleted_count = int(result.rowcount or 0)
        if deleted_count:
            log_event(
                "feedback",
                "delete_labels",
                "Deleted expert feedback labels",
                deleted_count=deleted_count,
            )
        return deleted_count

    def clear_all(self) -> None:
        with require_db_engine().begin() as conn:
            conn.execute(delete(self._table))
        log_event(
            "feedback",
            "clear_all",
            "Cleared all expert feedback labels",
        )

    def file_row_count(self) -> int:
        return self.count()


def get_latest_feedback() -> Dict[str, str]:
    return label_manager.get_label_statuses()


def get_latest_feedback_details() -> Dict[str, Dict[str, str]]:
    return label_manager.get_labels()


_DISPLAY_TEXT_PLACEHOLDERS = {"", "nan", "none", "null", "<na>", "nat"}


def _clean_display_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text_value = str(value).strip()
    return "" if text_value.lower() in _DISPLAY_TEXT_PLACEHOLDERS else text_value


def load_expert_knowledge_base() -> pd.DataFrame:
    _ensure_expert_knowledge_base_columns()
    engine = require_db_engine()
    query = select(EXPERT_KNOWLEDGE_BASE_TABLE).order_by(EXPERT_KNOWLEDGE_BASE_TABLE.c.updated_at.desc())
    df = pd.read_sql(query, engine)
    if df.empty:
        return df
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    for column_name in ["rule_id", "short_name", "material_code", "material_name", "supplier_code", "supplier_name", "vehicle_series", "rule_content"]:
        if column_name in df.columns:
            df[column_name] = df[column_name].map(_clean_display_text)
    identity_columns = [
        column_name
        for column_name in ["short_name", "material_code", "material_name", "vehicle_series"]
        if column_name in df.columns
    ]
    if identity_columns:
        identity_text = df[identity_columns].fillna("").astype(str).agg("".join, axis=1).str.strip()
        df = df[identity_text.ne("")].reset_index(drop=True)
    return df


def get_expert_knowledge_last_updated_at() -> Optional[pd.Timestamp]:
    _ensure_expert_knowledge_base_columns()
    engine = require_db_engine()
    with engine.connect() as conn:
        latest = conn.execute(select(func.max(EXPERT_KNOWLEDGE_BASE_TABLE.c.updated_at))).scalar_one_or_none()
    if latest is None:
        return None
    return pd.Timestamp(latest)


def save_expert_knowledge_rules(rules: Sequence[Dict[str, Any]]) -> int:
    if not rules:
        return 0
    _ensure_expert_knowledge_base_columns()
    data = pd.DataFrame(rules)
    for column_name in ["rule_id", "short_name", "material_code", "material_name", "supplier_code", "supplier_name", "vehicle_series", "rule_content"]:
        if column_name in data.columns:
            data[column_name] = data[column_name].map(_clean_display_text)
    rows = _rows_from_dataframe(data)
    _upsert_rows(
        EXPERT_KNOWLEDGE_BASE_TABLE,
        rows,
        conflict_columns=["rule_id"],
        update_columns=[
            "short_name",
            "material_code",
            "material_name",
            "supplier_code",
            "supplier_name",
            "vehicle_series",
            "rule_content",
            "confidence_score",
            "updated_at",
        ],
    )
    return len(rows)


def delete_expert_knowledge_rules(rule_ids: Sequence[str]) -> int:
    keys = [str(rule_id).strip() for rule_id in rule_ids if str(rule_id).strip()]
    if not keys:
        return 0
    _ensure_expert_knowledge_base_columns()
    with require_db_engine().begin() as conn:
        result = conn.execute(
            delete(EXPERT_KNOWLEDGE_BASE_TABLE).where(EXPERT_KNOWLEDGE_BASE_TABLE.c.rule_id.in_(keys))
        )
    return int(result.rowcount or 0)


def clear_expert_knowledge_base() -> int:
    _ensure_expert_knowledge_base_columns()
    with require_db_engine().begin() as conn:
        result = conn.execute(delete(EXPERT_KNOWLEDGE_BASE_TABLE))
    deleted_count = int(result.rowcount or 0)
    if deleted_count:
        log_event(
            "expert_knowledge",
            "clear_all",
            "Cleared all expert knowledge rules",
            deleted_count=deleted_count,
        )
    return deleted_count


def get_expert_knowledge_refresh_token() -> float:
    latest = get_expert_knowledge_last_updated_at()
    if latest is None:
        return 0.0
    return float(pd.Timestamp(latest).timestamp())


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
        "圈层编号": None,
        "圈层角色": None,
        "圈层置信度": None,
        "多圈合理区间": None,
        "样本量": 0,
        "status": "正常",
    }
    for source_column, default_value in optional_defaults.items():
        target_column = _ANOMALY_RESULT_COLUMN_MAPPING[source_column]
        if source_column not in data.columns and target_column not in data.columns:
            data[source_column] = default_value

    data["result_mode"] = result_mode
    data["computed_at"] = datetime.now()
    prepared = data.rename(columns=dict(_ANOMALY_RESULT_COLUMN_MAPPING))

    for column_name in target_columns:
        if column_name not in prepared.columns:
            prepared[column_name] = None

    prepared = prepared[target_columns]

    for column_name in _ANOMALY_NUMERIC_COLUMNS:
        if column_name in prepared.columns:
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

    valid_record_key_mask = prepared["_record_key"].notna()
    prepared.loc[valid_record_key_mask, "_record_key"] = (
        f"{result_mode}::" + prepared.loc[valid_record_key_mask, "_record_key"].astype(str)
    )
    prepared = prepared.drop_duplicates(subset=["_record_key"], keep="last")
    prepared = prepared.dropna(subset=["material_code"])
    prepared = prepared.dropna(subset=_ANOMALY_REQUIRED_COLUMNS)
    return prepared.reset_index(drop=True)


def _resolve_sqlite_multi_chunksize(conn, column_count: int, requested_chunksize: int = 10000) -> int:
    if column_count <= 0:
        return requested_chunksize

    dialect_name = str(getattr(conn.dialect, "name", "") or "").lower()
    if dialect_name != "sqlite":
        return requested_chunksize

    max_variable_number = 999
    try:
        for row in conn.exec_driver_sql("PRAGMA compile_options"):
            option_text = str(row[0] if row else "")
            if "MAX_VARIABLE_NUMBER=" in option_text:
                max_variable_number = int(option_text.split("MAX_VARIABLE_NUMBER=", 1)[1])
                break
    except Exception:
        pass

    # multi insert 会把“列数 * 行数”全部展开成参数，这里按 SQLite 实际上限回推安全批大小。
    safe_chunksize = max(1, (max_variable_number - 32) // max(column_count, 1))
    return max(1, min(requested_chunksize, safe_chunksize))


def _preview_column_names(columns: Sequence[Any], limit: int = 20) -> List[str]:
    preview = [str(column) for column in list(columns)[:limit]]
    if len(columns) > limit:
        preview.append(f"...另有 {len(columns) - limit} 列")
    return preview


def _build_cost_anomaly_write_failure_log_lines(
    result_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    result_mode: str,
    exc: Exception,
) -> List[str]:
    raw_row_count = 0 if result_df is None else int(len(result_df))
    prepared_row_count = 0 if prepared_df is None else int(len(prepared_df))
    raw_columns = [] if result_df is None else _preview_column_names(result_df.columns)
    prepared_columns = [] if prepared_df is None else _preview_column_names(prepared_df.columns)
    return [
        f"[cost_anomaly_results] 写入失败: {type(exc).__name__}",
        f"[cost_anomaly_results] result_mode={result_mode} 原始行数={raw_row_count} 待写入行数={prepared_row_count}",
        f"[cost_anomaly_results] 原始列名预览: {raw_columns}",
        f"[cost_anomaly_results] 待写入列名预览: {prepared_columns}",
        "[cost_anomaly_results] 异常详情和数据样本已省略，避免输出成本/物料明细。",
    ]


def save_cost_anomaly_results(result_df: pd.DataFrame, result_mode: str = "raw") -> int:
    total_started_at = time.perf_counter()
    _ensure_cost_anomaly_results_table()
    _ensure_cost_anomaly_result_runs_table()
    prepare_started_at = time.perf_counter()
    df = _prepare_anomaly_results(result_df, result_mode)
    prepare_seconds = time.perf_counter() - prepare_started_at
    if df.empty:
        print(
            f"[performance][存库阶段][{result_mode}] 无可写入记录，"
            f"清洗耗时={prepare_seconds:.3f}s"
        )
        return 0

    engine = require_db_engine()
    try:
        write_started_at = time.perf_counter()
        with engine.begin() as conn:
            effective_chunksize = _resolve_sqlite_multi_chunksize(conn, len(df.columns), requested_chunksize=10000)
            conn.execute(
                delete(COST_ANOMALY_RESULT_RUNS_TABLE).where(
                    COST_ANOMALY_RESULT_RUNS_TABLE.c.result_mode == result_mode
                )
            )
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
                chunksize=effective_chunksize,
                method="multi",
            )
        write_seconds = time.perf_counter() - write_started_at
        total_seconds = time.perf_counter() - total_started_at
        print(
            f"[performance][存库阶段][{result_mode}] 记录数={len(df)} "
            f"清洗耗时={prepare_seconds:.3f}s 写入耗时={write_seconds:.3f}s 批大小={effective_chunksize} 总耗时={total_seconds:.3f}s"
        )
    except Exception as exc:
        for line in _build_cost_anomaly_write_failure_log_lines(result_df, df, result_mode, exc):
            print(line)
        raise
    return len(df)


def _empty_cost_anomaly_results_frame() -> pd.DataFrame:
    ui_columns = [
        ui_column
        for ui_column, db_column in _ANOMALY_RESULT_COLUMN_MAPPING.items()
        if db_column in _ANOMALY_EXPORT_COLUMNS and ui_column not in {"result_mode", "computed_at"}
    ]
    return pd.DataFrame(columns=ui_columns)


def load_cost_anomaly_results(result_mode: str = "raw") -> pd.DataFrame:
    _ensure_cost_anomaly_results_table()
    engine = require_db_engine()
    query = (
        select(*(COST_ANOMALY_RESULTS_TABLE.c[column] for column in _ANOMALY_EXPORT_COLUMNS))
        .where(COST_ANOMALY_RESULTS_TABLE.c.result_mode == result_mode)
    )
    df = pd.read_sql(query, engine)
    if df.empty:
        return _empty_cost_anomaly_results_frame()

    reverse_mapping = {db_column: ui_column for ui_column, db_column in _ANOMALY_RESULT_COLUMN_MAPPING.items()}
    df = df.rename(columns=reverse_mapping)
    if "_record_key" in df.columns:
        df["_record_key"] = df["_record_key"].apply(canonicalize_record_key).astype("string")
    if "价格有效于" in df.columns:
        df["价格有效于"] = pd.to_datetime(df["价格有效于"], errors="coerce")
    ordered_columns = [
        ui_column
        for ui_column, db_column in _ANOMALY_RESULT_COLUMN_MAPPING.items()
        if db_column in _ANOMALY_EXPORT_COLUMNS
        and ui_column in df.columns
        and ui_column not in {"result_mode", "computed_at"}
    ]
    return df[ordered_columns].copy()


def record_cost_anomaly_result_run(
    result_mode: str,
    *,
    source_signature: str,
    options_signature: str,
    row_count: int,
) -> dict[str, Any]:
    _ensure_cost_anomaly_result_runs_table()
    normalized_mode = str(result_mode or "raw").strip() or "raw"
    row = {
        "result_mode": normalized_mode,
        "source_signature": str(source_signature or "").strip(),
        "options_signature": str(options_signature or "").strip(),
        "row_count": int(row_count or 0),
        "computed_at": datetime.now(),
    }
    engine = require_db_engine()
    with engine.begin() as conn:
        statement = sqlite_insert(COST_ANOMALY_RESULT_RUNS_TABLE).values(row)
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=["result_mode"],
                set_={
                    "source_signature": statement.excluded.source_signature,
                    "options_signature": statement.excluded.options_signature,
                    "row_count": statement.excluded.row_count,
                    "computed_at": statement.excluded.computed_at,
                },
            )
        )
    return {**row, "computed_at": row["computed_at"].isoformat()}


def load_fresh_cost_anomaly_results(
    result_mode: str = "raw",
    *,
    source_signature: str,
    options_signature: str,
) -> pd.DataFrame:
    _ensure_cost_anomaly_results_table()
    _ensure_cost_anomaly_result_runs_table()
    normalized_mode = str(result_mode or "raw").strip() or "raw"
    engine = require_db_engine()
    with engine.connect() as conn:
        run_row = conn.execute(
            select(COST_ANOMALY_RESULT_RUNS_TABLE).where(
                COST_ANOMALY_RESULT_RUNS_TABLE.c.result_mode == normalized_mode
            )
        ).mappings().first()

    if not run_row:
        return _empty_cost_anomaly_results_frame()
    if str(run_row.get("source_signature") or "") != str(source_signature or "").strip():
        return _empty_cost_anomaly_results_frame()
    if str(run_row.get("options_signature") or "") != str(options_signature or "").strip():
        return _empty_cost_anomaly_results_frame()

    df = load_cost_anomaly_results(normalized_mode)
    if df.empty:
        return df
    if int(run_row.get("row_count") or 0) != int(len(df)):
        return _empty_cost_anomaly_results_frame()
    return df


def get_local_database_health(
    *,
    max_freelist_ratio: float = 0.25,
    max_wal_mb: float = 64.0,
) -> dict[str, Any]:
    db_path = settings.db_path
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": str(db_path),
            "failures": [f"missing db: {db_path}"],
        }

    engine = require_db_engine()
    with engine.connect() as conn:
        page_size = int(conn.exec_driver_sql("PRAGMA page_size").scalar_one())
        page_count = int(conn.exec_driver_sql("PRAGMA page_count").scalar_one())
        freelist_count = int(conn.exec_driver_sql("PRAGMA freelist_count").scalar_one())
        journal_mode = str(conn.exec_driver_sql("PRAGMA journal_mode").scalar_one())

    wal_path = db_path.with_name(db_path.name + "-wal")
    shm_path = db_path.with_name(db_path.name + "-shm")
    freelist_ratio = freelist_count / max(page_count, 1)
    wal_mb = round((wal_path.stat().st_size if wal_path.exists() else 0) / 1024 / 1024, 2)
    failures: list[str] = []
    if freelist_ratio > max_freelist_ratio:
        failures.append(f"freelist ratio {freelist_ratio:.2%} exceeds {max_freelist_ratio:.2%}")
    if wal_mb > max_wal_mb:
        failures.append(f"WAL {wal_mb} MB exceeds {max_wal_mb} MB")

    return {
        "ok": not failures,
        "db_path": str(db_path),
        "db_mb": round(db_path.stat().st_size / 1024 / 1024, 2),
        "wal_mb": wal_mb,
        "shm_mb": round((shm_path.stat().st_size if shm_path.exists() else 0) / 1024 / 1024, 2),
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "freelist_ratio": round(freelist_ratio, 4),
        "journal_mode": journal_mode,
        "thresholds": {
            "max_freelist_ratio": max_freelist_ratio,
            "max_wal_mb": max_wal_mb,
        },
        "failures": failures,
    }


def _ensure_vehicle_config_tables() -> None:
    engine = require_db_engine()
    VEHICLE_RANK_CONFIG_TABLE.create(engine, checkfirst=True)
    VEHICLE_MARKET_PRICES_TABLE.create(engine, checkfirst=True)
    actual_columns = _get_table_columns(engine, VEHICLE_MARKET_PRICES_TABLE.name)
    if actual_columns and "failure_reason" not in actual_columns:
        with engine.begin() as conn:
            conn.execute(text(f'ALTER TABLE "{VEHICLE_MARKET_PRICES_TABLE.name}" ADD COLUMN failure_reason TEXT'))


def _save_vehicle_rank_config_impl(rank_rows: Sequence[Dict[str, Any]]) -> int:
    _ensure_vehicle_config_tables()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    updated_at = datetime.now()
    for fallback_order, row in enumerate(rank_rows or [], start=1):
        vehicle_series = str(row.get("vehicle_series") or row.get("适用车系") or "").strip()
        if not vehicle_series or vehicle_series in seen:
            continue
        try:
            rank_order = int(row.get("rank_order") or row.get("梯度排名") or fallback_order)
        except (TypeError, ValueError):
            rank_order = fallback_order
        rows.append(
            {
                "vehicle_series": vehicle_series,
                "rank_order": rank_order,
                "source": str(row.get("source") or row.get("来源") or "manual")[:64],
                "updated_at": updated_at,
            }
        )
        seen.add(vehicle_series)

    with require_db_engine().begin() as conn:
        conn.execute(delete(VEHICLE_RANK_CONFIG_TABLE))
        if rows:
            conn.execute(VEHICLE_RANK_CONFIG_TABLE.insert(), rows)
    return len(rows)


def save_vehicle_rank_config(rank_rows: Sequence[Dict[str, Any]]) -> int:
    return harness.run_db_action(
        "write",
        "storage_service.save_vehicle_rank_config",
        lambda: _save_vehicle_rank_config_impl(rank_rows),
    )


def _load_vehicle_rank_config_impl() -> pd.DataFrame:
    _ensure_vehicle_config_tables()
    query = select(VEHICLE_RANK_CONFIG_TABLE).order_by(VEHICLE_RANK_CONFIG_TABLE.c.rank_order.asc())
    df = pd.read_sql(query, require_db_engine())
    if df.empty:
        return pd.DataFrame(columns=["vehicle_series", "rank_order", "source", "updated_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return df[["vehicle_series", "rank_order", "source", "updated_at"]].copy()


def load_vehicle_rank_config() -> pd.DataFrame:
    return harness.run_db_action(
        "read",
        "storage_service.load_vehicle_rank_config",
        _load_vehicle_rank_config_impl,
    )


def _save_vehicle_market_prices_impl(price_rows: Sequence[Dict[str, Any]]) -> int:
    _ensure_vehicle_config_tables()
    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now()
    for row in price_rows or []:
        vehicle_series = str(row.get("vehicle_series") or "").strip()
        if not vehicle_series:
            continue
        market_price = pd.to_numeric(pd.Series([row.get("market_price")]), errors="coerce").iloc[0]
        rows.append(
            {
                "vehicle_series": vehicle_series,
                "market_price": float(market_price) if pd.notna(market_price) else None,
                "variant_name": str(row.get("variant_name") or "").strip(),
                "source_url": str(row.get("source_url") or "").strip(),
                "source_domain": str(row.get("source_domain") or "").strip(),
                "status": str(row.get("status") or "待确认").strip() or "待确认",
                "fetched_at": pd.to_datetime(row.get("fetched_at"), errors="coerce").to_pydatetime()
                if pd.notna(pd.to_datetime(row.get("fetched_at"), errors="coerce"))
                else fetched_at,
                "failure_reason": str(row.get("failure_reason") or "").strip(),
                "raw_response_json": row.get("raw_response_json")
                if isinstance(row.get("raw_response_json"), str)
                else _json.dumps(row.get("raw_response_json") or row, ensure_ascii=False, default=str),
            }
        )

    if not rows:
        return 0

    with require_db_engine().begin() as conn:
        for row in rows:
            stmt = sqlite_insert(VEHICLE_MARKET_PRICES_TABLE).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["vehicle_series"],
                set_={column_name: row[column_name] for column_name in row if column_name != "vehicle_series"},
            )
            conn.execute(stmt)
    return len(rows)


def save_vehicle_market_prices(price_rows: Sequence[Dict[str, Any]]) -> int:
    return harness.run_db_action(
        "write",
        "storage_service.save_vehicle_market_prices",
        lambda: _save_vehicle_market_prices_impl(price_rows),
    )


def _load_vehicle_market_prices_impl() -> pd.DataFrame:
    _ensure_vehicle_config_tables()
    query = select(VEHICLE_MARKET_PRICES_TABLE).order_by(VEHICLE_MARKET_PRICES_TABLE.c.vehicle_series.asc())
    df = pd.read_sql(query, require_db_engine())
    if df.empty:
        return pd.DataFrame(
            columns=[
                "vehicle_series",
                "market_price",
                "variant_name",
                "source_url",
                "source_domain",
                "status",
                "fetched_at",
                "failure_reason",
                "raw_response_json",
            ]
        )
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    return df[
        [
            "vehicle_series",
            "market_price",
            "variant_name",
            "source_url",
            "source_domain",
            "status",
            "fetched_at",
            "failure_reason",
            "raw_response_json",
        ]
    ].copy()


def load_vehicle_market_prices() -> pd.DataFrame:
    return harness.run_db_action(
        "read",
        "storage_service.load_vehicle_market_prices",
        _load_vehicle_market_prices_impl,
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
                "ring_id": "圈层编号",
                "ring_role": "圈层角色",
                "ring_confidence": "圈层置信度",
                "ring_intervals_json": "多圈合理区间",
                "expert_adjusted": "专家校准",
                "decision_basis": "判定依据",
            }
        )
        .drop(columns=["result_mode"], errors="ignore")
    )


def _normalize_skill_domain(domain: Optional[str]) -> str:
    domain_text = str(domain or SKILL_DOMAIN_COST).strip().lower()
    if domain_text in {SKILL_DOMAIN_COST, SKILL_DOMAIN_SHEET_METAL}:
        return domain_text
    return SKILL_DOMAIN_COST


def _get_skill_bounds_mapping(domain: str) -> Tuple[str, str]:
    if domain == SKILL_DOMAIN_SHEET_METAL:
        return "白痴指数合理区间", "基准指数"
    return "成本合理区间边界", "预测值"


def _build_skills_index(skills_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for skill in skills_list:
        short_name = str(skill.get("备件简称", "") or "").strip()
        if short_name:
            index[short_name] = skill
    return index


def save_skills(
    skills: list,
    sigma: float = 1.0,
    weight: int = 80,
    domain: str = SKILL_DOMAIN_COST,
) -> str:
    _ensure_skills_storage_tables()
    normalized_domain = _normalize_skill_domain(domain)
    bounds_key, base_key = _get_skill_bounds_mapping(normalized_domain)
    snapshot_id = str(uuid4())
    saved_at = datetime.now()
    snapshot_rows = [
        {
            "snapshot_id": snapshot_id,
            "module_type": normalized_domain,
            "skill_domain": normalized_domain,
            "version": "1.0",
            "saved_at": saved_at,
            "global_sigma": round(float(sigma), 4),
            "global_weight": int(weight),
        }
    ]

    rows = []
    for skill in skills:
        bounds = skill.get(bounds_key, {}) or {}
        alignment = skill.get("经验对齐率")
        alignment_rate = float(alignment) if isinstance(alignment, (int, float)) else None
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "module_type": normalized_domain,
                "skill_domain": normalized_domain,
                "short_name": str(skill.get("备件简称", "")),
                "algorithm_type": str(skill.get("适用算法", "")),
                "sigma_param": float(skill.get("当前σ参数", sigma)) if skill.get("当前σ参数") is not None else None,
                "expert_weight": int(skill.get("偏置权重", weight)) if skill.get("偏置权重") is not None else None,
                "alignment_rate": alignment_rate,
                "lower_bound": float(bounds.get("合理下限")) if bounds.get("合理下限") is not None else None,
                "upper_bound": float(bounds.get("合理上限")) if bounds.get("合理上限") is not None else None,
                "base_price": float(bounds.get(base_key)) if bounds.get(base_key) is not None else None,
                "skill_payload_json": _json.dumps(skill, ensure_ascii=False, default=str),
            }
        )

    with Session(require_db_engine()) as session:
        with session.begin():
            _insert_rows(SKILLS_SNAPSHOTS_TABLE, snapshot_rows, session=session)
            _insert_rows(SKILLS_ITEMS_TABLE, rows, session=session)
    log_event(
        "skills",
        "save_snapshot",
        "Saved skills snapshot into SQLite",
        snapshot_id=snapshot_id,
        module_type=normalized_domain,
        skills_count=len(rows),
        sigma=round(float(sigma), 4),
        weight=int(weight),
    )
    return "skills_snapshots / skills_items"


def load_skills(domain: str = SKILL_DOMAIN_COST) -> Optional[Dict]:
    if DB_ENGINE is None:
        return None
    _ensure_skills_storage_tables()
    normalized_domain = _normalize_skill_domain(domain)
    bounds_key, base_key = _get_skill_bounds_mapping(normalized_domain)
    try:
        engine = require_db_engine()
        with engine.connect() as conn:
            snapshot = conn.execute(
                select(SKILLS_SNAPSHOTS_TABLE)
                .where(SKILLS_SNAPSHOTS_TABLE.c.module_type == normalized_domain)
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
            .where(SKILLS_ITEMS_TABLE.c.module_type == normalized_domain)
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
            if normalized_domain == SKILL_DOMAIN_SHEET_METAL:
                if not isinstance(skill.get("白痴指数分布描述"), dict):
                    skill["白痴指数分布描述"] = {}
            elif not isinstance(skill.get("数据结构分布描述"), dict):
                skill["数据结构分布描述"] = {}
            if not isinstance(skill.get("异常统计"), dict):
                skill["异常统计"] = {}

            bounds = skill.get(bounds_key)
            if not isinstance(bounds, dict):
                bounds = {}
            skill[bounds_key] = {
                base_key: row.get("base_price") if row.get("base_price") is not None else bounds.get(base_key, 0.0),
                "合理下限": row.get("lower_bound") if row.get("lower_bound") is not None else bounds.get("合理下限", 0.0),
                "合理上限": row.get("upper_bound") if row.get("upper_bound") is not None else bounds.get("合理上限", 0.0),
            }

            if normalized_domain == SKILL_DOMAIN_COST:
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
        return {
            "snapshot_id": snapshot["snapshot_id"],
            "module_type": normalized_domain,
            "skill_domain": normalized_domain,
            "version": snapshot["version"],
            "saved_at": snapshot["saved_at"].isoformat() if snapshot["saved_at"] else None,
            "global_sigma": snapshot["global_sigma"],
            "global_weight": snapshot["global_weight"],
            "skills": skills_list,
            "index": _build_skills_index(skills_list),
        }
    except Exception:
        return None


def has_skills_snapshot(domain: str = SKILL_DOMAIN_COST) -> bool:
    if DB_ENGINE is None:
        return False
    _ensure_skills_storage_tables()
    normalized_domain = _normalize_skill_domain(domain)
    try:
        engine = require_db_engine()
        with engine.connect() as conn:
            snapshot_count = conn.execute(
                select(func.count())
                .select_from(SKILLS_SNAPSHOTS_TABLE)
                .where(SKILLS_SNAPSHOTS_TABLE.c.module_type == normalized_domain)
            ).scalar_one()
        return int(snapshot_count) > 0
    except Exception:
        return False


def delete_skills(domain: str = SKILL_DOMAIN_COST) -> Dict[str, int]:
    if DB_ENGINE is None:
        return {"snapshots": 0, "skills": 0}
    _ensure_skills_storage_tables()
    normalized_domain = _normalize_skill_domain(domain)
    engine = require_db_engine()
    with engine.begin() as conn:
        snapshot_ids = conn.execute(
            select(SKILLS_SNAPSHOTS_TABLE.c.snapshot_id)
            .where(SKILLS_SNAPSHOTS_TABLE.c.module_type == normalized_domain)
            .order_by(SKILLS_SNAPSHOTS_TABLE.c.saved_at.desc())
        ).scalars().all()
        if not snapshot_ids:
            return {"snapshots": 0, "skills": 0}

        skill_count = conn.execute(
            select(func.count())
            .select_from(SKILLS_ITEMS_TABLE)
            .where(SKILLS_ITEMS_TABLE.c.snapshot_id.in_(snapshot_ids))
        ).scalar_one()

        conn.execute(delete(SKILLS_ITEMS_TABLE).where(SKILLS_ITEMS_TABLE.c.snapshot_id.in_(snapshot_ids)))
        conn.execute(delete(SKILLS_SNAPSHOTS_TABLE).where(SKILLS_SNAPSHOTS_TABLE.c.snapshot_id.in_(snapshot_ids)))

    deleted = {"snapshots": len(snapshot_ids), "skills": int(skill_count)}
    log_event(
        "skills",
        "delete_snapshot",
        "Deleted skills snapshots from SQLite",
        module_type=normalized_domain,
        deleted_snapshots=deleted["snapshots"],
        deleted_skills=deleted["skills"],
    )
    return deleted


def get_latest_core_cost_lookup(material_codes: Sequence[str]) -> pd.DataFrame:
    normalized_codes = sorted(
        {
            str(material_code or "").strip()
            for material_code in material_codes
            if str(material_code or "").strip()
        }
    )
    engine = require_db_engine()
    available_columns = _get_table_columns(engine, CORE_COST_RECORDS_TABLE.name)
    if not normalized_codes or DB_ENGINE is None or not available_columns:
        return pd.DataFrame(columns=["material_code", "cost_amount", "monitor_date", "material_name", "vehicle_series", "factory"])

    normalized_available_columns = {column_name.lower(): column_name for column_name in available_columns}

    def _resolve_optional_column(alias_candidates: Sequence[str]) -> str | None:
        for alias in alias_candidates:
            matched_column = normalized_available_columns.get(str(alias).lower())
            if matched_column:
                return matched_column
        return None

    select_sql_columns = [
        '"material_code"',
        '"cost_amount"',
        '"monitor_date"',
        '"material_name"',
        '"vehicle_series"',
        '"factory"',
        '"cost_record_id"',
    ]
    optional_column_aliases = {
        "resource_developer": ["resource_developer", "resource_dev", "resource_owner", "资源开发", "开发", "开发负责人"],
        "pricing_owner": ["pricing_owner", "pricing_manager", "owner", "定价负责人", "负责人", "采购负责人"],
        "order_price": ["order_price", "purchase_price", "order_amount", "订购价", "订货价", "采购价"],
        "retail_price": ["retail_price", "sale_price", "list_price", "零售价", "销售价"],
    }
    resolved_optional_columns: list[str] = []
    for normalized_name, alias_candidates in optional_column_aliases.items():
        matched_column = _resolve_optional_column(alias_candidates)
        if matched_column:
            select_sql_columns.append(f'"{matched_column}" AS "{normalized_name}"')
            resolved_optional_columns.append(normalized_name)

    lookup_columns = [
        "material_code",
        "cost_amount",
        "monitor_date",
        "material_name",
        "vehicle_series",
        "factory",
        *resolved_optional_columns,
    ]

    frames: List[pd.DataFrame] = []
    chunk_size = 800
    for index in range(0, len(normalized_codes), chunk_size):
        code_chunk = normalized_codes[index : index + chunk_size]
        params = {f"code_{offset}": material_code for offset, material_code in enumerate(code_chunk)}
        placeholders = ", ".join(f":code_{offset}" for offset in range(len(code_chunk)))
        query = text(
            f'SELECT {", ".join(select_sql_columns)} '
            f'FROM "{CORE_COST_RECORDS_TABLE.name}" '
            f'WHERE "material_code" IN ({placeholders})'
        )
        chunk_df = pd.read_sql(query, engine, params=params)
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        return pd.DataFrame(columns=lookup_columns)

    lookup_df = pd.concat(frames, ignore_index=True)
    lookup_df["monitor_date"] = pd.to_datetime(lookup_df["monitor_date"], errors="coerce")

    vehicle_series_candidates = lookup_df[["material_code", "vehicle_series", "monitor_date", "cost_record_id"]].copy()
    vehicle_series_candidates["vehicle_series"] = (
        vehicle_series_candidates["vehicle_series"].fillna("").astype(str).str.strip()
    )
    vehicle_series_candidates = vehicle_series_candidates[vehicle_series_candidates["vehicle_series"].ne("")]
    if not vehicle_series_candidates.empty:
        preferred_vehicle_series = (
            vehicle_series_candidates.groupby(["material_code", "vehicle_series"], sort=False, dropna=False)
            .agg(
                record_count=("vehicle_series", "size"),
                latest_monitor_date=("monitor_date", "max"),
                latest_cost_record_id=("cost_record_id", "max"),
            )
            .reset_index()
            .sort_values(
                ["material_code", "record_count", "latest_monitor_date", "latest_cost_record_id", "vehicle_series"],
                ascending=[True, False, False, False, True],
                na_position="last",
            )
            .drop_duplicates(subset=["material_code"], keep="first")
            [["material_code", "vehicle_series"]]
            .rename(columns={"vehicle_series": "preferred_vehicle_series"})
        )
    else:
        preferred_vehicle_series = pd.DataFrame(columns=["material_code", "preferred_vehicle_series"])

    lookup_df = lookup_df.sort_values(
        ["material_code", "monitor_date", "cost_record_id"],
        ascending=[True, False, False],
        na_position="last",
    )
    lookup_df = lookup_df.drop_duplicates(subset=["material_code"], keep="first")
    lookup_df = lookup_df.merge(preferred_vehicle_series, on="material_code", how="left")
    lookup_df["vehicle_series"] = lookup_df["preferred_vehicle_series"].combine_first(lookup_df["vehicle_series"])
    lookup_df = lookup_df.drop(columns=["preferred_vehicle_series"])
    return lookup_df[lookup_columns].reset_index(drop=True)


def _ensure_database_columns() -> None:
    _ensure_core_cost_records_business_key_index()
    _ensure_expert_feedback_columns()
    _ensure_expert_knowledge_base_columns()
    _ensure_skills_storage_tables()


def _ensure_feedback_table_columns(table: Table) -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, table.name)
    if not actual_columns:
        table.create(engine, checkfirst=True)
        return
    if "remark" not in actual_columns:
        with engine.begin() as conn:
            conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN remark TEXT'))


def _ensure_expert_feedback_columns() -> None:
    _ensure_feedback_table_columns(EXPERT_FEEDBACK_TABLE)
    _ensure_feedback_table_columns(SHEET_METAL_FEEDBACK_TABLE)


def _ensure_expert_knowledge_base_columns() -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, EXPERT_KNOWLEDGE_BASE_TABLE.name)
    if not actual_columns:
        EXPERT_KNOWLEDGE_BASE_TABLE.create(engine, checkfirst=True)
        return

    expected_columns = {
        "rule_id": "VARCHAR(64)",
        "short_name": "VARCHAR(128)",
        "material_code": "VARCHAR(64)",
        "material_name": "VARCHAR(255)",
        "supplier_code": "VARCHAR(64)",
        "supplier_name": "VARCHAR(255)",
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
            conn.execute(text(f"ALTER TABLE expert_knowledge_base ADD COLUMN {column_name} {ddl}"))


def _ensure_skills_snapshots_columns() -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, SKILLS_SNAPSHOTS_TABLE.name)
    if not actual_columns:
        SKILLS_SNAPSHOTS_TABLE.create(engine, checkfirst=True)
        return

    expected_columns = {
        "module_type": "TEXT NOT NULL DEFAULT 'cost'",
        "skill_domain": "TEXT NOT NULL DEFAULT 'cost'",
    }
    missing_columns = [column_name for column_name in expected_columns if column_name not in actual_columns]
    for column_name in missing_columns:
        ddl = expected_columns[column_name]
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {SKILLS_SNAPSHOTS_TABLE.name} ADD COLUMN {column_name} {ddl}"))

    if missing_columns or "module_type" in actual_columns or "skill_domain" in actual_columns:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"UPDATE {SKILLS_SNAPSHOTS_TABLE.name} "
                    "SET module_type = COALESCE(NULLIF(TRIM(skill_domain), ''), 'cost') "
                    "WHERE module_type IS NULL OR TRIM(module_type) = ''"
                )
            )
            conn.execute(
                text(
                    f"UPDATE {SKILLS_SNAPSHOTS_TABLE.name} "
                    "SET skill_domain = COALESCE(NULLIF(TRIM(module_type), ''), 'cost') "
                    "WHERE skill_domain IS NULL OR TRIM(skill_domain) = ''"
                )
            )


def _ensure_skills_items_columns() -> None:
    engine = require_db_engine()
    actual_columns = _get_table_columns(engine, SKILLS_ITEMS_TABLE.name)
    if not actual_columns:
        SKILLS_ITEMS_TABLE.create(engine, checkfirst=True)
        return

    expected_columns = {
        "skill_payload_json": "TEXT",
        "module_type": "TEXT NOT NULL DEFAULT 'cost'",
        "skill_domain": "TEXT NOT NULL DEFAULT 'cost'",
    }
    missing_columns = [column_name for column_name in expected_columns if column_name not in actual_columns]
    for column_name in missing_columns:
        ddl = expected_columns[column_name]
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {SKILLS_ITEMS_TABLE.name} ADD COLUMN {column_name} {ddl}"))

    if missing_columns or "module_type" in actual_columns or "skill_domain" in actual_columns:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"UPDATE {SKILLS_ITEMS_TABLE.name} "
                    "SET module_type = COALESCE(NULLIF(TRIM(skill_domain), ''), 'cost') "
                    "WHERE module_type IS NULL OR TRIM(module_type) = ''"
                )
            )
            conn.execute(
                text(
                    f"UPDATE {SKILLS_ITEMS_TABLE.name} "
                    "SET skill_domain = COALESCE(NULLIF(TRIM(module_type), ''), 'cost') "
                    "WHERE skill_domain IS NULL OR TRIM(skill_domain) = ''"
                )
            )


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
        conn.execute(text("DROP TABLE IF EXISTS cost_anomaly_result_runs"))
        conn.execute(text(_COST_ANOMALY_RESULTS_CREATE_SQL))
    print("[cost_anomaly_results] 已按纯英文字段重置表结构")


def _ensure_cost_anomaly_result_runs_table() -> None:
    engine = require_db_engine()
    COST_ANOMALY_RESULT_RUNS_TABLE.create(engine, checkfirst=True)


def _ensure_cost_anomaly_results_table() -> None:
    global _COST_ANOMALY_RESULTS_RESET_DONE

    if _COST_ANOMALY_RESULTS_RESET_DONE:
        return

    engine = require_db_engine()
    actual_columns = _get_cost_anomaly_results_columns(engine)
    reset_requested = settings.reset_cost_anomaly_results_on_start

    if not actual_columns:
        with engine.begin() as conn:
            conn.execute(text(_COST_ANOMALY_RESULTS_CREATE_SQL))
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
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))


def _reset_table(table: Table, *, recreate: bool = True, cascade: bool = False) -> None:
    engine = require_db_engine()
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
        params={"snapshot_id": str(legacy_snapshot["snapshot_id"])} ,
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
                    "module_type": SKILL_DOMAIN_COST,
                    "skill_domain": SKILL_DOMAIN_COST,
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
    normalized_items["module_type"] = SKILL_DOMAIN_COST
    normalized_items["skill_domain"] = SKILL_DOMAIN_COST
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
                "module_type",
                "skill_domain",
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

    if actual_snapshots_columns:
        _ensure_skills_snapshots_columns()
        actual_snapshots_columns = _get_table_columns(engine, SKILLS_SNAPSHOTS_TABLE.name)

    if actual_items_columns and set(actual_items_columns).issubset(set(expected_items_columns)):
        _ensure_skills_items_columns()
        actual_items_columns = _get_table_columns(engine, SKILLS_ITEMS_TABLE.name)

    snapshots_mismatch = bool(actual_snapshots_columns) and set(actual_snapshots_columns) != set(expected_snapshots_columns)
    items_mismatch = bool(actual_items_columns) and set(actual_items_columns) != set(expected_items_columns)

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


def initialize_local_storage() -> None:
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
        print(f"Local SQLite storage ready: {settings.db_url}")
        _drop_legacy_skills_tables(engine)
        log_event(
            "storage",
            "initialized",
            "Initialized local SQLite storage",
            db_path=str(settings.db_path.name),
        )
        _DB_INIT_ERROR = None
    except Exception as exc:
        _DB_INIT_ERROR = exc


def vacuum_local_database() -> Dict[str, Any]:
    db_path = settings.db_path
    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    before_bytes = int(db_path.stat().st_size)
    wal_path = db_path.with_name(db_path.name + "-wal")
    before_wal_bytes = int(wal_path.stat().st_size) if wal_path.exists() else 0
    require_db_engine().dispose()
    with harness.managed_sqlite_connection(
        db_path,
        operation="write",
        source="storage_service.vacuum_local_database",
        timeout=30,
        isolation_level=None,
    ) as conn:
        checkpoint_before = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        conn.execute("VACUUM")
        checkpoint_after = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    after_bytes = int(db_path.stat().st_size)
    after_wal_bytes = int(wal_path.stat().st_size) if wal_path.exists() else 0
    result = {
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "saved_bytes": max(before_bytes - after_bytes, 0),
        "before_wal_bytes": before_wal_bytes,
        "after_wal_bytes": after_wal_bytes,
        "checkpoint_before": tuple(checkpoint_before or (0, 0, 0)),
        "checkpoint_after": tuple(checkpoint_after or (0, 0, 0)),
        "db_path": str(db_path),
    }
    log_event(
        "database",
        "vacuum",
        "Compacted local SQLite database",
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        saved_bytes=result["saved_bytes"],
        db_path=str(db_path.name),
    )
    return result


label_manager = LabelManager()
sheet_metal_label_manager = LabelManager(SHEET_METAL_FEEDBACK_TABLE.name)


class CostMonitoringService:
    def initialize_storage(self) -> None:
        harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.initialize_storage",
            initialize_local_storage,
        )

    def sync_core_cost_records(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None,
        mode: str = "incremental",
    ) -> int:
        from data_ingestion import persist_core_cost_records

        return harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.sync_core_cost_records",
            lambda: persist_core_cost_records(df, price_col=price_col, mode=mode),
        )

    def load_core_cost_records(self) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        from data_ingestion import load_core_cost_records

        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.load_core_cost_records",
            load_core_cost_records,
        )

    def get_core_cost_records_status(self) -> Dict[str, Any]:
        from data_ingestion import get_core_cost_records_status

        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.get_core_cost_records_status",
            get_core_cost_records_status,
        )

    def get_latest_core_cost_lookup(self, material_codes: Sequence[str]) -> pd.DataFrame:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.get_latest_core_cost_lookup",
            lambda: get_latest_core_cost_lookup(material_codes),
        )

    def get_feedback_details(self) -> Dict[str, Dict[str, str]]:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.get_feedback_details",
            label_manager.get_labels,
        )

    def get_feedback_statuses(self) -> Dict[str, str]:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.get_feedback_statuses",
            label_manager.get_label_statuses,
        )

    def replace_feedback(self, final_labels_df: pd.DataFrame) -> None:
        harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.replace_feedback",
            lambda: label_manager.replace_all(final_labels_df),
        )

    def delete_feedback(self, keys_to_remove) -> int:
        return harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.delete_feedback",
            lambda: label_manager.delete_labels(keys_to_remove),
        )

    def clear_feedback(self) -> None:
        harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.clear_feedback",
            label_manager.clear_all,
        )

    def get_feedback_row_count(self) -> int:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.get_feedback_row_count",
            label_manager.file_row_count,
        )

    def load_skills_snapshot(self, domain: str = SKILL_DOMAIN_COST) -> Optional[Dict]:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.load_skills_snapshot",
            lambda: load_skills(domain=domain),
        )

    def save_skills_snapshot(
        self,
        skills: list,
        sigma: float = 1.0,
        weight: int = 80,
        domain: str = SKILL_DOMAIN_COST,
    ) -> str:
        return harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.save_skills_snapshot",
            lambda: save_skills(skills, sigma=sigma, weight=weight, domain=domain),
        )

    def has_skills_snapshot(self, domain: str = SKILL_DOMAIN_COST) -> bool:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.has_skills_snapshot",
            lambda: has_skills_snapshot(domain=domain),
        )

    def delete_skills_snapshot(self, domain: str = SKILL_DOMAIN_COST) -> Dict[str, int]:
        return harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.delete_skills_snapshot",
            lambda: delete_skills(domain=domain),
        )

    def compact_local_database(self) -> Dict[str, Any]:
        return harness.run_db_action(
            "write",
            "storage_service.CostMonitoringService.compact_local_database",
            vacuum_local_database,
        )

    def get_local_database_health(self) -> Dict[str, Any]:
        return harness.run_db_action(
            "read",
            "storage_service.CostMonitoringService.get_local_database_health",
            get_local_database_health,
        )


service = CostMonitoringService()
