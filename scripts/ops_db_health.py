from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


REQUIRED_TABLES = {
    "core_cost_records",
    "cost_anomaly_results",
    "expert_feedback",
    "expert_knowledge_base",
    "skills_items",
    "skills_snapshots",
}


def bytes_to_mb(value: int | float) -> float:
    return round(float(value or 0) / 1024 / 1024, 2)


def _read_table_counts(cursor: sqlite3.Cursor, tables: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name in tables:
        counts[table_name] = int(cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])
    return counts


def collect_db_health(
    db_path: str | Path,
    *,
    max_freelist_ratio: float = 0.25,
    max_wal_mb: float = 64.0,
) -> dict[str, Any]:
    resolved_db_path = Path(db_path).resolve()
    if not resolved_db_path.exists():
        return {
            "ok": False,
            "error": f"missing db: {resolved_db_path}",
            "db_path": str(resolved_db_path),
            "failures": [f"missing db: {resolved_db_path}"],
        }

    connection = sqlite3.connect(f"file:{resolved_db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        cursor = connection.cursor()
        page_size = int(cursor.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(cursor.execute("PRAGMA page_count").fetchone()[0])
        freelist_count = int(cursor.execute("PRAGMA freelist_count").fetchone()[0])
        journal_mode = str(cursor.execute("PRAGMA journal_mode").fetchone()[0])
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        table_counts = _read_table_counts(cursor, tables)
    finally:
        connection.close()

    wal_path = resolved_db_path.with_name(resolved_db_path.name + "-wal")
    shm_path = resolved_db_path.with_name(resolved_db_path.name + "-shm")
    freelist_ratio = freelist_count / max(page_count, 1)
    missing_tables = sorted(REQUIRED_TABLES - set(tables))
    payload: dict[str, Any] = {
        "ok": True,
        "db_path": str(resolved_db_path),
        "db_mb": bytes_to_mb(resolved_db_path.stat().st_size),
        "wal_mb": bytes_to_mb(wal_path.stat().st_size) if wal_path.exists() else 0.0,
        "shm_mb": bytes_to_mb(shm_path.stat().st_size) if shm_path.exists() else 0.0,
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "freelist_mb": bytes_to_mb(freelist_count * page_size),
        "freelist_ratio": round(freelist_ratio, 4),
        "journal_mode": journal_mode,
        "table_counts": table_counts,
        "missing_tables": missing_tables,
        "thresholds": {
            "max_freelist_ratio": max_freelist_ratio,
            "max_wal_mb": max_wal_mb,
        },
    }
    failures: list[str] = []
    if missing_tables:
        failures.append(f"missing tables: {', '.join(missing_tables)}")
    if freelist_ratio > max_freelist_ratio:
        failures.append(f"freelist ratio {freelist_ratio:.2%} exceeds {max_freelist_ratio:.2%}")
    if payload["wal_mb"] > max_wal_mb:
        failures.append(f"WAL {payload['wal_mb']} MB exceeds {max_wal_mb} MB")
    payload["ok"] = not failures
    payload["failures"] = failures
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="cost_monitor_data.db")
    parser.add_argument("--max-freelist-ratio", type=float, default=0.25)
    parser.add_argument("--max-wal-mb", type=float, default=64.0)
    args = parser.parse_args(argv)

    payload = collect_db_health(
        args.db,
        max_freelist_ratio=args.max_freelist_ratio,
        max_wal_mb=args.max_wal_mb,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
