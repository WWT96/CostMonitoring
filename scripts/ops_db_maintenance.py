from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def bytes_to_mb(value: int | float) -> float:
    return round(float(value or 0) / 1024 / 1024, 2)


def collect_database_stats(db_path: str | Path) -> dict[str, Any]:
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"missing db: {resolved}")

    conn = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        cursor = conn.cursor()
        page_size = int(cursor.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(cursor.execute("PRAGMA page_count").fetchone()[0])
        freelist_count = int(cursor.execute("PRAGMA freelist_count").fetchone()[0])
        journal_mode = str(cursor.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        conn.close()

    wal_path = resolved.with_name(resolved.name + "-wal")
    shm_path = resolved.with_name(resolved.name + "-shm")
    return {
        "db_path": str(resolved),
        "db_bytes": int(resolved.stat().st_size),
        "db_mb": bytes_to_mb(resolved.stat().st_size),
        "wal_bytes": int(wal_path.stat().st_size) if wal_path.exists() else 0,
        "wal_mb": bytes_to_mb(wal_path.stat().st_size) if wal_path.exists() else 0.0,
        "shm_bytes": int(shm_path.stat().st_size) if shm_path.exists() else 0,
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "freelist_ratio": round(freelist_count / max(page_count, 1), 4),
        "journal_mode": journal_mode,
    }


def checkpoint_database(db_path: str | Path) -> dict[str, Any]:
    resolved = Path(db_path).resolve()
    conn = sqlite3.connect(resolved, timeout=30, isolation_level=None)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()

    busy, log_pages, checkpointed_pages = row if row is not None else (0, 0, 0)
    return {
        "busy": int(busy or 0),
        "log_pages": int(log_pages or 0),
        "checkpointed_pages": int(checkpointed_pages or 0),
    }


def vacuum_database(db_path: str | Path) -> None:
    resolved = Path(db_path).resolve()
    conn = sqlite3.connect(resolved, timeout=30, isolation_level=None)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()


def run_database_maintenance(db_path: str | Path, *, vacuum: bool = False) -> dict[str, Any]:
    before = collect_database_stats(db_path)
    operations = ["checkpoint"]
    checkpoint_before = checkpoint_database(db_path)

    if vacuum:
        operations.append("vacuum")
        vacuum_database(db_path)
        operations.append("checkpoint_after_vacuum")
        checkpoint_after = checkpoint_database(db_path)
    else:
        checkpoint_after = None

    after = collect_database_stats(db_path)
    return {
        "ok": checkpoint_before["busy"] == 0 and (checkpoint_after is None or checkpoint_after["busy"] == 0),
        "vacuum": bool(vacuum),
        "operations": operations,
        "before": before,
        "after": after,
        "saved_bytes": max(int(before["db_bytes"]) - int(after["db_bytes"]), 0),
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="cost_monitor_data.db")
    parser.add_argument("--vacuum", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = run_database_maintenance(args.db, vacuum=bool(args.vacuum))
    except Exception as exc:
        payload = {
            "ok": False,
            "db_path": str(Path(args.db).resolve()),
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
