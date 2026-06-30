# Backend API Contract

This contract is the migration boundary for the future Rust backend. The current Streamlit frontend remains the production UI until the backend boundary is stable. The later desktop target is Tauri + React/Vite, using the same command shapes.

## Goals

- Keep heavy compute and SQLite maintenance behind explicit commands.
- Make every command deterministic and testable from a CLI before wiring Tauri.
- Return JSON payloads only; no business rows should be printed to logs on failure.
- Preserve the existing SQLite file as the source of truth during migration.

## Commands

### `health`

Purpose: verify the backend binary starts and can report its contract version.

Request:

```json
{}
```

Response:

```json
{
  "ok": true,
  "service": "cost-monitor-backend",
  "contract_version": "0.1.0"
}
```

### `db-health`

Purpose: read-only SQLite health check equivalent to `scripts/ops_db_health.py`.

Request:

```json
{
  "db_path": "cost_monitor_data.db",
  "max_freelist_ratio": 0.25,
  "max_wal_mb": 64.0
}
```

Response: JSON fields must match the Python health script keys: `ok`, `db_mb`, `wal_mb`, `page_count`, `freelist_count`, `freelist_ratio`, `table_counts`, and `failures`.

### `compute-cost-anomaly`

Purpose: future Rust implementation of the raw cost anomaly slice.

Request:

```json
{
  "db_path": "cost_monitor_data.db",
  "result_mode": "raw",
  "source_signature": "sha256",
  "options_signature": "sha256"
}
```

Response:

```json
{
  "ok": true,
  "result_mode": "raw",
  "computed_rows": 1170,
  "persisted_rows": 1170,
  "elapsed_ms": 120
}
```

## Migration Order

1. Keep Python as orchestrator and call the Rust CLI for `health`.
2. Port `db-health` and compare output with `ops_db_health.py`.
3. Port read-only source loading and result metadata checks.
4. Port raw anomaly compute.
5. Port weighted compute after parity tests cover labels and parameter signatures.

## Acceptance

- Python and Rust health checks return compatible JSON.
- DB health values match Python within formatting tolerance.
- Raw/weighted parity tests compare row counts, record keys, statuses, and numeric columns within agreed tolerance.
- Rust commands do not mutate data unless the command name is explicitly write/maintenance oriented.
