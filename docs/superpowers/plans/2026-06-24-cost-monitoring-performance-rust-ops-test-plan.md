# Cost Monitoring Performance Rust Refactor and Operational Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a performance/root-cause baseline, decide the safe Rust backend migration boundary, and define operational test scripts plus acceptance criteria before any code refactor.

**Architecture:** Treat the current application as a Streamlit + pandas/scikit-learn + SQLite desktop-style monolith. Stabilize rerun/cache/database behavior first, then move compute/storage-heavy backend functions behind a narrow service boundary that can be reimplemented in Rust without forcing an immediate frontend rewrite.

**Tech Stack:** Current: Python, Streamlit, pandas, scikit-learn, SQLAlchemy, SQLite WAL. Candidate backend: Rust, SQLite, vectorized dataframe/statistics crates, optional local service/CLI. Candidate frontend: keep Streamlit short term; evaluate Tauri + React/Vite for a low-spec desktop rewrite.

---

## 1. Current Findings

### 1.1 Performance root causes

1. Streamlit rerun model is amplifying work.
   - `app.py` calls `bootstrap_app()` on every run.
   - `app_context.py:819-824` calls runtime governance and storage bootstrap before page rendering.
   - `harness.py:1086-1094` runs `audit_system_integrity()` during bootstrap.
   - `harness.py:919-1050` reads blueprint JSON, reads source files, parses AST, hashes sources, and evaluates rules.
   - Local timing: `audit_system_integrity()` itself took about `0.1831s`; full import + audit command took about `1.27s`.

2. Heavy computations are tied to page rendering.
   - `cost_monitor_ui.py:371-377` computes raw anomaly results while rendering the anomaly page.
   - `cost_monitor_ui.py:408-418` computes weighted anomaly results while rendering expert mode.
   - `compute_jobs.py:54-72` computes first and only then checks persisted results, so a cache miss still does full compute even when the DB already has a persisted result.

3. SQLite write amplification is severe.
   - `anomaly_engine.py:1374-1380` and `anomaly_engine.py:1464-1470` save anomaly results after compute.
   - `storage_service.py:1545-1586` deletes all rows for the result mode and appends the entire result set with `to_sql`.
   - Runtime log evidence:
     - `13,380` raw rows: compute `4.443s`, DB write `4.155s`.
     - `13,380` weighted rows: compute `5.377s`, DB write `4.229s`.
     - `10,000` weighted rows: compute `5.020s`, DB write `3.740s`.
     - Current `1,170` rows: raw compute `0.371s`, raw write `0.350s`; weighted compute `0.375s`, weighted write `0.342s`.

4. Database bloat matches the "gets slower over time" symptom.
   - Current DB files observed:
     - `cost_monitor_data.db`: about `99.87 MB`
     - `cost_monitor_data.db-wal`: about `16.94 MB`
   - SQLite PRAGMA snapshot after testing/restoring current derived results:
     - `page_size=4096`
     - `page_count=24382`
     - `freelist_count=22298`
   - Approximate free pages: `22298 * 4096 = 91.33 MB`, meaning most DB pages are deleted/reusable pages not returned to the filesystem.
   - SQLite documents `VACUUM` as rebuilding the database into a minimal file; SQLite WAL mode also keeps a separate `-wal` file while connections are open.

5. `st.cache_data` is useful but currently risky.
   - `app_context.py:567-633`, `assembly_ui.py:244-251`, and `sheet_metal_ui.py:52-74` cache DataFrame-heavy functions.
   - Streamlit documents `st.cache_data` as storing cached data and returning copies; its `max_entries` default is unbounded unless explicitly set.
   - Several cached functions include full DataFrames plus random refresh tokens in their cache keys, so repeated refreshes can accumulate large serialized entries.

6. UI rendering does expensive work even before explicit export.
   - `general_pages.py:578-584`, `cost_monitor_ui.py:607-618`, `sheet_metal_ui.py:552-562`, and `assembly_ui.py:789-807` build Excel bytes during page render for download buttons.
   - `page_ui_helpers.py:76-84` scans unique values for filter candidates.
   - `page_ui_helpers.py:124-131` copies/filter-combines text across rows for keyword search.

7. Parallel compute may hurt low-spec environments.
   - `anomaly_engine.py:1278-1307` uses `ProcessPoolExecutor` when thresholds are met.
   - On the observed machine it used up to `16` workers for medium data. On low-core/low-RAM machines, process startup and DataFrame pickling can dominate compute.

### 1.2 Current test baseline

- Command run: `python -m unittest tests.test_dgb_multiring`
- Result: `73` tests passed in about `2.857s`.
- Important isolation finding: the current tests call anomaly persistence paths and can rewrite the live `cost_anomaly_results` cache table. After noticing this, the derived raw/weighted cache was regenerated from the current `core_cost_records` table, resulting in `1170` raw rows and `1170` weighted rows.
- Acceptance requirement: future operational tests must use a copied database or explicit test database path before invoking anomaly functions.

---

## 2. Rust Backend Feasibility

### 2.1 Feasibility verdict

Rust backend refactor is feasible, especially for low-spec deployment, but only if the boundary is the backend service layer rather than a full rewrite on day one.

Best first Rust candidates:

- SQLite read/write layer, especially result caching, checkpointing, and incremental updates.
- Anomaly compute kernels currently centered in `anomaly_engine.py`.
- File/cache indexing and deterministic data preparation currently split across `data_ingestion.py`, `storage_service.py`, and `app_context.py`.
- Long-running AutoResearch loops currently in `skills_engine.py`.

Poor first Rust candidates:

- Streamlit UI code.
- Excel formatting/export parity.
- LLM prompt/response orchestration.
- Legacy record-key migration until there is exhaustive parity coverage.

### 2.2 Why Rust can help

- Rust gives predictable memory ownership and thread-safety guarantees, which is useful for long-running local tools.
- Rust can remove Python process-pool pickling overhead by using threads safely for CPU-bound native code.
- A Rust binary can be packaged smaller than the current Python + wheels distribution.
- Tauri's official architecture is Rust backend plus system WebView UI, which fits a low-spec desktop target.

### 2.3 Why Rust alone will not fix the current issue

The main root cause is architectural work amplification:

- Streamlit reruns recompute too much.
- Cache misses compute before checking persisted results.
- SQLite is rewritten wholesale for result sets.
- Export bytes and table filters are built during render.
- Dev governance checks run during ordinary app bootstrap.

If those behaviors are preserved, rewriting the compute function in Rust will make the compute slice faster but still leave rerun, DB bloat, and UI render costs.

### 2.4 Recommended Rust migration path

1. Stabilize Python first.
   - Gate harness integrity audit behind development mode.
   - Check persisted results before computing.
   - Add bounded cache settings.
   - Generate export bytes only on explicit export action.
   - Add DB health checks and VACUUM/checkpoint guidance.

2. Add a backend API boundary.
   - Define requests/responses for import, anomaly compute, skills extraction, DB health, and export.
   - Keep response payloads typed and table-shaped.
   - Use golden-output parity tests from the current Python implementation.

3. Port one backend vertical slice to Rust.
   - Start with DB health and result-cache management.
   - Then port raw anomaly compute for a fixed dataset.
   - Then port weighted anomaly compute and skill overrides.

4. Decide frontend after backend parity is measurable.
   - Keep Streamlit if it becomes fast enough for internal operations.
   - Move to Tauri + React/Vite if desktop packaging and low resource use are primary.
   - Use Next.js only if there is a real multi-user web deployment need.

---

## 3. Redundancy and Dead-Code Candidates

These are candidates for review, not deletion instructions.

1. Old subpart page path appears unused.
   - `general_pages.py:845` defines `render_subpart_cost_page()`.
   - `app.py` navigation uses `render_assembly_audit_page`, not `render_subpart_cost_page`.
   - `app_context.py:592-594` defines `cached_subpart_analysis()`, which appears only used by that unused page.
   - `data_ingestion.py:1235` defines `get_subpart_detail()`, with no current references found.

2. Storage service wrapper appears unused.
   - `storage_service.py:2621-2752` defines `CostMonitoringService` and `service`.
   - No other file imports or uses `service`; the app routes through `harness.execute_action()` instead.

3. Excel skills export responsibilities overlap.
   - `skills_engine.py` defines `skills_to_excel_table()` and `skills_to_excel_bytes()`.
   - `skills_excel_export.py` also defines flatten/export helpers.
   - Keep one canonical exporter per skill domain.

4. Compatibility aliases should be sunset after migration.
   - `config.py` keeps `data_folder_path` and `assembly_detail_data_path` compatibility.
   - `storage_service.py` keeps legacy record-key parsing/migration logic.
   - These are justified today, but should have a retention policy after stable releases.

5. Runtime governance should be separated from production runtime.
   - `harness.py` integrity checks are useful for development guardrails.
   - In operations, running AST/source audits on every Streamlit rerun is unnecessary overhead.

6. Repository/runtime artifacts are heavy.
   - `wheels/`: 95 files, about `465.91 MB`.
   - Local DB + WAL/SHM and packaged exe are present in the workspace.
   - These should be excluded from normal source review and not mixed with application code changes.

---

## 4. Frontend Language Decision

### 4.1 Recommendation

Do not change frontend language immediately. First fix backend work amplification and measure again.

### 4.2 Option assessment

| Option | Fit | Why |
| --- | --- | --- |
| Keep Streamlit | Good short-term | Fastest path, existing app works, suitable for internal tools after rerun/cache fixes. |
| Tauri + React/Vite | Best low-spec desktop target after backend boundary exists | Tauri uses Rust plus WebView and can call Rust backend commands; React/Vite gives fine-grained UI control without Streamlit full-script reruns. |
| Plain React/Vite web app | Good if deployed with a separate local/server backend | Better table virtualization and interaction control than Streamlit, but needs API/backend packaging work. |
| Next.js | Not recommended for low-spec local desktop first | Strong for production web apps and server rendering, but adds Node/server complexity that this local tool does not currently need. |

### 4.3 Clarification

Tauri is not a replacement frontend language. It is a desktop application framework that can host React, Svelte, Vue, or plain HTML/CSS/JS while using Rust for backend/system calls.

---

## 5. Operational Test Scripts

Create these scripts only after deciding to implement the test harness. Until then, this section is the exact script plan.

### Script A: `scripts/ops_db_health.py`

Purpose: read-only SQLite health check for bloat, WAL size, table counts, and required table presence.

```python
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REQUIRED_TABLES = {
    "core_cost_records",
    "cost_anomaly_results",
    "expert_feedback",
    "expert_knowledge_base",
    "skills_items",
    "skills_snapshots",
}


def bytes_to_mb(value: int) -> float:
    return round(float(value) / 1024 / 1024, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="cost_monitor_data.db")
    parser.add_argument("--max-freelist-ratio", type=float, default=0.25)
    parser.add_argument("--max-wal-mb", type=float, default=64.0)
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(json.dumps({"ok": False, "error": f"missing db: {db_path}"}, ensure_ascii=False, indent=2))
        return 2

    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    cur = con.cursor()
    page_size = int(cur.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(cur.execute("PRAGMA page_count").fetchone()[0])
    freelist_count = int(cur.execute("PRAGMA freelist_count").fetchone()[0])
    journal_mode = str(cur.execute("PRAGMA journal_mode").fetchone()[0])
    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    counts = {}
    for table_name in tables:
        counts[table_name] = int(cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])
    con.close()

    wal_path = db_path.with_name(db_path.name + "-wal")
    shm_path = db_path.with_name(db_path.name + "-shm")
    freelist_ratio = freelist_count / max(page_count, 1)
    missing_tables = sorted(REQUIRED_TABLES - set(tables))
    payload = {
        "ok": True,
        "db_path": str(db_path),
        "db_mb": bytes_to_mb(db_path.stat().st_size),
        "wal_mb": bytes_to_mb(wal_path.stat().st_size) if wal_path.exists() else 0.0,
        "shm_mb": bytes_to_mb(shm_path.stat().st_size) if shm_path.exists() else 0.0,
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "freelist_mb": bytes_to_mb(freelist_count * page_size),
        "freelist_ratio": round(freelist_ratio, 4),
        "journal_mode": journal_mode,
        "table_counts": counts,
        "missing_tables": missing_tables,
        "thresholds": {
            "max_freelist_ratio": args.max_freelist_ratio,
            "max_wal_mb": args.max_wal_mb,
        },
    }
    failures = []
    if missing_tables:
        failures.append(f"missing tables: {', '.join(missing_tables)}")
    if freelist_ratio > args.max_freelist_ratio:
        failures.append(f"freelist ratio {freelist_ratio:.2%} exceeds {args.max_freelist_ratio:.2%}")
    if payload["wal_mb"] > args.max_wal_mb:
        failures.append(f"WAL {payload['wal_mb']} MB exceeds {args.max_wal_mb} MB")
    payload["ok"] = not failures
    payload["failures"] = failures
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

Run:

```powershell
python scripts\ops_db_health.py --db cost_monitor_data.db
```

Expected current result:

- Fails on freelist ratio until the database is compacted.
- Passes after compaction/checkpoint if freelist ratio is at or below `25%` and WAL is at or below `64 MB`.

### Script B: `scripts/ops_log_perf_summary.py`

Purpose: summarize existing performance logs without touching the database.

```python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return round(ordered[index], 3)


def parse_perf_line(line: str) -> dict | None:
    if "[performance]" not in line:
        return None
    bracket_values = re.findall(r"\[([^\]]+)\]", line)
    seconds = [float(value) for value in re.findall(r"([0-9]+\.[0-9]+)s", line)]
    rows_match = re.search(r"(?:记录数|输出行数)=([0-9]+)", line)
    if len(bracket_values) < 2 or not seconds:
        return None
    return {
        "stage": bracket_values[1],
        "mode": bracket_values[2] if len(bracket_values) >= 3 else "",
        "rows": int(rows_match.group(1)) if rows_match else 0,
        "seconds": seconds[-1],
        "line": line.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    args = parser.parse_args()

    log_root = Path(args.logs)
    records = []
    for path in sorted(log_root.glob("*.log")) + sorted(Path(".").glob("*.log")):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                parsed = parse_perf_line(line)
                if parsed:
                    parsed["file"] = str(path)
                    records.append(parsed)
        except OSError:
            continue

    groups: dict[str, list[float]] = {}
    for record in records:
        key = f"{record['stage']}|{record['mode']}".strip("|")
        groups.setdefault(key, []).append(record["seconds"])

    summary = {
        key: {
            "count": len(values),
            "p50_seconds": percentile(values, 0.50),
            "p95_seconds": percentile(values, 0.95),
            "max_seconds": round(max(values), 3),
        }
        for key, values in sorted(groups.items())
    }
    print(json.dumps({"record_count": len(records), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```powershell
python scripts\ops_log_perf_summary.py --logs logs
```

Acceptance:

- Every operational test report includes p50/p95/max for compute and DB write stages.
- A regression is flagged if p95 total time grows by more than `30%` versus the approved baseline on the same hardware/data volume.

### Script C: `scripts/ops_streamlit_start_probe.py`

Purpose: measure cold start and basic HTTP readiness without clicking compute actions.

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def wait_for_http(url: str, timeout_seconds: float) -> float:
    start = time.perf_counter()
    deadline = start + timeout_seconds
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return time.perf_counter() - start
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25)
    raise TimeoutError(last_error or f"not ready within {timeout_seconds}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8591)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-cold-start", type=float, default=15.0)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless=true",
        f"--server.port={args.port}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready_seconds = wait_for_http(url, args.timeout)
        ok = ready_seconds <= args.max_cold_start
        print(json.dumps({"ok": ok, "url": url, "cold_start_seconds": round(ready_seconds, 3)}, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```powershell
python scripts\ops_streamlit_start_probe.py --port 8591 --max-cold-start 15
```

Acceptance:

- Low-spec target: app reaches HTTP ready state in `<= 15s`.
- Normal dev machine: app reaches HTTP ready state in `<= 8s`.

### Script D: `scripts/run_ops_regression.ps1`

Purpose: run existing unit tests and immediately warn that live database isolation is required.

```powershell
$ErrorActionPreference = "Stop"

Write-Host "Running unit regression suite..."
python -m unittest tests.test_dgb_multiring

Write-Host ""
Write-Host "Important: anomaly persistence tests must run against an isolated copied DB before being used in CI."
Write-Host "For local operational validation, run ops_db_health.py after this command."
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ops_regression.ps1
```

Acceptance:

- `73` current tests pass.
- CI or release validation fails if the test database is not isolated from `cost_monitor_data.db`.

---

## 6. Acceptance Criteria

### 6.1 Current Python stabilization acceptance

| Area | Acceptance |
| --- | --- |
| Startup | HTTP ready `<= 15s` on low-spec target, `<= 8s` on dev machine. |
| No idle recompute | Navigating sidebar or expanding filters does not emit new `[performance][计算阶段]` lines. |
| Current 1,170-row dataset | raw compute + write `<= 1.5s`; weighted compute + write `<= 1.5s`. |
| 10k operational dataset | raw compute + write `<= 12s`; weighted compute + write `<= 15s` on low-spec target. |
| DB health | freelist ratio `<= 25%` after maintenance; WAL `<= 64 MB` after idle/checkpoint. |
| Cache | all DataFrame-heavy `st.cache_data` functions have explicit `ttl` or `max_entries`, or are replaced by persisted-query keys. |
| Export | Excel bytes are generated only after explicit export action or a confirmed cached export key. |
| Tests | unit tests pass and do not mutate the live operational DB. |

### 6.2 Rust backend acceptance

| Area | Acceptance |
| --- | --- |
| Parity | Golden datasets produce same status classification and bounds within defined numeric tolerance. |
| Speed | Rust raw/weighted compute is at least `2x` faster than stabilized Python on 10k rows, or removes enough memory/packaging overhead to justify migration. |
| Memory | Peak backend memory on 10k rows remains `<= 512 MB` on low-spec target. |
| DB writes | Result save avoids delete-then-full-append when only parameters or labels change; write amplification is measured. |
| Recovery | Interrupted compute does not leave partially visible result sets. |
| Packaging | Desktop package does not require shipping the full `wheels/` directory. |

### 6.3 Frontend acceptance if replaced

| Area | Streamlit target | Tauri + React/Vite target |
| --- | --- | --- |
| Interaction | no full recompute on filter/table changes | no backend compute unless user clicks compute/refresh |
| Table UX | handles 1k visible rows without browser freeze | virtualized table handles 10k rows with smooth scroll |
| Packaging | acceptable for internal Python deployment | single desktop installer/bundle with Rust backend |
| Offline use | local DB works | local DB works |

---

## 7. Implementation Tasks

### Task 1: Establish safe operational baseline

**Files:**
- Create: `scripts/ops_db_health.py`
- Create: `scripts/ops_log_perf_summary.py`
- Create: `scripts/ops_streamlit_start_probe.py`
- Create: `scripts/run_ops_regression.ps1`

- [ ] Add the four scripts exactly as specified in Section 5.
- [ ] Run `python scripts\ops_db_health.py --db cost_monitor_data.db`.
- [ ] Run `python scripts\ops_log_perf_summary.py --logs logs`.
- [ ] Run `python scripts\ops_streamlit_start_probe.py --port 8591 --max-cold-start 15`.
- [ ] Run `powershell -ExecutionPolicy Bypass -File scripts\run_ops_regression.ps1`.
- [ ] Save the JSON outputs under `docs/ops-baselines/YYYY-MM-DD/`.

### Task 2: Stop accidental production work during rerun

**Files:**
- Modify: `app_context.py`
- Modify: `harness.py`
- Modify: `.streamlit/config.toml` if a runtime flag is chosen there.

- [ ] Gate `harness.bootstrap_runtime_governance()` behind a development/runtime flag.
- [ ] Keep sidebar warnings available in development mode.
- [ ] Verify sidebar navigation does not run AST audit in operations mode.

### Task 3: Fix result cache read-before-compute

**Files:**
- Modify: `compute_jobs.py`
- Modify: `storage_service.py` if a result metadata query is added.
- Test: `tests/test_dgb_multiring.py`

- [x] Add a metadata check for existing result mode, source data version, parameter hash, and label/skill hash.
- [x] Load persisted results before recomputing when metadata matches.
- [x] Add a failing test that a cache hit does not call the detector.
- [x] Verify raw and weighted result parity through the isolated regression suite.

### Task 4: Reduce DB bloat and write amplification

**Files:**
- Modify: `storage_service.py`
- Modify: `general_pages.py` if maintenance UX changes.
- Test: `tests/test_dgb_multiring.py`

- [x] Reduce write amplification by reusing persisted raw/weighted results when run metadata matches.
- [x] Add checkpoint/compact operations as explicit maintenance actions.
- [x] Add DB health warnings based on freelist ratio and WAL size.
- [x] Verify `ops_db_health.py` passes after maintenance.

### Task 5: Bound Streamlit cache and move exports behind explicit actions

**Files:**
- Modify: `app_context.py`
- Modify: `assembly_ui.py`
- Modify: `sheet_metal_ui.py`
- Modify: `general_pages.py`
- Modify: `cost_monitor_ui.py`

- [x] Add explicit `max_entries` or `ttl` to heavy `st.cache_data` functions.
- [x] Add stable source/options signatures for anomaly result reuse.
- [x] Generate major Excel bytes only after explicit user action.
- [x] Verify direct page-render Excel generation is blocked by static tests.

### Task 6: Rust backend proof of concept

**Files:**
- Create: `backend-rust/`
- Create: `backend-rust/README.md`
- Create: `docs/architecture/backend-api-contract.md`

- [x] Define request/response schemas for DB health and raw anomaly compute.
- [x] Create a zero-dependency Rust CLI POC with `health`.
- [ ] Implement DB health command first.
- [ ] Implement raw anomaly compute on a golden dataset.
- [ ] Compare output with current Python results.
- [ ] Decide whether the speed/memory gain justifies continuing to weighted compute.

---

## 8. Execution Results

Completed on 2026-06-24:

- Added operational scripts: `ops_db_health.py`, `ops_log_perf_summary.py`, `ops_streamlit_start_probe.py`, `ops_db_maintenance.py`, and isolated `run_ops_regression.py`.
- Saved baselines under `docs/ops-baselines/2026-06-24/`.
- Compacted the live SQLite DB from 95.24 MB to 8.09 MB; freelist ratio moved from 91.45% to 0.00%.
- Added runtime governance gating with `COST_MONITOR_RUNTIME_GOVERNANCE=1`.
- Added anomaly result run metadata and read-before-compute behavior.
- Added cache bounds and deferred major Excel export generation.
- Added Rust contract and `backend-rust` POC skeleton.

Verification:

- `python scripts\run_ops_regression.py --cwd . --db cost_monitor_data.db`: 93 tests passed with DB snapshot/restore.
- `python -m compileall app_context.py compute_jobs.py cost_monitor_ui.py general_pages.py harness.py page_ui_helpers.py sheet_metal_ui.py storage_service.py scripts tests`: passed.
- `python scripts\ops_db_health.py --db cost_monitor_data.db`: passed after maintenance.
- `cargo test`: not run because `cargo`/`rustc` are not installed on this machine.

---

## 9. Self Review

Spec coverage:

- Performance/root-cause analysis is covered in Sections 1 and 6.
- Rust backend feasibility is covered in Section 2.
- Redundant code and feature candidates are covered in Section 3.
- Frontend language analysis is covered in Section 4.
- Operational test scripts and acceptance standards are covered in Sections 5 and 6.

Placeholder scan:

- No unresolved placeholder step is intentionally left in this document.
- Script sections contain complete runnable draft code.

Risk review:

- Current unit tests can mutate the derived anomaly cache table if run directly; `scripts/run_ops_regression.py` now snapshots/restores the DB family and should be used for local regression.
- Database compaction is now an explicit operational action, not hidden inside normal page rendering.
- Rust compute migration should not continue until `db-health` and golden parity datasets are implemented.

---

## 10. Reference Sources

- Streamlit rerun/fragments and caching model: https://docs.streamlit.io/develop/concepts/architecture/fragments and https://docs.streamlit.io/develop/concepts/architecture/caching
- Streamlit `st.cache_data` API: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
- SQLite `VACUUM`: https://sqlite.org/lang_vacuum.html
- SQLite WAL: https://sqlite.org/wal.html
- Rust language overview: https://www.rust-lang.org/en-US
- Tauri overview and architecture: https://v2.tauri.app/start/ and https://v2.tauri.app/concept/architecture/
- React project guidance: https://react.dev/learn/creating-a-react-app
- Next.js docs: https://nextjs.org/docs
