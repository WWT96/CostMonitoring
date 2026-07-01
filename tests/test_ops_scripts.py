import json
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError


class OpsScriptsTests(unittest.TestCase):
    def _create_required_tables(self, db_path: Path) -> None:
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE core_cost_records (id INTEGER PRIMARY KEY);
                CREATE TABLE cost_anomaly_results (id INTEGER PRIMARY KEY, result_mode TEXT);
                CREATE TABLE expert_feedback (record_key TEXT PRIMARY KEY);
                CREATE TABLE expert_knowledge_base (rule_id TEXT PRIMARY KEY);
                CREATE TABLE skills_items (item_id INTEGER PRIMARY KEY);
                CREATE TABLE skills_snapshots (snapshot_id TEXT PRIMARY KEY);
                INSERT INTO core_cost_records VALUES (1);
                INSERT INTO cost_anomaly_results VALUES (1, 'raw');
                INSERT INTO cost_anomaly_results VALUES (2, 'weighted');
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_db_health_reports_table_counts_without_touching_live_database(self) -> None:
        from scripts.ops_db_health import collect_db_health

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "ops.db"
            self._create_required_tables(db_path)

            result = collect_db_health(db_path, max_freelist_ratio=1.0, max_wal_mb=64.0)

        self.assertTrue(result["ok"])
        self.assertEqual(result["table_counts"]["core_cost_records"], 1)
        self.assertEqual(result["table_counts"]["cost_anomaly_results"], 2)
        self.assertEqual(result["missing_tables"], [])
        self.assertEqual(result["failures"], [])

    def test_db_health_fails_when_required_tables_are_missing(self) -> None:
        from scripts.ops_db_health import collect_db_health

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "ops.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE core_cost_records (id INTEGER PRIMARY KEY)")
                conn.commit()
            finally:
                conn.close()

            result = collect_db_health(db_path, max_freelist_ratio=1.0, max_wal_mb=64.0)

        self.assertFalse(result["ok"])
        self.assertIn("cost_anomaly_results", result["missing_tables"])
        self.assertTrue(any("missing tables" in failure for failure in result["failures"]))

    def test_log_perf_summary_parses_chinese_performance_lines(self) -> None:
        from scripts.ops_log_perf_summary import summarize_performance_logs

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            (log_dir / "perf.log").write_text(
                "\n".join(
                    [
                        "[performance][计算阶段][raw] 记录数=1170 分组数=11 总耗时=0.359s",
                        "[performance][存库阶段][weighted] 记录数=1170 写入耗时=0.312s 总耗时=0.324s",
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_performance_logs(log_dir)

        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["summary"]["计算阶段|raw"]["max_seconds"], 0.359)
        self.assertEqual(summary["summary"]["存库阶段|weighted"]["p50_seconds"], 0.324)

    def test_streamlit_start_probe_waits_for_http_success(self) -> None:
        from scripts.ops_streamlit_start_probe import wait_for_http

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch("urllib.request.urlopen", return_value=_Response()):
            ready_seconds = wait_for_http("http://127.0.0.1:8591", timeout_seconds=1.0)

        self.assertGreaterEqual(ready_seconds, 0.0)

    def test_streamlit_start_probe_raises_on_timeout(self) -> None:
        from scripts.ops_streamlit_start_probe import wait_for_http

        with mock.patch("urllib.request.urlopen", side_effect=URLError("closed")):
            with mock.patch("time.sleep", return_value=None):
                with self.assertRaises(TimeoutError):
                    wait_for_http("http://127.0.0.1:8591", timeout_seconds=0.01)

    def test_log_perf_summary_json_payload_is_serializable(self) -> None:
        from scripts.ops_log_perf_summary import summarize_performance_logs

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = summarize_performance_logs(Path(tmp_dir))

        self.assertEqual(json.loads(json.dumps(payload, ensure_ascii=False)), payload)

    def test_log_perf_summary_cli_output_is_windows_console_safe(self) -> None:
        from scripts.ops_log_perf_summary import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            (log_dir / "perf.log").write_text(
                "[performance][读取阶段] 文件数=1 命中缓存=0 重建缓存=1 输出行数=1 总耗时=0.009s �\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["--logs", str(log_dir)])

        self.assertEqual(exit_code, 0)
        stdout.getvalue().encode("ascii")

    def test_regression_runner_restores_db_family_after_mutation(self) -> None:
        from scripts.run_ops_regression import restore_db_family, snapshot_db_family

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "cost_monitor_data.db"
            wal_path = root / "cost_monitor_data.db-wal"
            backup_dir = root / "backup"
            db_path.write_text("original-db", encoding="utf-8")
            wal_path.write_text("original-wal", encoding="utf-8")

            snapshot = snapshot_db_family(db_path, backup_dir)
            db_path.write_text("mutated-db", encoding="utf-8")
            wal_path.unlink()
            db_path.with_name("cost_monitor_data.db-shm").write_text("new-shm", encoding="utf-8")

            restore_db_family(db_path, snapshot)

            self.assertEqual(db_path.read_text(encoding="utf-8"), "original-db")
            self.assertEqual(wal_path.read_text(encoding="utf-8"), "original-wal")
            self.assertFalse(db_path.with_name("cost_monitor_data.db-shm").exists())

    def test_regression_runner_builds_default_unittest_command(self) -> None:
        from scripts.run_ops_regression import build_unittest_command

        command = build_unittest_command(["tests.test_dgb_multiring", "tests.test_ops_scripts"])

        self.assertEqual(command[1:3], ["-m", "unittest"])
        self.assertEqual(command[-2:], ["tests.test_dgb_multiring", "tests.test_ops_scripts"])

    def test_cost_anomaly_result_run_metadata_gates_fresh_loads(self) -> None:
        import pandas as pd
        import storage_service
        from sqlalchemy import create_engine

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "ops.db"
            engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
            result_df = pd.DataFrame(
                [
                    {
                        "_record_key": "rk-fresh",
                        "物料编码": "MAT-001",
                        "备件简称": "门板",
                        "实际成本": 10.0,
                        "预测值": 10.0,
                        "status": "正常",
                        "价格有效于": "2026-06-01",
                    }
                ]
            )

            try:
                with (
                    mock.patch.object(storage_service, "DB_ENGINE", engine),
                    mock.patch.object(storage_service, "_COST_ANOMALY_RESULTS_RESET_DONE", False),
                    mock.patch.object(storage_service.harness, "authorize_db_operation", return_value={}),
                ):
                    storage_service.save_cost_anomaly_results(result_df, result_mode="raw")

                    missing_meta = storage_service.load_fresh_cost_anomaly_results(
                        "raw",
                        source_signature="source-a",
                        options_signature="opts-a",
                    )
                    storage_service.record_cost_anomaly_result_run(
                        "raw",
                        source_signature="source-a",
                        options_signature="opts-a",
                        row_count=1,
                    )
                    fresh = storage_service.load_fresh_cost_anomaly_results(
                        "raw",
                        source_signature="source-a",
                        options_signature="opts-a",
                    )
                    stale = storage_service.load_fresh_cost_anomaly_results(
                        "raw",
                        source_signature="source-a",
                        options_signature="opts-b",
                    )
            finally:
                engine.dispose()

        self.assertTrue(missing_meta.empty)
        self.assertEqual(fresh["status"].tolist(), ["正常"])
        self.assertTrue(stale.empty)

    def test_db_maintenance_checkpoint_is_explicit_and_non_destructive(self) -> None:
        from scripts.ops_db_maintenance import run_database_maintenance

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "ops.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO sample(value) VALUES ('kept')")
                conn.commit()
            finally:
                conn.close()

            result = run_database_maintenance(db_path, vacuum=False)
            conn = sqlite3.connect(db_path)
            try:
                rows = conn.execute("SELECT value FROM sample").fetchall()
            finally:
                conn.close()

        self.assertTrue(result["ok"])
        self.assertFalse(result["vacuum"])
        self.assertIn("checkpoint", result["operations"])
        self.assertEqual(rows, [("kept",)])

    def test_db_maintenance_vacuum_reports_before_and_after_stats(self) -> None:
        from scripts.ops_db_maintenance import run_database_maintenance

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "ops.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
                conn.executemany("INSERT INTO sample(value) VALUES (?)", [(str(idx),) for idx in range(50)])
                conn.execute("DELETE FROM sample WHERE id <= 45")
                conn.commit()
            finally:
                conn.close()

            result = run_database_maintenance(db_path, vacuum=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["vacuum"])
        self.assertIn("before", result)
        self.assertIn("after", result)
        self.assertGreaterEqual(result["before"]["db_bytes"], result["after"]["db_bytes"])

    def test_streamlit_data_caches_are_bounded(self) -> None:
        cache_files = [
            Path("app_context.py"),
            Path("sheet_metal_ui.py"),
            Path("cost_monitor_ui.py"),
        ]
        bare_cache_lines: list[str] = []
        for file_path in cache_files:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.strip() == "@st.cache_data":
                    bare_cache_lines.append(f"{file_path}:{line_number}")

        self.assertEqual(bare_cache_lines, [])

    def test_streamlit_toolbar_hides_developer_cache_actions(self) -> None:
        config_text = Path(".streamlit/config.toml").read_text(encoding="utf-8")

        self.assertIn('toolbarMode = "viewer"', config_text)

    def test_large_excel_exports_are_deferred_until_requested(self) -> None:
        ui_files = [
            Path("general_pages.py"),
            Path("cost_monitor_ui.py"),
            Path("sheet_metal_ui.py"),
        ]
        direct_export_lines: list[str] = []
        patterns = [
            "export_data=to_excel_bytes(",
            "excel_data=to_excel_bytes(",
            "audit_bytes=to_excel_bytes(",
            "data=to_excel_bytes(",
            "data=skills_engine.skills_to_excel_bytes(",
            "skills_excel_bytes=sheet_metal_logic.sheet_metal_skills_to_excel_bytes(",
            "data=sheet_metal_logic.sheet_metal_skills_to_excel_bytes(",
        ]
        for file_path in ui_files:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if any(pattern in line.replace(" ", "") for pattern in patterns):
                    direct_export_lines.append(f"{file_path}:{line_number}")

        self.assertEqual(direct_export_lines, [])

    def test_deferred_download_payload_is_size_limited_and_clearable(self) -> None:
        from page_ui_helpers import clear_deferred_download_payload, store_deferred_download_payload

        state = {}
        store_deferred_download_payload(
            state,
            key="demo",
            payload=b"small",
            file_name="demo.xlsx",
            max_bytes=10,
        )
        self.assertEqual(state["_deferred_download_payload_demo"], b"small")

        clear_deferred_download_payload(state, "demo")
        self.assertNotIn("_deferred_download_payload_demo", state)
        self.assertNotIn("_deferred_download_file_demo", state)

        with self.assertRaises(ValueError):
            store_deferred_download_payload(
                state,
                key="big",
                payload=b"01234567890",
                file_name="big.xlsx",
                max_bytes=10,
            )
        self.assertNotIn("_deferred_download_payload_big", state)

    def test_rust_backend_contract_and_poc_skeleton_exist(self) -> None:
        required_paths = [
            Path("docs/architecture/backend-api-contract.md"),
            Path("backend-rust/Cargo.toml"),
            Path("backend-rust/src/main.rs"),
            Path("backend-rust/README.md"),
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]

        self.assertEqual(missing_paths, [])
        self.assertIn("cost-monitor-backend", Path("backend-rust/Cargo.toml").read_text(encoding="utf-8"))
        self.assertIn("health", Path("backend-rust/src/main.rs").read_text(encoding="utf-8"))

    def test_runtime_memory_governance_harness_records_current_standards(self) -> None:
        harness_path = Path("harness/runtime_memory_governance.json")

        payload = json.loads(harness_path.read_text(encoding="utf-8"))
        checklist_ids = {item["id"] for item in payload["verification_checklist"]}

        self.assertEqual(payload["status"], "active")
        self.assertIn("deferred_download_payloads", checklist_ids)
        self.assertIn("bounded_streamlit_caches", checklist_ids)
        self.assertIn("explicit_session_release", checklist_ids)
        self.assertIn("low_spec_parallelism_gate", checklist_ids)


if __name__ == "__main__":
    unittest.main()
