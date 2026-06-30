from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from anomaly_engine import _detect_group_anomalies_worker, apply_accepted_ring_union_status, infer_anomaly_reason
from app_context import build_calibration_management_df, clear_cost_feedback_and_ai_state
from assembly_logic import _load_relationship_rows, build_abnormal_ratio_export_df, sort_assembly_summary_by_cost_ratio
from config import _build_llm_api_configs
from cost_monitor_ui import (
    build_cost_anomaly_run_request,
    build_cost_anomaly_chart_scope,
    build_interval_lower_bound_rank_analysis,
    build_interval_compare_display_labels,
    export_cost_anomaly_result_excel,
    filter_label_details_to_management_rows,
    filter_cost_anomaly_scope,
    filter_latest_material_cost_records_for_monitoring,
    sort_interval_compare_chart_data,
)
from data_ingestion import (
    _prepare_core_cost_records,
    build_default_vehicle_rank,
    build_builtin_template_excel_bytes,
    build_manual_vehicle_rank_from_display,
    build_manual_vehicle_market_price_rows_from_display,
    build_vehicle_market_price_display_df,
    extract_first_vehicle_series_name,
    extract_missing_vehicle_market_price_series,
    extract_vehicle_rank_candidates,
    filter_latest_cost_increase_rows,
    generate_pivot_report,
    get_material_metrics,
    get_vehicle_gradient_comparison,
    load_data_from_uploaded_files,
    prioritize_latest_cost_increases,
    process_dataframe,
)
from compute_jobs import ComputeJob
from import_jobs import ImportJob
from general_pages import apply_vehicle_market_price_auto_rank
from llm_engine import (
    _build_group_payloads,
    _chat_completions_url,
    _call_llm,
    _load_feedback_records_for_knowledge,
    _normalize_direct_llm_url,
    _post_chat_completion_with_fallback,
    explain_vehicle_gradient_deviations,
    fetch_vehicle_market_prices,
    normalize_vehicle_market_price_result,
)
from page_ui_helpers import get_vehicle_market_price_manual_editable_columns
from sheet_metal_logic import (
    build_sheet_metal_audit_report,
    build_sheet_metal_calibration_management_df,
    detect_sheet_metal_anomalies,
    sheet_metal_skills_to_excel_bytes,
)
from sheet_metal_ui import build_sheet_metal_review_run_request, export_sheet_metal_review_result_excel
from skills_engine import (
    build_cost_skill_overrides_json,
    export_cost_skills_excel_artifacts,
    extract_skills,
    generate_audit_report,
    load_latest_cost_skills_excel,
    run_auto_research,
    skills_to_excel_bytes,
)
from storage_service import (
    EXPERT_KNOWLEDGE_BASE_TABLE,
    _prepare_anomaly_results,
    canonicalize_record_key,
    find_feedback_rows_missing_required_remarks,
    load_expert_knowledge_base,
    load_vehicle_rank_config,
    make_record_key,
    save_cost_anomaly_results,
    save_vehicle_rank_config,
    split_record_key,
)
from ui_utils import render_merged_html_table


def _group_frame(values: list[float], *, short_name: str = "测试件") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "物料编码": [f"MAT-{idx:03d}" for idx in range(len(values))],
            "物料名称": [short_name] * len(values),
            "适用车系": ["测试车系"] * len(values),
            "工厂": ["测试工厂"] * len(values),
            "备件简称": [short_name] * len(values),
            "实际成本": values,
            "价格有效于": pd.date_range("2025-01-01", periods=len(values), freq="D"),
            "样本量": [len(values)] * len(values),
            "_recency_weight": np.ones(len(values), dtype=float),
        }
    )


class _UploadedFileStub:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


@contextlib.contextmanager
def _workspace_temp_dir(name: str):
    temp_dir = Path(".test_tmp") / name
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class DgbMultiRingTests(unittest.TestCase):
    def test_single_ring_keeps_one_normal_main_ring(self) -> None:
        values = np.linspace(95.0, 105.0, 40).round(4).tolist()

        result = _detect_group_anomalies_worker(
            _group_frame(values),
            {"weighted": False, "gap_k": 4.0, "baseline_quantile": 0.5},
        )

        self.assertIn("圈层编号", result.columns)
        self.assertIn("圈层角色", result.columns)
        self.assertIn("圈层置信度", result.columns)
        self.assertIn("多圈合理区间", result.columns)
        self.assertEqual(set(result["圈层角色"]), {"主邻居圈"})
        self.assertEqual(set(result["status"]), {"正常（主邻居圈）"})

    def test_secondary_supported_ring_is_normal_without_bridging_the_gap(self) -> None:
        values = (
            np.linspace(96.0, 104.0, 30).round(4).tolist()
            + np.linspace(298.0, 302.0, 2).round(4).tolist()
            + np.linspace(496.0, 504.0, 10).round(4).tolist()
        )

        result = _detect_group_anomalies_worker(
            _group_frame(values),
            {"weighted": False, "gap_k": 4.0, "baseline_quantile": 0.5},
        )

        main_or_secondary = result[result["圈层角色"].isin(["主邻居圈", "次邻居圈"])]
        self.assertGreaterEqual(main_or_secondary["圈层编号"].nunique(), 2)
        self.assertEqual(set(result.loc[result["实际成本"].between(496.0, 504.0), "status"]), {"正常（次邻居圈）"})
        self.assertTrue(result.loc[result["实际成本"].between(298.0, 302.0), "status"].str.contains("异常").all())
        intervals = result["多圈合理区间"].dropna().astype(str).iloc[0]
        self.assertIn("主邻居圈", intervals)
        self.assertIn("次邻居圈", intervals)

    def test_small_isolated_component_remains_anomaly(self) -> None:
        values = np.linspace(96.0, 104.0, 36).round(4).tolist() + [500.0, 502.0]

        result = _detect_group_anomalies_worker(
            _group_frame(values),
            {"weighted": False, "gap_k": 4.0, "baseline_quantile": 0.5},
        )

        isolated = result[result["实际成本"] >= 500.0]
        self.assertEqual(set(isolated["圈层角色"]), {"未采纳孤立圈"})
        self.assertTrue(isolated["status"].str.contains("异常").all())

    def test_expert_normal_anchor_preserves_small_secondary_ring(self) -> None:
        values = np.linspace(96.0, 104.0, 36).round(4).tolist() + [500.0, 502.0]
        group = _group_frame(values)
        group["_is_expert_normal"] = group["实际成本"] >= 500.0

        result = _detect_group_anomalies_worker(
            group,
            {
                "weighted": True,
                "sigma_multiplier": 1.0,
                "expert_weight": 80,
                "decay_alpha": 1.0,
                "gap_k": 4.0,
                "baseline_quantile": 0.5,
                "group_source": "测试专家锚点",
            },
        )

        anchored = result[result["实际成本"] >= 500.0]
        self.assertEqual(set(anchored["圈层角色"]), {"次邻居圈"})
        self.assertEqual(set(anchored["status"]), {"正常（次邻居圈）"})

    def test_extract_skills_keeps_multi_ring_intervals(self) -> None:
        values = (
            np.linspace(96.0, 104.0, 30).round(4).tolist()
            + np.linspace(298.0, 302.0, 2).round(4).tolist()
            + np.linspace(496.0, 504.0, 10).round(4).tolist()
        )
        result = _detect_group_anomalies_worker(
            _group_frame(values),
            {"weighted": False, "gap_k": 4.0, "baseline_quantile": 0.55},
        )
        result["_record_key"] = [f"rk-{idx}" for idx in range(len(result))]

        skills = extract_skills(result, {}, baseline_quantile=0.55)

        self.assertEqual(len(skills), 1)
        skill = skills[0]
        self.assertEqual(skill["适用算法"], "DGB-MultiRing KDE+KNN+Elbow 密度连接异常检测")
        self.assertEqual(skill["Baseline Quantile"], 0.55)
        self.assertEqual(skill["次邻居圈数量"], 1)
        self.assertGreaterEqual(len(skill["多邻居圈合理区间"]), 2)

    def test_storage_prepare_maps_ring_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "_record_key": "rk-storage",
                    "物料编码": "MAT-001",
                    "备件简称": "测试件",
                    "实际成本": 100.0,
                    "预测值": 100.0,
                    "合理下限": 95.0,
                    "合理上限": 105.0,
                    "偏离数值": 0.0,
                    "偏离比例": 0.0,
                    "status": "正常（主邻居圈）",
                    "圈层编号": 1,
                    "圈层角色": "主邻居圈",
                    "圈层置信度": 1.0,
                    "多圈合理区间": "[]",
                }
            ]
        )

        prepared = _prepare_anomaly_results(frame, "raw")

        self.assertIn("ring_id", prepared.columns)
        self.assertIn("ring_role", prepared.columns)
        self.assertIn("ring_confidence", prepared.columns)
        self.assertIn("ring_intervals_json", prepared.columns)
        self.assertEqual(int(prepared["ring_id"].iloc[0]), 1)
        self.assertEqual(prepared["ring_role"].iloc[0], "主邻居圈")

    def test_small_sample_baseline_stays_inside_bounds_for_cost_and_sheet_metal(self) -> None:
        result = _detect_group_anomalies_worker(
            _group_frame([10.4, 25.6]),
            {"weighted": False, "gap_k": 4.0, "baseline_quantile": 0.5},
        )

        self.assertTrue((result["合理下限"] <= result["预测值"]).all())
        self.assertTrue((result["预测值"] <= result["合理上限"]).all())

        sheet_df = pd.DataFrame(
            {
                "物料编码": ["SM-001", "SM-002"],
                "物料名称": ["钣金A", "钣金B"],
                "适用车系": ["测试车系", "测试车系"],
                "工厂": ["钣金", "钣金"],
                "备件简称": ["A柱加强板中段", "A柱加强板中段"],
                "白痴指数": [10.4, 25.6],
                "monitor_date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
                "静态快照时间": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            }
        )

        sheet_result = detect_sheet_metal_anomalies(sheet_df)

        self.assertTrue((sheet_result["合理下限"] <= sheet_result["基准指数"]).all())
        self.assertTrue((sheet_result["基准指数"] <= sheet_result["合理上限"]).all())

    def test_dense_minor_high_tail_does_not_become_main_reasonable_interval(self) -> None:
        low_cluster = np.linspace(100.0, 1100.0, 30).round(4).tolist()
        dense_high_tail = np.linspace(2701.0, 2751.0, 8).round(4).tolist()
        result = _detect_group_anomalies_worker(
            _group_frame(low_cluster + dense_high_tail),
            {"weighted": False, "gap_k": 4.0, "baseline_quantile": 0.5},
        )

        main_rows = result[result["圈层角色"].eq("主邻居圈")]

        self.assertFalse(main_rows.empty)
        self.assertLess(float(main_rows["合理上限"].max()), max(dense_high_tail))
        self.assertEqual(set(result.loc[result["实际成本"].between(2701.0, 2751.0), "圈层角色"]), {"次邻居圈"})

    def test_monitoring_keeps_only_latest_cost_date_per_material_code(self) -> None:
        frame = pd.DataFrame(
            {
                "物料编码": ["A", "A", "A", "B"],
                "价格有效于": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-03-01", "2026-02-01"]),
                "工厂": ["一厂", "一厂", "二厂", "一厂"],
                "实际成本": [10.0, 11.0, 12.0, 20.0],
                "status": ["异常偏高", "异常偏高", "异常偏高", "正常（主邻居圈）"],
            }
        )

        latest = filter_latest_material_cost_records_for_monitoring(frame)

        self.assertEqual(latest["物料编码"].tolist(), ["A", "A", "B"])
        self.assertNotIn(pd.Timestamp("2026-01-01"), set(latest["价格有效于"]))

    def test_single_material_metrics_include_freight_factor_and_cost_drop(self) -> None:
        frame = pd.DataFrame(
            {
                "物料编码": ["A", "A", "A", "A", "A"],
                "工厂": ["X990", "总装", "X990", "总装", "X990"],
                "monitor_date": pd.to_datetime(
                    ["2025-12-31", "2025-12-31", "2026-06-01", "2026-05-20", "2024-12-31"]
                ),
                "成本": [120.0, 100.0, 90.0, 80.0, 130.0],
            }
        )

        metrics = get_material_metrics(frame, "成本")

        self.assertEqual(metrics["latest_factory"], "X990")
        self.assertAlmostEqual(metrics["freight_factor"], 90.0 / 80.0)
        self.assertEqual(metrics["cost_drop_factory"], "X990")
        self.assertEqual(metrics["cost_drop_reference_price"], 120.0)
        self.assertEqual(pd.Timestamp(metrics["cost_drop_reference_date"]), pd.Timestamp("2025-12-31"))
        self.assertEqual(metrics["cost_drop_amount"], -30.0)

    def test_assembly_relationship_loader_reads_layer1_to_layer2_columns_only(self) -> None:
        with _workspace_temp_dir("assembly_relationship") as folder:
            file_path = folder / "assembly.xlsx"
            pd.DataFrame(
                {
                    "层级0编码": ["OLD-PARENT"],
                    "层级0名称": ["旧父级"],
                    "层级1编码": ["PARENT-001"],
                    "层级1名称": ["父级零件"],
                    "层级2编码": ["CHILD-001"],
                    "层级2名称": ["子级零件"],
                }
            ).to_excel(file_path, index=False)

            relationship_df, warnings, contract_detected = _load_relationship_rows([file_path])

        self.assertEqual(warnings, [])
        self.assertTrue(contract_detected)
        self.assertEqual(relationship_df.loc[0, "层级0编码"], "PARENT-001")
        self.assertEqual(relationship_df.loc[0, "层级0名称"], "父级零件")
        self.assertEqual(relationship_df.loc[0, "层级1编码"], "CHILD-001")
        self.assertEqual(relationship_df.loc[0, "层级1名称"], "子级零件")

    def test_skills_markdown_uses_responsive_two_column_report_grids(self) -> None:
        skills = [
            {
                "备件简称": "测试件",
                "适用算法": "DGB-MultiRing KDE+KNN+Elbow 密度连接异常检测",
                "数据结构分布描述": {"样本量": 2, "均值": 10.0},
                "当前σ参数": 1.0,
                "偏置权重": 80,
                "时序敏感度 (Decay Alpha)": 1.0,
                "圈子严格度 (Gap K)": 4.0,
                "Baseline Quantile": 0.5,
                "参数语义说明": [],
                "本组专家标注数": 0,
                "成本合理区间边界": {"预测值": 10.0, "合理下限": 9.0, "合理上限": 11.0},
                "多邻居圈合理区间": [],
                "主邻居圈编号": 1,
                "次邻居圈数量": 0,
                "异常统计": {"正常": 1, "异常偏高": 0, "异常偏低": 1},
                "语义校准报告": {"引用规律数": 0, "主要匹配方式": [], "参考文本规律": []},
                "经验对齐率": "N/A",
            }
        ]

        markdown = __import__("skills_engine").skills_to_markdown(skills)

        self.assertGreaterEqual(markdown.count("skills-report-grid"), 2)
        self.assertIn("skills-report-panel", markdown)
        self.assertIn("grid-template-columns: repeat(2", markdown)

    def test_skills_markdown_escapes_dynamic_text_before_html_rendering(self) -> None:
        skills = [
            {
                "备件简称": "<script>alert('x')</script>",
                "适用算法": "<img src=x onerror=alert(1)>",
                "数据结构分布描述": {"样本量": 2, "均值": 10.0},
                "当前σ参数": 1.0,
                "偏置权重": 80,
                "时序敏感度 (Decay Alpha)": 1.0,
                "圈子严格度 (Gap K)": 4.0,
                "Baseline Quantile": 0.5,
                "参数语义说明": ["<b>不要渲染</b>"],
                "本组专家标注数": 0,
                "成本合理区间边界": {"预测值": 10.0, "合理下限": 9.0, "合理上限": 11.0},
                "多邻居圈合理区间": [],
                "主邻居圈编号": 1,
                "次邻居圈数量": 0,
                "异常统计": {"正常": 1, "异常偏高": 0, "异常偏低": 1},
                "语义校准报告": {"引用规律数": 0, "主要匹配方式": [], "参考文本规律": []},
                "经验对齐率": "N/A",
            }
        ]

        markdown = __import__("skills_engine").skills_to_markdown(skills)

        self.assertNotIn("<script>alert('x')</script>", markdown)
        self.assertNotIn("<img src=x onerror=alert(1)>", markdown)
        self.assertNotIn("<b>不要渲染</b>", markdown)
        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", markdown)
        self.assertIn("&lt;b&gt;不要渲染&lt;/b&gt;", markdown)

    def test_uploaded_file_loader_reports_partial_parse_failures(self) -> None:
        valid_csv = (
            "物料编码,物料名称,适用车系,备件简称,工厂,价格有效于,成本\n"
            "MAT-001,零件A,车系A,简称A,工厂A,2026-06-01,12.3\n"
        ).encode("utf-8")
        invalid_csv = "foo,bar\n1,2\n".encode("utf-8")

        merged_df, price_col, error_msg, warnings = load_data_from_uploaded_files(
            [
                _UploadedFileStub("valid.csv", valid_csv),
                _UploadedFileStub("broken.csv", invalid_csv),
            ]
        )

        self.assertIsNone(error_msg)
        self.assertEqual(price_col, "成本")
        self.assertEqual(len(merged_df), 1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("broken.csv", warnings[0])
        self.assertIn("缺少必要列", warnings[0])

    def test_import_job_blocks_uploaded_sync_when_any_file_fails(self) -> None:
        valid_csv = (
            "物料编码,物料名称,适用车系,备件简称,工厂,价格有效于,成本\n"
            "MAT-001,零件A,车系A,简称A,工厂A,2026-06-01,12.3\n"
        ).encode("utf-8")
        invalid_csv = "foo,bar\n1,2\n".encode("utf-8")
        persisted: list[tuple[int, str]] = []

        result = ImportJob(
            persist_func=lambda df, price_col: persisted.append((len(df), price_col)) or len(df),
        ).import_uploaded(
            [
                _UploadedFileStub("valid.csv", valid_csv),
                _UploadedFileStub("broken.csv", invalid_csv),
            ]
        )

        self.assertFalse(result.success)
        self.assertEqual(persisted, [])
        self.assertEqual(result.scanned_file_count, 2)
        self.assertEqual(result.loaded_file_count, 1)
        self.assertEqual(result.failed_file_count, 1)
        self.assertIn("broken.csv", result.message)
        self.assertIn("已阻止本次写库", result.message)

    def test_import_job_blocks_folder_sync_when_any_file_fails(self) -> None:
        with _workspace_temp_dir("import_folder_failure") as folder:
            (folder / "valid.csv").write_text(
                "物料编码,物料名称,适用车系,备件简称,工厂,价格有效于,成本\n"
                "MAT-001,零件A,车系A,简称A,工厂A,2026-06-01,12.3\n",
                encoding="utf-8",
            )
            (folder / "broken.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
            persisted: list[tuple[int, str]] = []

            result = ImportJob(
                persist_func=lambda df, price_col: persisted.append((len(df), price_col)) or len(df),
            ).import_folder(str(folder))

        self.assertFalse(result.success)
        self.assertEqual(persisted, [])
        self.assertEqual(result.loaded_file_count, 1)
        self.assertEqual(result.failed_file_count, 1)
        self.assertIn("broken.csv", result.message)
        self.assertIn("部分文件未导入", result.message)

    def test_import_job_scans_nested_folder_files_before_sync(self) -> None:
        with _workspace_temp_dir("import_nested_folder") as folder:
            nested = folder / "nested"
            nested.mkdir()
            (nested / "valid.csv").write_text(
                "物料编码,物料名称,适用车系,备件简称,工厂,价格有效于,成本\n"
                "MAT-001,零件A,车系A,简称A,工厂A,2026-06-01,12.3\n",
                encoding="utf-8",
            )
            persisted: list[tuple[int, str]] = []

            result = ImportJob(
                persist_func=lambda df, price_col: persisted.append((len(df), price_col)) or len(df),
            ).import_folder(str(folder))

        self.assertTrue(result.success)
        self.assertEqual(result.scanned_file_count, 1)
        self.assertEqual(result.loaded_file_count, 1)
        self.assertEqual(persisted, [(1, "成本")])

    def test_core_cost_prepare_preserves_distinct_supplier_rows_with_same_material_factory_date(self) -> None:
        raw_df = pd.DataFrame(
            [
                {
                    "物料编码": "MAT-001",
                    "物料名称": "零件A",
                    "适用车系": "车系A",
                    "备件简称": "简称A",
                    "供应商名称": "供应商A",
                    "供应商代码": "SUP-A",
                    "工厂": "X990",
                    "价格": 100.0,
                    "monitor_date": "2026-06-01",
                    "价格有效期至": "2026-12-31",
                },
                {
                    "物料编码": "MAT-001",
                    "物料名称": "零件A",
                    "适用车系": "车系A",
                    "备件简称": "简称A",
                    "供应商名称": "供应商B",
                    "供应商代码": "SUP-B",
                    "工厂": "X990",
                    "价格": 101.0,
                    "monitor_date": "2026-06-01",
                    "价格有效期至": "2026-12-31",
                },
            ]
        )

        prepared, resolved_price_col = _prepare_core_cost_records(raw_df, "价格")

        self.assertEqual(resolved_price_col, "价格")
        self.assertEqual(len(prepared), 2)
        self.assertIn("source_row_hash", prepared.columns)
        self.assertEqual(prepared["source_row_hash"].nunique(), 2)
        self.assertEqual(prepared["supplier_code"].tolist(), ["SUP-A", "SUP-B"])

    def test_llm_call_falls_back_to_second_env_config_when_primary_request_fails(self) -> None:
        class FakeSettings:
            llm_api_key = "primary-key"
            llm_api_base_url = "https://primary.example/direct"
            llm_api_model = "primary-model"
            llm_api_direct_url = True
            llm_api_configs = [
                {
                    "name": "primary",
                    "api_key": "primary-key",
                    "base_url": "https://primary.example/direct",
                    "model": "primary-model",
                    "direct_url": True,
                },
                {
                    "name": "backup",
                    "api_key": "backup-key",
                    "base_url": "https://backup.example",
                    "model": "backup-model",
                    "direct_url": False,
                },
            ]
            llm_timeout_seconds = 3
            llm_temperature = 0.2

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "{\"rule_content\":\"ok\"}"}}]}

        calls: list[dict] = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            if len(calls) == 1:
                raise RuntimeError("primary network down")
            return FakeResponse()

        with (
            patch("llm_engine.settings", FakeSettings()),
            patch("llm_engine.requests.post", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            result = _call_llm("test prompt")

        self.assertEqual(result["rule_content"], "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["url"], "https://primary.example/direct")
        self.assertEqual(calls[1]["url"], "https://backup.example/v1/chat/completions")
        self.assertEqual(calls[1]["json"]["model"], "backup-model")
        self.assertEqual(calls[1]["headers"]["Authorization"], "Bearer backup-key")

    def test_byd_llm_wildcard_config_posts_to_documented_endpoint_with_chat_model(self) -> None:
        class FakeSettings:
            llm_api_key = "primary-key"
            llm_api_base_url = "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/*"
            llm_api_model = "deepseek-chat"
            llm_api_direct_url = True
            llm_api_configs = [
                {
                    "name": "DeepSeek-V4-Flash",
                    "api_key": "primary-key",
                    "base_url": "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/*",
                    "model": "deepseek-chat",
                    "direct_url": True,
                }
            ]
            llm_timeout_seconds = 45
            llm_temperature = 0.2

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "{\"ok\":true}"}}]}

        calls: list[dict] = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

        with patch("llm_engine.settings", FakeSettings()), patch("llm_engine.requests.post", side_effect=fake_post):
            response = _post_chat_completion_with_fallback([{"role": "user", "content": "ping"}], temperature=0.0)

        self.assertEqual(response["choices"][0]["message"]["content"], "{\"ok\":true}")
        self.assertEqual(calls[0]["url"], "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/chat/completions")
        self.assertEqual(calls[0]["json"]["model"], "deepseek-chat")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer primary-key")

    def test_qwen_env_config_can_be_ranked_first_with_no_think_flag(self) -> None:
        configs = _build_llm_api_configs(
            {
                "LLM_API_KEY": "qwen-key",
                "LLM_API_URL": "https://eisapi.byd.com/open-api/1.0/llm/v1/qwen-open-14b/*",
                "LLM_API_MODEL": "qwen3-14b-awq",
                "LLM_API_DIRECT_URL": "true",
                "LLM_API_NAME": "Qwen3-14B",
                "LLM_API_NO_THINK": "true",
                "LLM_API_2_KEY": "deepseek-key",
                "LLM_API_2_URL": "https://api.deepseek.com",
                "LLM_API_2_MODEL": "deepseek-v4-pro",
                "LLM_API_2_DIRECT_URL": "false",
                "LLM_API_2_NAME": "deepseek-v4-pro",
                "LLM_API_3_KEY": "flash-key",
                "LLM_API_3_URL": "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/*",
                "LLM_API_3_MODEL": "deepseek-chat",
                "LLM_API_3_DIRECT_URL": "true",
                "LLM_API_3_NAME": "DeepSeek-V4-Flash",
            }
        )

        self.assertEqual([config["name"] for config in configs], ["Qwen3-14B", "deepseek-v4-pro", "DeepSeek-V4-Flash"])
        self.assertEqual(configs[0]["base_url"], "https://eisapi.byd.com/open-api/1.0/llm/v1/qwen-open-14b/*")
        self.assertEqual(configs[0]["model"], "qwen3-14b-awq")
        self.assertTrue(configs[0]["direct_url"])
        self.assertTrue(configs[0]["append_no_think"])
        self.assertFalse(configs[1]["direct_url"])
        self.assertFalse(configs[1]["append_no_think"])
        self.assertTrue(configs[2]["direct_url"])

    def test_qwen_no_think_config_appends_suffix_to_user_messages_only(self) -> None:
        class FakeSettings:
            llm_api_key = "qwen-key"
            llm_api_base_url = "https://eisapi.byd.com/open-api/1.0/llm/v1/qwen-open-14b/*"
            llm_api_model = "qwen3-14b-awq"
            llm_api_direct_url = True
            llm_api_configs = [
                {
                    "name": "Qwen3-14B",
                    "api_key": "qwen-key",
                    "base_url": "https://eisapi.byd.com/open-api/1.0/llm/v1/qwen-open-14b/*",
                    "model": "qwen3-14b-awq",
                    "direct_url": True,
                    "append_no_think": True,
                }
            ]
            llm_timeout_seconds = 45
            llm_temperature = 0.2

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "{\"ok\":true}"}}]}

        calls: list[dict] = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "ping"},
        ]
        with patch("llm_engine.settings", FakeSettings()), patch("llm_engine.requests.post", side_effect=fake_post):
            _post_chat_completion_with_fallback(messages, temperature=0.0)

        self.assertEqual(calls[0]["url"], "https://eisapi.byd.com/open-api/1.0/llm/v1/qwen-open-14b/chat/completions")
        self.assertEqual(calls[0]["json"]["model"], "qwen3-14b-awq")
        self.assertEqual(calls[0]["json"]["messages"][0]["content"], "system prompt")
        self.assertEqual(calls[0]["json"]["messages"][1]["content"], "ping/no_think")

    def test_vehicle_market_price_fetch_accepts_json_array_response(self) -> None:
        response_content = json.dumps(
            [
                {
                    "vehicle_series": "汉",
                    "variant_name": "汉 DM-i 次顶配",
                    "market_price": "20.98万",
                    "confidence": 0.82,
                    "basis": "LLM 基于自身知识估算",
                }
            ],
            ensure_ascii=False,
        )

        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch(
                "llm_engine._post_chat_completion_with_fallback",
                return_value={"choices": [{"message": {"content": response_content}}]},
            ),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            rows = fetch_vehicle_market_prices(["汉", "唐"])

        self.assertEqual(len(rows), 2)
        han_row = next(row for row in rows if row["vehicle_series"] == "汉")
        tang_row = next(row for row in rows if row["vehicle_series"] == "唐")
        self.assertEqual(han_row["status"], "LLM估算")
        self.assertEqual(han_row["source_domain"], "")
        self.assertAlmostEqual(han_row["market_price"], 209800.0)
        self.assertEqual(tang_row["status"], "待确认")
        self.assertIn("LLM 未返回该车系", tang_row["failure_reason"])

    def test_vehicle_market_price_prompt_requires_sorted_all_vehicle_estimates(self) -> None:
        captured_messages = []
        response_content = json.dumps(
            [
                {"vehicle_series": "腾势N9", "variant_name": "腾势N9 次顶配", "market_price": "38.98万"},
                {"vehicle_series": "秦PLUS", "variant_name": "秦PLUS 次顶配", "market_price": "12.98万"},
            ],
            ensure_ascii=False,
        )

        def fake_post(messages, *, temperature):
            captured_messages.extend(messages)
            return {"choices": [{"message": {"content": response_content}}]}

        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch("llm_engine._post_chat_completion_with_fallback", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            fetch_vehicle_market_prices(["秦PLUS", "腾势N9"])

        serialized_prompt = json.dumps(captured_messages, ensure_ascii=False)
        self.assertIn("所有车系", serialized_prompt)
        self.assertIn("次顶配", serialized_prompt)
        self.assertIn("从高到低", serialized_prompt)

    def test_vehicle_market_price_prompt_forbids_empty_values_and_prefers_reliable_sources(self) -> None:
        captured_messages = []
        response_content = json.dumps(
            [
                {"vehicle_series": "秦PLUS", "variant_name": "秦PLUS 次顶配", "market_price": "12.98万"},
            ],
            ensure_ascii=False,
        )

        def fake_post(messages, *, temperature):
            captured_messages.extend(messages)
            return {"choices": [{"message": {"content": response_content}}]}

        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch("llm_engine._post_chat_completion_with_fallback", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            fetch_vehicle_market_prices(["秦PLUS"])

        serialized_prompt = json.dumps(captured_messages, ensure_ascii=False)
        self.assertIn("不得为空", serialized_prompt)
        self.assertIn("严禁 null", serialized_prompt)
        self.assertGreaterEqual(serialized_prompt.count("强制"), 3)
        self.assertIn("尽可能准确", serialized_prompt)
        self.assertIn("依据", serialized_prompt)

    def test_vehicle_market_price_fetch_repairs_blank_llm_estimates(self) -> None:
        first_response = json.dumps(
            [
                {"vehicle_series": "汉", "variant_name": "", "market_price": None},
                {"vehicle_series": "唐", "variant_name": "唐 DM-i 次顶配", "market_price": "18.98万"},
            ],
            ensure_ascii=False,
        )
        repair_response = json.dumps(
            [
                {"vehicle_series": "汉", "variant_name": "汉 DM-i 次顶配", "market_price": "20.98万"},
            ],
            ensure_ascii=False,
        )
        calls: list[list[dict[str, str]]] = []

        def fake_post(messages, *, temperature):
            calls.append(messages)
            content = first_response if len(calls) == 1 else repair_response
            return {"choices": [{"message": {"content": content}}]}

        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch("llm_engine._post_chat_completion_with_fallback", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            rows = fetch_vehicle_market_prices(["汉", "唐"])

        han_row = next(row for row in rows if row["vehicle_series"] == "汉")
        self.assertEqual(len(calls), 2)
        repair_payload = json.loads(calls[1][-1]["content"])
        self.assertEqual(repair_payload["vehicle_series"], ["汉"])
        self.assertEqual(han_row["status"], "LLM估算")
        self.assertEqual(han_row["variant_name"], "汉 DM-i 次顶配")
        self.assertEqual(han_row["market_price"], 209800.0)

    def test_vehicle_market_price_recheck_targets_only_blank_estimates(self) -> None:
        display_df = pd.DataFrame(
            [
                {"梯度排名": 1, "车系": "汉", "次顶配车型": "汉 DM-i 次顶配", "估算价格（元）": 209800},
                {"梯度排名": 2, "车系": "唐", "次顶配车型": "", "估算价格（元）": None},
                {"梯度排名": 3, "车系": "宋PLUS", "次顶配车型": "", "估算价格（元）": pd.NA},
                {"梯度排名": 4, "车系": "唐", "次顶配车型": "", "估算价格（元）": ""},
                {"梯度排名": 5, "车系": "海豹", "次顶配车型": "海豹 次顶配", "估算价格（元）": "189800"},
            ]
        )

        missing_vehicle_series = extract_missing_vehicle_market_price_series(display_df)

        self.assertEqual(missing_vehicle_series, ["唐", "宋PLUS"])

    def test_vehicle_market_price_recheck_refreshes_rank_by_completed_prices(self) -> None:
        session_state = {
            "vehicle_rank_manual_order": ["秦PLUS", "腾势N9", "唐"],
            "vehicle_rank": ["秦PLUS", "腾势N9", "唐"],
            "vehicle_rank_text": "秦PLUS\n腾势N9\n唐",
        }
        source_df = pd.DataFrame(
            [
                {"适用车系": "秦PLUS", "成本": 100.0},
                {"适用车系": "腾势N9", "成本": 100.0},
                {"适用车系": "唐", "成本": 100.0},
            ]
        )
        completed_prices = pd.DataFrame(
            [
                {"vehicle_series": "秦PLUS", "market_price": 129800.0},
                {"vehicle_series": "腾势N9", "market_price": 389800.0},
                {"vehicle_series": "唐", "market_price": 239800.0},
            ]
        )

        auto_rank = apply_vehicle_market_price_auto_rank(
            session_state,
            source_df,
            completed_prices,
            price_col="成本",
        )

        self.assertEqual(auto_rank, ["腾势N9", "唐", "秦PLUS"])
        self.assertEqual(session_state["vehicle_rank"], ["腾势N9", "唐", "秦PLUS"])
        self.assertEqual(session_state["vehicle_rank_text"], "腾势N9\n唐\n秦PLUS")
        self.assertEqual(session_state["vehicle_rank_manual_order"], [])

    def test_vehicle_market_price_display_uses_same_chinese_series_fallback_for_local_candidates(self) -> None:
        market_prices = pd.DataFrame(
            [
                {"vehicle_series": "秦L", "market_price": 129800.0, "variant_name": "秦L DM-i 次顶配"},
                {"vehicle_series": "海豹08EV", "market_price": 239800.0, "variant_name": "海豹 700km 性能版"},
                {"vehicle_series": "唐", "market_price": 309800.0, "variant_name": "唐 EV 次顶配"},
                {"vehicle_series": "秦PLUS", "market_price": None, "variant_name": ""},
                {"vehicle_series": "海狮06", "market_price": None, "variant_name": ""},
            ]
        )

        display_df = build_vehicle_market_price_display_df(
            market_prices,
            vehicle_candidates=["秦PLUS", "海豹06", "海狮06"],
        )
        by_vehicle = display_df.set_index("车系")

        self.assertEqual(set(display_df["车系"]), {"秦PLUS", "海豹06", "海狮06"})
        self.assertEqual(int(by_vehicle.loc["秦PLUS", "估算价格（元）"]), 129800)
        self.assertEqual(int(by_vehicle.loc["海豹06", "估算价格（元）"]), 239800)
        self.assertTrue(pd.isna(by_vehicle.loc["海狮06", "估算价格（元）"]))
        self.assertNotIn("唐", set(display_df["车系"]))

    def test_vehicle_market_price_rank_uses_same_chinese_fallback_without_cross_series_match(self) -> None:
        source_df = pd.DataFrame(
            [
                {"适用车系": "秦PLUS", "成本": 100.0},
                {"适用车系": "海狮06", "成本": 100.0},
                {"适用车系": "海豹06", "成本": 100.0},
                {"适用车系": "腾势N9", "成本": 100.0},
            ]
        )
        market_prices = pd.DataFrame(
            [
                {"vehicle_series": "秦L", "market_price": 129800.0},
                {"vehicle_series": "海豹08EV", "market_price": 239800.0},
                {"vehicle_series": "腾势N9", "market_price": 499800.0},
            ]
        )

        rank = build_default_vehicle_rank(source_df, market_prices, price_col="成本")

        self.assertEqual(rank, ["腾势N9", "海豹06", "秦PLUS", "海狮06"])
        display_df = build_vehicle_market_price_display_df(
            market_prices,
            vehicle_candidates=source_df["适用车系"].tolist(),
        )
        self.assertEqual(extract_missing_vehicle_market_price_series(display_df), ["海狮06"])

    def test_vehicle_gradient_deviation_explanations_use_llm_with_local_logic_context(self) -> None:
        response_content = json.dumps(
            [
                {
                    "row_id": "7",
                    "explanation": "按原有逻辑，成本排序第1而梯度排名第2，偏差超过25%，因此需要复核。",
                }
            ],
            ensure_ascii=False,
        )
        captured_messages: list[list[dict[str, str]]] = []

        def fake_post(messages, *, temperature):
            captured_messages.append(messages)
            return {"choices": [{"message": {"content": response_content}}]}

        rows = [
            {
                "row_id": "7",
                "vehicle_series": "车系B",
                "part_name": "门板",
                "gradient_rank": 2,
                "cost_rank": 1,
                "deviation_rate": 0.5,
                "is_abnormal": True,
            }
        ]

        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch("llm_engine._post_chat_completion_with_fallback", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            explanations = explain_vehicle_gradient_deviations(rows)

        self.assertEqual(explanations["7"], "按原有逻辑，成本排序第1而梯度排名第2，偏差超过25%，因此需要复核。")
        serialized_prompt = json.dumps(captured_messages, ensure_ascii=False)
        self.assertIn("原有逻辑", serialized_prompt)
        self.assertIn("25%", serialized_prompt)
        self.assertIn("成本排序", serialized_prompt)

    def test_vehicle_market_price_repair_failure_keeps_primary_successes(self) -> None:
        first_response = json.dumps(
            [
                {"vehicle_series": "汉", "variant_name": "", "market_price": None},
                {"vehicle_series": "唐", "variant_name": "唐 DM-i 次顶配", "market_price": "18.98万"},
            ],
            ensure_ascii=False,
        )
        calls: list[list[dict[str, str]]] = []

        def fake_post(messages, *, temperature):
            calls.append(messages)
            if len(calls) == 1:
                return {"choices": [{"message": {"content": first_response}}]}
            raise RuntimeError("repair timeout")

        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch("llm_engine._post_chat_completion_with_fallback", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            rows = fetch_vehicle_market_prices(["汉", "唐"])

        han_row = next(row for row in rows if row["vehicle_series"] == "汉")
        tang_row = next(row for row in rows if row["vehicle_series"] == "唐")
        self.assertEqual(len(calls), 2)
        self.assertEqual(tang_row["status"], "LLM估算")
        self.assertEqual(tang_row["market_price"], 189800.0)
        self.assertEqual(han_row["status"], "待确认")
        self.assertIsNone(han_row["market_price"])

    def test_vehicle_market_price_fetch_splits_large_vehicle_list_into_batches(self) -> None:
        captured_batches: list[list[str]] = []

        def fake_post(messages, *, temperature):
            payload = json.loads(messages[-1]["content"])
            batch = payload["vehicle_series"]
            captured_batches.append(batch)
            response_content = json.dumps(
                [
                    {
                        "vehicle_series": vehicle,
                        "variant_name": f"{vehicle} 次顶配",
                        "market_price": 100000 + index * 1000,
                    }
                    for index, vehicle in enumerate(batch, start=1)
                ],
                ensure_ascii=False,
            )
            return {"choices": [{"message": {"content": response_content}}]}

        vehicles = [f"车系{index}" for index in range(1, 13)]
        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch("llm_engine._post_chat_completion_with_fallback", side_effect=fake_post),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            rows = fetch_vehicle_market_prices(vehicles)

        self.assertGreater(len(captured_batches), 1)
        self.assertEqual([vehicle for batch in captured_batches for vehicle in batch], vehicles)
        self.assertEqual(len(rows), len(vehicles))
        self.assertTrue(all(row["market_price"] is not None for row in rows))

    def test_vehicle_market_price_fetch_marks_pending_when_all_llm_configs_fail(self) -> None:
        with (
            patch("llm_engine.is_llm_configured", return_value=True),
            patch(
                "llm_engine._post_chat_completion_with_fallback",
                side_effect=RuntimeError("LLM 网络请求失败，所有本地配置均不可用：primary down；backup timeout"),
            ),
            patch("llm_engine.harness.run_llm_action", side_effect=lambda _name, fn, request_payload=None: fn()),
        ):
            rows = fetch_vehicle_market_prices(["汉", "唐"])

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["vehicle_series"] for row in rows}, {"汉", "唐"})
        self.assertEqual({row["status"] for row in rows}, {"待确认"})
        self.assertTrue(all(row["market_price"] is None for row in rows))
        self.assertTrue(all("LLM 网络请求失败" in row["failure_reason"] for row in rows))

    def test_vehicle_market_price_display_sorts_by_price_desc_and_shows_values(self) -> None:
        market_prices = pd.DataFrame(
            [
                {
                    "vehicle_series": "元PLUS",
                    "market_price": 139800.0,
                    "variant_name": "元PLUS 次顶配",
                    "status": "LLM估算",
                    "fetched_at": "2026-06-18 10:00:00",
                },
                {
                    "vehicle_series": "汉",
                    "market_price": 209800.0,
                    "variant_name": "汉 DM-i 次顶配",
                    "status": "LLM估算",
                    "fetched_at": "2026-06-18 10:01:00",
                },
                {
                    "vehicle_series": "唐",
                    "market_price": None,
                    "variant_name": "",
                    "source_url": "",
                    "status": "待确认",
                    "fetched_at": "2026-06-18 10:02:00",
                    "raw_response_json": json.dumps({"failure_reason": "LLM 网络请求失败"}, ensure_ascii=False),
                },
            ]
        )

        display_df = build_vehicle_market_price_display_df(market_prices)

        self.assertEqual(display_df.columns.tolist(), ["梯度排名", "车系", "次顶配车型", "估算价格（元）"])
        self.assertEqual(display_df["车系"].tolist(), ["汉", "元PLUS", "唐"])
        self.assertEqual(display_df["梯度排名"].tolist(), [1, 2, 3])
        self.assertEqual(int(display_df.loc[0, "估算价格（元）"]), 209800)
        self.assertTrue(pd.isna(display_df.loc[2, "估算价格（元）"]))

    def test_vehicle_market_price_display_respects_manual_rank_order(self) -> None:
        market_prices = pd.DataFrame(
            [
                {"vehicle_series": "元PLUS", "market_price": 139800.0, "variant_name": "元PLUS 次顶配"},
                {"vehicle_series": "汉", "market_price": 209800.0, "variant_name": "汉 DM-i 次顶配"},
                {"vehicle_series": "唐", "market_price": 189800.0, "variant_name": "唐 DM-i 次顶配"},
            ]
        )

        display_df = build_vehicle_market_price_display_df(market_prices, rank_order=["唐", "元PLUS", "汉"])

        self.assertEqual(display_df["车系"].tolist(), ["唐", "元PLUS", "汉"])
        self.assertEqual(display_df["梯度排名"].tolist(), [1, 2, 3])

    def test_llm_estimated_prices_override_local_vehicle_read_order_for_rank(self) -> None:
        local_order_df = pd.DataFrame(
            [
                {"适用车系": "秦MAX DM-i", "价格": 100000.0},
                {"适用车系": "腾势N9", "价格": 100000.0},
                {"适用车系": "秦PLUS", "价格": 100000.0},
            ]
        )
        market_prices = pd.DataFrame(
            [
                {"vehicle_series": "秦MAX DM-i", "market_price": 129800.0},
                {"vehicle_series": "腾势N9", "market_price": 389800.0},
                {"vehicle_series": "秦PLUS", "market_price": 119800.0},
            ]
        )

        rank = build_default_vehicle_rank(local_order_df, market_prices, price_col="价格")

        self.assertEqual(rank, ["腾势N9", "秦MAX DM-i", "秦PLUS"])

    def test_manual_vehicle_rank_from_display_uses_edited_rank_numbers(self) -> None:
        edited_display = pd.DataFrame(
            [
                {"梯度排名": 2, "车系": "秦MAX DM-i"},
                {"梯度排名": 1, "车系": "腾势N9"},
                {"梯度排名": 3, "车系": "秦PLUS"},
            ]
        )

        rank = build_manual_vehicle_rank_from_display(edited_display)

        self.assertEqual(rank, ["腾势N9", "秦MAX DM-i", "秦PLUS"])

    def test_manual_vehicle_market_price_rows_from_display_persists_edited_prices(self) -> None:
        edited_display = pd.DataFrame(
            [
                {"梯度排名": 2, "车系": "秦MAX DM-i", "次顶配车型": "秦MAX 次顶配", "估算价格（元）": 139800},
                {"梯度排名": 1, "车系": "腾势N9", "次顶配车型": "腾势N9 次顶配", "估算价格（元）": 489800},
                {"梯度排名": 3, "车系": "秦PLUS", "次顶配车型": "", "估算价格（元）": None},
            ]
        )

        rows = build_manual_vehicle_market_price_rows_from_display(edited_display)
        by_name = {row["vehicle_series"]: row for row in rows}

        self.assertEqual(len(rows), 3)
        self.assertEqual(by_name["腾势N9"]["market_price"], 489800.0)
        self.assertEqual(by_name["腾势N9"]["variant_name"], "腾势N9 次顶配")
        self.assertEqual(by_name["腾势N9"]["status"], "人工修正")
        self.assertIsNone(by_name["秦PLUS"]["market_price"])
        self.assertEqual(by_name["秦PLUS"]["status"], "待确认")

    def test_vehicle_market_price_manual_editor_allows_variant_name_input(self) -> None:
        self.assertEqual(
            get_vehicle_market_price_manual_editable_columns(),
            ["梯度排名", "次顶配车型", "估算价格（元）"],
        )

    def test_cost_anomaly_write_failure_log_does_not_dump_business_rows(self) -> None:
        class FailingEngine:
            def begin(self):
                raise RuntimeError("db locked for SECRET-MAT-001 12345.67")

        result_df = pd.DataFrame(
            [
                {
                    "_record_key": "rk-secret",
                    "物料编码": "SECRET-MAT-001",
                    "备件简称": "敏感件",
                    "实际成本": 12345.67,
                    "预测值": 12000.0,
                    "合理下限": 11000.0,
                    "合理上限": 13000.0,
                    "偏离数值": 345.67,
                    "偏离比例": 0.0288,
                    "status": "异常偏高",
                }
            ]
        )
        output = io.StringIO()

        with (
            patch("storage_service._ensure_cost_anomaly_results_table", return_value=None),
            patch("storage_service._ensure_cost_anomaly_result_runs_table", return_value=None),
            patch("storage_service.require_db_engine", return_value=FailingEngine()),
            contextlib.redirect_stdout(output),
            self.assertRaises(RuntimeError),
        ):
            save_cost_anomaly_results(result_df, result_mode="raw")

        log_text = output.getvalue()
        self.assertIn("[cost_anomaly_results] 写入失败", log_text)
        self.assertIn("待写入行数=1", log_text)
        self.assertNotIn("SECRET-MAT-001", log_text)
        self.assertNotIn("12345.67", log_text)
        self.assertNotIn("待写入前5行", log_text)

    def test_sqlite_engine_uses_busy_timeout_and_wal_pragmas(self) -> None:
        from storage_service import SQLITE_CONNECTION_PRAGMAS, _build_sqlite_connect_args

        connect_args = _build_sqlite_connect_args()

        self.assertGreaterEqual(connect_args.get("timeout", 0), 30)
        self.assertIn("PRAGMA journal_mode=WAL", SQLITE_CONNECTION_PRAGMAS)
        self.assertIn("PRAGMA busy_timeout=30000", SQLITE_CONNECTION_PRAGMAS)

    def test_gitignore_excludes_generated_data_cache(self) -> None:
        gitignore_text = Path(".gitignore").read_text(encoding="utf-8")

        self.assertIn("data/cache/", gitignore_text)
        self.assertIn("data/benchmark_runtime/", gitignore_text)

    def test_database_health_warning_only_reports_actionable_bloat(self) -> None:
        from general_pages import build_database_health_warning

        self.assertEqual(build_database_health_warning({"ok": True, "freelist_ratio": 0.01, "wal_mb": 0.0}), "")

        warning = build_database_health_warning(
            {
                "ok": False,
                "freelist_ratio": 0.91,
                "wal_mb": 80.0,
            }
        )

        self.assertIn("空闲页", warning)
        self.assertIn("WAL", warning)

    def test_runtime_governance_bootstrap_is_disabled_by_default(self) -> None:
        from app_context import maybe_bootstrap_runtime_governance

        state = {}
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("app_context.harness.bootstrap_runtime_governance") as bootstrap,
        ):
            payload = maybe_bootstrap_runtime_governance(state)

        bootstrap.assert_not_called()
        self.assertEqual(payload, {"enabled": False})
        self.assertIsNone(state.get("_harness_audit_result"))

    def test_runtime_governance_bootstrap_runs_when_env_flag_is_enabled(self) -> None:
        from app_context import maybe_bootstrap_runtime_governance

        state = {}
        expected_payload = {"audit_result": {"status": "success"}}
        with (
            patch.dict(os.environ, {"COST_MONITOR_RUNTIME_GOVERNANCE": "1"}, clear=True),
            patch("app_context.harness.bootstrap_runtime_governance", return_value=expected_payload) as bootstrap,
        ):
            payload = maybe_bootstrap_runtime_governance(state)

        bootstrap.assert_called_once_with(state)
        self.assertEqual(payload, {"enabled": True, **expected_payload})

    def test_loaded_data_state_can_be_released_without_dropping_navigation(self) -> None:
        from app_context import clear_loaded_data_state

        state = {
            "data": pd.DataFrame({"物料编码": ["MAT-001"]}),
            "price_col": "成本",
            "loaded_data_origin": "local_db",
            "current_page": "系统设置",
            "_deferred_download_payload_cost": b"xlsx",
            "_deferred_download_file_cost": "cost.xlsx",
        }

        result = clear_loaded_data_state(state)

        self.assertEqual(result["released_rows"], 1)
        self.assertIsNone(state["data"])
        self.assertEqual(state["price_col"], "")
        self.assertEqual(state["loaded_data_origin"], "")
        self.assertEqual(state["current_page"], "系统设置")
        self.assertNotIn("_deferred_download_payload_cost", state)
        self.assertNotIn("_deferred_download_file_cost", state)

    def test_parallel_detection_is_disabled_by_default_for_low_spec_desktop(self) -> None:
        from anomaly_engine import _should_use_parallel_detection

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("anomaly_engine.os.cpu_count", return_value=8),
        ):
            self.assertFalse(_should_use_parallel_detection(total_rows=1_000_000, group_count=1_000))

        with (
            patch.dict(os.environ, {"COST_MONITOR_ENABLE_PROCESS_POOL": "1"}, clear=True),
            patch("anomaly_engine.os.cpu_count", return_value=8),
        ):
            self.assertTrue(_should_use_parallel_detection(total_rows=1_000_000, group_count=1_000))

    def test_compute_job_returns_fresh_persisted_cost_anomaly_without_recompute(self) -> None:
        source_df = pd.DataFrame(
            {
                "物料编码": ["MAT-001"],
                "备件简称": ["门板"],
                "工厂": ["F1"],
                "monitor_date": pd.to_datetime(["2026-06-01"]),
                "成本": [1.0],
            }
        )
        persisted_df = pd.DataFrame({"status": ["persisted"]})
        calls: list[str] = []

        job = ComputeJob(
            raw_detector=lambda df, price_col: calls.append(f"detect:{price_col}") or pd.DataFrame(),
            fresh_result_loader=lambda result_mode, source_signature, options_signature: (
                calls.append(f"fresh:{result_mode}:{bool(source_signature)}:{bool(options_signature)}") or persisted_df
            ),
        )

        result = job.run_cost_anomaly(source_df, "成本", result_mode="raw")

        self.assertEqual(calls, ["fresh:raw:True:True"])
        self.assertEqual(result["status"].tolist(), ["persisted"])

    def test_compute_job_records_result_run_metadata_after_compute(self) -> None:
        source_df = pd.DataFrame(
            {
                "物料编码": ["MAT-001"],
                "备件简称": ["门板"],
                "工厂": ["F1"],
                "monitor_date": pd.to_datetime(["2026-06-01"]),
                "成本": [1.0],
            }
        )
        computed_df = pd.DataFrame({"status": ["computed"]})
        persisted_df = pd.DataFrame({"status": ["persisted"]})
        calls: list[tuple[str, str, str]] = []

        job = ComputeJob(
            raw_detector=lambda df, price_col: calls.append(("detect", price_col, "")) or computed_df,
            fresh_result_loader=lambda result_mode, source_signature, options_signature: (
                calls.append(("fresh", result_mode, source_signature)) or pd.DataFrame()
            ),
            raw_loader=lambda result_mode: calls.append(("load", result_mode, "")) or persisted_df,
            result_run_recorder=lambda result_mode, source_signature, options_signature, row_count: calls.append(
                ("mark", result_mode, f"{bool(source_signature)}:{bool(options_signature)}:{row_count}")
            ),
        )

        result = job.run_cost_anomaly(source_df, "成本", result_mode="raw")

        self.assertEqual(calls[0][0], "fresh")
        self.assertEqual(calls[1], ("detect", "成本", ""))
        self.assertEqual(calls[2], ("mark", "raw", "True:True:1"))
        self.assertEqual(calls[3], ("load", "raw", ""))
        self.assertEqual(result["status"].tolist(), ["persisted"])

    def test_weighted_cost_anomaly_options_signature_tracks_parameters(self) -> None:
        source_df = pd.DataFrame(
            {
                "物料编码": ["MAT-001"],
                "备件简称": ["门板"],
                "工厂": ["F1"],
                "monitor_date": pd.to_datetime(["2026-06-01"]),
                "成本": [1.0],
            }
        )
        option_signatures: list[str] = []
        job = ComputeJob(
            weighted_detector=lambda df, price_col, labels, **kwargs: pd.DataFrame({"status": ["computed"]}),
            fresh_result_loader=lambda result_mode, source_signature, options_signature: (
                option_signatures.append(options_signature) or pd.DataFrame()
            ),
            raw_loader=lambda result_mode: pd.DataFrame({"status": ["persisted"]}),
            result_run_recorder=lambda result_mode, source_signature, options_signature, row_count: None,
        )

        job.run_weighted_cost_anomaly(source_df, "成本", tuple(), result_mode="weighted", sigma_multiplier=1.0)
        job.run_weighted_cost_anomaly(source_df, "成本", tuple(), result_mode="weighted", sigma_multiplier=2.0)

        self.assertEqual(len(option_signatures), 2)
        self.assertNotEqual(option_signatures[0], option_signatures[1])

    def test_compute_job_precomputes_cost_skills_and_filters_labeled_groups(self) -> None:
        anomaly_df = pd.DataFrame(
            {
                "备件简称": ["门板", "门板", "灯具"],
                "_record_key": ["rk-1", "rk-2", "rk-3"],
            }
        )
        skill_rows = [
            {"备件简称": "门板", "当前σ参数": 1.0},
            {"备件简称": "灯具", "当前σ参数": 1.0},
        ]
        job = ComputeJob(cost_skill_extractor=lambda df, labels: skill_rows)

        result = job.precompute_cost_skills(anomaly_df, {"rk-2": "正常"})

        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.covered_count, 1)
        self.assertEqual([skill["备件简称"] for skill in result.all_skills], ["门板", "灯具"])
        self.assertEqual([skill["备件简称"] for skill in result.export_skills], ["门板"])

    def test_compute_job_precomputes_sheet_metal_skills_and_filters_labeled_groups(self) -> None:
        review_df = pd.DataFrame(
            {
                "备件简称": ["钣金A", "钣金A", "钣金B"],
                "_record_key": ["rk-a1", "rk-a2", "rk-b1"],
            }
        )
        skill_rows = [
            {"备件简称": "钣金A", "当前σ参数": 1.2},
            {"备件简称": "钣金B", "当前σ参数": 1.2},
        ]
        calls: list[tuple[float, int]] = []

        def fake_extractor(df, labels, sigma_multiplier, expert_weight):
            calls.append((sigma_multiplier, expert_weight))
            return skill_rows

        job = ComputeJob(sheet_metal_skill_extractor=fake_extractor)

        result = job.precompute_sheet_metal_skills(
            review_df,
            {"rk-b1": "正常"},
            sigma_multiplier=1.2,
            expert_weight=90,
        )

        self.assertEqual(calls, [(1.2, 90)])
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.covered_count, 1)
        self.assertEqual([skill["备件简称"] for skill in result.all_skills], ["钣金A", "钣金B"])
        self.assertEqual([skill["备件简称"] for skill in result.export_skills], ["钣金B"])

    def test_accepted_ring_union_marks_only_main_or_secondary_intervals_normal(self) -> None:
        ring_payload = json.dumps(
            [
                {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 95.0, "合理上限": 105.0, "预测值": 100.0},
                {"圈层编号": 2, "圈层角色": "次邻居圈", "合理下限": 495.0, "合理上限": 505.0, "预测值": 500.0},
            ],
            ensure_ascii=False,
        )
        frame = pd.DataFrame(
            {
                "实际成本": [100.0, 150.0, 500.0, 650.0],
                "预测值": [100.0, 100.0, 100.0, 100.0],
                "合理下限": [95.0, 95.0, 95.0, 95.0],
                "合理上限": [105.0, 105.0, 105.0, 105.0],
                "偏离数值": [0.0, 50.0, 400.0, 550.0],
                "偏离比例": [0.0, 0.5, 4.0, 5.5],
                "status": ["异常偏高", "异常偏高", "异常偏高", "异常偏高"],
                "圈层角色": ["未采纳孤立圈"] * 4,
                "多圈合理区间": [ring_payload] * 4,
            }
        )

        result = apply_accepted_ring_union_status(frame, value_col="实际成本")

        self.assertEqual(result.loc[0, "status"], "正常（主邻居圈）")
        self.assertEqual(result.loc[2, "status"], "正常（次邻居圈）")
        self.assertIn("异常", result.loc[1, "status"])
        self.assertIn("异常", result.loc[3, "status"])

    def test_sheet_metal_uses_same_ring_union_logic(self) -> None:
        ring_payload = json.dumps(
            [
                {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 10.0, "合理上限": 12.0, "预测值": 11.0},
                {"圈层编号": 2, "圈层角色": "次邻居圈", "合理下限": 24.0, "合理上限": 26.0, "预测值": 25.0},
            ],
            ensure_ascii=False,
        )
        frame = pd.DataFrame(
            {
                "白痴指数": [11.0, 18.0, 25.0],
                "基准指数": [11.0, 11.0, 11.0],
                "合理下限": [10.0, 10.0, 10.0],
                "合理上限": [12.0, 12.0, 12.0],
                "偏离指数": [0.0, 7.0, 14.0],
                "偏离比例": [0.0, 0.6, 1.2],
                "status": ["异常偏高", "异常偏高", "异常偏高"],
                "圈层角色": ["未采纳孤立圈"] * 3,
                "多圈合理区间": [ring_payload] * 3,
            }
        )

        result = apply_accepted_ring_union_status(
            frame,
            value_col="白痴指数",
            baseline_col="基准指数",
            deviation_col="偏离指数",
        )

        self.assertEqual(result.loc[0, "status"], "正常（主邻居圈）")
        self.assertEqual(result.loc[2, "status"], "正常（次邻居圈）")
        self.assertIn("异常", result.loc[1, "status"])

    def test_feedback_rows_marked_normal_require_remarks(self) -> None:
        rows = pd.DataFrame(
            [
                {"record_key": "rk-ok", "label": "正常", "remark": "材质差异导致合理"},
                {"record_key": "rk-missing", "label": "正常", "remark": "   "},
                {"record_key": "rk-other", "label": "", "remark": ""},
            ]
        )

        missing = find_feedback_rows_missing_required_remarks(rows)

        self.assertEqual(missing, ["rk-missing"])

    def test_clear_cost_feedback_also_clears_ai_knowledge_and_cached_analysis(self) -> None:
        with (
            patch("app_context.harness.execute_action") as execute_action,
            patch("app_context.cached_enrich_anomaly_with_ai.clear") as clear_ai_cache,
        ):
            clear_cost_feedback_and_ai_state()

        executed_actions = [call.args[0] for call in execute_action.call_args_list]
        self.assertEqual(executed_actions, ["clear_feedback", "clear_expert_knowledge_base"])
        clear_ai_cache.assert_called_once()

    def test_cost_skills_excel_flattens_dynamic_ring_columns(self) -> None:
        skills = [
            {
                "备件简称": "A件",
                "适用算法": "算法A",
                "当前σ参数": 1.0,
                "偏置权重": 80,
                "时序敏感度 (Decay Alpha)": 1.0,
                "圈子严格度 (Gap K)": 4.0,
                "Baseline Quantile": 0.5,
                "本组专家标注数": 1,
                "成本合理区间边界": {"预测值": 10.0, "合理下限": 9.0, "合理上限": 11.0},
                "数据结构分布描述": {"样本量": 3, "均值": 10.0},
                "异常统计": {"正常": 2, "异常偏高": 1, "异常偏低": 0},
                "语义校准报告": {"引用规律数": 1, "主要匹配方式": ["规则"], "参考文本规律": ["原因"]},
                "多邻居圈合理区间": [
                    {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 9.0, "预测值": 10.0, "合理上限": 11.0, "样本量": 2, "圈层置信度": 1.0},
                    {"圈层编号": 2, "圈层角色": "次邻居圈", "合理下限": 20.0, "预测值": 21.0, "合理上限": 22.0, "样本量": 1, "圈层置信度": 0.8},
                ],
                "经验对齐率": "N/A",
            },
            {
                "备件简称": "B件",
                "适用算法": "算法B",
                "当前σ参数": 1.0,
                "偏置权重": 80,
                "本组专家标注数": 0,
                "成本合理区间边界": {"预测值": 30.0, "合理下限": 29.0, "合理上限": 31.0},
                "数据结构分布描述": {"样本量": 1},
                "异常统计": {"正常": 1},
                "多邻居圈合理区间": [],
            },
        ]

        exported = pd.read_excel(io.BytesIO(skills_to_excel_bytes(skills)))

        self.assertIn("邻居圈2_角色", exported.columns)
        self.assertEqual(exported.loc[0, "邻居圈2_角色"], "次邻居圈")
        self.assertTrue(pd.isna(exported.loc[1, "邻居圈2_角色"]))

    def test_sheet_metal_skills_excel_uses_same_flattening_rules(self) -> None:
        skills = [
            {
                "备件简称": "钣金A",
                "适用算法": "算法",
                "当前σ参数": 1.0,
                "偏置权重": 80,
                "本组专家标注数": 0,
                "白痴指数合理区间": {"基准指数": 10.0, "合理下限": 9.0, "合理上限": 11.0},
                "白痴指数分布描述": {"样本量": 2, "均值": 10.0},
                "异常统计": {"正常": 2},
                "多邻居圈合理区间": [
                    {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 9.0, "预测值": 10.0, "合理上限": 11.0}
                ],
                "经验对齐率": "N/A",
            }
        ]

        exported = pd.read_excel(io.BytesIO(sheet_metal_skills_to_excel_bytes(skills)))

        self.assertIn("合理区间_基准指数", exported.columns)
        self.assertIn("邻居圈1_角色", exported.columns)

    def test_cost_skills_excel_artifacts_write_model_and_expert_report_paths(self) -> None:
        skills = [
            {
                "备件简称": "门板",
                "适用算法": "算法",
                "当前σ参数": 2.5,
                "偏置权重": 120,
                "时序敏感度 (Decay Alpha)": 1.2,
                "圈子严格度 (Gap K)": 4.5,
                "Baseline Quantile": 0.55,
                "本组专家标注数": 1,
                "成本合理区间边界": {"预测值": 100.0, "合理下限": 90.0, "合理上限": 110.0},
                "数据结构分布描述": {"样本量": 3},
                "异常统计": {"正常": 2, "异常偏高": 1},
                "多邻居圈合理区间": [
                    {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 90.0, "预测值": 100.0, "合理上限": 110.0},
                ],
            }
        ]
        base_dir = Path(".test_tmp") / "skills_artifacts"
        shutil.rmtree(base_dir, ignore_errors=True)
        model_dir = base_dir / "model"
        report_dir = base_dir / "report"
        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = export_cost_skills_excel_artifacts(
                skills,
                model_export_path=str(model_dir),
                expert_report_export_path=str(report_dir),
                generated_at=pd.Timestamp("2026-06-22 10:11:12"),
                force_new=True,
            )

            model_path = Path(result["model_export_path"])
            report_path = Path(result["expert_report_export_path"])
            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(model_path.parent, Path(model_dir))
            self.assertEqual(report_path.parent, Path(report_dir))
            self.assertEqual(model_path.suffix, ".xlsx")
            self.assertEqual(report_path.suffix, ".xlsx")
            self.assertIn("全量", model_path.name)
            self.assertIn("优化后", report_path.name)
            exported = pd.read_excel(report_path)
            self.assertEqual(exported.loc[0, "备件简称"], "门板")
            self.assertEqual(float(exported.loc[0, "合理区间_合理下限"]), 90.0)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_latest_cost_skills_excel_loader_reads_newest_file_and_builds_interval_overrides(self) -> None:
        old_skills = [
            {
                "备件简称": "门板",
                "适用算法": "旧算法",
                "当前σ参数": 1.0,
                "偏置权重": 80,
                "时序敏感度 (Decay Alpha)": 1.0,
                "圈子严格度 (Gap K)": 4.0,
                "Baseline Quantile": 0.5,
                "本组专家标注数": 0,
                "成本合理区间边界": {"预测值": 10.0, "合理下限": 9.0, "合理上限": 11.0},
                "数据结构分布描述": {},
                "异常统计": {},
                "多邻居圈合理区间": [],
            }
        ]
        new_skills = [
            {
                "备件简称": "门板",
                "适用算法": "新算法",
                "当前σ参数": 3.3,
                "偏置权重": 180,
                "时序敏感度 (Decay Alpha)": 1.7,
                "圈子严格度 (Gap K)": 6.0,
                "Baseline Quantile": 0.6,
                "本组专家标注数": 2,
                "成本合理区间边界": {"预测值": 100.0, "合理下限": 90.0, "合理上限": 110.0},
                "数据结构分布描述": {},
                "异常统计": {},
                "多邻居圈合理区间": [
                    {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 90.0, "预测值": 100.0, "合理上限": 110.0},
                    {"圈层编号": 2, "圈层角色": "次邻居圈", "合理下限": 490.0, "预测值": 500.0, "合理上限": 510.0},
                ],
            }
        ]
        report_dir = Path(".test_tmp") / "skills_latest_report"
        shutil.rmtree(report_dir, ignore_errors=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        try:
            old_path = report_dir / "成本区间Skills_优化后_20260101_000000.xlsx"
            new_path = report_dir / "成本区间Skills_优化后_20260622_101112.xlsx"
            old_path.write_bytes(skills_to_excel_bytes(old_skills))
            new_path.write_bytes(skills_to_excel_bytes(new_skills))

            loaded = load_latest_cost_skills_excel(str(report_dir))
            overrides_json = build_cost_skill_overrides_json(loaded)
            overrides = json.loads(overrides_json)
        finally:
            shutil.rmtree(report_dir, ignore_errors=True)

        self.assertIsNotNone(loaded)
        self.assertEqual(Path(loaded["source_path"]).name, new_path.name)
        self.assertEqual(loaded["skills"][0]["备件简称"], "门板")
        self.assertEqual(float(loaded["skills"][0]["当前σ参数"]), 3.3)
        self.assertEqual(len(loaded["skills"][0]["多邻居圈合理区间"]), 2)
        self.assertEqual(overrides["门板"]["sigma"], 3.3)
        self.assertEqual(len(overrides["门板"]["fixed_intervals"]), 2)

    def test_weighted_detection_applies_fixed_intervals_loaded_from_cost_skills_excel(self) -> None:
        frame = _group_frame([100.0, 500.0, 750.0], short_name="门板")
        frame["_is_expert_normal"] = [False, False, False]
        result = _detect_group_anomalies_worker(
            frame,
            {
                "weighted": True,
                "fixed_intervals": [
                    {"圈层编号": 1, "圈层角色": "主邻居圈", "合理下限": 90.0, "预测值": 100.0, "合理上限": 110.0},
                    {"圈层编号": 2, "圈层角色": "次邻居圈", "合理下限": 490.0, "预测值": 500.0, "合理上限": 510.0},
                ],
                "group_source": "专家经验报告Excel",
            },
        )

        self.assertEqual(result.loc[result["实际成本"].eq(100.0), "status"].iloc[0], "正常（主邻居圈）")
        self.assertEqual(result.loc[result["实际成本"].eq(500.0), "status"].iloc[0], "正常（次邻居圈）")
        self.assertIn("异常", result.loc[result["实际成本"].eq(750.0), "status"].iloc[0])
        self.assertEqual(set(result["判定依据"]), {"专家经验报告Excel"})

    def test_auto_research_history_includes_expert_short_name_scope(self) -> None:
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "物料编码": [f"DOOR-{idx:03d}" for idx in range(12)],
                        "物料名称": ["门板"] * 12,
                        "适用车系": ["汉"] * 12,
                        "工厂": ["X990"] * 12,
                        "备件简称": ["门板"] * 12,
                        "成本": np.linspace(95.0, 106.0, 12),
                        "monitor_date": pd.date_range("2026-01-01", periods=12, freq="D"),
                    }
                ),
                pd.DataFrame(
                    {
                        "物料编码": [f"LAMP-{idx:03d}" for idx in range(12)],
                        "物料名称": ["灯具"] * 12,
                        "适用车系": ["唐"] * 12,
                        "工厂": ["A990"] * 12,
                        "备件简称": ["灯具"] * 12,
                        "成本": np.linspace(45.0, 56.0, 12),
                        "monitor_date": pd.date_range("2026-02-01", periods=12, freq="D"),
                    }
                ),
            ],
            ignore_index=True,
        )
        key_frame = frame.rename(columns={"成本": "实际成本", "monitor_date": "价格有效于"})
        expert_labels = {
            make_record_key(key_frame.iloc[0]): "正常",
            make_record_key(key_frame.iloc[12]): "正常",
        }

        result = run_auto_research(frame, "成本", expert_labels, n_iterations=0)

        self.assertEqual(result["history"][0]["备件简称"], "门板、灯具")

    def test_builtin_templates_are_valid_excel_workbooks(self) -> None:
        for template_name, expected_columns in {
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
        }.items():
            template_df = pd.read_excel(io.BytesIO(build_builtin_template_excel_bytes(template_name)))
            self.assertEqual(template_df.columns.tolist(), expected_columns)

    def test_settings_path_management_groups_separate_import_and_cost_export_paths(self) -> None:
        from general_pages import build_path_management_groups

        groups = build_path_management_groups()
        group_keys = {group["title"]: [field["setting_key"] for field in group["fields"]] for group in groups}

        self.assertEqual(
            group_keys["导入路径"],
            ["input_data_path", "sheet_metal_base_info_path", "assembly_data_path"],
        )
        self.assertEqual(
            group_keys["备件成本导出路径"],
            ["quantitative_skills_path", "qualitative_skills_path"],
        )
        self.assertEqual(
            group_keys["钣金模块导出路径"],
            ["sheet_metal_model_export_path", "sheet_metal_report_export_path"],
        )

    def test_cost_template_columns_can_be_imported_without_column_renaming(self) -> None:
        template_df = pd.DataFrame(
            [
                {
                    "物料编码": "MAT-001",
                    "物料名称": "零件A",
                    "适用车系": "车系A",
                    "备件简称": "简称A",
                    "供应商名称": "供应商A",
                    "供应商代码": "SUP-001",
                    "工厂": "工厂A",
                    "价格": 12.3,
                    "价格有效期于": "2026-06-01",
                    "价格有效期至": "2026-12-31",
                }
            ]
        )

        processed_df, price_col, error_msg = process_dataframe(template_df)

        self.assertIsNone(error_msg)
        self.assertEqual(price_col, "价格")
        self.assertEqual(str(processed_df.loc[0, "一级总成供应商代码"]), "SUP-001")
        self.assertEqual(pd.Timestamp(processed_df.loc[0, "monitor_date"]), pd.Timestamp("2026-06-01"))

    def test_full_cost_report_displays_period_delta_next_to_price_changes(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "物料编码": "MAT-001",
                    "物料名称": "零件A",
                    "适用车系": "车系A",
                    "备件简称": "简称A",
                    "工厂": "工厂A",
                    "价格变动1": "250.0",
                    "价格变动2": "300.0",
                    "价格变动3": "250.0",
                }
            ]
        )

        html = render_merged_html_table(
            report_df,
            ["价格变动1", "价格变动2", "价格变动3"],
            is_trend_mode=False,
        )

        self.assertIn("300.00（▲ +50.00）", html)
        self.assertIn("250.00（▼ -50.00）", html)
        self.assertIn("#e74c3c", html)
        self.assertIn("#27ae60", html)

    def test_full_cost_report_prioritizes_latest_cost_increases(self) -> None:
        source_df = pd.DataFrame(
            [
                {"物料编码": "MAT-DOWN", "物料名称": "下降件", "适用车系": "车系A", "备件简称": "下降件", "工厂": "总装", "monitor_date": "2026-01-01", "成本": 200.0},
                {"物料编码": "MAT-DOWN", "物料名称": "下降件", "适用车系": "车系A", "备件简称": "下降件", "工厂": "总装", "monitor_date": "2026-02-01", "成本": 180.0},
                {"物料编码": "MAT-UP", "物料名称": "上涨件", "适用车系": "车系B", "备件简称": "上涨件", "工厂": "总装", "monitor_date": "2026-01-01", "成本": 100.0},
                {"物料编码": "MAT-UP", "物料名称": "上涨件", "适用车系": "车系B", "备件简称": "上涨件", "工厂": "总装", "monitor_date": "2026-02-01", "成本": 130.0},
            ]
        )

        report_df = generate_pivot_report(source_df, "成本")
        prioritized_df = prioritize_latest_cost_increases(report_df)

        self.assertEqual(prioritized_df["物料编码"].tolist()[0], "MAT-UP")

    def test_latest_cost_increase_export_returns_only_rising_details(self) -> None:
        report_df = pd.DataFrame(
            [
                {"物料编码": "MAT-UP", "物料名称": "上涨件", "适用车系": "车系B", "备件简称": "上涨件", "工厂": "总装", "价格变动1": 100.0, "价格变动2": 130.0},
                {"物料编码": "MAT-FLAT", "物料名称": "持平件", "适用车系": "车系C", "备件简称": "持平件", "工厂": "总装", "价格变动1": 90.0, "价格变动2": 90.0},
                {"物料编码": "MAT-DOWN", "物料名称": "下降件", "适用车系": "车系A", "备件简称": "下降件", "工厂": "总装", "价格变动1": 200.0, "价格变动2": 180.0},
            ]
        )

        export_df = filter_latest_cost_increase_rows(report_df)

        self.assertEqual(export_df["物料编码"].tolist(), ["MAT-UP"])
        self.assertEqual(export_df["最新成本"].tolist(), [130.0])
        self.assertEqual(export_df["上期成本"].tolist(), [100.0])
        self.assertEqual(export_df["最新变动额"].tolist(), [30.0])

    def test_direct_llm_url_keeps_configured_wildcard_pattern(self) -> None:
        url = _normalize_direct_llm_url("https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/*")

        self.assertEqual(url, "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/*")

    def test_byd_wildcard_direct_url_resolves_to_chat_completions_endpoint(self) -> None:
        url = _chat_completions_url(
            {
                "base_url": "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/*",
                "direct_url": True,
            }
        )

        self.assertEqual(url, "https://eisapi.byd.com/open-api/1.0/llm/v1/deepseek-v4-flash/chat/completions")

    def test_assembly_summary_sorts_by_cost_ratio_desc_and_exports_abnormal_parents(self) -> None:
        summary_df = pd.DataFrame(
            [
                {"层级0编码": "P-LOW", "成本比例": 0.8},
                {"层级0编码": "P-HIGH", "成本比例": 1.35},
                {"层级0编码": "P-MISSING", "成本比例": np.nan},
            ]
        )
        export_df = pd.DataFrame(
            [
                {"层级0编码": "P-HIGH", "层级": "层级0", "零件名称": "高比例总成"},
                {"层级0编码": "P-HIGH", "层级": "层级1", "零件名称": "高比例子件"},
                {"层级0编码": "P-LOW", "层级": "层级0", "零件名称": "低比例总成"},
            ]
        )

        sorted_summary = sort_assembly_summary_by_cost_ratio(summary_df)
        abnormal_export = build_abnormal_ratio_export_df(summary_df, export_df)

        self.assertEqual(sorted_summary["层级0编码"].tolist(), ["P-HIGH", "P-LOW", "P-MISSING"])
        self.assertEqual(abnormal_export["层级0编码"].tolist(), ["P-HIGH", "P-HIGH"])

    def test_cost_anomaly_scope_requires_selected_short_names_and_limits_factories(self) -> None:
        raw_df = pd.DataFrame(
            [
                {"备件简称": "A件", "工厂": "X990", "物料编码": "A-X"},
                {"备件简称": "A件", "工厂": "总装", "物料编码": "A-Z"},
                {"备件简称": "B件", "工厂": "a990", "物料编码": "B-A"},
                {"备件简称": "C件", "工厂": "C990", "物料编码": "C-C"},
            ]
        )

        empty_scope = filter_cost_anomaly_scope(raw_df, selected_short_names=[], include_all_short_names=False)
        selected_scope = filter_cost_anomaly_scope(raw_df, selected_short_names=["A件", "B件"], include_all_short_names=False)
        all_scope = filter_cost_anomaly_scope(raw_df, selected_short_names=[], include_all_short_names=True)

        self.assertTrue(empty_scope.empty)
        self.assertEqual(selected_scope["物料编码"].tolist(), ["A-X", "B-A"])
        self.assertEqual(all_scope["物料编码"].tolist(), ["A-X", "B-A"])

    def test_cost_anomaly_run_request_requires_explicit_calculate_click(self) -> None:
        idle = build_cost_anomaly_run_request(
            selected_short_names=["A件"],
            calculate_selected_clicked=False,
            calculate_all_clicked=False,
        )
        selected = build_cost_anomaly_run_request(
            selected_short_names=["A件", "B件"],
            calculate_selected_clicked=True,
            calculate_all_clicked=False,
        )
        all_names = build_cost_anomaly_run_request(
            selected_short_names=["A件"],
            calculate_selected_clicked=False,
            calculate_all_clicked=True,
        )

        self.assertFalse(idle["should_run"])
        self.assertTrue(selected["should_run"])
        self.assertFalse(selected["include_all_short_names"])
        self.assertEqual(selected["selected_short_names"], ["A件", "B件"])
        self.assertEqual(selected["scope_label"], "所选2个简称")
        self.assertTrue(all_names["include_all_short_names"])
        self.assertEqual(all_names["selected_short_names"], [])
        self.assertEqual(all_names["scope_label"], "全量简称")

    def test_cost_anomaly_result_export_writes_excel_to_model_path(self) -> None:
        result_df = pd.DataFrame(
            [
                {"备件简称": "门板", "物料编码": "MAT-001", "status": "正常", "合理下限": 90.0, "合理上限": 110.0},
            ]
        )

        with _workspace_temp_dir("cost_anomaly_result_export") as export_dir:
            output_path = export_cost_anomaly_result_excel(
                result_df,
                str(export_dir),
                scope_label="全量简称",
                generated_at=pd.Timestamp("2026-06-25 13:20:00"),
            )
            exported = pd.read_excel(output_path)

        self.assertIn("成本异常监控_全量简称_20260625_132000", Path(output_path).name)
        self.assertEqual(exported.loc[0, "备件简称"], "门板")
        self.assertEqual(float(exported.loc[0, "合理下限"]), 90.0)

    def test_sheet_metal_review_run_request_requires_explicit_calculate_click(self) -> None:
        idle = build_sheet_metal_review_run_request(
            selected_short_names=["翼子板"],
            calculate_selected_clicked=False,
            calculate_all_clicked=False,
        )
        selected = build_sheet_metal_review_run_request(
            selected_short_names=["翼子板"],
            calculate_selected_clicked=True,
            calculate_all_clicked=False,
        )
        all_names = build_sheet_metal_review_run_request(
            selected_short_names=["翼子板"],
            calculate_selected_clicked=False,
            calculate_all_clicked=True,
        )

        self.assertFalse(idle["should_run"])
        self.assertTrue(selected["should_run"])
        self.assertEqual(selected["selected_short_names"], ["翼子板"])
        self.assertEqual(selected["scope_label"], "所选1个简称")
        self.assertTrue(all_names["include_all_short_names"])
        self.assertEqual(all_names["selected_short_names"], [])
        self.assertEqual(all_names["scope_label"], "全量简称")

    def test_sheet_metal_review_result_export_writes_excel_to_model_path(self) -> None:
        result_df = pd.DataFrame(
            [
                {"备件简称": "翼子板", "物料编码": "SM-001", "status": "正常", "合理下限": 9.0, "合理上限": 11.0},
            ]
        )

        with _workspace_temp_dir("sheet_metal_review_result_export") as export_dir:
            output_path = export_sheet_metal_review_result_excel(
                result_df,
                str(export_dir),
                scope_label="全量简称",
                generated_at=pd.Timestamp("2026-06-25 13:21:00"),
            )
            exported = pd.read_excel(output_path)

        self.assertIn("钣金件白痴指数复核_全量简称_20260625_132100", Path(output_path).name)
        self.assertEqual(exported.loc[0, "备件简称"], "翼子板")
        self.assertEqual(float(exported.loc[0, "合理上限"]), 11.0)

    def test_cost_anomaly_chart_scope_uses_multi_selected_short_names(self) -> None:
        working_df = pd.DataFrame(
            [
                {"备件简称": "A件", "物料编码": "A-1"},
                {"备件简称": "B件", "物料编码": "B-1"},
                {"备件简称": "C件", "物料编码": "C-1"},
            ]
        )
        monitoring_df = pd.DataFrame(
            [
                {"备件简称": "A件", "物料编码": "A-1"},
                {"备件简称": "B件", "物料编码": "B-1"},
            ]
        )

        chart_df, monitoring_chart_df, chart_title = build_cost_anomaly_chart_scope(
            working_df,
            monitoring_df,
            selected_short_names=["A件", "B件"],
            include_all_short_names=False,
        )

        self.assertEqual(chart_df["物料编码"].tolist(), ["A-1", "B-1"])
        self.assertEqual(monitoring_chart_df["物料编码"].tolist(), ["A-1", "B-1"])
        self.assertEqual(chart_title, "A件、B件 - 成本分布")

    def test_interval_compare_part_mode_groups_rings_on_vehicle_axis(self) -> None:
        chart_df = pd.DataFrame(
            [
                {"适用车系": "汉", "备件简称": "门板", "圈层显示": "#1 主邻居圈"},
                {"适用车系": "汉", "备件简称": "门板", "圈层显示": "#2 次邻居圈"},
                {"适用车系": "唐", "备件简称": "门板", "圈层显示": "#1 主邻居圈"},
            ]
        )

        result = build_interval_compare_display_labels(chart_df, compare_mode="part")

        self.assertEqual(result["显示标签"].tolist(), ["汉", "汉", "唐"])
        self.assertEqual(result.loc[result["适用车系"] == "汉", "显示标签"].nunique(), 1)
        self.assertNotIn("邻居圈", result.loc[0, "显示标签"])

    def test_interval_lower_bound_rank_analysis_compares_part_order_against_reversed_vehicle_gradient(self) -> None:
        interval_df = pd.DataFrame(
            [
                {"适用车系": "低配车", "备件简称": "门板", "圈层角色": "主邻居圈", "圈层编号": 1, "合理下限": 300.0, "预测值": 320.0, "合理上限": 360.0},
                {"适用车系": "中配车", "备件简称": "门板", "圈层角色": "主邻居圈", "圈层编号": 1, "合理下限": 100.0, "预测值": 130.0, "合理上限": 160.0},
                {"适用车系": "高配车", "备件简称": "门板", "圈层角色": "主邻居圈", "圈层编号": 1, "合理下限": 200.0, "预测值": 220.0, "合理上限": 260.0},
                {"适用车系": "低配车", "备件简称": "灯具", "圈层角色": "主邻居圈", "圈层编号": 1, "合理下限": 80.0, "预测值": 90.0, "合理上限": 100.0},
                {"适用车系": "中配车", "备件简称": "灯具", "圈层角色": "主邻居圈", "圈层编号": 1, "合理下限": 120.0, "预测值": 130.0, "合理上限": 140.0},
                {"适用车系": "高配车", "备件简称": "灯具", "圈层角色": "主邻居圈", "圈层编号": 1, "合理下限": 180.0, "预测值": 190.0, "合理上限": 200.0},
                {"适用车系": "低配车", "备件简称": "门板", "圈层角色": "次邻居圈", "圈层编号": 2, "合理下限": 1.0, "预测值": 2.0, "合理上限": 3.0},
            ]
        )
        vehicle_rank_order_map = {
            "高配车": 1,
            "中配车": 2,
            "低配车": 3,
        }

        summary_df, detail_df, heatmap_df = build_interval_lower_bound_rank_analysis(
            interval_df,
            vehicle_rank_order_map,
        )

        detail_key = detail_df.set_index(["备件简称", "适用车系"])
        self.assertEqual(int(detail_key.loc[("门板", "中配车"), "区间下限排名"]), 1)
        self.assertEqual(int(detail_key.loc[("门板", "中配车"), "期望低到高排名"]), 2)
        self.assertEqual(int(detail_key.loc[("门板", "中配车"), "排名偏差"]), -1)
        self.assertEqual(detail_key.loc[("门板", "中配车"), "偏差方向"], "整体偏低")
        self.assertEqual(int(detail_key.loc[("门板", "低配车"), "合理下限"]), 300)
        self.assertNotEqual(int(detail_key.loc[("门板", "低配车"), "合理下限"]), 1)

        summary_key = summary_df.set_index("备件简称")
        self.assertEqual(int(summary_key.loc["门板", "最大排名偏差"]), 2)
        self.assertEqual(int(summary_key.loc["灯具", "最大排名偏差"]), 0)
        self.assertLess(float(summary_key.loc["门板", "排序一致性"]), float(summary_key.loc["灯具", "排序一致性"]))
        self.assertEqual(int(summary_key.loc["门板", "异常车系数"]), 3)

        self.assertEqual(heatmap_df.loc[heatmap_df["备件简称"].eq("门板"), "低配车"].iloc[0], 2)
        self.assertEqual(heatmap_df.loc[heatmap_df["备件简称"].eq("灯具"), "高配车"].iloc[0], 0)

    def test_vehicle_series_candidates_use_first_name_from_multi_value_field(self) -> None:
        df = pd.DataFrame(
            [
                {"适用车系": "汉、唐", "成本": 10.0},
                {"适用车系": " 海豹 / 宋 ", "成本": 20.0},
                {"适用车系": "", "成本": 30.0},
            ]
        )

        self.assertEqual(extract_first_vehicle_series_name("汉、唐"), "汉")
        self.assertEqual(extract_first_vehicle_series_name(" 海豹 / 宋 "), "海豹")
        self.assertEqual(extract_vehicle_rank_candidates(df), ["汉", "海豹"])

    def test_vehicle_gradient_comparison_sorts_by_gradient_and_hides_cost_rank(self) -> None:
        df = pd.DataFrame(
            [
                {"备件简称": "门板", "适用车系": "车系A", "物料编码": "A", "工厂": "X990", "monitor_date": "2026-01-01", "成本": 100.0},
                {"备件简称": "门板", "适用车系": "车系B", "物料编码": "B", "工厂": "X990", "monitor_date": "2026-01-01", "成本": 300.0},
                {"备件简称": "门板", "适用车系": "车系C", "物料编码": "C", "工厂": "X990", "monitor_date": "2026-01-01", "成本": 200.0},
            ]
        )

        result = get_vehicle_gradient_comparison(df, "成本", "门板", ["车系A", "车系B", "车系C"])

        self.assertNotIn("成本序号", result.columns)
        self.assertNotIn("梯度偏差率", result.columns)
        self.assertEqual(result["适用车系"].tolist(), ["车系A", "车系B", "车系C"])
        self.assertEqual(result["梯度排名"].tolist(), [1, 2, 3])
        self.assertTrue(bool(result.loc[1, "梯度偏差异常"]))

    def test_interval_compare_vehicle_mode_groups_rings_on_part_axis_and_sorts_by_lower_bound(self) -> None:
        chart_df = pd.DataFrame(
            [
                {"适用车系": "汉", "备件简称": "高价件", "圈层编号": 1, "圈层显示": "#1 主邻居圈", "合理下限": 300.0},
                {"适用车系": "汉", "备件简称": "低价件", "圈层编号": 1, "圈层显示": "#1 主邻居圈", "合理下限": 100.0},
                {"适用车系": "汉", "备件简称": "低价件", "圈层编号": 2, "圈层显示": "#2 次邻居圈", "合理下限": 120.0},
            ]
        )

        labeled = build_interval_compare_display_labels(chart_df, compare_mode="vehicle")
        sorted_df = sort_interval_compare_chart_data(labeled, mode="vehicle")

        self.assertEqual(labeled.loc[labeled["备件简称"] == "低价件", "显示标签"].nunique(), 1)
        self.assertEqual(sorted_df["备件简称"].tolist(), ["低价件", "低价件", "高价件"])
        self.assertNotIn("邻居圈", labeled.loc[0, "显示标签"])

    def test_calibration_management_uses_source_context_when_record_key_only_has_identity(self) -> None:
        source_row = {
            "物料编码": "MAT-001",
            "物料名称": "左前门板",
            "适用车系": "汉",
            "备件简称": "门板",
            "工厂": "X990",
            "价格有效于": "2026-06-01",
            "实际成本": 123.456,
        }
        record_key = make_record_key(source_row)

        result = build_calibration_management_df(
            {record_key: {"label": "正常", "remark": "铝板材质差异"}},
            fallback_source_df=pd.DataFrame([source_row]),
            fallback_price_col="实际成本",
        )

        self.assertEqual(result.loc[0, "物料编码"], "MAT-001")
        self.assertEqual(result.loc[0, "物料名称"], "左前门板")
        self.assertEqual(result.loc[0, "备件简称"], "门板")

    def test_calibration_management_backfills_names_when_exact_match_has_blank_placeholders(self) -> None:
        record_key = make_record_key(
            {
                "物料编码": "MAT-001",
                "工厂": "X990",
                "价格有效于": "2026-06-01",
                "实际成本": 100.0,
            }
        )
        exact_blank_context = pd.DataFrame(
            [
                {
                    "_record_key": record_key,
                    "物料编码": "MAT-001",
                    "物料名称": "",
                    "备件简称": "nan",
                    "工厂": "X990",
                    "价格有效于": "2026-06-01",
                    "实际成本": 100.0,
                }
            ]
        )
        core_context = pd.DataFrame(
            [
                {
                    "物料编码": "MAT-001",
                    "物料名称": "左前门板总成",
                    "备件简称": "门板",
                    "工厂": "A990",
                    "monitor_date": "2026-05-01",
                    "成本": 98.0,
                }
            ]
        )

        result = build_calibration_management_df(
            {record_key: {"label": "正常", "remark": "材质差异"}},
            core_source_df=core_context,
            fallback_source_df=exact_blank_context,
            fallback_price_col="实际成本",
        )

        self.assertEqual(result.loc[0, "物料名称"], "左前门板总成")
        self.assertEqual(result.loc[0, "备件简称"], "门板")
        self.assertNotIn("nan", result.astype(str).agg("|".join, axis=1).iloc[0].lower())

    def test_calibration_management_omits_orphan_labels_without_current_source_context(self) -> None:
        record_key = make_record_key(
            {
                "物料编码": "STALE-001",
                "工厂": "X990",
                "价格有效于": "2026-06-01",
                "实际成本": 100.0,
            }
        )

        result = build_calibration_management_df(
            {record_key: {"label": "正常", "remark": "旧数据批注"}},
            core_source_df=pd.DataFrame(columns=["物料编码", "物料名称", "备件简称", "工厂", "成本", "monitor_date"]),
            fallback_source_df=pd.DataFrame(columns=["物料编码", "物料名称", "备件简称", "工厂", "实际成本", "价格有效于"]),
            fallback_price_col="实际成本",
        )

        self.assertTrue(result.empty)

    def test_label_details_filter_uses_only_visible_management_rows(self) -> None:
        active_key = make_record_key(
            {
                "物料编码": "MAT-001",
                "工厂": "X990",
                "价格有效于": "2026-06-01",
                "实际成本": 100.0,
            }
        )
        stale_key = make_record_key(
            {
                "物料编码": "STALE-001",
                "工厂": "X990",
                "价格有效于": "2026-06-01",
                "实际成本": 100.0,
            }
        )
        management_df = pd.DataFrame([{"record_key": active_key, "物料名称": "左前门板", "备件简称": "门板"}])

        active_details = filter_label_details_to_management_rows(
            {
                f"raw::{active_key}": {"label": "正常", "remark": "有效"},
                stale_key: {"label": "正常", "remark": "旧数据"},
            },
            management_df,
        )

        self.assertEqual(list(active_details), [f"raw::{active_key}"])

    def test_result_mode_prefixed_record_key_is_canonicalized_and_split(self) -> None:
        source_row = {
            "物料编码": "MAT-001",
            "工厂": "X990",
            "价格有效于": "2026-06-01",
            "实际成本": 123.456,
        }
        raw_key = make_record_key(source_row)
        prefixed_key = f"raw::{raw_key}"

        self.assertEqual(canonicalize_record_key(prefixed_key), raw_key)
        parsed = split_record_key(prefixed_key)
        self.assertEqual(parsed["物料编码"], "MAT-001")
        self.assertEqual(parsed["工厂"], "X990")
        self.assertEqual(parsed["_join_date_key"], "2026-06-01")
        self.assertEqual(parsed["_join_price_key"], "123.4560")

    def test_calibration_management_hydrates_all_columns_from_prefixed_anomaly_result_key(self) -> None:
        raw_key = make_record_key(
            {
                "物料编码": "MAT-001",
                "工厂": "X990",
                "价格有效于": "2026-06-01",
                "实际成本": 123.456,
            }
        )
        prefixed_key = f"raw::{raw_key}"
        anomaly_context = pd.DataFrame(
            [
                {
                    "_record_key": prefixed_key,
                    "物料编码": "MAT-001",
                    "物料名称": "左前门板",
                    "备件简称": "门板",
                    "工厂": "X990",
                    "价格有效于": "2026-06-01",
                    "实际成本": 123.456,
                }
            ]
        )

        result = build_calibration_management_df(
            {prefixed_key: {"label": "正常", "remark": "铝板材质差异"}},
            fallback_source_df=anomaly_context,
            fallback_price_col="实际成本",
        )

        self.assertEqual(result.loc[0, "物料编码"], "MAT-001")
        self.assertEqual(result.loc[0, "物料名称"], "左前门板")
        self.assertEqual(result.loc[0, "备件简称"], "门板")
        self.assertEqual(result.loc[0, "工厂"], "X990")
        self.assertEqual(str(result.loc[0, "价格有效期于"])[:10], "2026-06-01")
        self.assertAlmostEqual(float(result.loc[0, "价格"]), 123.456)

    def test_feedback_knowledge_loader_hydrates_prefixed_key_from_raw_results_when_weighted_misses(self) -> None:
        source_row = {
            "物料编码": "MAT-001",
            "物料名称": "左前门板",
            "适用车系": "汉",
            "备件简称": "门板",
            "工厂": "X990",
            "价格有效于": "2026-06-01",
            "实际成本": 123.456,
            "供应商名称": "供应商A",
            "供应商代码": "SUP-001",
        }
        raw_key = make_record_key(source_row)
        prefixed_key = f"raw::{raw_key}"
        feedback_df = pd.DataFrame(
            [
                {
                    "record_key": prefixed_key,
                    "label": "正常",
                    "remark": "铝板材质差异",
                    "labeled_at": pd.Timestamp("2026-06-02"),
                }
            ]
        )
        weighted_df = pd.DataFrame(
            [
                {
                    "_record_key": "weighted::other-key",
                    "物料编码": "OTHER",
                    "备件简称": "其他件",
                    "适用车系": "汉",
                    "实际成本": 1.0,
                    "预测值": 1.0,
                    "物料名称": "其他",
                    "status": "异常偏高",
                    "价格有效于": "2026-06-01",
                }
            ]
        )
        raw_df = pd.DataFrame(
            [
                {
                    "_record_key": prefixed_key,
                    "物料编码": "MAT-001",
                    "物料名称": "左前门板",
                    "备件简称": "门板",
                    "适用车系": "汉",
                    "实际成本": 123.456,
                    "预测值": 100.0,
                    "status": "异常偏高",
                    "价格有效于": "2026-06-01",
                }
            ]
        )
        core_df = pd.DataFrame([source_row]).rename(columns={"实际成本": "成本"})
        core_df["monitor_date"] = pd.to_datetime(core_df["价格有效于"])

        def fake_execute(action: str, **kwargs):
            if action == "get_feedback_records":
                return feedback_df
            if action == "load_cost_anomaly_results":
                return weighted_df if kwargs.get("result_mode") == "weighted" else raw_df
            if action == "load_core_cost_records":
                return core_df, "成本", None
            raise AssertionError(f"Unexpected action: {action}")

        with patch("llm_engine.harness.execute_action", side_effect=fake_execute):
            records = _load_feedback_records_for_knowledge()

        self.assertEqual(len(records), 1)
        self.assertEqual(records.loc[0, "record_key"], raw_key)
        self.assertEqual(records.loc[0, "material_code"], "MAT-001")
        self.assertEqual(records.loc[0, "material_name"], "左前门板")
        self.assertEqual(records.loc[0, "short_name"], "门板")
        self.assertEqual(records.loc[0, "vehicle_series"], "汉")
        self.assertEqual(records.loc[0, "supplier_name"], "供应商A")
        self.assertEqual(records.loc[0, "supplier_code"], "SUP-001")

    def test_feedback_knowledge_loader_replaces_nan_placeholders_with_core_context(self) -> None:
        source_row = {
            "物料编码": "MAT-001",
            "物料名称": "左前门板",
            "适用车系": "汉",
            "备件简称": "门板",
            "工厂": "X990",
            "价格有效于": "2026-06-01",
            "实际成本": 123.456,
            "供应商名称": "供应商A",
            "供应商代码": "SUP-001",
        }
        raw_key = make_record_key(source_row)
        feedback_df = pd.DataFrame(
            [
                {
                    "record_key": raw_key,
                    "label": "正常",
                    "remark": "铝板材质差异",
                    "labeled_at": pd.Timestamp("2026-06-02"),
                }
            ]
        )
        raw_df = pd.DataFrame(
            [
                {
                    "_record_key": raw_key,
                    "物料编码": "MAT-001",
                    "物料名称": "nan",
                    "备件简称": "nan",
                    "适用车系": "nan",
                    "实际成本": 123.456,
                    "预测值": 100.0,
                    "status": "异常偏高",
                    "价格有效于": "2026-06-01",
                }
            ]
        )
        core_df = pd.DataFrame([source_row]).rename(columns={"实际成本": "成本"})
        core_df["monitor_date"] = pd.to_datetime(core_df["价格有效于"])

        def fake_execute(action: str, **kwargs):
            if action == "get_feedback_records":
                return feedback_df
            if action == "load_cost_anomaly_results":
                return pd.DataFrame() if kwargs.get("result_mode") == "weighted" else raw_df
            if action == "load_core_cost_records":
                return core_df, "成本", None
            raise AssertionError(f"Unexpected action: {action}")

        with patch("llm_engine.harness.execute_action", side_effect=fake_execute):
            records = _load_feedback_records_for_knowledge()

        self.assertEqual(records.loc[0, "material_name"], "左前门板")
        self.assertEqual(records.loc[0, "short_name"], "门板")
        self.assertEqual(records.loc[0, "vehicle_series"], "汉")
        self.assertNotIn("nan", records[["material_name", "short_name", "vehicle_series"]].astype(str).agg("|".join, axis=1).iloc[0].lower())

    def test_feedback_knowledge_loader_omits_orphan_feedback_without_cost_context(self) -> None:
        stale_key = make_record_key(
            {
                "物料编码": "STALE-001",
                "工厂": "X990",
                "价格有效于": "2026-06-01",
                "实际成本": 100.0,
            }
        )
        feedback_df = pd.DataFrame(
            [
                {
                    "record_key": stale_key,
                    "label": "正常",
                    "remark": "旧数据批注",
                    "labeled_at": pd.Timestamp("2026-06-02"),
                }
            ]
        )

        def fake_execute(action: str, **kwargs):
            if action == "get_feedback_records":
                return feedback_df
            if action == "load_cost_anomaly_results":
                return pd.DataFrame()
            if action == "load_core_cost_records":
                return pd.DataFrame(), "成本", None
            raise AssertionError(f"Unexpected action: {action}")

        with patch("llm_engine.harness.execute_action", side_effect=fake_execute):
            records = _load_feedback_records_for_knowledge()

        self.assertTrue(records.empty)

    def test_expert_knowledge_payload_groups_by_vehicle_supplier_and_short_name_with_material_context(self) -> None:
        records_df = pd.DataFrame(
            [
                {
                    "record_key": "rk-1",
                    "material_code": "MAT-001",
                    "material_name": "左前门板",
                    "short_name": "门板",
                    "vehicle_series": "汉",
                    "supplier_code": "SUP-001",
                    "supplier_name": "供应商A",
                    "status": "异常偏高",
                    "remark": "铝板材质差异",
                    "labeled_at": pd.Timestamp("2026-06-01"),
                },
                {
                    "record_key": "rk-2",
                    "material_code": "MAT-002",
                    "material_name": "右前门板",
                    "short_name": "门板",
                    "vehicle_series": "汉",
                    "supplier_code": "SUP-001",
                    "supplier_name": "供应商A",
                    "status": "异常偏高",
                    "remark": "同供应商模具摊销",
                    "labeled_at": pd.Timestamp("2026-06-02"),
                },
            ]
        )

        payloads = _build_group_payloads(records_df)

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload["short_name"], "门板")
        self.assertEqual(payload["vehicle_series"], "汉")
        self.assertEqual(payload["supplier_code"], "SUP-001")
        self.assertIn("MAT-001", payload["material_codes"])
        self.assertEqual(payload["representative_material_code"], "MAT-001")
        self.assertIn("汉车系门板类备件对于供应商A来说普遍异常", payload["rule_template"])

    def test_load_expert_knowledge_base_normalizes_nan_placeholders_for_display(self) -> None:
        import storage_service
        from sqlalchemy import create_engine

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "knowledge.db"
            engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
            EXPERT_KNOWLEDGE_BASE_TABLE.create(engine)
            with engine.begin() as conn:
                conn.execute(
                    EXPERT_KNOWLEDGE_BASE_TABLE.insert(),
                    [
                        {
                            "rule_id": "rule-nan",
                            "short_name": "nan",
                            "material_code": "nan",
                            "material_name": "None",
                            "supplier_code": "",
                            "supplier_name": "NULL",
                            "vehicle_series": "nan",
                            "rule_content": "当备件存在配置较高、结构复杂时需要复核",
                            "confidence_score": 0.75,
                            "updated_at": pd.Timestamp("2026-06-24 10:00:00").to_pydatetime(),
                        },
                        {
                            "rule_id": "rule-ok",
                            "short_name": "气帘",
                            "material_code": "15919779-00",
                            "material_name": "UJUW-960384-气帘-星空蓝",
                            "supplier_code": "GD6480",
                            "supplier_name": "欣旺达-华北",
                            "vehicle_series": "海豹06",
                            "rule_content": "海豹06气帘结构复杂，需要结合供应商报价复核",
                            "confidence_score": 0.85,
                            "updated_at": pd.Timestamp("2026-06-24 11:00:00").to_pydatetime(),
                        }
                    ],
                )

            try:
                with (
                    patch.object(storage_service, "DB_ENGINE", engine),
                    patch.object(storage_service.harness, "authorize_db_operation", return_value={}),
                ):
                    knowledge_df = load_expert_knowledge_base()
            finally:
                engine.dispose()

        self.assertEqual(knowledge_df["rule_id"].tolist(), ["rule-ok"])
        self.assertEqual(knowledge_df.loc[0, "short_name"], "气帘")
        self.assertEqual(knowledge_df.loc[0, "material_code"], "15919779-00")
        self.assertEqual(knowledge_df.loc[0, "material_name"], "UJUW-960384-气帘-星空蓝")
        self.assertEqual(knowledge_df.loc[0, "supplier_name"], "欣旺达-华北")
        self.assertEqual(knowledge_df.loc[0, "vehicle_series"], "海豹06")
        visible_text = knowledge_df[
            ["short_name", "material_code", "material_name", "supplier_code", "supplier_name", "vehicle_series"]
        ].astype(str).agg("|".join, axis=1).iloc[0].lower()
        self.assertNotIn("nan", visible_text)

    def test_ai_analysis_references_same_vehicle_supplier_material_and_expert_remark(self) -> None:
        knowledge_df = pd.DataFrame(
            [
                {
                    "rule_id": "rule-1",
                    "short_name": "门板",
                    "supplier_code": "SUP-001",
                    "supplier_name": "供应商A",
                    "vehicle_series": "汉",
                    "material_code": "MAT-001",
                    "material_name": "左前门板",
                    "rule_content": "铝板材质差异导致成本偏高",
                    "confidence_score": 0.9,
                    "updated_at": pd.Timestamp("2026-06-03"),
                }
            ]
        )
        anomaly_record = {
            "物料编码": "MAT-999",
            "备件简称": "门板",
            "适用车系": "汉",
            "供应商代码": "SUP-001",
            "供应商名称": "供应商A",
            "status": "异常偏高",
            "实际成本": 150.0,
            "预测值": 100.0,
        }

        reason = infer_anomaly_reason(anomaly_record, knowledge_df=knowledge_df)

        self.assertIn("MAT-001", reason)
        self.assertIn("同车系", reason)
        self.assertIn("同供应商", reason)
        self.assertIn("铝板材质差异导致成本偏高", reason)

    def test_cost_audit_report_only_includes_labeled_records(self) -> None:
        original_df = pd.DataFrame(
            [
                {"_record_key": "rk-labeled", "物料编码": "MAT-001", "物料名称": "门板A", "备件简称": "门板", "实际成本": 150.0, "status": "异常偏高"},
                {"_record_key": "rk-unlabeled", "物料编码": "MAT-002", "物料名称": "门板B", "备件简称": "门板", "实际成本": 90.0, "status": "正常"},
            ]
        )
        optimized_df = original_df.copy()
        optimized_df["status"] = ["正常（专家校准）", "正常"]
        optimized_df["AI 辅助分析"] = ["参考专家经验", ""]

        report = generate_audit_report(original_df, optimized_df, {"rk-labeled": "正常"})

        self.assertEqual(report["物料编码"].tolist(), ["MAT-001"])
        self.assertEqual(report["专家反馈"].tolist(), ["正常"])

    def test_sheet_metal_audit_report_only_includes_labeled_records(self) -> None:
        original_df = pd.DataFrame(
            [
                {"_record_key": "rk-labeled", "物料编码": "SM-001", "物料描述": "翼子板A", "备件简称": "翼子板", "工厂": "F1", "白痴指数": 12.0, "status": "异常偏高"},
                {"_record_key": "rk-unlabeled", "物料编码": "SM-002", "物料描述": "翼子板B", "备件简称": "翼子板", "工厂": "F1", "白痴指数": 10.0, "status": "正常"},
            ]
        )
        optimized_df = original_df.copy()
        optimized_df["status"] = ["正常（专家校准）", "正常"]

        report = build_sheet_metal_audit_report(original_df, optimized_df, expert_labels={"rk-labeled": "正常"})

        self.assertEqual(report["物料编码"].tolist(), ["SM-001"])
        self.assertEqual(report["原始结论"].tolist(), ["异常偏高"])

    def test_sheet_metal_calibration_management_uses_material_description_as_name(self) -> None:
        source_row = {
            "物料编码": "SM-001",
            "物料描述": "左翼子板总成",
            "备件简称": "翼子板",
            "工厂": "F1",
            "白痴指数": 12.0,
            "价格有效于": "2026-06-01",
        }
        record_key = make_record_key(
            {
                "物料编码": "SM-001",
                "工厂": "F1",
                "价格有效于": "2026-06-01",
                "实际成本": 12.0,
            }
        )

        result = build_sheet_metal_calibration_management_df(
            {record_key: {"label": "正常", "remark": "材质差异"}},
            source_df=pd.DataFrame([source_row]),
        )

        self.assertEqual(result.loc[0, "物料名称"], "左翼子板总成")
        self.assertEqual(result.loc[0, "备件简称"], "翼子板")

    def test_vehicle_rank_config_persists_in_sqlite(self) -> None:
        previous = load_vehicle_rank_config()
        try:
            save_vehicle_rank_config(
                [
                    {"vehicle_series": "车系A", "rank_order": 1, "source": "unit-test"},
                    {"vehicle_series": "车系B", "rank_order": 2, "source": "unit-test"},
                ]
            )

            loaded = load_vehicle_rank_config()

            self.assertEqual(loaded["vehicle_series"].tolist(), ["车系A", "车系B"])
            self.assertEqual(loaded["rank_order"].tolist(), [1, 2])
        finally:
            save_vehicle_rank_config(previous.to_dict(orient="records") if not previous.empty else [])

    def test_vehicle_market_price_result_accepts_llm_estimate_without_official_source_url(self) -> None:
        without_source = normalize_vehicle_market_price_result(
            {
                "vehicle_series": "汉",
                "market_price": "21.98万",
                "variant_name": "次顶配",
                "confidence": 0.9,
            }
        )
        non_official_source = normalize_vehicle_market_price_result(
            {
                "vehicle_series": "唐",
                "market_price": "18.98万",
                "variant_name": "次顶配",
                "source_url": "https://example.com/tang",
            }
        )

        self.assertEqual(without_source["status"], "LLM估算")
        self.assertEqual(without_source["source_domain"], "")
        self.assertEqual(without_source["market_price"], 219800.0)
        self.assertEqual(non_official_source["status"], "LLM估算")
        self.assertEqual(non_official_source["market_price"], 189800.0)


if __name__ == "__main__":
    unittest.main()
