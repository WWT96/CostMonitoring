import io
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import harness
from compute_jobs import ComputeJob
import llm_engine
import skills_engine
import ui_utils
from anomaly_engine import DEFAULT_DECAY_ALPHA, DEFAULT_GAP_K, calculate_recency_weight_series, parse_accepted_ring_intervals
from app_context import (
    NO_DATA_WARNING,
    bump_cost_refresh_token,
    build_calibration_management_df,
    cached_anomaly_report,
    cached_anomaly_report_weighted,
    cached_enrich_anomaly_with_ai,
    clear_cost_feedback_and_ai_state,
    ensure_selectbox_state,
    get_cost_refresh_token,
    inject_css,
    render_knowledge_sync_status,
    require_price_col,
    reset_session_key,
    sync_ai_knowledge_base,
)
from config import settings
from data_ingestion import normalize_vehicle_name, to_excel_bytes
from page_ui_helpers import dataframe_export_fingerprint, prepare_table_view, render_deferred_download_button, render_standard_data_editor
from storage_service import canonicalize_record_key, find_feedback_rows_missing_required_remarks


ALLOWED_COST_ANOMALY_FACTORIES = {"X990", "A990"}


def _skills_export_fingerprint(skills: list[dict]) -> str:
    preview = [
        f"{item.get('备件简称', '')}:{item.get('当前σ参数', '')}:{item.get('偏置权重', '')}"
        for item in list(skills or [])[:50]
    ]
    return f"count={len(skills or [])}|{'|'.join(map(str, preview))}"


def _safe_export_label(value: object, default: str = "测算结果") -> str:
    text = str(value or "").strip() or default
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:48] or default


def build_cost_anomaly_run_request(
    *,
    selected_short_names: list[str] | tuple[str, ...] | set[str],
    calculate_selected_clicked: bool,
    calculate_all_clicked: bool,
) -> dict:
    if calculate_all_clicked:
        return {
            "should_run": True,
            "include_all_short_names": True,
            "selected_short_names": [],
            "scope_label": "全量简称",
            "message": "",
        }

    selected = []
    seen = set()
    for value in selected_short_names or []:
        text = str(value or "").strip()
        if text and text not in seen:
            selected.append(text)
            seen.add(text)

    if calculate_selected_clicked and selected:
        return {
            "should_run": True,
            "include_all_short_names": False,
            "selected_short_names": selected,
            "scope_label": f"所选{len(selected)}个简称",
            "message": "",
        }

    message = "请选择至少一个备件简称后再点击计算。" if calculate_selected_clicked else ""
    return {
        "should_run": False,
        "include_all_short_names": False,
        "selected_short_names": selected,
        "scope_label": "",
        "message": message,
    }


def export_cost_anomaly_result_excel(
    result_df: pd.DataFrame,
    export_path: str,
    *,
    scope_label: str,
    generated_at: datetime | pd.Timestamp | None = None,
) -> str:
    target_dir = Path(str(export_path or "").strip()).expanduser()
    if not str(export_path or "").strip():
        raise ValueError("请先在系统设置中配置成本分析模型导出路径。")
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp(generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_label = _safe_export_label(scope_label)
    output_path = target_dir / f"成本异常监控_{safe_label}_{timestamp}.xlsx"
    export_df = result_df.copy() if result_df is not None else pd.DataFrame()
    export_df = export_df.drop(columns=[column for column in export_df.columns if str(column).startswith("_ai_")], errors="ignore")
    output_path.write_bytes(to_excel_bytes(export_df, sheet_name="成本异常监控"))
    return str(output_path)


def filter_cost_anomaly_scope(
    df: pd.DataFrame,
    *,
    selected_short_names: list[str] | tuple[str, ...] | set[str],
    include_all_short_names: bool,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "工厂" not in df.columns or "备件简称" not in df.columns:
        return df.iloc[0:0].copy()

    scoped_df = df.copy()
    factory_values = scoped_df["工厂"].fillna("").astype(str).str.strip().str.upper()
    scoped_df = scoped_df[factory_values.isin(ALLOWED_COST_ANOMALY_FACTORIES)].copy()
    if scoped_df.empty:
        return scoped_df.reset_index(drop=True)

    if include_all_short_names:
        return scoped_df.reset_index(drop=True)

    selected = {str(value).strip() for value in selected_short_names if str(value).strip()}
    if not selected:
        return scoped_df.iloc[0:0].copy()
    return scoped_df[scoped_df["备件简称"].astype(str).isin(selected)].copy().reset_index(drop=True)


def reset_cost_anomaly_short_name_filters() -> None:
    st.session_state["cost_anomaly_select_all_short_names"] = False
    st.session_state["cost_anomaly_short_names"] = []
    st.session_state.pop("cost_anomaly_active_run_request", None)


def filter_label_details_to_management_rows(
    label_details: dict[str, dict[str, str]],
    management_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    if not label_details or management_df is None or management_df.empty or "record_key" not in management_df.columns:
        return {}
    active_keys = {
        canonicalize_record_key(str(record_key))
        for record_key in management_df["record_key"].astype(str).tolist()
        if canonicalize_record_key(str(record_key))
    }
    return {
        record_key: payload
        for record_key, payload in label_details.items()
        if canonicalize_record_key(str(record_key)) in active_keys
    }


def build_cost_anomaly_chart_scope(
    working_df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    *,
    selected_short_names: list[str] | tuple[str, ...] | set[str],
    include_all_short_names: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    chart_df = working_df.copy()
    monitoring_chart_df = monitoring_df.copy()
    if include_all_short_names:
        return chart_df, monitoring_chart_df, "全部备件简称 - 成本分布"

    selected = [str(value).strip() for value in selected_short_names if str(value).strip()]
    if not selected:
        return chart_df.iloc[0:0].copy(), monitoring_chart_df.iloc[0:0].copy(), "未选择备件简称 - 成本分布"

    selected_set = set(selected)
    chart_df = chart_df[chart_df["备件简称"].astype(str).isin(selected_set)].copy()
    monitoring_chart_df = monitoring_chart_df[monitoring_chart_df["备件简称"].astype(str).isin(selected_set)].copy()
    if len(selected) <= 3:
        title_prefix = "、".join(selected)
    else:
        title_prefix = f"{len(selected)} 个备件简称"
    return chart_df, monitoring_chart_df, f"{title_prefix} - 成本分布"


def _load_vehicle_rank_order_map() -> dict[str, int]:
    try:
        rank_df = harness.execute_action("load_vehicle_rank_config")
    except Exception:
        return {}
    if rank_df is None or rank_df.empty or "vehicle_series" not in rank_df.columns:
        return {}
    rank_df = rank_df.sort_values("rank_order") if "rank_order" in rank_df.columns else rank_df.copy()
    return {
        normalize_vehicle_name(str(vehicle)): int(index)
        for index, vehicle in enumerate(rank_df["vehicle_series"].astype(str).tolist(), start=1)
        if normalize_vehicle_name(str(vehicle))
    }


def _normalize_vehicle_rank_order_map(vehicle_rank_order_map: dict[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for vehicle, rank_value in (vehicle_rank_order_map or {}).items():
        vehicle_key = normalize_vehicle_name(str(vehicle))
        rank_numeric = pd.to_numeric(pd.Series([rank_value]), errors="coerce").iloc[0]
        if vehicle_key and pd.notna(rank_numeric):
            normalized[vehicle_key] = int(rank_numeric)
    return normalized


def build_interval_lower_bound_rank_analysis(
    interval_df: pd.DataFrame,
    vehicle_rank_order_map: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_columns = ["备件简称", "覆盖车系数", "排序一致性", "最大排名偏差", "异常车系数", "重点异常车系"]
    detail_columns = [
        "备件简称",
        "适用车系",
        "车系梯度排名",
        "期望低到高排名",
        "区间下限排名",
        "排名偏差",
        "偏差方向",
        "合理下限",
        "基准价",
        "合理上限",
        "圈层角色",
    ]
    if interval_df is None or interval_df.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=detail_columns), pd.DataFrame()

    needed_columns = {"适用车系", "备件简称", "合理下限"}
    if not needed_columns.issubset(set(interval_df.columns)):
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=detail_columns), pd.DataFrame()

    data = interval_df.copy()
    for column_name in ["预测值", "合理上限", "圈层角色"]:
        if column_name not in data.columns:
            data[column_name] = None
    data["适用车系"] = data["适用车系"].fillna("").astype(str).str.strip()
    data["备件简称"] = data["备件简称"].fillna("").astype(str).str.strip()
    data["合理下限"] = pd.to_numeric(data["合理下限"], errors="coerce")
    data["预测值"] = pd.to_numeric(data["预测值"], errors="coerce")
    data["合理上限"] = pd.to_numeric(data["合理上限"], errors="coerce")
    data = data[data["适用车系"].ne("") & data["备件简称"].ne("")].dropna(subset=["合理下限"]).copy()
    if data.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=detail_columns), pd.DataFrame()

    if data["圈层角色"].astype(str).eq("主邻居圈").any():
        data = data[data["圈层角色"].astype(str).eq("主邻居圈")].copy()

    data["圈层角色"] = data["圈层角色"].fillna("主邻居圈").astype(str)
    grouped = (
        data.groupby(["备件简称", "适用车系"], as_index=False)
        .agg({"合理下限": "median", "预测值": "median", "合理上限": "median", "圈层角色": "first"})
        .rename(columns={"预测值": "基准价"})
    )
    normalized_rank_map = _normalize_vehicle_rank_order_map(vehicle_rank_order_map)
    grouped["_vehicle_key"] = grouped["适用车系"].map(lambda value: normalize_vehicle_name(str(value)))
    grouped["车系梯度排名"] = grouped["_vehicle_key"].map(normalized_rank_map)
    max_gradient_rank = max(normalized_rank_map.values()) if normalized_rank_map else 0
    grouped["期望低到高排名"] = grouped["车系梯度排名"].map(
        lambda value: int(max_gradient_rank - int(value) + 1) if pd.notna(value) and max_gradient_rank else pd.NA
    )
    grouped["区间下限排名"] = grouped.groupby("备件简称")["合理下限"].rank(method="dense", ascending=True).astype("Int64")
    expected_numeric = pd.to_numeric(grouped["期望低到高排名"], errors="coerce")
    lower_rank_numeric = pd.to_numeric(grouped["区间下限排名"], errors="coerce")
    grouped["排名偏差"] = (lower_rank_numeric - expected_numeric).astype("Int64")

    def _direction(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric) or float(numeric) == 0:
            return "一致"
        return "整体偏高" if float(numeric) > 0 else "整体偏低"

    grouped["偏差方向"] = grouped["排名偏差"].map(_direction)
    grouped["合理下限"] = grouped["合理下限"].clip(lower=0).round(2)
    grouped["基准价"] = grouped["基准价"].round(2)
    grouped["合理上限"] = grouped["合理上限"].round(2)

    detail_df = grouped[detail_columns].copy()
    detail_df["_abs_rank_delta"] = pd.to_numeric(detail_df["排名偏差"], errors="coerce").abs().fillna(0)
    summary_rows: list[dict] = []
    for part_name, part_df in detail_df.groupby("备件简称", sort=False):
        valid_corr_df = part_df.dropna(subset=["期望低到高排名", "区间下限排名"])
        if len(valid_corr_df) >= 2:
            consistency = valid_corr_df["期望低到高排名"].corr(valid_corr_df["区间下限排名"], method="spearman")
            consistency = float(consistency) if pd.notna(consistency) else 0.0
        else:
            consistency = np.nan
        abnormal_df = part_df[part_df["_abs_rank_delta"].gt(0)].copy()
        abnormal_df = abnormal_df.sort_values(["_abs_rank_delta", "适用车系"], ascending=[False, True], kind="mergesort")
        summary_rows.append(
            {
                "备件简称": part_name,
                "覆盖车系数": int(part_df["适用车系"].nunique()),
                "排序一致性": round(consistency, 4) if pd.notna(consistency) else pd.NA,
                "最大排名偏差": int(part_df["_abs_rank_delta"].max()) if not part_df.empty else 0,
                "异常车系数": int(abnormal_df["适用车系"].nunique()),
                "重点异常车系": "、".join(abnormal_df["适用车系"].head(3).astype(str).tolist()),
            }
        )

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["排序一致性", "最大排名偏差", "异常车系数", "备件简称"],
            ascending=[True, False, False, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)

    heatmap_df = detail_df.pivot_table(
        index="备件简称",
        columns="适用车系",
        values="排名偏差",
        aggfunc="first",
    ).reset_index()
    detail_df = detail_df.drop(columns=["_abs_rank_delta"]).sort_values(
        ["备件简称", "区间下限排名", "适用车系"],
        kind="mergesort",
    ).reset_index(drop=True)
    return summary_df, detail_df, heatmap_df


def _append_vehicle_rank_sort_columns(df: pd.DataFrame, rank_order_map: dict[str, int]) -> pd.DataFrame:
    sorted_df = df.copy()
    if "适用车系" not in sorted_df.columns or not rank_order_map:
        sorted_df["_vehicle_rank_order"] = 9999
        return sorted_df
    sorted_df["_vehicle_rank_order"] = sorted_df["适用车系"].astype(str).map(lambda value: rank_order_map.get(normalize_vehicle_name(value), 9999)).astype(int)
    return sorted_df


def build_interval_compare_display_labels(chart_data: pd.DataFrame, *, compare_mode: str) -> pd.DataFrame:
    display_df = chart_data.copy()
    if compare_mode == "vehicle":
        display_df["显示标签"] = display_df["备件简称"].astype(str)
    elif compare_mode == "part":
        display_df["显示标签"] = display_df["适用车系"].astype(str)
    else:
        raise ValueError(f"Unsupported interval compare mode: {compare_mode}")
    return display_df


def sort_interval_compare_chart_data(chart_data: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    if chart_data is None or chart_data.empty:
        return pd.DataFrame() if chart_data is None else chart_data.copy()

    sorted_df = chart_data.copy()
    sorted_df["合理下限"] = pd.to_numeric(sorted_df["合理下限"], errors="coerce")
    if mode == "vehicle":
        label_col = "显示标签" if "显示标签" in sorted_df.columns else "备件简称"
        lower_order = sorted_df.groupby(label_col)["合理下限"].transform("min")
        sorted_df["_interval_lower_order"] = lower_order
        sorted_df = sorted_df.sort_values(
            ["_interval_lower_order", label_col, "圈层编号"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        axis_order = {label: index for index, label in enumerate(dict.fromkeys(sorted_df[label_col].astype(str).tolist()))}
        sorted_df["_interval_axis_order"] = sorted_df[label_col].astype(str).map(axis_order).astype(int)
        return sorted_df.drop(columns=["_interval_lower_order"], errors="ignore").reset_index(drop=True)

    if mode == "part":
        sort_cols = [column for column in ["_vehicle_rank_order", "适用车系", "圈层编号"] if column in sorted_df.columns]
        return sorted_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True) if sort_cols else sorted_df.reset_index(drop=True)

    return sorted_df.reset_index(drop=True)


def filter_latest_material_cost_records_for_monitoring(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df is None or result_df.empty or "物料编码" not in result_df.columns or "价格有效于" not in result_df.columns:
        return result_df.copy() if result_df is not None else pd.DataFrame()

    data = result_df.copy()
    material_codes = data["物料编码"].fillna("").astype(str).str.strip()
    date_values = pd.to_datetime(data["价格有效于"], errors="coerce")
    latest_dates = date_values.groupby(material_codes, dropna=False).transform("max")
    latest_mask = date_values.eq(latest_dates) | latest_dates.isna()
    return data.loc[latest_mask].copy()


def render_cost_anomaly_page() -> None:
    inject_css(is_overview=False)
    st.title("📌 成本异常监控")
    if st.session_state.data is None:
        st.warning(NO_DATA_WARNING)
        return

    df = st.session_state.data
    price_col = require_price_col(df)

    mode_col, stat_col = st.columns([3, 2])
    with mode_col:
        st.radio(
            "测算模式",
            options=["原始测算", "优化后测算（专家纠偏）"],
            key="anomaly_mode",
            horizontal=True,
        )

    label_details = harness.execute_action("get_feedback_details")
    core_source_error = ""
    mgmt_df = pd.DataFrame()
    if label_details:
        core_source_df, _, core_source_error = harness.execute_action("load_core_cost_records")
        anomaly_context_frames = [df]
        for result_mode in ["raw", "weighted"]:
            try:
                persisted_context_df = harness.execute_action("load_cost_anomaly_results", result_mode=result_mode)
            except Exception:
                persisted_context_df = pd.DataFrame()
            if persisted_context_df is not None and not persisted_context_df.empty:
                anomaly_context_frames.append(persisted_context_df)
        fallback_context_df = pd.concat(anomaly_context_frames, ignore_index=True, sort=False)
        mgmt_df = build_calibration_management_df(
            label_details,
            core_source_df=core_source_df,
            fallback_source_df=fallback_context_df,
            fallback_price_col=price_col,
        )
        active_label_details = filter_label_details_to_management_rows(label_details, mgmt_df)
    else:
        active_label_details = {}

    label_statuses = {record_key: payload.get("label", "") for record_key, payload in active_label_details.items()}
    label_remarks = {record_key: payload.get("remark", "") for record_key, payload in active_label_details.items()}
    with stat_col:
        label_count = len(active_label_details)
        st.metric("已由专家校准的记录", f"{label_count} 条")
        stale_label_count = max(len(label_details) - label_count, 0)
        if stale_label_count:
            st.caption(f"已隐藏 {stale_label_count} 条无法匹配当前源数据的旧标注")

    render_knowledge_sync_status()

    if label_count > 0:
        with st.expander("📋 查看/管理已校准记录", expanded=False):
            if core_source_error and (core_source_df is None or core_source_df.empty):
                st.caption(f"主库关联提示：{core_source_error}")
            mgmt_display_columns = [
                "物料编码",
                "物料名称",
                "备件简称",
                "工厂",
                "价格有效期于",
                "价格",
                "当前标注",
                "标注备注",
                "撤回标注",
            ]
            mgmt_editor_source = mgmt_df.set_index("record_key", drop=True)
            mgmt_filtered_df, mgmt_visible_columns = prepare_table_view(
                mgmt_editor_source,
                "calibration_mgmt",
                display_columns=mgmt_display_columns,
                default_search_columns=["物料编码", "物料名称", "备件简称", "工厂", "标注备注"],
                locked_columns=mgmt_display_columns,
                filter_title="已校准记录",
            )
            mgmt_edited = render_standard_data_editor(
                mgmt_filtered_df[mgmt_visible_columns],
                "calibration_mgmt",
                editable_columns=["标注备注", "撤回标注"],
                column_config={
                    "物料编码": st.column_config.TextColumn("物料编码", disabled=True),
                    "物料名称": st.column_config.TextColumn("物料名称", disabled=True, width="large"),
                    "备件简称": st.column_config.TextColumn("备件简称", disabled=True),
                    "工厂": st.column_config.TextColumn("工厂", disabled=True),
                    "价格有效期于": st.column_config.TextColumn("价格有效期于", disabled=True),
                    "价格": st.column_config.NumberColumn("价格", disabled=True, format="%.4f"),
                    "当前标注": st.column_config.TextColumn("当前标注", disabled=True),
                    "撤回标注": st.column_config.CheckboxColumn("撤回标注", help="勾选后点击下方按钮撤回此标注", default=False),
                    "标注备注": st.column_config.TextColumn(
                        "标注备注",
                        help="可直接编辑备注，例如材质、供应商或批次原因说明。",
                        width="large",
                    ),
                },
                max_height=320,
            )

            mgmt_c1, mgmt_c2, mgmt_c3 = st.columns(3)
            with mgmt_c1:
                if st.button("💾 保存修改", type="primary"):
                    final_mgmt_df = mgmt_df.copy()
                    edited_remark_map = {
                        str(record_key): str(remark or "")
                        for record_key, remark in zip(mgmt_edited.index, mgmt_edited["标注备注"])
                    }
                    final_mgmt_df["标注备注"] = final_mgmt_df["record_key"].astype(str).map(edited_remark_map).fillna(final_mgmt_df["标注备注"])
                    final_labels_df = final_mgmt_df[["record_key", "当前标注", "标注备注"]].rename(
                        columns={"当前标注": "label", "标注备注": "remark"}
                    )
                    missing_remark_keys = find_feedback_rows_missing_required_remarks(final_labels_df)
                    if missing_remark_keys:
                        st.error(f"标注为正常的记录必须填写批注原因，共 {len(missing_remark_keys)} 条。")
                    else:
                        try:
                            harness.execute_action("replace_feedback", final_labels_df=final_labels_df)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            bump_cost_refresh_token()
                            sync_ai_knowledge_base(force_full=True)
                            st.success(f"✅ 已保存 {len(final_labels_df)} 条标注备注修改")
                            time.sleep(0.5)
                            st.rerun()
            with mgmt_c2:
                if st.button("🗑️ 撤回选中的标注"):
                    keys_to_revoke = [str(record_key) for record_key, row in mgmt_edited.iterrows() if bool(row["撤回标注"])]
                    if keys_to_revoke:
                        revoked = harness.execute_action("delete_feedback", keys_to_remove=keys_to_revoke)
                        bump_cost_refresh_token()
                        st.success(f"✅ 已撤回 {revoked} 条标注")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning("未选中任何记录")
            with mgmt_c3:
                if st.button("⚠️ 清空所有标注"):
                    st.session_state["_confirm_clear_labels"] = True
                if st.session_state.get("_confirm_clear_labels"):
                    st.warning("确定要清空所有专家标注吗？此操作不可撤销。")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("✅ 确认清空", type="primary"):
                            clear_cost_feedback_and_ai_state()
                            bump_cost_refresh_token()
                            st.session_state["_confirm_clear_labels"] = False
                            st.success("✅ 已清空所有标注与 AI 辅助分析缓存")
                            time.sleep(0.5)
                            st.rerun()
                    with cc2:
                        if st.button("取消"):
                            st.session_state["_confirm_clear_labels"] = False
                            st.rerun()

    st.markdown("---")

    factory_scoped_df = filter_cost_anomaly_scope(
        df,
        selected_short_names=[],
        include_all_short_names=True,
    )
    if factory_scoped_df.empty:
        st.warning("成本异常监控仅核查工厂类型 X990 / A990；当前数据中没有可参与测算的记录。")
        st.stop()

    short_name_options = sorted(factory_scoped_df["备件简称"].astype(str).unique().tolist())
    st.markdown("#### 🔍 测算范围")
    scope_col, selected_run_col, full_run_col, reset_scope_col = st.columns([4, 1.5, 1.7, 1], vertical_alignment="bottom")
    with scope_col:
        selected_short_names = st.multiselect(
            "备件简称筛选（仅选择不触发测算）",
            options=short_name_options,
            key="cost_anomaly_short_names",
            help="选择后点击“计算所选简称”；进入页面默认不会进行任何异常测算。",
        )
    with selected_run_col:
        calculate_selected_clicked = st.button(
            "计算所选简称",
            key="cost_anomaly_calculate_selected",
            width="stretch",
            disabled=not selected_short_names,
        )
    with full_run_col:
        calculate_all_clicked = st.button(
            "一键计算全量简称",
            key="cost_anomaly_calculate_all",
            width="stretch",
        )
    with reset_scope_col:
        st.button(
            "重置",
            key="reset_cost_anomaly_short_names",
            width="stretch",
            on_click=reset_cost_anomaly_short_name_filters,
        )

    current_run_request = build_cost_anomaly_run_request(
        selected_short_names=selected_short_names,
        calculate_selected_clicked=calculate_selected_clicked,
        calculate_all_clicked=calculate_all_clicked,
    )
    should_export_run_result = bool(current_run_request["should_run"])
    if current_run_request["should_run"]:
        st.session_state["cost_anomaly_active_run_request"] = current_run_request
    elif current_run_request.get("message"):
        st.warning(current_run_request["message"])

    active_run_request = st.session_state.get("cost_anomaly_active_run_request") or current_run_request
    if not active_run_request.get("should_run"):
        st.info("初始状态不会进行成本异常测算。请先选择备件简称后点击“计算所选简称”，或点击“一键计算全量简称”。")
        st.stop()

    include_all_short_names = bool(active_run_request.get("include_all_short_names"))
    selected_short_names = list(active_run_request.get("selected_short_names") or [])
    scoped_df = filter_cost_anomaly_scope(
        df,
        selected_short_names=selected_short_names,
        include_all_short_names=include_all_short_names,
    )
    if scoped_df.empty:
        st.info("当前测算范围为空，请调整备件简称后重新点击计算。")
        st.stop()

    selected_scope_text = str(active_run_request.get("scope_label") or ("全量简称" if include_all_short_names else f"所选{len(selected_short_names)}个简称"))
    st.caption(f"当前仅纳入工厂 X990 / A990，共 {len(scoped_df)} 条记录，范围：{selected_scope_text}。")

    unique_short_name = scoped_df["备件简称"].astype(str).nunique()
    if len(scoped_df) >= 30000 or unique_short_name >= 300:
        st.info("正在进行大规模深度测算，请稍候...")

    try:
        with st.spinner("原始测算进行中，请稍候..."):
            anomaly_df = cached_anomaly_report(
                scoped_df,
                price_col,
                get_cost_refresh_token(),
            )
    except ImportError as exc:
        st.error(str(exc))
        st.info("安装完成后重启应用，再进入本页面即可。")
        st.code("pip install scikit-learn")
        st.stop()
    except Exception as exc:
        st.error(f"异常检测失败: {exc}")
        st.stop()

    if anomaly_df.empty:
        st.info("当前数据暂无可检测记录。")
        st.stop()

    is_expert_mode = st.session_state.anomaly_mode == "优化后测算（专家纠偏）"

    skills_data = None
    skills_source = ""
    skills_json = ""
    skills_loaded = False
    if is_expert_mode:
        skills_data = skills_engine.load_latest_cost_skills_excel(settings.qualitative_skills_path)
        if skills_data:
            skills_source = f"专家经验报告导出路径最新 Excel：{skills_data.get('source_path', '')}"
        else:
            skills_data = harness.execute_action("load_skills_snapshot", domain="cost")
            if skills_data:
                skills_source = f"本地数据库 Skills 快照：{skills_data.get('saved_at', '未知时间')}"
        skills_json = skills_engine.build_cost_skill_overrides_json(skills_data)
        skills_loaded = bool(skills_json)

    if is_expert_mode:
        expert_labels = dict(label_statuses)
        if expert_labels:
            with st.spinner("专家纠偏测算进行中，请稍候..."):
                working_df = cached_anomaly_report_weighted(
                    scoped_df,
                    price_col,
                    tuple(sorted(expert_labels.items())),
                    get_cost_refresh_token(),
                    skills_overrides_json=skills_json,
                )
        else:
            working_df = anomaly_df.copy()
            st.info("💡 暂无专家标注数据，显示原始测算结果。请先在下方表格中勾选并保存标注。")
    else:
        working_df = anomaly_df.copy()

    knowledge_refresh_token = harness.execute_action("get_expert_knowledge_refresh_token")
    working_df = cached_enrich_anomaly_with_ai(working_df, knowledge_refresh_token)

    if "_record_key" in working_df.columns:
        working_df["专家校准"] = working_df["_record_key"].astype(str).map(lambda key: "✅" if label_statuses.get(key) == "正常" else "")
        working_df["专家备注"] = working_df["_record_key"].astype(str).map(lambda key: label_remarks.get(key, ""))
        working_df["专家备注"] = working_df["专家备注"].fillna("")

    if is_expert_mode:
        if skills_loaded:
            skill_count = len(skills_data.get("skills", []))
            st.info(
                f"📘 正在应用最新 Skills 技能书（{skill_count} 个备件简称，来源：{skills_source}），匹配的备件将使用文件中的成本区间与参数进行专家纠偏。"
            )
        elif skills_data is None and harness.execute_action("has_skills_snapshot", domain="cost"):
            st.warning("⚠️ Skills 技能书快照读取异常，已回退到默认算法。请重新运行 AutoResearch 生成。")

    if should_export_run_result:
        try:
            exported_path = export_cost_anomaly_result_excel(
                working_df.drop(columns=["_record_key"], errors="ignore"),
                settings.quantitative_skills_path,
                scope_label=selected_scope_text,
                generated_at=datetime.now(),
            )
            st.success(f"📁 本次成本异常测算结果已保存至成本分析模型导出路径：`{exported_path}`")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.warning(f"本次测算已完成，但写入成本分析模型导出路径失败：{exc}")

    monitoring_df = filter_latest_material_cost_records_for_monitoring(working_df)
    historical_rows = max(int(len(working_df) - len(monitoring_df)), 0)
    if historical_rows > 0:
        st.caption(f"异常监控列表仅核查每个物料编码的最新日期成本；{historical_rows} 条旧日期记录只参与合理区间测算。")

    high_count = int(monitoring_df["status"].astype(str).str.contains("异常偏高").sum())
    low_count = int(monitoring_df["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum())
    abnormal_total = high_count + low_count

    m1, m2, m3 = st.columns(3)
    m1.metric("异常总数", f"{abnormal_total}")
    m2.metric("异常偏高", f"{high_count}")
    m3.metric("异常偏低", f"{low_count}")

    if is_expert_mode:
        orig_monitoring_df = filter_latest_material_cost_records_for_monitoring(anomaly_df)
        orig_high = int(orig_monitoring_df["status"].astype(str).str.contains("异常偏高").sum())
        orig_low = int(orig_monitoring_df["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum())
        orig_total = orig_high + orig_low
        if orig_total > 0:
            reduced = orig_total - abnormal_total
            st.caption(
                f"💡 加权自学习算法重新测算后，异常记录从 {orig_total} 条降至 {abnormal_total} 条（减少 {reduced} 条，降幅 {reduced / orig_total:.1%}）"
            )

    filtered_anomaly_df = monitoring_df.copy()

    abnormal_view = filtered_anomaly_df[
        filtered_anomaly_df["status"].astype(str).str.contains("异常偏高|异常偏低|严重异常偏低")
    ].copy()
    st.markdown(f"**共找到 {len(abnormal_view)} 条异常记录**")

    preview_abnormal_view = abnormal_view
    if len(abnormal_view) > 1000:
        preview_abnormal_view = abnormal_view.head(100).copy()
        st.warning("异常结果超过 1000 条。为避免前端渲染卡顿，当前仅预览前 100 条，其余记录请通过“导出异常成本报表”查看。")

    if not preview_abnormal_view.empty and "_record_key" in preview_abnormal_view.columns:
        edit_df = preview_abnormal_view.copy()
        existing_labels = dict(label_statuses)
        existing_remarks = dict(label_remarks)
        edit_df["标注为正常"] = edit_df["_record_key"].apply(lambda key: existing_labels.get(key) == "正常")
        edit_df["专家校准"] = edit_df["_record_key"].apply(lambda key: "✅" if existing_labels.get(key) == "正常" else "")
        edit_df["标注备注"] = edit_df["_record_key"].apply(lambda key: existing_remarks.get(key, ""))
        edit_df["采纳AI建议"] = False

        display_cols = [column_name for column_name in edit_df.columns if column_name != "_record_key"]
        if "专家备注" in display_cols:
            display_cols.remove("专家备注")
        priority_cols = [
            column_name
            for column_name in ["标注为正常", "标注备注", "采纳AI建议", "AI 辅助分析", "专家校准"]
            if column_name in display_cols
        ]
        display_cols = priority_cols + [column_name for column_name in display_cols if column_name not in priority_cols]

        column_config = {
            "标注为正常": st.column_config.CheckboxColumn("标注为正常", help="勾选此项将该记录标注为「正常」（专家纠偏）", default=False),
            "标注备注": st.column_config.TextColumn("标注备注", help="填写专家备注；若勾选“采纳AI建议”，保存时会自动写入 AI 辅助分析文本。"),
            "采纳AI建议": st.column_config.CheckboxColumn("采纳AI建议", help="勾选后，保存时自动将 AI 辅助分析填入标注备注，并视为已校准。", default=False),
            "物料编码": st.column_config.TextColumn("物料编码", disabled=True),
            "物料名称": st.column_config.TextColumn("物料名称", disabled=True),
            "适用车系": st.column_config.TextColumn("适用车系", disabled=True),
            "工厂": st.column_config.TextColumn("工厂", disabled=True),
            "备件简称": st.column_config.TextColumn("备件简称", disabled=True),
            "实际成本": st.column_config.NumberColumn("实际成本", disabled=True, format="%.2f"),
            "价格有效于": st.column_config.DateColumn("价格有效于", disabled=True),
            "样本量": st.column_config.NumberColumn("样本量", disabled=True),
            "预测值": st.column_config.NumberColumn("预测值", disabled=True, format="%.2f"),
            "合理下限": st.column_config.NumberColumn("合理下限", disabled=True, format="%.2f"),
            "合理上限": st.column_config.NumberColumn("合理上限", disabled=True, format="%.2f"),
            "偏离数值": st.column_config.NumberColumn("偏离数值", disabled=True, format="%.2f"),
            "偏离比例": st.column_config.NumberColumn("偏离比例", disabled=True, format="%.2%%"),
            "status": st.column_config.TextColumn("status", disabled=True),
            "AI 辅助分析": st.column_config.TextColumn("AI 辅助分析", disabled=True),
        }
        if "专家校准" in display_cols:
            column_config["专家校准"] = st.column_config.TextColumn("专家校准", disabled=True)
        if "判定依据" in display_cols:
            column_config["判定依据"] = st.column_config.TextColumn("判定依据", disabled=True)

        visible_edit_df, anomaly_visible_columns = prepare_table_view(
            edit_df,
            "anomaly_editor",
            display_columns=display_cols,
            default_search_columns=["物料编码", "物料名称", "备件简称", "适用车系", "status"],
            locked_columns=["标注为正常", "标注备注", "采纳AI建议", "专家校准"],
            filter_title="异常记录",
        )
        visible_edit_df = visible_edit_df.reset_index(drop=True)
        edited = render_standard_data_editor(
            visible_edit_df[anomaly_visible_columns],
            "anomaly_editor",
            editable_columns=["标注为正常", "标注备注", "采纳AI建议"],
            column_config=column_config,
            max_height=460,
        )

        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            if st.button("💾 保存专家标注", type="primary", width="stretch"):
                final_labels = {
                    key: {"label": payload.get("label", ""), "remark": payload.get("remark", "")}
                    for key, payload in label_details.items()
                }
                for index, (_, orig_row) in enumerate(visible_edit_df.iterrows()):
                    record_key = orig_row["_record_key"]
                    checked = bool(edited.iloc[index]["标注为正常"])
                    adopt_ai = bool(edited.iloc[index].get("采纳AI建议", False))
                    remark_text = str(edited.iloc[index].get("标注备注", "") or "").strip()
                    if adopt_ai:
                        ai_text = str(orig_row.get("AI 辅助分析", "") or "").strip()
                        if ai_text:
                            remark_text = ai_text
                            checked = True

                    should_keep = checked or bool(remark_text)
                    if should_keep:
                        final_labels[record_key] = {"label": "正常", "remark": remark_text}
                    elif record_key in final_labels and final_labels[record_key].get("label") == "正常":
                        del final_labels[record_key]

                final_labels_df = pd.DataFrame(
                    [{"record_key": key, "label": value.get("label", ""), "remark": value.get("remark", "")} for key, value in final_labels.items()]
                )
                missing_remark_keys = find_feedback_rows_missing_required_remarks(final_labels_df)
                if missing_remark_keys:
                    st.error(f"标注为正常的记录必须填写批注原因，共 {len(missing_remark_keys)} 条。")
                else:
                    try:
                        harness.execute_action("replace_feedback", final_labels_df=final_labels_df)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        bump_cost_refresh_token()
                        sync_ai_knowledge_base(force_full=True)
                        st.success(f"✅ 标注已保存！当前共 {len(final_labels)} 条专家校准记录。")
                        time.sleep(0.5)
                        st.rerun()
    elif preview_abnormal_view.empty:
        st.info("当前筛选条件下暂无异常记录。")
    else:
        preview_abnormal_view, readonly_visible_columns = prepare_table_view(
            preview_abnormal_view,
            "anomaly_readonly",
            display_columns=[column_name for column_name in preview_abnormal_view.columns if column_name != "_record_key"],
            default_search_columns=["物料编码", "物料名称", "备件简称", "适用车系", "status"],
            filter_title="异常记录（只读）",
        )
        render_standard_data_editor(
            preview_abnormal_view[readonly_visible_columns],
            "anomaly_readonly",
            column_config={
                "实际成本": st.column_config.NumberColumn("实际成本", format="%.2f"),
                "预测值": st.column_config.NumberColumn("预测值", format="%.2f"),
                "合理下限": st.column_config.NumberColumn("合理下限", format="%.2f"),
                "合理上限": st.column_config.NumberColumn("合理上限", format="%.2f"),
                "偏离数值": st.column_config.NumberColumn("偏离数值", format="%.2f"),
                "偏离比例": st.column_config.NumberColumn("偏离比例", format="%.2%%"),
            },
            max_height=460,
        )

    if "action_col2" not in locals():
        _, action_col2 = st.columns([1, 1])
    export_df = abnormal_view.drop(columns=["_record_key"], errors="ignore")
    export_df = export_df.drop(columns=[column_name for column_name in export_df.columns if str(column_name).startswith("_ai_")], errors="ignore")
    with action_col2:
        render_deferred_download_button(
            label="📥 下载异常成本报表",
            prepare_label="准备导出异常成本报表",
            data_builder=lambda export_frame=export_df.copy(): to_excel_bytes(export_frame),
            file_name=f"异常成本监控_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="cost_anomaly_export",
            fingerprint=dataframe_export_fingerprint(export_df),
            width="stretch",
        )

    chart_df, monitoring_chart_df, chart_title = build_cost_anomaly_chart_scope(
        working_df,
        monitoring_df,
        selected_short_names=selected_short_names,
        include_all_short_names=include_all_short_names,
    )
    if not chart_df.empty:
        chart_df = chart_df.copy()
        chart_df["偏离比例显示"] = chart_df["偏离比例"].apply(
            lambda value: f"{value:.2%}" if isinstance(value, (int, float)) and value == value else ""
        )
        chart_df["_recency_weight"] = calculate_recency_weight_series(chart_df["价格有效于"]).to_numpy(dtype=float)
        chart_df["_time_decay_note"] = np.where(
            chart_df["_recency_weight"] < 0.999,
            "权重受时序衰减影响",
            "近6个月样本权重最高",
        )
        nbins = min(80, max(10, int(len(chart_df) ** 0.5 * 4)))
        cost_values = chart_df["实际成本"].to_numpy(dtype=float)
        if float(np.min(cost_values)) == float(np.max(cost_values)):
            center_value = float(cost_values[0])
            span = max(abs(center_value) * 0.05, 1.0)
            bin_edges = np.array([center_value - span, center_value + span], dtype=float)
        else:
            bin_edges = np.unique(np.histogram_bin_edges(cost_values, bins=nbins).astype(float))
            if bin_edges.size < 2:
                center_value = float(np.mean(cost_values))
                span = max(abs(center_value) * 0.05, 1.0)
                bin_edges = np.array([center_value - span, center_value + span], dtype=float)

        bin_ids = np.searchsorted(bin_edges, cost_values, side="right") - 1
        bin_ids = np.clip(bin_ids, 0, bin_edges.size - 2)
        chart_df["_hist_bin"] = bin_ids

        histogram_rows = []
        for bin_idx, bin_group in chart_df.groupby("_hist_bin", sort=True):
            left_edge = float(bin_edges[int(bin_idx)])
            right_edge = float(bin_edges[int(bin_idx) + 1])
            histogram_rows.append(
                {
                    "bin_mid": (left_edge + right_edge) / 2.0,
                    "bin_width": max(right_edge - left_edge, np.finfo(float).eps) * 0.95,
                    "bin_left": left_edge,
                    "bin_right": right_edge,
                    "raw_count": int(len(bin_group)),
                    "weighted_count": float(bin_group["_recency_weight"].sum()),
                    "time_decay_note": "权重受时序衰减影响" if (bin_group["_recency_weight"] < 0.999).any() else "近6个月样本权重最高",
                }
            )

        histogram_df = pd.DataFrame(histogram_rows)
        anchor_group = chart_df.groupby("备件简称", as_index=False).size().sort_values("size", ascending=False).iloc[0]["备件简称"]
        anchor_df = chart_df[chart_df["备件简称"] == anchor_group]
        ring_payload = ""
        if "多圈合理区间" in anchor_df.columns and not anchor_df["多圈合理区间"].dropna().empty:
            ring_payload = str(anchor_df["多圈合理区间"].dropna().astype(str).iloc[0])
        accepted_intervals = parse_accepted_ring_intervals(ring_payload)
        role_colors = {
            "主邻居圈": "#3b82f6",
            "次邻居圈": "#f59e0b",
            "异常区间": "#ef4444",
        }
        if not histogram_df.empty:
            def _classify_bin(midpoint: float) -> str:
                for interval in accepted_intervals:
                    if float(interval["合理下限"]) <= midpoint <= float(interval["合理上限"]):
                        return str(interval["圈层角色"])
                return "异常区间"

            histogram_df["ring_class"] = histogram_df["bin_mid"].apply(_classify_bin)
        fig = go.Figure()
        if not histogram_df.empty:
            for ring_class in ["主邻居圈", "次邻居圈", "异常区间"]:
                class_df = histogram_df[histogram_df["ring_class"].eq(ring_class)]
                if class_df.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=class_df["bin_mid"],
                        y=class_df["raw_count"],
                        width=class_df["bin_width"],
                        name=ring_class,
                        opacity=0.78,
                        marker=dict(color=role_colors[ring_class], line=dict(color=role_colors[ring_class], width=1)),
                        customdata=class_df[["bin_left", "bin_right", "weighted_count", "time_decay_note", "ring_class"]].values,
                        hovertemplate=(
                            "区间类型: %{customdata[4]}"
                            "<br>价格区间: %{customdata[0]:,.2f} - %{customdata[1]:,.2f}"
                            "<br>样本频数: %{y}"
                            "<br>时序加权样本: %{customdata[2]:.2f}"
                            "<br>%{customdata[3]}"
                            "<extra></extra>"
                        ),
                    )
                )

        if accepted_intervals:
            for interval in accepted_intervals:
                role = str(interval["圈层角色"])
                color = role_colors.get(role, "#4a90e2")
                lower = float(interval["合理下限"])
                upper = float(interval["合理上限"])
                baseline = float(interval["预测值"])
                fig.add_vrect(x0=lower, x1=upper, fillcolor=color, opacity=0.08, line_width=0)
                fig.add_vline(x=baseline, line_dash="dash", line_color=color, annotation_text=f"{role}基准", annotation_position="top")
                fig.add_vline(x=lower, line_dash="dot", line_color=color)
                fig.add_vline(x=upper, line_dash="dot", line_color=color)
        else:
            baseline = float(anchor_df["预测值"].median())
            upper = float(anchor_df["合理上限"].median())
            lower = float(anchor_df["合理下限"].median())
            fig.add_vline(x=baseline, line_dash="dash", line_color="#1f77b4", annotation_text="基准合理价", annotation_position="top")
            fig.add_vline(x=upper, line_dash="dash", line_color="#d62728", annotation_text="合理上限", annotation_position="top")
            fig.add_vline(x=lower, line_dash="dash", line_color="#d62728", annotation_text="合理下限", annotation_position="top")

        monitoring_chart_df = monitoring_chart_df.copy()
        monitoring_chart_df["偏离比例显示"] = monitoring_chart_df["偏离比例"].apply(
            lambda value: f"{value:.2%}" if isinstance(value, (int, float)) and value == value else ""
        )
        monitoring_chart_df["_recency_weight"] = calculate_recency_weight_series(monitoring_chart_df["价格有效于"]).to_numpy(dtype=float)
        monitoring_chart_df["_time_decay_note"] = np.where(
            monitoring_chart_df["_recency_weight"] < 0.999,
            "权重受时序衰减影响",
            "近6个月样本权重最高",
        )
        abnormal_points = monitoring_chart_df[
            monitoring_chart_df["status"].astype(str).str.contains("异常偏高|异常偏低|严重异常偏低")
        ].copy()
        if not abnormal_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=abnormal_points["实际成本"],
                    y=[0] * len(abnormal_points),
                    mode="markers",
                    marker=dict(size=10, color="#e74c3c"),
                    name="异常点",
                    customdata=abnormal_points[["物料编码", "实际成本", "偏离比例显示", "_time_decay_note"]].values,
                    hovertemplate=(
                        "物料编码: %{customdata[0]}<br>实际价格: %{customdata[1]:,.2f}"
                        "<br>偏离比例: %{customdata[2]}"
                        "<br>%{customdata[3]}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(text=f"{chart_title}<br><sup>算法已自动识别并保护梯度定价区间，剔除孤立离群点。</sup>"),
            xaxis_title="成本",
            yaxis_title="频数",
            template="plotly_white",
            bargap=0.05,
        )
        st.plotly_chart(fig, width="stretch")


def render_cost_skills_page() -> None:
    inject_css(is_overview=False)
    st.title("🧠 成本区间 Skills")
    render_knowledge_sync_status()

    anomaly_df = _load_interval_compare_anomaly_df()
    if anomaly_df is None:
        return

    df = st.session_state.data
    price_col = require_price_col(df)

    expert_labels = harness.execute_action("get_feedback_statuses")
    file_rows = harness.execute_action("get_feedback_row_count")
    if file_rows != len(expert_labels):
        bump_cost_refresh_token()
        expert_labels = harness.execute_action("get_feedback_statuses")

    if expert_labels:
        labels_tuple = tuple(sorted(expert_labels.items()))
        skills_data = skills_engine.load_latest_cost_skills_excel(settings.qualitative_skills_path)
        if not skills_data:
            skills_data = harness.execute_action("load_skills_snapshot", domain="cost")
        skills_json = ""
        if skills_data:
            skills_json = skills_engine.build_cost_skill_overrides_json(skills_data)
        try:
            optimized_df = cached_anomaly_report_weighted(
                df,
                price_col,
                labels_tuple,
                get_cost_refresh_token(),
                skills_overrides_json=skills_json,
            )
        except Exception:
            optimized_df = anomaly_df
    else:
        optimized_df = anomaly_df

    knowledge_refresh_token = harness.execute_action("get_expert_knowledge_refresh_token")
    optimized_df = cached_enrich_anomaly_with_ai(optimized_df, knowledge_refresh_token)

    st.markdown("## 📋 Skills 技能书")
    st.markdown("从当前异常检测结果中提取每个备件简称的算法参数与分布特征，可作为系统**知识资产**下载。")

    skills_result = ComputeJob().precompute_cost_skills(optimized_df, expert_labels)
    skills = skills_result.all_skills
    skills_filtered = skills_result.export_skills

    try:
        exported_paths = skills_engine.export_cost_skills_excel_artifacts(
            skills,
            model_export_path=settings.quantitative_skills_path,
            generated_at=datetime.now(),
            force_new=False,
        )
        if exported_paths.get("model_export_path"):
            st.caption(f"📁 全量成本区间 Skills 已自动写入成本分析模型导出路径：{exported_paths['model_export_path']}")
        elif not str(settings.quantitative_skills_path or "").strip():
            st.caption("📁 成本分析模型导出路径未配置；当前仅在页面中展示并支持手动下载。")
    except Exception as exc:
        st.warning(f"⚠️ 全量成本区间 Skills 自动写入成本分析模型导出路径失败：{exc}")

    sk_all, sk_covered = skills_result.total_count, skills_result.covered_count
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.metric("备件简称总数", sk_all)
    with c_m2:
        st.metric("专家标注覆盖简称数", sk_covered)

    with st.expander("预览 Skills 技能书（全部备件）", expanded=False):
        st.markdown(skills_engine.skills_to_markdown(skills), unsafe_allow_html=True)

    export_skills = skills_filtered if expert_labels else skills
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            f"📥 下载 Skills (JSON) — {len(export_skills)} 个简称",
            data=skills_engine.skills_to_json_bytes(export_skills),
            file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width="stretch",
        )
    with dl2:
        st.download_button(
            "📥 下载 Skills (Markdown)",
            data=skills_engine.skills_to_markdown(export_skills).encode("utf-8"),
            file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch",
        )
    with dl3:
        render_deferred_download_button(
            label="📥 下载 Skills (Excel)",
            prepare_label="准备 Skills Excel",
            data_builder=lambda skills_payload=list(export_skills): skills_engine.skills_to_excel_bytes(skills_payload),
            file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="cost_skills_excel_export",
            fingerprint=_skills_export_fingerprint(export_skills),
            width="stretch",
        )

    st.markdown("---")
    st.markdown("## 🧾 专家经验看板")
    st.markdown("基于专家备注做增量知识蒸馏，沉淀为可复用的定价经验规则。")

    kb_df = harness.execute_action("load_expert_knowledge_base")
    kb_c1, kb_c2, kb_c3 = st.columns(3)
    with kb_c1:
        st.metric("知识规则数", f"{len(kb_df)}")
    with kb_c2:
        if kb_df.empty or "updated_at" not in kb_df.columns or kb_df["updated_at"].isna().all():
            st.metric("最近更新时间", "暂无")
        else:
            latest_ts = pd.to_datetime(kb_df["updated_at"], errors="coerce").max()
            st.metric("最近更新时间", latest_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_ts) else "暂无")
    with kb_c3:
        if st.button("🤖 刷新 AI 知识库"):
            sync_ai_knowledge_base(force_full=True, spinner_text="🤖 正在蒸馏专家备注并刷新 AI 知识库...")
            st.rerun()

    llm_settings_ready = bool(settings.llm_enabled)
    if kb_df.empty:
        if llm_settings_ready:
            st.info("当前还没有 AI 蒸馏出的专家经验规则。请先补充专家备注，然后保存备注或运行 AutoResearch。")
        else:
            st.info("LLM 尚未从本地 `.env` 检测到完整配置，当前无法生成 AI 经验规则。")
    else:
        kb_view = kb_df.rename(
            columns={
                "short_name": "备件简称",
                "material_code": "代表物料编码",
                "material_name": "代表物料名称",
                "supplier_code": "供应商代码",
                "supplier_name": "供应商名称",
                "vehicle_series": "适用车系",
                "rule_content": "车系供应商简称分析",
                "confidence_score": "可信度",
                "updated_at": "更新时间",
            }
        )
        for column_name in ["代表物料编码", "代表物料名称", "供应商名称"]:
            if column_name not in kb_view.columns:
                kb_view[column_name] = ""
        kb_display_columns = [
            "备件简称",
            "代表物料编码",
            "代表物料名称",
            "供应商代码",
            "供应商名称",
            "适用车系",
            "车系供应商简称分析",
            "可信度",
            "更新时间",
        ]
        kb_table_df, kb_visible_columns = prepare_table_view(
            kb_view[kb_display_columns],
            "knowledge_base_table",
            default_search_columns=["备件简称", "代表物料编码", "供应商代码", "供应商名称", "适用车系", "车系供应商简称分析"],
            filter_title="AI 经验库",
        )
        render_standard_data_editor(
            kb_table_df[kb_visible_columns],
            "knowledge_base_table",
            column_config={"车系供应商简称分析": st.column_config.TextColumn("车系供应商简称分析", disabled=True, width="large")},
            max_height=360,
        )

        with st.expander("预览 Markdown 格式报告", expanded=False):
            st.markdown(llm_engine.knowledge_base_to_markdown(kb_df))

        kb_dl1, kb_dl2, kb_dl3 = st.columns(3)
        with kb_dl1:
            st.download_button(
                "📥 下载 AI 经验库 (JSON)",
                data=llm_engine.knowledge_base_to_json_bytes(kb_df),
                file_name=f"expert_knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with kb_dl2:
            st.download_button(
                "📥 下载 AI 经验库 (Markdown)",
                data=llm_engine.knowledge_base_to_markdown(kb_df).encode("utf-8"),
                file_name=f"expert_knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        with kb_dl3:
            kb_export_df = kb_table_df[kb_visible_columns].copy()
            render_deferred_download_button(
                label="📥 下载 AI 经验库 (Excel)",
                prepare_label="准备 AI 经验库 Excel",
                data_builder=lambda export_df=kb_export_df: to_excel_bytes(export_df),
                file_name=f"expert_knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="knowledge_base_excel_export",
                fingerprint=dataframe_export_fingerprint(kb_export_df),
            )

    st.markdown("---")
    st.markdown("## 🔬 AutoResearch 棘轮迭代")
    st.markdown("系统自动微调 **σ 系数**、**偏置权重**、**时序衰减系数**、**断层倍数** 与 **基准分位点**，并通过随机参数搜索控制 10 轮寻优时长。")

    if not expert_labels:
        st.warning("⚠️ 暂无专家标注数据。请先在「成本异常监控」页面中标注并保存。")
    else:
        iter_options = [5, 10, 20]
        n_iters = st.select_slider("迭代次数", options=iter_options, value=10)
        if st.button("🚀 启动 AutoResearch", type="primary"):
            progress_bar = st.progress(0, text="初始化...")
            status_text = st.empty()

            def _on_progress(current, total, best_score, trial_score, trial_params, best_params):
                pct = current / total
                progress_bar.progress(pct, text=f"迭代 {current}/{total} — 最佳得分 {best_score:.2%}")
                status_text.caption(
                    "本轮试验: "
                    f"σ={trial_params['sigma']:.4f} | "
                    f"权重={trial_params['weight']}× | "
                    f"α={trial_params['decay_alpha']:.4f} | "
                    f"GapK={trial_params['gap_k']:.4f} | "
                    f"Q={trial_params.get('baseline_quantile', 0.5):.4f} | "
                    f"试验得分 {trial_score:.2%} | "
                    f"当前最佳 α={best_params['decay_alpha']:.4f}, Q={best_params.get('baseline_quantile', 0.5):.4f}"
                )

            result = skills_engine.run_auto_research(df, price_col, expert_labels, n_iters, progress_callback=_on_progress)
            progress_bar.progress(1.0, text="✅ 迭代完成")
            status_text.empty()

            st.success(
                f"**最优参数**: σ = {result['best_sigma']}, 偏置权重 = {result['best_weight']}×, 时序衰减系数 = {result['best_decay_alpha']}, 断层倍数 = {result['best_gap_k']}, 基准分位点 = {result.get('best_baseline_quantile', 0.5)}, 准确率 = {result['best_score']:.2%}, 冲突数 = {result['best_conflicts']}/{result['total_expert']}"
            )
            st.caption(f"搜索策略：随机参数搜索，共耗时 {result['elapsed_seconds']:.2f} 秒。")

            with st.expander("迭代历史", expanded=False):
                history_df = pd.DataFrame(result["history"])
                history_df, history_visible_columns = prepare_table_view(
                    history_df,
                    "autoresearch_history",
                    default_search_columns=list(history_df.columns[: min(4, len(history_df.columns))]),
                    filter_title="AutoResearch 迭代历史",
                )
                render_standard_data_editor(history_df[history_visible_columns], "autoresearch_history", max_height=320)

            st.markdown("---")
            st.markdown("### 📋 优化后 Skills")
            opt_skills = ComputeJob().precompute_cost_skills(
                cached_enrich_anomaly_with_ai(result["result_df"], knowledge_refresh_token),
                expert_labels,
                sigma_multiplier=result["best_sigma"],
                expert_weight=result["best_weight"],
                decay_alpha=result["best_decay_alpha"],
                gap_k=result["best_gap_k"],
                baseline_quantile=result.get("best_baseline_quantile", 0.5),
            ).all_skills

            saved_path = harness.execute_action(
                "save_skills_snapshot",
                skills=opt_skills,
                sigma=result["best_sigma"],
                weight=result["best_weight"],
                domain="cost",
            )
            exported_paths = skills_engine.export_cost_skills_excel_artifacts(
                opt_skills,
                model_export_path=settings.quantitative_skills_path,
                expert_report_export_path=settings.qualitative_skills_path,
                generated_at=datetime.now(),
                force_new=True,
            )
            bump_cost_refresh_token()
            st.success(f"✅ Skills 已自动保存至 `{saved_path}`，下次异常检测将自动加载。")
            if exported_paths.get("model_export_path"):
                st.success(f"📁 全量优化后 Skills 已写入成本分析模型导出路径：`{exported_paths['model_export_path']}`")
            if exported_paths.get("expert_report_export_path"):
                st.success(f"📁 优化后 Skills 已写入专家经验报告导出路径：`{exported_paths['expert_report_export_path']}`")
            if not exported_paths.get("model_export_path") or not exported_paths.get("expert_report_export_path"):
                st.warning("⚠️ 有导出路径未配置，未配置的路径不会自动生成 Excel 文件。")

            sync_result = sync_ai_knowledge_base(force_full=True, spinner_text="🤖 正在蒸馏专家备注并更新 AI 知识库...")
            if sync_result.get("status") == "success":
                st.success(f"🤖 {sync_result.get('message', '')}")
            elif sync_result.get("status") in {"no_changes", "no_data", "skipped"}:
                st.info(f"🤖 {sync_result.get('message', '')}")
            else:
                st.warning(f"🤖 {sync_result.get('message', '')}")

            opt_dl1, opt_dl2, opt_dl3 = st.columns(3)
            with opt_dl1:
                st.download_button(
                    "📥 下载优化后 Skills (JSON)",
                    data=skills_engine.skills_to_json_bytes(opt_skills),
                    file_name=f"skills_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width="stretch",
                )
            with opt_dl2:
                st.download_button(
                    "📥 下载优化后 Skills (Markdown)",
                    data=skills_engine.skills_to_markdown(opt_skills).encode("utf-8"),
                    file_name=f"skills_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    width="stretch",
                )
            with opt_dl3:
                render_deferred_download_button(
                    label="📥 下载优化后 Skills (Excel)",
                    prepare_label="准备优化后 Skills Excel",
                    data_builder=lambda skills_payload=list(opt_skills): skills_engine.skills_to_excel_bytes(skills_payload),
                    file_name=f"skills_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="optimized_cost_skills_excel_export",
                    fingerprint=_skills_export_fingerprint(opt_skills),
                    width="stretch",
                )

            st.markdown("### 📊 深度审计报表")
            st.markdown("本轮标注备件对照：原始结论 vs 专家反馈 vs 最终优化结论")
            audit_df = skills_engine.generate_audit_report(
                anomaly_df,
                cached_enrich_anomaly_with_ai(result["result_df"], knowledge_refresh_token),
                expert_labels,
            )
            audit_df, audit_visible_columns = prepare_table_view(
                audit_df,
                "audit_report_table",
                default_search_columns=["物料编码", "物料名称", "备件简称", "AI辅助分析"],
                filter_title="深度审计报表",
            )
            render_standard_data_editor(
                audit_df[audit_visible_columns],
                "audit_report_table",
                column_config={"AI辅助分析": st.column_config.TextColumn("AI辅助分析", disabled=True, width="large")},
                max_height=460,
            )

            audit_export_df = audit_df[audit_visible_columns].copy()
            render_deferred_download_button(
                label="📥 下载深度审计报表",
                prepare_label="准备导出深度审计报表",
                data_builder=lambda export_df=audit_export_df: to_excel_bytes(export_df),
                file_name=f"深度审计报表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="audit_report_export",
                fingerprint=dataframe_export_fingerprint(audit_export_df),
            )

def _load_interval_compare_anomaly_df() -> pd.DataFrame | None:
    render_knowledge_sync_status()

    if st.session_state.data is None:
        st.warning(NO_DATA_WARNING)
        return None

    df = st.session_state.data
    price_col = require_price_col(df)

    try:
        anomaly_df = cached_anomaly_report(df, price_col, get_cost_refresh_token())
    except Exception as exc:
        st.error(f"异常检测失败: {exc}")
        return None

    if anomaly_df.empty:
        st.info("当前数据暂无可检测记录。")
        return None

    return anomaly_df


def render_interval_compare_page() -> None:
    inject_css(is_overview=False)
    st.title("📐 车系-备件成本区间对照")

    anomaly_df = _load_interval_compare_anomaly_df()
    if anomaly_df is None:
        return

    _render_interval_compare_section(anomaly_df, show_heading=False)


def _render_interval_compare_section(anomaly_df: pd.DataFrame, *, show_heading: bool = True) -> None:
    if show_heading:
        st.markdown("## 📐 车系-备件成本区间对照")
    if anomaly_df.empty:
        st.info("当前数据暂无可检测记录，无法生成成本区间对照。")
        return

    needed_columns = ["适用车系", "备件简称", "预测值", "合理下限", "合理上限"]
    if not all(column_name in anomaly_df.columns for column_name in needed_columns):
        st.warning("异常检测结果中缺少必要列（预测值/合理下限/合理上限），无法生成图表。")
        return

    has_ring_columns = all(column_name in anomaly_df.columns for column_name in ["圈层编号", "圈层角色"])
    if has_ring_columns:
        interval_source = anomaly_df[
            anomaly_df["圈层角色"].astype(str).isin(["主邻居圈", "次邻居圈"])
        ].copy()
        if interval_source.empty:
            interval_source = anomaly_df.copy()
            interval_source["圈层编号"] = 1
            interval_source["圈层角色"] = "主邻居圈"
        interval_df = interval_source.groupby(["适用车系", "备件简称", "圈层编号", "圈层角色"], as_index=False).agg(
            {"合理下限": "median", "合理上限": "median", "预测值": "median"}
        )
        interval_df["圈层显示"] = interval_df["圈层编号"].fillna(0).astype(int).astype(str).radd("#") + " " + interval_df["圈层角色"].astype(str)
    else:
        interval_df = anomaly_df.groupby(["适用车系", "备件简称"], as_index=False).agg({"合理下限": "median", "合理上限": "median", "预测值": "median"})
        interval_df["圈层编号"] = 1
        interval_df["圈层角色"] = "主邻居圈"
        interval_df["圈层显示"] = "#1 主邻居圈"
    if interval_df.empty:
        st.info("当前筛选组合下暂无测算出的合理区间信息")
        return
    vehicle_rank_order_map = _load_vehicle_rank_order_map()
    interval_df = _append_vehicle_rank_sort_columns(interval_df, vehicle_rank_order_map)

    export_df = interval_df[["适用车系", "备件简称", "圈层编号", "圈层角色", "合理下限", "预测值", "合理上限"]].copy()
    export_df = export_df.rename(columns={"预测值": "基准价"})
    export_df["合理下限"] = export_df["合理下限"].clip(lower=0).round(2)
    export_df["合理上限"] = export_df["合理上限"].round(2)
    export_df["基准价"] = export_df["基准价"].round(2)
    export_df = export_df.drop_duplicates()
    export_df = _append_vehicle_rank_sort_columns(export_df, vehicle_rank_order_map)
    export_df = export_df.sort_values(["备件简称", "_vehicle_rank_order", "适用车系", "圈层编号"]).drop(columns=["_vehicle_rank_order"]).reset_index(drop=True)

    @st.cache_data(max_entries=2, ttl=900)
    def _build_interval_excel(df_for_export):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_for_export.to_excel(writer, index=False, sheet_name="成本区间")
        return buffer.getvalue()

    @st.cache_data(max_entries=2, ttl=900)
    def _build_lower_bound_rank_excel(summary_for_export, detail_for_export, heatmap_for_export):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            summary_for_export.to_excel(writer, index=False, sheet_name="排序总览")
            detail_for_export.to_excel(writer, index=False, sheet_name="排序明细")
            heatmap_for_export.to_excel(writer, index=False, sheet_name="偏差热力矩阵")
        return buffer.getvalue()

    fc1, fc2, fc3 = st.columns([2, 2, 1.5])
    all_vehicles = sorted(interval_df["适用车系"].astype(str).unique().tolist())
    all_parts = sorted(interval_df["备件简称"].astype(str).unique().tolist())
    vehicle_options = ["全部"] + all_vehicles
    part_options = ["全部"] + all_parts
    ensure_selectbox_state("interval_vehicle", vehicle_options, "全部")
    ensure_selectbox_state("interval_part", part_options, "全部")

    with fc1:
        vehicle_select_col, vehicle_reset_col = st.columns([4, 1])
        with vehicle_select_col:
            selected_vehicle = st.selectbox("筛选车系", options=vehicle_options, key="interval_vehicle")
        with vehicle_reset_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button(
                "重置",
                key="reset_interval_vehicle",
                width="stretch",
                on_click=reset_session_key,
                args=("interval_vehicle", "全部"),
            )
    with fc2:
        part_select_col, part_reset_col = st.columns([4, 1])
        with part_select_col:
            selected_part = st.selectbox("筛选备件简称", options=part_options, key="interval_part")
        with part_reset_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button(
                "重置",
                key="reset_interval_part",
                width="stretch",
                on_click=reset_session_key,
                args=("interval_part", "全部"),
            )
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        file_name = f"车系备件成本区间对标表_{datetime.now().strftime('%Y%m%d')}.xlsx"
        st.download_button(
            label="📥 导出全量成本区间表",
            data=_build_interval_excel(export_df),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    rank_summary_df, rank_detail_df, rank_heatmap_df = build_interval_lower_bound_rank_analysis(
        interval_df,
        vehicle_rank_order_map,
    )
    if not rank_summary_df.empty and not rank_detail_df.empty:
        st.markdown("### 🧭 车系-备件区间排序一致性分析")
        st.caption("按每个备件简称内各车系的合理区间下限从低到高排序，并与车系梯度反向后的“期望低到高排名”比较。默认只取主邻居圈。")
        low_consistency_count = int(pd.to_numeric(rank_summary_df["排序一致性"], errors="coerce").lt(0.6).sum())
        max_delta = int(pd.to_numeric(rank_summary_df["最大排名偏差"], errors="coerce").fillna(0).max())
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("参与简称", f"{len(rank_summary_df)} 个")
        metric_b.metric("一致性偏低简称", f"{low_consistency_count} 个")
        metric_c.metric("最大排名偏差", f"{max_delta} 位")

        summary_view_df, summary_visible_columns = prepare_table_view(
            rank_summary_df,
            "interval_lower_bound_rank_summary",
            display_columns=["备件简称", "覆盖车系数", "排序一致性", "最大排名偏差", "异常车系数", "重点异常车系"],
            default_search_columns=["备件简称", "重点异常车系"],
            filter_title="区间下限排序总览",
        )
        render_standard_data_editor(
            summary_view_df[summary_visible_columns],
            "interval_lower_bound_rank_summary",
            column_config={
                "排序一致性": st.column_config.NumberColumn("排序一致性", format="%.4f", disabled=True),
                "最大排名偏差": st.column_config.NumberColumn("最大排名偏差", disabled=True),
                "异常车系数": st.column_config.NumberColumn("异常车系数", disabled=True),
            },
            max_height=320,
        )

        if not rank_heatmap_df.empty and len(rank_heatmap_df.columns) > 1:
            heatmap_matrix = rank_heatmap_df.set_index("备件简称")
            heatmap_matrix = heatmap_matrix.apply(pd.to_numeric, errors="coerce")
            heatmap_height = max(360, min(900, 42 * len(heatmap_matrix.index) + 120))
            fig = px.imshow(
                heatmap_matrix,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                text_auto=True,
                labels=dict(x="车系", y="备件简称", color="排名偏差"),
                title="各备件简称下车系区间下限排序偏差热力图",
            )
            fig.update_layout(
                height=heatmap_height,
                template="plotly_white",
                margin=dict(l=10, r=20, t=60, b=10),
            )
            fig.update_xaxes(side="top")
            st.plotly_chart(fig, width="stretch")

        rank_part_options = rank_summary_df["备件简称"].astype(str).tolist()
        ensure_selectbox_state("interval_lower_bound_rank_part", rank_part_options, rank_part_options[0] if rank_part_options else None)
        selected_rank_part = st.selectbox(
            "查看单个备件简称的车系区间下限排序",
            options=rank_part_options,
            key="interval_lower_bound_rank_part",
        )
        selected_rank_detail = rank_detail_df[rank_detail_df["备件简称"].astype(str).eq(str(selected_rank_part))].copy()
        if not selected_rank_detail.empty:
            selected_rank_detail = selected_rank_detail.sort_values(["区间下限排名", "适用车系"], kind="mergesort").reset_index(drop=True)
            bar_fig = go.Figure()
            bar_colors = selected_rank_detail["偏差方向"].map(
                {"整体偏高": "#ef4444", "整体偏低": "#3b82f6", "一致": "#10b981"}
            ).fillna("#64748b")
            bar_fig.add_trace(
                go.Bar(
                    y=selected_rank_detail["适用车系"].astype(str),
                    x=selected_rank_detail["合理下限"],
                    orientation="h",
                    marker=dict(color=bar_colors),
                    customdata=selected_rank_detail[["区间下限排名", "期望低到高排名", "排名偏差", "偏差方向", "基准价", "合理上限"]].values,
                    hovertemplate=(
                        "车系: %{y}<br>合理下限: %{x:,.2f}"
                        "<br>区间下限排名: %{customdata[0]}"
                        "<br>期望低到高排名: %{customdata[1]}"
                        "<br>排名偏差: %{customdata[2]}（%{customdata[3]}）"
                        "<br>基准价: %{customdata[4]:,.2f}"
                        "<br>合理上限: %{customdata[5]:,.2f}<extra></extra>"
                    ),
                )
            )
            bar_fig.update_layout(
                title=dict(text=f"{selected_rank_part} — 各车系合理区间下限由低到高", x=0.5),
                template="plotly_white",
                height=max(360, 42 * len(selected_rank_detail) + 120),
                margin=dict(l=10, r=20, t=60, b=10),
                xaxis_title="合理区间下限",
                yaxis_title="",
            )
            bar_fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(selected_rank_detail["适用车系"].astype(str).tolist())))
            st.plotly_chart(bar_fig, width="stretch")

            detail_view_df, detail_visible_columns = prepare_table_view(
                selected_rank_detail,
                "interval_lower_bound_rank_detail",
                display_columns=[
                    "区间下限排名",
                    "适用车系",
                    "车系梯度排名",
                    "期望低到高排名",
                    "排名偏差",
                    "偏差方向",
                    "合理下限",
                    "基准价",
                    "合理上限",
                ],
                default_search_columns=["适用车系", "偏差方向"],
                filter_title="区间下限排序明细",
            )
            render_standard_data_editor(
                detail_view_df[detail_visible_columns],
                "interval_lower_bound_rank_detail",
                column_config={
                    "合理下限": st.column_config.NumberColumn("合理下限", format="%.2f", disabled=True),
                    "基准价": st.column_config.NumberColumn("基准价", format="%.2f", disabled=True),
                    "合理上限": st.column_config.NumberColumn("合理上限", format="%.2f", disabled=True),
                },
                max_height=360,
            )

        render_deferred_download_button(
            label="📥 导出区间下限排序分析",
            prepare_label="准备导出排序一致性分析",
            data_builder=lambda summary_df=rank_summary_df.copy(), detail_df=rank_detail_df.copy(), heatmap_df=rank_heatmap_df.copy(): _build_lower_bound_rank_excel(summary_df, detail_df, heatmap_df),
            file_name=f"车系备件区间下限排序分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="interval_lower_bound_rank_export",
            fingerprint=dataframe_export_fingerprint(rank_detail_df),
            width="stretch",
        )

    def _render_interval_chart(chart_data: pd.DataFrame, y_col: str, title_text: str, slider_key: str, *, group_rings_on_axis: bool = False):
        chart_data = chart_data.copy()
        global_max = float(chart_data["合理上限"].max())
        range_ceil = max(5000, int(np.ceil(global_max / 1000) * 1000))
        step = 100 if range_ceil <= 20000 else 500
        default_hi = min(range_ceil, max(5000, int(np.ceil(global_max / 1000) * 1000)))
        x_min, x_max = st.slider(
            "金额显示范围 (CNY)",
            min_value=0,
            max_value=range_ceil,
            value=(0, default_hi),
            step=step,
            key=slider_key,
        )

        chart_data = chart_data[(chart_data["合理上限"] >= x_min) & (chart_data["合理下限"] <= x_max)]
        if chart_data.empty:
            st.info("当前金额范围内无可显示的区间数据，请调整滑动条。")
            return

        if "_interval_axis_order" in chart_data.columns:
            chart_data = chart_data.sort_values(["_interval_axis_order", "圈层编号"], kind="mergesort").reset_index(drop=True)
        elif "_vehicle_rank_order" in chart_data.columns and "适用车系" in chart_data.columns:
            chart_data = chart_data.sort_values(["_vehicle_rank_order", "适用车系", y_col, "圈层编号"]).reset_index(drop=True)
        else:
            chart_data = chart_data.sort_values([y_col, "圈层编号"]).reset_index(drop=True)
        labels = chart_data[y_col].astype(str).tolist()
        y_axis_labels = list(dict.fromkeys(labels))
        role_colors = {
            "主邻居圈": "#3b82f6",
            "次邻居圈": "#f59e0b",
        }
        fallback_colors = px.colors.qualitative.Plotly
        shown_legends: set[str] = set()

        fig = go.Figure()
        for index, (_, row) in enumerate(chart_data.iterrows()):
            label = labels[index]
            lower = float(row["合理下限"])
            upper = float(row["合理上限"])
            mid = float(row["预测值"])
            ring_role = str(row.get("圈层角色", "合理区间"))
            ring_display = str(row.get("圈层显示", ring_role))
            color = role_colors.get(ring_role, fallback_colors[index % len(fallback_colors)])
            legend_name = ring_role if group_rings_on_axis else ring_display
            show_legend = bool(group_rings_on_axis and legend_name not in shown_legends)
            if show_legend:
                shown_legends.add(legend_name)

            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[upper - lower],
                    base=[lower],
                    orientation="h",
                    marker=dict(color=color, line=dict(color="rgba(15,23,42,0.18)", width=1)),
                    opacity=0.72 if group_rings_on_axis else 0.9,
                    name=legend_name,
                    legendgroup=legend_name,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{ui_utils.escape_html_text(label)}</b><br>"
                        f"圈层: {ui_utils.escape_html_text(ring_display)}<br>"
                        f"合理下限: {lower:,.2f}<br>"
                        f"合理上限: {upper:,.2f}<br>"
                        f"基准价: {mid:,.2f}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=[label],
                    x=[mid],
                    mode="markers",
                    marker=dict(symbol="line-ns-open", size=16, color="#2c3e50", line_width=2),
                    showlegend=False,
                    hovertemplate=f"基准价: {mid:,.2f}<extra></extra>",
                )
            )

        chart_height = max(520 if group_rings_on_axis else 800, len(y_axis_labels) * (48 if group_rings_on_axis else 35))
        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=16)),
            template="plotly_white",
            showlegend=group_rings_on_axis,
            height=chart_height,
            bargap=0.35,
            barmode="overlay" if group_rings_on_axis else "relative",
            margin=dict(l=10, r=20, t=60, b=10),
            plot_bgcolor="rgba(248,249,250,0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title="", tickformat=",", showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False, range=[x_min, x_max], rangeslider_visible=False)
        fig.update_yaxes(
            title="",
            automargin=True,
            ticklabelstandoff=20,
            showgrid=False,
            tickfont=dict(size=12),
            categoryorder="array",
            categoryarray=list(reversed(y_axis_labels)),
        )
        st.plotly_chart(fig, width="stretch")

    if selected_vehicle != "全部":
        chart_data = interval_df[interval_df["适用车系"] == selected_vehicle].copy()
        if chart_data.empty:
            st.info("当前筛选组合下暂无测算出的合理区间信息")
        else:
            chart_data = build_interval_compare_display_labels(chart_data, compare_mode="vehicle")
            chart_data = sort_interval_compare_chart_data(chart_data, mode="vehicle")
            _render_interval_chart(
                chart_data,
                "显示标签",
                f"车系「{selected_vehicle}」— 各备件成本合理区间",
                slider_key="interval_slider_a",
                group_rings_on_axis=True,
            )
    elif selected_part != "全部":
        chart_data = interval_df[interval_df["备件简称"] == selected_part].copy()
        if chart_data.empty:
            st.info("当前筛选组合下暂无测算出的合理区间信息")
        else:
            chart_data = build_interval_compare_display_labels(chart_data, compare_mode="part")
            chart_data = sort_interval_compare_chart_data(chart_data, mode="part")
            _render_interval_chart(
                chart_data,
                "显示标签",
                f"备件「{selected_part}」— 各车系成本合理区间",
                slider_key="interval_slider_b",
                group_rings_on_axis=True,
            )
    else:
        st.info("请在上方选择一个「车系」或一个「备件简称」以生成区间对照图。")
