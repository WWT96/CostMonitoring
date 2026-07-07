from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import harness
import sheet_metal_logic
from anomaly_engine import _EXPERT_WEIGHT, parse_accepted_ring_intervals
from app_context import (
    bump_sheet_metal_refresh_token,
    clear_sheet_metal_feedback_state,
    ensure_selectbox_state,
    get_path_setting,
    get_sheet_metal_refresh_token,
    inject_css,
    reset_session_key,
)
from compute_jobs import ComputeJob
from data_ingestion import to_excel_bytes
from page_ui_helpers import dataframe_export_fingerprint, prepare_table_view, render_deferred_download_button, render_standard_data_editor
from storage_service import find_feedback_rows_missing_required_remarks


SHEET_METAL_SKILL_DOMAIN = "sheet_metal"
_NON_MATERIAL_ANCHOR_STATE_KEY = "sheet_metal_non_material_active_anchor"
_NON_MATERIAL_RESULT_STATE_KEY = "sheet_metal_non_material_result"
_NON_MATERIAL_REVIEW_STATE_KEY = "sheet_metal_non_material_review"
_NON_MATERIAL_SCOPE_STATE_KEY = "sheet_metal_non_material_scope"
_NON_MATERIAL_MANUAL_PANEL_STATE_KEY = "sheet_metal_non_material_manual_panel_visible"
_NON_MATERIAL_EXCLUSION_LABELS = {
    "not_reasonable": "白痴指数不合理",
    "cost_missing": "成本缺失",
    "weight_missing": "重量缺失",
    "weight_invalid": "重量小于等于0",
    "steel_anchor_missing": "钢价锚点缺失",
    "material_cost_invalid": "材料成本无效",
    "short_name_missing": "备件简称缺失",
}
SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS = {
    "材料时令价格": "%.2f",
    "成本": "%.2f",
    "重量": "%.2f",
    "白痴指数": "%.2f",
    "材料成本": "%.2f",
    "非材料成本系数": "%.2f%%",
}


def _sheet_metal_skills_fingerprint(skills: list[dict]) -> str:
    preview = [
        f"{item.get('备件简称', '')}:{item.get('当前σ参数', '')}:{item.get('偏置权重', '')}"
        for item in list(skills or [])[:50]
    ]
    return f"count={len(skills or [])}|{'|'.join(map(str, preview))}"


def _safe_sheet_metal_export_label(value: object, default: str = "测算结果") -> str:
    text = str(value or "").strip() or default
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:48] or default


def build_sheet_metal_review_run_request(
    *,
    selected_short_names: list[str] | tuple[str, ...] | set[str],
    calculate_selected_clicked: bool,
    calculate_all_clicked: bool,
) -> dict:
    if calculate_all_clicked:
        return {
            "should_run": True,
            "should_export_result": True,
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
            "should_export_result": False,
            "include_all_short_names": False,
            "selected_short_names": selected,
            "scope_label": f"所选{len(selected)}个简称",
            "message": "",
        }

    message = "请选择至少一个备件简称后再点击计算。" if calculate_selected_clicked else ""
    return {
        "should_run": False,
        "should_export_result": False,
        "include_all_short_names": False,
        "selected_short_names": selected,
        "scope_label": "",
        "message": message,
    }


def build_sheet_metal_non_material_run_request(
    *,
    selected_short_names: list[str] | tuple[str, ...] | set[str],
    calculate_selected_clicked: bool,
    calculate_all_clicked: bool,
) -> dict:
    return build_sheet_metal_review_run_request(
        selected_short_names=selected_short_names,
        calculate_selected_clicked=calculate_selected_clicked,
        calculate_all_clicked=calculate_all_clicked,
    )


def export_sheet_metal_review_result_excel(
    result_df: pd.DataFrame,
    export_path: str,
    *,
    scope_label: str,
    generated_at: datetime | pd.Timestamp | None = None,
) -> str:
    if not str(export_path or "").strip():
        raise ValueError("请先在系统设置中配置钣金指数分析模型导出路径。")
    output_dir = Path(str(export_path).strip()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp(generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_label = _safe_sheet_metal_export_label(scope_label)
    output_path = output_dir / f"钣金件白痴指数复核_{safe_label}_{timestamp}.xlsx"
    export_df = result_df.copy() if result_df is not None else pd.DataFrame()
    output_path.write_bytes(to_excel_bytes(export_df, sheet_name="钣金复核结果"))
    return str(output_path)


def _inject_sheet_metal_table_css() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stDataEditor"] [role="columnheader"],
            [data-testid="stDataFrame"] [role="columnheader"] {
                justify-content: center !important;
                text-align: center !important;
            }
            [data-testid="stDataEditor"] [role="gridcell"],
            [data-testid="stDataFrame"] [role="gridcell"] {
                justify-content: flex-start !important;
                text-align: left !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(max_entries=3, ttl=900)
def _cached_load_sheet_metal_base_data(folder_path: str, refresh_token: int):
    return sheet_metal_logic.load_sheet_metal_base_data(folder_path)


@st.cache_data(max_entries=3, ttl=900)
def _cached_detect_sheet_metal_anomalies(
    base_df: pd.DataFrame,
    expert_labels_tuple: tuple,
    optimized: bool,
    refresh_token: int,
    sigma_multiplier: float = 1.0,
    expert_weight_override: int = 0,
    skills_overrides_json: str = "",
):
    return sheet_metal_logic.detect_sheet_metal_anomalies(
        base_df,
        expert_labels=dict(expert_labels_tuple),
        optimized=optimized,
        sigma_multiplier=sigma_multiplier,
        expert_weight_override=expert_weight_override,
        skills_overrides_json=skills_overrides_json,
    )


def _load_sheet_metal_skills_snapshot() -> tuple[dict | None, bool, str]:
    skills_data = harness.execute_action("load_skills_snapshot", domain=SHEET_METAL_SKILL_DOMAIN)
    snapshot_exists = harness.execute_action("has_skills_snapshot", domain=SHEET_METAL_SKILL_DOMAIN)
    skills_json = sheet_metal_logic.build_sheet_metal_skills_overrides_json(skills_data)
    return skills_data, snapshot_exists, skills_json


def _save_artifacts_to_directory(directory_path: str, artifacts: list[tuple[str, bytes]]) -> tuple[list[str], str | None]:
    target_dir = str(directory_path or "").strip()
    if not target_dir:
        return [], "请先在“系统设置”中配置对应的钣金导出路径。"

    try:
        output_dir = Path(target_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for file_name, file_bytes in artifacts:
            output_path = output_dir / file_name
            output_path.write_bytes(file_bytes)
            saved_paths.append(str(output_path))
        return saved_paths, None
    except Exception as exc:
        return [], f"保存到本地目录失败：{exc}"


def _get_sheet_metal_source() -> tuple[pd.DataFrame | None, str | None]:
    folder_path = get_path_setting("sheet_metal_base_info_path")
    if not folder_path:
        return None, "请先在“系统设置”中配置钣金件基础数据路径"
    return _cached_load_sheet_metal_base_data(folder_path, get_sheet_metal_refresh_token())


def reset_sheet_metal_review_compute_filters() -> None:
    st.session_state["sheet_metal_review_compute_short_names"] = []
    st.session_state.pop("sheet_metal_review_active_run_request", None)


def reset_sheet_metal_non_material_state() -> None:
    st.session_state["sheet_metal_non_material_short_names"] = []
    for key in [_NON_MATERIAL_RESULT_STATE_KEY, _NON_MATERIAL_REVIEW_STATE_KEY, _NON_MATERIAL_SCOPE_STATE_KEY]:
        st.session_state.pop(key, None)


def _parse_manual_steel_price_values(raw_text: object) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"\d+(?:\.\d+)?", str(raw_text or "")):
        number = pd.to_numeric(pd.Series([token]), errors="coerce").iloc[0]
        if pd.notna(number):
            value = float(number)
            if 1000 <= value <= 20000:
                values.append(value)
    return values


def _build_manual_steel_prices_from_values(raw_values: dict[str, object]) -> dict[str, list[float]]:
    manual_prices: dict[str, list[float]] = {}
    for category_name in sheet_metal_logic._STEEL_MARKET_CATEGORIES:
        values = _parse_manual_steel_price_values(raw_values.get(category_name, ""))
        if values:
            manual_prices[category_name] = values
    return manual_prices


def _build_manual_steel_prices_from_state() -> dict[str, list[float]]:
    return _build_manual_steel_prices_from_values(
        {
            category_name: st.session_state.get(f"sheet_metal_manual_steel_{category_name}", "")
            for category_name in sheet_metal_logic._STEEL_MARKET_CATEGORIES
        }
    )


def build_sheet_metal_manual_steel_anchor_request(
    *,
    manual_panel_visible: bool,
    open_manual_clicked: bool,
    save_manual_clicked: bool,
    manual_prices: dict[str, list[float]],
) -> dict:
    show_manual_inputs = bool(manual_panel_visible or open_manual_clicked or save_manual_clicked)
    normalized_prices = {
        str(name): [float(value) for value in values]
        for name, values in (manual_prices or {}).items()
        if values
    }
    should_save = bool(save_manual_clicked and normalized_prices)
    message = ""
    if save_manual_clicked and not normalized_prices:
        message = "请至少录入一个有效钢材大类价格。"
    return {
        "show_manual_inputs": show_manual_inputs,
        "should_save_manual_anchor": should_save,
        "manual_prices": normalized_prices,
        "message": message,
    }


def _render_non_material_anchor_summary(anchor: dict | None) -> None:
    if not anchor or not anchor.get("average_price_per_ton"):
        st.info("当前未设置有效钢材锚点。")
        return

    categories = anchor.get("categories") or []
    anchor_cols = st.columns(3)
    anchor_cols[0].metric("材料时令价格", f"{float(anchor['average_price_per_ton']):,.2f} 元/吨")
    anchor_cols[1].metric("钢材大类数", f"{len(categories)}")
    anchor_cols[2].metric("锚点日期", str(anchor.get("date") or "未记录"))

    if categories:
        anchor_df = pd.DataFrame(categories)
        display_cols = [column for column in ["category", "average", "source", "date"] if column in anchor_df.columns]
        anchor_df = anchor_df[display_cols].rename(
            columns={"category": "钢材大类", "average": "大类均价", "source": "来源", "date": "日期"}
        )
        with st.expander("查看钢价锚点明细", expanded=False):
            render_standard_data_editor(anchor_df, "sheet_metal_non_material_anchor_detail", max_height=220)


def _render_non_material_exclusion_summary(summary: dict | None) -> None:
    summary = summary or {}
    rows = [
        {"排除原因": _NON_MATERIAL_EXCLUSION_LABELS.get(key, key), "数量": int(summary.get(key, 0) or 0)}
        for key in _NON_MATERIAL_EXCLUSION_LABELS
        if int(summary.get(key, 0) or 0) > 0
    ]
    if rows:
        st.markdown("#### 排除统计")
        render_standard_data_editor(pd.DataFrame(rows), "sheet_metal_non_material_exclusion_summary", max_height=240)


def _render_sheet_metal_histogram(chart_df: pd.DataFrame, chart_title: str) -> None:
    if chart_df.empty:
        return

    chart_df = chart_df.copy()
    chart_df["偏离比例显示"] = chart_df["偏离比例"].apply(
        lambda value: f"{value:.2%}" if isinstance(value, (int, float)) and value == value else ""
    )

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

    nbins = min(80, max(10, int(len(chart_df) ** 0.5 * 4)))
    index_values = chart_df["白痴指数"].to_numpy(dtype=float)
    if float(np.min(index_values)) == float(np.max(index_values)):
        center_value = float(index_values[0])
        span = max(abs(center_value) * 0.05, 1.0)
        bin_edges = np.array([center_value - span, center_value + span], dtype=float)
    else:
        bin_edges = np.unique(np.histogram_bin_edges(index_values, bins=nbins).astype(float))
        if bin_edges.size < 2:
            center_value = float(np.mean(index_values))
            span = max(abs(center_value) * 0.05, 1.0)
            bin_edges = np.array([center_value - span, center_value + span], dtype=float)

    bin_ids = np.searchsorted(bin_edges, index_values, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, bin_edges.size - 2)
    chart_df["_hist_bin"] = bin_ids
    histogram_rows = []
    for bin_idx, bin_group in chart_df.groupby("_hist_bin", sort=True):
        left_edge = float(bin_edges[int(bin_idx)])
        right_edge = float(bin_edges[int(bin_idx) + 1])
        midpoint = (left_edge + right_edge) / 2.0
        ring_class = "异常区间"
        for interval in accepted_intervals:
            if float(interval["合理下限"]) <= midpoint <= float(interval["合理上限"]):
                ring_class = str(interval["圈层角色"])
                break
        histogram_rows.append(
            {
                "bin_mid": midpoint,
                "bin_width": max(right_edge - left_edge, np.finfo(float).eps) * 0.95,
                "bin_left": left_edge,
                "bin_right": right_edge,
                "raw_count": int(len(bin_group)),
                "ring_class": ring_class,
            }
        )

    histogram_df = pd.DataFrame(histogram_rows)
    fig = go.Figure()
    for ring_class in ["主邻居圈", "次邻居圈", "异常区间"]:
        class_df = histogram_df[histogram_df["ring_class"].eq(ring_class)] if not histogram_df.empty else pd.DataFrame()
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
                customdata=class_df[["bin_left", "bin_right", "ring_class"]].values,
                hovertemplate=(
                    "区间类型: %{customdata[2]}"
                    "<br>指数区间: %{customdata[0]:,.4f} - %{customdata[1]:,.4f}"
                    "<br>样本频数: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    if accepted_intervals:
        for interval in accepted_intervals:
            role = str(interval["圈层角色"])
            color = role_colors.get(role, "#3b82f6")
            lower = float(interval["合理下限"])
            upper = float(interval["合理上限"])
            baseline = float(interval["预测值"])
            fig.add_vrect(x0=lower, x1=upper, fillcolor=color, opacity=0.08, line_width=0)
            fig.add_vline(x=baseline, line_dash="dash", line_color=color, annotation_text=f"{role}基准", annotation_position="top")
            fig.add_vline(x=lower, line_dash="dot", line_color=color)
            fig.add_vline(x=upper, line_dash="dot", line_color=color)
    else:
        baseline = float(anchor_df["基准指数"].median())
        upper = float(anchor_df["合理上限"].median())
        lower = float(anchor_df["合理下限"].median())
        fig.add_vline(x=baseline, line_dash="dash", line_color="#1f77b4", annotation_text="基准指数", annotation_position="top")
        fig.add_vline(x=upper, line_dash="dash", line_color="#d62728", annotation_text="合理上限", annotation_position="top")
        fig.add_vline(x=lower, line_dash="dash", line_color="#d62728", annotation_text="合理下限", annotation_position="top")

    abnormal_points = chart_df[chart_df["status"].astype(str).str.contains("异常偏高|异常偏低|严重异常偏低")].copy()
    if not abnormal_points.empty:
        fig.add_trace(
            go.Scatter(
                x=abnormal_points["白痴指数"],
                y=[0] * len(abnormal_points),
                mode="markers",
                marker=dict(size=10, color="#e74c3c"),
                name="异常点",
                customdata=abnormal_points[["物料编码", "白痴指数", "偏离比例显示"]].values,
                hovertemplate="物料编码: %{customdata[0]}<br>白痴指数: %{customdata[1]:,.4f}<br>偏离比例: %{customdata[2]}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=f"{chart_title}<br><sup>静态钣金件复核模式：按物料编码保留最后一条记录，不构建时间序列。</sup>"),
        xaxis_title="白痴指数",
        yaxis_title="频数",
        template="plotly_white",
        bargap=0.05,
    )
    st.plotly_chart(fig, width="stretch")


def render_sheet_metal_review_page() -> None:
    inject_css(is_overview=False)
    _inject_sheet_metal_table_css()
    st.title("🧩 钣金件白痴指数复核")
    st.caption("静态复核模式：系统仅读取钣金件基础数据目录中的 Excel，按物料编码保留最后一条记录，不显示折线图或任何时间动态组件。")

    base_df, error_message = _get_sheet_metal_source()
    if error_message:
        st.warning(error_message)
        return
    if base_df is None or base_df.empty:
        st.info("当前钣金件基础数据为空。")
        return

    label_details = harness.execute_action("get_sheet_metal_feedback_details")
    label_statuses = {record_key: payload.get("label", "") for record_key, payload in label_details.items()}
    label_remarks = {record_key: payload.get("remark", "") for record_key, payload in label_details.items()}
    skills_data, skills_snapshot_exists, skills_overrides_json = _load_sheet_metal_skills_snapshot()
    skills_loaded = bool(skills_data and skills_data.get("skills"))
    current_sigma = float((skills_data or {}).get("global_sigma") or 1.0)
    current_weight = int((skills_data or {}).get("global_weight") or _EXPERT_WEIGHT)

    mode_col, stat_col = st.columns([3, 2])
    with mode_col:
        st.radio(
            "复核模式",
            options=["原始测算", "优化后测算（专家纠偏）"],
            key="sheet_metal_anomaly_mode",
            horizontal=True,
        )
    with stat_col:
        st.metric("独立钣金标注数", f"{len(label_details)} 条")

    st.info("钣金件专家标注独立保存在本地表 `sheet_metal_feedback`，不会与成本异常标注混淆。")
    if skills_loaded:
        st.info(
            f"📘 已加载数据库中的最新钣金指数技能书（{len(skills_data.get('skills', []))} 个备件简称，保存于 {skills_data.get('saved_at', '未知')}）。"
            "当前测算会优先应用个性化 σ 边界，并在报表中标记 [技能书校验]。"
        )
    elif skills_data is None and skills_snapshot_exists:
        st.warning("⚠️ 钣金指数技能书快照读取异常，当前已回退到默认算法。请前往“钣金件指数技能书”页面重新生成。")

    if label_details:
        with st.expander("📋 查看/管理已校准钣金记录", expanded=False):
            mgmt_df = sheet_metal_logic.build_sheet_metal_calibration_management_df(label_details, source_df=base_df)
            mgmt_display_columns = ["物料编码", "物料名称", "备件简称", "工厂", "白痴指数", "当前标注", "标注备注", "撤回标注"]
            mgmt_editor_source = mgmt_df.set_index("record_key", drop=True)
            mgmt_filtered_df, mgmt_visible_columns = prepare_table_view(
                mgmt_editor_source,
                "sheet_metal_calibration_mgmt",
                display_columns=mgmt_display_columns,
                default_search_columns=["物料编码", "物料名称", "备件简称", "工厂", "标注备注"],
                locked_columns=mgmt_display_columns,
                filter_title="已校准钣金记录",
            )
            mgmt_edited = render_standard_data_editor(
                mgmt_filtered_df[mgmt_visible_columns],
                "sheet_metal_calibration_mgmt",
                editable_columns=["标注备注", "撤回标注"],
                column_config={
                    "物料编码": st.column_config.TextColumn("物料编码", disabled=True),
                    "物料名称": st.column_config.TextColumn("物料名称", disabled=True, width="large"),
                    "备件简称": st.column_config.TextColumn("备件简称", disabled=True),
                    "工厂": st.column_config.TextColumn("工厂", disabled=True),
                    "白痴指数": st.column_config.NumberColumn("白痴指数", disabled=True, format="%.4f"),
                    "当前标注": st.column_config.TextColumn("当前标注", disabled=True),
                    "撤回标注": st.column_config.CheckboxColumn("撤回标注", help="勾选后点击下方按钮撤回此标注", default=False),
                    "标注备注": st.column_config.TextColumn("标注备注", help="可直接编辑备注，例如材质、工艺或供应商异常原因。", width="large"),
                },
                max_height=320,
            )

            mgmt_c1, mgmt_c2, mgmt_c3 = st.columns(3)
            with mgmt_c1:
                if st.button("💾 保存钣金标注修改", type="primary"):
                    final_mgmt_df = mgmt_df.copy()
                    edited_remark_map = {
                        str(record_key): str(remark or "")
                        for record_key, remark in zip(mgmt_edited.index, mgmt_edited["标注备注"])
                    }
                    final_mgmt_df["标注备注"] = final_mgmt_df["record_key"].astype(str).map(edited_remark_map).fillna(final_mgmt_df["标注备注"])
                    final_labels_df = final_mgmt_df[["record_key", "当前标注", "标注备注"]].rename(columns={"当前标注": "label", "标注备注": "remark"})
                    missing_remark_keys = find_feedback_rows_missing_required_remarks(final_labels_df)
                    if missing_remark_keys:
                        st.error(f"标注为正常的记录必须填写批注原因，共 {len(missing_remark_keys)} 条。")
                    else:
                        try:
                            harness.execute_action("replace_sheet_metal_feedback", final_labels_df=final_labels_df)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            bump_sheet_metal_refresh_token()
                            st.success(f"✅ 已保存 {len(final_labels_df)} 条钣金标注备注修改")
                            st.rerun()
            with mgmt_c2:
                if st.button("🗑️ 撤回选中的钣金标注"):
                    keys_to_revoke = [str(record_key) for record_key, row in mgmt_edited.iterrows() if bool(row["撤回标注"])]
                    if keys_to_revoke:
                        revoked = harness.execute_action("delete_sheet_metal_feedback", keys_to_remove=keys_to_revoke)
                        bump_sheet_metal_refresh_token()
                        st.success(f"✅ 已撤回 {revoked} 条钣金标注")
                        st.rerun()
                    else:
                        st.warning("未选中任何记录")
            with mgmt_c3:
                if st.button("⚠️ 清空所有钣金标注"):
                    st.session_state["_confirm_clear_sheet_metal_labels"] = True
                if st.session_state.get("_confirm_clear_sheet_metal_labels"):
                    st.warning("确定要清空所有钣金专家标注吗？此操作不可撤销。")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("✅ 确认清空钣金标注", type="primary"):
                            clear_sheet_metal_feedback_state()
                            st.session_state["_confirm_clear_sheet_metal_labels"] = False
                            st.success("✅ 已清空所有钣金标注")
                            st.rerun()
                    with cc2:
                        if st.button("取消钣金清空"):
                            st.session_state["_confirm_clear_sheet_metal_labels"] = False
                            st.rerun()

    st.markdown("---")

    st.markdown("#### 🔍 测算范围")
    short_name_source_options = sorted(base_df["备件简称"].dropna().astype(str).unique().tolist()) if "备件简称" in base_df.columns else []
    compute_col, selected_run_col, full_run_col, reset_col = st.columns([4, 1.5, 1.7, 1], vertical_alignment="bottom")
    with compute_col:
        selected_compute_short_names = st.multiselect(
            "钣金备件简称筛选（仅选择不触发测算）",
            options=short_name_source_options,
            key="sheet_metal_review_compute_short_names",
            help="选择后点击“计算所选简称”；进入页面默认不会进行钣金异常测算。",
        )
    with selected_run_col:
        calculate_selected_clicked = st.button(
            "计算所选简称",
            key="sheet_metal_review_calculate_selected",
            width="stretch",
            disabled=not selected_compute_short_names,
        )
    with full_run_col:
        calculate_all_clicked = st.button(
            "一键计算全量简称",
            key="sheet_metal_review_calculate_all",
            width="stretch",
        )
    with reset_col:
        st.button(
            "重置",
            key="reset_sheet_metal_review_compute_short_names",
            width="stretch",
            on_click=reset_sheet_metal_review_compute_filters,
        )

    current_run_request = build_sheet_metal_review_run_request(
        selected_short_names=selected_compute_short_names,
        calculate_selected_clicked=calculate_selected_clicked,
        calculate_all_clicked=calculate_all_clicked,
    )
    should_export_run_result = bool(current_run_request.get("should_export_result"))
    if current_run_request["should_run"]:
        st.session_state["sheet_metal_review_active_run_request"] = current_run_request
    elif current_run_request.get("message"):
        st.warning(current_run_request["message"])

    active_run_request = st.session_state.get("sheet_metal_review_active_run_request") or current_run_request
    if not active_run_request.get("should_run"):
        st.info("初始状态不会进行钣金件白痴指数复核测算。请先选择备件简称后点击“计算所选简称”，或点击“一键计算全量简称”。")
        return

    include_all_short_names = bool(active_run_request.get("include_all_short_names"))
    selected_compute_short_names = list(active_run_request.get("selected_short_names") or [])
    review_source_df = base_df.copy()
    if not include_all_short_names:
        selected_set = set(selected_compute_short_names)
        review_source_df = review_source_df[review_source_df["备件简称"].astype(str).isin(selected_set)].copy()
    if review_source_df.empty:
        st.info("当前钣金测算范围为空，请调整备件简称后重新点击计算。")
        return

    selected_scope_text = str(active_run_request.get("scope_label") or ("全量简称" if include_all_short_names else f"所选{len(selected_compute_short_names)}个简称"))
    st.caption(f"当前钣金测算范围：{selected_scope_text}，共 {len(review_source_df)} 条记录。")

    is_expert_mode = st.session_state.sheet_metal_anomaly_mode == "优化后测算（专家纠偏）"
    active_labels_tuple = tuple(sorted(label_statuses.items())) if is_expert_mode and bool(label_statuses) else tuple()
    review_df = _cached_detect_sheet_metal_anomalies(
        review_source_df,
        active_labels_tuple,
        is_expert_mode and bool(label_statuses),
        get_sheet_metal_refresh_token(),
        current_sigma,
        current_weight,
        skills_overrides_json,
    )

    if review_df.empty:
        st.info("当前钣金件数据暂无可检测记录。")
        return

    if "_record_key" in review_df.columns:
        review_df["专家校准"] = review_df["_record_key"].astype(str).map(lambda key: "✅" if label_statuses.get(key) == "正常" else "")
        review_df["专家备注"] = review_df["_record_key"].astype(str).map(lambda key: label_remarks.get(key, "")).fillna("")

    if should_export_run_result:
        try:
            exported_path = export_sheet_metal_review_result_excel(
                review_df.drop(columns=["_record_key"], errors="ignore"),
                get_path_setting("sheet_metal_model_export_path"),
                scope_label=selected_scope_text,
                generated_at=datetime.now(),
            )
            st.success(f"📁 本次钣金复核测算结果已保存至钣金指数分析模型导出路径：`{exported_path}`")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.warning(f"本次钣金复核已完成，但写入钣金指数分析模型导出路径失败：{exc}")

    high_count = int(review_df["status"].astype(str).str.contains("异常偏高").sum())
    low_count = int(review_df["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum())
    abnormal_total = high_count + low_count

    m1, m2, m3 = st.columns(3)
    m1.metric("异常总数", f"{abnormal_total}")
    m2.metric("异常偏高", f"{high_count}")
    m3.metric("异常偏低", f"{low_count}")

    short_name_options = ["全部"] + sorted(review_df["备件简称"].astype(str).unique().tolist())
    ensure_selectbox_state("sheet_metal_short_name", short_name_options, "全部")
    short_filter_col, short_reset_col = st.columns([5, 1])
    with short_filter_col:
        selected_short_name = st.selectbox("🔍 备件简称筛选", short_name_options, key="sheet_metal_short_name")
    with short_reset_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button(
            "重置",
            key="reset_sheet_metal_short_name",
            width="stretch",
            on_click=reset_session_key,
            args=("sheet_metal_short_name", "全部"),
        )

    filtered_df = review_df.copy()
    if selected_short_name != "全部":
        filtered_df = filtered_df[filtered_df["备件简称"].astype(str) == selected_short_name].copy()

    abnormal_view = filtered_df[filtered_df["status"].astype(str).str.contains("异常偏高|异常偏低|严重异常偏低")].copy()
    st.markdown(f"**共找到 {len(abnormal_view)} 条异常记录**")

    if not abnormal_view.empty and "_record_key" in abnormal_view.columns:
        edit_df = abnormal_view.copy()
        edit_df["标注为正常"] = edit_df["_record_key"].astype(str).map(lambda key: label_statuses.get(key) == "正常")
        edit_df["专家校准"] = edit_df["_record_key"].astype(str).map(lambda key: "✅" if label_statuses.get(key) == "正常" else "")
        edit_df["标注备注"] = edit_df["_record_key"].astype(str).map(lambda key: label_remarks.get(key, ""))

        preferred_display_cols = [
            "标注为正常",
            "标注备注",
            "专家校准",
            "判定依据",
            "车型",
            "车系",
            "车型梯度",
            "物料编码",
            "物料描述",
            "备件简称",
            "产品成本",
            "出厂单价",
            "包装费",
            "净重",
            "包装后重量",
            "白痴指数",
            "基准指数",
            "合理下限",
            "合理上限",
            "偏离指数",
            "偏离比例",
            "工厂",
            "适用车系",
            "样本量",
            "status",
        ]
        display_cols = [
            column_name
            for column_name in preferred_display_cols
            if column_name in edit_df.columns and column_name not in {"_record_key", "静态快照时间", "monitor_date", "数据来源文件", "数据来源工作表", "专家备注"}
        ]
        display_cols += [
            column_name
            for column_name in edit_df.columns
            if column_name not in display_cols and column_name not in {"_record_key", "静态快照时间", "monitor_date", "数据来源文件", "数据来源工作表", "专家备注"}
        ]
        priority_cols = [column_name for column_name in ["标注为正常", "标注备注", "专家校准"] if column_name in display_cols]
        display_cols = priority_cols + [column_name for column_name in display_cols if column_name not in priority_cols]

        visible_edit_df, visible_columns = prepare_table_view(
            edit_df,
            "sheet_metal_editor",
            display_columns=display_cols,
            default_search_columns=["物料编码", "物料描述", "备件简称", "车系", "status"],
            locked_columns=["标注为正常", "标注备注", "专家校准"],
            filter_title="钣金异常记录",
        )
        visible_edit_df = visible_edit_df.reset_index(drop=True)
        edited_df = render_standard_data_editor(
            visible_edit_df[visible_columns],
            "sheet_metal_editor",
            editable_columns=["标注为正常", "标注备注"],
            column_config={
                "标注为正常": st.column_config.CheckboxColumn("标注为正常", help="勾选此项将该记录标注为「正常」", default=False),
                "标注备注": st.column_config.TextColumn("标注备注", help="填写钣金专家备注，例如材质、重量或工艺原因说明。", width="large"),
                "专家校准": st.column_config.TextColumn("专家校准", disabled=True),
                "判定依据": st.column_config.TextColumn("判定依据", disabled=True),
                "白痴指数": st.column_config.NumberColumn("白痴指数", disabled=True, format="%.4f"),
                "基准指数": st.column_config.NumberColumn("基准指数", disabled=True, format="%.4f"),
                "合理下限": st.column_config.NumberColumn("合理下限", disabled=True, format="%.4f"),
                "合理上限": st.column_config.NumberColumn("合理上限", disabled=True, format="%.4f"),
                "偏离指数": st.column_config.NumberColumn("偏离指数", disabled=True, format="%.4f"),
                "偏离比例": st.column_config.NumberColumn("偏离比例", disabled=True, format="%.2%%"),
            },
            max_height=460,
        )

        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            if st.button("💾 保存钣金专家标注", type="primary", width="stretch"):
                final_labels = {
                    key: {"label": payload.get("label", ""), "remark": payload.get("remark", "")}
                    for key, payload in label_details.items()
                }
                for index, (_, orig_row) in enumerate(visible_edit_df.iterrows()):
                    record_key = str(orig_row["_record_key"])
                    checked = bool(edited_df.iloc[index]["标注为正常"])
                    remark_text = str(edited_df.iloc[index].get("标注备注", "") or "").strip()
                    if checked or remark_text:
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
                        harness.execute_action("replace_sheet_metal_feedback", final_labels_df=final_labels_df)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        bump_sheet_metal_refresh_token()
                        st.success(f"✅ 钣金标注已保存！当前共 {len(final_labels)} 条独立钣金标注记录。")
                        st.rerun()
    elif abnormal_view.empty:
        st.info("当前筛选条件下暂无异常记录。")
    else:
        abnormal_view, readonly_visible_columns = prepare_table_view(
            abnormal_view,
            "sheet_metal_readonly",
            display_columns=[column_name for column_name in [
                "车型",
                "车系",
                "车型梯度",
                "物料编码",
                "物料描述",
                "备件简称",
                "产品成本",
                "出厂单价",
                "包装费",
                "净重",
                "包装后重量",
                "白痴指数",
                "基准指数",
                "合理下限",
                "合理上限",
                "偏离指数",
                "偏离比例",
                "工厂",
                "适用车系",
                "样本量",
                "status",
                "判定依据",
                "专家校准",
                "标注备注",
            ] if column_name in abnormal_view.columns],
            default_search_columns=["物料编码", "物料描述", "备件简称", "车系", "status"],
            filter_title="钣金异常记录（只读）",
        )
        render_standard_data_editor(abnormal_view[readonly_visible_columns], "sheet_metal_readonly", max_height=460)

    if "action_col2" not in locals():
        _, action_col2 = st.columns([1, 1])
    export_df = abnormal_view.drop(columns=["_record_key", "monitor_date"], errors="ignore")
    with action_col2:
        render_deferred_download_button(
            label="📥 下载钣金件白痴指数复核报表",
            prepare_label="准备导出钣金件复核报表",
            data_builder=lambda export_frame=export_df.copy(): to_excel_bytes(export_frame),
            file_name=f"钣金件白痴指数复核_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sheet_metal_review_export",
            fingerprint=dataframe_export_fingerprint(export_df),
            width="stretch",
        )

    chart_title = "全部备件简称 - 白痴指数分布" if selected_short_name == "全部" else f"{selected_short_name} - 白痴指数分布"
    _render_sheet_metal_histogram(filtered_df, chart_title)


def render_sheet_metal_non_material_coefficients_page() -> None:
    inject_css(is_overview=False)
    _inject_sheet_metal_table_css()
    st.title("🧮 钣金件非材料成本系数")
    st.caption("基于钢材大类时令价锚点和钣金白痴指数合理样本，反推备件简称级非材料成本系数。")

    base_df, error_message = _get_sheet_metal_source()
    if error_message:
        st.warning(error_message)
        return
    if base_df is None or base_df.empty:
        st.info("当前钣金件基础数据为空。")
        return

    st.markdown("### 材料锚点")
    anchor_date = st.date_input("锚点日期", value=datetime.now().date(), key="sheet_metal_non_material_anchor_date")
    fetch_col, manual_col, clear_col = st.columns([1.4, 1.4, 1], vertical_alignment="bottom")
    with fetch_col:
        fetch_clicked = st.button("抓取生意社钢价", key="sheet_metal_fetch_public_steel_anchor", width="stretch")
    with manual_col:
        manual_clicked = st.button("使用手动钢价", key="sheet_metal_use_manual_steel_anchor", width="stretch")
    with clear_col:
        clear_clicked = st.button("清空结果", key="sheet_metal_non_material_clear", width="stretch")

    if clear_clicked:
        st.session_state.pop(_NON_MATERIAL_ANCHOR_STATE_KEY, None)
        st.session_state[_NON_MATERIAL_MANUAL_PANEL_STATE_KEY] = False
        reset_sheet_metal_non_material_state()
        st.rerun()

    if fetch_clicked:
        st.session_state[_NON_MATERIAL_MANUAL_PANEL_STATE_KEY] = False
        with st.spinner("正在抓取生意社公开钢价..."):
            anchor = sheet_metal_logic.load_sheet_metal_steel_market_anchor(as_of_date=anchor_date)
        st.session_state[_NON_MATERIAL_ANCHOR_STATE_KEY] = anchor
        if anchor.get("average_price_per_ton"):
            st.success("已更新生意社钢价锚点。")
        else:
            st.warning("生意社钢价抓取未得到有效均价，请使用手动钢价。")

    if manual_clicked:
        st.session_state[_NON_MATERIAL_MANUAL_PANEL_STATE_KEY] = True

    save_manual_clicked = False
    manual_input_values: dict[str, object] = {}
    if st.session_state.get(_NON_MATERIAL_MANUAL_PANEL_STATE_KEY, False):
        with st.form("sheet_metal_manual_steel_anchor_form", clear_on_submit=False):
            anchor_cols = st.columns(4)
            for index, category_name in enumerate(sheet_metal_logic._STEEL_MARKET_CATEGORIES):
                with anchor_cols[index % len(anchor_cols)]:
                    manual_input_values[category_name] = st.text_input(
                        f"{category_name}（元/吨）",
                        key=f"sheet_metal_manual_steel_{category_name}",
                        placeholder="可录入多个报价，用逗号分隔",
                    )
            save_manual_clicked = st.form_submit_button("保存手动钢价", type="primary")

    manual_prices = (
        _build_manual_steel_prices_from_values(manual_input_values)
        if manual_input_values
        else _build_manual_steel_prices_from_state()
    )
    manual_request = build_sheet_metal_manual_steel_anchor_request(
        manual_panel_visible=bool(st.session_state.get(_NON_MATERIAL_MANUAL_PANEL_STATE_KEY, False)),
        open_manual_clicked=manual_clicked,
        save_manual_clicked=save_manual_clicked,
        manual_prices=manual_prices,
    )
    st.session_state[_NON_MATERIAL_MANUAL_PANEL_STATE_KEY] = bool(manual_request["show_manual_inputs"])
    if save_manual_clicked:
        if manual_request["should_save_manual_anchor"]:
            st.session_state[_NON_MATERIAL_ANCHOR_STATE_KEY] = sheet_metal_logic.load_sheet_metal_steel_market_anchor(
                manual_prices=manual_request["manual_prices"],
                as_of_date=anchor_date,
            )
            st.success("已保存手动钢价锚点。")
        else:
            st.warning(str(manual_request.get("message") or "请至少录入一个有效钢材大类价格。"))

    active_anchor = st.session_state.get(_NON_MATERIAL_ANCHOR_STATE_KEY)
    _render_non_material_anchor_summary(active_anchor)

    st.markdown("---")
    st.markdown("### 测算范围")

    skills_data, skills_snapshot_exists, skills_overrides_json = _load_sheet_metal_skills_snapshot()
    skills_loaded = bool(skills_data and skills_data.get("skills"))
    current_sigma = float((skills_data or {}).get("global_sigma") or 1.0)
    current_weight = int((skills_data or {}).get("global_weight") or _EXPERT_WEIGHT)
    label_details = harness.execute_action("get_sheet_metal_feedback_details")
    label_statuses = {record_key: payload.get("label", "") for record_key, payload in label_details.items()}

    st.radio(
        "测算模式",
        options=["原始测算", "优化后测算（专家纠偏）"],
        key="sheet_metal_non_material_mode",
        horizontal=True,
    )

    if skills_loaded:
        st.info(f"已加载数据库中的最新钣金指数技能书，当前测算会应用已保存的个性化边界。")
    elif skills_data is None and skills_snapshot_exists:
        st.warning("钣金指数技能书快照读取异常，当前已回退到默认算法。")

    short_name_options = sorted(base_df["备件简称"].dropna().astype(str).unique().tolist()) if "备件简称" in base_df.columns else []
    filter_col, reset_col = st.columns([5, 1], vertical_alignment="bottom")
    with filter_col:
        selected_short_names = st.multiselect(
            "备件简称筛选（用于计算所选简称）",
            options=short_name_options,
            key="sheet_metal_non_material_short_names",
        )
    with reset_col:
        st.button(
            "重置",
            key="sheet_metal_non_material_reset",
            width="stretch",
            on_click=reset_sheet_metal_non_material_state,
        )

    selected_calc_col, full_calc_col = st.columns([1.3, 1.3], vertical_alignment="bottom")
    with selected_calc_col:
        calculate_selected_clicked = st.button(
            "计算所选简称",
            type="secondary",
            key="sheet_metal_non_material_calculate_selected",
            width="stretch",
            disabled=not selected_short_names,
        )
    with full_calc_col:
        calculate_all_clicked = st.button(
            "一键计算全量简称",
            type="primary",
            key="sheet_metal_non_material_calculate_all",
            width="stretch",
        )

    run_request = build_sheet_metal_non_material_run_request(
        selected_short_names=selected_short_names,
        calculate_selected_clicked=calculate_selected_clicked,
        calculate_all_clicked=calculate_all_clicked,
    )

    if run_request.get("message"):
        st.warning(str(run_request["message"]))

    if run_request["should_run"]:
        if not active_anchor or not active_anchor.get("average_price_per_ton"):
            st.warning("请先抓取或录入有效钢材锚点。")
        else:
            compute_source_df = base_df.copy()
            if not run_request.get("include_all_short_names"):
                selected_set = set(map(str, run_request.get("selected_short_names") or []))
                compute_source_df = compute_source_df[compute_source_df["备件简称"].astype(str).isin(selected_set)].copy()

            if compute_source_df.empty:
                st.warning("当前测算范围为空，请调整备件简称。")
            else:
                is_expert_mode = st.session_state.sheet_metal_non_material_mode == "优化后测算（专家纠偏）"
                active_labels_tuple = tuple(sorted(label_statuses.items())) if is_expert_mode and bool(label_statuses) else tuple()
                with st.spinner("正在计算非材料成本系数..."):
                    review_df = _cached_detect_sheet_metal_anomalies(
                        compute_source_df,
                        active_labels_tuple,
                        is_expert_mode and bool(label_statuses),
                        get_sheet_metal_refresh_token(),
                        current_sigma,
                        current_weight,
                        skills_overrides_json,
                    )
                    samples_df = sheet_metal_logic.build_reasonable_sheet_metal_samples(review_df)
                    result_df = sheet_metal_logic.calculate_non_material_coefficients(samples_df, active_anchor)

                scope_label = str(run_request.get("scope_label") or "测算结果")
                st.session_state[_NON_MATERIAL_RESULT_STATE_KEY] = result_df
                st.session_state[_NON_MATERIAL_REVIEW_STATE_KEY] = review_df
                st.session_state[_NON_MATERIAL_SCOPE_STATE_KEY] = scope_label
                if run_request.get("should_export_result"):
                    st.success("全量非材料成本系数已计算完成，可在下方准备并下载 Excel。")

    result_df = st.session_state.get(_NON_MATERIAL_RESULT_STATE_KEY)
    review_df = st.session_state.get(_NON_MATERIAL_REVIEW_STATE_KEY)
    scope_label = st.session_state.get(_NON_MATERIAL_SCOPE_STATE_KEY) or "未测算"

    if result_df is None:
        st.info("初始状态不会进行钣金非材料成本系数测算。")
        return

    summary = result_df.attrs.get("excluded_summary", {}) if isinstance(result_df, pd.DataFrame) else {}
    valid_count = len(result_df) if isinstance(result_df, pd.DataFrame) else 0
    reasonable_count = int(len(sheet_metal_logic.build_reasonable_sheet_metal_samples(review_df))) if isinstance(review_df, pd.DataFrame) else 0
    excluded_total = int(sum(int(value or 0) for value in summary.values()))

    st.markdown("### 测算结果")
    metric_cols = st.columns(4)
    metric_cols[0].metric("测算范围", str(scope_label))
    metric_cols[1].metric("合理样本数", f"{reasonable_count}")
    metric_cols[2].metric("有效输出行", f"{valid_count}")
    metric_cols[3].metric("排除数量", f"{excluded_total}")

    _render_non_material_exclusion_summary(summary)

    if result_df.empty:
        st.warning("当前没有可输出的非材料成本系数结果。")
        return

    preview_df, visible_columns = prepare_table_view(
        result_df,
        "sheet_metal_non_material_coefficients",
        display_columns=sheet_metal_logic.SHEET_METAL_NON_MATERIAL_OUTPUT_COLUMNS,
        default_search_columns=["物料编码", "物料名称", "备件简称"],
        filter_title="钣金件非材料成本系数",
    )
    render_standard_data_editor(
        preview_df[visible_columns],
        "sheet_metal_non_material_coefficients",
        column_config={
            "样本数": st.column_config.NumberColumn("样本数", disabled=True),
            "材料时令价格": st.column_config.NumberColumn(
                "材料时令价格",
                disabled=True,
                format=SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS["材料时令价格"],
            ),
            "成本": st.column_config.NumberColumn(
                "成本",
                disabled=True,
                format=SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS["成本"],
            ),
            "重量": st.column_config.NumberColumn(
                "重量",
                disabled=True,
                format=SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS["重量"],
            ),
            "白痴指数": st.column_config.NumberColumn(
                "白痴指数",
                disabled=True,
                format=SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS["白痴指数"],
            ),
            "材料成本": st.column_config.NumberColumn(
                "材料成本",
                disabled=True,
                format=SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS["材料成本"],
            ),
            "非材料成本系数": st.column_config.NumberColumn(
                "非材料成本系数",
                disabled=True,
                format=SHEET_METAL_NON_MATERIAL_RESULT_NUMBER_FORMATS["非材料成本系数"],
            ),
        },
        max_height=520,
    )

    export_scope_label = _safe_sheet_metal_export_label(scope_label, default="测算结果")
    export_name = f"钣金件非材料成本系数_{export_scope_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    render_deferred_download_button(
        label="下载钣金件非材料成本系数",
        prepare_label="准备导出钣金件非材料成本系数",
        data_builder=lambda export_frame=result_df.copy(): to_excel_bytes(export_frame, sheet_name="非材料成本系数"),
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="sheet_metal_non_material_export",
        fingerprint=dataframe_export_fingerprint(result_df),
        width="stretch",
    )


def render_sheet_metal_skills_page() -> None:
    inject_css(is_overview=False)
    _inject_sheet_metal_table_css()
    st.title("🧠 钣金件指数技能书")
    st.caption("钣金件指数技能书基于静态白痴指数区间提取，并在本页面集中完成自学习训练、导出与报告沉淀。")

    base_df, error_message = _get_sheet_metal_source()
    if error_message:
        st.warning(error_message)
        return
    if base_df is None or base_df.empty:
        st.info("当前钣金件基础数据为空。")
        return

    expert_labels = harness.execute_action("get_sheet_metal_feedback_statuses")
    skills_data, skills_snapshot_exists, _ = _load_sheet_metal_skills_snapshot()
    auto_research_result = None
    auto_research_audit_df = pd.DataFrame()

    st.markdown("### 🔬 钣金指数自学习训练")
    st.caption("系统将基于 DGB + 策略 B 加权方案执行棘轮迭代，自动寻找更优的 σ 参数与偏置权重，并与普通备件 AutoResearch 保持一致。")

    train_c1, train_c2 = st.columns([2, 2])
    with train_c1:
        auto_iterations = st.select_slider(
            "自学习迭代轮数",
            options=[5, 10, 20],
            value=10,
            key="sheet_metal_autoresearch_iterations",
        )
    with train_c2:
        auto_research_clicked = st.button(
            "🚀 运行钣金指数自学习训练",
            type="primary",
            width="stretch",
            disabled=not bool(expert_labels),
        )

    if not expert_labels:
        st.warning("当前尚无钣金专家标注，暂不能启动自学习训练。请先在“钣金件白痴指数复核”页面完成标注并保存。")

    if auto_research_clicked:
        original_review_df = _cached_detect_sheet_metal_anomalies(
            base_df,
            tuple(),
            False,
            get_sheet_metal_refresh_token(),
        )
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def _progress_callback(current: int, total: int, best_score: float, trial_score: float, sigma: float, weight: int) -> None:
            ratio = min(current / max(total, 1), 1.0)
            progress_bar.progress(ratio)
            progress_text.caption(
                f"第 {current}/{total} 轮 | 当前最佳对齐率 {best_score:.1%} | 本轮对齐率 {trial_score:.1%} | σ={sigma:.4f} | 权重={weight}x"
            )

        auto_research_result = sheet_metal_logic.run_sheet_metal_auto_research(
            base_df,
            expert_labels,
            n_iterations=int(auto_iterations),
            progress_callback=_progress_callback,
        )
        generated_skills = ComputeJob().precompute_sheet_metal_skills(
            auto_research_result["result_df"],
            expert_labels,
            sigma_multiplier=auto_research_result["best_sigma"],
            expert_weight=auto_research_result["best_weight"],
        ).all_skills
        harness.execute_action(
            "save_skills_snapshot",
            skills=generated_skills,
            sigma=auto_research_result["best_sigma"],
            weight=auto_research_result["best_weight"],
            domain=SHEET_METAL_SKILL_DOMAIN,
        )
        bump_sheet_metal_refresh_token()
        skills_data, skills_snapshot_exists, _ = _load_sheet_metal_skills_snapshot()
        auto_research_audit_df = sheet_metal_logic.build_sheet_metal_audit_report(
            original_review_df,
            auto_research_result["result_df"],
            expert_labels=expert_labels,
        )
        progress_bar.progress(1.0)
        progress_text.caption(
            f"自学习训练完成 | 最优 σ={auto_research_result['best_sigma']:.4f} | 最优权重={auto_research_result['best_weight']}x | 对齐率 {auto_research_result['best_score']:.1%}"
        )

        result_c1, result_c2, result_c3 = st.columns(3)
        result_c1.metric("最优指数 σ", f"{auto_research_result['best_sigma']:.4f}")
        result_c2.metric("最优偏置权重", f"{auto_research_result['best_weight']}x")
        result_c3.metric(
            "专家经验对齐率",
            f"{auto_research_result['best_score']:.1%}",
        )
        st.success("钣金指数自学习训练已完成，新的钣金指数技能书已写入本地数据库。")

        if not auto_research_audit_df.empty:
            with st.expander("查看原始测算与专家校准后结论对比", expanded=True):
                audit_preview_df, audit_visible_columns = prepare_table_view(
                    auto_research_audit_df,
                    "sheet_metal_autoresearch_audit",
                    default_search_columns=["物料编码", "物料描述", "备件简称", "原始结论", "优化后结论", "判定依据"],
                    filter_title="钣金训练结论对比",
                )
                render_standard_data_editor(
                    audit_preview_df[audit_visible_columns],
                    "sheet_metal_autoresearch_audit",
                    max_height=280,
                )

        history_df = pd.DataFrame(auto_research_result.get("history", []))
        if not history_df.empty:
            with st.expander("查看自学习迭代记录", expanded=False):
                history_view_df, history_visible_columns = prepare_table_view(
                    history_df,
                    "sheet_metal_autoresearch_history",
                    default_search_columns=list(history_df.columns[: min(4, len(history_df.columns))]),
                    filter_title="钣金自学习迭代记录",
                )
                render_standard_data_editor(
                    history_view_df[history_visible_columns],
                    "sheet_metal_autoresearch_history",
                    max_height=260,
                )

    st.markdown("---")

    if skills_data and skills_data.get("skills"):
        skills = list(skills_data.get("skills", []))
        st.success(
            f"✅ 当前展示的是数据库中已保存的钣金指数技能书（保存时间：{skills_data.get('saved_at', '未知')}）。"
        )
    else:
        if skills_data is None and skills_snapshot_exists:
            st.warning("⚠️ 钣金指数技能书快照读取异常，当前回退为基于最新测算结果的临时预览。")

        review_df = _cached_detect_sheet_metal_anomalies(
            base_df,
            tuple(sorted(expert_labels.items())),
            bool(expert_labels),
            get_sheet_metal_refresh_token(),
        )
        if review_df.empty:
            st.info("当前钣金件数据暂无可生成技能书的记录。")
            return

        skills = ComputeJob().precompute_sheet_metal_skills(review_df, expert_labels).all_skills
        st.info("当前暂无已保存的钣金指数技能书，以下为基于最新测算结果的临时预览。完成自学习训练后可保存、复用或删除。")

    skills_table = sheet_metal_logic.sheet_metal_skills_to_table(skills)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("备件简称总数", len(skills))
    with c2:
        st.metric("独立钣金标注覆盖简称数", int((skills_table["本组专家标注数"] > 0).sum()) if not skills_table.empty else 0)
    with c3:
        st.metric("当前保存 σ", f"{float((skills_data or {}).get('global_sigma') or 1.0):.4f}")

    st.caption(f"钣金指数分析模型导出路径：{get_path_setting('sheet_metal_model_export_path') or '未配置'}")
    st.caption(f"钣金专家经验报告导出路径：{get_path_setting('sheet_metal_report_export_path') or '未配置'}")

    if skills_data and skills_data.get("skills"):
        delete_c1, delete_c2 = st.columns([2, 2])
        with delete_c1:
            st.caption(f"当前保存权重：{int((skills_data or {}).get('global_weight') or _EXPERT_WEIGHT)}x")
        with delete_c2:
            if st.button("🗑️ 删除已保存钣金指数技能书", width="stretch"):
                st.session_state["_confirm_delete_sheet_metal_skills"] = True
        if st.session_state.get("_confirm_delete_sheet_metal_skills"):
            st.warning("确定要删除当前数据库中的全部钣金指数技能书吗？此操作不可撤销。")
            confirm_c1, confirm_c2 = st.columns(2)
            with confirm_c1:
                if st.button("✅ 确认删除钣金指数技能书", type="primary"):
                    deleted = harness.execute_action("delete_skills_snapshot", domain=SHEET_METAL_SKILL_DOMAIN)
                    st.session_state["_confirm_delete_sheet_metal_skills"] = False
                    bump_sheet_metal_refresh_token()
                    st.success(f"✅ 已删除 {deleted['skills']} 条钣金指数技能书记录，快照 {deleted['snapshots']} 个。")
                    st.rerun()
            with confirm_c2:
                if st.button("取消删除钣金指数技能书"):
                    st.session_state["_confirm_delete_sheet_metal_skills"] = False
                    st.rerun()

    if not skills_table.empty:
        preview_df, visible_columns = prepare_table_view(
            skills_table,
            "sheet_metal_skills_table",
            default_search_columns=["备件简称"],
            filter_title="钣金件指数技能书",
        )
        render_standard_data_editor(preview_df[visible_columns], "sheet_metal_skills_table", max_height=360)

    with st.expander("预览 Markdown 格式钣金指数技能书", expanded=False):
        st.markdown(sheet_metal_logic.sheet_metal_skills_to_markdown(skills))

    export_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    skills_json_name = f"钣金指数技能书_{export_stamp}.json"
    skills_markdown_name = f"钣金指数技能书_{export_stamp}.md"
    skills_excel_name = f"钣金指数技能书_{export_stamp}.xlsx"
    report_markdown_name = f"钣金专家经验报告_{export_stamp}.md"
    report_excel_name = f"钣金专家经验报告_{export_stamp}.xlsx"

    skills_json_bytes = sheet_metal_logic.sheet_metal_skills_to_json_bytes(skills)
    skills_markdown_bytes = sheet_metal_logic.sheet_metal_skills_to_markdown(skills).encode("utf-8")

    st.markdown("### 📦 技能书导出与报告沉淀")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        if st.download_button(
            "📥 下载钣金指数技能书（JSON）",
            data=skills_json_bytes,
            file_name=skills_json_name,
            mime="application/json",
            width="stretch",
        ):
            saved_paths, error_message = _save_artifacts_to_directory(
                get_path_setting("sheet_metal_model_export_path"),
                [(skills_json_name, skills_json_bytes)],
            )
            if error_message:
                st.warning(error_message)
            elif saved_paths:
                st.success(f"技能书文件已自动保存至：{saved_paths[0]}")
    with dl2:
        if st.download_button(
            "📥 下载钣金指数技能书（Markdown）",
            data=skills_markdown_bytes,
            file_name=skills_markdown_name,
            mime="text/markdown",
            width="stretch",
        ):
            saved_paths, error_message = _save_artifacts_to_directory(
                get_path_setting("sheet_metal_model_export_path"),
                [(skills_markdown_name, skills_markdown_bytes)],
            )
            if error_message:
                st.warning(error_message)
            elif saved_paths:
                st.success(f"技能书文件已自动保存至：{saved_paths[0]}")
    with dl3:
        render_deferred_download_button(
            label="📥 下载钣金指数技能书（Excel）",
            prepare_label="准备钣金指数技能书 Excel",
            data_builder=lambda skills_payload=list(skills): sheet_metal_logic.sheet_metal_skills_to_excel_bytes(skills_payload),
            file_name=skills_excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sheet_metal_skills_excel_export",
            fingerprint=_sheet_metal_skills_fingerprint(skills),
            width="stretch",
        )

    report_c1, report_c2 = st.columns([2, 1])
    with report_c1:
        st.caption("点击“生成钣金专家经验报告”后，系统会将 Markdown 报告和 Excel 对照表同时写入已配置的钣金专家经验报告导出路径。")
    with report_c2:
        if st.button("🧾 生成钣金专家经验报告", width="stretch"):
            report_excel_bytes = sheet_metal_logic.sheet_metal_skills_to_excel_bytes(skills)
            report_saved_paths, error_message = _save_artifacts_to_directory(
                get_path_setting("sheet_metal_report_export_path"),
                [
                    (report_markdown_name, skills_markdown_bytes),
                    (report_excel_name, report_excel_bytes),
                ],
            )
            if error_message:
                st.warning(error_message)
            elif report_saved_paths:
                st.success("钣金专家经验报告已生成并保存至：\n" + "\n".join(report_saved_paths))
