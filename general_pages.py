import os
from html import escape
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

import harness
from app_context import (
    NO_DATA_WARNING,
    apply_session_path_updates,
    cached_pivot_report,
    cached_subpart_analysis,
    cached_vehicle_compare,
    clear_loaded_data_state,
    clear_session_path_settings,
    ensure_selectbox_state,
    format_bytes,
    get_cost_refresh_token,
    get_path_setting,
    inject_css,
    load_folder_data_into_session,
    render_bootstrap_status,
    require_price_col,
    reset_and_reload_saved_path_data_into_session,
    reset_search_callback,
    reset_session_key,
)
from config import PATH_SETTING_LABELS, settings
from data_ingestion import (
    BUILTIN_IMPORT_TEMPLATE_NAMES,
    build_default_vehicle_rank,
    build_builtin_template_excel_bytes,
    build_manual_vehicle_rank_from_display,
    build_manual_vehicle_market_price_rows_from_display,
    build_vehicle_candidate_display_df,
    build_vehicle_market_price_display_df,
    extract_missing_vehicle_market_price_series,
    extract_vehicle_rank_candidates,
    filter_latest_cost_increase_rows,
    filter_report_df,
    get_material_metrics,
    paginate_by_material,
    prioritize_latest_cost_increases,
    to_excel_bytes,
)
from page_ui_helpers import (
    dataframe_export_fingerprint,
    get_vehicle_market_price_manual_editable_columns,
    inject_center_aligned_table_css,
    prepare_table_view,
    render_deferred_download_button,
    render_standard_data_editor,
)
from ui_utils import BASE_COLS, render_merged_html_table


def build_path_management_groups() -> list[dict]:
    return [
        {
            "title": "导入路径",
            "expanded": True,
            "caption": "此分组包含原始成本数据、钣金件基础数据、一级件明细数据三个独立目录。",
            "columns": 3,
            "fields": [
                {
                    "setting_key": "input_data_path",
                    "label": "原始数据存放路径",
                    "session_key": "settings_input_data_path",
                    "placeholder": "例如：D:/成本监控/原始数据",
                    "help": "系统将自动扫描该目录下的 Excel 和 CSV 文件，并在启动时优先加载。",
                },
                {
                    "setting_key": "sheet_metal_base_info_path",
                    "label": "钣金件基础数据路径",
                    "session_key": "settings_sheet_metal_base_info_path",
                    "placeholder": "例如：D:/成本监控/钣金基础数据",
                    "help": "钣金件模块会在进入对应页面时静态扫描该目录下的 Excel 文件，不会影响成本主库。",
                },
                {
                    "setting_key": "assembly_data_path",
                    "label": "一级件明细数据路径",
                    "session_key": "settings_assembly_data_path",
                    "placeholder": "例如：D:/成本监控/一级件明细",
                    "help": "拆分件成本监控模块会按需扫描该目录下的 Excel 文件，并结合本地 SQLite 自动补齐缺失成本。",
                },
            ],
        },
        {
            "title": "备件成本导出路径",
            "expanded": True,
            "caption": "此分组包含成本分析模型导出与专家经验报告导出两个独立目录。",
            "columns": 2,
            "fields": [
                {
                    "setting_key": "quantitative_skills_path",
                    "label": "成本分析模型导出路径",
                    "session_key": "settings_quantitative_skills_path",
                    "placeholder": "例如：D:/成本监控/模型导出",
                    "help": "系统将在此目录保存成本分析模型结果、参数快照和相关导出文件。",
                },
                {
                    "setting_key": "qualitative_skills_path",
                    "label": "专家经验报告导出路径",
                    "session_key": "settings_qualitative_skills_path",
                    "placeholder": "例如：D:/成本监控/专家报告",
                    "help": "系统将在此目录保存专家经验总结、知识蒸馏结果和相关报告。",
                },
            ],
        },
        {
            "title": "钣金模块导出路径",
            "expanded": False,
            "caption": "此分组包含钣金模型导出与钣金专家报告导出两个独立目录，用于保持设置页主路径项整齐展示。",
            "columns": 2,
            "fields": [
                {
                    "setting_key": "sheet_metal_model_export_path",
                    "label": "钣金指数分析模型导出路径",
                    "session_key": "settings_sheet_metal_model_export_path",
                    "placeholder": "例如：D:/成本监控/钣金模型导出",
                    "help": "系统将在此目录自动保存钣金指数技能书、参数快照和相关模型导出文件。",
                },
                {
                    "setting_key": "sheet_metal_report_export_path",
                    "label": "钣金专家经验报告导出路径",
                    "session_key": "settings_sheet_metal_report_export_path",
                    "placeholder": "例如：D:/成本监控/钣金专家报告",
                    "help": "系统将在此目录自动保存钣金专家经验报告、结论对比表和相关说明文件。",
                },
            ],
        },
    ]


def build_database_health_warning(
    health: dict,
    *,
    max_freelist_ratio: float = 0.25,
    max_wal_mb: float = 64.0,
) -> str:
    if not isinstance(health, dict) or health.get("ok") is True:
        return ""

    warnings = []
    freelist_ratio = float(health.get("freelist_ratio", 0.0) or 0.0)
    wal_mb = float(health.get("wal_mb", 0.0) or 0.0)
    if freelist_ratio > max_freelist_ratio:
        warnings.append(f"空闲页比例 {freelist_ratio:.1%}，建议在空闲时执行数据库压缩")
    if wal_mb > max_wal_mb:
        warnings.append(f"WAL 文件 {wal_mb:.2f} MB，建议先执行 checkpoint 或压缩")
    return "；".join(warnings)


def apply_vehicle_market_price_auto_rank(
    session_state,
    source_df: pd.DataFrame,
    market_prices_df: pd.DataFrame | None,
    *,
    price_col: str,
) -> list[str]:
    auto_rank = build_default_vehicle_rank(source_df, market_prices_df, price_col=price_col)
    session_state["vehicle_rank_text"] = "\n".join(auto_rank)
    session_state["vehicle_rank"] = auto_rank
    session_state["vehicle_rank_manual_order"] = []
    return auto_rank


def _render_path_management_groups() -> dict[str, str]:
    path_values = {}
    for group in build_path_management_groups():
        with st.expander(str(group["title"]), expanded=bool(group.get("expanded", False))):
            caption = str(group.get("caption", "") or "").strip()
            if caption:
                st.caption(caption)
            fields = list(group.get("fields", []))
            column_count = max(1, min(int(group.get("columns", 2) or 2), len(fields) or 1))
            columns = st.columns(column_count)
            for index, field in enumerate(fields):
                with columns[index % column_count]:
                    setting_key = str(field["setting_key"])
                    path_values[setting_key] = st.text_input(
                        str(field["label"]),
                        value=get_path_setting(setting_key),
                        key=str(field["session_key"]),
                        placeholder=str(field.get("placeholder", "") or ""),
                        help=str(field.get("help", "") or ""),
                    )
    return path_values


def render_overview_page() -> None:
    inject_css(is_overview=True)

    def _perform_overview_reset() -> bool:
        with st.spinner("正在重置并重新读取本地路径数据..."):
            success, message = reset_and_reload_saved_path_data_into_session()
        st.session_state["_startup_bootstrap_status"] = {
            "kind": "success" if success else "error",
            "message": message,
        }
        return success

    if str(st.query_params.get("overview_reset", "") or "") == "1":
        try:
            del st.query_params["overview_reset"]
        except Exception:
            pass
        if _perform_overview_reset():
            st.rerun()

    if st.session_state.data is not None:
        count_display = st.session_state.data["物料编码"].nunique()
        subtitle_text = "个备件的成本变动"
    else:
        count_display = "-"
        subtitle_text = "等待加载数据..."

    st.markdown(
        f"""
        <div class="overview-title">与您一起守护了</div>
        <div class="overview-metric">{count_display}</div>
        <div class="overview-subtitle">{subtitle_text}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🛠️ 本地数据源状态")
    render_bootstrap_status()

    try:
        local_db_status = harness.execute_action("get_core_cost_records_status")
    except Exception as exc:
        local_db_status = {"row_count": 0, "updated_at": None, "price_col": None, "error": str(exc)}

    st.markdown(
        """
        <style>
            .overview-db-status-card {
                height: 134px;
                width: 100%;
                box-sizing: border-box;
                padding: 22px 18px;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: #f8f9fa;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 12px;
            }
            .overview-db-status-label {
                color: #2c3e50;
                font-size: 14px;
                line-height: 1.2;
                white-space: nowrap;
            }
            .overview-db-status-value {
                color: #2c3e50;
                font-size: 34px;
                line-height: 1.05;
                font-weight: 400;
                letter-spacing: 0;
                white-space: nowrap;
            }
            .overview-db-status-value.compact {
                font-size: 21px;
            }
            div.st-key-overview_reset_button {
                height: 134px;
            }
            div.st-key-overview_reset_button button {
                height: 134px;
                min-height: 134px;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: #f8f9fa;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                color: #2c3e50;
                font-weight: 700;
                white-space: normal;
            }
            div.st-key-overview_reset_button button:hover {
                border-color: #d0d7de;
                background: #f3f6f8;
                color: #2c3e50;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _render_overview_db_card(label: str, value: str, *, compact: bool = False) -> None:
        value_class = "overview-db-status-value compact" if compact else "overview-db-status-value"
        st.markdown(
            f"""
            <div class="overview-db-status-card">
                <div class="overview-db-status-label">{escape(label)}</div>
                <div class="{value_class}">{escape(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    db_c1, db_c2, db_c3 = st.columns(3)
    updated_at = local_db_status.get("updated_at")
    updated_at_text = updated_at.strftime("%Y-%m-%d %H:%M") if updated_at else "暂无"
    with db_c1:
        _render_overview_db_card("本地数据库记录数", f"{local_db_status.get('row_count', 0)}")
    with db_c2:
        _render_overview_db_card("最近写入时间", updated_at_text, compact=True)
    with db_c3:
        if st.button(
            "🔄 重置并更新本地数据",
            key="overview_reset_button",
            use_container_width=True,
            help="按已配置的原始数据存放路径重新读取文件，并覆盖本地核心成本表。",
        ):
            if _perform_overview_reset():
                st.rerun()


def render_settings_page() -> None:
    inject_css(is_overview=False)
    st.title("⚙️ 系统设置")
    settings.reload()
    render_bootstrap_status()

    st.markdown("### 📁 路径管理")
    st.caption("原始数据存放路径用于加载成本源数据；一级件明细数据路径仅供拆分件审计模块读取；钣金件基础数据路径仅供钣金模块静态读取，不会写入成本主库；各模块路径彼此独立。")

    path_group_values = _render_path_management_groups()
    saved_input_data_path = path_group_values.get("input_data_path", "")
    saved_sheet_metal_base_info_path = path_group_values.get("sheet_metal_base_info_path", "")
    saved_assembly_data_path = path_group_values.get("assembly_data_path", "")
    saved_quantitative_skills_path = path_group_values.get("quantitative_skills_path", "")
    saved_qualitative_skills_path = path_group_values.get("qualitative_skills_path", "")
    saved_sheet_metal_model_export_path = path_group_values.get("sheet_metal_model_export_path", "")
    saved_sheet_metal_report_export_path = path_group_values.get("sheet_metal_report_export_path", "")

    path_c1, path_c2 = st.columns(2)
    with path_c1:
        if st.button("保存并应用路径配置", type="primary", width="stretch"):
            path_updates = {
                "input_data_path": saved_input_data_path.strip(),
                "assembly_data_path": saved_assembly_data_path.strip(),
                "sheet_metal_base_info_path": saved_sheet_metal_base_info_path.strip(),
                "quantitative_skills_path": saved_quantitative_skills_path.strip(),
                "qualitative_skills_path": saved_qualitative_skills_path.strip(),
                "sheet_metal_model_export_path": saved_sheet_metal_model_export_path.strip(),
                "sheet_metal_report_export_path": saved_sheet_metal_report_export_path.strip(),
            }
            invalid_paths = [
                f"{PATH_SETTING_LABELS[path_key]}: {path_updates[path_key]}"
                for path_key in path_updates
                if path_updates[path_key] and not os.path.isdir(path_updates[path_key])
            ]
            if invalid_paths:
                st.error("以下目录不存在：\n" + "\n".join(invalid_paths))
            else:
                harness.execute_action("save_path_settings", path_updates=path_updates)
                apply_session_path_updates(path_updates)
                harness.execute_action("ensure_session_paths", session_state=st.session_state)
                st.session_state["_startup_bootstrap_complete"] = False
                st.toast("路径配置已保存并应用")

                if path_updates["input_data_path"]:
                    with st.spinner("正在校验原始数据目录并写入本地数据库..."):
                        success, message = load_folder_data_into_session(
                            path_updates["input_data_path"],
                            origin="settings_path",
                        )
                    st.session_state["_startup_bootstrap_status"] = {
                        "kind": "success" if success else "error",
                        "message": message,
                    }
                    if success:
                        st.success(f"路径配置已保存到本地配置文件。{message}")
                    else:
                        st.error(message)
                else:
                    st.session_state["_startup_bootstrap_status"] = {
                        "kind": "info",
                        "message": "路径配置已保存到本地配置文件；当前未填写原始数据存放路径。",
                    }
                    st.success("路径配置已保存到本地配置文件")
                    if path_updates["assembly_data_path"]:
                        st.info("一级件明细数据路径已保存，拆分件成本监控模块将在进入对应页面时按需加载。")
                    if path_updates["sheet_metal_base_info_path"]:
                        st.info("钣金件基础数据路径已保存，钣金模块将在进入对应页面时按需静态加载。")

    with path_c2:
        if st.button("清空全部路径配置", width="stretch"):
            harness.execute_action(
                "save_path_settings",
                path_updates={
                    "input_data_path": "",
                    "assembly_data_path": "",
                    "sheet_metal_base_info_path": "",
                    "quantitative_skills_path": "",
                    "qualitative_skills_path": "",
                    "sheet_metal_model_export_path": "",
                    "sheet_metal_report_export_path": "",
                },
            )
            clear_session_path_settings()
            harness.execute_action("ensure_session_paths", session_state=st.session_state)
            st.session_state["_startup_bootstrap_complete"] = False
            st.session_state["_startup_bootstrap_status"] = {
                "kind": "info",
                "message": "已清空本地配置文件中保存的全部路径配置",
            }
            st.info("已清空本地配置文件中保存的全部路径配置")

    db_status = harness.execute_action("get_core_cost_records_status")
    db_file_size = settings.db_path.stat().st_size if settings.db_path.exists() else 0
    path_info_c1, path_info_c2, path_info_c3, path_info_c4 = st.columns(4)
    path_info_c1.metric("当前本地数据库记录数", f"{db_status.get('row_count', 0)}")
    path_info_c2.metric(
        "最近本地写入",
        db_status["updated_at"].strftime("%Y-%m-%d %H:%M") if db_status.get("updated_at") else "暂无",
    )
    path_info_c3.metric("本地数据库状态", "已启用" if settings.db_path.exists() else "未创建")
    path_info_c4.metric("数据库体积", format_bytes(db_file_size))

    if st.button("释放当前内存数据", width="stretch"):
        release_result = clear_loaded_data_state(st.session_state)
        st.success(
            f"已释放当前会话数据 {release_result['released_rows']} 行，"
            f"并清理 {release_result['cleared_download_keys']} 个导出缓存项。"
        )

    try:
        db_health = harness.execute_action("get_local_database_health")
    except Exception:
        db_health = {}
    db_health_warning = build_database_health_warning(db_health)
    if db_health_warning:
        st.warning(db_health_warning)

    if st.button("🧹 压缩并整理数据库", width="stretch"):
        with st.spinner("正在压缩并整理本地数据库..."):
            vacuum_result = harness.execute_action("compact_local_database")
        st.success(
            "数据库压缩完成："
            f"{format_bytes(vacuum_result['before_bytes'])} → {format_bytes(vacuum_result['after_bytes'])}，"
            f"释放 {format_bytes(vacuum_result['saved_bytes'])}"
        )

    st.markdown("---")
    st.markdown("### 📄 内置导入模板")
    st.caption("下载后直接按列填数再导入，字段与当前导入校验规则保持一致。")
    template_columns = st.columns(3)
    for column, template_name in zip(template_columns, ["cost", "assembly", "sheet_metal"]):
        with column:
            label = BUILTIN_IMPORT_TEMPLATE_NAMES[template_name]
            st.download_button(
                f"下载{label}",
                data=build_builtin_template_excel_bytes(template_name),
                file_name=f"{label}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    st.markdown("---")
    st.markdown("### 🤖 大模型联网配置")
    st.caption("LLM 已改为本地 `.env` 固定配置，页面不再提供手动密钥输入。")
    llm_status_col1, llm_status_col2, llm_status_col3 = st.columns(3)
    llm_configs = getattr(settings, "llm_api_configs", []) or []
    llm_status_col1.metric("配置来源", ".env" if settings.llm_config_source == "env" else "未检测到 .env")
    llm_status_col2.metric("模型", settings.llm_api_model or "未配置")
    llm_status_col3.metric("可用配置数", f"{len(llm_configs)} 组" if llm_configs else "未配置")
    if settings.llm_env_configured:
        fallback_names = [str(item.get("name") or item.get("model") or "LLM") for item in llm_configs[1:]]
        fallback_text = f"；备用：{'、'.join(fallback_names)}" if fallback_names else ""
        st.success(f"当前 LLM 已由本地环境文件固定配置：主模型 {settings.llm_api_model}{fallback_text}")
        if settings.llm_api_direct_url:
            st.info("主模型已启用直连模式：配置地址以 `/*` 结尾时会按接口文档解析到 `/chat/completions`；备用模型按各自配置自动处理。")
    else:
        st.warning(f"未检测到完整 `.env` 配置，知识蒸馏功能将禁用。环境文件路径：{settings.llm_env_file_path}")


def _format_optional_float(value, *, decimals: int = 2, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)
    sign_prefix = "+" if signed and number > 0 else ""
    return f"{sign_prefix}{number:,.{decimals}f}"


def _build_vehicle_gradient_deviation_context_rows(compare_df: pd.DataFrame) -> list[dict]:
    if compare_df.empty or "梯度偏差异常" not in compare_df.columns:
        return []

    data = compare_df.copy().reset_index(drop=True)
    data["_latest_cost_numeric"] = pd.to_numeric(data.get("最新成本"), errors="coerce")
    data["_cost_rank"] = data["_latest_cost_numeric"].rank(method="first", ascending=False)
    context_rows: list[dict] = []
    for row_index, row in data.iterrows():
        if not bool(row.get("梯度偏差异常")):
            continue
        gradient_rank = pd.to_numeric(pd.Series([row.get("梯度排名")]), errors="coerce").iloc[0]
        cost_rank = pd.to_numeric(pd.Series([row.get("_cost_rank")]), errors="coerce").iloc[0]
        deviation_rate = None
        if pd.notna(gradient_rank) and float(gradient_rank) > 0 and pd.notna(cost_rank):
            deviation_rate = abs(float(cost_rank) - float(gradient_rank)) / float(gradient_rank)
        context_rows.append(
            {
                "row_id": str(row_index),
                "vehicle_series": str(row.get("适用车系") or "").strip(),
                "part_name": str(row.get("备件简称") or "").strip(),
                "gradient_rank": int(gradient_rank) if pd.notna(gradient_rank) else None,
                "cost_rank": int(cost_rank) if pd.notna(cost_rank) else None,
                "deviation_rate": deviation_rate,
                "is_abnormal": True,
            }
        )
    return context_rows


def _append_vehicle_gradient_deviation_explanations(compare_df: pd.DataFrame) -> pd.DataFrame:
    result = compare_df.copy().reset_index(drop=True)
    result["梯度偏差解释"] = ""
    context_rows = _build_vehicle_gradient_deviation_context_rows(result)
    if not context_rows:
        return result

    fingerprint_columns = [
        column_name
        for column_name in ["梯度排名", "梯度偏差异常", "适用车系", "备件简称", "最新成本", "最新成本有效期"]
        if column_name in result.columns
    ]
    fingerprint = dataframe_export_fingerprint(result[fingerprint_columns], columns=fingerprint_columns)
    cache_key = f"vehicle_gradient_deviation_explanations::{fingerprint}"
    if st.session_state.get("_vehicle_gradient_deviation_explanation_key") != cache_key:
        with st.spinner("正在补全梯度偏差解释..."):
            explanations = harness.execute_action("explain_vehicle_gradient_deviations", rows=context_rows)
        st.session_state["_vehicle_gradient_deviation_explanation_key"] = cache_key
        st.session_state["_vehicle_gradient_deviation_explanations"] = explanations

    explanations = st.session_state.get("_vehicle_gradient_deviation_explanations") or {}
    for context_row in context_rows:
        row_id = str(context_row["row_id"])
        if row_id in explanations:
            result.loc[int(row_id), "梯度偏差解释"] = explanations[row_id]
    return result


def render_single_material_page() -> None:
    inject_css(is_overview=False)
    st.markdown(
        """
        <style>
        .single-material-metrics [data-testid="stMetric"] {
            min-height: 134px !important;
            height: 134px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }
        .single-material-metrics [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            min-height: 1.35rem !important;
        }
        .single-material-metrics [data-testid="stMetric"] [data-testid="stMetricValue"] {
            line-height: 1.1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("📈 单个物料成本监控")

    if st.session_state.data is None:
        st.warning(NO_DATA_WARNING)
        return

    df = st.session_state.data
    price_col = require_price_col(df)
    items = sorted(df["物料编码"].astype(str).unique())
    if not items:
        st.info("当前数据中没有可选择的物料编码")
        return
    ensure_selectbox_state("single_material_code", items, items[0] if items else None)
    item_col, reset_col = st.columns([5, 1])
    with item_col:
        selected_item = st.selectbox("🔍 搜索/选择物料编码", items, key="single_material_code")
    with reset_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button(
            "重置",
            key="reset_single_material_code",
            width="stretch",
            on_click=reset_session_key,
            args=("single_material_code", items[0] if items else ""),
        )

    item_data = df[df["物料编码"].astype(str) == selected_item].sort_values("monitor_date")
    if item_data.empty:
        st.info("该物料暂无有效数据")
        return

    metrics = get_material_metrics(item_data, price_col)
    st.markdown("### 📊 核心指标")
    st.markdown('<div class="single-material-metrics">', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    latest_factory = str(metrics.get("latest_factory") or "未知工厂")
    m1.metric(f"最新成本（{latest_factory}）", f" {metrics['latest_price']:,.2f}")
    m2.metric("历史最低", f" {metrics['min_price']:,.2f}")
    m3.metric("历史最高", f" {metrics['max_price']:,.2f}")
    m4.metric("包运费系数", _format_optional_float(metrics.get("freight_factor"), decimals=4))
    cost_drop_factory = str(metrics.get("cost_drop_factory") or latest_factory)
    reference_year_end = metrics.get("cost_drop_reference_year_end")
    reference_suffix = ""
    if reference_year_end is not None and not pd.isna(reference_year_end):
        reference_suffix = f"较{pd.Timestamp(reference_year_end).year}年末"
    m5.metric(
        f"成本变动（{cost_drop_factory}）",
        _format_optional_float(metrics.get("cost_drop_amount"), signed=True),
        delta=reference_suffix or None,
        delta_color="normal",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    fig = px.line(
        item_data,
        x="monitor_date",
        y=price_col,
        color="工厂",
        title=f"📈 物料 {selected_item} 多工厂成本走势对比",
        markers=True,
        hover_data={"工厂": True, "monitor_date": "|%Y-%m-%d", price_col: ":.2f"},
    )
    fig.update_layout(
        xaxis_title="日期",
        yaxis_title="价格 (CNY)",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="工厂",
    )
    st.plotly_chart(fig, width="stretch")


def render_report_page(page_name: str) -> None:
    inject_css(is_overview=False)
    st.title(f"📑 {page_name}")

    if st.session_state.data is None:
        st.warning(NO_DATA_WARNING)
        return

    df = st.session_state.data
    price_col = require_price_col(df)

    with st.spinner(f"正在生成{page_name}..."):
        report_df = cached_pivot_report(df, price_col, get_cost_refresh_token())
    report_df = prioritize_latest_cost_increases(report_df)

    st.markdown("#### 🔍 筛选条件与导出")
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
    with c1:
        st.text_input("搜索物料编码 (支持空格分隔多值)", key="search_code")
    with c2:
        st.text_input("搜索备件简称", key="search_name")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("🔄 重置检索", width="stretch", on_click=reset_search_callback)

    current_search_hash = f"{page_name}_{st.session_state.search_code}_{st.session_state.search_name}"
    if st.session_state.last_search_hash != current_search_hash:
        st.session_state.report_page_number = 1
        st.session_state.last_search_hash = current_search_hash

    filtered_df = filter_report_df(
        report_df,
        st.session_state.search_code,
        st.session_state.search_name,
    )
    filtered_df, visible_columns = prepare_table_view(
        filtered_df,
        f"{page_name}_report_table",
        default_search_columns=["物料编码", "物料名称", "备件简称", "适用车系", "工厂"],
        locked_columns=[column_name for column_name in BASE_COLS if column_name in filtered_df.columns],
        filter_title=page_name,
    )

    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        report_export_df = filtered_df[visible_columns].copy()
        render_deferred_download_button(
            label="📥 下载报表",
            prepare_label="准备导出报表",
            data_builder=lambda export_df=report_export_df: to_excel_bytes(export_df),
            file_name=f"{page_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{page_name}_report_export",
            fingerprint=dataframe_export_fingerprint(report_export_df),
            width="stretch",
        )
    with c5:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        increase_export_df = filter_latest_cost_increase_rows(filtered_df)
        render_deferred_download_button(
            label="📥 下载上涨明细",
            prepare_label="准备上涨明细",
            data_builder=lambda export_df=increase_export_df: to_excel_bytes(export_df, sheet_name="最新上涨明细"),
            file_name=f"{page_name}_最新成本上涨明细_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{page_name}_latest_increase_export",
            fingerprint=dataframe_export_fingerprint(increase_export_df),
            width="stretch",
        )

    page_info = paginate_by_material(filtered_df, st.session_state.report_page_number, 50)
    st.session_state.report_page_number = page_info["page_number"]
    page_data = page_info["page_df"][visible_columns]
    value_columns = [column_name for column_name in visible_columns if column_name not in BASE_COLS]

    st.markdown(f"**共找到 {len(filtered_df)} 条匹配记录**")
    st.markdown(
        render_merged_html_table(
            page_data,
            value_columns,
            is_trend_mode=False,
            preserve_order=True,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    p1, p2, p3 = st.columns([1, 1, 3])
    with p1:
        if st.button("⬅️ 上一页", disabled=st.session_state.report_page_number <= 1):
            st.session_state.report_page_number -= 1
            st.rerun()
    with p2:
        if st.button("下一页 ➡️", disabled=st.session_state.report_page_number >= page_info["total_pages"]):
            st.session_state.report_page_number += 1
            st.rerun()
    with p3:
        st.markdown(
            f"<div style='line-height:2.5;text-align:center;color:#666;'>当前第 {st.session_state.report_page_number} 页 / 共 {page_info['total_pages']} 页</div>",
            unsafe_allow_html=True,
        )


def render_vehicle_rank_config_page() -> None:
    inject_css(is_overview=False)
    inject_center_aligned_table_css()
    st.title("📁 车系梯度配置")
    st.markdown("系统从本地表格读取车系列表，LLM 估算次顶配价格后生成梯度；本地读取顺序不作为梯度结论。")
    rank_notice = st.session_state.pop("vehicle_rank_notice", None)
    if rank_notice:
        st.success(rank_notice)

    saved_rank_df = harness.execute_action("load_vehicle_rank_config")
    market_prices_df = harness.execute_action("load_vehicle_market_prices")
    vehicle_candidates: list[str] = []
    if st.session_state.data is not None:
        vehicle_candidates = extract_vehicle_rank_candidates(st.session_state.data)

    if not st.session_state.get("vehicle_rank") and saved_rank_df is not None and not saved_rank_df.empty:
        persisted_rank = saved_rank_df.sort_values("rank_order")["vehicle_series"].astype(str).tolist()
        st.session_state.vehicle_rank = persisted_rank
        st.session_state.vehicle_rank_text = "\n".join(persisted_rank)

    if st.session_state.data is None:
        st.info("加载全量成本数据后，可自动抽取适用车系并生成默认梯度。")
    else:
        estimated_price_count = 0
        if market_prices_df is not None and not market_prices_df.empty and "market_price" in market_prices_df.columns:
            local_price_df = build_vehicle_market_price_display_df(
                market_prices_df,
                vehicle_candidates=vehicle_candidates,
            )
            estimated_price_count = int(local_price_df["估算价格（元）"].notna().sum())
        st.caption(f"已从全量成本识别 {len(vehicle_candidates)} 个车系；已有估算价格 {estimated_price_count} 条。")

    st.markdown("#### 本地表格车系列表")
    candidate_df = build_vehicle_candidate_display_df(vehicle_candidates)
    if candidate_df.empty:
        st.info("暂无可展示车系。")
    else:
        render_standard_data_editor(
            candidate_df,
            "vehicle_candidate_source_list",
            max_height=360,
        )

    st.markdown("#### LLM估算价格结果")
    action_display_price_df = (
        build_vehicle_market_price_display_df(market_prices_df, vehicle_candidates=vehicle_candidates)
        if market_prices_df is not None and not market_prices_df.empty
        else pd.DataFrame(columns=["梯度排名", "车系", "次顶配车型", "估算价格（元）"])
    )
    missing_price_series = extract_missing_vehicle_market_price_series(action_display_price_df)
    action_col1, action_col2, action_col3, action_col4 = st.columns([1.4, 1.4, 1.2, 2.6], vertical_alignment="center")
    with action_col1:
        refresh_clicked = st.button(
            "🤖 LLM估算并排序",
            width="stretch",
            disabled=not vehicle_candidates,
            help="调用本地配置的 LLM，估算列表中所有车系次顶配价格，并要求按价格从高到低输出。",
        )
    with action_col2:
        repair_missing_clicked = st.button(
            "🔎 只重检空价格",
            width="stretch",
            disabled=not missing_price_series,
            help=f"仅重新检索估算价格为空的车系；已有价格的车系不会再次检索。当前空价车系 {len(missing_price_series)} 个。",
        )
    with action_col3:
        manual_clicked = st.button(
            "✋ 人工修正",
            width="stretch",
            disabled=market_prices_df is None or market_prices_df.empty,
            type="secondary" if not st.session_state.get("vehicle_rank_manual_edit_mode") else "primary",
            help="进入可编辑模式，手工填写梯度排名和估算价格。",
        )
    with action_col4:
        if market_prices_df is not None and not market_prices_df.empty:
            latest_fetch = pd.to_datetime(market_prices_df.get("fetched_at"), errors="coerce").max()
            if pd.notna(latest_fetch):
                st.caption(f"估算价格最近刷新：{latest_fetch.strftime('%Y-%m-%d %H:%M')}")
            if missing_price_series:
                st.caption(f"仍有 {len(missing_price_series)} 个车系缺少估算价格，可只重检空值。")

    if manual_clicked:
        st.session_state.vehicle_rank_manual_edit_mode = not st.session_state.get("vehicle_rank_manual_edit_mode", False)
        st.rerun()

    if refresh_clicked:
        with st.spinner("正在调用 LLM 估算次顶配价格并生成价格梯度，请稍候..."):
            price_rows = harness.execute_action("fetch_vehicle_market_prices", vehicle_series=vehicle_candidates)
            saved_count = harness.execute_action("save_vehicle_market_prices", price_rows=price_rows)
            market_prices_df = harness.execute_action("load_vehicle_market_prices")
            apply_vehicle_market_price_auto_rank(
                st.session_state,
                st.session_state.data,
                market_prices_df,
                price_col=require_price_col(st.session_state.data),
            )
            st.session_state.vehicle_rank_manual_edit_mode = False
        st.success(f"已刷新并保存 {saved_count} 条估算价格结果；无法估算的车系会排在末尾。")
        st.rerun()

    if repair_missing_clicked:
        with st.spinner("正在只重检估算价格为空的车系，并强制要求 LLM 输出价格..."):
            price_rows = harness.execute_action("fetch_vehicle_market_prices", vehicle_series=missing_price_series)
            saved_count = harness.execute_action("save_vehicle_market_prices", price_rows=price_rows)
            market_prices_df = harness.execute_action("load_vehicle_market_prices")
            if st.session_state.data is not None:
                apply_vehicle_market_price_auto_rank(
                    st.session_state,
                    st.session_state.data,
                    market_prices_df,
                    price_col=require_price_col(st.session_state.data),
                )
            st.session_state.vehicle_rank_manual_edit_mode = False
        st.success(f"已只重检 {len(missing_price_series)} 个空价车系并保存 {saved_count} 条结果；已按补全后的价格自动刷新梯度排名。")
        st.rerun()

    edited_price_df = pd.DataFrame(columns=["梯度排名", "车系", "次顶配车型", "估算价格（元）"])
    if market_prices_df is not None and not market_prices_df.empty:
        manual_order = st.session_state.get("vehicle_rank_manual_order") or []
        display_price_df = build_vehicle_market_price_display_df(
            market_prices_df,
            rank_order=manual_order if manual_order else None,
            vehicle_candidates=vehicle_candidates,
        )
        if not manual_order and not st.session_state.get("vehicle_rank_manual_edit_mode"):
            estimated_rank = build_manual_vehicle_rank_from_display(display_price_df)
            st.session_state.vehicle_rank = estimated_rank
            st.session_state.vehicle_rank_text = "\n".join(estimated_rank)
        estimated_rows = int(display_price_df["估算价格（元）"].notna().sum()) if "估算价格（元）" in display_price_df.columns else 0
        if estimated_rows:
            st.caption("默认按 LLM 估算价格从高到低展示；人工修正后，可按填写的梯度排名刷新显示顺序。")
        else:
            st.warning("本次未取得可用估算价格，可稍后重试 LLM 估算，或先手动调整上方车系列表。")
        if st.session_state.get("vehicle_rank_manual_edit_mode"):
            edited_price_df = render_standard_data_editor(
                display_price_df,
                "vehicle_market_price_manual_cache",
                editable_columns=get_vehicle_market_price_manual_editable_columns(),
                column_config={
                    "梯度排名": st.column_config.NumberColumn("梯度排名", min_value=1, step=1, format="%d"),
                    "次顶配车型": st.column_config.TextColumn("次顶配车型", width="medium"),
                    "估算价格（元）": st.column_config.NumberColumn("估算价格（元）", min_value=0, step=1000, format="%d"),
                },
                max_height=360,
            )
            st.caption("修改“梯度排名”、“次顶配车型”或“估算价格（元）”后，点击下方刷新按钮，列表会按人工排名重新排序并保存内容。")
        else:
            edited_price_df = display_price_df
            render_standard_data_editor(
                display_price_df,
                "vehicle_market_price_cache",
                max_height=360,
            )
    else:
        st.info("暂无 LLM 估算结果。点击“LLM估算并排序”后，会在这里显示估算价格和梯度排名。")

    if not edited_price_df.empty:
        price_export_df = edited_price_df.copy()
        render_deferred_download_button(
            label="📥 下载LLM估算结果",
            prepare_label="准备导出LLM估算结果",
            data_builder=lambda export_df=price_export_df: to_excel_bytes(export_df, sheet_name="LLM估算结果"),
            file_name=f"车系LLM估算价格结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="vehicle_market_price_export",
            fingerprint=dataframe_export_fingerprint(price_export_df),
            width="stretch",
        )

    save_col1, save_col2, _ = st.columns([1.5, 1.4, 3], vertical_alignment="center")
    with save_col1:
        apply_manual_clicked = st.button(
            "🔄 按人工排名刷新",
            width="stretch",
            disabled=not st.session_state.get("vehicle_rank_manual_edit_mode") or edited_price_df.empty,
        )
    with save_col2:
        save_clicked = st.button(
            "💾 保存梯度配置",
            type="primary",
            width="stretch",
            disabled=edited_price_df.empty and not st.session_state.get("vehicle_rank"),
        )

    if apply_manual_clicked:
        manual_price_rows = build_manual_vehicle_market_price_rows_from_display(edited_price_df)
        if manual_price_rows:
            harness.execute_action("save_vehicle_market_prices", price_rows=manual_price_rows)
        manual_rank = build_manual_vehicle_rank_from_display(edited_price_df)
        st.session_state.vehicle_rank_manual_order = manual_rank
        st.session_state.vehicle_rank = manual_rank
        st.session_state.vehicle_rank_text = "\n".join(manual_rank)
        st.session_state.vehicle_rank_notice = f"已按人工排名刷新，共 {len(manual_rank)} 个车系。"
        st.rerun()

    if save_clicked:
        manual_price_rows = build_manual_vehicle_market_price_rows_from_display(edited_price_df)
        if manual_price_rows:
            harness.execute_action("save_vehicle_market_prices", price_rows=manual_price_rows)
        rank_list = list(st.session_state.get("vehicle_rank") or [])
        if st.session_state.get("vehicle_rank_manual_edit_mode") and not edited_price_df.empty:
            rank_list = build_manual_vehicle_rank_from_display(edited_price_df)
        elif not rank_list and not edited_price_df.empty:
            rank_list = build_manual_vehicle_rank_from_display(edited_price_df)
        st.session_state.vehicle_rank_text = "\n".join(rank_list)
        st.session_state.vehicle_rank = rank_list
        st.session_state.vehicle_rank_manual_order = rank_list
        harness.execute_action(
            "save_vehicle_rank_config",
            rank_rows=[
                {"vehicle_series": vehicle_name, "rank_order": index, "source": "manual"}
                for index, vehicle_name in enumerate(rank_list, start=1)
            ],
        )
        st.session_state.vehicle_rank_notice = f"配置已生效，已识别 {len(rank_list)} 个梯度车系"
        st.rerun()


def render_vehicle_gradient_compare_page() -> None:
    inject_css(is_overview=False)
    inject_center_aligned_table_css()
    st.title("📊 车系梯度成本对比")

    if st.session_state.data is None:
        st.warning(NO_DATA_WARNING)
        return

    df = st.session_state.data
    price_col = require_price_col(df)
    part_options = sorted(df["备件简称"].astype(str).unique())

    saved_rank_df = harness.execute_action("load_vehicle_rank_config")
    if saved_rank_df is not None and not saved_rank_df.empty:
        st.session_state.vehicle_rank = saved_rank_df.sort_values("rank_order")["vehicle_series"].astype(str).tolist()

    selected_part = st.selectbox("备件简称筛选", part_options)
    compare_df = cached_vehicle_compare(
        df,
        price_col,
        selected_part,
        tuple(st.session_state.vehicle_rank),
        get_cost_refresh_token(),
    )
    compare_df = _append_vehicle_gradient_deviation_explanations(compare_df)
    compare_df, compare_visible_columns = prepare_table_view(
        compare_df,
        "vehicle_compare_table",
        default_search_columns=["适用车系", "备件简称", "最新成本"],
        filter_title="车系梯度成本对比",
    )

    st.markdown(f"**共找到 {len(compare_df)} 条匹配记录**")
    render_standard_data_editor(
        compare_df[compare_visible_columns],
        "vehicle_compare_table",
        column_config={
            "梯度排名": st.column_config.NumberColumn("梯度排名", disabled=True, format="%d"),
            "梯度偏差异常": st.column_config.CheckboxColumn("梯度偏差异常", disabled=True, help="超过 25% 时勾选"),
            "梯度偏差解释": st.column_config.TextColumn("梯度偏差解释", disabled=True, width="large"),
        },
        max_height=500,
    )

    compare_export_df = compare_df[compare_visible_columns].copy()
    render_deferred_download_button(
        label="📥 下载对比结果",
        prepare_label="准备导出对比结果",
        data_builder=lambda export_df=compare_export_df: to_excel_bytes(export_df),
        file_name=f"车系梯度成本对比_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="vehicle_compare_export",
        fingerprint=dataframe_export_fingerprint(compare_export_df),
    )


def render_subpart_cost_page() -> None:
    inject_css(is_overview=False)
    st.title("🔩 拆分件成本监控")

    if st.session_state.data is None:
        st.warning(NO_DATA_WARNING)
        return

    df = st.session_state.data
    price_col = require_price_col(df)

    if "一级总成料号" not in df.columns:
        st.warning("⚠️ 当前数据中缺少「一级总成料号」字段，无法进行拆分件分析。请检查数据源。")
        return

    with st.spinner("正在分析拆分件成本..."):
        subpart_df = cached_subpart_analysis(df, price_col, get_cost_refresh_token())

    if subpart_df.empty:
        st.info("当前数据中没有一级总成料号不为空的记录。")
        return

    total_assy = len(subpart_df)
    abnormal_count = int((subpart_df["结论状态"] == "异常").sum())
    normal_count = int((subpart_df["结论状态"] == "正常").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("总成总数", f"{total_assy}")
    m2.metric("异常", f"{abnormal_count}")
    m3.metric("正常", f"{normal_count}")

    show_mode = st.radio("显示范围", options=["仅异常", "全部"], horizontal=True, key="subpart_show_mode")

    display_df = subpart_df.copy()
    if show_mode == "仅异常":
        display_df = display_df[display_df["结论状态"] == "异常"].copy()

    fc1, fc2 = st.columns(2)
    with fc1:
        filter_assy = st.text_input("筛选 一级总成料号", key="subpart_filter_assy", placeholder="输入关键字筛选...")
    with fc2:
        filter_desc = st.text_input("筛选 一级总成品名描述", key="subpart_filter_desc", placeholder="输入关键字筛选...")

    if filter_assy:
        display_df = display_df[display_df["一级总成料号"].astype(str).str.contains(filter_assy, case=False, na=False)]
    if filter_desc:
        display_df = display_df[display_df["一级总成品名描述"].astype(str).str.contains(filter_desc, case=False, na=False)]

    st.markdown(f"**共 {len(display_df)} 条记录**")

    ordered_cols = [
        "一级总成料号",
        "一级总成品名描述",
        "一级总成成本",
        "子零件数量",
        "子零件加权总和",
        "测算总成成本",
        "测算比值",
        "结论状态",
    ]
    extra_cols = [column_name for column_name in display_df.columns if column_name not in ordered_cols]
    final_cols = [column_name for column_name in ordered_cols if column_name in display_df.columns] + extra_cols
    display_df = display_df[final_cols]

    if "测算比值" in display_df.columns:
        display_df = display_df.copy()
        display_df["测算比值"] = display_df["测算比值"].apply(
            lambda value: f"{value:.4%}" if isinstance(value, (int, float)) and value == value else ""
        )

    column_config = {
        "一级总成料号": st.column_config.TextColumn("一级总成料号"),
        "一级总成品名描述": st.column_config.TextColumn("一级总成品名描述"),
        "一级总成成本": st.column_config.NumberColumn("一级总成成本", format="%.2f"),
        "子零件数量": st.column_config.NumberColumn("子零件数量"),
        "子零件加权总和": st.column_config.NumberColumn("子零件加权总和", format="%.2f"),
        "测算总成成本": st.column_config.NumberColumn("测算总成成本", format="%.2f"),
        "测算比值": st.column_config.TextColumn("测算比值"),
        "结论状态": st.column_config.TextColumn("结论状态"),
    }

    display_df, subpart_visible_columns = prepare_table_view(
        display_df,
        "subpart_table",
        default_search_columns=["一级总成料号", "一级总成品名描述", "结论状态"],
        filter_title="拆分件成本监控",
    )
    render_standard_data_editor(
        display_df[subpart_visible_columns],
        "subpart_table",
        column_config=column_config,
        max_height=600,
    )

    if not display_df.empty:
        status_html_parts = ['<div style="margin-top: 8px; font-size: 13px;">']
        status_html_parts.append(
            '<span style="display:inline-block;padding:2px 8px;background-color:#e74c3c;color:white;border-radius:4px;margin-right:8px;">异常</span> 测算比值 &gt; 120%（子件加价20%后超过总成价）'
        )
        status_html_parts.append(
            '&nbsp;&nbsp;&nbsp;<span style="display:inline-block;padding:2px 8px;background-color:#27ae60;color:white;border-radius:4px;margin-right:8px;">正常</span> 测算比值 ≤ 120%'
        )
        status_html_parts.append('</div>')
        st.markdown("".join(status_html_parts), unsafe_allow_html=True)

    subpart_export_df = display_df[subpart_visible_columns].copy()
    render_deferred_download_button(
        label="📥 下载异常拆分件报表",
        prepare_label="准备导出异常拆分件报表",
        data_builder=lambda export_df=subpart_export_df: to_excel_bytes(export_df),
        file_name=f"拆分件成本监控_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="subpart_export",
        fingerprint=dataframe_export_fingerprint(subpart_export_df),
    )
