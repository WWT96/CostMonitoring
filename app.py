
import io
import time
from datetime import datetime

from config import settings

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

import processor

st.set_page_config(
    page_title="备件成本监控看板",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)


def inject_css(is_overview: bool = False):
    base_css = """
    <style>
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 98% !important;
        }
        [data-testid="stSidebarCollapseButton"] {
            z-index: 99999 !important;
            visibility: visible !important;
            display: block !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            z-index: 99999 !important;
            visibility: visible !important;
            display: block !important;
            left: 10px !important;
            top: 10px !important;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e9ecef;
            background-color: #f8f9fa;
        }
        /* 主目录（Expander 标题）层级样式 */
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            font-weight: 700 !important;
            color: #2f3e52 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
            font-size: 15px !important;
            font-weight: 700 !important;
            color: #2f3e52 !important;
        }
        /* 子目录按钮样式：更小字号、单行、省略号、紧凑间距 */
        [data-testid="stSidebar"] div[data-testid="stExpander"] div.stButton > button {
            font-size: 13px !important;
            font-weight: 500 !important;
            color: #6b7280 !important;
            padding: 0.25rem 0.5rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            line-height: 1.2 !important;
        }
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0.25rem !important;
        }
        header[data-testid="stHeader"] {
            background: transparent;
            pointer-events: none;
        }
        header[data-testid="stHeader"] > div:first-child {
            pointer-events: auto;
        }
        footer {
            display: none;
        }
        [data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        body {
            font-family: "Microsoft YaHei", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if is_overview:
        st.markdown(
            """
            <style>
                .stApp {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    color: #2c3e50;
                }
                .overview-title {
                    text-align: center;
                    font-size: 2.2rem;
                    font-weight: 700;
                    color: #2c3e50;
                    margin-top: 10vh;
                    margin-bottom: 1.5rem;
                }
                .overview-metric {
                    font-size: 6rem;
                    font-weight: 800;
                    color: #2c3e50;
                    text-align: center;
                    margin: 0;
                }
                .overview-subtitle {
                    text-align: center;
                    font-size: 1.4rem;
                    color: #576574;
                    margin-bottom: 5vh;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                .stApp {
                    background: white;
                    color: inherit;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

def reset_search_callback():
    st.session_state.search_code = ""
    st.session_state.search_name = ""
    st.session_state.report_page_number = 1


@st.cache_data
def cached_load_data(folder_path: str):
    return processor.load_data_from_folder(folder_path)


@st.cache_data
def cached_pivot_report(df, price_col: str):
    return processor.generate_pivot_report(df, price_col)


@st.cache_data
def cached_trend_report(df, price_col: str):
    return processor.generate_trend_report(df, price_col)


@st.cache_data
def cached_vehicle_compare(df, price_col: str, part_name: str, rank_tuple: tuple):
    return processor.get_vehicle_gradient_comparison(df, price_col, part_name, list(rank_tuple))


@st.cache_data
def cached_anomaly_report(df, price_col: str):
    return processor.detect_cost_anomalies(df, price_col)


@st.cache_data
def cached_subpart_analysis(df, price_col: str):
    return processor.analyze_subpart_costs(df, price_col)


@st.cache_data
def cached_anomaly_report_weighted(
    df, price_col: str, expert_labels_tuple: tuple,
    sigma_multiplier: float = 1.0, expert_weight_override: int = 0,
    skills_overrides_json: str = "",
):
    return processor.detect_cost_anomalies_weighted(
        df, price_col, expert_labels_tuple,
        sigma_multiplier=sigma_multiplier,
        expert_weight_override=expert_weight_override,
        skills_overrides_json=skills_overrides_json,
    )


@st.cache_data
def cached_load_api_data(refresh_token: float):
    """从 Supabase 的 core_cost_records 表加载数据。refresh_token 用于手动失效缓存。"""
    try:
        return processor.load_core_cost_records()
    except Exception as e:
        return None, None, f"读取数据库失败: {e}"


def require_price_col(df):
    price_col = st.session_state.get("price_col", "")
    if price_col and price_col in df.columns:
        return price_col

    detected_price_col = processor.detect_price_column(df.columns)
    if detected_price_col:
        st.session_state.price_col = detected_price_col
        return detected_price_col

    st.error("当前数据中未找到可用价格列，请检查源数据列名。")
    st.stop()


for key, default in {
    "data": None,
    "price_col": "",
    "data_source": "local",
    "anomaly_mode": "原始测算",
    "folder_path": "",
    "report_page_number": 1,
    "search_code": "",
    "search_name": "",
    "last_search_hash": "",
    "vehicle_rank_text": "",
    "vehicle_rank": [],
    "active_page": "概览",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.sidebar.title("🚀 功能导航")

st.sidebar.radio(
    "数据源",
    options=["local", "api"],
    format_func=lambda x: "📁 本地文件" if x == "local" else "🌐 云端数据库",
    key="data_source",
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

if st.sidebar.button(
    f"{'✅ ' if st.session_state.active_page == '概览' else ''}📊 概览",
    use_container_width=True,
    key="nav_overview",
):
    st.session_state.active_page = "概览"

board_pages = ["单个物料监控", "全量成本报表", "成本变动趋势", "车系梯度配置", "车系梯度成本对比", "拆分件成本监控"]
with st.sidebar.expander("全量成本看板", expanded=st.session_state.active_page in board_pages):
    for p in board_pages:
        label_map = {
            "单个物料监控": "📈 单个物料监控",
            "全量成本报表": "📑 全量成本报表",
            "成本变动趋势": "📉 成本变动趋势",
            "车系梯度配置": "📁 车系梯度配置",
            "车系梯度成本对比": "📊 车系梯度成本对比",
            "拆分件成本监控": "🔩 拆分件成本监控",
        }
        if st.button(
            f"{'✅ ' if st.session_state.active_page == p else ''}{label_map[p]}",
            use_container_width=True,
            key=f"nav_{p}",
        ):
            st.session_state.active_page = p

with st.sidebar.expander("异常成本监控体系", expanded=st.session_state.active_page == "异常成本监控体系"):
    if st.button(
        f"{'✅ ' if st.session_state.active_page == '异常成本监控体系' else ''}📌 异常成本监控体系",
        use_container_width=True,
        key="nav_abnormal",
    ):
        st.session_state.active_page = "异常成本监控体系"

skills_pages = ["Skills 技能引擎", "车系-备件成本区间对照"]
with st.sidebar.expander("Skills 技能引擎", expanded=st.session_state.active_page in skills_pages):
    _skills_label_map = {
        "Skills 技能引擎": "🧠 Skills 技能引擎",
        "车系-备件成本区间对照": "📐 车系-备件成本区间对照",
    }
    for _sp in skills_pages:
        if st.button(
            f"{'✅ ' if st.session_state.active_page == _sp else ''}{_skills_label_map[_sp]}",
            use_container_width=True,
            key=f"nav_{_sp}",
        ):
            st.session_state.active_page = _sp

page = st.session_state.active_page


if page == "概览":
    inject_css(is_overview=True)

    if st.session_state.data is not None:
        unique_items = st.session_state.data["物料编码"].nunique()
        count_display = unique_items
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
    st.markdown("### 🛠️ 数据源设置")
    if st.session_state.data_source == "local":
        col1, col2 = st.columns([4, 1])

        with col1:
            current_path = st.text_input(
                "本地数据文件夹路径",
                value=st.session_state.folder_path,
                placeholder="请手动粘贴本地数据文件夹路径...",
                label_visibility="collapsed",
            )
            if current_path != st.session_state.folder_path:
                st.session_state.folder_path = current_path
            st.caption("提示：云端版本请手动输入路径或使用同步功能，浏览器无法直接调起本地选择框。")
            st.caption("优先建议使用“从云端数据库加载数据”或 API 同步功能，避免依赖本机文件系统。")

        with col2:
            if st.button("🔄 同步本地数据", type="primary", use_container_width=True):
                if not st.session_state.folder_path:
                    st.warning("请先输入文件夹路径，或切换到云端数据库 / API 同步模式")
                else:
                    with st.spinner("正在扫描并合并数据，请稍候..."):
                        merged_df, price_col, error_msg = cached_load_data(st.session_state.folder_path)
                    if error_msg:
                        st.error(error_msg)
                    else:
                        st.session_state.data = merged_df
                        st.session_state.price_col = price_col or ""
                        st.success(f"✅ 已加载 {len(merged_df)} 条记录")
                        time.sleep(1)
                        st.rerun()
    else:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.text_input(
                "云端数据源",
                value="Supabase / core_cost_records",
                disabled=True,
                label_visibility="collapsed",
            )
        with col2:
            if st.button("🔄 加载云端数据", type="primary", use_container_width=True):
                with st.spinner("正在加载云端数据库数据..."):
                    merged_df, price_col, error_msg = cached_load_api_data(time.time())
                if error_msg:
                    st.error(error_msg)
                else:
                    st.session_state.data = merged_df
                    st.session_state.price_col = price_col or ""
                    st.success(f"✅ 已加载 {len(merged_df)} 条云端记录")
                    time.sleep(1)
                    st.rerun()
        st.caption(
            f"💡 通过 POST http://localhost:{settings.api_port}/sync_data"
            " 向此服务推送数据后，记录会写入 Supabase，再点击上方按钮加载。"
        )

elif page == "单个物料监控":
    inject_css(is_overview=False)
    st.title("📈 单个物料成本监控")

    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        df = st.session_state.data
        price_col = require_price_col(df)
        items = sorted(df["物料编码"].astype(str).unique())
        selected_item = st.selectbox("🔍 搜索/选择物料编码", items)

        item_data = df[df["物料编码"].astype(str) == selected_item].sort_values("monitor_date")
        if item_data.empty:
            st.info("该物料暂无有效数据")
        else:
            metrics = processor.get_material_metrics(item_data, price_col)
            st.markdown("### 📊 核心指标 (全工厂聚合)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("最新成本", f" {metrics['latest_price']:,.2f}")
            m2.metric("历史最低", f" {metrics['min_price']:,.2f}")
            m3.metric("历史最高", f" {metrics['max_price']:,.2f}")
            m4.metric("最大变化幅度", f"{metrics['max_change_pct']:.2%}")
            m5.metric("累计变化幅度", f"{metrics['cum_change_pct']:.2%}", delta_color="inverse")

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
            st.plotly_chart(fig, use_container_width=True)

elif page in ["全量成本报表", "成本变动趋势"]:
    inject_css(is_overview=False)
    st.title(f"📑 {page}")

    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        df = st.session_state.data
        price_col = require_price_col(df)

        with st.spinner(f"正在生成{page}..."):
            if page == "成本变动趋势":
                report_df = cached_trend_report(df, price_col)
            else:
                report_df = cached_pivot_report(df, price_col)

        st.markdown("#### 🔍 筛选条件与导出")
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        with c1:
            st.text_input("搜索物料编码 (支持空格分隔多值)", key="search_code")
        with c2:
            st.text_input("搜索备件简称", key="search_name")
        with c3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("🔄 重置检索", use_container_width=True, on_click=reset_search_callback)

        current_search_hash = f"{page}_{st.session_state.search_code}_{st.session_state.search_name}"
        if st.session_state.last_search_hash != current_search_hash:
            st.session_state.report_page_number = 1
            st.session_state.last_search_hash = current_search_hash

        filtered_df = processor.filter_report_df(
            report_df,
            st.session_state.search_code,
            st.session_state.search_name,
        )

        with c4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            try:
                excel_data = processor.to_excel_bytes(filtered_df)
                file_name = f"{page}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    "📥 导出报表",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"导出失败: {e}")

        page_info = processor.paginate_by_material(filtered_df, st.session_state.report_page_number, 50)
        st.session_state.report_page_number = page_info["page_number"]
        page_data = page_info["page_df"]

        st.markdown(f"**共找到 {len(filtered_df)} 条匹配记录**")

        value_prefix = "变动趋势" if page == "成本变动趋势" else "价格变动"
        value_cols = [col for col in page_data.columns if col.startswith(value_prefix)]
        value_cols.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))

        html = processor.render_merged_html_table(
            page_data,
            value_cols,
            is_trend_mode=(page == "成本变动趋势"),
        )
        st.markdown(html, unsafe_allow_html=True)

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

elif page == "车系梯度配置":
    inject_css(is_overview=False)
    st.title("📁 车系梯度配置")
    st.markdown("每行输入一个车系名称，顺序即梯度排名顺序。系统将自动忽略空格差异进行匹配。")

    rank_text = st.text_area(
        "车系列表",
        value=st.session_state.vehicle_rank_text,
        height=260,
        placeholder="例如:\n远景Max\n洞明S\nE300",
    )

    if st.button("💾 保存配置", type="primary"):
        rank_list = processor.parse_vehicle_rank_config(rank_text)
        st.session_state.vehicle_rank_text = rank_text
        st.session_state.vehicle_rank = rank_list
        st.success(f"配置已生效，已识别 {len(rank_list)} 个梯度车系")

    if st.session_state.vehicle_rank:
        st.markdown("#### 当前配置")
        for i, name in enumerate(st.session_state.vehicle_rank, start=1):
            st.write(f"{i}. {name}")

elif page == "车系梯度成本对比":
    inject_css(is_overview=False)
    st.title("📊 车系梯度成本对比")

    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        df = st.session_state.data
        price_col = require_price_col(df)
        part_options = sorted(df["备件简称"].astype(str).unique())

        selected_part = st.selectbox("备件简称筛选", part_options)
        compare_df = cached_vehicle_compare(df, price_col, selected_part, tuple(st.session_state.vehicle_rank))

        st.markdown(f"**共找到 {len(compare_df)} 条匹配记录**")
        st.markdown(processor.render_center_table_html(compare_df), unsafe_allow_html=True)

        try:
            export_data = processor.to_excel_bytes(compare_df)
            st.download_button(
                "📥 导出对比结果",
                data=export_data,
                file_name=f"车系梯度成本对比_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"导出失败: {e}")

elif page == "拆分件成本监控":
    inject_css(is_overview=False)
    st.title("🔩 拆分件成本监控")

    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        df = st.session_state.data
        price_col = require_price_col(df)

        if "一级总成料号" not in df.columns:
            st.warning("⚠️ 当前数据中缺少「一级总成料号」字段，无法进行拆分件分析。请检查数据源。")
        else:
            with st.spinner("正在分析拆分件成本..."):
                subpart_df = cached_subpart_analysis(df, price_col)

            if subpart_df.empty:
                st.info("当前数据中没有一级总成料号不为空的记录。")
            else:
                # ── 顶部汇总指标 ──────────────────────────────
                total_assy = len(subpart_df)
                abnormal_count = int((subpart_df["结论状态"] == "异常").sum())
                normal_count = int((subpart_df["结论状态"] == "正常").sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("总成总数", f"{total_assy}")
                m2.metric("异常", f"{abnormal_count}")
                m3.metric("正常", f"{normal_count}")

                # ── 显示开关：仅异常 / 全部 ──────────────────
                show_mode = st.radio(
                    "显示范围",
                    options=["仅异常", "全部"],
                    horizontal=True,
                    key="subpart_show_mode",
                )

                display_df = subpart_df.copy()
                if show_mode == "仅异常":
                    display_df = display_df[display_df["结论状态"] == "异常"].copy()

                # ── 关键字筛选 ────────────────────────────────
                fc1, fc2 = st.columns(2)
                with fc1:
                    filter_assy = st.text_input(
                        "筛选 一级总成料号",
                        key="subpart_filter_assy",
                        placeholder="输入关键字筛选...",
                    )
                with fc2:
                    filter_desc = st.text_input(
                        "筛选 一级总成品名描述",
                        key="subpart_filter_desc",
                        placeholder="输入关键字筛选...",
                    )

                if filter_assy:
                    display_df = display_df[
                        display_df["一级总成料号"].astype(str).str.contains(filter_assy, case=False, na=False)
                    ]
                if filter_desc:
                    display_df = display_df[
                        display_df["一级总成品名描述"].astype(str).str.contains(filter_desc, case=False, na=False)
                    ]

                st.markdown(f"**共 {len(display_df)} 条记录**")

                # ── 报表列顺序 ────────────────────────────────
                ordered_cols = [
                    "一级总成料号", "一级总成品名描述", "一级总成成本",
                    "子零件数量", "子零件加权总和", "测算总成成本",
                    "测算比值", "结论状态",
                ]
                # 保留存在的列，并附加其余列
                extra_cols = [c for c in display_df.columns if c not in ordered_cols]
                final_cols = [c for c in ordered_cols if c in display_df.columns] + extra_cols
                display_df = display_df[final_cols]

                # ── 测算比值转百分比展示列 ─────────────────────
                if "测算比值" in display_df.columns:
                    display_df = display_df.copy()
                    display_df["测算比值"] = display_df["测算比值"].apply(
                        lambda v: f"{v:.4%}" if isinstance(v, (int, float)) and v == v else ""
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

                st.data_editor(
                    display_df,
                    column_config=column_config,
                    disabled=True,
                    use_container_width=True,
                    height=min(600, 35 * len(display_df) + 38),
                    hide_index=True,
                    key="subpart_table",
                )

                # ── 使用 HTML 展示颜色标注的结论状态 ──────────
                if not display_df.empty:
                    status_html_parts = ['<div style="margin-top: 8px; font-size: 13px;">']
                    status_html_parts.append(
                        '<span style="display:inline-block;padding:2px 8px;'
                        'background-color:#e74c3c;color:white;border-radius:4px;'
                        'margin-right:8px;">异常</span> 测算比值 &gt; 120%（子件加价20%后超过总成价）'
                    )
                    status_html_parts.append(
                        '&nbsp;&nbsp;&nbsp;'
                        '<span style="display:inline-block;padding:2px 8px;'
                        'background-color:#27ae60;color:white;border-radius:4px;'
                        'margin-right:8px;">正常</span> 测算比值 ≤ 120%'
                    )
                    status_html_parts.append('</div>')
                    st.markdown("".join(status_html_parts), unsafe_allow_html=True)

                # ── 导出按钮 ─────────────────────────────────
                try:
                    export_data = processor.to_excel_bytes(display_df)
                    st.download_button(
                        "📥 导出异常拆分件报表",
                        data=export_data,
                        file_name=f"拆分件成本监控_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"导出失败: {e}")

elif page == "异常成本监控体系":
    inject_css(is_overview=False)
    st.title("📌 异常成本监控体系")
    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        df = st.session_state.data
        price_col = require_price_col(df)

        # ── 双模式开关 + 专家标注统计 ──────────────────────────
        mode_col, stat_col = st.columns([3, 2])
        with mode_col:
            st.radio(
                "测算模式",
                options=["原始测算", "优化后测算（专家纠偏）"],
                key="anomaly_mode",
                horizontal=True,
            )
        with stat_col:
            label_count = processor.label_manager.count()
            st.metric("已由专家校准的记录", f"{label_count} 条")

        # ── 专家校准管理中心 ─────────────────────────────────
        if label_count > 0:
            with st.expander("📋 查看/管理已校准记录", expanded=False):
                all_labels = processor.label_manager.get_labels()
                mgmt_rows = [
                    {"记录主键": k, "当前标注": v} for k, v in all_labels.items()
                ]
                mgmt_df = pd.DataFrame(mgmt_rows)
                mgmt_df["撤回标注"] = False

                mgmt_edited = st.data_editor(
                    mgmt_df,
                    column_config={
                        "撤回标注": st.column_config.CheckboxColumn(
                            "撤回标注", help="勾选后点击下方按钮撤回此标注", default=False,
                        ),
                        "记录主键": st.column_config.TextColumn("记录主键", disabled=True),
                        "当前标注": st.column_config.TextColumn("当前标注", disabled=True),
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=min(300, 35 * len(mgmt_df) + 38),
                    key="calibration_mgmt",
                )

                mgmt_c1, mgmt_c2 = st.columns(2)
                with mgmt_c1:
                    if st.button("🗑️ 撤回选中的标注", type="primary"):
                        keys_to_revoke = []
                        for i, row in mgmt_edited.iterrows():
                            if row["撤回标注"]:
                                keys_to_revoke.append(mgmt_df.iloc[i]["记录主键"])
                        if keys_to_revoke:
                            revoked = processor.label_manager.delete_labels(keys_to_revoke)
                            st.cache_data.clear()
                            st.success(f"✅ 已撤回 {revoked} 条标注")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.warning("未选中任何记录")
                with mgmt_c2:
                    if st.button("⚠️ 清空所有标注"):
                        st.session_state["_confirm_clear_labels"] = True
                    if st.session_state.get("_confirm_clear_labels"):
                        st.warning("确定要清空所有专家标注吗？此操作不可撤销。")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button("✅ 确认清空", type="primary"):
                                processor.label_manager.clear_all()
                                st.cache_data.clear()
                                st.session_state["_confirm_clear_labels"] = False
                                st.success("✅ 已清空所有标注")
                                time.sleep(0.5)
                                st.rerun()
                        with cc2:
                            if st.button("取消"):
                                st.session_state["_confirm_clear_labels"] = False
                                st.rerun()

        st.markdown("---")

        unique_short_name = df["备件简称"].astype(str).nunique()
        if len(df) >= 30000 or unique_short_name >= 300:
            st.info("正在进行大规模深度测算，请稍候...")

        try:
            anomaly_df = cached_anomaly_report(df, price_col)
        except ImportError as e:
            st.error(str(e))
            st.info("安装完成后重启应用，再进入本页面即可。")
            st.code("pip install scikit-learn")
            st.stop()
        except Exception as e:
            st.error(f"异常检测失败: {e}")
            st.stop()

        if anomaly_df.empty:
            st.info("当前数据暂无可检测记录。")
            st.stop()

        # ── 优化后模式：使用加权算法重新计算 ──────────────────
        is_expert_mode = st.session_state.anomaly_mode == "优化后测算（专家纠偏）"

        # ── 闭环自学习：加载 Skills 技能书参数 ────────────────
        _skills_data = processor.load_skills()
        _skills_json = ""
        _skills_loaded = False
        if _skills_data and is_expert_mode:
            try:
                import json as _json
                _so = {}
                for sk in _skills_data["skills"]:
                    _so[sk["备件简称"]] = {
                        "sigma": sk.get("当前σ参数", 1.0),
                        "weight": sk.get("偏置权重", 80),
                    }
                _skills_json = _json.dumps(_so, ensure_ascii=False)
                _skills_loaded = True
            except Exception:
                _skills_json = ""

        if is_expert_mode:
            expert_labels = processor.label_manager.get_labels()
            if expert_labels:
                working_df = cached_anomaly_report_weighted(
                    df, price_col, tuple(sorted(expert_labels.items())),
                    skills_overrides_json=_skills_json,
                )
            else:
                working_df = anomaly_df.copy()
                st.info("💡 暂无专家标注数据，显示原始测算结果。请先在下方表格中勾选并保存标注。")
        else:
            working_df = anomaly_df.copy()

        # ── Skills 闭环状态提示 ───────────────────────────────
        if is_expert_mode:
            if _skills_loaded:
                _sk_count = len(_skills_data.get("skills", []))
                _sk_time = _skills_data.get("saved_at", "未知")
                st.info(
                    f"📘 已加载 Skills 技能书（{_sk_count} 个备件简称，保存于 {_sk_time}），"
                    f"匹配的备件将使用个性化 σ/权重参数。"
                )
            elif _skills_data is None and processor.has_skills_snapshot():
                st.warning("⚠️ Skills 技能书快照读取异常，已回退到默认算法。请重新运行 AutoResearch 生成。")

        # ── 顶部汇总 ──────────────────────────────────────────
        high_count = int(working_df["status"].astype(str).str.contains("异常偏高").sum())
        low_count = int(working_df["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum())
        abnormal_total = high_count + low_count

        m1, m2, m3 = st.columns(3)
        m1.metric("异常总数", f"{abnormal_total}")
        m2.metric("异常偏高", f"{high_count}")
        m3.metric("异常偏低", f"{low_count}")

        if is_expert_mode:
            orig_high = int(anomaly_df["status"].astype(str).str.contains("异常偏高").sum())
            orig_low = int(anomaly_df["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum())
            orig_total = orig_high + orig_low
            if orig_total > 0:
                reduced = orig_total - abnormal_total
                st.caption(
                    f"💡 加权自学习算法重新测算后，异常记录从 {orig_total} 条降至 {abnormal_total} 条"
                    f"（减少 {reduced} 条，降幅 {reduced / orig_total:.1%}）"
                )

        # ── 筛选 ──────────────────────────────────────────────
        short_name_options = ["全部"] + sorted(working_df["备件简称"].astype(str).unique().tolist())
        selected_short_name = st.selectbox("🔍 备件简称筛选", short_name_options)

        filtered_anomaly_df = working_df.copy()
        if selected_short_name != "全部":
            filtered_anomaly_df = filtered_anomaly_df[
                filtered_anomaly_df["备件简称"].astype(str) == selected_short_name
            ].copy()

        # ── 异常记录表格 + 手动标注 ──────────────────────────
        abnormal_view = filtered_anomaly_df[
            filtered_anomaly_df["status"].astype(str).str.contains("异常偏高|异常偏低|严重异常偏低")
        ].copy()

        st.markdown(f"**共找到 {len(abnormal_view)} 条异常记录**")

        if not abnormal_view.empty and "_record_key" in abnormal_view.columns:
            edit_df = abnormal_view.copy()
            existing_labels = processor.label_manager.get_labels()
            edit_df["标注为正常"] = edit_df["_record_key"].apply(
                lambda k: existing_labels.get(k) == "正常"
            )

            display_cols = [c for c in edit_df.columns if c != "_record_key"]
            if "标注为正常" in display_cols:
                display_cols.remove("标注为正常")
            display_cols = ["标注为正常"] + display_cols

            # 丰富 column_config：每列指定类型以获得内置筛选/排序能力
            column_config = {
                "标注为正常": st.column_config.CheckboxColumn(
                    "标注为正常",
                    help="勾选此项将该记录标注为「正常」（专家纠偏）",
                    default=False,
                ),
                "物料编码": st.column_config.TextColumn("物料编码", disabled=True),
                "物料名称": st.column_config.TextColumn("物料名称", disabled=True),
                "适用车系": st.column_config.TextColumn("适用车系", disabled=True),
                "工厂": st.column_config.TextColumn("工厂", disabled=True),
                "备件简称": st.column_config.TextColumn("备件简称", disabled=True),
                "实际成本": st.column_config.NumberColumn(
                    "实际成本", disabled=True, format="%.2f",
                ),
                "价格有效于": st.column_config.DateColumn("价格有效于", disabled=True),
                "样本量": st.column_config.NumberColumn("样本量", disabled=True),
                "预测值": st.column_config.NumberColumn(
                    "预测值", disabled=True, format="%.2f",
                ),
                "合理下限": st.column_config.NumberColumn(
                    "合理下限", disabled=True, format="%.2f",
                ),
                "合理上限": st.column_config.NumberColumn(
                    "合理上限", disabled=True, format="%.2f",
                ),
                "偏离数值": st.column_config.NumberColumn(
                    "偏离数值", disabled=True, format="%.2f",
                ),
                "偏离比例": st.column_config.NumberColumn(
                    "偏离比例", disabled=True, format="%.2%%",
                ),
                "status": st.column_config.TextColumn("status", disabled=True),
            }
            if "专家校准" in display_cols:
                column_config["专家校准"] = st.column_config.TextColumn("专家校准", disabled=True)
            if "判定依据" in display_cols:
                column_config["判定依据"] = st.column_config.TextColumn("判定依据", disabled=True)

            edited = st.data_editor(
                edit_df[display_cols],
                column_config=column_config,
                use_container_width=True,
                height=420,
                hide_index=True,
                key="anomaly_editor",
            )

            if st.button("💾 保存专家标注", type="primary"):
                final_labels = dict(existing_labels)
                for i, (_, orig_row) in enumerate(edit_df.iterrows()):
                    rk = orig_row["_record_key"]
                    checked = bool(edited.iloc[i]["标注为正常"])
                    if checked:
                        final_labels[rk] = "正常"
                    elif rk in final_labels and final_labels[rk] == "正常":
                        del final_labels[rk]

                processor.label_manager._flush(final_labels)
                st.cache_data.clear()
                st.success(f"✅ 标注已保存！当前共 {len(final_labels)} 条专家校准记录。")
                time.sleep(0.5)
                st.rerun()
        elif abnormal_view.empty:
            st.info("当前筛选条件下暂无异常记录。")
        else:
            # 无 _record_key 时仍展示 data_editor（只读），保留排序/筛选
            _ro_cols = [c for c in abnormal_view.columns if c != "_record_key"]
            _ro_config = {
                "实际成本": st.column_config.NumberColumn("实际成本", format="%.2f"),
                "预测值": st.column_config.NumberColumn("预测值", format="%.2f"),
                "合理下限": st.column_config.NumberColumn("合理下限", format="%.2f"),
                "合理上限": st.column_config.NumberColumn("合理上限", format="%.2f"),
                "偏离数值": st.column_config.NumberColumn("偏离数值", format="%.2f"),
                "偏离比例": st.column_config.NumberColumn("偏离比例", format="%.2%%"),
            }
            st.data_editor(
                abnormal_view[_ro_cols],
                column_config=_ro_config,
                disabled=True,
                use_container_width=True,
                height=420,
                hide_index=True,
                key="anomaly_readonly",
            )

        # ── 导出（包含所有计算指标，不限异常）──────────────────
        try:
            export_df = filtered_anomaly_df.drop(columns=["_record_key"], errors="ignore")
            export_data = processor.to_excel_bytes(export_df)
            st.download_button(
                "📥 导出异常成本报表",
                data=export_data,
                file_name=f"异常成本监控_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"导出失败: {e}")

        # ── 可视化 ────────────────────────────────────────────
        if selected_short_name == "全部":
            chart_df = filtered_anomaly_df.copy()
            chart_title = "全部备件简称 - 成本分布"
        else:
            chart_df = filtered_anomaly_df.copy()
            chart_title = f"{selected_short_name} - 成本分布"

        if not chart_df.empty:
            chart_df = chart_df.copy()
            chart_df["偏离比例显示"] = chart_df["偏离比例"].apply(
                lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and x == x else ""
            )

            nbins = min(80, max(10, int(len(chart_df) ** 0.5 * 4)))
            fig = px.histogram(
                chart_df,
                x="实际成本",
                nbins=nbins,
                title=chart_title,
                opacity=0.75,
                color_discrete_sequence=["#4a90e2"],
            )

            # 基准/上下限辅助线：严格取“当前筛选下最大样本群体”的边界
            anchor_group = (
                chart_df.groupby("备件简称", as_index=False)
                .size()
                .sort_values("size", ascending=False)
                .iloc[0]["备件简称"]
            )
            anchor_df = chart_df[chart_df["备件简称"] == anchor_group]
            baseline = float(anchor_df["预测值"].median())
            upper = float(anchor_df["合理上限"].median())
            lower = float(anchor_df["合理下限"].median())

            fig.add_vline(
                x=baseline,
                line_dash="dash",
                line_color="#1f77b4",
                annotation_text="基准合理价",
                annotation_position="top",
            )
            fig.add_vline(
                x=upper,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="合理上限",
                annotation_position="top",
            )
            fig.add_vline(
                x=lower,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="合理下限",
                annotation_position="top",
            )

            # 异常点散点层（hover 展示编码/价格/偏离比例）
            abnormal_points = chart_df[
                chart_df["status"].astype(str).str.contains("异常偏高|异常偏低|严重异常偏低")
            ].copy()
            if not abnormal_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=abnormal_points["实际成本"],
                        y=[0] * len(abnormal_points),
                        mode="markers",
                        marker=dict(size=10, color="#e74c3c"),
                        name="异常点",
                        customdata=abnormal_points[["物料编码", "实际成本", "偏离比例显示"]].values,
                        hovertemplate=(
                            "物料编码: %{customdata[0]}<br>"
                            "实际价格: %{customdata[1]:,.2f}<br>"
                            "偏离比例: %{customdata[2]}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title=dict(
                    text=(
                        f"{chart_title}<br>"
                        "<sup>算法已自动识别并保护梯度定价区间，剔除孤立离群点。</sup>"
                    )
                ),
                xaxis_title="成本",
                yaxis_title="频数",
                template="plotly_white",
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Skills 技能引擎":
    inject_css(is_overview=False)
    st.title("🧠 Skills 技能引擎")

    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        import skills_engine

        df = st.session_state.data
        price_col = require_price_col(df)

        try:
            anomaly_df = cached_anomaly_report(df, price_col)
        except Exception as e:
            st.error(f"异常检测失败: {e}")
            st.stop()

        if anomaly_df.empty:
            st.info("当前数据暂无可检测记录。")
            st.stop()

        expert_labels = processor.get_latest_feedback()

        # ── 防错校验：CSV 行数 vs 内存标注数不一致时强制清缓存 ──
        _file_rows = processor.label_manager.file_row_count()
        if _file_rows != len(expert_labels):
            st.cache_data.clear()
            expert_labels = processor.get_latest_feedback()

        # 若有专家标注，使用加权算法结果；否则使用原始结果
        if expert_labels:
            labels_tuple = tuple(sorted(expert_labels.items()))
            # 闭环：加载 Skills 参数用于加权检测
            _sk_data = processor.load_skills()
            _sk_json_skills = ""
            if _sk_data:
                try:
                    import json as _json_sk
                    _so_sk = {}
                    for _sk_item in _sk_data["skills"]:
                        _so_sk[_sk_item["备件简称"]] = {
                            "sigma": _sk_item.get("当前σ参数", 1.0),
                            "weight": _sk_item.get("偏置权重", 80),
                        }
                    _sk_json_skills = _json_sk.dumps(_so_sk, ensure_ascii=False)
                except Exception:
                    pass
            try:
                optimized_df = cached_anomaly_report_weighted(
                    df, price_col, labels_tuple,
                    skills_overrides_json=_sk_json_skills,
                )
            except Exception:
                optimized_df = anomaly_df
        else:
            optimized_df = anomaly_df

        # ── Section 1: Skills 技能书 ──────────────────────────
        st.markdown("## 📋 Skills 技能书")
        st.markdown("从当前异常检测结果中提取每个备件简称的算法参数与分布特征，可作为系统**知识资产**下载。")

        skills = skills_engine.extract_skills(optimized_df, expert_labels)

        # 统一口径：仅保留至少有 1 条有效专家标注的备件简称
        if expert_labels and "_record_key" in optimized_df.columns:
            _labeled_keys = set(expert_labels.keys())
            _covered_names = set()
            for _sn, _grp in optimized_df.groupby("备件简称", sort=False):
                if _labeled_keys & set(_grp["_record_key"].values):
                    _covered_names.add(str(_sn))
            skills_filtered = [s for s in skills if s["备件简称"] in _covered_names]
        else:
            skills_filtered = skills

        sk_all, sk_covered = len(skills), len(skills_filtered)
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.metric("备件简称总数", sk_all)
        with c_m2:
            st.metric("专家标注覆盖简称数", sk_covered)

        with st.expander("预览 Skills 技能书（全部备件）", expanded=False):
            md_report = skills_engine.skills_to_markdown(skills)
            st.markdown(md_report)

        # 导出使用覆盖简称的筛选集（确保统计口径一致）
        _export_skills = skills_filtered if expert_labels else skills
        dl1, dl2 = st.columns(2)
        with dl1:
            json_bytes = skills_engine.skills_to_json_bytes(_export_skills)
            st.download_button(
                f"📥 下载 Skills (JSON) — {len(_export_skills)} 个简称",
                data=json_bytes,
                file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with dl2:
            md_bytes = skills_engine.skills_to_markdown(_export_skills).encode("utf-8")
            st.download_button(
                "📥 下载 Skills (Markdown)",
                data=md_bytes,
                file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

        st.markdown("---")

        # ── Section 2: AutoResearch 棘轮迭代 ─────────────────
        st.markdown("## 🔬 AutoResearch 棘轮迭代")
        st.markdown(
            "系统自动微调 **σ 系数**与**偏置权重**，严格棘轮规则保证每次迭代只进不退。"
        )

        if not expert_labels:
            st.warning("⚠️ 暂无专家标注数据。请先在「异常成本监控体系」页面中标注并保存。")
        else:
            iter_options = [5, 10, 20]
            n_iters = st.select_slider("迭代次数", options=iter_options, value=10)

            if st.button("🚀 启动 AutoResearch", type="primary"):
                progress_bar = st.progress(0, text="初始化...")
                status_text = st.empty()

                def _on_progress(current, total, best_score, trial_score, sigma):
                    pct = current / total
                    progress_bar.progress(
                        pct,
                        text=f"迭代 {current}/{total} — 最佳得分 {best_score:.2%}",
                    )
                    status_text.caption(
                        f"本轮试验: σ={sigma:.4f} | "
                        f"试验得分 {trial_score:.2%} | 最佳得分 {best_score:.2%}"
                    )

                result = skills_engine.run_auto_research(
                    df, price_col, expert_labels, n_iters, progress_callback=_on_progress
                )

                progress_bar.progress(1.0, text="✅ 迭代完成")
                status_text.empty()

                st.success(
                    f"**最优参数**: σ = {result['best_sigma']}, "
                    f"偏置权重 = {result['best_weight']}×, "
                    f"准确率 = {result['best_score']:.2%}, "
                    f"冲突数 = {result['best_conflicts']}/{result['total_expert']}"
                )

                # 迭代历史
                with st.expander("迭代历史", expanded=False):
                    history_df = pd.DataFrame(result["history"])
                    st.dataframe(history_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # ── 优化后 Skills 下载 ────────────────────────
                st.markdown("### 📋 优化后 Skills")
                opt_skills = skills_engine.extract_skills(
                    result["result_df"],
                    expert_labels,
                    sigma_multiplier=result["best_sigma"],
                    expert_weight=result["best_weight"],
                )

                # ★ 闭环：自动将优化后 Skills 保存到本地，供下次检测使用
                _saved_path = processor.save_skills(
                    opt_skills,
                    sigma=result["best_sigma"],
                    weight=result["best_weight"],
                )
                st.cache_data.clear()
                st.success(f"✅ Skills 已自动保存至 `{_saved_path}`，下次异常检测将自动加载。")

                opt_json = skills_engine.skills_to_json_bytes(opt_skills)
                st.download_button(
                    "📥 下载优化后 Skills (JSON)",
                    data=opt_json,
                    file_name=f"skills_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

                # ── 深度审计报表 ──────────────────────────────
                st.markdown("### 📊 深度审计报表")
                st.markdown(
                    "全量备件对照：原始结论 vs 专家反馈 vs 最终优化结论"
                )
                audit_df = skills_engine.generate_audit_report(
                    anomaly_df, result["result_df"], expert_labels
                )
                st.dataframe(audit_df, use_container_width=True, height=420, hide_index=True)

                try:
                    audit_bytes = processor.to_excel_bytes(audit_df)
                    st.download_button(
                        "📥 导出深度审计报表",
                        data=audit_bytes,
                        file_name=f"深度审计报表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"导出失败: {e}")

elif page == "车系-备件成本区间对照":
    inject_css(is_overview=False)
    st.title("📐 车系-备件成本区间对照")

    if st.session_state.data is None:
        st.warning("⚠️ 请先在概览页配置数据路径并同步数据")
    else:
        df = st.session_state.data
        price_col = require_price_col(df)

        try:
            anomaly_df = cached_anomaly_report(df, price_col)
        except Exception as e:
            st.error(f"异常检测失败: {e}")
            st.stop()

        if anomaly_df.empty:
            st.info("当前数据暂无可检测记录，无法生成成本区间对照。")
            st.stop()

        # 准备区间数据：按 适用车系+备件简称 聚合取中位数
        _needed = ["适用车系", "备件简称", "预测值", "合理下限", "合理上限"]
        if not all(c in anomaly_df.columns for c in _needed):
            st.warning("异常检测结果中缺少必要列（预测值/合理下限/合理上限），无法生成图表。")
            st.stop()

        interval_df = (
            anomaly_df.groupby(["适用车系", "备件简称"], as_index=False)
            .agg({"合理下限": "median", "合理上限": "median", "预测值": "median"})
        )

        if interval_df.empty:
            st.info("当前筛选组合下暂无测算出的合理区间信息")
            st.stop()

        # ── 导出数据准备 ─────────────────────────────────────
        export_df = interval_df[["适用车系", "备件简称", "合理下限", "预测值", "合理上限"]].copy()
        export_df = export_df.rename(columns={"预测值": "基准价"})
        export_df["合理下限"] = export_df["合理下限"].clip(lower=0).round(2)
        export_df["合理上限"] = export_df["合理上限"].round(2)
        export_df["基准价"] = export_df["基准价"].round(2)
        export_df = export_df.drop_duplicates().sort_values(["适用车系", "备件简称"]).reset_index(drop=True)

        @st.cache_data
        def _build_interval_excel(_df):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                _df.to_excel(writer, index=False, sheet_name="成本区间")
            return buf.getvalue()

        # ── 筛选栏 + 导出按钮 ────────────────────────────────
        fc1, fc2, fc3 = st.columns([2, 2, 1.5])
        all_vehicles = sorted(interval_df["适用车系"].astype(str).unique().tolist())
        all_parts = sorted(interval_df["备件简称"].astype(str).unique().tolist())

        with fc1:
            selected_vehicle = st.selectbox(
                "筛选车系",
                options=["全部"] + all_vehicles,
                key="interval_vehicle",
            )
        with fc2:
            selected_part = st.selectbox(
                "筛选备件简称",
                options=["全部"] + all_parts,
                key="interval_part",
            )
        with fc3:
            st.markdown("<br>", unsafe_allow_html=True)
            fname = f"车系备件成本区间对标表_{datetime.now().strftime('%Y%m%d')}.xlsx"
            st.download_button(
                label="📥 导出全量成本区间表",
                data=_build_interval_excel(export_df),
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # ── 公共绘图函数 ──────────────────────────────────────
        def _render_interval_chart(chart_data, y_col, title_text, slider_key):
            """绘制水平悬浮长方形区间图 + 基准价竖线标记 + Streamlit 原生金额滑动条。"""
            chart_data = chart_data.copy()

            # ── 金额滑动条 ────────────────────────────────────
            global_max = float(chart_data["合理上限"].max())
            range_ceil = max(5000, int(np.ceil(global_max / 1000) * 1000))
            step = 100 if range_ceil <= 20000 else 500
            default_hi = min(range_ceil, max(5000, int(np.ceil(global_max / 1000) * 1000)))

            x_range = st.slider(
                "金额显示范围 (CNY)",
                min_value=0,
                max_value=range_ceil,
                value=(0, default_hi),
                step=step,
                key=slider_key,
            )
            x_min, x_max = x_range

            # ── 过滤：区间完全落在可视范围外的条目不显示 ──────
            chart_data = chart_data[
                (chart_data["合理上限"] >= x_min) & (chart_data["合理下限"] <= x_max)
            ]
            if chart_data.empty:
                st.info("当前金额范围内无可显示的区间数据，请调整滑动条。")
                return

            chart_data = chart_data.sort_values(y_col).reset_index(drop=True)
            labels = chart_data[y_col].astype(str).tolist()
            colors = px.colors.qualitative.Plotly

            fig = go.Figure()
            for i, (_, row) in enumerate(chart_data.iterrows()):
                lbl = labels[i]
                lo = float(row["合理下限"])
                hi = float(row["合理上限"])
                mid = float(row["预测值"])
                clr = colors[i % len(colors)]

                # 悬浮长方形色块
                fig.add_trace(go.Bar(
                    y=[lbl],
                    x=[hi - lo],
                    base=[lo],
                    orientation="h",
                    marker=dict(color=clr, line_width=0),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{processor.escape_html_text(lbl)}</b><br>"
                        f"合理下限: {lo:,.2f}<br>"
                        f"合理上限: {hi:,.2f}<br>"
                        f"基准价: {mid:,.2f}<extra></extra>"
                    ),
                ))
                # 基准价竖线标记（中轴线）
                fig.add_trace(go.Scatter(
                    y=[lbl],
                    x=[mid],
                    mode="markers",
                    marker=dict(
                        symbol="line-ns-open",
                        size=16,
                        color="#2c3e50",
                        line_width=2,
                    ),
                    showlegend=False,
                    hovertemplate=f"基准价: {mid:,.2f}<extra></extra>",
                ))

            chart_height = max(800, len(chart_data) * 35)
            fig.update_layout(
                title=dict(text=title_text, x=0.5, font=dict(size=16)),
                template="plotly_white",
                showlegend=False,
                height=chart_height,
                bargap=0.35,
                margin=dict(l=10, r=20, t=60, b=10),
                plot_bgcolor="rgba(248,249,250,0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_xaxes(
                title="",
                tickformat=",",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                range=[x_min, x_max],
                rangeslider_visible=False,
            )
            fig.update_yaxes(
                title="",
                automargin=True,
                ticklabelstandoff=20,
                showgrid=False,
                tickfont=dict(size=12),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 视图 A：选定车系 → 纵向对比不同备件 ─────────────
        if selected_vehicle != "全部":
            chart_data = interval_df[interval_df["适用车系"] == selected_vehicle].copy()
            if chart_data.empty:
                st.info("当前筛选组合下暂无测算出的合理区间信息")
            else:
                _render_interval_chart(
                    chart_data, "备件简称",
                    f"车系「{selected_vehicle}」— 各备件成本合理区间",
                    slider_key="interval_slider_a",
                )

        # ── 视图 B：选定备件简称 → 横向对比不同车系 ─────────
        elif selected_part != "全部":
            chart_data = interval_df[interval_df["备件简称"] == selected_part].copy()
            if chart_data.empty:
                st.info("当前筛选组合下暂无测算出的合理区间信息")
            else:
                _render_interval_chart(
                    chart_data, "适用车系",
                    f"备件「{selected_part}」— 各车系成本合理区间",
                    slider_key="interval_slider_b",
                )

        # ── 未选定任何维度时的提示 ────────────────────────────
        else:
            st.info("请在上方选择一个「车系」或一个「备件简称」以生成区间对照图。")
