import multiprocessing
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT_ROOT)

import streamlit as st

from assembly_ui import render_assembly_audit_page
from app_context import bootstrap_app
import harness
from cost_monitor_ui import render_cost_anomaly_page, render_cost_skills_page, render_interval_compare_page
from general_pages import (
    render_overview_page,
    render_report_page,
    render_settings_page,
    render_single_material_page,
    render_vehicle_gradient_compare_page,
    render_vehicle_rank_config_page,
)
from sheet_metal_ui import (
    render_sheet_metal_non_material_coefficients_page,
    render_sheet_metal_price_suggestion_page,
    render_sheet_metal_review_page,
    render_sheet_metal_skills_page,
)


st.set_page_config(
    page_title="备件成本监控看板",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)


BOARD_PAGES = [
    ("单个物料监控", "📈 单个物料监控", render_single_material_page),
    ("全量成本报表", "📑 全量成本报表", lambda: render_report_page("全量成本报表")),
    ("车系梯度配置", "📁 车系梯度配置", render_vehicle_rank_config_page),
    ("车系梯度成本对比", "📊 车系梯度对比", render_vehicle_gradient_compare_page),
    ("拆分件成本监控", "🔩 拆分件成本监控", render_assembly_audit_page),
]

ANOMALY_PAGES = [
    ("成本异常监控", "📌 成本异常监控", render_cost_anomaly_page),
    ("钣金件白痴指数复核", "🧩 钣金件白痴指数复核", render_sheet_metal_review_page),
    ("钣金件非材料成本系数", "🧮 钣金件非材料成本系数", render_sheet_metal_non_material_coefficients_page),
    ("钣金件价格建议", "💰 钣金件价格建议", render_sheet_metal_price_suggestion_page),
]

SKILLS_PAGES = [
    ("成本区间 Skills", "🧠 成本区间 Skills", render_cost_skills_page),
    ("钣金件指数 Skills", "🧠 钣金件指数 Skills", render_sheet_metal_skills_page),
    ("车系-备件成本区间对照", "📐 车系-备件成本区间对照", render_interval_compare_page),
]

SECTION_DEFINITIONS = [
    ("概览", "🏠 概览", []),
    ("系统设置", "⚙️ 系统设置", []),
    ("全量成本看板", "📊 全量成本看板", BOARD_PAGES),
    ("异常监控体系", "📌 异常监控体系", ANOMALY_PAGES),
    ("Skills 技能引擎", "🧠 Skills 技能引擎", SKILLS_PAGES),
]

SECTION_LABELS = {section_name: label for section_name, label, _ in SECTION_DEFINITIONS}
SECTION_PAGE_GROUPS = {section_name: pages for section_name, _, pages in SECTION_DEFINITIONS if pages}
PAGE_TO_SECTION = {
    page_name: section_name
    for section_name, _, pages in SECTION_DEFINITIONS
    for page_name, _, _ in pages
}

PAGE_RENDERERS = {
    "概览": render_overview_page,
    "系统设置": render_settings_page,
    **{page_name: renderer for page_name, _, renderer in BOARD_PAGES},
    **{page_name: renderer for page_name, _, renderer in ANOMALY_PAGES},
    **{page_name: renderer for page_name, _, renderer in SKILLS_PAGES},
}


def _inject_sidebar_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] div.stButton {
            margin: 0 0 0.32rem 0;
        }

        [data-testid="stSidebar"] div.stButton > button {
            width: 100%;
            justify-content: flex-start;
            min-height: 2.45rem;
            padding: 0.48rem 0.72rem;
            border-radius: 14px;
            border: 1px solid #d7dde4;
            background: linear-gradient(180deg, #ffffff 0%, #f4f7f8 100%);
            color: #1f2d3d;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            transition: transform 0.14s ease, box-shadow 0.14s ease, border-color 0.14s ease, background 0.14s ease;
        }

        [data-testid="stSidebar"] div.stButton > button p {
            margin: 0;
            width: 100%;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            line-height: 1.25;
            font-size: inherit;
        }

        [data-testid="stSidebar"] div.stButton > button:hover {
            border-color: #b8c4cf;
            background: linear-gradient(180deg, #ffffff 0%, #eef3f5 100%);
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            border: 2px solid #79c993;
            background: #ffffff;
            color: #173a26;
            box-shadow: 0 7px 16px rgba(46, 125, 50, 0.12);
        }

        [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
            border-color: #63ba83;
            background: #ffffff;
            box-shadow: 0 8px 18px rgba(46, 125, 50, 0.16);
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: none;
            background: transparent;
            margin-bottom: 0.24rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            border: none;
            background: transparent;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            padding: 0.14rem 0.08rem 0.28rem;
            border-radius: 10px;
            cursor: pointer;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background: rgba(116, 129, 141, 0.08);
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
            font-weight: 700;
            font-size: 1rem;
            color: #24323f;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
            margin: 0.1rem 0 0.42rem;
            padding: 0.45rem;
            background: #f0f2f6;
            border: 1px solid #e1e6ec;
            border-radius: 14px;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div.stButton {
            margin: 0 0 0.18rem 0;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div.stButton:last-child {
            margin-bottom: 0;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div.stButton > button {
            min-height: 2rem;
            padding: 0.36rem 0.48rem;
            border-radius: 11px;
            border: 1px solid transparent;
            background: transparent;
            box-shadow: none;
            font-size: 1.01rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div.stButton > button:hover {
            background: rgba(255, 255, 255, 0.82);
            border-color: #d7dde4;
            box-shadow: none;
            transform: none;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div.stButton > button[kind="primary"] {
            border: 2px solid #79c993;
            background: #ffffff;
            box-shadow: 0 4px 10px rgba(46, 125, 50, 0.10);
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div.stButton > button[kind="primary"]:hover {
            background: #ffffff;
            box-shadow: 0 5px 12px rgba(46, 125, 50, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_current_page(page_name: str) -> None:
    st.session_state.current_page = page_name
    st.session_state.active_page = page_name
    active_section = _resolve_active_section(page_name)
    if active_section in SECTION_PAGE_GROUPS:
        _keep_section_expanded(active_section)


def _resolve_active_section(current_page: str) -> str:
    if current_page in {"概览", "系统设置"}:
        return current_page
    return PAGE_TO_SECTION.get(current_page, "概览")


def _get_sidebar_expanded_sections(current_page: str) -> dict[str, bool]:
    active_section = _resolve_active_section(current_page)
    stored_sections = st.session_state.get("sidebar_expanded_sections")
    if not isinstance(stored_sections, dict):
        stored_sections = {}

    expanded_sections = {
        section_name: bool(stored_sections.get(section_name, section_name == active_section))
        for section_name in SECTION_PAGE_GROUPS
    }

    if active_section and not any(expanded_sections.values()):
        expanded_sections[active_section] = True

    st.session_state.sidebar_expanded_sections = expanded_sections
    return expanded_sections


def _keep_section_expanded(section_name: str) -> None:
    expanded_sections = dict(st.session_state.get("sidebar_expanded_sections", {}))
    for existing_section in SECTION_PAGE_GROUPS:
        expanded_sections[existing_section] = bool(expanded_sections.get(existing_section, False))
    if section_name in expanded_sections:
        expanded_sections[section_name] = True
    st.session_state.sidebar_expanded_sections = expanded_sections


def _render_nav_button(page_name: str, label: str, *, key: str, is_selected: bool, sidebar_container=None) -> bool:
    button_label = f"✅ {label}" if is_selected else label
    button_owner = sidebar_container or st.sidebar
    return button_owner.button(
        button_label,
        key=key,
        type="primary" if is_selected else "secondary",
        width="stretch",
    )


def _render_sidebar() -> str:
    _inject_sidebar_styles()
    st.sidebar.title("🚀 功能导航")
    harness.render_sidebar_integrity_warning()
    current_page = str(st.session_state.get("current_page", "概览") or "概览")
    active_section = _resolve_active_section(current_page)
    expanded_sections = _get_sidebar_expanded_sections(current_page)

    if _render_nav_button(
        "概览",
        "🏠 概览",
        key="nav_quick_overview",
        is_selected=current_page == "概览",
    ):
        _set_current_page("概览")
        st.rerun()

    if _render_nav_button(
        "系统设置",
        "⚙️ 系统设置",
        key="nav_quick_settings",
        is_selected=current_page == "系统设置",
    ):
        _set_current_page("系统设置")
        st.rerun()

    st.sidebar.divider()

    for section_name in ["全量成本看板", "异常监控体系", "Skills 技能引擎"]:
        pages = SECTION_PAGE_GROUPS[section_name]

        expander = st.sidebar.expander(
            SECTION_LABELS[section_name],
            expanded=expanded_sections.get(section_name, False) or active_section == section_name,
        )
        with expander:
            for page_name, label, _ in pages:
                if _render_nav_button(
                    page_name,
                    label,
                    key=f"sidebar_nav_{section_name}_{page_name}",
                    is_selected=current_page == page_name,
                    sidebar_container=expander,
                ):
                    _set_current_page(page_name)
                    st.rerun()

    harness.render_sidebar_governance_status()
    return str(st.session_state.get("current_page", "概览") or "概览")


def main() -> None:
    bootstrap_app()
    active_page = _render_sidebar()

    renderer = PAGE_RENDERERS.get(active_page, render_overview_page)
    renderer()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
