from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app_context import get_path_setting, inject_css
from assembly_logic import (
    build_abnormal_ratio_export_df,
    is_supported_assembly_excel_file,
    load_and_process_assembly_data,
    sort_assembly_summary_by_cost_ratio,
)
from config import settings
from data_ingestion import to_excel_bytes
from local_logging import log_event
from ui_utils import escape_html_text


TREE_HEADERS = [
    "适用车系",
    "零件编码",
    "子物料数量",
    "零件名称",
    "成本",
    "订购价",
    "零售价",
    "资源开发",
    "定价负责人",
    "成本比例",
    "订购价比例",
    "零售价比例",
]

ASSEMBLY_EXPANDED_CODES_KEY = "expanded_parts"
ASSEMBLY_PAGE_KEY = "assembly_page"
ASSEMBLY_PAGE_INPUT_KEY = "assembly_page_input"
ASSEMBLY_PARENT_FILTER_KEY = "assembly_filter_parent_code"
ASSEMBLY_CHILD_FILTER_KEY = "assembly_filter_child_code"
ASSEMBLY_OWNER_FILTER_KEY = "assembly_filter_owner"
ASSEMBLY_PAGE_SIZE = 20
ASSEMBLY_DATA_SCHEMA_VERSION = "0.1.2-layer2-source-child-count"
ASSEMBLY_RENDER_COLUMNS = [
    "适用车系",
    "层级0编码",
    "子物料数量",
    "层级0名称",
    "层级0成本",
    "层级0订购价",
    "层级0零售价",
    "资源开发",
    "定价负责人",
    "成本比例",
    "订购价比例",
    "零售价比例",
]
ASSEMBLY_TABLE_HEADER_HEIGHT_REM = 2.8
ASSEMBLY_TABLE_ROW_HEIGHT_REM = 2.8
ASSEMBLY_TOGGLE_LAYOUT = [0.34, 9.66]


def _inject_assembly_table_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --assembly-header-height: 2.8rem;
                --assembly-row-height: 2.8rem;
                --assembly-toggle-icon-height: 1.05rem;
            }
            .assembly-table-shell {
                width: 100%;
                overflow-x: auto;
                margin-top: 0.35rem;
                margin-left: -0.45rem;
            }
            .assembly-audit-table {
                width: 100%;
                border-collapse: collapse;
                table-layout: auto;
                font-size: 0.84rem;
                color: #223244;
                background: #ffffff;
                border: 1px solid #d9e0e7;
            }
            .assembly-audit-table th,
            .assembly-audit-table td {
                border: 1px solid #d9e0e7;
                padding: 0 0.65rem;
                vertical-align: middle;
                height: var(--assembly-row-height);
            }
            .assembly-audit-table th {
                background: #f2f5f7;
                color: #405264;
                font-weight: 700;
                text-align: center;
                white-space: nowrap;
                height: var(--assembly-header-height);
            }
            .assembly-audit-table td {
                text-align: left;
                line-height: 1.35;
                white-space: nowrap;
            }
            .assembly-audit-table td.assembly-numeric-cell {
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
            }
            .assembly-audit-table td.assembly-ratio-alert {
                background: #fff0f0 !important;
                color: #b42318;
                font-weight: 800 !important;
                box-shadow: inset 4px 0 0 #e74c3c;
            }
            .assembly-audit-table tr.assembly-parent-row td {
                background: #ffffff;
                font-weight: 600;
            }
            .assembly-audit-table tr.assembly-child-row td {
                background: #fafafa;
                color: #304255;
                font-weight: 400;
            }
            .assembly-audit-table td.assembly-child-code,
            .assembly-audit-table td.assembly-child-name {
                padding-left: 1.35rem;
            }
            .assembly-toggle-rail-spacer {
                height: 100%;
            }
            [data-testid="stColumn"]:has([class*="st-key-assembly_toggle_"]) [data-testid="stVerticalBlock"] {
                gap: 0 !important;
            }
            [data-testid="stColumn"]:has([class*="st-key-assembly_toggle_"]) [data-testid="stElementContainer"],
            [data-testid="stColumn"]:has([class*="st-key-assembly_toggle_"]) .element-container {
                margin: 0 !important;
                height: var(--assembly-row-height) !important;
                min-height: var(--assembly-row-height) !important;
            }
            [data-testid="stColumn"]:has([class*="st-key-assembly_toggle_"]) [data-testid="stElementContainer"]:has(.assembly-toggle-rail-spacer),
            [data-testid="stColumn"]:has([class*="st-key-assembly_toggle_"]) .element-container:has(.assembly-toggle-rail-spacer) {
                height: calc(var(--assembly-header-height) + 0.38rem) !important;
                min-height: calc(var(--assembly-header-height) + 0.38rem) !important;
            }
            [data-testid="stColumn"]:has([class*="st-key-assembly_toggle_"]) div[data-testid="stButton"] {
                height: var(--assembly-row-height) !important;
                min-height: var(--assembly-row-height) !important;
                margin: 0 !important;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            [class*="st-key-assembly_toggle_"] button[data-testid^="stBaseButton"] {
                width: 100% !important;
                height: var(--assembly-row-height) !important;
                min-height: var(--assembly-row-height) !important;
                padding: 0 !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                background: transparent !important;
                color: #1f2d3d !important;
                font-size: 0.78rem !important;
                line-height: 1 !important;
            }
            [class*="st-key-assembly_toggle_"] button[data-testid^="stBaseButton"]:hover,
            [class*="st-key-assembly_toggle_"] button[data-testid^="stBaseButton"]:focus {
                border: none !important;
                box-shadow: none !important;
                background: transparent !important;
                color: #2f6f4e !important;
            }
            .assembly-toggle-placeholder {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: var(--assembly-row-height);
                min-height: var(--assembly-row-height);
            }
            .assembly-pagination-status {
                width: 100%;
                min-height: 38px;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                color: #425466;
                font-size: 0.9rem;
                white-space: nowrap;
            }
            .assembly-tree-footer {
                width: 100%;
                margin-top: 0.55rem;
            }
            div[data-testid="stNumberInput"] button {
                display: none !important;
            }
            div[data-testid="stNumberInput"] {
                max-width: 80px !important;
            }
            div[data-testid="stNumberInput"] input {
                padding-right: 10px !important;
                text-align: center !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_folder_signature(folder_path: str) -> tuple[tuple[str, int, int], ...]:
    base_path = Path(folder_path)
    if not base_path.exists() or not base_path.is_dir():
        return tuple()

    signature_rows: list[tuple[str, int, int]] = []
    for file_path in sorted(base_path.rglob("*")):
        if not is_supported_assembly_excel_file(file_path):
            continue
        stat_result = file_path.stat()
        signature_rows.append(
            (
                str(file_path.relative_to(base_path)).replace("\\", "/"),
                int(stat_result.st_mtime_ns),
                int(stat_result.st_size),
            )
        )
    return tuple(signature_rows)


def _build_db_signature() -> tuple[float, int, int]:
    refresh_token = float(st.session_state.get("_local_db_refresh_token", 0.0) or 0.0)
    db_path = settings.db_path
    if not db_path.exists():
        return refresh_token, 0, 0

    stat_result = db_path.stat()
    return refresh_token, int(stat_result.st_mtime_ns), int(stat_result.st_size)


@st.cache_data
def _cached_load_assembly_bundle(
    folder_path: str,
    folder_signature: tuple[tuple[str, int, int], ...],
    db_signature: tuple[float, int, int],
    schema_version: str,
):
    return load_and_process_assembly_data(folder_path)


def _format_money(value) -> str:
    return "" if pd.isna(value) else f"{float(value):,.2f}"


def _format_ratio(value) -> str:
    return "" if pd.isna(value) else f"{float(value):.2%}"


def _format_text(value, empty_text: str = "-") -> str:
    text = str(value or "").strip()
    return text or empty_text


def _build_html_cell(
    value: str,
    *,
    tag: str = "td",
    classes: tuple[str, ...] = (),
    rowspan: int = 1,
    raw: bool = False,
    include_rowspan: bool = False,
) -> str:
    class_text = " ".join(class_name for class_name in classes if class_name)
    class_attr = f" class='{class_text}'" if class_text else ""
    try:
        normalized_rowspan = max(int(rowspan), 1)
    except (TypeError, ValueError):
        normalized_rowspan = 1
    rowspan_attr = f" rowspan='{normalized_rowspan}'" if include_rowspan or normalized_rowspan > 1 else ""

    if raw:
        content = value or "&nbsp;"
    else:
        text = "" if value is None else str(value)
        content = escape_html_text(text) if text.strip() else "&nbsp;"

    return f"<{tag}{class_attr}{rowspan_attr}>{content}</{tag}>"


def _log_assembly_tree_render_event(action: str, message: str, **details: object) -> None:
    log_event("assembly_tree", action, message, **details)
    if details:
        print(f"[assembly_tree] {action}: {message} | {details}")
    else:
        print(f"[assembly_tree] {action}: {message}")


def _render_tree_header() -> str:
    return "<thead><tr>" + "".join(_build_html_cell(header_name, tag="th") for header_name in TREE_HEADERS) + "</tr></thead>"


def _render_tree_row(parent_cells: str, child_rows: list[str] | None = None) -> str:
    rows = [f"<tr class='assembly-parent-row'>{parent_cells}</tr>"]
    rows.extend(child_rows or [])
    return "".join(rows)


def _get_expanded_parent_codes() -> set[str]:
    raw_value = st.session_state.get(ASSEMBLY_EXPANDED_CODES_KEY, set())
    if isinstance(raw_value, dict):
        normalized = {str(code) for code, is_open in raw_value.items() if is_open}
    elif isinstance(raw_value, (list, tuple, set)):
        normalized = {str(code) for code in raw_value if str(code).strip()}
    else:
        normalized = set()
    st.session_state[ASSEMBLY_EXPANDED_CODES_KEY] = set(normalized)
    return set(normalized)


def _set_expanded_parent_codes(expanded_codes: set[str]) -> None:
    st.session_state[ASSEMBLY_EXPANDED_CODES_KEY] = set(expanded_codes)


def _toggle_assembly_parent(parent_code: str) -> None:
    normalized_parent_code = str(parent_code or "").strip()
    if not normalized_parent_code:
        return

    expanded_codes = _get_expanded_parent_codes()
    if normalized_parent_code in expanded_codes:
        expanded_codes.remove(normalized_parent_code)
    else:
        expanded_codes.add(normalized_parent_code)
    _set_expanded_parent_codes(expanded_codes)


def _get_current_assembly_page(total_pages: int) -> int:
    try:
        current_page = int(st.session_state.get(ASSEMBLY_PAGE_KEY) or 1)
    except (TypeError, ValueError):
        current_page = 1

    normalized_page = min(max(current_page, 1), total_pages)
    st.session_state[ASSEMBLY_PAGE_KEY] = normalized_page
    if ASSEMBLY_PAGE_INPUT_KEY not in st.session_state or normalized_page != current_page:
        st.session_state[ASSEMBLY_PAGE_INPUT_KEY] = normalized_page
    return normalized_page


def _set_current_assembly_page(target_page: int, total_pages: int) -> int:
    normalized_page = min(max(int(target_page), 1), total_pages)
    st.session_state[ASSEMBLY_PAGE_KEY] = normalized_page
    st.session_state[ASSEMBLY_PAGE_INPUT_KEY] = normalized_page
    return normalized_page


def _handle_assembly_filter_change() -> None:
    st.session_state[ASSEMBLY_PAGE_KEY] = 1
    st.session_state[ASSEMBLY_PAGE_INPUT_KEY] = 1


def reset_assembly_filters() -> None:
    st.session_state[ASSEMBLY_PARENT_FILTER_KEY] = ""
    st.session_state[ASSEMBLY_CHILD_FILTER_KEY] = ""
    st.session_state[ASSEMBLY_OWNER_FILTER_KEY] = ""
    st.session_state[ASSEMBLY_PAGE_KEY] = 1
    st.session_state[ASSEMBLY_PAGE_INPUT_KEY] = 1


def _paginate_parent_rows(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    total_parents = int(len(summary_df))
    total_pages = max(1, (total_parents + ASSEMBLY_PAGE_SIZE - 1) // ASSEMBLY_PAGE_SIZE)
    current_page = _get_current_assembly_page(total_pages)
    page_start = (current_page - 1) * ASSEMBLY_PAGE_SIZE
    page_end = page_start + ASSEMBLY_PAGE_SIZE
    return summary_df.iloc[page_start:page_end].reset_index(drop=True), current_page, total_pages


def _build_detail_lookup(detail_df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    if detail_df.empty:
        return {}

    detail_lookup: dict[str, list[dict[str, str]]] = {}
    required_detail_columns = [
        "层级0编码",
        "层级1编码",
        "层级1名称",
        "层级1成本",
        "层级1订购价",
        "层级1零售价",
    ]
    optional_owner_columns = [
        "资源开发",
        "定价负责人",
    ]
    missing_columns = [column for column in required_detail_columns if column not in detail_df.columns]
    if missing_columns:
        _log_assembly_tree_render_event(
            "missing_detail_columns",
            "二级件明细缺少树表渲染所需字段。",
            missing_columns=missing_columns,
            detail_columns=list(detail_df.columns),
        )
        return {}

    render_df = detail_df[required_detail_columns].copy()
    for column_name in optional_owner_columns:
        render_df[column_name] = (
            detail_df[column_name] if column_name in detail_df.columns else pd.Series(["未配置"] * len(detail_df), index=detail_df.index)
        )

    for (
        parent_code,
        child_code,
        child_name,
        child_cost,
        child_order_price,
        child_retail_price,
        child_resource_owner,
        child_pricing_owner,
    ) in render_df[[*required_detail_columns, *optional_owner_columns]].itertuples(index=False, name=None):
        normalized_parent_code = str(parent_code or "").strip()
        detail_lookup.setdefault(normalized_parent_code, []).append(
            {
                "适用车系": "",
                "零件编码": f"└─ {_format_text(child_code)}",
                "零件名称": f"└─ {_format_text(child_name)}",
                "成本": _format_money(child_cost),
                "订购价": _format_money(child_order_price),
                "零售价": _format_money(child_retail_price),
                "资源开发": _format_text(child_resource_owner, empty_text="未配置"),
                "定价负责人": _format_text(child_pricing_owner, empty_text="未配置"),
                "成本比例": "",
                "订购价比例": "",
                "零售价比例": "",
            }
        )
    return detail_lookup


def _build_tree_render_groups(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    expanded_codes: set[str],
    current_page: int,
) -> list[dict[str, object]]:
    if summary_df.empty:
        _log_assembly_tree_render_event(
            "empty_summary_page",
            "分页后的一级件数据为空，未生成 HTML 树表。",
            current_page=current_page,
            detail_row_count=int(len(detail_df)),
        )
        return []

    missing_columns = [column for column in ASSEMBLY_RENDER_COLUMNS if column not in summary_df.columns]
    if missing_columns:
        _log_assembly_tree_render_event(
            "missing_summary_columns",
            "一级件汇总数据缺少树表渲染所需字段。",
            current_page=current_page,
            missing_columns=missing_columns,
            summary_columns=list(summary_df.columns),
        )
        return []

    detail_lookup = _build_detail_lookup(detail_df)
    render_groups: list[dict[str, object]] = []
    for row_index, (
        vehicle_series,
        parent_code,
        child_count,
        parent_name,
        parent_cost,
        parent_order_price,
        parent_retail_price,
        resource_owner,
        pricing_owner,
        cost_ratio,
        order_ratio,
        retail_ratio,
    ) in enumerate(summary_df[ASSEMBLY_RENDER_COLUMNS].itertuples(index=False, name=None), start=1):
        normalized_parent_code = str(parent_code or "").strip()
        child_rows = detail_lookup.get(normalized_parent_code, [])
        has_children = bool(child_rows)
        is_expanded = has_children and normalized_parent_code in expanded_codes
        rendered_children = child_rows if is_expanded else []
        render_groups.append(
            {
                "button_key": f"{row_index}_{normalized_parent_code or 'empty'}",
                "parent_code": normalized_parent_code,
                "child_count": _format_text(child_count, empty_text="0"),
                "vehicle_series": _format_text(vehicle_series, empty_text="未配置"),
                "parent_name": _format_text(parent_name),
                "parent_cost": _format_money(parent_cost),
                "parent_order_price": _format_money(parent_order_price),
                "parent_retail_price": _format_money(parent_retail_price),
                "resource_owner": _format_text(resource_owner),
                "pricing_owner": _format_text(pricing_owner),
                "cost_ratio": _format_ratio(cost_ratio),
                "cost_ratio_alert": pd.notna(cost_ratio) and float(cost_ratio) > 1.2,
                "order_ratio": _format_ratio(order_ratio),
                "retail_ratio": _format_ratio(retail_ratio),
                "has_children": has_children,
                "is_expanded": is_expanded,
                "rendered_children": rendered_children,
                "rowspan": max(1, 1 + len(rendered_children)),
            }
        )
    return render_groups


def _build_rowspan_tree_table_html(
    render_groups: list[dict[str, object]],
    current_page: int,
    detail_row_count: int,
) -> str:
    if not render_groups:
        return ""
    body_rows: list[str] = []
    current_parent_code = ""

    try:
        for group in render_groups:
            current_parent_code = str(group.get("parent_code") or "").strip() or f"<empty-parent-{len(body_rows) + 1}>"
            parent_rowspan = max(int(group.get("rowspan") or 1), 1)
            rendered_children = list(group.get("rendered_children") or [])
            parent_cells = (
                _build_html_cell(str(group.get("vehicle_series") or ""), rowspan=parent_rowspan, include_rowspan=True)
                + _build_html_cell(str(group.get("parent_code") or ""))
                + _build_html_cell(str(group.get("child_count") or ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(str(group.get("parent_name") or ""))
                + _build_html_cell(str(group.get("parent_cost") or ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(str(group.get("parent_order_price") or ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(str(group.get("parent_retail_price") or ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(str(group.get("resource_owner") or ""))
                + _build_html_cell(str(group.get("pricing_owner") or ""))
                + _build_html_cell(
                    str(group.get("cost_ratio") or ""),
                    classes=("assembly-numeric-cell", "assembly-ratio-alert" if bool(group.get("cost_ratio_alert")) else ""),
                    rowspan=parent_rowspan,
                    include_rowspan=True,
                )
                + _build_html_cell(str(group.get("order_ratio") or ""), classes=("assembly-numeric-cell",), rowspan=parent_rowspan, include_rowspan=True)
                + _build_html_cell(str(group.get("retail_ratio") or ""), classes=("assembly-numeric-cell",), rowspan=parent_rowspan, include_rowspan=True)
            )
            child_row_html = [
                "<tr class='assembly-child-row'>"
                + _build_html_cell(child_row.get("零件编码", ""), classes=("assembly-child-code",))
                + _build_html_cell("")
                + _build_html_cell(child_row.get("零件名称", ""), classes=("assembly-child-name",))
                + _build_html_cell(child_row.get("成本", ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(child_row.get("订购价", ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(child_row.get("零售价", ""), classes=("assembly-numeric-cell",))
                + _build_html_cell(child_row.get("资源开发", ""))
                + _build_html_cell(child_row.get("定价负责人", ""))
                + "</tr>"
                for child_row in rendered_children
            ]
            body_rows.append(_render_tree_row(parent_cells, child_row_html))
    except Exception as exc:
        _log_assembly_tree_render_event(
            "build_failed",
            "HTML 树表生成中断。",
            current_page=current_page,
            parent_code=current_parent_code,
            summary_row_count=int(len(render_groups)),
            detail_row_count=detail_row_count,
            error=repr(exc),
        )
        return ""

    html = (
        "<div class='assembly-table-shell'><table class='assembly-audit-table'>"
        + _render_tree_header()
        + "<tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )
    if not body_rows or "<table" not in html or "<tbody>" not in html:
        _log_assembly_tree_render_event(
            "empty_html",
            "HTML 树表生成结果为空或结构不完整。",
            current_page=current_page,
            parent_code=current_parent_code,
            summary_row_count=int(len(render_groups)),
            detail_row_count=detail_row_count,
            body_row_count=len(body_rows),
            html_preview=html[:240],
        )
        return ""
    return html


def _render_toggle_placeholder() -> None:
    st.markdown("<div class='assembly-toggle-placeholder'>&nbsp;</div>", unsafe_allow_html=True)


def _render_assembly_toggle_buttons(render_groups: list[dict[str, object]]) -> None:
    st.markdown("<div class='assembly-toggle-rail-spacer'></div>", unsafe_allow_html=True)
    for group in render_groups:
        if bool(group.get("has_children")):
            is_expanded = bool(group.get("is_expanded"))
            st.button(
                "▼" if is_expanded else "▶",
                key=f"assembly_toggle_{group.get('button_key')}",
                help="收起二级件" if is_expanded else "展开二级件",
                type="secondary",
                width="stretch",
                on_click=_toggle_assembly_parent,
                args=(str(group.get("parent_code") or ""),),
            )
        else:
            _render_toggle_placeholder()

        for _ in list(group.get("rendered_children") or []):
            _render_toggle_placeholder()


def _handle_assembly_page_jump(total_pages: int) -> None:
    jump_page = int(st.session_state.get(ASSEMBLY_PAGE_INPUT_KEY) or 1)
    st.session_state[ASSEMBLY_PAGE_KEY] = min(max(jump_page, 1), total_pages)


def _render_pagination_controls(current_page: int, total_pages: int) -> None:
    footer_container = st.container()
    with footer_container:
        st.markdown("<div class='assembly-tree-footer'></div>", unsafe_allow_html=True)
        prev_col, next_col, status_col, jump_col = st.columns([1, 1, 2.6, 0.65], gap="small", vertical_alignment="center")

        with prev_col:
            if st.button(
                "上一页",
                key="assembly_page_prev",
                type="secondary",
                width="stretch",
                disabled=current_page <= 1,
            ):
                _set_current_assembly_page(current_page - 1, total_pages)
                st.rerun()

        with next_col:
            if st.button(
                "下一页",
                key="assembly_page_next",
                type="secondary",
                width="stretch",
                disabled=current_page >= total_pages,
            ):
                _set_current_assembly_page(current_page + 1, total_pages)
                st.rerun()

        with status_col:
            st.markdown(
                f"<div class='assembly-pagination-status'>当前第 {current_page} 页 / 共 {total_pages} 页</div>",
                unsafe_allow_html=True,
            )

        with jump_col:
            st.number_input(
                "跳转页码",
                min_value=1,
                max_value=total_pages,
                step=1,
                key=ASSEMBLY_PAGE_INPUT_KEY,
                label_visibility="collapsed",
                help=f"输入 1 到 {total_pages} 的页码并回车跳转",
                on_change=_handle_assembly_page_jump,
                args=(total_pages,),
            )


def render_assembly_audit_page() -> None:
    inject_css(is_overview=False)
    _inject_assembly_table_css()
    st.title("🔩 拆分件成本监控")

    folder_path = get_path_setting("assembly_data_path")
    if not folder_path:
        st.info("请先前往“⚙️ 系统设置”配置一级件明细数据路径。")
        return

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        st.info(f"当前一级件明细数据路径不存在，请先到“⚙️ 系统设置”修正路径：{folder_path}")
        return

    folder_signature = _build_folder_signature(folder_path)
    db_signature = _build_db_signature()
    with st.spinner("正在加载一级件明细并执行跨表成本补全..."):
        payload = _cached_load_assembly_bundle(
            folder_path,
            folder_signature,
            db_signature,
            ASSEMBLY_DATA_SCHEMA_VERSION,
        )

    error_message = payload.get("error_message")
    if error_message:
        st.warning(error_message)
        return

    info_message = str(payload.get("info_message") or "").strip()

    warnings = payload.get("warnings", []) or []
    for warning_message in warnings:
        st.warning(warning_message)

    summary_df = payload.get("summary_df", pd.DataFrame())
    detail_df = payload.get("detail_df", pd.DataFrame())
    export_df = payload.get("export_df", pd.DataFrame())
    stats = payload.get("stats", {}) or {}

    if summary_df.empty:
        st.info(info_message or "当前一级件映射数据暂无可审计记录。")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("一级件数", f"{int(stats.get('层级0数', 0))}")
    m2.metric("二级件数", f"{int(stats.get('层级1数', 0))}")
    m3.metric("一级件成本补齐", f"{int(stats.get('自动补齐层级0成本行数', 0))}")
    m4.metric("二级件成本补齐", f"{int(stats.get('自动补齐层级1成本行数', 0))}")
    m5.metric("仍缺失成本", f"{int(stats.get('仍缺失成本行数', 0))}")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 0.4], gap="medium", vertical_alignment="bottom")
    with filter_col1:
        assembly_code_keyword = st.text_input(
            "一级件编码",
            key=ASSEMBLY_PARENT_FILTER_KEY,
            placeholder="输入一级件编码关键字",
            on_change=_handle_assembly_filter_change,
        )
    with filter_col2:
        child_code_keyword = st.text_input(
            "二级件编码",
            key=ASSEMBLY_CHILD_FILTER_KEY,
            placeholder="输入二级件编码关键字",
            on_change=_handle_assembly_filter_change,
        )
    with filter_col3:
        owner_keyword = st.text_input(
            "定价负责人",
            key=ASSEMBLY_OWNER_FILTER_KEY,
            placeholder="输入定价负责人关键字",
            on_change=_handle_assembly_filter_change,
        )
    with filter_col4:
        st.button(
            "🔄 重置",
            key="assembly_reset_filters",
            width="stretch",
            on_click=reset_assembly_filters,
        )

    filtered_summary = summary_df.copy()
    if assembly_code_keyword:
        filtered_summary = filtered_summary[
            filtered_summary["层级0编码"].astype(str).str.contains(assembly_code_keyword, case=False, na=False)
        ]
    if child_code_keyword:
        matched_parent_codes = set(
            detail_df[
                detail_df["层级1编码"].astype(str).str.contains(child_code_keyword, case=False, na=False)
            ]["层级0编码"].astype(str).str.strip().tolist()
        )
        filtered_summary = filtered_summary[
            filtered_summary["层级0编码"].astype(str).str.strip().isin(matched_parent_codes)
        ]
    if owner_keyword:
        filtered_summary = filtered_summary[
            filtered_summary["定价负责人"].astype(str).str.contains(owner_keyword, case=False, na=False)
        ]
    filtered_summary = sort_assembly_summary_by_cost_ratio(filtered_summary)

    visible_parent_codes = filtered_summary["层级0编码"].astype(str).str.strip().tolist()
    filtered_detail = detail_df[detail_df["层级0编码"].astype(str).str.strip().isin(visible_parent_codes)].copy()
    filtered_export = export_df[export_df["层级0编码"].astype(str).str.strip().isin(visible_parent_codes)].copy()
    abnormal_export = build_abnormal_ratio_export_df(filtered_summary, filtered_export)
    abnormal_ratio_count = int(pd.to_numeric(filtered_summary.get("成本比例"), errors="coerce").gt(1.2).sum()) if "成本比例" in filtered_summary.columns else 0

    summary_col, export_col, abnormal_export_col = st.columns([4.4, 1.2, 1.2], gap="medium", vertical_alignment="center")
    with summary_col:
        st.markdown(f"**当前筛选后共 {len(filtered_summary)} 个一级件**")
        st.caption(f"成本比例默认按降序排列；当前超过 1.2 的一级件共 {abnormal_ratio_count} 个，已在表格中用重点色标注。")
    with export_col:
        export_bytes = to_excel_bytes(filtered_export) if not filtered_export.empty else b""
        st.download_button(
            "📥 导出分级平铺报表",
            data=export_bytes,
            file_name=f"层级映射审计_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            disabled=filtered_export.empty,
        )
    with abnormal_export_col:
        abnormal_export_bytes = to_excel_bytes(abnormal_export) if not abnormal_export.empty else b""
        st.download_button(
            "📥 导出异常比例",
            data=abnormal_export_bytes,
            file_name=f"拆分件异常比例_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            disabled=abnormal_export.empty,
        )

    if filtered_summary.empty:
        _set_current_assembly_page(1, 1)
        st.info("当前筛选条件下没有匹配的一级件记录。")
        return

    paged_summary, current_page, total_pages = _paginate_parent_rows(filtered_summary)
    paged_parent_codes = paged_summary["层级0编码"].astype(str).str.strip().tolist()
    paged_detail = filtered_detail[filtered_detail["层级0编码"].astype(str).str.strip().isin(paged_parent_codes)].copy()

    expanded_codes = _get_expanded_parent_codes()
    render_groups = _build_tree_render_groups(paged_summary, paged_detail, expanded_codes, current_page)

    st.caption("点击每行左侧箭头展开或收起二级件明细。")
    rowspan_table_html = _build_rowspan_tree_table_html(render_groups, current_page, int(len(paged_detail)))
    if "<table" not in rowspan_table_html or "rowspan" not in rowspan_table_html:
        _log_assembly_tree_render_event(
            "invalid_html",
            "render_assembly_audit_page 收到了无效的 HTML rowspan tree table。",
            current_page=current_page,
            paged_parent_codes=paged_parent_codes,
            expanded_codes=sorted(expanded_codes),
            render_group_count=len(render_groups),
            html_length=len(rowspan_table_html),
            html_preview=rowspan_table_html[:240],
        )
        st.error("拆分件树表渲染失败：未生成有效的 HTML rowspan tree table。")
        return

    toggle_col, table_col = st.columns(ASSEMBLY_TOGGLE_LAYOUT, gap="small", vertical_alignment="top")
    with toggle_col:
        _render_assembly_toggle_buttons(render_groups)
    with table_col:
        st.write(rowspan_table_html, unsafe_allow_html=True)

    _render_pagination_controls(current_page, total_pages)
