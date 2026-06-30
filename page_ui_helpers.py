import re
from collections.abc import MutableMapping
from typing import Callable

import pandas as pd
import streamlit as st

DEFERRED_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024


def get_table_height(row_count: int, min_height: int = 180, max_height: int = 560) -> int:
    return max(min_height, min(max_height, 38 + 35 * max(row_count, 1)))


def get_vehicle_market_price_manual_editable_columns() -> list[str]:
    return ["梯度排名", "次顶配车型", "估算价格（元）"]


def dataframe_export_fingerprint(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df is None:
        return "none"
    selected_columns = [column for column in (columns or df.columns.tolist()) if column in df.columns]
    index_preview = [str(value) for value in list(df.index[:5]) + list(df.index[-5:])]
    return f"rows={len(df)}|cols={','.join(map(str, selected_columns))}|idx={','.join(index_preview)}"


def _deferred_download_state_keys(key: str) -> tuple[str, str, str]:
    return (
        f"_deferred_download_payload_{key}",
        f"_deferred_download_file_{key}",
        f"_deferred_download_error_{key}",
    )


def clear_deferred_download_payload(state: MutableMapping, key: str) -> None:
    for state_key in _deferred_download_state_keys(key):
        state.pop(state_key, None)


def store_deferred_download_payload(
    state: MutableMapping,
    *,
    key: str,
    payload: bytes,
    file_name: str,
    max_bytes: int = DEFERRED_DOWNLOAD_MAX_BYTES,
) -> None:
    payload_key, file_key, error_key = _deferred_download_state_keys(key)
    clear_deferred_download_payload(state, key)
    payload_size = len(payload or b"")
    if payload_size > max_bytes:
        raise ValueError(f"导出文件 {payload_size / 1024 / 1024:.2f} MB，超过 {max_bytes / 1024 / 1024:.2f} MB 上限")
    state[payload_key] = payload
    state[file_key] = file_name
    state[error_key] = ""


def render_deferred_download_button(
    *,
    label: str,
    prepare_label: str,
    data_builder: Callable[[], bytes],
    file_name: str,
    mime: str,
    key: str,
    fingerprint: str = "",
    max_bytes: int = DEFERRED_DOWNLOAD_MAX_BYTES,
    width: str | None = "stretch",
) -> None:
    payload_key, file_key, error_key = _deferred_download_state_keys(key)
    fingerprint_key = f"_deferred_download_fingerprint_{key}"

    if st.session_state.get(fingerprint_key) != fingerprint:
        clear_deferred_download_payload(st.session_state, key)
        st.session_state[fingerprint_key] = fingerprint

    if st.button(prepare_label, key=f"{key}_prepare", width=width):
        try:
            store_deferred_download_payload(
                st.session_state,
                key=key,
                payload=data_builder(),
                file_name=file_name,
                max_bytes=max_bytes,
            )
        except Exception as exc:
            st.session_state[error_key] = f"导出失败: {exc}"

    error_message = st.session_state.get(error_key)
    if error_message:
        st.error(error_message)
        return

    payload = st.session_state.get(payload_key)
    if payload:
        downloaded = st.download_button(
            label,
            data=payload,
            file_name=st.session_state.get(file_key) or file_name,
            mime=mime,
            width=width,
        )
        if downloaded:
            clear_deferred_download_payload(st.session_state, key)


def inject_center_aligned_table_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="gridcell"],
        [data-testid="stDataFrame"] [role="columnheader"] > div,
        [data-testid="stDataFrame"] [role="gridcell"] > div {
            justify-content: center !important;
            text-align: center !important;
            align-items: center !important;
        }
        [data-testid="stDataFrame"] [data-testid="stMarkdownContainer"],
        [data-testid="stDataFrame"] p {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def prepare_table_view(
    source_df: pd.DataFrame,
    key_prefix: str,
    *,
    display_columns: list[str] | None = None,
    default_search_columns: list[str] | None = None,
    locked_columns: list[str] | None = None,
    filter_title: str = "当前表格",
) -> tuple[pd.DataFrame, list[str]]:
    allowed_columns = [
        column_name
        for column_name in (display_columns or source_df.columns.tolist())
        if column_name in source_df.columns
    ]
    if not allowed_columns:
        return source_df.copy(), []
    if source_df.empty:
        return source_df.copy(), allowed_columns

    default_search_columns = [
        column_name
        for column_name in (default_search_columns or allowed_columns[: min(4, len(allowed_columns))])
        if column_name in allowed_columns
    ] or allowed_columns[: min(4, len(allowed_columns))]
    locked_columns = [column_name for column_name in (locked_columns or []) if column_name in allowed_columns]

    filter_priority = [
        "status",
        "结论状态",
        "工厂",
        "备件简称",
        "适用车系",
        "当前标注",
        "专家反馈",
        "最终优化结论",
        "供应商代码",
    ]
    filter_candidates: list[str] = []
    seen_columns: set[str] = set()
    for column_name in filter_priority + allowed_columns:
        if column_name in seen_columns or column_name not in allowed_columns:
            continue
        unique_count = source_df[column_name].dropna().astype(str).nunique()
        if 1 < unique_count <= 40:
            filter_candidates.append(column_name)
            seen_columns.add(column_name)
        if len(filter_candidates) >= 4:
            break

    category_filters: dict[str, list[str]] = {}
    with st.expander(f"🔎 表格筛选与列显示 · {filter_title}", expanded=False):
        filter_c1, filter_c2 = st.columns([2, 2])
        with filter_c1:
            search_columns = st.multiselect(
                "搜索列",
                options=allowed_columns,
                default=default_search_columns,
                key=f"{key_prefix}_search_columns",
            )
        with filter_c2:
            keyword = st.text_input(
                "关键词搜索",
                key=f"{key_prefix}_keyword",
                placeholder="支持空格分隔多关键词，例如 远景Max 异常偏高",
            )

        selected_visible_columns = st.multiselect(
            "显示列",
            options=allowed_columns,
            default=allowed_columns,
            key=f"{key_prefix}_visible_columns",
        )

        if filter_candidates:
            filter_columns = st.columns(len(filter_candidates))
            for idx, column_name in enumerate(filter_candidates):
                options = sorted(source_df[column_name].dropna().astype(str).unique().tolist())
                with filter_columns[idx]:
                    category_filters[column_name] = st.multiselect(
                        f"{column_name} 筛选",
                        options=options,
                        key=f"{key_prefix}_filter_{column_name}",
                    )
        else:
            keyword = st.session_state.get(f"{key_prefix}_keyword", "")
            search_columns = st.session_state.get(f"{key_prefix}_search_columns", default_search_columns)

    filtered_df = source_df.copy()
    search_columns = [column_name for column_name in search_columns if column_name in filtered_df.columns]
    if keyword and search_columns:
        combined_text = filtered_df[search_columns].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
        search_mask = pd.Series(True, index=filtered_df.index)
        for token in [token.strip().lower() for token in keyword.split() if token.strip()]:
            search_mask &= combined_text.str.contains(re.escape(token), regex=True, na=False)
        filtered_df = filtered_df[search_mask]

    for column_name, selected_values in category_filters.items():
        if selected_values:
            filtered_df = filtered_df[
                filtered_df[column_name].fillna("").astype(str).isin(selected_values)
            ]

    visible_columns = [column_name for column_name in selected_visible_columns if column_name in allowed_columns]
    if locked_columns:
        visible_columns = locked_columns + [
            column_name for column_name in visible_columns if column_name not in locked_columns
        ]
    visible_columns = visible_columns or allowed_columns
    return filtered_df, visible_columns


def build_table_column_config(
    display_df: pd.DataFrame,
    *,
    overrides: dict[str, object] | None = None,
    editable_columns: list[str] | None = None,
) -> dict[str, object]:
    overrides = overrides or {}
    editable_set = set(editable_columns or [])
    percent_columns = {"偏离比例", "可信度"}
    money_like_columns = {
        "最新成本",
        "实际成本",
        "预测值",
        "合理下限",
        "合理上限",
        "偏离数值",
        "成本数值",
        "一级总成成本",
        "子零件加权总和",
        "测算总成成本",
        "基准价",
    }
    integer_like_columns = {"样本量", "子零件数量"}
    named_large_text_columns = {"AI 辅助分析", "AI辅助分析", "经验规律", "标注备注", "专家备注", "判定依据"}

    final_config: dict[str, object] = {}
    for column_name in display_df.columns:
        if column_name in overrides:
            final_config[column_name] = overrides[column_name]
            continue

        disabled = editable_columns is None or column_name not in editable_set
        series = display_df[column_name]
        sample_lengths = series.dropna().astype(str).str.len() if not series.empty else pd.Series(dtype=int)
        is_large_text = column_name in named_large_text_columns or (not sample_lengths.empty and sample_lengths.max() > 36)

        if pd.api.types.is_bool_dtype(series):
            final_config[column_name] = st.column_config.CheckboxColumn(column_name, disabled=disabled)
        elif column_name == "测算比值":
            final_config[column_name] = st.column_config.TextColumn(column_name, disabled=disabled)
        elif column_name in percent_columns:
            final_config[column_name] = st.column_config.NumberColumn(column_name, disabled=disabled, format="%.2%%")
        elif column_name in integer_like_columns or pd.api.types.is_integer_dtype(series):
            final_config[column_name] = st.column_config.NumberColumn(column_name, disabled=disabled, format="%d")
        elif column_name in money_like_columns or pd.api.types.is_float_dtype(series):
            final_config[column_name] = st.column_config.NumberColumn(column_name, disabled=disabled, format="%.2f")
        else:
            final_config[column_name] = st.column_config.TextColumn(
                column_name,
                disabled=disabled,
                width="large" if is_large_text else "medium",
            )
    return final_config


def render_standard_data_editor(
    display_df: pd.DataFrame,
    key_prefix: str,
    *,
    editable_columns: list[str] | None = None,
    column_config: dict[str, object] | None = None,
    max_height: int = 560,
):
    final_config = build_table_column_config(
        display_df,
        overrides=column_config,
        editable_columns=editable_columns,
    )
    if editable_columns is None:
        disabled_setting: bool | list[str] = True
    else:
        editable_set = set(editable_columns)
        disabled_setting = [column_name for column_name in display_df.columns if column_name not in editable_set]

    return st.data_editor(
        display_df,
        column_config=final_config,
        disabled=disabled_setting,
        hide_index=True,
        width="stretch",
        height=get_table_height(len(display_df), max_height=max_height),
        key=f"{key_prefix}_grid",
    )
