import html
from typing import List

import pandas as pd


MERGE_GROUP_COLS = ["物料编码", "物料名称", "适用车系", "备件简称"]
BASE_COLS = [*MERGE_GROUP_COLS, "工厂"]


def escape_html_text(value) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value), quote=True)


def _coerce_table_number(value):
    if isinstance(value, bool) or pd.isna(value) or value == "":
        return None
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return None
    return float(numeric_value)


def _format_table_value(
    value,
    column_name: str,
    value_cols: List[str],
    is_trend_mode: bool,
    comparison_delta: float | None = None,
) -> tuple[str, str]:
    cell_style = ""
    if pd.isna(value) or value == "":
        return "", cell_style

    numeric_value = _coerce_table_number(value)
    if numeric_value is not None:
        display_text = f"{numeric_value:,.2f}"
        if comparison_delta is not None and column_name.startswith("价格变动"):
            if comparison_delta > 0:
                cell_style = " color: #e74c3c; font-weight: 700;"
                display_text = f"{display_text}（▲ +{comparison_delta:,.2f}）"
            elif comparison_delta < 0:
                cell_style = " color: #27ae60; font-weight: 700;"
                display_text = f"{display_text}（▼ {comparison_delta:,.2f}）"
            else:
                display_text = f"{display_text}（0.00）"
        elif is_trend_mode and column_name in value_cols:
            if value > 0:
                cell_style = " color: #e74c3c; font-weight: 700;"
                display_text = f"▲ {display_text}"
            elif value < 0:
                cell_style = " color: #27ae60; font-weight: 700;"
                display_text = f"▼ {display_text}"
            else:
                cell_style = " color: #2c3e50;"
                display_text = "-"
        return display_text, cell_style

    return str(value), cell_style


def render_merged_html_table(
    df: pd.DataFrame,
    value_cols: List[str],
    is_trend_mode: bool = False,
    preserve_order: bool = False,
) -> str:
    if df.empty:
        return "<div style='text-align:center; padding: 20px;'>暂无数据</div>"

    all_cols = [column_name for column_name in [*BASE_COLS, *value_cols] if column_name in df.columns]
    group_cols = [column_name for column_name in MERGE_GROUP_COLS if column_name in all_cols]
    detail_cols = [column_name for column_name in all_cols if column_name not in group_cols]

    working_df = df[all_cols].copy()
    sort_columns = [column_name for column_name in [*group_cols, "工厂"] if column_name in working_df.columns]
    if sort_columns and not preserve_order:
        working_df = working_df.sort_values(sort_columns, kind="stable")

    if group_cols:
        rowspan_df = (
            working_df.groupby(group_cols, sort=False, dropna=False)
            .size()
            .rename("__rowspan__")
            .reset_index()
        )
        working_df = working_df.merge(rowspan_df, on=group_cols, how="left")
    else:
        working_df["__rowspan__"] = 1

    render_columns = [*all_cols, "__rowspan__"]
    column_index = {column_name: idx for idx, column_name in enumerate(render_columns)}

    html_parts = [
        """
        <style>
            .merged-report-table {
                width: 100%;
                border-collapse: collapse;
                table-layout: auto;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                border: 1px solid #dee2e6;
            }
            .merged-report-table th,
            .merged-report-table td {
                text-align: center !important;
                vertical-align: middle !important;
                border: 1px solid #dee2e6;
            }
            .merged-report-table th {
                padding: 12px;
                background-color: #f8f9fa;
                color: #495057;
                font-weight: 600;
                white-space: nowrap;
            }
            .merged-report-table td {
                padding: 10px 8px;
                background-color: #ffffff;
            }
        </style>
        <div style="width: 100%; overflow-x: auto; margin-bottom: 20px;">
            <table class="merged-report-table">
                <thead>
                    <tr>
        """
    ]

    for col in all_cols:
        html_parts.append(f"<th>{escape_html_text(col)}</th>")

    html_parts.append("</tr></thead><tbody>")

    last_group_key = None
    for row in working_df[render_columns].itertuples(index=False, name=None):
        current_group_key = tuple(row[column_index[column_name]] for column_name in group_cols)
        rowspan = int(row[column_index["__rowspan__"]] or 1)

        html_parts.append("<tr>")
        if current_group_key != last_group_key:
            for column_name in group_cols:
                cell_value = escape_html_text(row[column_index[column_name]])
                html_parts.append(f'<td rowspan="{rowspan}">{cell_value}</td>')

        for column_name in detail_cols:
            comparison_delta = None
            if column_name.startswith("价格变动"):
                try:
                    current_idx = int(column_name.replace("价格变动", ""))
                except ValueError:
                    current_idx = 0
                previous_col = f"价格变动{current_idx - 1}"
                if current_idx > 1 and previous_col in column_index:
                    current_value = pd.to_numeric(pd.Series([row[column_index[column_name]]]), errors="coerce").iloc[0]
                    previous_value = pd.to_numeric(pd.Series([row[column_index[previous_col]]]), errors="coerce").iloc[0]
                    if pd.notna(current_value) and pd.notna(previous_value):
                        comparison_delta = float(current_value - previous_value)
            display_val, extra_style = _format_table_value(
                row[column_index[column_name]],
                column_name,
                value_cols,
                is_trend_mode,
                comparison_delta,
            )
            style_attr = f' style="{extra_style.strip()}"' if extra_style.strip() else ""
            html_parts.append(f"<td{style_attr}>{escape_html_text(display_val)}</td>")

        html_parts.append("</tr>")
        last_group_key = current_group_key

    html_parts.append("</tbody></table></div>")
    return "".join(html_parts)


def render_center_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div style='text-align:center; padding: 20px;'>暂无数据</div>"

    html_parts = [
        """
        <div style="width: 100%; overflow-x: auto; margin-bottom: 20px;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 14px; border: 1px solid #dee2e6;">
                <thead>
                    <tr style="background-color: #f8f9fa; color: #495057;">
        """
    ]
    for col in df.columns:
        html_parts.append(
            f'<th style="padding: 12px; border: 1px solid #dee2e6; text-align: center !important; font-weight: 600;">{escape_html_text(col)}</th>'
        )
    html_parts.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        html_parts.append(
            '<tr style="background-color: #ffffff; transition: background-color 0.1s ease;" onmouseover="this.style.backgroundColor=\'#f1f3f5\'" onmouseout="this.style.backgroundColor=\'#ffffff\'">'
        )
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                show_val = ""
            elif col == "最新成本" and isinstance(val, (int, float)):
                show_val = f"{val:,.2f}"
            else:
                show_val = str(val)
            html_parts.append(
                f'<td style="padding: 10px; border: 1px solid #dee2e6; text-align: center !important;">{escape_html_text(show_val)}</td>'
            )
        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div>")
    return "".join(html_parts)
