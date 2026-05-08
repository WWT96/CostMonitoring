import html
from typing import List

import pandas as pd


BASE_COLS = ["物料编码", "物料名称", "适用车系", "备件简称", "工厂"]


def escape_html_text(value) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value), quote=True)


def render_merged_html_table(df: pd.DataFrame, value_cols: List[str], is_trend_mode: bool = False) -> str:
    if df.empty:
        return "<div style='text-align:center; padding: 20px;'>暂无数据</div>"

    all_cols = BASE_COLS + value_cols
    html_parts = [
        """
        <div style="width: 100%; overflow-x: auto; margin-bottom: 20px;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 14px; border: 1px solid #dee2e6;">
                <thead>
                    <tr style="background-color: #f8f9fa; color: #495057;">
        """
    ]

    for col in all_cols:
        html_parts.append(
            f'<th style="padding: 12px; border: 1px solid #dee2e6; text-align: center !important; vertical-align: middle !important; font-weight: 600; white-space: nowrap;">{escape_html_text(col)}</th>'
        )

    html_parts.append("</tr></thead><tbody>")

    grouped = df.sort_values(["物料编码", "工厂"]).groupby("物料编码", sort=False)
    for _, group in grouped:
        row_count = len(group)
        first_row = True
        for _, row in group.iterrows():
            html_parts.append(
                '<tr style="background-color: #ffffff; transition: background-color 0.1s ease;" onmouseover="this.style.backgroundColor=\'#f1f3f5\'" onmouseout="this.style.backgroundColor=\'#ffffff\'">'
            )
            for col_idx, col_name in enumerate(all_cols):
                val = row.get(col_name, "")
                if col_idx < 4:
                    if first_row:
                        val_text = escape_html_text(val)
                        html_parts.append(
                            f'<td rowspan="{row_count}" style="padding: 10px; border: 1px solid #dee2e6; text-align: center !important; vertical-align: middle !important; background-color: #fff;">{val_text}</td>'
                        )
                    continue

                cell_style = (
                    "padding: 8px; border: 1px solid #dee2e6; text-align: center !important; vertical-align: middle !important;"
                )
                display_val = ""

                if pd.isna(val) or val == "":
                    display_val = ""
                elif isinstance(val, (int, float)):
                    display_val = f"{val:,.2f}"
                    if is_trend_mode and col_name in value_cols:
                        if val > 0:
                            cell_style += " color: #e74c3c; font-weight: bold;"
                            display_val = f"▲ {display_val}"
                        elif val < 0:
                            cell_style += " color: #27ae60; font-weight: bold;"
                            display_val = f"▼ {display_val}"
                        else:
                            cell_style += " color: #2c3e50;"
                            display_val = "-"
                else:
                    display_val = str(val)

                html_parts.append(f'<td style="{cell_style}">{escape_html_text(display_val)}</td>')
            html_parts.append("</tr>")
            first_row = False

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