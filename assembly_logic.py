from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import streamlit as st

import harness


SUPPORTED_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".xlsb"}
REQUIRED_LAYER_COLUMNS_MESSAGE = "请确保文件包含层级1编码、层级1名称、层级2编码、层级2名称"

RELATION_COLUMNS = ["层级0编码", "层级0名称", "层级1编码", "层级1名称"]

SUMMARY_COLUMNS = [
    "适用车系",
    "层级0编码",
    "子物料数量",
    "层级0名称",
    "层级0成本",
    "层级1成本",
    "资源开发",
    "定价负责人",
    "成本比例",
    "订购价比例",
    "零售价比例",
    "层级0订购价",
    "层级0订购价来源",
    "层级0零售价",
    "层级0零售价来源",
    "层级1订购价汇总",
    "层级1零售价汇总",
]

DETAIL_COLUMNS = [
    "层级0编码",
    "层级1编码",
    "层级1名称",
    "层级1成本",
    "层级1订购价",
    "层级1零售价",
    "资源开发",
    "定价负责人",
]

EXPORT_COLUMNS = [
    "层级",
    "适用车系",
    "层级0编码",
    "子物料数量",
    "层级0名称",
    "层级0成本",
    "层级1成本",
    "资源开发",
    "定价负责人",
    "成本比例",
    "订购价比例",
    "零售价比例",
    "层级1编码",
    "层级1名称",
    "层级1订购价",
    "层级1零售价",
    "层级0订购价",
    "层级0订购价来源",
    "层级0零售价",
    "层级0零售价来源",
]

RELATION_COLUMN_ALIASES: Dict[str, list[str]] = {
    "层级0编码": ["层级1编码"],
    "层级0名称": ["层级1名称"],
    "层级1编码": ["层级2编码"],
    "层级1名称": ["层级2名称"],
}


def _empty_payload(
    error_message: str | None = None,
    warnings: list[str] | None = None,
    info_message: str | None = None,
) -> dict[str, Any]:
    return {
        "summary_df": pd.DataFrame(columns=SUMMARY_COLUMNS),
        "detail_df": pd.DataFrame(columns=DETAIL_COLUMNS),
        "export_df": pd.DataFrame(columns=EXPORT_COLUMNS),
        "stats": {
            "层级0数": 0,
            "层级1数": 0,
            "自动补齐层级0成本行数": 0,
            "自动补齐层级1成本行数": 0,
            "仍缺失成本行数": 0,
            "源文件数": 0,
        },
        "error_message": error_message,
        "warnings": warnings or [],
        "info_message": info_message,
    }


def _normalize_column_name(column_name: Any) -> str:
    text = str(column_name or "").strip().lower()
    return (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("\n", "")
        .replace("\r", "")
        .replace("\t", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _first_matching_column(columns: Sequence[Any], aliases: Sequence[str]) -> str | None:
    column_lookup = {_normalize_column_name(column_name): str(column_name) for column_name in columns}
    for alias in aliases:
        matched = column_lookup.get(_normalize_column_name(alias))
        if matched:
            return matched
    return None


def _clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.replace({"nan": "", "None": "", "NaT": ""})


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.replace(",", "", regex=False).str.strip()
    normalized = normalized.mask(normalized.isin({"", "nan", "None", "-"}))
    return pd.to_numeric(normalized, errors="coerce")


def is_supported_assembly_excel_file(file_path: Path) -> bool:
    return (
        file_path.is_file()
        and not file_path.name.startswith("~$")
        and file_path.suffix.lower() in SUPPORTED_EXCEL_SUFFIXES
    )


def _list_assembly_files(folder_path: str) -> list[Path]:
    base_path = Path(folder_path)
    return [
        file_path
        for file_path in sorted(base_path.rglob("*"))
        if is_supported_assembly_excel_file(file_path)
    ]


def _load_relationship_rows(file_paths: Sequence[Path]) -> tuple[pd.DataFrame, list[str], bool]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    contract_detected = False

    for file_path in file_paths:
        try:
            workbook = pd.ExcelFile(file_path)
        except Exception as exc:
            warnings.append(f"读取文件失败：{file_path.name} ({exc})")
            continue

        with workbook:
            for sheet_name in workbook.sheet_names:
                try:
                    header_df = workbook.parse(sheet_name=sheet_name, nrows=0)
                except Exception as exc:
                    warnings.append(f"读取工作表失败：{file_path.name} / {sheet_name} ({exc})")
                    continue

                stripped_headers = pd.Index([str(column_name).strip() for column_name in header_df.columns])
                header_lookup = {
                    stripped_column: original_column
                    for stripped_column, original_column in zip(stripped_headers, header_df.columns)
                }
                matched_columns = {
                    column_name: None
                    for column_name in RELATION_COLUMNS
                }
                for column_name in RELATION_COLUMNS:
                    matched_header = _first_matching_column(stripped_headers, RELATION_COLUMN_ALIASES[column_name])
                    if matched_header is not None:
                        matched_columns[column_name] = header_lookup.get(matched_header)

                if not all(matched_columns.values()):
                    continue
                contract_detected = True

                use_columns = [matched_column for matched_column in matched_columns.values() if matched_column]
                if not use_columns:
                    continue

                try:
                    sheet_df = workbook.parse(sheet_name=sheet_name, usecols=use_columns, dtype=object)
                except Exception as exc:
                    warnings.append(f"读取工作表失败：{file_path.name} / {sheet_name} ({exc})")
                    continue

                if sheet_df.empty:
                    continue

                relationship_df = pd.DataFrame(index=sheet_df.index)
                for column_name in RELATION_COLUMNS:
                    matched_column = matched_columns.get(column_name)
                    if matched_column:
                        relationship_df[column_name] = sheet_df[matched_column]
                    else:
                        relationship_df[column_name] = pd.Series([np.nan] * len(sheet_df), index=sheet_df.index)

                for column_name in RELATION_COLUMNS:
                    relationship_df[column_name] = _clean_text_series(relationship_df[column_name])
                relationship_df = relationship_df[
                    relationship_df["层级0编码"].ne("") & relationship_df["层级1编码"].ne("")
                ]
                if not relationship_df.empty:
                    frames.append(relationship_df)

    if not frames:
        return pd.DataFrame(columns=RELATION_COLUMNS), warnings, contract_detected

    relationship_df = pd.concat(frames, ignore_index=True)
    relationship_df = relationship_df.drop_duplicates(subset=["层级0编码", "层级1编码"], keep="first")
    return relationship_df.reset_index(drop=True), warnings, contract_detected


def _prepare_lookup_columns(lookup_df: pd.DataFrame) -> pd.DataFrame:
    prepared = lookup_df.copy()
    for column_name in [
        "material_name",
        "vehicle_series",
        "resource_developer",
        "pricing_owner",
        "order_price",
        "retail_price",
    ]:
        if column_name not in prepared.columns:
            prepared[column_name] = pd.NA
    return prepared


def _build_enriched_relationship_df(relationship_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    material_codes = sorted(
        {
            *relationship_df["层级0编码"].astype(str).tolist(),
            *relationship_df["层级1编码"].astype(str).tolist(),
        }
    )
    lookup_df = _prepare_lookup_columns(
        harness.execute_action("get_latest_core_cost_lookup", material_codes=material_codes)
    )

    parent_lookup = lookup_df.rename(
        columns={
            "material_code": "层级0编码",
            "material_name": "数据库层级0名称",
            "vehicle_series": "适用车系",
            "cost_amount": "层级0成本",
            "resource_developer": "资源开发",
            "pricing_owner": "定价负责人",
            "order_price": "层级0订购价_数据库",
            "retail_price": "层级0零售价_数据库",
        }
    )[
        [
            "层级0编码",
            "数据库层级0名称",
            "适用车系",
            "层级0成本",
            "资源开发",
            "定价负责人",
            "层级0订购价_数据库",
            "层级0零售价_数据库",
        ]
    ].copy()
    parent_lookup["层级0数据库命中"] = True
    child_lookup = lookup_df.rename(
        columns={
            "material_code": "层级1编码",
            "material_name": "数据库层级1名称",
            "cost_amount": "层级1成本",
            "resource_developer": "层级1资源开发",
            "pricing_owner": "层级1定价负责人",
            "order_price": "层级1订购价_数据库",
            "retail_price": "层级1零售价_数据库",
        }
    )[
        [
            "层级1编码",
            "数据库层级1名称",
            "层级1成本",
            "层级1资源开发",
            "层级1定价负责人",
            "层级1订购价_数据库",
            "层级1零售价_数据库",
        ]
    ]

    enriched_df = relationship_df.merge(parent_lookup, on="层级0编码", how="left")
    enriched_df = enriched_df.merge(child_lookup, on="层级1编码", how="left")
    enriched_df["层级0数据库命中"] = enriched_df["层级0数据库命中"].eq(True)

    layer0_name_series = _clean_text_series(enriched_df["层级0名称"]).replace("", pd.NA)
    layer1_name_series = _clean_text_series(enriched_df["层级1名称"]).replace("", pd.NA)
    db_layer0_name = _clean_text_series(enriched_df["数据库层级0名称"]).replace("", pd.NA)
    db_layer1_name = _clean_text_series(enriched_df["数据库层级1名称"]).replace("", pd.NA)
    enriched_df["层级0名称"] = layer0_name_series.combine_first(db_layer0_name).fillna("未匹配")
    enriched_df["层级1名称"] = layer1_name_series.combine_first(db_layer1_name).fillna("未匹配")
    vehicle_series_series = _clean_text_series(enriched_df["适用车系"])
    enriched_df["适用车系"] = np.where(
        vehicle_series_series.ne(""),
        vehicle_series_series,
        np.where(enriched_df["层级0数据库命中"], "未知", "未配置"),
    )
    enriched_df["资源开发"] = _clean_text_series(enriched_df["资源开发"]).replace("", "未配置")
    enriched_df["定价负责人"] = _clean_text_series(enriched_df["定价负责人"]).replace("", "未配置")
    enriched_df["层级1资源开发"] = _clean_text_series(enriched_df["层级1资源开发"]).replace("", "未配置")
    enriched_df["层级1定价负责人"] = _clean_text_series(enriched_df["层级1定价负责人"]).replace("", "未配置")

    enriched_df["层级0成本"] = _coerce_numeric_series(enriched_df["层级0成本"])
    enriched_df["层级1成本"] = _coerce_numeric_series(enriched_df["层级1成本"])

    layer0_order_raw = _coerce_numeric_series(enriched_df["层级0订购价_数据库"])
    layer0_retail_raw = _coerce_numeric_series(enriched_df["层级0零售价_数据库"])
    layer1_order_raw = _coerce_numeric_series(enriched_df["层级1订购价_数据库"])
    layer1_retail_raw = _coerce_numeric_series(enriched_df["层级1零售价_数据库"])

    enriched_df["层级0订购价"] = layer0_order_raw.fillna(0.0)
    enriched_df["层级0订购价来源"] = np.where(layer0_order_raw.notna(), "SQLite字段", "数据库缺失，按0处理")
    enriched_df["层级0零售价"] = layer0_retail_raw.fillna(0.0)
    enriched_df["层级0零售价来源"] = np.where(layer0_retail_raw.notna(), "SQLite字段", "数据库缺失，按0处理")
    enriched_df["层级1订购价"] = layer1_order_raw.fillna(0.0)
    enriched_df["层级1零售价"] = layer1_retail_raw.fillna(0.0)

    stats = {
        "自动补齐层级0成本行数": int(enriched_df["层级0成本"].notna().sum()),
        "自动补齐层级1成本行数": int(enriched_df["层级1成本"].notna().sum()),
        "仍缺失成本行数": int(enriched_df["层级0成本"].isna().sum() + enriched_df["层级1成本"].isna().sum()),
    }
    return enriched_df, stats


def _first_non_empty(series: pd.Series) -> str:
    cleaned = _clean_text_series(series)
    non_empty = cleaned[cleaned.ne("")]
    if non_empty.empty:
        return ""
    return str(non_empty.iloc[0])


def _first_numeric(series: pd.Series) -> float:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return float("nan")
    return float(numeric_series.iloc[0])


def _vectorized_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numeric_numerator = pd.to_numeric(numerator, errors="coerce")
    numeric_denominator = pd.to_numeric(denominator, errors="coerce")
    valid_mask = numeric_denominator.notna() & numeric_denominator.ne(0)
    ratio = pd.Series(np.nan, index=numeric_numerator.index, dtype=float)
    ratio.loc[valid_mask] = numeric_numerator.loc[valid_mask] / numeric_denominator.loc[valid_mask]
    return ratio


def sort_assembly_summary_by_cost_ratio(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "成本比例" not in summary_df.columns:
        return summary_df.copy()
    sorted_df = summary_df.copy()
    sorted_df["_成本比例排序"] = pd.to_numeric(sorted_df["成本比例"], errors="coerce")
    secondary_cols = [column_name for column_name in ["层级0编码", "层级0名称"] if column_name in sorted_df.columns]
    sorted_df = sorted_df.sort_values(
        ["_成本比例排序", *secondary_cols],
        ascending=[False, *([True] * len(secondary_cols))],
        na_position="last",
    )
    return sorted_df.drop(columns=["_成本比例排序"]).reset_index(drop=True)


def build_abnormal_ratio_export_df(
    summary_df: pd.DataFrame,
    export_df: pd.DataFrame,
    *,
    threshold: float = 1.2,
) -> pd.DataFrame:
    if summary_df.empty or export_df.empty or "层级0编码" not in summary_df.columns or "层级0编码" not in export_df.columns:
        return export_df.iloc[0:0].copy()
    if "成本比例" not in summary_df.columns:
        return export_df.iloc[0:0].copy()

    ratio_values = pd.to_numeric(summary_df["成本比例"], errors="coerce")
    abnormal_parent_codes = set(
        summary_df.loc[ratio_values.gt(float(threshold)), "层级0编码"].astype(str).str.strip()
    )
    if not abnormal_parent_codes:
        return export_df.iloc[0:0].copy()

    export_parent_codes = export_df["层级0编码"].astype(str).str.strip()
    return export_df[export_parent_codes.isin(abnormal_parent_codes)].copy().reset_index(drop=True)


def _build_summary_df(enriched_df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = enriched_df.groupby("层级0编码", sort=True, as_index=False).agg(
        适用车系=("适用车系", _first_non_empty),
        层级0名称=("层级0名称", _first_non_empty),
        层级0成本=("层级0成本", _first_numeric),
        资源开发=("资源开发", _first_non_empty),
        定价负责人=("定价负责人", _first_non_empty),
        层级0订购价=("层级0订购价", _first_numeric),
        层级0订购价来源=("层级0订购价来源", _first_non_empty),
        层级0零售价=("层级0零售价", _first_numeric),
        层级0零售价来源=("层级0零售价来源", _first_non_empty),
        子物料数量=("层级1编码", lambda values: int(_clean_text_series(values).ne("").sum())),
        层级1成本=("层级1成本", lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0).sum())),
        层级1订购价汇总=("层级1订购价", lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0).sum())),
        层级1零售价汇总=("层级1零售价", lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0).sum())),
    )
    if grouped_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped_df["成本比例"] = _vectorized_ratio(grouped_df["层级1成本"], grouped_df["层级0成本"])
    grouped_df["订购价比例"] = _vectorized_ratio(grouped_df["层级1订购价汇总"], grouped_df["层级0订购价"])
    grouped_df["零售价比例"] = _vectorized_ratio(grouped_df["层级1零售价汇总"], grouped_df["层级0零售价"])
    grouped_df["资源开发"] = grouped_df["资源开发"].replace("", "未配置")
    grouped_df["定价负责人"] = grouped_df["定价负责人"].replace("", "未配置")
    grouped_df["适用车系"] = grouped_df["适用车系"].replace("", "未配置")
    return sort_assembly_summary_by_cost_ratio(grouped_df[SUMMARY_COLUMNS])


def _build_detail_df(enriched_df: pd.DataFrame) -> pd.DataFrame:
    detail_df = enriched_df[
        ["层级0编码", "层级1编码", "层级1名称", "层级1成本", "层级1订购价", "层级1零售价"]
    ].copy()
    detail_df["资源开发"] = _clean_text_series(enriched_df["层级1资源开发"]).replace("", "未配置")
    detail_df["定价负责人"] = _clean_text_series(enriched_df["层级1定价负责人"]).replace("", "未配置")
    if detail_df.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    return (
        detail_df[DETAIL_COLUMNS]
        .sort_values(["层级0编码", "层级1编码", "层级1名称"], na_position="last")
        .reset_index(drop=True)
    )


def _build_export_df(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    summary_export = summary_df.copy()
    summary_export["层级"] = "层级0"
    summary_export["层级1编码"] = ""
    summary_export["层级1名称"] = ""
    summary_export["层级1订购价"] = np.nan
    summary_export["层级1零售价"] = np.nan
    summary_export["_sort_order"] = 0
    summary_export = summary_export[EXPORT_COLUMNS + ["_sort_order"]]

    detail_export = detail_df.merge(
        summary_df[
            [
                "适用车系",
                "层级0编码",
                "子物料数量",
                "层级0名称",
                "层级0成本",
                "层级0订购价",
                "层级0订购价来源",
                "层级0零售价",
                "层级0零售价来源",
            ]
        ],
        on="层级0编码",
        how="left",
    )
    detail_export["层级0成本"] = detail_export["层级0成本"]
    detail_export["层级"] = "层级1"
    detail_export["成本比例"] = np.nan
    detail_export["订购价比例"] = np.nan
    detail_export["零售价比例"] = np.nan
    detail_export["_sort_order"] = 1
    detail_export = detail_export[EXPORT_COLUMNS + ["_sort_order"]]

    export_df = pd.concat([summary_export, detail_export], ignore_index=True)
    export_df = export_df.sort_values(["层级0编码", "_sort_order", "层级1编码"], na_position="last")
    return export_df.drop(columns=["_sort_order"]).reset_index(drop=True)


def _get_session_assembly_path() -> str:
    try:
        return str(st.session_state.get("assembly_data_path") or "").strip()
    except Exception:
        return ""


def load_and_process_assembly_data(folder_path: str | None = None) -> dict[str, Any]:
    normalized_path = _get_session_assembly_path() or str(folder_path or "").strip()
    if not normalized_path:
        return _empty_payload("请先在“系统设置”中配置一级件明细数据路径")

    folder = Path(normalized_path)
    if not folder.exists() or not folder.is_dir():
        return _empty_payload(f"一级件明细数据路径不存在：{normalized_path}")

    file_paths = _list_assembly_files(normalized_path)
    if not file_paths:
        return _empty_payload(info_message="当前目录下未找到可用的一级件明细 Excel 文件")

    relationship_df, warnings, contract_detected = _load_relationship_rows(file_paths)
    if relationship_df.empty:
        if not contract_detected:
            return _empty_payload(REQUIRED_LAYER_COLUMNS_MESSAGE, warnings=warnings)
        return _empty_payload(REQUIRED_LAYER_COLUMNS_MESSAGE, warnings=warnings)

    enriched_df, patch_stats = _build_enriched_relationship_df(relationship_df)
    summary_df = _build_summary_df(enriched_df)
    detail_df = _build_detail_df(enriched_df)
    export_df = _build_export_df(summary_df, detail_df)
    stats = {
        "层级0数": int(len(summary_df)),
        "层级1数": int(len(detail_df)),
        "自动补齐层级0成本行数": patch_stats["自动补齐层级0成本行数"],
        "自动补齐层级1成本行数": patch_stats["自动补齐层级1成本行数"],
        "仍缺失成本行数": patch_stats["仍缺失成本行数"],
        "源文件数": int(len(file_paths)),
    }
    return {
        "summary_df": summary_df,
        "detail_df": detail_df,
        "export_df": export_df,
        "stats": stats,
        "error_message": None,
        "warnings": warnings,
        "info_message": None,
    }


def load_assembly_audit_bundle(folder_path: str | None = None) -> dict[str, Any]:
    return load_and_process_assembly_data(folder_path)
