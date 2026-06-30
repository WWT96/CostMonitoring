import os
import random
import time
from typing import Any, MutableMapping

import numpy as np
import pandas as pd
import streamlit as st

import harness
from compute_jobs import ComputeJob
from anomaly_engine import (
    enrich_anomaly_with_inferred_reasons,
)
from config import settings
from data_ingestion import (
    analyze_subpart_costs,
    detect_price_column,
    generate_pivot_report,
    generate_trend_report,
    get_vehicle_gradient_comparison,
)
from import_jobs import ImportJob
from storage_service import canonicalize_record_key, split_record_key


NO_DATA_WARNING = "⚠️ 请先在“⚙️ 系统设置”中配置本地数据路径，或导入本地文件"
RUNTIME_GOVERNANCE_ENV = "COST_MONITOR_RUNTIME_GOVERNANCE"
TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}
PATH_KEY_ALIASES = {
    "assembly_detail_data_path": "assembly_data_path",
}

SESSION_DEFAULTS = {
    "data": None,
    "price_col": "",
    "anomaly_mode": "原始测算",
    "sheet_metal_anomaly_mode": "原始测算",
    "input_data_path": settings.input_data_path,
    "quantitative_skills_path": settings.quantitative_skills_path,
    "qualitative_skills_path": settings.qualitative_skills_path,
    "assembly_data_path": settings.assembly_data_path,
    "sheet_metal_base_info_path": settings.sheet_metal_base_info_path,
    "sheet_metal_model_export_path": settings.sheet_metal_model_export_path,
    "sheet_metal_report_export_path": settings.sheet_metal_report_export_path,
    "loaded_data_origin": "",
    "report_page_number": 1,
    "search_code": "",
    "search_name": "",
    "last_search_hash": "",
    "vehicle_rank_text": "",
    "vehicle_rank": [],
    "current_page": "概览",
    "active_page": "概览",
    "knowledge_sync_status": None,
    "cost_refresh_token": 0,
    "sheet_metal_refresh_token": 0,
    "_storage_initialized": False,
    "_local_db_refresh_token": 0.0,
    "_startup_bootstrap_complete": False,
    "_startup_bootstrap_status": None,
    "_harness_audit_result": None,
    "_harness_path_status": None,
}

PATH_SESSION_KEYS = tuple(
    key for key in SESSION_DEFAULTS if key.endswith("_path")
)


def _new_refresh_token() -> int:
    return random.getrandbits(63)


def initialize_session_state() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if not st.session_state.get("cost_refresh_token"):
        st.session_state["cost_refresh_token"] = _new_refresh_token()
    if not st.session_state.get("sheet_metal_refresh_token"):
        st.session_state["sheet_metal_refresh_token"] = _new_refresh_token()
    harness.execute_action("ensure_session_paths", session_state=st.session_state)


def get_cost_refresh_token() -> int:
    return int(st.session_state.get("cost_refresh_token") or 0)


def bump_cost_refresh_token() -> int:
    refresh_token = _new_refresh_token()
    st.session_state["cost_refresh_token"] = refresh_token
    return refresh_token


def get_sheet_metal_refresh_token() -> int:
    return int(st.session_state.get("sheet_metal_refresh_token") or 0)


def bump_sheet_metal_refresh_token() -> int:
    refresh_token = _new_refresh_token()
    st.session_state["sheet_metal_refresh_token"] = refresh_token
    return refresh_token


def reset_session_key(key: str, value: Any = "") -> None:
    st.session_state[key] = value


def ensure_selectbox_state(key: str, options: list[Any], default: Any | None = None) -> Any:
    if not options:
        return default
    fallback = default if default in options else options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback
    return st.session_state[key]


def _normalize_path_key(path_key: str) -> str:
    return PATH_KEY_ALIASES.get(path_key, path_key)


def get_path_setting(path_key: str) -> str:
    normalized_key = _normalize_path_key(path_key)
    if normalized_key in st.session_state:
        return str(st.session_state.get(normalized_key) or "").strip()
    return str(getattr(settings, normalized_key, getattr(settings, path_key, "")) or "").strip()


def apply_session_path_updates(path_updates: dict[str, str]) -> None:
    normalized_updates = {
        _normalize_path_key(path_key): str(path_value or "").strip()
        for path_key, path_value in path_updates.items()
    }
    for path_key in PATH_SESSION_KEYS:
        if path_key in normalized_updates:
            st.session_state[path_key] = normalized_updates[path_key]


def clear_session_path_settings() -> None:
    apply_session_path_updates({path_key: "" for path_key in PATH_SESSION_KEYS})


def clear_loaded_data_state(session_state: MutableMapping[str, Any] | None = None) -> dict[str, int]:
    state = session_state if session_state is not None else st.session_state
    data = state.get("data")
    released_rows = int(len(data)) if isinstance(data, pd.DataFrame) else 0
    cleared_download_keys = 0

    for key in list(state.keys()):
        if str(key).startswith("_deferred_download_"):
            state.pop(key, None)
            cleared_download_keys += 1

    state["data"] = None
    state["price_col"] = ""
    state["loaded_data_origin"] = ""
    state["cost_refresh_token"] = _new_refresh_token()
    state["sheet_metal_refresh_token"] = _new_refresh_token()

    if state is st.session_state:
        for cache_name in [
            "cached_load_data",
            "cached_pivot_report",
            "cached_trend_report",
            "cached_vehicle_compare",
            "cached_anomaly_report",
            "cached_subpart_analysis",
            "cached_anomaly_report_weighted",
            "cached_load_local_database",
            "cached_enrich_anomaly_with_ai",
        ]:
            cache_func = globals().get(cache_name)
            if cache_func is not None:
                try:
                    cache_func.clear()
                except Exception:
                    pass

    return {
        "released_rows": released_rows,
        "cleared_download_keys": cleared_download_keys,
    }


def normalize_active_page_alias() -> None:
    alias_map = {
        "异常成本监控体系": "成本异常监控",
        "Skills 技能引擎": "成本区间 Skills",
        "成本变动趋势": "全量成本报表",
    }
    current_page = str(
        st.session_state.get("current_page")
        or st.session_state.get("active_page")
        or "概览"
    )
    normalized_page = alias_map.get(current_page, current_page)
    st.session_state.current_page = normalized_page
    st.session_state.active_page = normalized_page


def ensure_storage_initialized() -> None:
    if st.session_state.get("_storage_initialized", False):
        return
    harness.execute_action("initialize_storage")
    st.session_state["_storage_initialized"] = True


def is_runtime_governance_enabled() -> bool:
    return str(os.environ.get(RUNTIME_GOVERNANCE_ENV, "") or "").strip().lower() in TRUE_ENV_VALUES


def maybe_bootstrap_runtime_governance(session_state: Any) -> dict[str, Any]:
    if not is_runtime_governance_enabled():
        session_state[harness.HARNESS_AUDIT_STATE_KEY] = None
        return {"enabled": False}

    payload = harness.bootstrap_runtime_governance(session_state)
    return {"enabled": True, **payload}


def inject_css(is_overview: bool = False) -> None:
    base_css = """
    <style>
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: 98% !important;
        }
        [data-testid="stDataEditor"],
        [data-testid="stDataFrame"] {
            width: 100% !important;
        }
        [data-testid="stDataEditor"] [role="grid"],
        [data-testid="stDataFrame"] [role="grid"] {
            border: 1px solid #e9ecef;
            border-radius: 10px;
            overflow-x: auto !important;
        }
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
        [data-testid="stDataEditor"] [role="gridcell"] *,
        [data-testid="stDataFrame"] [role="gridcell"] * {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            line-height: 1.35 !important;
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
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            font-weight: 700 !important;
            color: #2f3e52 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
            font-size: 15px !important;
            font-weight: 700 !important;
            color: #2f3e52 !important;
        }
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
            min-height: 134px !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
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


def build_calibration_management_df(
    label_details: dict[str, dict[str, str]],
    *,
    core_source_df: pd.DataFrame | None = None,
    fallback_source_df: pd.DataFrame | None = None,
    fallback_price_col: str = "",
) -> pd.DataFrame:
    display_columns = [
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
    rows = []
    for record_key, payload in label_details.items():
        parsed = split_record_key(record_key)
        rows.append(
            {
                "record_key": parsed["record_key"],
                "物料编码": parsed["物料编码"],
                "工厂": parsed["工厂"],
                "价格有效期于": parsed["价格有效期于"],
                "价格": parsed["价格"],
                "当前标注": payload.get("label", ""),
                "标注备注": payload.get("remark", ""),
                "撤回标注": False,
                "_join_date_key": parsed["_join_date_key"],
                "_join_price_key": parsed["_join_price_key"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=["record_key"] + display_columns)

    placeholder_text_values = {"", "nan", "none", "null", "<na>", "nat"}

    def _clean_text_series(values: Any, index: pd.Index | None = None) -> pd.Series:
        if isinstance(values, pd.Series):
            series = values.copy()
        else:
            series = pd.Series(values, index=index)
        cleaned = series.astype("string").str.strip()
        return cleaned.mask(cleaned.str.lower().isin(placeholder_text_values), pd.NA)

    def _coalesce_text_columns(*values: Any, index: pd.Index | None = None) -> pd.Series:
        result = pd.Series(pd.NA, index=index, dtype="string")
        for value in values:
            result = result.combine_first(_clean_text_series(value, index=index))
        return result.fillna("").astype(str)

    def _normalize_lookup_frame(source_df: pd.DataFrame | None, price_column: str = "") -> pd.DataFrame:
        if source_df is None or source_df.empty or "物料编码" not in source_df.columns:
            return pd.DataFrame(
                columns=[
                    "record_key",
                    "物料编码",
                    "物料名称",
                    "备件简称",
                    "工厂",
                    "价格",
                    "_join_date_key",
                    "_join_price_key",
                    "monitor_date",
                    "_display_context_score",
                ]
            )

        lookup_df = source_df.copy()
        if "monitor_date" in lookup_df.columns:
            lookup_df["monitor_date"] = pd.to_datetime(lookup_df["monitor_date"], errors="coerce")
        elif "价格有效于" in lookup_df.columns:
            lookup_df["monitor_date"] = pd.to_datetime(lookup_df["价格有效于"], errors="coerce")
        elif "价格有效期于" in lookup_df.columns:
            lookup_df["monitor_date"] = pd.to_datetime(lookup_df["价格有效期于"], errors="coerce")
        else:
            lookup_df["monitor_date"] = pd.NaT

        if "成本" in lookup_df.columns:
            lookup_df["价格"] = pd.to_numeric(lookup_df["成本"], errors="coerce")
        elif price_column and price_column in lookup_df.columns:
            lookup_df["价格"] = pd.to_numeric(lookup_df[price_column], errors="coerce")
        elif "实际成本" in lookup_df.columns:
            lookup_df["价格"] = pd.to_numeric(lookup_df["实际成本"], errors="coerce")
        else:
            lookup_df["价格"] = np.nan

        lookup_df["物料编码"] = _clean_text_series(lookup_df["物料编码"]).fillna("").astype(str)
        for column_name in ["物料名称", "备件简称", "工厂"]:
            if column_name not in lookup_df.columns:
                lookup_df[column_name] = ""
            lookup_df[column_name] = _clean_text_series(lookup_df[column_name]).fillna("").astype(str)

        lookup_df["_join_date_key"] = lookup_df["monitor_date"].dt.strftime("%Y-%m-%d").fillna("")
        lookup_df["_join_price_key"] = lookup_df["价格"].apply(
            lambda value: f"{float(value):.4f}" if pd.notna(value) else ""
        )
        if "_record_key" in lookup_df.columns:
            lookup_df["record_key"] = lookup_df["_record_key"].fillna("").astype(str).map(canonicalize_record_key)
        else:
            lookup_df["record_key"] = ""
        lookup_df["_display_context_score"] = (
            lookup_df[["物料名称", "备件简称"]]
            .replace("", pd.NA)
            .notna()
            .sum(axis=1)
        )
        lookup_df = lookup_df.sort_values("monitor_date", na_position="last")
        return lookup_df[
            [
                "record_key",
                "物料编码",
                "物料名称",
                "备件简称",
                "工厂",
                "价格",
                "_join_date_key",
                "_join_price_key",
                "monitor_date",
                "_display_context_score",
            ]
        ]

    base_df = pd.DataFrame(rows)
    lookup_frames = []
    primary_lookup_df = _normalize_lookup_frame(core_source_df)
    if not primary_lookup_df.empty:
        lookup_frames.append(primary_lookup_df.assign(_source_priority=0))
    fallback_lookup_df = _normalize_lookup_frame(fallback_source_df, fallback_price_col)
    if not fallback_lookup_df.empty:
        lookup_frames.append(fallback_lookup_df.assign(_source_priority=1))

    if lookup_frames:
        lookup_df = pd.concat(lookup_frames, ignore_index=True, sort=False)
        lookup_df = lookup_df.sort_values(["_source_priority", "monitor_date"], na_position="last")

        base_df["_canonical_record_key"] = base_df["record_key"].astype(str).map(canonicalize_record_key)
        keyed_lookup_df = lookup_df[lookup_df["record_key"].astype(str).str.strip().ne("")].drop_duplicates(
            subset=["record_key"],
            keep="first",
        )
        if not keyed_lookup_df.empty:
            base_df = base_df.merge(
                keyed_lookup_df[
                    ["record_key", "物料编码", "工厂", "_join_date_key", "_join_price_key", "价格", "物料名称", "备件简称"]
                ].rename(
                    columns={
                        "record_key": "_canonical_record_key",
                        "物料编码": "_keyed_物料编码",
                        "工厂": "_keyed_工厂",
                        "_join_date_key": "_keyed_join_date_key",
                        "_join_price_key": "_keyed_join_price_key",
                        "价格": "_keyed_价格",
                        "物料名称": "_keyed_物料名称",
                        "备件简称": "_keyed_备件简称",
                    }
                ),
                on="_canonical_record_key",
                how="left",
            )
        else:
            base_df["_keyed_物料编码"] = ""
            base_df["_keyed_工厂"] = ""
            base_df["_keyed_join_date_key"] = ""
            base_df["_keyed_join_price_key"] = ""
            base_df["_keyed_价格"] = np.nan
            base_df["_keyed_物料名称"] = ""
            base_df["_keyed_备件简称"] = ""

        exact_lookup_df = lookup_df.drop_duplicates(
            subset=["物料编码", "工厂", "_join_date_key", "_join_price_key"],
            keep="first",
        )
        base_df = base_df.merge(
            exact_lookup_df[
                ["物料编码", "工厂", "_join_date_key", "_join_price_key", "物料名称", "备件简称"]
            ],
            on=["物料编码", "工厂", "_join_date_key", "_join_price_key"],
            how="left",
        )
        base_df["物料编码"] = (
            base_df["物料编码"].astype("string")
            .replace("", pd.NA)
            .combine_first(base_df["_keyed_物料编码"].astype("string").replace("", pd.NA))
            .fillna("")
            .astype(str)
        )
        base_df["工厂"] = (
            base_df["工厂"].astype("string")
            .replace("", pd.NA)
            .combine_first(base_df["_keyed_工厂"].astype("string").replace("", pd.NA))
            .fillna("")
            .astype(str)
        )
        base_df["_join_date_key"] = (
            base_df["_join_date_key"].astype("string")
            .replace("", pd.NA)
            .combine_first(base_df["_keyed_join_date_key"].astype("string").replace("", pd.NA))
            .fillna("")
            .astype(str)
        )
        base_df["_join_price_key"] = (
            base_df["_join_price_key"].astype("string")
            .replace("", pd.NA)
            .combine_first(base_df["_keyed_join_price_key"].astype("string").replace("", pd.NA))
            .fillna("")
            .astype(str)
        )
        base_df["价格有效期于"] = (
            base_df["价格有效期于"].astype("string")
            .replace("", pd.NA)
            .combine_first(base_df["_join_date_key"].astype("string").replace("", pd.NA))
            .fillna("")
            .astype(str)
        )
        base_df["价格"] = pd.to_numeric(base_df["价格"], errors="coerce").combine_first(
            pd.to_numeric(base_df["_keyed_价格"], errors="coerce")
        )

        fallback_name_df = lookup_df.sort_values(
            ["物料编码", "_display_context_score", "_source_priority", "monitor_date"],
            ascending=[True, False, True, True],
            na_position="last",
            kind="mergesort",
        ).drop_duplicates(subset=["物料编码"], keep="first").rename(
            columns={
                "物料名称": "_fallback_物料名称",
                "备件简称": "_fallback_备件简称",
            }
        )
        base_df = base_df.merge(
            fallback_name_df[["物料编码", "_fallback_物料名称", "_fallback_备件简称"]],
            on="物料编码",
            how="left",
        )
        base_df["物料名称"] = _coalesce_text_columns(
            base_df.get("物料名称", ""),
            base_df.get("_keyed_物料名称", ""),
            base_df.get("_fallback_物料名称", ""),
            index=base_df.index,
        )
        base_df["备件简称"] = _coalesce_text_columns(
            base_df.get("备件简称", ""),
            base_df.get("_keyed_备件简称", ""),
            base_df.get("_fallback_备件简称", ""),
            index=base_df.index,
        )
        has_display_context = (
            _clean_text_series(base_df["物料名称"], index=base_df.index).notna()
            | _clean_text_series(base_df["备件简称"], index=base_df.index).notna()
        )
        base_df = base_df.loc[has_display_context].copy()
        base_df = base_df.drop(
            columns=[
                "_canonical_record_key",
                "_keyed_物料编码",
                "_keyed_工厂",
                "_keyed_join_date_key",
                "_keyed_join_price_key",
                "_keyed_价格",
                "_keyed_物料名称",
                "_keyed_备件简称",
                "_fallback_物料名称",
                "_fallback_备件简称",
            ],
            errors="ignore",
        )
    else:
        base_df["物料名称"] = ""
        base_df["备件简称"] = ""
        base_df = base_df.iloc[0:0].copy()

    return base_df[["record_key"] + display_columns].copy()


def reset_search_callback() -> None:
    st.session_state.search_code = ""
    st.session_state.search_name = ""
    st.session_state.report_page_number = 1


@st.cache_data(max_entries=4, ttl=900)
def cached_load_data(folder_path: str, refresh_token: int):
    return harness.execute_action("load_data_from_folder", folder_path=folder_path)


@st.cache_data(max_entries=6, ttl=900)
def cached_pivot_report(df, price_col: str, refresh_token: int):
    return generate_pivot_report(df, price_col)


@st.cache_data(max_entries=6, ttl=900)
def cached_trend_report(df, price_col: str, refresh_token: int):
    return generate_trend_report(df, price_col)


@st.cache_data(max_entries=6, ttl=900)
def cached_vehicle_compare(df, price_col: str, part_name: str, rank_tuple: tuple, refresh_token: int):
    return get_vehicle_gradient_comparison(df, price_col, part_name, list(rank_tuple))


@st.cache_data(max_entries=3, ttl=900)
def cached_anomaly_report(df, price_col: str, refresh_token: int):
    return ComputeJob().run_cost_anomaly(df, price_col, result_mode="raw")


@st.cache_data(max_entries=4, ttl=900)
def cached_subpart_analysis(df, price_col: str, refresh_token: int):
    return analyze_subpart_costs(df, price_col)


@st.cache_data(max_entries=3, ttl=900)
def cached_anomaly_report_weighted(
    df,
    price_col: str,
    expert_labels_tuple: tuple,
    refresh_token: int,
    sigma_multiplier: float = 1.0,
    expert_weight_override: int = 0,
    decay_alpha: float = 1.0,
    gap_k: float = 4.0,
    baseline_quantile: float = 0.5,
    skills_overrides_json: str = "",
):
    return ComputeJob().run_weighted_cost_anomaly(
        df,
        price_col,
        expert_labels_tuple,
        result_mode="weighted",
        sigma_multiplier=sigma_multiplier,
        expert_weight_override=expert_weight_override,
        decay_alpha=decay_alpha,
        gap_k=gap_k,
        baseline_quantile=baseline_quantile,
        skills_overrides_json=skills_overrides_json,
    )


@st.cache_data(max_entries=3, ttl=300)
def cached_load_local_database(refresh_token: float, cost_refresh_token: int):
    try:
        return harness.execute_action("load_core_cost_records")
    except Exception as exc:
        return None, None, f"读取本地数据库失败: {exc}"


@st.cache_data(max_entries=4, ttl=900)
def cached_enrich_anomaly_with_ai(result_df: pd.DataFrame, knowledge_refresh_token: float):
    return enrich_anomaly_with_inferred_reasons(result_df)


def require_price_col(df: pd.DataFrame) -> str:
    price_col = st.session_state.get("price_col", "")
    if price_col and price_col in df.columns:
        return price_col

    detected_price_col = detect_price_column(df.columns)
    if detected_price_col:
        st.session_state.price_col = detected_price_col
        return detected_price_col

    st.error("当前数据中未找到可用价格列，请检查源数据列名。")
    st.stop()


def set_loaded_data(df: pd.DataFrame, price_col: str, origin: str) -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("_deferred_download_"):
            st.session_state.pop(key, None)
    st.session_state.data = df
    st.session_state.price_col = price_col or ""
    st.session_state.loaded_data_origin = origin


def format_bytes(size_in_bytes: int | float) -> str:
    value = float(size_in_bytes or 0)
    units = ["字节", "千字节", "兆字节", "吉字节", "太字节"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "字节" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.2f} 太字节"


def persist_dataframe_to_local_database(
    df: pd.DataFrame,
    price_col: str,
    *,
    invalidate_cache: bool = True,
) -> int:
    synced_rows = harness.execute_action(
        "sync_core_cost_records",
        df=df,
        price_col=price_col,
        mode="full",
    )
    st.session_state["_local_db_refresh_token"] = time.time()
    if invalidate_cache:
        bump_cost_refresh_token()
    return synced_rows


def validate_local_database_sync_count(expected_rows: int, synced_rows: int) -> tuple[bool, str]:
    try:
        db_status = harness.execute_action("get_core_cost_records_status")
        db_rows = int(db_status.get("row_count", 0) or 0)
    except Exception as exc:
        return False, f"本地数据库写入后校验失败：{exc}"

    if int(expected_rows) != int(synced_rows) or int(expected_rows) != db_rows:
        return (
            False,
            f"本地数据同步校验未通过：有效源数据 {expected_rows} 条，准备写入 {synced_rows} 条，数据库当前 {db_rows} 条。"
            "请检查是否存在字段缺失、数据库锁定或旧版本索引残留。",
        )
    return True, f"本地数据同步校验通过：有效源数据、写入记录、数据库记录均为 {expected_rows} 条。"


def load_uploaded_files_into_session(uploaded_files: list[Any], *, origin: str) -> tuple[bool, str]:
    job = ImportJob(
        persist_func=lambda df, price_col: persist_dataframe_to_local_database(
            df,
            price_col,
            invalidate_cache=True,
        )
    )
    result = job.import_uploaded(uploaded_files)
    if not result.success:
        return False, result.message

    sync_ok, sync_message = validate_local_database_sync_count(len(result.dataframe), result.synced_rows)
    if not sync_ok:
        return False, sync_message
    set_loaded_data(result.dataframe, result.price_col or "", origin=origin)
    return True, f"✅ {result.message}{sync_message}"


def load_folder_data_into_session(folder_path: str, *, origin: str) -> tuple[bool, str]:
    normalized_path = str(folder_path or "").strip()
    if not normalized_path:
        return False, "请先输入本地文件夹路径"

    job = ImportJob(
        persist_func=lambda df, price_col: persist_dataframe_to_local_database(
            df,
            price_col,
            invalidate_cache=True,
        )
    )
    result = job.import_folder(normalized_path)
    if not result.success:
        return False, result.message

    sync_ok, sync_message = validate_local_database_sync_count(len(result.dataframe), result.synced_rows)
    if not sync_ok:
        return False, sync_message
    set_loaded_data(result.dataframe, result.price_col or "", origin=origin)
    st.session_state["input_data_path"] = normalized_path
    return True, f"{result.message}{sync_message}"


def reset_and_reload_saved_path_data_into_session() -> tuple[bool, str]:
    saved_path = get_path_setting("input_data_path")
    if not saved_path:
        return False, "当前尚未配置原始数据存放路径，请先在“系统设置”中保存路径。"
    if not os.path.isdir(saved_path):
        return False, f"原始数据存放路径不存在：{saved_path}"

    try:
        cached_load_data.clear()
        cached_load_local_database.clear()
    except Exception:
        pass
    return load_folder_data_into_session(saved_path, origin="overview_reset")


def load_local_database_into_session(*, origin: str = "local_db") -> tuple[bool, str]:
    refresh_token = bump_cost_refresh_token()
    merged_df, price_col, error_msg = cached_load_local_database(
        st.session_state.get("_local_db_refresh_token", 0.0),
        refresh_token,
    )
    if error_msg:
        return False, error_msg
    if merged_df is None or merged_df.empty:
        return False, "本地数据库中暂无核心成本数据"

    set_loaded_data(merged_df, price_col or "", origin=origin)
    return True, f"已从本地数据库恢复 {len(merged_df)} 条记录"


def bootstrap_local_data() -> None:
    settings.reload()
    harness.execute_action("ensure_session_paths", session_state=st.session_state)
    saved_path = settings.input_data_path.strip()

    if st.session_state.get("_startup_bootstrap_complete", False):
        return

    status_payload = None
    db_status = harness.execute_action("get_core_cost_records_status")
    db_has_data = int(db_status.get("row_count", 0) or 0) > 0

    if db_has_data:
        db_df, db_price_col, db_error = cached_load_local_database(
            st.session_state.get("_local_db_refresh_token", 0.0),
            get_cost_refresh_token(),
        )
        if not db_error and db_df is not None and not db_df.empty:
            set_loaded_data(db_df, db_price_col or "", origin="local_db")
            status_payload = {
                "kind": "info",
                "message": f"已从本地数据库恢复 {len(db_df)} 条记录",
            }
    elif saved_path:
        if os.path.isdir(saved_path):
            load_success, message = load_folder_data_into_session(saved_path, origin="settings_path")
            status_payload = {
                "kind": "success" if load_success else "error",
                "message": message,
            }
        else:
            status_payload = {
                "kind": "warning",
                "message": f"本地配置文件中保存的路径不存在：{saved_path}",
            }
    elif status_payload is None:
        status_payload = {
            "kind": "info",
            "message": "当前尚未配置本地数据路径，请先在“⚙️ 系统设置”中保存路径。",
        }

    st.session_state["_startup_bootstrap_status"] = status_payload
    st.session_state["_startup_bootstrap_complete"] = True


def bootstrap_app() -> None:
    initialize_session_state()
    normalize_active_page_alias()
    maybe_bootstrap_runtime_governance(st.session_state)
    ensure_storage_initialized()
    bootstrap_local_data()


def render_bootstrap_status() -> None:
    status_payload = st.session_state.get("_startup_bootstrap_status")
    if not status_payload:
        return

    kind = str(status_payload.get("kind", "info")).strip().lower()
    message = str(status_payload.get("message", "")).strip()
    if not message:
        return

    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def sync_ai_knowledge_base(force_full: bool = False, spinner_text: str = "🤖 正在同步更新 AI 知识库..."):
    with st.spinner(spinner_text):
        sync_result = harness.execute_action("sync_expert_knowledge_base", force_full=force_full)
    st.session_state.knowledge_sync_status = sync_result
    return sync_result


def clear_cost_feedback_and_ai_state() -> None:
    harness.execute_action("clear_feedback")
    harness.execute_action("clear_expert_knowledge_base")
    try:
        cached_enrich_anomaly_with_ai.clear()
    except Exception:
        pass
    try:
        st.session_state.knowledge_sync_status = {
            "status": "no_data",
            "message": "已清空专家批注与 AI 经验库。",
        }
    except Exception:
        pass


def clear_sheet_metal_feedback_state() -> None:
    harness.execute_action("clear_sheet_metal_feedback")
    bump_sheet_metal_refresh_token()


def render_knowledge_sync_status() -> None:
    sync_result = st.session_state.get("knowledge_sync_status")
    if not sync_result:
        return
    message = str(sync_result.get("message", "")).strip()
    if not message:
        return
    status = sync_result.get("status", "")
    if status == "success":
        st.success(f"🤖 {message}")
    elif status in {"no_changes", "no_data", "skipped"}:
        st.info(f"🤖 {message}")
    else:
        st.warning(f"🤖 {message}")
