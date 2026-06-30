from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

import harness
from anomaly_engine import detect_cost_anomalies, detect_cost_anomalies_weighted
import sheet_metal_logic
import skills_engine
from storage_service import build_record_keys


RawDetector = Callable[[pd.DataFrame, str], pd.DataFrame]
ResultLoader = Callable[[str], pd.DataFrame]
FreshResultLoader = Callable[[str, str, str], pd.DataFrame]
ResultRunRecorder = Callable[[str, str, str, int], Any]
CostSkillExtractor = Callable[..., list[dict]]
SheetMetalSkillExtractor = Callable[[pd.DataFrame, dict, float, int], list[dict]]


@dataclass
class SkillsPrecomputeResult:
    all_skills: list[dict]
    export_skills: list[dict]

    @property
    def total_count(self) -> int:
        return len(self.all_skills)

    @property
    def covered_count(self) -> int:
        return len(self.export_skills)


class ComputeJob:
    """Run heavy computations, then hand UI callers the persisted result view."""

    def __init__(
        self,
        *,
        raw_detector: RawDetector | None = None,
        weighted_detector: Callable[..., pd.DataFrame] | None = None,
        raw_loader: ResultLoader | None = None,
        fresh_result_loader: FreshResultLoader | None = None,
        result_run_recorder: ResultRunRecorder | None = None,
        cost_skill_extractor: CostSkillExtractor | None = None,
        sheet_metal_skill_extractor: SheetMetalSkillExtractor | None = None,
    ) -> None:
        self.raw_detector = raw_detector or detect_cost_anomalies
        self.weighted_detector = weighted_detector or detect_cost_anomalies_weighted
        self.raw_loader = raw_loader or (
            lambda result_mode: harness.execute_action("load_cost_anomaly_results", result_mode=result_mode)
        )
        self.fresh_result_loader = fresh_result_loader or (
            lambda result_mode, source_signature, options_signature: harness.execute_action(
                "load_fresh_cost_anomaly_results",
                result_mode=result_mode,
                source_signature=source_signature,
                options_signature=options_signature,
            )
        )
        self.result_run_recorder = result_run_recorder or (
            lambda result_mode, source_signature, options_signature, row_count: harness.execute_action(
                "record_cost_anomaly_result_run",
                result_mode=result_mode,
                source_signature=source_signature,
                options_signature=options_signature,
                row_count=row_count,
            )
        )
        self.cost_skill_extractor = cost_skill_extractor or skills_engine.extract_skills
        self.sheet_metal_skill_extractor = sheet_metal_skill_extractor or sheet_metal_logic.extract_sheet_metal_skills

    def run_cost_anomaly(self, df: pd.DataFrame, price_col: str, *, result_mode: str = "raw") -> pd.DataFrame:
        source_signature = build_cost_anomaly_source_signature(df, price_col)
        options_signature = build_cost_anomaly_options_signature(
            result_mode=result_mode,
            price_col=price_col,
        )
        persisted_df = self._load_fresh_persisted(result_mode, source_signature, options_signature)
        if isinstance(persisted_df, pd.DataFrame) and not persisted_df.empty:
            return persisted_df

        computed_df = self.raw_detector(df, price_col)
        self._record_result_run(result_mode, source_signature, options_signature, computed_df)
        return self._load_persisted_or_computed(result_mode, computed_df)

    def run_weighted_cost_anomaly(
        self,
        df: pd.DataFrame,
        price_col: str,
        expert_labels_tuple: tuple,
        *,
        result_mode: str = "weighted",
        **kwargs,
    ) -> pd.DataFrame:
        source_signature = build_cost_anomaly_source_signature(df, price_col)
        options_signature = build_cost_anomaly_options_signature(
            result_mode=result_mode,
            price_col=price_col,
            expert_labels_tuple=expert_labels_tuple,
            options=kwargs,
        )
        persisted_df = self._load_fresh_persisted(result_mode, source_signature, options_signature)
        if isinstance(persisted_df, pd.DataFrame) and not persisted_df.empty:
            return persisted_df

        computed_df = self.weighted_detector(df, price_col, expert_labels_tuple, **kwargs)
        self._record_result_run(result_mode, source_signature, options_signature, computed_df)
        return self._load_persisted_or_computed(result_mode, computed_df)

    def _load_fresh_persisted(
        self,
        result_mode: str,
        source_signature: str,
        options_signature: str,
    ) -> pd.DataFrame:
        if not source_signature or not options_signature:
            return pd.DataFrame()
        return self.fresh_result_loader(result_mode, source_signature, options_signature)

    def _record_result_run(
        self,
        result_mode: str,
        source_signature: str,
        options_signature: str,
        computed_df: pd.DataFrame,
    ) -> None:
        if not source_signature or not options_signature:
            return
        row_count = int(len(computed_df)) if isinstance(computed_df, pd.DataFrame) else 0
        self.result_run_recorder(result_mode, source_signature, options_signature, row_count)

    def _load_persisted_or_computed(self, result_mode: str, computed_df: pd.DataFrame) -> pd.DataFrame:
        persisted_df = self.raw_loader(result_mode)
        if isinstance(persisted_df, pd.DataFrame) and not persisted_df.empty:
            return persisted_df
        return computed_df

    def precompute_cost_skills(
        self,
        anomaly_df: pd.DataFrame,
        expert_labels: dict | None,
        **skill_kwargs,
    ) -> SkillsPrecomputeResult:
        labels = dict(expert_labels or {})
        all_skills = self.cost_skill_extractor(anomaly_df, labels, **skill_kwargs)
        export_skills = self._filter_skills_to_labeled_groups(all_skills, anomaly_df, labels)
        return SkillsPrecomputeResult(all_skills=all_skills, export_skills=export_skills)

    def precompute_sheet_metal_skills(
        self,
        review_df: pd.DataFrame,
        expert_labels: dict | None,
        *,
        sigma_multiplier: float = 1.0,
        expert_weight: int = 80,
    ) -> SkillsPrecomputeResult:
        labels = dict(expert_labels or {})
        all_skills = self.sheet_metal_skill_extractor(review_df, labels, sigma_multiplier, expert_weight)
        export_skills = self._filter_skills_to_labeled_groups(all_skills, review_df, labels)
        return SkillsPrecomputeResult(all_skills=all_skills, export_skills=export_skills)

    @staticmethod
    def _filter_skills_to_labeled_groups(
        skills: list[dict],
        source_df: pd.DataFrame,
        labels: dict,
        *,
        group_col: str = "备件简称",
    ) -> list[dict]:
        if not labels or source_df is None or source_df.empty or "_record_key" not in source_df.columns or group_col not in source_df.columns:
            return skills

        labeled_keys = set(str(key) for key in labels.keys())
        covered_names = set()
        for group_name, group in source_df.groupby(group_col, sort=False):
            if labeled_keys & set(group["_record_key"].astype(str).values):
                covered_names.add(str(group_name))
        return [skill for skill in skills if str(skill.get(group_col)) in covered_names]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_cost_anomaly_source_signature(df: pd.DataFrame, price_col: str) -> str:
    if df is None or df.empty or price_col not in df.columns:
        return ""

    if "_record_key" in df.columns:
        keys = [str(value) for value in df["_record_key"].dropna().astype(str).tolist() if str(value).strip()]
        return _hash_payload({"row_count": len(keys), "record_keys": sorted(keys)}) if keys else ""

    date_column = "monitor_date" if "monitor_date" in df.columns else "价格有效于" if "价格有效于" in df.columns else ""
    required_columns = ["物料编码", "备件简称", date_column, price_col]
    if not date_column or any(column not in df.columns for column in required_columns):
        return ""

    source_df = df.dropna(subset=required_columns).copy()
    if source_df.empty:
        return ""

    keys = build_record_keys(
        source_df,
        date_column=date_column,
        value_column=price_col,
    ).dropna()
    key_values = [str(value) for value in keys.astype(str).tolist() if str(value).strip()]
    return _hash_payload({"row_count": len(key_values), "record_keys": sorted(key_values)}) if key_values else ""


def build_cost_anomaly_options_signature(
    *,
    result_mode: str,
    price_col: str,
    expert_labels_tuple: tuple = tuple(),
    options: dict[str, Any] | None = None,
) -> str:
    return _hash_payload(
        {
            "result_mode": str(result_mode or "").strip(),
            "price_col": str(price_col or "").strip(),
            "expert_labels_tuple": expert_labels_tuple,
            "options": options or {},
        }
    )
