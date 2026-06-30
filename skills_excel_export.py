from __future__ import annotations

import io
from typing import Any, Dict, List, Sequence

import pandas as pd


BASE_SKILL_COLUMNS = [
    "备件简称",
    "适用算法",
    "当前σ参数",
    "偏置权重",
    "时序敏感度 (Decay Alpha)",
    "圈子严格度 (Gap K)",
    "Baseline Quantile",
    "本组专家标注数",
    "经验对齐率",
]
RING_FIELD_MAP = [
    ("圈层编号", "编号"),
    ("圈层角色", "角色"),
    ("合理下限", "合理下限"),
    ("预测值", "预测值"),
    ("合理上限", "合理上限"),
    ("样本量", "样本量"),
    ("加权样本量", "加权样本量"),
    ("圈层置信度", "置信度"),
    ("专家锚点", "专家锚点"),
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(item) for item in value if str(item).strip())
    return str(value) if value not in (None, "") else ""


def _max_ring_count(skills: Sequence[dict]) -> int:
    return max((len(_as_list(skill.get("多邻居圈合理区间"))) for skill in skills), default=0)


def flatten_skills_for_excel(
    skills: Sequence[dict],
    *,
    interval_key: str,
    distribution_key: str,
) -> pd.DataFrame:
    max_ring_count = _max_ring_count(skills)
    rows: List[Dict[str, Any]] = []
    for skill in skills:
        row: Dict[str, Any] = {}
        for column_name in BASE_SKILL_COLUMNS:
            row[column_name] = skill.get(column_name, "")

        for key, value in _as_dict(skill.get(interval_key)).items():
            row[f"合理区间_{key}"] = value
        for key, value in _as_dict(skill.get(distribution_key)).items():
            row[f"分布_{key}"] = value
        for key, value in _as_dict(skill.get("异常统计")).items():
            row[f"异常统计_{key}"] = value

        semantic_report = _as_dict(skill.get("语义校准报告"))
        row["语义_引用规律数"] = semantic_report.get("引用规律数", "")
        row["语义_主要匹配方式"] = _join_list(semantic_report.get("主要匹配方式", []))
        row["语义_参考文本规律"] = _join_list(semantic_report.get("参考文本规律", []))

        rings = _as_list(skill.get("多邻居圈合理区间"))
        for ring_idx in range(max_ring_count):
            ring = _as_dict(rings[ring_idx]) if ring_idx < len(rings) else {}
            column_prefix = f"邻居圈{ring_idx + 1}"
            for source_key, export_key in RING_FIELD_MAP:
                row[f"{column_prefix}_{export_key}"] = ring.get(source_key, "")
        rows.append(row)

    return pd.DataFrame(rows)


def skills_dataframe_to_excel_bytes(df: pd.DataFrame, *, sheet_name: str) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Skills")
    return buffer.getvalue()


def skills_to_excel_bytes(
    skills: Sequence[dict],
    *,
    interval_key: str,
    distribution_key: str,
    sheet_name: str,
) -> bytes:
    table = flatten_skills_for_excel(
        skills,
        interval_key=interval_key,
        distribution_key=distribution_key,
    )
    return skills_dataframe_to_excel_bytes(table, sheet_name=sheet_name)
