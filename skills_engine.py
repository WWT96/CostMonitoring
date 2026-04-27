"""Skills 技能引擎 & AutoResearch 棘轮迭代。

提供：
1. Skills 技能书 — 每个备件简称的算法参数、分布特征、合理区间边界 (JSON / Markdown)
2. AutoResearch — 严格棘轮循环自动调参（σ × 偏置权重）
3. 深度审计报表 — 物料级原始结论 vs 优化结论对照
"""

import json
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import processor


# ── 1. Skills 技能书 ──────────────────────────────────────────────────────


def extract_skills(
    anomaly_df: pd.DataFrame,
    expert_labels: Dict[str, str],
    sigma_multiplier: float = 1.0,
    expert_weight: int = 80,
) -> List[dict]:
    """为每个备件简称提取 Skills 条目。

    Parameters
    ----------
    anomaly_df : 异常检测结果 DataFrame（含 _record_key / 预测值 / 合理上下限 等列）。
    expert_labels : {record_key: label} 专家标注字典。
    sigma_multiplier : 当前使用的 σ 缩放系数。
    expert_weight : 当前使用的偏置权重倍数。
    """
    has_rk = "_record_key" in anomaly_df.columns
    skills: List[dict] = []

    for short_name, group in anomaly_df.groupby("备件简称", sort=True):
        costs = group["实际成本"].to_numpy(dtype=float)

        # 本组专家标注数
        if has_rk:
            rkeys = set(group["_record_key"].values)
            expert_count = sum(1 for k, v in expert_labels.items() if v == "正常" and k in rkeys)
        else:
            expert_count = 0

        dist: dict = {
            "样本量": int(len(costs)),
            "均值": round(float(np.mean(costs)), 4),
            "标准差": round(float(np.std(costs)), 4),
            "中位数": round(float(np.median(costs)), 4),
            "最小值": round(float(np.min(costs)), 4),
            "最大值": round(float(np.max(costs)), 4),
        }
        if len(costs) > 2:
            dist["偏度"] = round(float(pd.Series(costs).skew()), 4)

        skill = {
            "备件简称": str(short_name),
            "适用算法": "KDE+KNN+Elbow 密度连接异常检测",
            "数据结构分布描述": dist,
            "当前σ参数": round(sigma_multiplier, 4),
            "偏置权重": expert_weight,
            "本组专家标注数": expert_count,
            "成本合理区间边界": {
                "预测值": round(float(group["预测值"].iloc[0]), 4),
                "合理下限": round(float(group["合理下限"].iloc[0]), 4),
                "合理上限": round(float(group["合理上限"].iloc[0]), 4),
            },
            "异常统计": {
                "正常": int(group["status"].astype(str).str.contains("正常").sum()),
                "异常偏高": int(group["status"].astype(str).str.contains("异常偏高").sum()),
                "异常偏低": int(
                    group["status"].astype(str).str.contains("异常偏低|严重异常偏低").sum()
                ),
            },
        }

        # ── 经验对齐率：专家标注"正常"的记录中，实际落在合理区间内的比例 ──
        if has_rk and expert_count > 0:
            lower = float(group["合理下限"].iloc[0])
            upper = float(group["合理上限"].iloc[0])
            cost_map = dict(zip(group["_record_key"], group["实际成本"].astype(float)))
            aligned = sum(
                1 for k, v in expert_labels.items()
                if v == "正常" and k in cost_map and lower <= cost_map[k] <= upper
            )
            skill["经验对齐率"] = round(aligned / expert_count, 4)
        else:
            skill["经验对齐率"] = "N/A"

        skills.append(skill)

    return skills


def skills_to_json_bytes(skills: List[dict]) -> bytes:
    """导出 Skills 为可下载的 JSON 字节。"""
    payload = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "skills_count": len(skills),
        "skills": skills,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def skills_to_markdown(skills: List[dict]) -> str:
    """将 Skills 转为人类可读的 Markdown 技能书报告。"""
    lines = [
        "# Skills 技能书报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**备件简称总数**: {len(skills)}",
        "",
        "---",
        "",
    ]

    for i, sk in enumerate(skills, 1):
        lines.append(f"## {i}. {sk['备件简称']}")
        lines.append("")
        lines.append(f"- **适用算法**: {sk['适用算法']}")
        lines.append(f"- **当前 σ 参数**: {sk['当前σ参数']}")
        lines.append(f"- **偏置权重**: {sk['偏置权重']}×")
        lines.append(f"- **本组专家标注数**: {sk['本组专家标注数']}")
        align = sk.get("经验对齐率", "N/A")
        if isinstance(align, float):
            lines.append(f"- **经验对齐率**: {align:.2%}")
        else:
            lines.append(f"- **经验对齐率**: {align}")
        lines.append("")

        bounds = sk["成本合理区间边界"]
        lines.append("### 成本合理区间")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 预测值（基准合理价） | {bounds['预测值']:,.4f} |")
        lines.append(f"| 合理下限 | {bounds['合理下限']:,.4f} |")
        lines.append(f"| 合理上限 | {bounds['合理上限']:,.4f} |")
        lines.append("")

        dist = sk["数据结构分布描述"]
        lines.append("### 数据分布特征")
        lines.append("")
        lines.append("| 统计量 | 数值 |")
        lines.append("|--------|------|")
        for k, v in dist.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:,.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

        stats = sk["异常统计"]
        lines.append("### 异常统计")
        lines.append("")
        lines.append("| 分类 | 数量 |")
        lines.append("|------|------|")
        for k, v in stats.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── 2. AutoResearch 棘轮迭代 ──────────────────────────────────────────────


def _score_params(
    result_df: pd.DataFrame,
    expert_labels: Dict[str, str],
) -> tuple:
    """准确率得分 + 冲突数。

    返回 (score, conflict_count, total_expert)。
    score = 自然落入合理区间的专家标注记录比例。
    conflict_count = 未落入合理区间的专家标注记录数（冲突数）。
    """
    normal_keys = {k for k, v in expert_labels.items() if v == "正常"}
    if not normal_keys:
        return 1.0, 0, 0

    total = 0
    correct = 0
    for key in normal_keys:
        rows = result_df[result_df["_record_key"] == key]
        if rows.empty:
            continue
        total += 1
        r = rows.iloc[0]
        if r["合理下限"] <= r["实际成本"] <= r["合理上限"]:
            correct += 1

    score = correct / total if total > 0 else 0.0
    conflict = total - correct
    return score, conflict, total


def run_auto_research(
    df: pd.DataFrame,
    price_col: str,
    expert_labels: Dict[str, str],
    n_iterations: int = 10,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """棘轮循环自动调参。

    Parameters
    ----------
    df : 全量原始数据（含 price_col, monitor_date）。
    price_col : 价格列名。
    expert_labels : {record_key: label} 专家标注。
    n_iterations : 迭代轮数（5 / 10 / 20）。
    progress_callback : ``(current, total, best_score, trial_score, sigma)`` 回调。

    Returns
    -------
    dict
        best_sigma, best_weight, best_score, history, result_df
    """
    labels_tuple = tuple(sorted(expert_labels.items()))

    if not expert_labels:
        baseline_df = processor.detect_cost_anomalies_weighted(df, price_col, ())
        return {
            "best_sigma": 1.0,
            "best_weight": processor._EXPERT_WEIGHT,
            "best_score": 1.0,
            "best_conflicts": 0,
            "total_expert": 0,
            "history": [],
            "result_df": baseline_df,
        }

    rng = np.random.RandomState(42)

    # ── 初始基线 ──
    best_sigma = 1.0
    best_weight = processor._EXPERT_WEIGHT
    best_df = processor.detect_cost_anomalies_weighted(
        df, price_col, labels_tuple,
        sigma_multiplier=best_sigma,
        expert_weight_override=best_weight,
    )
    best_score, best_conflicts, total_expert = _score_params(best_df, expert_labels)

    prev_best = {"sigma": best_sigma, "weight": best_weight, "score": best_score, "conflicts": best_conflicts}
    rollback_used = False

    history = [
        {
            "迭代": 0,
            "σ系数": best_sigma,
            "偏置权重": best_weight,
            "得分": round(best_score, 4),
            "冲突数": best_conflicts,
            "是否采纳": "✅",
            "备注": "初始基线",
        }
    ]

    for i in range(n_iterations):
        # 随机扰动（围绕当前最佳微调）
        trial_sigma = best_sigma * (1 + rng.uniform(-0.3, 0.4))
        trial_weight = best_weight + int(rng.randint(-40, 80))
        trial_sigma = max(0.1, min(5.0, round(trial_sigma, 4)))
        trial_weight = max(1, min(500, trial_weight))

        try:
            trial_df = processor.detect_cost_anomalies_weighted(
                df, price_col, labels_tuple,
                sigma_multiplier=trial_sigma,
                expert_weight_override=trial_weight,
            )
            trial_score, trial_conflicts, _ = _score_params(trial_df, expert_labels)
        except Exception as exc:
            note = f"计算错误: {exc}"
            if not rollback_used:
                best_sigma = prev_best["sigma"]
                best_weight = prev_best["weight"]
                best_score = prev_best["score"]
                best_conflicts = prev_best["conflicts"]
                note += "（已回滚一个版本）"
                rollback_used = True
            history.append({
                "迭代": i + 1,
                "σ系数": trial_sigma,
                "偏置权重": trial_weight,
                "得分": None,
                "冲突数": None,
                "是否采纳": "❌",
                "备注": note,
            })
            if progress_callback:
                progress_callback(i + 1, n_iterations, best_score, 0.0, best_sigma)
            continue

        # 棘轮：仅当冲突数减少（或同等冲突但得分更高）时才采纳
        accepted = (trial_conflicts < best_conflicts) or (
            trial_conflicts == best_conflicts and trial_score > best_score
        )
        if accepted:
            prev_best = {"sigma": best_sigma, "weight": best_weight, "score": best_score, "conflicts": best_conflicts}
            best_sigma = trial_sigma
            best_weight = trial_weight
            best_score = trial_score
            best_conflicts = trial_conflicts
            best_df = trial_df
            note = f"冲突 {prev_best['conflicts']}→{best_conflicts} 得分 {prev_best['score']:.2%}→{best_score:.2%}"
        else:
            note = f"冲突 {trial_conflicts}≥{best_conflicts} 得分 {trial_score:.2%}≤{best_score:.2%}"

        history.append({
            "迭代": i + 1,
            "σ系数": trial_sigma,
            "偏置权重": trial_weight,
            "得分": round(trial_score, 4),
            "冲突数": trial_conflicts,
            "是否采纳": "✅" if accepted else "❌",
            "备注": note,
        })

        if progress_callback:
            progress_callback(i + 1, n_iterations, best_score, trial_score, best_sigma)

        if best_score >= 1.0 and best_conflicts == 0:
            break

    return {
        "best_sigma": round(best_sigma, 4),
        "best_weight": best_weight,
        "best_score": round(best_score, 4),
        "best_conflicts": best_conflicts,
        "total_expert": total_expert,
        "history": history,
        "result_df": best_df,
    }


# ── 3. 深度审计报表 ──────────────────────────────────────────────────────


def generate_audit_report(
    original_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    expert_labels: Dict[str, str],
) -> pd.DataFrame:
    """生成全量深度审计报表。

    字段：物料编码、物料名称、备件简称、成本数值、原始结论、专家反馈、最终优化结论
    """
    report = original_df[["_record_key", "物料编码"]].copy()

    report["物料名称"] = (
        original_df["物料名称"].values
        if "物料名称" in original_df.columns
        else "未知"
    )
    report["备件简称"] = original_df["备件简称"].values
    report["成本数值"] = original_df["实际成本"].values
    report["原始结论"] = original_df["status"].values

    report["专家反馈"] = report["_record_key"].map(
        lambda k: expert_labels.get(k, "未标注")
    )

    opt_map = dict(zip(optimized_df["_record_key"], optimized_df["status"]))
    report["最终优化结论"] = report["_record_key"].map(
        lambda k: opt_map.get(k, "—")
    )

    report = report.drop(columns=["_record_key"])
    return report
