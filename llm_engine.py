from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

from config import settings
import processor


_SYSTEM_PROMPT = """
你是一名资深采购审计师，负责把专家备注蒸馏成可复用的定价知识。
请严格根据输入记录总结，不要臆造不存在的供应商、车系或材料原因。
输出必须是单个 JSON 对象，不要附加额外解释。
""".strip()


def is_llm_configured() -> bool:
    return bool(settings.llm_api_key)


def build_rule_id(short_name: str, supplier_code: str, vehicle_series: str) -> str:
    raw_key = f"{short_name.strip()}|{supplier_code.strip()}|{vehicle_series.strip()}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _join_unique(values: Sequence[Any], default: str = "") -> str:
    cleaned = sorted({_normalize_text(value) for value in values if _normalize_text(value)})
    if not cleaned:
        return default
    return " / ".join(cleaned)


def _chat_completions_url() -> str:
    base = settings.llm_api_base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        text = fence_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _coerce_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.35
    if score > 1 and score <= 100:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _call_llm(user_prompt: str) -> Dict[str, Any]:
    payload = {
        "model": settings.llm_api_model,
        "temperature": settings.llm_temperature,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        _chat_completions_url(),
        headers=headers,
        json=payload,
        timeout=settings.llm_timeout_seconds,
    )
    response.raise_for_status()
    response_json = response.json()
    content = (
        response_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not content:
        raise ValueError("LLM 响应缺少 message.content")
    return _extract_json_object(content)


def _load_feedback_records_for_knowledge() -> pd.DataFrame:
    feedback_df = processor.label_manager.get_label_records()
    if feedback_df.empty:
        return pd.DataFrame()

    anomaly_df = processor.load_cost_anomaly_results(result_mode="weighted")
    if anomaly_df.empty:
        anomaly_df = processor.load_cost_anomaly_results(result_mode="raw")
    if anomaly_df.empty:
        return pd.DataFrame()

    anomaly_df = anomaly_df.copy()
    anomaly_df["_record_key"] = anomaly_df["_record_key"].astype(str)
    anomaly_df = anomaly_df.rename(
        columns={
            "备件简称": "short_name",
            "适用车系": "vehicle_series",
            "实际成本": "actual_price",
            "预测值": "predicted_price",
            "物料编码": "material_code",
            "价格有效于": "effective_date",
        }
    )

    core_df, _, _ = processor.load_core_cost_records()
    if core_df is None or core_df.empty:
        core_lookup = pd.DataFrame(columns=["_record_key", "supplier_name", "supplier_code", "short_name_fallback", "vehicle_series_fallback"])
    else:
        core_lookup = core_df.copy()
        core_lookup["价格有效于"] = pd.to_datetime(core_lookup.get("monitor_date"), errors="coerce")
        core_lookup["实际成本"] = pd.to_numeric(core_lookup.get("成本"), errors="coerce")
        core_lookup["_record_key"] = core_lookup.apply(processor.make_record_key, axis=1)
        core_lookup = core_lookup.rename(
            columns={
                "一级总成供应商名称": "supplier_name",
                "一级总成供应商代码": "supplier_code",
                "备件简称": "short_name_fallback",
                "适用车系": "vehicle_series_fallback",
            }
        )
        core_lookup = core_lookup[
            [
                "_record_key",
                "supplier_name",
                "supplier_code",
                "short_name_fallback",
                "vehicle_series_fallback",
            ]
        ].drop_duplicates(subset=["_record_key"], keep="last")

    merged = feedback_df.merge(
        anomaly_df[
            [
                "_record_key",
                "short_name",
                "vehicle_series",
                "actual_price",
                "predicted_price",
                "material_code",
                "effective_date",
            ]
        ],
        left_on="record_key",
        right_on="_record_key",
        how="left",
    )
    merged = merged.merge(core_lookup, on="_record_key", how="left")

    merged["remark"] = merged["remark"].fillna("").astype(str).str.strip()
    merged["short_name"] = merged["short_name"].fillna(merged.get("short_name_fallback", "")).astype(str)
    merged["vehicle_series"] = merged["vehicle_series"].fillna(merged.get("vehicle_series_fallback", "")).astype(str)
    merged["supplier_name"] = merged.get("supplier_name", "").fillna("").astype(str)
    merged["supplier_code"] = merged.get("supplier_code", "").fillna("").astype(str)
    merged["material_code"] = merged.get("material_code", "").fillna("").astype(str)
    merged["actual_price"] = pd.to_numeric(merged.get("actual_price"), errors="coerce")
    merged["predicted_price"] = pd.to_numeric(merged.get("predicted_price"), errors="coerce")
    merged["effective_date"] = pd.to_datetime(merged.get("effective_date"), errors="coerce")
    merged["labeled_at"] = pd.to_datetime(merged.get("labeled_at"), errors="coerce")

    merged["short_name"] = merged["short_name"].replace("", "未知备件")
    merged["vehicle_series"] = merged["vehicle_series"].replace("", "未识别车系")
    merged["group_mode"] = merged["supplier_code"].apply(lambda value: "supplier" if _normalize_text(value) else "vehicle")
    merged["group_scope"] = merged.apply(
        lambda row: _normalize_text(row["supplier_code"], default="未识别供应商")
        if row["group_mode"] == "supplier"
        else _normalize_text(row["vehicle_series"], default="未识别车系"),
        axis=1,
    )
    merged["group_key"] = merged.apply(
        lambda row: f"{row['short_name']}||{row['group_mode']}||{row['group_scope']}",
        axis=1,
    )
    return merged


def _build_group_payloads(records_df: pd.DataFrame) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    if records_df.empty:
        return payloads

    for _, group in records_df.groupby("group_key", sort=True):
        group = group.sort_values(["labeled_at", "record_key"], na_position="last")
        short_name = _normalize_text(group["short_name"].iloc[0], "未知备件")
        group_mode = _normalize_text(group["group_mode"].iloc[0], "vehicle")
        supplier_code = _join_unique(group["supplier_code"], default="") if group_mode == "supplier" else ""
        supplier_name = _join_unique(group["supplier_name"], default="")
        vehicle_series = _join_unique(group["vehicle_series"], default="未识别车系")
        rule_id = build_rule_id(short_name, supplier_code, vehicle_series)

        evidence_rows = []
        for row in group.to_dict(orient="records"):
            actual_price = row.get("actual_price")
            predicted_price = row.get("predicted_price")
            deviation_pct = None
            if predicted_price not in (None, 0) and not pd.isna(predicted_price) and not pd.isna(actual_price):
                deviation_pct = (float(actual_price) - float(predicted_price)) / float(predicted_price)
            evidence_rows.append(
                {
                    "record_key": row.get("record_key", ""),
                    "material_code": _normalize_text(row.get("material_code")),
                    "vehicle_series": _normalize_text(row.get("vehicle_series"), "未识别车系"),
                    "supplier_name": _normalize_text(row.get("supplier_name"), "未提供"),
                    "supplier_code": _normalize_text(row.get("supplier_code"), "未提供"),
                    "actual_price": None if pd.isna(actual_price) else round(float(actual_price), 4),
                    "predicted_price": None if pd.isna(predicted_price) else round(float(predicted_price), 4),
                    "deviation_pct": None if deviation_pct is None else round(float(deviation_pct), 4),
                    "label": _normalize_text(row.get("label"), "未标注"),
                    "remark": _normalize_text(row.get("remark")),
                    "effective_date": row.get("effective_date").strftime("%Y-%m-%d")
                    if pd.notna(row.get("effective_date"))
                    else "",
                }
            )

        payloads.append(
            {
                "rule_id": rule_id,
                "short_name": short_name,
                "supplier_code": supplier_code,
                "supplier_name": supplier_name,
                "vehicle_series": vehicle_series,
                "group_mode": group_mode,
                "records": evidence_rows,
            }
        )

    return payloads


def _build_distillation_prompt(group_payload: Dict[str, Any]) -> str:
    lines = [
        "请基于以下专家标注记录，蒸馏一条可复用的备件定价经验规则。",
        f"分组方式: {'备件简称+供应商' if group_payload['group_mode'] == 'supplier' else '备件简称+车系'}",
        f"备件简称: {group_payload['short_name']}",
        f"供应商代码: {group_payload['supplier_code'] or '未提供'}",
        f"供应商名称: {group_payload['supplier_name'] or '未提供'}",
        f"车系: {group_payload['vehicle_series'] or '未提供'}",
        "",
        "记录明细:",
    ]
    for idx, record in enumerate(group_payload["records"], start=1):
        deviation_display = "未知"
        if record["deviation_pct"] is not None:
            deviation_display = f"{record['deviation_pct']:.2%}"
        lines.extend(
            [
                f"{idx}. 物料编码={record['material_code'] or '未知'} | 实际价格={record['actual_price']} | 预测值={record['predicted_price']} | 偏离={deviation_display}",
                f"   专家标注={record['label']} | 车系={record['vehicle_series']} | 供应商={record['supplier_name']}({record['supplier_code']})",
                f"   专家备注={record['remark']}",
            ]
        )

    lines.extend(
        [
            "",
            "请严格输出如下 JSON 对象，不要输出其它文字:",
            "{",
            '  "rule_content": "2到4句中文规则总结，指出溢价/降价原因、适用范围和采购判断",',
            '  "confidence_score": 0.0,',
            '  "key_signals": ["最多3条证据点"],',
            '  "markdown_summary": "以 ### 标题开头的 Markdown 摘要"',
            "}",
            "如果证据不足，也要给出谨慎结论，并降低 confidence_score。",
        ]
    )
    return "\n".join(lines)


def distill_expert_knowledge(records_df: pd.DataFrame) -> Dict[str, Any]:
    grouped_payloads = _build_group_payloads(records_df)
    if not grouped_payloads:
        return {"rules": [], "markdown_report": "# 专家经验知识库\n\n暂无可蒸馏的专家备注。", "json_summary": []}

    rules: List[Dict[str, Any]] = []
    markdown_blocks: List[str] = ["# 专家经验知识库", "", f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for group_payload in grouped_payloads:
        llm_result = _call_llm(_build_distillation_prompt(group_payload))
        rule_content = _normalize_text(llm_result.get("rule_content"), "证据不足，暂无法形成稳定规则。")
        confidence_score = _coerce_confidence(llm_result.get("confidence_score"))
        key_signals = llm_result.get("key_signals")
        if not isinstance(key_signals, list):
            key_signals = []
        markdown_summary = _normalize_text(llm_result.get("markdown_summary"))
        if not markdown_summary:
            markdown_summary = (
                f"### {group_payload['short_name']}\n\n"
                f"- 规则: {rule_content}\n"
                f"- 置信度: {confidence_score:.0%}\n"
            )

        persisted_rule = {
            "rule_id": group_payload["rule_id"],
            "short_name": group_payload["short_name"],
            "supplier_code": group_payload["supplier_code"],
            "vehicle_series": group_payload["vehicle_series"],
            "rule_content": rule_content,
            "confidence_score": confidence_score,
            "updated_at": datetime.now(),
        }
        rules.append(
            {
                **persisted_rule,
                "key_signals": key_signals[:3],
                "markdown_summary": markdown_summary,
            }
        )
        markdown_blocks.extend([markdown_summary, ""])

    return {
        "rules": rules,
        "markdown_report": "\n".join(markdown_blocks).strip(),
        "json_summary": rules,
    }


def sync_expert_knowledge_base(force_full: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "skipped",
        "message": "",
        "updated_rule_count": 0,
        "deleted_rule_count": 0,
        "failure_count": 0,
        "markdown_report": "",
        "json_summary": [],
    }

    knowledge_df = processor.load_expert_knowledge_base()
    if not is_llm_configured():
        result["status"] = "skipped"
        result["message"] = "LLM 未配置，已跳过知识蒸馏。"
        result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
        result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
        return result

    try:
        source_df = _load_feedback_records_for_knowledge()
        if source_df.empty:
            result["status"] = "no_data"
            result["message"] = "暂无可供蒸馏的专家备注或缺少异常测算上下文。"
            result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
            result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
            return result

        last_updated = None if force_full else processor.get_expert_knowledge_last_updated_at()
        if last_updated is None:
            affected_group_keys = set(source_df[source_df["remark"] != ""]["group_key"])
        else:
            affected_group_keys = set(source_df[source_df["labeled_at"] > last_updated]["group_key"])

        if not affected_group_keys:
            result["status"] = "no_changes"
            result["message"] = "没有新增或修改过的专家备注，已跳过增量蒸馏。"
            result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
            result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
            return result

        eligible_df = source_df[source_df["remark"] != ""].copy()
        affected_records_df = eligible_df[eligible_df["group_key"].isin(affected_group_keys)].copy()

        stale_rule_ids = []
        remaining_group_keys = set(affected_records_df["group_key"])
        for group_key in sorted(affected_group_keys - remaining_group_keys):
            group_rows = source_df[source_df["group_key"] == group_key]
            if group_rows.empty:
                continue
            short_name = _normalize_text(group_rows["short_name"].iloc[0], "未知备件")
            group_mode = _normalize_text(group_rows["group_mode"].iloc[0], "vehicle")
            supplier_code = _join_unique(group_rows["supplier_code"], default="") if group_mode == "supplier" else ""
            vehicle_series = _join_unique(group_rows["vehicle_series"], default="未识别车系")
            stale_rule_ids.append(build_rule_id(short_name, supplier_code, vehicle_series))

        if stale_rule_ids:
            result["deleted_rule_count"] = processor.delete_expert_knowledge_rules(stale_rule_ids)

        if affected_records_df.empty:
            knowledge_df = processor.load_expert_knowledge_base()
            result["status"] = "success"
            result["message"] = f"已清理 {result['deleted_rule_count']} 条失效知识规则。"
            result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
            result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
            return result

        distillation_result = distill_expert_knowledge(affected_records_df)
        persisted_rules = []
        for rule in distillation_result["rules"]:
            persisted_rules.append(
                {
                    "rule_id": rule["rule_id"],
                    "short_name": rule["short_name"],
                    "supplier_code": rule["supplier_code"],
                    "vehicle_series": rule["vehicle_series"],
                    "rule_content": rule["rule_content"],
                    "confidence_score": rule["confidence_score"],
                    "updated_at": rule["updated_at"],
                }
            )
        result["updated_rule_count"] = processor.save_expert_knowledge_rules(persisted_rules)
        knowledge_df = processor.load_expert_knowledge_base()
        result["status"] = "success"
        result["message"] = (
            f"已更新 {result['updated_rule_count']} 条 AI 经验规则"
            + (f"，并清理 {result['deleted_rule_count']} 条失效规则。" if result["deleted_rule_count"] else "。")
        )
        result["markdown_report"] = distillation_result["markdown_report"]
        result["json_summary"] = distillation_result["json_summary"]
        result["knowledge_df"] = knowledge_df
        return result
    except requests.RequestException as exc:
        result["status"] = "error"
        result["message"] = f"LLM 网络请求失败：{exc}"
    except Exception as exc:
        result["status"] = "error"
        result["message"] = f"AI 知识蒸馏失败：{exc}"

    knowledge_df = processor.load_expert_knowledge_base()
    result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
    result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
    return result


def knowledge_base_to_markdown(knowledge_df: Optional[pd.DataFrame] = None) -> str:
    if knowledge_df is None:
        knowledge_df = processor.load_expert_knowledge_base()
    if knowledge_df is None or knowledge_df.empty:
        return "# 专家经验知识库\n\n暂无 AI 蒸馏出的专家经验规则。"

    lines = [
        "# 专家经验知识库",
        "",
        f"规则总数: {len(knowledge_df)}",
        "",
    ]
    for _, row in knowledge_df.iterrows():
        supplier_code = _normalize_text(row.get("supplier_code"), "未提供")
        vehicle_series = _normalize_text(row.get("vehicle_series"), "未提供")
        confidence = _coerce_confidence(row.get("confidence_score"))
        updated_at = pd.to_datetime(row.get("updated_at"), errors="coerce")
        updated_text = updated_at.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(updated_at) else "未知"
        lines.extend(
            [
                f"## {row.get('short_name', '未知备件')}",
                "",
                f"- 供应商代码: {supplier_code}",
                f"- 车系: {vehicle_series}",
                f"- 可信度: {confidence:.0%}",
                f"- 更新时间: {updated_text}",
                f"- 经验规则: {row.get('rule_content', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def knowledge_base_to_json_bytes(knowledge_df: Optional[pd.DataFrame] = None) -> bytes:
    if knowledge_df is None:
        knowledge_df = processor.load_expert_knowledge_base()
    payload = {
        "generated_at": datetime.now().isoformat(),
        "rules": knowledge_df.to_dict(orient="records") if knowledge_df is not None and not knowledge_df.empty else [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")