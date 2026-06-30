from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import pandas as pd
import requests

from config import settings
import harness
from storage_service import canonicalize_record_key, make_record_key


_SYSTEM_PROMPT = """
你是一名资深采购审计师，负责把专家备注蒸馏成可复用的定价知识。
请严格根据输入记录总结，不要臆造不存在的供应商、车系或材料原因。
输出必须是单个 JSON 对象，不要附加额外解释。
""".strip()
_VEHICLE_MARKET_PRICE_BATCH_SIZE = 5


def is_llm_configured() -> bool:
    return bool(_iter_llm_api_configs())


def build_rule_id(short_name: str, supplier_code: str, vehicle_series: str) -> str:
    raw_key = f"{short_name.strip()}|{supplier_code.strip()}|{vehicle_series.strip()}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return default
    return text if text else default


def _join_unique(values: Sequence[Any], default: str = "") -> str:
    cleaned = sorted({_normalize_text(value) for value in values if _normalize_text(value)})
    if not cleaned:
        return default
    return " / ".join(cleaned)


_LLM_ALLOWED_PROMPT_KEYS = {
    "direction",
    "material_code",
    "material_codes",
    "material_name",
    "material_names",
    "records",
    "remark",
    "representative_material_code",
    "representative_material_name",
    "rule_template",
    "short_name",
    "status",
    "supplier_code",
    "supplier_name",
    "vehicle_series",
}
_LLM_BLOCKED_KEY_TOKENS = (
    "actual",
    "amount",
    "baseline",
    "bound",
    "cost",
    "deviation",
    "lower",
    "predicted",
    "price",
    "ratio",
    "sample",
    "sigma",
    "upper",
    "weight",
)


def _sanitize_remark_text(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?%?", "[数值已脱敏]", text)


def sanitize_data_for_llm(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized: Dict[str, Any] = {}
        for raw_key, raw_value in payload.items():
            key = _normalize_text(raw_key)
            lowered_key = key.lower()
            if not key or key not in _LLM_ALLOWED_PROMPT_KEYS:
                continue
            if any(token in lowered_key for token in _LLM_BLOCKED_KEY_TOKENS):
                continue

            if key == "remark":
                sanitized_value = _sanitize_remark_text(raw_value)
            else:
                sanitized_value = sanitize_data_for_llm(raw_value)

            if sanitized_value in (None, "", [], {}):
                continue
            sanitized[key] = sanitized_value
        return sanitized

    if isinstance(payload, list):
        sanitized_list = []
        for item in payload:
            sanitized_item = sanitize_data_for_llm(item)
            if sanitized_item not in (None, "", [], {}):
                sanitized_list.append(sanitized_item)
        return sanitized_list

    if isinstance(payload, str):
        return _normalize_text(payload)

    return None


def _iter_llm_api_configs() -> List[Dict[str, Any]]:
    configured = getattr(settings, "llm_api_configs", None)
    if isinstance(configured, list) and configured:
        return [
            {
                "name": _normalize_text(item.get("name"), item.get("model", "LLM")),
                "api_key": _normalize_text(item.get("api_key")),
                "base_url": _normalize_text(item.get("base_url")),
                "model": _normalize_text(item.get("model")),
                "direct_url": bool(item.get("direct_url", False)),
                "append_no_think": bool(item.get("append_no_think", False)),
            }
            for item in configured
            if isinstance(item, dict)
            and _normalize_text(item.get("api_key"))
            and _normalize_text(item.get("base_url"))
            and _normalize_text(item.get("model"))
        ]

    if not (settings.llm_api_key and settings.llm_api_base_url and settings.llm_api_model):
        return []
    return [
        {
            "name": _normalize_text(getattr(settings, "llm_api_name", ""), settings.llm_api_model),
            "api_key": settings.llm_api_key,
            "base_url": settings.llm_api_base_url,
            "model": settings.llm_api_model,
            "direct_url": bool(getattr(settings, "llm_api_direct_url", False)),
            "append_no_think": bool(getattr(settings, "llm_api_append_no_think", False)),
        }
    ]


def _chat_completions_url(config: Optional[Dict[str, Any]] = None) -> str:
    llm_config = config or (_iter_llm_api_configs()[0] if _iter_llm_api_configs() else {})
    base_url = str(llm_config.get("base_url") or settings.llm_api_base_url)
    direct_url = bool(llm_config.get("direct_url", getattr(settings, "llm_api_direct_url", False)))
    if direct_url:
        direct = _normalize_direct_llm_url(base_url)
        lowered = direct.lower()
        if direct.endswith("/*") or lowered.endswith("/%2a"):
            return f"{direct.rsplit('/', 1)[0]}/chat/completions"
        if direct.endswith("*"):
            return f"{direct[:-1].rstrip('/')}/chat/completions"
        return direct.rstrip("/")
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _normalize_direct_llm_url(url: str) -> str:
    normalized = str(url or "").strip()
    return normalized


def _messages_for_llm_config(messages: List[Dict[str, str]], config: Dict[str, Any]) -> List[Dict[str, str]]:
    prepared_messages = [dict(message) for message in messages]
    if not bool(config.get("append_no_think", False)):
        return prepared_messages

    for message in prepared_messages:
        if _normalize_text(message.get("role")).lower() != "user":
            continue
        content = str(message.get("content") or "").rstrip()
        if content and not content.endswith("/no_think"):
            message["content"] = f"{content}/no_think"
    return prepared_messages


def _official_byd_domain(source_url: Any) -> str:
    parsed = urlparse(str(source_url or "").strip())
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    for domain in ("byd.com", "bydauto.com.cn"):
        if host == domain or host.endswith(f".{domain}"):
            return domain
    return ""


def _parse_market_price_to_yuan(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        numeric = float(value)
        return numeric * 10000 if numeric < 10000 else numeric

    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    numeric = float(match.group(0))
    if "万" in text or numeric < 10000:
        numeric *= 10000
    return numeric


def normalize_vehicle_market_price_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    vehicle_series = _normalize_text(raw_result.get("vehicle_series") or raw_result.get("车系"))
    source_url = _normalize_text(raw_result.get("source_url") or raw_result.get("官方URL") or raw_result.get("来源链接"))
    source_domain = _official_byd_domain(source_url)
    market_price = _parse_market_price_to_yuan(raw_result.get("market_price") or raw_result.get("price") or raw_result.get("价格"))
    variant_name = _normalize_text(raw_result.get("variant_name") or raw_result.get("车型") or raw_result.get("次顶配车型"))
    failure_reason = _normalize_text(raw_result.get("failure_reason") or raw_result.get("失败原因"))
    raw_status = _normalize_text(raw_result.get("status") or raw_result.get("状态"))

    if vehicle_series and market_price is not None:
        status = raw_status if raw_status in {"LLM估算", "人工修正", "已确认"} else "LLM估算"
        failure_reason = ""
    else:
        status = "待确认"
        market_price = None
        if not failure_reason:
            failure_reason = "LLM 未返回有效估算价格"

    return {
        "vehicle_series": vehicle_series,
        "market_price": market_price,
        "variant_name": variant_name,
        "source_url": source_url,
        "source_domain": source_domain,
        "status": status,
        "fetched_at": datetime.now(),
        "failure_reason": failure_reason,
        "raw_response_json": json.dumps(raw_result, ensure_ascii=False, default=str),
    }


def _coerce_vehicle_market_price_items(parsed_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed_payload, list):
        raw_items = parsed_payload
    elif isinstance(parsed_payload, dict):
        raw_items = parsed_payload.get("results", parsed_payload.get("items", parsed_payload.get("data")))
        if raw_items is None and (
            parsed_payload.get("vehicle_series")
            or parsed_payload.get("车系")
            or parsed_payload.get("market_price")
            or parsed_payload.get("price")
            or parsed_payload.get("价格")
        ):
            raw_items = [parsed_payload]
    else:
        raw_items = []

    if isinstance(raw_items, dict):
        if (
            raw_items.get("vehicle_series")
            or raw_items.get("车系")
            or raw_items.get("market_price")
            or raw_items.get("price")
            or raw_items.get("价格")
        ):
            raw_items = [raw_items]
        else:
            raw_items = list(raw_items.values())

    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _build_pending_vehicle_market_price_rows(vehicles: Sequence[str], failure_reason: str) -> List[Dict[str, Any]]:
    reason = _normalize_text(failure_reason, "LLM 自动估算失败，请人工确认车系梯度。")
    if len(reason) > 800:
        reason = f"{reason[:800]}..."
    return [
        normalize_vehicle_market_price_result(
            {
                "vehicle_series": vehicle,
                "failure_reason": reason,
            }
        )
        for vehicle in vehicles
    ]


def _is_complete_vehicle_market_price_row(row: Dict[str, Any]) -> bool:
    return bool(_normalize_text(row.get("vehicle_series"))) and row.get("market_price") is not None and bool(
        _normalize_text(row.get("variant_name"))
    )


def _vehicle_series_key(value: Any) -> str:
    return _normalize_text(value).replace(" ", "").lower()


def _build_vehicle_market_price_messages(vehicles: Sequence[str], *, repair_reason: str = "") -> List[Dict[str, str]]:
    task_prefix = "请基于你自身知识查阅并估算列表中所有比亚迪车系的次顶配车型价格，用于生成车系价格梯度。"
    if repair_reason:
        task_prefix = f"上一次响应存在空白或缺失字段：{repair_reason}。请重新补全这些车系的次顶配车型价格。"
    user_prompt = json.dumps(
        {
            "task": (
                f"{task_prefix}请先完成全局比较，再把所有车系按次顶配价格从高到低排序输出。"
                "不要联网，不需要官方URL。必须返回 JSON，不要附加解释。"
                "强制要求：每辆车的结果不能为空值，variant_name 和 market_price 都不得为空；严禁 null、None、空字符串。"
                "强制输出价格：即使你不完全确定，也必须基于自身知识给出最可能的次顶配车型名称和估算价格。"
                "强制、强制、强制：market_price 必须是人民币元数值，不能留空，不能返回未知，不能要求人工补填。"
            ),
            "accuracy_requirement": (
                "尽可能准确。优先依据你已知的厂商指导价、公开上市售价、官方车型配置、主流汽车媒体公开报价等知识进行估算；"
                "如存在年款差异，选择当前更常见/较新的在售或近似在售版本，并在 basis 里写明依据。"
            ),
            "sort_requirement": "输出数组必须覆盖输入中的所有车系，并按 market_price 从高到低排列；价格最高的车系排在第一位。",
            "non_empty_requirement": "每个输入车系必须且只能输出一条记录；vehicle_series、variant_name、market_price、basis 不得为空；market_price 必须是人民币元数值。",
            "uncertainty_rule": "不允许用空价格表示不确定；不确定时仍强制给估算价格，将 confidence 调低，并在 basis 中说明估算依据。",
            "output_schema": [
                "rank_order",
                "vehicle_series",
                "variant_name",
                "market_price",
                "confidence",
                "basis",
                "failure_reason",
            ],
            "vehicle_series": list(vehicles),
        },
        ensure_ascii=False,
    )

    return [
        {
            "role": "system",
            "content": (
                "你只返回 JSON。价格是用于排序的 LLM 估算值。每个输入车系必须输出完整结果；"
                "variant_name 和 market_price 不得为空，严禁 null、None、空字符串。"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _merge_vehicle_market_price_rows(
    vehicles: Sequence[str],
    primary_rows: Sequence[Dict[str, Any]],
    repair_rows: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    primary_by_vehicle = {
        _vehicle_series_key(row.get("vehicle_series")): row
        for row in primary_rows
        if _vehicle_series_key(row.get("vehicle_series"))
    }
    repair_by_vehicle = {
        _vehicle_series_key(row.get("vehicle_series")): row
        for row in (repair_rows or [])
        if _is_complete_vehicle_market_price_row(row)
    }

    merged_rows: list[dict[str, Any]] = []
    for vehicle in vehicles:
        vehicle_key = _vehicle_series_key(vehicle)
        if vehicle_key in repair_by_vehicle:
            merged_rows.append(repair_by_vehicle[vehicle_key])
        elif vehicle_key in primary_by_vehicle:
            merged_rows.append(primary_by_vehicle[vehicle_key])
        else:
            merged_rows.append(
                normalize_vehicle_market_price_result(
                    {"vehicle_series": vehicle, "failure_reason": "LLM 未返回该车系的估算价格"}
                )
            )

    return sorted(
        merged_rows,
        key=lambda row: (
            row.get("market_price") is None,
            -float(row.get("market_price") or 0),
            _normalize_text(row.get("vehicle_series")),
        ),
    )


def _post_chat_completion_with_fallback(messages: List[Dict[str, str]], *, temperature: float) -> Dict[str, Any]:
    configs = _iter_llm_api_configs()
    if not configs:
        raise RuntimeError("LLM 未配置，无法发起请求。")

    errors: list[str] = []
    for config in configs:
        payload = {
            "model": config["model"],
            "temperature": temperature,
            "messages": _messages_for_llm_config(messages, config),
        }
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                _chat_completions_url(config),
                headers=headers,
                json=payload,
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            config_name = _normalize_text(config.get("name"), config.get("model", "LLM"))
            errors.append(f"{config_name}: {type(exc).__name__}: {exc}")
            continue

    raise RuntimeError("LLM 网络请求失败，所有本地配置均不可用：" + "；".join(errors))


def fetch_vehicle_market_prices(vehicle_series: Sequence[str]) -> List[Dict[str, Any]]:
    vehicles = [_normalize_text(value) for value in vehicle_series if _normalize_text(value)]
    if not vehicles:
        return []
    if not is_llm_configured():
        return _build_pending_vehicle_market_price_rows(vehicles, "LLM 未配置，无法自动估算车系价格")

    result_rows: List[Dict[str, Any]] = []
    for start_index in range(0, len(vehicles), _VEHICLE_MARKET_PRICE_BATCH_SIZE):
        batch_vehicles = vehicles[start_index : start_index + _VEHICLE_MARKET_PRICE_BATCH_SIZE]
        result_rows.extend(_fetch_vehicle_market_price_batch(batch_vehicles))
    return result_rows


def _fetch_vehicle_market_price_batch(vehicles: Sequence[str]) -> List[Dict[str, Any]]:
    messages = _build_vehicle_market_price_messages(vehicles)

    def _execute_request() -> List[Dict[str, Any]]:
        response_json = _post_chat_completion_with_fallback(messages, temperature=0.0)
        content = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            raise ValueError("LLM 响应缺少 message.content")
        parsed = _extract_json_object(content)
        raw_items = _coerce_vehicle_market_price_items(parsed)
        normalized_rows = [normalize_vehicle_market_price_result(item) for item in raw_items]
        primary_rows = _merge_vehicle_market_price_rows(vehicles, normalized_rows)
        incomplete_vehicles = [
            row["vehicle_series"]
            for row in primary_rows
            if not _is_complete_vehicle_market_price_row(row)
        ]
        if not incomplete_vehicles:
            return primary_rows

        repair_messages = _build_vehicle_market_price_messages(
            incomplete_vehicles,
            repair_reason="部分车系缺少次顶配车型名称或估算价格",
        )
        try:
            repair_response_json = _post_chat_completion_with_fallback(repair_messages, temperature=0.0)
            repair_content = (
                repair_response_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not repair_content:
                return primary_rows
            repair_items = _coerce_vehicle_market_price_items(_extract_json_object(repair_content))
            repair_rows = [normalize_vehicle_market_price_result(item) for item in repair_items]
        except (RuntimeError, ValueError, json.JSONDecodeError):
            return primary_rows
        return _merge_vehicle_market_price_rows(vehicles, primary_rows, repair_rows)

    try:
        return harness.run_llm_action(
            "llm_engine.fetch_vehicle_market_prices",
            _execute_request,
            request_payload={"messages": messages},
        )
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return _build_pending_vehicle_market_price_rows(
            vehicles,
            f"LLM 网络请求失败或响应不可用，已转为待确认：{exc}",
        )


def _normalize_gradient_explanation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row_id = _normalize_text(row.get("row_id"))
    gradient_rank = pd.to_numeric(pd.Series([row.get("gradient_rank")]), errors="coerce").iloc[0]
    cost_rank = pd.to_numeric(pd.Series([row.get("cost_rank")]), errors="coerce").iloc[0]
    deviation_rate = pd.to_numeric(pd.Series([row.get("deviation_rate")]), errors="coerce").iloc[0]
    return {
        "row_id": row_id,
        "vehicle_series": _normalize_text(row.get("vehicle_series"), "未识别车系"),
        "part_name": _normalize_text(row.get("part_name"), "未知备件"),
        "gradient_rank": int(gradient_rank) if pd.notna(gradient_rank) else None,
        "cost_rank": int(cost_rank) if pd.notna(cost_rank) else None,
        "deviation_rate": float(deviation_rate) if pd.notna(deviation_rate) else None,
        "is_abnormal": bool(row.get("is_abnormal")),
    }


def _fallback_vehicle_gradient_deviation_explanation(row: Dict[str, Any]) -> str:
    normalized = _normalize_gradient_explanation_row(row)
    vehicle_series = normalized["vehicle_series"]
    part_name = normalized["part_name"]
    gradient_rank = normalized["gradient_rank"]
    cost_rank = normalized["cost_rank"]
    deviation_rate = normalized["deviation_rate"]
    if gradient_rank and cost_rank and deviation_rate is not None:
        return (
            f"按原有逻辑，{vehicle_series}的{part_name}配置梯度排名第{gradient_rank}，"
            f"但最新成本排序第{cost_rank}，偏差率{deviation_rate:.1%}，超过25%阈值，因此判为梯度偏差异常。"
        )
    return (
        f"按原有逻辑，{vehicle_series}的{part_name}缺少有效梯度排名或成本排序，"
        "无法完成梯度一致性校验，需人工复核。"
    )


def _build_vehicle_gradient_deviation_messages(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized_rows = [_normalize_gradient_explanation_row(row) for row in rows if _normalize_text(row.get("row_id"))]
    prompt_payload = {
        "task": (
            "请补全车系梯度偏差异常解释。必须严格按照代码原有逻辑解释："
            "成本排序与配置梯度排名的偏差率 = abs(成本排序 - 梯度排名) / 梯度排名；"
            "偏差率超过 25% 时判定为梯度偏差异常。"
        ),
        "output_rule": "只返回 JSON 数组，每条包含 row_id 和 explanation，不要输出其它文字。",
        "writing_rule": "解释要短、清晰，说明配置梯度排名、成本排序、25%阈值和为什么需要复核。",
        "rows": [
            {
                "row_id": row["row_id"],
                "车系": row["vehicle_series"],
                "备件简称": row["part_name"],
                "配置梯度排名": f"第{row['gradient_rank']}" if row["gradient_rank"] else "缺失",
                "成本排序": f"第{row['cost_rank']}" if row["cost_rank"] else "缺失",
                "偏差率": f"{row['deviation_rate']:.1%}" if row["deviation_rate"] is not None else "无法计算",
                "是否异常": "是" if row["is_abnormal"] else "否",
            }
            for row in normalized_rows
        ],
    }
    return [
        {
            "role": "system",
            "content": "你是采购审计解释助手。你只返回 JSON，且只能解释输入中的车系梯度偏差判定逻辑。",
        },
        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
    ]


def _coerce_vehicle_gradient_explanation_items(parsed_payload: Any) -> list[dict[str, Any]]:
    if isinstance(parsed_payload, dict):
        raw_items = parsed_payload.get("items") or parsed_payload.get("rows") or parsed_payload.get("explanations")
        if raw_items is None and (parsed_payload.get("row_id") or parsed_payload.get("explanation")):
            raw_items = [parsed_payload]
    else:
        raw_items = parsed_payload
    if isinstance(raw_items, dict):
        raw_items = list(raw_items.values())
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def explain_vehicle_gradient_deviations(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    normalized_rows = [_normalize_gradient_explanation_row(row) for row in rows if _normalize_text(row.get("row_id"))]
    fallback_explanations = {
        row["row_id"]: _fallback_vehicle_gradient_deviation_explanation(row)
        for row in normalized_rows
        if row["row_id"]
    }
    if not normalized_rows:
        return {}
    if not is_llm_configured():
        return fallback_explanations

    messages = _build_vehicle_gradient_deviation_messages(normalized_rows)

    def _execute_request() -> Dict[str, str]:
        response_json = _post_chat_completion_with_fallback(messages, temperature=0.0)
        content = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            return fallback_explanations
        parsed = _extract_json_object(content)
        explanations = dict(fallback_explanations)
        for item in _coerce_vehicle_gradient_explanation_items(parsed):
            row_id = _normalize_text(item.get("row_id"))
            explanation = _normalize_text(item.get("explanation") or item.get("解释"))
            if row_id and explanation:
                explanations[row_id] = explanation[:500]
        return explanations

    try:
        return harness.run_llm_action(
            "llm_engine.explain_vehicle_gradient_deviations",
            _execute_request,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError):
        return fallback_explanations


def _extract_json_object(raw_text: str) -> Any:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fence_match:
        text = fence_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        object_start = text.find("{")
        array_start = text.find("[")
        candidates = [idx for idx in [object_start, array_start] if idx >= 0]
        if not candidates:
            raise
        start = min(candidates)
        end_char = "}" if text[start] == "{" else "]"
        end = text.rfind(end_char)
        if end > start:
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
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    def _execute_request() -> Dict[str, Any]:
        response_json = _post_chat_completion_with_fallback(messages, temperature=settings.llm_temperature)
        content = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            raise ValueError("LLM 响应缺少 message.content")
        parsed = _extract_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM 响应不是 JSON 对象")
        return parsed

    return harness.run_llm_action(
        "llm_engine._call_llm",
        _execute_request,
        request_payload={"messages": messages},
    )


def _load_feedback_records_for_knowledge() -> pd.DataFrame:
    feedback_df = harness.execute_action("get_feedback_records")
    if feedback_df.empty:
        return pd.DataFrame()

    feedback_df = feedback_df.copy()
    feedback_df["record_key"] = feedback_df["record_key"].astype(str).map(canonicalize_record_key)

    anomaly_frames: list[pd.DataFrame] = []
    for priority, result_mode in enumerate(["weighted", "raw"]):
        loaded_df = harness.execute_action("load_cost_anomaly_results", result_mode=result_mode)
        if loaded_df is None or loaded_df.empty:
            continue
        anomaly_frames.append(loaded_df.assign(_source_priority=priority))
    if not anomaly_frames:
        return pd.DataFrame()

    anomaly_df = pd.concat(anomaly_frames, ignore_index=True, sort=False)
    anomaly_df["_record_key"] = anomaly_df["_record_key"].astype(str).map(canonicalize_record_key)
    anomaly_df = (
        anomaly_df.sort_values(["_source_priority"], kind="mergesort")
        .drop_duplicates(subset=["_record_key"], keep="first")
        .drop(columns=["_source_priority"], errors="ignore")
    )
    anomaly_df = anomaly_df.rename(
        columns={
            "备件简称": "short_name",
            "适用车系": "vehicle_series",
            "实际成本": "actual_price",
            "预测值": "predicted_price",
            "物料编码": "material_code",
            "物料名称": "material_name",
            "价格有效于": "effective_date",
        }
    )
    for column_name in ["material_name", "status"]:
        if column_name not in anomaly_df.columns:
            anomaly_df[column_name] = ""

    core_df, _, _ = harness.execute_action("load_core_cost_records")
    if core_df is None or core_df.empty:
        core_lookup = pd.DataFrame(
            columns=[
                "_record_key",
                "supplier_name",
                "supplier_code",
                "material_name_fallback",
                "short_name_fallback",
                "vehicle_series_fallback",
            ]
        )
    else:
        core_lookup = core_df.copy()
        core_lookup["价格有效于"] = pd.to_datetime(core_lookup.get("monitor_date"), errors="coerce")
        core_lookup["实际成本"] = pd.to_numeric(core_lookup.get("成本"), errors="coerce")
        core_lookup["_record_key"] = core_lookup.apply(make_record_key, axis=1)
        core_lookup["_record_key"] = core_lookup["_record_key"].astype(str).map(canonicalize_record_key)
        core_lookup = core_lookup.rename(
            columns={
                "一级总成供应商名称": "supplier_name",
                "一级总成供应商代码": "supplier_code",
                "物料名称": "material_name_fallback",
                "备件简称": "short_name_fallback",
                "适用车系": "vehicle_series_fallback",
            }
        )
        if "supplier_name" not in core_lookup.columns:
            core_lookup["supplier_name"] = core_lookup.get("供应商名称", "")
        if "supplier_code" not in core_lookup.columns:
            core_lookup["supplier_code"] = core_lookup.get("供应商代码", "")
        core_lookup = core_lookup[
            [
                "_record_key",
                "supplier_name",
                "supplier_code",
                "material_name_fallback",
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
                "material_name",
                "status",
                "effective_date",
            ]
        ],
        left_on="record_key",
        right_on="_record_key",
        how="left",
    )
    merged["_record_key"] = merged["_record_key"].fillna(merged["record_key"])
    merged = merged.merge(core_lookup, on="_record_key", how="left")

    placeholder_text_values = {"", "nan", "none", "null", "<na>", "nat"}

    def _text_series(column_name: str) -> pd.Series:
        if column_name in merged.columns:
            return merged[column_name]
        return pd.Series("", index=merged.index)

    def _clean_series(series: pd.Series) -> pd.Series:
        cleaned = series.astype("string").str.strip()
        return cleaned.mask(cleaned.str.lower().isin(placeholder_text_values), pd.NA)

    def _coalesce_series(primary_column: str, fallback_column: str = "", default: str = "") -> pd.Series:
        result = _clean_series(_text_series(primary_column))
        if fallback_column:
            result = result.combine_first(_clean_series(_text_series(fallback_column)))
        return result.fillna(default).astype(str)

    merged["remark"] = _coalesce_series("remark")
    merged["short_name"] = _coalesce_series("short_name", "short_name_fallback")
    merged["vehicle_series"] = _coalesce_series("vehicle_series", "vehicle_series_fallback")
    merged["supplier_name"] = _coalesce_series("supplier_name")
    merged["supplier_code"] = _coalesce_series("supplier_code")
    merged["material_code"] = _coalesce_series("material_code")
    merged["material_name"] = _coalesce_series("material_name", "material_name_fallback")
    merged["status"] = _coalesce_series("status")
    has_cost_context = (
        merged["material_name"].astype(str).str.strip().ne("")
        | merged["short_name"].astype(str).str.strip().ne("")
        | merged["vehicle_series"].astype(str).str.strip().ne("")
    )
    merged = merged.loc[has_cost_context].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["actual_price"] = pd.to_numeric(merged.get("actual_price"), errors="coerce")
    merged["predicted_price"] = pd.to_numeric(merged.get("predicted_price"), errors="coerce")
    merged["effective_date"] = pd.to_datetime(merged.get("effective_date"), errors="coerce")
    merged["labeled_at"] = pd.to_datetime(merged.get("labeled_at"), errors="coerce")

    merged["short_name"] = merged["short_name"].replace("", "未知备件")
    merged["vehicle_series"] = merged["vehicle_series"].replace("", "未识别车系")
    merged["group_key"] = merged.apply(
        lambda row: "||".join(
            [
                _normalize_text(row["vehicle_series"], "未识别车系"),
                _normalize_text(row["supplier_code"], "未识别供应商"),
                _normalize_text(row["short_name"], "未知备件"),
            ]
        ),
        axis=1,
    )
    return merged


def _build_group_payloads(records_df: pd.DataFrame) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    if records_df.empty:
        return payloads
    records_df = records_df.copy()
    for column_name in ["vehicle_series", "supplier_code", "supplier_name", "short_name", "material_code", "material_name", "status", "remark"]:
        if column_name not in records_df.columns:
            records_df[column_name] = ""
    if "group_key" not in records_df.columns:
        records_df["group_key"] = records_df.apply(
            lambda row: "||".join(
                [
                    _normalize_text(row.get("vehicle_series"), "未识别车系"),
                    _normalize_text(row.get("supplier_code"), "未识别供应商"),
                    _normalize_text(row.get("short_name"), "未知备件"),
                ]
            ),
            axis=1,
        )

    for _, group in records_df.groupby("group_key", sort=True):
        group = group.sort_values(["labeled_at", "record_key"], na_position="last")
        short_name = _normalize_text(group["short_name"].iloc[0], "未知备件")
        supplier_code = _join_unique(group["supplier_code"], default="")
        supplier_name = _join_unique(group["supplier_name"], default="")
        vehicle_series = _join_unique(group["vehicle_series"], default="未识别车系")
        rule_id = build_rule_id(short_name, supplier_code, vehicle_series)
        material_codes = [
            value for value in dict.fromkeys(group["material_code"].fillna("").astype(str).str.strip().tolist()) if value
        ]
        material_names = [
            value for value in dict.fromkeys(group["material_name"].fillna("").astype(str).str.strip().tolist()) if value
        ]
        representative_material_code = material_codes[0] if material_codes else ""
        representative_material_name = material_names[0] if material_names else ""
        direction_text = "偏高" if group["status"].astype(str).str.contains("偏高").sum() >= group["status"].astype(str).str.contains("偏低").sum() else "偏低"
        supplier_text = supplier_name or supplier_code or "未识别供应商"
        rule_template = f"{vehicle_series}车系{short_name}类备件对于{supplier_text}来说普遍异常"

        evidence_rows = []
        for row in group.to_dict(orient="records"):
            evidence_rows.append(
                {
                    "material_code": _normalize_text(row.get("material_code")),
                    "material_name": _normalize_text(row.get("material_name")),
                    "status": _normalize_text(row.get("status")),
                    "remark": _normalize_text(row.get("remark")),
                }
            )

        payloads.append(
            {
                "rule_id": rule_id,
                "short_name": short_name,
                "material_codes": material_codes,
                "material_names": material_names,
                "representative_material_code": representative_material_code,
                "representative_material_name": representative_material_name,
                "supplier_code": supplier_code,
                "supplier_name": supplier_name,
                "vehicle_series": vehicle_series,
                "direction": direction_text,
                "rule_template": rule_template,
                "records": evidence_rows,
            }
        )

    return payloads


def _build_distillation_prompt(group_payload: Dict[str, Any]) -> str:
    lines = [
        "请基于以下专家备注，蒸馏一条可复用的备件定价经验规则。",
        "以下输入已经过脱敏处理，不包含成本数值、价格字段、σ参数或其它算法字段。",
        "请优先围绕车系、供应商、备件简称三者的组合关系总结，例如“X车系Y类备件对于Z供应商来说普遍异常”。",
        f"车系: {group_payload.get('vehicle_series', '未识别车系')}",
        f"备件简称: {group_payload['short_name']}",
        f"供应商: {group_payload.get('supplier_name') or group_payload.get('supplier_code') or '未识别供应商'}",
        f"代表物料编码: {group_payload.get('representative_material_code', '')}",
        "",
        "专家备注清单:",
    ]
    for idx, record in enumerate(group_payload["records"], start=1):
        lines.append(
            f"{idx}. 物料{record.get('material_code', '') or '未知'}，结论{record.get('status', '') or '未知'}，"
            f"备注: {record.get('remark', '未提供') or '未提供'}"
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


def _fallback_distillation_for_group(group_payload: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    short_name = _normalize_text(group_payload.get("short_name"), "未知备件")
    vehicle_series = _normalize_text(group_payload.get("vehicle_series"), "未识别车系")
    supplier_text = (
        _normalize_text(group_payload.get("supplier_name"))
        or _normalize_text(group_payload.get("supplier_code"))
        or "未识别供应商"
    )
    representative_code = _normalize_text(group_payload.get("representative_material_code"))
    representative_name = _normalize_text(group_payload.get("representative_material_name"))
    direction = _normalize_text(group_payload.get("direction"), "异常")
    remarks = [
        _normalize_text(record.get("remark"))
        for record in group_payload.get("records", [])
        if _normalize_text(record.get("remark"))
    ]
    remark_summary = "；".join(dict.fromkeys(remarks)) or "专家已确认该样本具备合理业务原因"
    material_text = representative_code
    if representative_name:
        material_text = f"{material_text}（{representative_name}）" if material_text else representative_name
    if not material_text:
        material_text = "代表物料"
    rule_content = (
        f"{vehicle_series}车系{short_name}类备件对于{supplier_text}来说存在专家校准记录。"
        f"参考{material_text}的专家批注：{remark_summary}，可用于同车系、同供应商、同简称备件的{direction}复核。"
    )
    if reason:
        rule_content += " 本条规则由本地确定性逻辑生成，原因是 LLM 蒸馏暂不可用。"
    return {
        "rule_content": rule_content,
        "confidence_score": 0.55 if reason else 0.65,
        "key_signals": [
            f"车系={vehicle_series}",
            f"供应商={supplier_text}",
            f"专家备注={remark_summary[:80]}",
        ],
        "markdown_summary": (
            f"### {short_name}\n\n"
            f"- 规则: {rule_content}\n"
            f"- 代表物料: {material_text}\n"
            f"- 置信度: {(0.55 if reason else 0.65):.0%}\n"
        ),
    }


def distill_expert_knowledge(records_df: pd.DataFrame) -> Dict[str, Any]:
    grouped_payloads = _build_group_payloads(records_df)
    if not grouped_payloads:
        return {"rules": [], "markdown_report": "# 专家经验知识库\n\n暂无可蒸馏的专家备注。", "json_summary": []}

    rules: List[Dict[str, Any]] = []
    markdown_blocks: List[str] = ["# 专家经验知识库", "", f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for group_payload in grouped_payloads:
        sanitized_payload = sanitize_data_for_llm(group_payload) or {}
        prompt_payload = dict(sanitized_payload)
        prompt_payload["short_name"] = _normalize_text(prompt_payload.get("short_name"), "未知备件")
        prompt_payload["records"] = prompt_payload.get("records", [])
        prompt_payload = harness.sanitize_and_validate_llm_payload(
            prompt_payload,
            source="llm_engine.distill_expert_knowledge",
        )
        try:
            llm_result = _call_llm(_build_distillation_prompt(prompt_payload))
        except Exception as exc:
            llm_result = _fallback_distillation_for_group(group_payload, reason=str(exc))
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
            "material_code": group_payload["representative_material_code"],
            "material_name": group_payload["representative_material_name"],
            "supplier_code": group_payload["supplier_code"],
            "supplier_name": group_payload["supplier_name"],
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

    knowledge_df = harness.execute_action("load_expert_knowledge_base")
    if not is_llm_configured():
        result["status"] = "skipped"
        result["message"] = "LLM 未配置，已跳过知识蒸馏。"
        result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
        result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
        return result

    try:
        if force_full and knowledge_df is not None and not knowledge_df.empty:
            result["deleted_rule_count"] += int(harness.execute_action("clear_expert_knowledge_base") or 0)
            knowledge_df = harness.execute_action("load_expert_knowledge_base")

        source_df = _load_feedback_records_for_knowledge()
        if source_df.empty:
            result["status"] = "no_data"
            result["message"] = "暂无可供蒸馏的专家备注或缺少异常测算上下文。"
            result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
            result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
            return result

        last_updated = None if force_full else harness.execute_action("get_expert_knowledge_last_updated_at")
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
            supplier_code = _join_unique(group_rows["supplier_code"], default="")
            vehicle_series = _join_unique(group_rows["vehicle_series"], default="未识别车系")
            stale_rule_ids.append(build_rule_id(short_name, supplier_code, vehicle_series))

        if stale_rule_ids:
            result["deleted_rule_count"] = harness.execute_action(
                "delete_expert_knowledge_rules",
                rule_ids=stale_rule_ids,
            )

        if affected_records_df.empty:
            knowledge_df = harness.execute_action("load_expert_knowledge_base")
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
                    "material_code": rule.get("material_code", ""),
                    "material_name": rule.get("material_name", ""),
                    "supplier_code": rule["supplier_code"],
                    "supplier_name": rule.get("supplier_name", ""),
                    "vehicle_series": rule["vehicle_series"],
                    "rule_content": rule["rule_content"],
                    "confidence_score": rule["confidence_score"],
                    "updated_at": rule["updated_at"],
                }
            )
        result["updated_rule_count"] = harness.execute_action(
            "save_expert_knowledge_rules",
            rules=persisted_rules,
        )
        knowledge_df = harness.execute_action("load_expert_knowledge_base")
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

    knowledge_df = harness.execute_action("load_expert_knowledge_base")
    result["markdown_report"] = knowledge_base_to_markdown(knowledge_df)
    result["json_summary"] = knowledge_df.to_dict(orient="records") if not knowledge_df.empty else []
    return result


def knowledge_base_to_markdown(knowledge_df: Optional[pd.DataFrame] = None) -> str:
    if knowledge_df is None:
        knowledge_df = harness.execute_action("load_expert_knowledge_base")
    if knowledge_df is None or knowledge_df.empty:
        return "# 专家经验知识库\n\n暂无 AI 蒸馏出的专家经验规则。"

    lines = [
        "# 专家经验知识库",
        "",
        f"规则总数: {len(knowledge_df)}",
        "",
    ]
    for _, row in knowledge_df.iterrows():
        material_code = _normalize_text(row.get("material_code"), "未提供")
        material_name = _normalize_text(row.get("material_name"), "未提供")
        supplier_code = _normalize_text(row.get("supplier_code"), "未提供")
        supplier_name = _normalize_text(row.get("supplier_name"), "未提供")
        vehicle_series = _normalize_text(row.get("vehicle_series"), "未提供")
        confidence = _coerce_confidence(row.get("confidence_score"))
        updated_at = pd.to_datetime(row.get("updated_at"), errors="coerce")
        updated_text = updated_at.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(updated_at) else "未知"
        lines.extend(
            [
                f"## {row.get('short_name', '未知备件')}",
                "",
                f"- 代表物料编码: {material_code}",
                f"- 代表物料名称: {material_name}",
                f"- 供应商代码: {supplier_code}",
                f"- 供应商名称: {supplier_name}",
                f"- 车系: {vehicle_series}",
                f"- 可信度: {confidence:.0%}",
                f"- 更新时间: {updated_text}",
                f"- 车系/供应商/简称分析: {row.get('rule_content', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def knowledge_base_to_json_bytes(knowledge_df: Optional[pd.DataFrame] = None) -> bytes:
    if knowledge_df is None:
        knowledge_df = harness.execute_action("load_expert_knowledge_base")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "rules": knowledge_df.to_dict(orient="records") if knowledge_df is not None and not knowledge_df.empty else [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
