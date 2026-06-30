"""本地配置中心。

系统仅使用 settings.json + 本地 SQLite。
"""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from local_logging import log_event


PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
ENV_FILE = PROJECT_ROOT / ".env"
LOCAL_DB_FILENAME = "cost_monitor_data.db"
LOCAL_DB_PATH = PROJECT_ROOT / LOCAL_DB_FILENAME
LOCAL_DB_URL = f"sqlite:///{LOCAL_DB_PATH.as_posix()}"
LEGACY_PATH_KEY_ALIASES = {
    "assembly_detail_data_path": "assembly_data_path",
}
PATH_SETTING_LABELS = {
    "input_data_path": "原始数据存放路径",
    "quantitative_skills_path": "成本分析模型导出路径",
    "qualitative_skills_path": "专家经验报告导出路径",
    "assembly_data_path": "一级件明细数据路径",
    "sheet_metal_base_info_path": "钣金件基础数据路径",
    "sheet_metal_model_export_path": "钣金指数分析模型导出路径",
    "sheet_metal_report_export_path": "钣金专家经验报告导出路径",
}


def _strip_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mask_secret(value: Any) -> str:
    secret = _strip_string(value)
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def _default_llm_api_config() -> Dict[str, Any]:
    return {
        "api_key": "",
        "base_url": "",
        "model": "",
        "timeout_seconds": 45,
        "temperature": 0.2,
    }


def _default_paths_payload() -> Dict[str, str]:
    return {path_key: "" for path_key in PATH_SETTING_LABELS}


def _default_settings_payload() -> Dict[str, Any]:
    return {
        **_default_paths_payload(),
        "llm_api_config": _default_llm_api_config(),
    }


def _merge_settings_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = deepcopy(_default_settings_payload())
    if not isinstance(payload, dict):
        return merged

    legacy_input_data_path = _strip_string(payload.get("data_folder_path", ""))
    for path_key in PATH_SETTING_LABELS:
        if path_key in payload:
            merged[path_key] = _strip_string(payload.get(path_key, ""))
    for legacy_key, canonical_key in LEGACY_PATH_KEY_ALIASES.items():
        legacy_value = _strip_string(payload.get(legacy_key, ""))
        if legacy_value and not merged[canonical_key]:
            merged[canonical_key] = legacy_value
    if legacy_input_data_path and not merged["input_data_path"]:
        merged["input_data_path"] = legacy_input_data_path

    llm_payload = payload.get("llm_api_config")
    if isinstance(llm_payload, dict):
        merged["llm_api_config"].update(
            {
                "api_key": _strip_string(llm_payload.get("api_key"), merged["llm_api_config"]["api_key"]),
                "base_url": _strip_string(llm_payload.get("base_url"), merged["llm_api_config"]["base_url"]),
                "model": _strip_string(llm_payload.get("model"), merged["llm_api_config"]["model"]),
                "timeout_seconds": _coerce_int(
                    llm_payload.get("timeout_seconds"),
                    merged["llm_api_config"]["timeout_seconds"],
                ),
                "temperature": _coerce_float(
                    llm_payload.get("temperature"),
                    merged["llm_api_config"]["temperature"],
                ),
            }
        )

    return merged


def _read_settings_payload() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return _default_settings_payload()

    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = _default_settings_payload()
    return _merge_settings_payload(payload)


def _parse_env_payload(text: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            payload[key] = value
    return payload


def _read_env_payload() -> Dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    try:
        return _parse_env_payload(ENV_FILE.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _env_flag(value: Any, default: bool = False) -> bool:
    text = _strip_string(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _llm_config_from_env(env_payload: Dict[str, str], prefix: str, *, direct_default: bool = False) -> Dict[str, Any] | None:
    key_name = f"{prefix}KEY"
    url_name = f"{prefix}URL"
    model_name = f"{prefix}MODEL"
    api_key = _strip_string(env_payload.get(key_name, ""))
    base_url = _strip_string(env_payload.get(url_name, ""))
    model = _strip_string(env_payload.get(model_name, ""))
    if not (api_key and base_url and model):
        return None
    name = _strip_string(env_payload.get(f"{prefix}NAME", "")) or model
    return {
        "name": name,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "direct_url": _env_flag(env_payload.get(f"{prefix}DIRECT_URL"), direct_default),
        "append_no_think": _env_flag(env_payload.get(f"{prefix}NO_THINK"), False),
    }


def _build_llm_api_configs(env_payload: Dict[str, str]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    primary = _llm_config_from_env(env_payload, "LLM_API_", direct_default=False)
    if primary:
        configs.append(primary)

    indexed_suffixes = sorted(
        {
            match.group(1)
            for key in env_payload
            if (match := re.match(r"LLM_API_(\d+)_KEY$", str(key)))
        },
        key=lambda value: int(value),
    )
    for suffix in indexed_suffixes:
        config = _llm_config_from_env(env_payload, f"LLM_API_{suffix}_", direct_default=False)
        if config:
            configs.append(config)
    return configs


def _write_settings_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = _merge_settings_payload(payload)
    SETTINGS_FILE.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return merged


class Settings:
    def __init__(self) -> None:
        self.reset_cost_anomaly_results_on_start = False
        self.reload()

    def reload(self) -> None:
        payload = _read_settings_payload()
        if not SETTINGS_FILE.exists():
            payload = _write_settings_payload(payload)

        self._payload = payload
        self.input_data_path = _strip_string(payload.get("input_data_path", ""))
        self.quantitative_skills_path = _strip_string(payload.get("quantitative_skills_path", ""))
        self.qualitative_skills_path = _strip_string(payload.get("qualitative_skills_path", ""))
        self.assembly_data_path = _strip_string(payload.get("assembly_data_path", payload.get("assembly_detail_data_path", "")))
        self.assembly_detail_data_path = self.assembly_data_path
        self.sheet_metal_base_info_path = _strip_string(payload.get("sheet_metal_base_info_path", ""))
        self.sheet_metal_model_export_path = _strip_string(payload.get("sheet_metal_model_export_path", ""))
        self.sheet_metal_report_export_path = _strip_string(payload.get("sheet_metal_report_export_path", ""))
        self.data_folder_path = self.input_data_path

        llm_api_config = payload.get("llm_api_config", {})
        self.llm_timeout_seconds = _coerce_int(llm_api_config.get("timeout_seconds"), 45)
        self.llm_temperature = _coerce_float(llm_api_config.get("temperature"), 0.2)
        env_payload = _read_env_payload()
        self.llm_api_configs = _build_llm_api_configs(env_payload)
        primary_llm_config = self.llm_api_configs[0] if self.llm_api_configs else {}
        self.llm_api_key = _strip_string(primary_llm_config.get("api_key", ""))
        self.llm_api_base_url = _strip_string(primary_llm_config.get("base_url", ""))
        self.llm_api_model = _strip_string(primary_llm_config.get("model", ""))
        self.llm_api_direct_url = bool(primary_llm_config.get("direct_url", False))
        self.llm_api_append_no_think = bool(primary_llm_config.get("append_no_think", False))
        self.llm_env_file_path = ENV_FILE
        self.llm_config_source = "env" if env_payload else "missing_env"
        self.llm_env_configured = bool(self.llm_api_configs)

    @property
    def db_url(self) -> str:
        return LOCAL_DB_URL

    @property
    def db_path(self) -> Path:
        return LOCAL_DB_PATH

    @property
    def llm_model(self) -> str:
        return self.llm_api_model

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_env_configured)

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._payload)

    def save(
        self,
        *,
        path_updates: Dict[str, str] | None = None,
        data_folder_path: str | None = None,
        llm_api_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        previous_payload = self.to_dict()
        payload = deepcopy(previous_payload)
        normalized_path_updates: Dict[str, str] = {}
        if path_updates is not None:
            for path_key, path_value in path_updates.items():
                canonical_key = LEGACY_PATH_KEY_ALIASES.get(path_key, path_key)
                if canonical_key in PATH_SETTING_LABELS:
                    normalized_path_updates[canonical_key] = _strip_string(path_value)
        if data_folder_path is not None:
            normalized_path_updates["input_data_path"] = _strip_string(data_folder_path)
        if normalized_path_updates:
            payload.update(normalized_path_updates)
        if llm_api_config is not None:
            current_llm_config = dict(payload.get("llm_api_config", {}))
            current_llm_config.update(llm_api_config)
            payload["llm_api_config"] = current_llm_config

        payload = _write_settings_payload(payload)
        self.reload()
        current_payload = self.to_dict()

        if normalized_path_updates:
            for path_key, path_label in PATH_SETTING_LABELS.items():
                if path_key not in normalized_path_updates:
                    continue
                old_path = _strip_string(previous_payload.get(path_key, ""))
                new_path = _strip_string(current_payload.get(path_key, ""))
                if old_path != new_path:
                    log_event(
                        "settings",
                        "path_updated" if new_path else "path_cleared",
                        f"Updated {path_label}",
                        path_key=path_key,
                        path_label=path_label,
                        old_path=old_path,
                        new_path=new_path,
                    )

        if llm_api_config is not None:
            previous_llm = previous_payload.get("llm_api_config", {}) or {}
            current_llm = current_payload.get("llm_api_config", {}) or {}
            changed_fields = [
                field_name
                for field_name in ["api_key", "base_url", "model", "timeout_seconds", "temperature"]
                if previous_llm.get(field_name) != current_llm.get(field_name)
            ]
            if changed_fields:
                log_event(
                    "settings",
                    "llm_config_updated",
                    "Updated local LLM configuration",
                    changed_fields=changed_fields,
                    api_key_configured=bool(current_llm.get("api_key")),
                    api_key_preview=_mask_secret(current_llm.get("api_key")) if "api_key" in changed_fields else "",
                    base_url=current_llm.get("base_url", ""),
                    model=current_llm.get("model", ""),
                )

        return payload

    def save_data_folder_path(self, folder_path: str) -> Dict[str, Any]:
        return self.save(data_folder_path=folder_path)

    def save_paths(self, **paths: str) -> Dict[str, Any]:
        return self.save(path_updates=paths)

    def save_llm_api_config(self, **kwargs: Any) -> Dict[str, Any]:
        return self.save(llm_api_config=kwargs)


settings = Settings()
