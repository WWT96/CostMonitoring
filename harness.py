from __future__ import annotations

import ast
import contextvars
import hashlib
import inspect
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, MutableMapping, Sequence

try:
    import streamlit as st
except Exception:  # pragma: no cover - bare python execution
    st = None

from config import PATH_SETTING_LABELS, PROJECT_ROOT, settings


HARNESS_ROOT = PROJECT_ROOT / "harness"
BLUEPRINTS_DIR = HARNESS_ROOT / "blueprints"
STATE_DIR = HARNESS_ROOT / "state"
BLUEPRINT_STATE_FILE = STATE_DIR / "blueprint_updates.json"
PACKAGING_RELEASE_HARNESS_FILE = HARNESS_ROOT / "packaging_release_harness.json"

CONFIRMATION_PHRASE = "这个新功能已确定，请将其作为新蓝图"
HARNESS_AUDIT_STATE_KEY = "_harness_audit_result"
HARNESS_PATH_STATE_KEY = "_harness_path_status"
GOVERNANCE_STATUS_TEXT = "🛡️ Harness 治理引擎：核心逻辑保护已锁定 (V1.1)"

MANDATORY_SESSION_PATH_KEYS = (
    "input_data_path",
    "quantitative_skills_path",
    "qualitative_skills_path",
    "assembly_data_path",
    "sheet_metal_base_info_path",
    "sheet_metal_model_export_path",
)
OPTIONAL_SESSION_PATH_KEYS = (
    "sheet_metal_report_export_path",
)

SENSITIVE_LLM_FIELD_TOKENS = (
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
    "成本",
    "价格",
    "订购价",
    "零售价",
    "比例",
    "白痴指数",
    "权重",
    "σ",
)
SENSITIVE_LLM_INLINE_PATTERNS = (
    re.compile(r"(cost|price|ratio|sigma|weight|actual|predicted|baseline|amount|bound)\s*[:：=]\s*[-+]?\d", re.I),
    re.compile(r"(成本|价格|订购价|零售价|比例|白痴指数|权重|σ)\s*[:：=]\s*[-+]?\d"),
)

_ACTION_SCOPE_STACK: contextvars.ContextVar[tuple[dict[str, Any], ...]] = contextvars.ContextVar(
    "harness_action_scope_stack",
    default=(),
)
_DB_BOOTSTRAP_SOURCES = {
    "storage_service._build_db_engine",
}
_DB_ACTION_TYPES = {
    "compact_local_database",
    "clear_expert_knowledge_base",
    "delete_expert_knowledge_rules",
    "delete_feedback",
    "delete_sheet_metal_feedback",
    "delete_skills_snapshot",
    "get_core_cost_records_status",
    "get_expert_knowledge_last_updated_at",
    "get_expert_knowledge_refresh_token",
    "get_feedback_details",
    "get_feedback_records",
    "get_feedback_row_count",
    "get_feedback_statuses",
    "get_latest_core_cost_lookup",
    "get_local_database_health",
    "load_vehicle_market_prices",
    "load_vehicle_rank_config",
    "get_sheet_metal_feedback_details",
    "get_sheet_metal_feedback_statuses",
    "has_skills_snapshot",
    "initialize_storage",
    "load_core_cost_records",
    "load_cost_anomaly_results",
    "load_fresh_cost_anomaly_results",
    "load_expert_knowledge_base",
    "load_skills_snapshot",
    "replace_feedback",
    "replace_sheet_metal_feedback",
    "save_cost_anomaly_results",
    "record_cost_anomaly_result_run",
    "save_expert_knowledge_rules",
    "save_vehicle_market_prices",
    "save_vehicle_rank_config",
    "save_skills_snapshot",
    "sync_core_cost_records",
    "clear_feedback",
    "clear_sheet_metal_feedback",
}
_LLM_ACTION_TYPES = {
    "explain_vehicle_gradient_deviations",
    "fetch_vehicle_market_prices",
    "sync_expert_knowledge_base",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_harness_layout() -> None:
    BLUEPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not BLUEPRINT_STATE_FILE.exists():
        BLUEPRINT_STATE_FILE.write_text("{}\n", encoding="utf-8")


def _load_json_file(file_path: Path, *, default: Any) -> Any:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json_file(file_path: Path, payload: Any) -> None:
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_json_payload(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_blueprints() -> list[dict[str, Any]]:
    _ensure_harness_layout()
    blueprints: list[dict[str, Any]] = []
    for blueprint_path in sorted(BLUEPRINTS_DIR.glob("*.json")):
        payload = _load_json_file(blueprint_path, default={})
        if isinstance(payload, dict) and payload.get("module_name") and payload.get("relative_path"):
            blueprints.append(payload)
    return blueprints


def _load_blueprint_locks() -> dict[str, Any]:
    _ensure_harness_layout()
    payload = _load_json_file(BLUEPRINT_STATE_FILE, default={})
    return payload if isinstance(payload, dict) else {}


def load_packaging_release_harness() -> dict[str, Any]:
    _ensure_harness_layout()
    payload = _load_json_file(PACKAGING_RELEASE_HARNESS_FILE, default={})
    return payload if isinstance(payload, dict) else {}


def get_packaging_release_checklist() -> list[dict[str, Any]]:
    payload = load_packaging_release_harness()
    checklist = payload.get("verification_checklist", [])
    return [item for item in checklist if isinstance(item, dict)]


def _save_blueprint_locks(payload: dict[str, Any]) -> None:
    _ensure_harness_layout()
    _write_json_file(BLUEPRINT_STATE_FILE, payload)


def _resolve_source_path(relative_path: str) -> Path:
    return PROJECT_ROOT / str(relative_path).replace("\\", "/")


def _read_source(relative_path: str) -> str:
    return _resolve_source_path(relative_path).read_text(encoding="utf-8")


def _hash_source(relative_path: str) -> str:
    source_bytes = _resolve_source_path(relative_path).read_bytes()
    return hashlib.sha256(source_bytes).hexdigest()


def _scope_signature(scope_stack: Sequence[dict[str, Any]]) -> str:
    if not scope_stack:
        return ""
    return " > ".join(str(scope.get("action_type", "")) for scope in scope_stack if scope.get("action_type"))


@contextmanager
def _open_action_scope(
    action_type: str,
    *,
    source: str,
    allow_db: bool = False,
    allow_llm: bool = False,
) -> Iterator[None]:
    scope_stack = list(_ACTION_SCOPE_STACK.get())
    scope_stack.append(
        {
            "action_type": str(action_type or "").strip(),
            "source": str(source or "").strip(),
            "allow_db": bool(allow_db),
            "allow_llm": bool(allow_llm),
            "opened_at": _iso_now(),
        }
    )
    token = _ACTION_SCOPE_STACK.set(tuple(scope_stack))
    try:
        yield
    finally:
        _ACTION_SCOPE_STACK.reset(token)


def _is_db_access_authorized(source: str) -> bool:
    normalized_source = str(source or "").strip()
    if normalized_source in _DB_BOOTSTRAP_SOURCES:
        return True
    return any(bool(scope.get("allow_db")) for scope in _ACTION_SCOPE_STACK.get())


def _collect_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for child in target.elts:
            names.update(_collect_target_names(child))
        return names
    if isinstance(target, ast.Attribute):
        return {target.attr}
    return set()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent_name = _call_name(node.value)
        return f"{parent_name}.{node.attr}" if parent_name else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arg_names = [arg.arg for arg in node.args.posonlyargs]
    arg_names.extend(arg.arg for arg in node.args.args)
    if node.args.vararg:
        arg_names.append(f"*{node.args.vararg.arg}")
    arg_names.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        arg_names.append(f"**{node.args.kwarg.arg}")
    return arg_names


def _extract_scope_metadata(node: ast.AST) -> dict[str, list[str]]:
    calls: set[str] = set()
    string_literals: set[str] = set()
    assignments: set[str] = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            call_text = _call_name(child.func)
            if call_text:
                calls.add(call_text)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            literal_text = child.value.strip()
            if literal_text:
                string_literals.add(literal_text)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                assignments.update(_collect_target_names(target))
        elif isinstance(child, ast.AnnAssign):
            assignments.update(_collect_target_names(child.target))

    return {
        "calls": sorted(calls),
        "strings": sorted(string_literals),
        "assignments": sorted(assignments),
    }


def _build_module_structure(relative_path: str) -> tuple[dict[str, Any], str, str]:
    source_text = _read_source(relative_path)
    parsed = ast.parse(source_text, filename=relative_path)
    module_assignments: set[str] = set()
    functions: dict[str, Any] = {}
    classes: dict[str, Any] = {}

    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                module_assignments.update(_collect_target_names(target))
            continue
        if isinstance(node, ast.AnnAssign):
            module_assignments.update(_collect_target_names(node.target))
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope_meta = _extract_scope_metadata(node)
            functions[node.name] = {
                "args": _function_args(node),
                **scope_meta,
            }
            continue
        if isinstance(node, ast.ClassDef):
            methods: dict[str, Any] = {}
            for class_child in node.body:
                if isinstance(class_child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope_meta = _extract_scope_metadata(class_child)
                    methods[class_child.name] = {
                        "args": _function_args(class_child),
                        **scope_meta,
                    }
            classes[node.name] = {
                "bases": sorted(_call_name(base) for base in node.bases if _call_name(base)),
                "methods": methods,
            }

    structure = {
        "relative_path": relative_path,
        "module_assignments": sorted(module_assignments),
        "functions": functions,
        "classes": classes,
    }
    return structure, _hash_json_payload(structure), source_text


def _resolve_structure_scope(structure: dict[str, Any], scope_name: str) -> dict[str, Any] | None:
    normalized_scope = str(scope_name or "").strip()
    if not normalized_scope:
        return None

    functions = structure.get("functions", {})
    if normalized_scope in functions:
        return functions[normalized_scope]

    if "." in normalized_scope:
        class_name, method_name = normalized_scope.split(".", 1)
        class_payload = structure.get("classes", {}).get(class_name, {})
        return class_payload.get("methods", {}).get(method_name)
    return None


def _structure_finding(
    *,
    module_name: str,
    relative_path: str,
    label: str,
    message: str,
    details: Sequence[str],
    source_text: str,
    anchor_text: str = "",
    severity: str = "error",
) -> dict[str, Any]:
    line = _find_line_number(source_text, anchor_text=anchor_text or label, patterns=list(details))
    return {
        "module_name": module_name,
        "relative_path": relative_path,
        "severity": severity,
        "label": label,
        "message": message,
        "details": list(details),
        "line": line,
    }


def _evaluate_structure_contract(
    structure: dict[str, Any],
    source_text: str,
    blueprint: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = blueprint.get("structure") or {}
    if not isinstance(spec, dict):
        return []

    module_name = str(blueprint.get("module_name", "unknown"))
    relative_path = str(blueprint.get("relative_path", ""))
    findings: list[dict[str, Any]] = []

    required_assignments = [str(name) for name in spec.get("required_assignments", []) if str(name)]
    if required_assignments:
        missing_assignments = sorted(
            assignment_name
            for assignment_name in required_assignments
            if assignment_name not in set(structure.get("module_assignments", []))
        )
        if missing_assignments:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="结构赋值缺失",
                    message=f"{module_name} 缺少锁定的模块级赋值节点",
                    details=missing_assignments,
                    source_text=source_text,
                    severity="error",
                )
            )

    for function_spec in spec.get("required_functions", []):
        if not isinstance(function_spec, dict):
            continue
        function_name = str(function_spec.get("name", "")).strip()
        if not function_name:
            continue
        actual_function = structure.get("functions", {}).get(function_name)
        if actual_function is None:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="核心函数缺失",
                    message=f"{module_name} 缺少锁定函数: {function_name}",
                    details=[function_name],
                    source_text=source_text,
                    anchor_text=f"def {function_name}",
                    severity=str(function_spec.get("severity", "error")),
                )
            )
            continue

        expected_args = [str(arg_name) for arg_name in function_spec.get("args", []) if str(arg_name)]
        if expected_args and list(actual_function.get("args", [])) != expected_args:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="函数签名漂移",
                    message=f"{module_name}.{function_name} 的参数签名已变化",
                    details=[f"expected={expected_args}", f"actual={actual_function.get('args', [])}"],
                    source_text=source_text,
                    anchor_text=f"def {function_name}",
                    severity=str(function_spec.get("severity", "error")),
                )
            )

    for class_spec in spec.get("required_classes", []):
        if not isinstance(class_spec, dict):
            continue
        class_name = str(class_spec.get("name", "")).strip()
        if not class_name:
            continue
        actual_class = structure.get("classes", {}).get(class_name)
        if actual_class is None:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="核心类缺失",
                    message=f"{module_name} 缺少锁定类: {class_name}",
                    details=[class_name],
                    source_text=source_text,
                    anchor_text=f"class {class_name}",
                    severity=str(class_spec.get("severity", "error")),
                )
            )
            continue

        for method_spec in class_spec.get("methods", []):
            if not isinstance(method_spec, dict):
                continue
            method_name = str(method_spec.get("name", "")).strip()
            if not method_name:
                continue
            actual_method = actual_class.get("methods", {}).get(method_name)
            if actual_method is None:
                findings.append(
                    _structure_finding(
                        module_name=module_name,
                        relative_path=relative_path,
                        label="核心方法缺失",
                        message=f"{module_name}.{class_name} 缺少锁定方法: {method_name}",
                        details=[f"{class_name}.{method_name}"],
                        source_text=source_text,
                        anchor_text=f"def {method_name}",
                        severity=str(method_spec.get("severity", "error")),
                    )
                )
                continue

            expected_args = [str(arg_name) for arg_name in method_spec.get("args", []) if str(arg_name)]
            if expected_args and list(actual_method.get("args", [])) != expected_args:
                findings.append(
                    _structure_finding(
                        module_name=module_name,
                        relative_path=relative_path,
                        label="方法签名漂移",
                        message=f"{module_name}.{class_name}.{method_name} 的参数签名已变化",
                        details=[f"expected={expected_args}", f"actual={actual_method.get('args', [])}"],
                        source_text=source_text,
                        anchor_text=f"def {method_name}",
                        severity=str(method_spec.get("severity", "error")),
                    )
                )

    for call_spec in spec.get("required_function_calls", []):
        if not isinstance(call_spec, dict):
            continue
        scope_name = str(call_spec.get("scope", "")).strip()
        calls = [str(call_name) for call_name in call_spec.get("calls", []) if str(call_name)]
        if not scope_name or not calls:
            continue
        scope_payload = _resolve_structure_scope(structure, scope_name)
        if scope_payload is None:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="结构范围缺失",
                    message=f"{module_name} 缺少用于比对的结构范围: {scope_name}",
                    details=[scope_name],
                    source_text=source_text,
                    anchor_text=scope_name,
                    severity=str(call_spec.get("severity", "error")),
                )
            )
            continue

        actual_calls = set(scope_payload.get("calls", []))
        missing_calls = sorted(call_name for call_name in calls if call_name not in actual_calls)
        if missing_calls:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="关键调用缺失",
                    message=f"{module_name}.{scope_name} 缺少锁定的关键调用",
                    details=missing_calls,
                    source_text=source_text,
                    anchor_text=scope_name,
                    severity=str(call_spec.get("severity", "error")),
                )
            )

    for string_spec in spec.get("required_string_literals", []):
        if not isinstance(string_spec, dict):
            continue
        scope_name = str(string_spec.get("scope", "")).strip()
        values = [str(value) for value in string_spec.get("values", []) if str(value)]
        if not scope_name or not values:
            continue
        scope_payload = _resolve_structure_scope(structure, scope_name)
        if scope_payload is None:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="结构范围缺失",
                    message=f"{module_name} 缺少用于比对的字符串范围: {scope_name}",
                    details=[scope_name],
                    source_text=source_text,
                    anchor_text=scope_name,
                    severity=str(string_spec.get("severity", "error")),
                )
            )
            continue

        actual_values = set(scope_payload.get("strings", []))
        missing_values = sorted(value for value in values if value not in actual_values)
        if missing_values:
            findings.append(
                _structure_finding(
                    module_name=module_name,
                    relative_path=relative_path,
                    label="视觉蓝图字面量缺失",
                    message=f"{module_name}.{scope_name} 缺少锁定的结构字面量",
                    details=missing_values,
                    source_text=source_text,
                    anchor_text=scope_name,
                    severity=str(string_spec.get("severity", "error")),
                )
            )

    return findings


def _line_number_from_index(source_text: str, index: int) -> int | None:
    if index < 0:
        return None
    return source_text.count("\n", 0, index) + 1


def _find_line_number(source_text: str, *, anchor_text: str = "", patterns: Sequence[str] = (), regex_mode: bool = False) -> int | None:
    if anchor_text:
        anchor_index = source_text.find(anchor_text)
        if anchor_index >= 0:
            return _line_number_from_index(source_text, anchor_index)
    for pattern in patterns:
        if regex_mode:
            match = re.search(pattern, source_text, flags=re.M)
            if match:
                return _line_number_from_index(source_text, match.start())
        else:
            matched_index = source_text.find(pattern)
            if matched_index >= 0:
                return _line_number_from_index(source_text, matched_index)
    return None


def _evaluate_rule(source_text: str, rule: dict[str, Any]) -> dict[str, Any] | None:
    mode = str(rule.get("mode", "contains_all")).strip()
    patterns = [str(pattern) for pattern in rule.get("patterns", []) if str(pattern)]
    anchor_text = str(rule.get("anchor_text", "")).strip()
    label = str(rule.get("label", rule.get("id", "未命名规则"))).strip()
    severity = str(rule.get("severity", "error")).strip() or "error"

    missing: list[str] = []
    violated: list[str] = []
    regex_mode = mode.startswith("regex")

    if mode in {"contains_all", "regex_all"}:
        for pattern in patterns:
            if regex_mode:
                if not re.search(pattern, source_text, flags=re.M):
                    missing.append(pattern)
            elif pattern not in source_text:
                missing.append(pattern)
        if missing:
            return {
                "severity": severity,
                "label": label,
                "message": f"{label} 缺少锁定签名",
                "details": missing,
                "line": _find_line_number(source_text, anchor_text=anchor_text, patterns=patterns, regex_mode=regex_mode),
            }

    if mode in {"contains_any", "regex_any"}:
        matched = False
        for pattern in patterns:
            if regex_mode and re.search(pattern, source_text, flags=re.M):
                matched = True
                break
            if not regex_mode and pattern in source_text:
                matched = True
                break
        if not matched:
            return {
                "severity": severity,
                "label": label,
                "message": f"{label} 未命中任何有效签名",
                "details": patterns,
                "line": _find_line_number(source_text, anchor_text=anchor_text, patterns=patterns, regex_mode=regex_mode),
            }

    if mode == "forbid_any":
        for pattern in patterns:
            if pattern in source_text:
                violated.append(pattern)
        if violated:
            return {
                "severity": severity,
                "label": label,
                "message": f"{label} 命中了禁用签名",
                "details": violated,
                "line": _find_line_number(source_text, anchor_text=anchor_text, patterns=violated),
            }

    return None


def _get_function_source_text(source_text: str, function_name: str) -> str:
    try:
        module_tree = ast.parse(source_text)
    except SyntaxError:
        return ""

    source_lines = source_text.splitlines()
    for node in module_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            start_line = max(node.lineno - 1, 0)
            end_line = getattr(node, "end_lineno", node.lineno)
            return "\n".join(source_lines[start_line:end_line])
    return ""


def _assembly_ui_finding(
    source_text: str,
    *,
    label: str,
    message: str,
    details: Sequence[str],
    anchor_text: str = "",
    patterns: Sequence[str] = (),
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "module_name": "assembly_ui",
        "relative_path": "assembly_ui.py",
        "severity": severity,
        "label": label,
        "message": message,
        "details": list(details),
        "line": _find_line_number(source_text, anchor_text=anchor_text, patterns=patterns),
    }


def _audit_assembly_ui_rowspan_contract(source_text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    render_source = _get_function_source_text(source_text, "render_assembly_audit_page")
    html_builder_source = _get_function_source_text(source_text, "_build_rowspan_tree_table_html")
    helper_sources = {
        helper_name: _get_function_source_text(source_text, helper_name)
        for helper_name in ("_render_tree_header", "_render_tree_row", "_render_assembly_toggle_buttons", "_toggle_assembly_parent")
    }

    if not html_builder_source:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="拆分件监控缺少 HTML 树表构造函数",
                message="assembly_ui 缺少 _build_rowspan_tree_table_html，无法生成 HTML rowspan 树表。",
                details=["_build_rowspan_tree_table_html"],
                anchor_text="def _build_rowspan_tree_table_html",
            )
        )
        return findings

    banned_builder_calls = [pattern for pattern in ("st.columns(", "st.container(") if pattern in html_builder_source]
    if banned_builder_calls:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="HTML 树表核心函数禁止使用 Streamlit 多列排版",
                message="_build_rowspan_tree_table_html 不得再通过 st.columns 或 st.container 参与树表主体排版。",
                details=banned_builder_calls,
                anchor_text="def _build_rowspan_tree_table_html",
                patterns=banned_builder_calls,
            )
        )

    for helper_name, helper_source in helper_sources.items():
        helper_banned_calls = [pattern for pattern in ("st.columns(", "st.container(") if pattern in helper_source]
        if helper_banned_calls:
            findings.append(
                _assembly_ui_finding(
                    source_text,
                    label="HTML 树表辅助函数禁止使用 Streamlit 多列排版",
                    message=f"{helper_name} 不得再通过 st.columns 或 st.container 参与树表主体排版。",
                    details=helper_banned_calls,
                    anchor_text=f"def {helper_name}",
                    patterns=helper_banned_calls,
                )
            )

        banned_toggle_navigation_signatures = [
            pattern
            for pattern in (
                "ASSEMBLY_TOGGLE_QUERY_KEY",
                "ASSEMBLY_QUERY_PAGE_KEY",
                "_build_toggle_link(",
                "_get_query_param_text(",
                "_clear_assembly_query_params(",
                "_apply_pending_assembly_toggle(",
                "st.query_params",
                "href='?",
                'href="?',
                "<a class='assembly-toggle-link'",
            )
            if pattern in source_text
        ]
        if banned_toggle_navigation_signatures:
            findings.append(
                _assembly_ui_finding(
                    source_text,
                    label="拆分件展开交互不得依赖链接或查询参数",
                    message="assembly_ui 的展开/收起必须由原生按钮与 session state 驱动，不得保留 href 或 query param 跳转链路。",
                    details=banned_toggle_navigation_signatures,
                    patterns=banned_toggle_navigation_signatures,
                )
            )

    if "rowspan" not in html_builder_source or "_build_html_cell(" not in html_builder_source:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="HTML 树表必须输出 rowspan 单元格",
                message="_build_rowspan_tree_table_html 未生成带 rowspan 的 HTML 单元格输出。",
                details=["rowspan", "_build_html_cell("],
                anchor_text="def _build_rowspan_tree_table_html",
                patterns=["rowspan", "_build_html_cell("],
            )
        )

    hidden_signatures = [
        pattern
        for pattern in ("assembly-audit-table-hidden", "aria-hidden='true'", 'aria-hidden="true"')
        if pattern in source_text
    ]
    if hidden_signatures:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="HTML 树表不得隐藏",
                message="assembly_ui 不得把合规的 HTML rowspan 树表标记为隐藏节点。",
                details=hidden_signatures,
                patterns=hidden_signatures,
            )
        )

    toggle_helper_source = helper_sources.get("_render_assembly_toggle_buttons", "")
    if not toggle_helper_source or "st.button(" not in toggle_helper_source:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="拆分件展开交互必须使用原生按钮",
                message="assembly_ui 缺少基于 st.button 的一级件展开/收起触发器。",
                details=["_render_assembly_toggle_buttons", "st.button("],
                anchor_text="def _render_assembly_toggle_buttons",
                patterns=["st.button("],
            )
        )

    toggle_callback_source = helper_sources.get("_toggle_assembly_parent", "")
    if not toggle_callback_source or "_set_expanded_parent_codes(" not in toggle_callback_source:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="拆分件按钮回调必须直接更新展开状态",
                message="assembly_ui 缺少只负责维护 expanded_parts 的原生按钮回调。",
                details=["_toggle_assembly_parent", "_set_expanded_parent_codes("],
                anchor_text="def _toggle_assembly_parent",
                patterns=["_set_expanded_parent_codes("],
            )
        )

    legacy_grid_signatures = [pattern for pattern in ("def _build_tree_rows(",) if pattern in source_text]
    if legacy_grid_signatures:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="拆分件树表不得保留旧 grid 渲染器",
                message="assembly_ui 仍保留旧的树表 grid 渲染函数，存在回退到 st.columns 的风险。",
                details=legacy_grid_signatures,
                patterns=legacy_grid_signatures,
            )
        )

    if not render_source:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="拆分件监控缺少主渲染函数",
                message="assembly_ui 缺少 render_assembly_audit_page，无法验证最终渲染路径。",
                details=["render_assembly_audit_page"],
                anchor_text="def render_assembly_audit_page",
            )
        )
        return findings

    if "st.write(rowspan_table_html, unsafe_allow_html=True)" not in render_source:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="HTML 树表必须显式写入页面",
                message="render_assembly_audit_page 必须通过 st.write(..., unsafe_allow_html=True) 输出 rowspan 树表。",
                details=["st.write(rowspan_table_html, unsafe_allow_html=True)"],
                anchor_text="def render_assembly_audit_page",
                patterns=["st.write(rowspan_table_html, unsafe_allow_html=True)"],
            )
        )

    legacy_render_calls = [
        pattern
        for pattern in ("_build_tree_rows(", "_render_tree_header(", "_render_tree_row(")
        if pattern in render_source
    ]
    if legacy_render_calls:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="最终树表渲染路径仍指向旧 grid 逻辑",
                message="render_assembly_audit_page 仍在调用旧的 st.columns 树表渲染路径。",
                details=legacy_render_calls,
                anchor_text="def render_assembly_audit_page",
                patterns=legacy_render_calls,
            )
        )

    required_render_signatures = [
        "_get_expanded_parent_codes()",
        "_build_tree_render_groups(",
        "_build_rowspan_tree_table_html(",
        "_render_assembly_toggle_buttons(",
        "_render_pagination_controls(",
    ]
    missing_render_signatures = [pattern for pattern in required_render_signatures if pattern not in render_source]
    if missing_render_signatures:
        findings.append(
            _assembly_ui_finding(
                source_text,
                label="HTML 树表状态与分页链路不完整",
                message="render_assembly_audit_page 缺少展开状态或分页所需的关键调用。",
                details=missing_render_signatures,
                anchor_text="def render_assembly_audit_page",
                patterns=missing_render_signatures,
            )
        )

    return findings


def audit_system_integrity() -> dict[str, Any]:
    _ensure_harness_layout()
    locks = _load_blueprint_locks()
    findings: list[dict[str, Any]] = []
    modules: list[dict[str, Any]] = []

    for blueprint in _load_blueprints():
        module_name = str(blueprint["module_name"])
        relative_path = str(blueprint["relative_path"])
        source_path = _resolve_source_path(relative_path)

        module_findings: list[dict[str, Any]] = []
        if not source_path.exists():
            module_findings.append(
                {
                    "module_name": module_name,
                    "relative_path": relative_path,
                    "severity": "error",
                    "label": "模块文件缺失",
                    "message": f"蓝图模块文件不存在: {relative_path}",
                    "details": [relative_path],
                    "line": None,
                }
            )
            modules.append(
                {
                    "module_name": module_name,
                    "relative_path": relative_path,
                    "locked": False,
                    "ok": False,
                    "source_sha256": "",
                    "locked_sha256": "",
                    "finding_count": 1,
                }
            )
            findings.extend(module_findings)
            continue

        try:
            structure_payload, current_structure_hash, source_text = _build_module_structure(relative_path)
        except SyntaxError as exc:
            module_findings.append(
                {
                    "module_name": module_name,
                    "relative_path": relative_path,
                    "severity": "error",
                    "label": "结构解析失败",
                    "message": f"{module_name} 无法完成 AST 结构解析: {exc.msg}",
                    "details": [f"line={exc.lineno}", f"offset={exc.offset}"],
                    "line": exc.lineno,
                }
            )
            modules.append(
                {
                    "module_name": module_name,
                    "relative_path": relative_path,
                    "locked": False,
                    "ok": False,
                    "source_sha256": "",
                    "locked_sha256": "",
                    "structure_sha256": "",
                    "locked_structure_sha256": "",
                    "finding_count": 1,
                }
            )
            findings.extend(module_findings)
            continue

        current_hash = _hash_source(relative_path)
        lock_state = locks.get(module_name, {})
        locked_hash = ""
        locked_structure_hash = ""
        if isinstance(lock_state, dict):
            locked_hash = str(lock_state.get("sha256", "")).strip()
            locked_structure_hash = str(lock_state.get("structure_sha256", "")).strip()
        elif isinstance(lock_state, str):
            locked_hash = str(lock_state).strip()

        if locked_structure_hash and locked_structure_hash != current_structure_hash:
            module_findings.append(
                {
                    "module_name": module_name,
                    "relative_path": relative_path,
                    "severity": "warning",
                    "label": "蓝图结构签名不一致",
                    "message": f"{module_name} 结构签名已变化，尚未确认更新蓝图",
                    "details": [f"expected={locked_structure_hash[:12]}", f"current={current_structure_hash[:12]}"],
                    "line": _find_line_number(source_text, anchor_text=str(blueprint.get("anchor_text", "")).strip()),
                }
            )

        module_findings.extend(_evaluate_structure_contract(structure_payload, source_text, blueprint))

        for rule in blueprint.get("rules", []):
            finding = _evaluate_rule(source_text, rule)
            if finding:
                finding.update(
                    {
                        "module_name": module_name,
                        "relative_path": relative_path,
                    }
                )
                module_findings.append(finding)

        if module_name == "assembly_ui":
            module_findings.extend(_audit_assembly_ui_rowspan_contract(source_text))

        modules.append(
            {
                "module_name": module_name,
                "relative_path": relative_path,
                "locked": bool(locked_hash),
                "ok": not module_findings,
                "source_sha256": current_hash,
                "locked_sha256": locked_hash,
                "structure_sha256": current_structure_hash,
                "locked_structure_sha256": locked_structure_hash,
                "finding_count": len(module_findings),
            }
        )
        findings.extend(module_findings)

    findings.sort(key=lambda item: (item.get("severity") != "error", item.get("relative_path", ""), item.get("line") or 0))
    has_error = any(str(item.get("severity", "")).lower() == "error" for item in findings)
    return {
        "status": "success" if not findings else ("error" if has_error else "warning"),
        "checked_at": _iso_now(),
        "module_count": len(modules),
        "finding_count": len(findings),
        "findings": findings,
        "modules": modules,
    }


def _require_streamlit_session_state() -> MutableMapping[str, Any]:
    if st is None:
        raise RuntimeError("当前运行环境未启用 Streamlit session_state")
    return st.session_state


def ensure_session_paths(session_state: MutableMapping[str, Any] | None = None) -> dict[str, Any]:
    settings.reload()
    state = session_state if session_state is not None else _require_streamlit_session_state()
    changed_keys: list[str] = []
    nonempty_mandatory_keys: list[str] = []

    for path_key in [*MANDATORY_SESSION_PATH_KEYS, *OPTIONAL_SESSION_PATH_KEYS]:
        expected_value = str(getattr(settings, path_key, "") or "").strip()
        current_value = str(state.get(path_key, "") or "").strip() if path_key in state else None
        if current_value != expected_value:
            state[path_key] = expected_value
            changed_keys.append(path_key)
        if path_key in MANDATORY_SESSION_PATH_KEYS and expected_value:
            nonempty_mandatory_keys.append(path_key)

    status = {
        "checked_at": _iso_now(),
        "mandatory_path_keys": list(MANDATORY_SESSION_PATH_KEYS),
        "optional_path_keys": list(OPTIONAL_SESSION_PATH_KEYS),
        "backfilled_keys": changed_keys,
        "nonempty_mandatory_keys": nonempty_mandatory_keys,
        "path_labels": {key: PATH_SETTING_LABELS.get(key, key) for key in [*MANDATORY_SESSION_PATH_KEYS, *OPTIONAL_SESSION_PATH_KEYS]},
    }
    state[HARNESS_PATH_STATE_KEY] = status
    return status


def bootstrap_runtime_governance(session_state: MutableMapping[str, Any] | None = None) -> dict[str, Any]:
    state = session_state if session_state is not None else _require_streamlit_session_state()
    path_status = ensure_session_paths(state)
    audit_result = audit_system_integrity()
    state[HARNESS_AUDIT_STATE_KEY] = audit_result
    return {
        "path_status": path_status,
        "audit_result": audit_result,
    }


def render_sidebar_integrity_warning(audit_result: dict[str, Any] | None = None) -> None:
    if st is None:
        return
    payload = audit_result or st.session_state.get(HARNESS_AUDIT_STATE_KEY)
    if not isinstance(payload, dict):
        return
    findings = payload.get("findings") or []
    if not findings:
        return

    st.sidebar.error("❗ 警告：检测到核心架构退行")
    for finding in findings[:6]:
        relative_path = str(finding.get("relative_path", ""))
        line = finding.get("line")
        location_text = f"{relative_path}:{line}" if line else relative_path
        st.sidebar.markdown(
            f"- **{finding.get('module_name', 'unknown')}**: {finding.get('message', '')}  \n"
            f"  位置: {location_text}"
        )
    overflow_count = max(len(findings) - 6, 0)
    if overflow_count:
        st.sidebar.caption(f"另有 {overflow_count} 处待处理冲突，详见 Harness 审计结果。")


def render_sidebar_governance_status(audit_result: dict[str, Any] | None = None) -> None:
    if st is None:
        return
    payload = audit_result or st.session_state.get(HARNESS_AUDIT_STATE_KEY) or {}
    status = str(payload.get("status", "ok")).strip().lower()
    finding_count = int(payload.get("finding_count", 0) or 0)

    st.sidebar.markdown("---")
    st.sidebar.caption(GOVERNANCE_STATUS_TEXT)
    if finding_count <= 0:
        st.sidebar.caption("Harness 审计通过，未发现待确认变更。")
    elif status == "error":
        st.sidebar.caption(f"当前存在 {finding_count} 处强制门禁告警，蓝图更新前不会被静默忽略。")
    else:
        st.sidebar.caption(f"当前存在 {finding_count} 处待确认结构变更，请使用蓝图确认流程处理。")


def _infer_source(default: str) -> str:
    current_file = Path(__file__).name
    for frame in inspect.stack()[2:]:
        filename = Path(frame.filename).name
        if filename == current_file:
            continue
        return f"{filename}:{frame.function}"
    return default


def authorize_db_operation(operation: str, source: str | None = None) -> dict[str, str]:
    normalized_source = str(source or _infer_source("harness.authorize_db_operation")).strip()
    if not _is_db_access_authorized(normalized_source):
        scope_signature = _scope_signature(_ACTION_SCOPE_STACK.get())
        scope_hint = f"；当前作用域: {scope_signature}" if scope_signature else ""
        raise PermissionError(f"禁止绕过 Harness 访问数据库: {normalized_source}{scope_hint}")
    return {
        "operation": str(operation or "read").strip() or "read",
        "source": normalized_source,
        "authorized_at": _iso_now(),
    }


def run_db_action(operation: str, source: str, action: Callable[[], Any]) -> Any:
    with _open_action_scope(
        f"db:{operation}",
        source=source,
        allow_db=True,
    ):
        authorize_db_operation(operation, source)
        return action()


@contextmanager
def managed_sqlite_connection(
    db_path: str | Path,
    *,
    operation: str,
    source: str,
    **connect_kwargs: Any,
) -> Iterator[sqlite3.Connection]:
    authorize_db_operation(operation, source)
    with sqlite3.connect(str(db_path), **connect_kwargs) as conn:
        yield conn


def _value_contains_number(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        return bool(re.search(r"[-+]?\d", value))
    if isinstance(value, dict):
        return any(_value_contains_number(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_value_contains_number(item) for item in value)
    return False


def _assert_safe_llm_payload(payload: Any, *, source: str, parent_key: str = "") -> None:
    if isinstance(payload, dict):
        for raw_key, raw_value in payload.items():
            key_text = str(raw_key or "").strip()
            lowered_key = key_text.lower()
            if any(token in lowered_key for token in SENSITIVE_LLM_FIELD_TOKENS) and _value_contains_number(raw_value):
                raise ValueError(f"{source} 试图向 LLM 发送敏感数值字段: {key_text}")
            _assert_safe_llm_payload(raw_value, source=source, parent_key=key_text or parent_key)
        return

    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            _assert_safe_llm_payload(item, source=source, parent_key=parent_key)
        return

    if isinstance(payload, str):
        for pattern in SENSITIVE_LLM_INLINE_PATTERNS:
            if pattern.search(payload):
                context_key = parent_key or "prompt"
                raise ValueError(f"{source} 试图向 LLM 发送敏感数值文本: {context_key}")


def sanitize_and_validate_llm_payload(payload: Any, *, source: str) -> Any:
    _assert_safe_llm_payload(payload, source=source)
    return payload


def run_llm_action(source: str, action: Callable[[], Any], *, request_payload: Any | None = None) -> Any:
    with _open_action_scope("llm", source=source, allow_llm=True):
        if request_payload is not None:
            sanitize_and_validate_llm_payload(request_payload, source=source)
        return action()


def execute_action(action_type: str, **kwargs: Any) -> Any:
    normalized_action = str(action_type or "").strip()
    if not normalized_action:
        raise ValueError("action_type 不能为空")

    with _open_action_scope(
        f"execute:{normalized_action}",
        source="harness.execute_action",
        allow_db=normalized_action in _DB_ACTION_TYPES,
        allow_llm=normalized_action in _LLM_ACTION_TYPES,
    ):
        if normalized_action == "initialize_storage":
            from storage_service import initialize_local_storage

            return initialize_local_storage()

        if normalized_action == "load_data_from_folder":
            from data_ingestion import load_data_from_folder

            return load_data_from_folder(str(kwargs.get("folder_path") or ""))

        if normalized_action == "load_data_from_uploaded_files":
            from data_ingestion import load_data_from_uploaded_files

            return load_data_from_uploaded_files(list(kwargs.get("uploaded_files") or []))

        if normalized_action == "sync_core_cost_records":
            from data_ingestion import persist_core_cost_records

            return persist_core_cost_records(
                kwargs.get("df"),
                price_col=kwargs.get("price_col"),
                mode=str(kwargs.get("mode") or "incremental"),
            )

        if normalized_action == "load_core_cost_records":
            from data_ingestion import load_core_cost_records

            return load_core_cost_records()

        if normalized_action == "get_core_cost_records_status":
            from data_ingestion import get_core_cost_records_status

            return get_core_cost_records_status()

        if normalized_action == "get_local_database_health":
            from storage_service import get_local_database_health

            return get_local_database_health()

        if normalized_action == "get_latest_core_cost_lookup":
            from storage_service import get_latest_core_cost_lookup

            return get_latest_core_cost_lookup(kwargs.get("material_codes") or [])

        if normalized_action == "load_vehicle_rank_config":
            from storage_service import load_vehicle_rank_config

            return load_vehicle_rank_config()

        if normalized_action == "save_vehicle_rank_config":
            from storage_service import save_vehicle_rank_config

            return save_vehicle_rank_config(kwargs.get("rank_rows") or [])

        if normalized_action == "load_vehicle_market_prices":
            from storage_service import load_vehicle_market_prices

            return load_vehicle_market_prices()

        if normalized_action == "save_vehicle_market_prices":
            from storage_service import save_vehicle_market_prices

            return save_vehicle_market_prices(kwargs.get("price_rows") or [])

        if normalized_action == "get_feedback_details":
            from storage_service import label_manager

            return label_manager.get_labels()

        if normalized_action == "get_feedback_statuses":
            from storage_service import label_manager

            return label_manager.get_label_statuses()

        if normalized_action == "get_feedback_records":
            from storage_service import label_manager

            return label_manager.get_label_records()

        if normalized_action == "get_feedback_row_count":
            from storage_service import label_manager

            return label_manager.file_row_count()

        if normalized_action == "replace_feedback":
            from storage_service import label_manager

            return label_manager.replace_all(kwargs.get("final_labels_df"))

        if normalized_action == "delete_feedback":
            from storage_service import label_manager

            return label_manager.delete_labels(kwargs.get("keys_to_remove") or [])

        if normalized_action == "clear_feedback":
            from storage_service import label_manager

            return label_manager.clear_all()

        if normalized_action == "get_sheet_metal_feedback_details":
            from storage_service import sheet_metal_label_manager

            return sheet_metal_label_manager.get_labels()

        if normalized_action == "get_sheet_metal_feedback_statuses":
            from storage_service import sheet_metal_label_manager

            return sheet_metal_label_manager.get_label_statuses()

        if normalized_action == "replace_sheet_metal_feedback":
            from storage_service import sheet_metal_label_manager

            return sheet_metal_label_manager.replace_all(kwargs.get("final_labels_df"))

        if normalized_action == "delete_sheet_metal_feedback":
            from storage_service import sheet_metal_label_manager

            return sheet_metal_label_manager.delete_labels(kwargs.get("keys_to_remove") or [])

        if normalized_action == "clear_sheet_metal_feedback":
            from storage_service import sheet_metal_label_manager

            return sheet_metal_label_manager.clear_all()

        if normalized_action == "load_skills_snapshot":
            from storage_service import load_skills

            return load_skills(domain=str(kwargs.get("domain") or "cost"))

        if normalized_action == "save_skills_snapshot":
            from storage_service import save_skills

            return save_skills(
                kwargs.get("skills") or [],
                sigma=float(kwargs.get("sigma") or 1.0),
                weight=int(kwargs.get("weight") or 80),
                domain=str(kwargs.get("domain") or "cost"),
            )

        if normalized_action == "has_skills_snapshot":
            from storage_service import has_skills_snapshot

            return has_skills_snapshot(domain=str(kwargs.get("domain") or "cost"))

        if normalized_action == "delete_skills_snapshot":
            from storage_service import delete_skills

            return delete_skills(domain=str(kwargs.get("domain") or "cost"))

        if normalized_action == "compact_local_database":
            from storage_service import vacuum_local_database

            return vacuum_local_database()

        if normalized_action == "load_expert_knowledge_base":
            from storage_service import load_expert_knowledge_base

            return load_expert_knowledge_base()

        if normalized_action == "get_expert_knowledge_last_updated_at":
            from storage_service import get_expert_knowledge_last_updated_at

            return get_expert_knowledge_last_updated_at()

        if normalized_action == "get_expert_knowledge_refresh_token":
            from storage_service import get_expert_knowledge_refresh_token

            return get_expert_knowledge_refresh_token()

        if normalized_action == "save_expert_knowledge_rules":
            from storage_service import save_expert_knowledge_rules

            return save_expert_knowledge_rules(kwargs.get("rules") or [])

        if normalized_action == "delete_expert_knowledge_rules":
            from storage_service import delete_expert_knowledge_rules

            return delete_expert_knowledge_rules(kwargs.get("rule_ids") or [])

        if normalized_action == "clear_expert_knowledge_base":
            from storage_service import clear_expert_knowledge_base

            return clear_expert_knowledge_base()

        if normalized_action == "load_cost_anomaly_results":
            from storage_service import load_cost_anomaly_results

            return load_cost_anomaly_results(result_mode=str(kwargs.get("result_mode") or "raw"))

        if normalized_action == "load_fresh_cost_anomaly_results":
            from storage_service import load_fresh_cost_anomaly_results

            return load_fresh_cost_anomaly_results(
                result_mode=str(kwargs.get("result_mode") or "raw"),
                source_signature=str(kwargs.get("source_signature") or ""),
                options_signature=str(kwargs.get("options_signature") or ""),
            )

        if normalized_action == "save_cost_anomaly_results":
            from storage_service import save_cost_anomaly_results

            return save_cost_anomaly_results(
                kwargs.get("result_df"),
                result_mode=str(kwargs.get("result_mode") or "raw"),
            )

        if normalized_action == "record_cost_anomaly_result_run":
            from storage_service import record_cost_anomaly_result_run

            return record_cost_anomaly_result_run(
                str(kwargs.get("result_mode") or "raw"),
                source_signature=str(kwargs.get("source_signature") or ""),
                options_signature=str(kwargs.get("options_signature") or ""),
                row_count=int(kwargs.get("row_count") or 0),
            )

        if normalized_action == "save_path_settings":
            path_updates = dict(kwargs.get("path_updates") or {})
            settings.save_paths(**path_updates)
            settings.reload()
            return settings.to_dict()

        if normalized_action == "save_llm_api_config":
            settings.save_llm_api_config(
                api_key=str(kwargs.get("api_key") or ""),
                base_url=str(kwargs.get("base_url") or ""),
                model=str(kwargs.get("model") or ""),
                timeout_seconds=int(kwargs.get("timeout_seconds") or 45),
                temperature=float(kwargs.get("temperature") or 0.2),
            )
            settings.reload()
            return settings.to_dict()

        if normalized_action == "ensure_session_paths":
            return ensure_session_paths(kwargs.get("session_state"))

        if normalized_action == "sync_expert_knowledge_base":
            return sync_expert_knowledge_base(force_full=bool(kwargs.get("force_full", False)))

        if normalized_action == "fetch_vehicle_market_prices":
            import llm_engine

            return llm_engine.fetch_vehicle_market_prices(kwargs.get("vehicle_series") or [])

        if normalized_action == "explain_vehicle_gradient_deviations":
            import llm_engine

            return llm_engine.explain_vehicle_gradient_deviations(kwargs.get("rows") or [])

        raise KeyError(f"未注册的 Harness 动作: {normalized_action}")


def sync_expert_knowledge_base(*, force_full: bool = False) -> dict[str, Any]:
    import llm_engine

    return run_llm_action(
        "harness.sync_expert_knowledge_base",
        lambda: llm_engine.sync_expert_knowledge_base(force_full=force_full),
    )


def lock_new_blueprint(module_name: str, confirmation_text: str) -> dict[str, Any]:
    normalized_module_name = str(module_name or "").strip()
    if not normalized_module_name:
        raise ValueError("module_name 不能为空")
    if CONFIRMATION_PHRASE not in str(confirmation_text or ""):
        raise PermissionError("未检测到蓝图更新确认语句，拒绝更新锁定状态")

    blueprints = {item["module_name"]: item for item in _load_blueprints()}
    if normalized_module_name not in blueprints:
        raise KeyError(f"未找到模块蓝图: {normalized_module_name}")

    blueprint = blueprints[normalized_module_name]
    current_hash = _hash_source(str(blueprint["relative_path"]))
    _, current_structure_hash, _ = _build_module_structure(str(blueprint["relative_path"]))
    payload = _load_blueprint_locks()
    payload[normalized_module_name] = {
        "relative_path": str(blueprint["relative_path"]),
        "sha256": current_hash,
        "structure_sha256": current_structure_hash,
        "confirmed_at": _iso_now(),
        "confirmation_phrase": CONFIRMATION_PHRASE,
    }
    _save_blueprint_locks(payload)
    return {
        "status": "updated",
        "module_name": normalized_module_name,
        "relative_path": str(blueprint["relative_path"]),
        "sha256": current_hash,
        "structure_sha256": current_structure_hash,
        "confirmed_at": payload[normalized_module_name]["confirmed_at"],
    }


def confirm_and_update_blueprint(module_name: str, confirmation_text: str) -> dict[str, Any]:
    return lock_new_blueprint(module_name, confirmation_text)
