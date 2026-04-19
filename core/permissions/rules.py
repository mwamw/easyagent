"""Permission rule matching and risk derivation helpers."""

from __future__ import annotations

from typing import Any

from Tool.BaseTool import Tool

from .context import PermissionContext
from .types import PermissionRule, RiskCategory


PATH_PARAM_NAMES = {
    "file_path",
    "path",
    "notebook_path",
    "directory",
    "cwd",
    "workspace_root",
}
COMMAND_PARAM_NAMES = {"command", "cmd"}


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def extract_path_values(parameters: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in parameters.items():
        if key in PATH_PARAM_NAMES and isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def extract_command_values(parameters: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in parameters.items():
        if key in COMMAND_PARAM_NAMES and isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def derive_risk_categories(tool: Tool, parameters: dict[str, Any]) -> list[str]:
    spec = tool.get_spec()
    explicit = [str(item) for item in getattr(spec, "risk_categories", []) or []]
    if explicit:
        return explicit

    names = {tool.name, tool.name.lower()}
    derived: list[str] = []

    if spec.read_only:
        if names & {"fileread", "glob", "grep"}:
            derived.append(RiskCategory.FILESYSTEM_READ.value)
        if names & {"webfetch", "websearch", "web_search"}:
            derived.append(RiskCategory.NETWORK.value)
    else:
        if spec.destructive or names & {"filewrite", "fileedit", "notebookedit"}:
            derived.append(RiskCategory.FILESYSTEM_WRITE.value)
        if names & {"bash"}:
            derived.extend(
                [
                    RiskCategory.SHELL.value,
                    RiskCategory.PROCESS.value,
                    RiskCategory.SIDE_EFFECT.value,
                ]
            )
        if names & {"taskstop"}:
            derived.extend([RiskCategory.PROCESS.value, RiskCategory.SIDE_EFFECT.value])
        if names & {"enterworktree", "exitworktree"}:
            derived.extend([RiskCategory.FILESYSTEM_WRITE.value, RiskCategory.SIDE_EFFECT.value])

    if names & {"listmcpresources", "readmcpresource"}:
        derived.append(RiskCategory.MCP.value)
    if tool.get_spec().metadata.get("mcp_server"):
        derived.append(RiskCategory.MCP.value)

    seen: set[str] = set()
    result: list[str] = []
    for item in derived:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _rule_tool_matches(rule: PermissionRule, tool: Tool) -> bool:
    return rule.tool_name in {"*", tool.name}


def _rule_matches_risk(rule: PermissionRule, risk_categories: list[str]) -> bool:
    expected = _normalize_str_list(rule.matcher.get("risk_categories"))
    if not expected:
        return True
    return bool(set(expected) & set(risk_categories))


def _rule_matches_path(rule: PermissionRule, parameters: dict[str, Any]) -> bool:
    prefixes = _normalize_str_list(rule.matcher.get("path_prefixes"))
    if not prefixes:
        return True
    values = extract_path_values(parameters)
    return any(any(value.startswith(prefix) for prefix in prefixes) for value in values)


def _rule_matches_command(rule: PermissionRule, parameters: dict[str, Any]) -> bool:
    prefixes = _normalize_str_list(rule.matcher.get("command_prefixes"))
    if not prefixes:
        return True
    values = extract_command_values(parameters)
    return any(any(value.startswith(prefix) for prefix in prefixes) for value in values)


def _rule_matches_param_equals(rule: PermissionRule, parameters: dict[str, Any]) -> bool:
    expected = rule.matcher.get("param_equals") or {}
    if not expected:
        return True
    for key, value in expected.items():
        if parameters.get(key) != value:
            return False
    return True


def _rule_matches_param_contains(rule: PermissionRule, parameters: dict[str, Any]) -> bool:
    expected = rule.matcher.get("param_contains") or {}
    if not expected:
        return True
    for key, value in expected.items():
        haystack = str(parameters.get(key, ""))
        if str(value) not in haystack:
            return False
    return True


def find_matching_rule(
    context: PermissionContext,
    tool: Tool,
    parameters: dict[str, Any],
    *,
    risk_categories: list[str],
) -> PermissionRule | None:
    for rule in list(context.rules or []):
        if not _rule_tool_matches(rule, tool):
            continue
        if not _rule_matches_risk(rule, risk_categories):
            continue
        if not _rule_matches_path(rule, parameters):
            continue
        if not _rule_matches_command(rule, parameters):
            continue
        if not _rule_matches_param_equals(rule, parameters):
            continue
        if not _rule_matches_param_contains(rule, parameters):
            continue
        return rule
    return None

