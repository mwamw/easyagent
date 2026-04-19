"""Permission engine for tool authorization."""

from __future__ import annotations

from Tool.BaseTool import Tool

from .context import PermissionContext
from .rules import derive_risk_categories, find_matching_rule
from .types import PermissionDecision, PermissionMode, RiskCategory


_PLAN_BLOCKED_RISKS = {
    RiskCategory.FILESYSTEM_WRITE.value,
    RiskCategory.SHELL.value,
    RiskCategory.NETWORK.value,
    RiskCategory.PROCESS.value,
    RiskCategory.SIDE_EFFECT.value,
}

_EDIT_RISKS = {
    RiskCategory.FILESYSTEM_READ.value,
    RiskCategory.FILESYSTEM_WRITE.value,
}

_HIGH_RISK_CATEGORIES = {
    RiskCategory.SHELL.value,
    RiskCategory.NETWORK.value,
    RiskCategory.PROCESS.value,
    RiskCategory.MCP.value,
    RiskCategory.SIDE_EFFECT.value,
}


class PermissionEngine:
    def authorize(
        self,
        tool: Tool,
        parameters: dict[str, object],
        context: PermissionContext | None = None,
    ) -> PermissionDecision:
        context = context or PermissionContext()
        spec = tool.get_spec()
        risk_categories = derive_risk_categories(tool, parameters)

        if context.mode == PermissionMode.BYPASS:
            return PermissionDecision.allow(
                tool_name=tool.name,
                reason="当前权限模式为 bypass。",
                risk_categories=risk_categories,
            )

        matched_rule = find_matching_rule(
            context,
            tool,
            parameters,
            risk_categories=risk_categories,
        )
        if matched_rule is not None:
            if matched_rule.behavior.value == "allow":
                return PermissionDecision.allow(
                    tool_name=tool.name,
                    reason=matched_rule.description or "命中允许规则。",
                    matched_rule_source=matched_rule.source,
                    risk_categories=risk_categories,
                )
            if matched_rule.behavior.value == "deny":
                return PermissionDecision.deny(
                    tool_name=tool.name,
                    reason=matched_rule.description or "命中拒绝规则。",
                    matched_rule_source=matched_rule.source,
                    risk_categories=risk_categories,
                )
            return PermissionDecision.ask(
                tool_name=tool.name,
                reason=matched_rule.description or "命中确认规则，需要用户确认。",
                matched_rule_source=matched_rule.source,
                risk_categories=risk_categories,
            )

        if context.mode == PermissionMode.PLAN and (
            spec.destructive or bool(set(risk_categories) & _PLAN_BLOCKED_RISKS)
        ):
            return PermissionDecision.deny(
                tool_name=tool.name,
                reason="当前处于 plan 模式，高风险工具执行被阻止。",
                risk_categories=risk_categories,
            )

        if context.mode == PermissionMode.ACCEPT_EDITS:
            if risk_categories and set(risk_categories).issubset(_EDIT_RISKS):
                return PermissionDecision.allow(
                    tool_name=tool.name,
                    reason="当前权限模式为 accept_edits，文件读写类工具允许直接执行。",
                    risk_categories=risk_categories,
                )

        if context.mode == PermissionMode.DONT_ASK and spec.requires_confirmation:
            return PermissionDecision.deny(
                tool_name=tool.name,
                reason="当前权限模式为 dont_ask，需确认工具被自动拒绝。",
                risk_categories=risk_categories,
            )

        if context.mode == PermissionMode.DONT_ASK and (
            spec.destructive or bool(set(risk_categories) & _HIGH_RISK_CATEGORIES)
        ):
            return PermissionDecision.deny(
                tool_name=tool.name,
                reason="当前权限模式为 dont_ask，高风险工具被自动拒绝。",
                risk_categories=risk_categories,
            )

        if spec.requires_confirmation:
            return PermissionDecision.ask(
                tool_name=tool.name,
                reason=f"工具 '{tool.name}' 需要用户确认后才能执行。",
                risk_categories=risk_categories,
            )

        return PermissionDecision.allow(
            tool_name=tool.name,
            reason="未命中特殊权限规则，默认允许执行。",
            risk_categories=risk_categories,
        )
