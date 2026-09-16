"""Default Claude Code-style policy engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..base import BasePermissionEngine
from ..models import (
    ApprovalMode,
    PermissionBehavior,
    PermissionDecision,
    PermissionGrant,
    PermissionRequest,
    ToolPermissionDescriptor,
)
from .matcher import RuleMatcher
from .policy import PolicyRule, PolicySet, load_policy_file, policy_locations


class RuleBasedPermissionEngine(BasePermissionEngine):
    """Deterministic policy engine; all persistence and matching is product-owned."""

    def __init__(
        self,
        *,
        approval_mode: ApprovalMode = ApprovalMode.ASK_FOR_APPROVAL,
        workspace_root: str | None = None,
        policy_directory: str | None = None,
        policy_paths: list[str] | None = None,
        load_policies: bool = True,
    ) -> None:
        self.approval_mode = ApprovalMode(approval_mode)
        self.workspace_root = workspace_root
        self._policy = PolicySet()
        self._grants: list[PermissionGrant] = []
        self._policy_revision = 0
        self._matcher = RuleMatcher()
        if load_policies:
            for location in policy_locations(policy_directory=policy_directory, policy_paths=policy_paths):
                loaded = load_policy_file(location, source=str(location))
                self._policy.extend(loaded.rules)
            self._policy_revision = self._policy.revision

    def describe_tool(self, tool: Any) -> ToolPermissionDescriptor:
        getter = getattr(tool, "get_permission_descriptor", None)
        if callable(getter):
            descriptor = getter()
            if isinstance(descriptor, ToolPermissionDescriptor):
                return descriptor
        raise TypeError(f"Tool {getattr(tool, 'name', '<unknown>')} has no ToolPermissionDescriptor")

    def set_approval_mode(self, mode: ApprovalMode | str) -> None:
        self.approval_mode = ApprovalMode(mode)
        self._policy_revision += 1

    def add_rule(self, *, tool_name: str, behavior: PermissionBehavior | str,
                 matcher: dict[str, Any] | None = None, source: str = "session",
                 description: str = "", pattern: str | None = None) -> None:
        rule = PolicyRule(behavior, tool_name, matcher or {}, source, description, pattern)
        self._policy.rules.append(rule)
        self._policy_revision += 1

    def clear_rules(self, source: str | None = None) -> None:
        self._policy.clear(source)
        self._policy_revision += 1

    def rules(self) -> list[dict[str, Any]]:
        return [rule.to_dict() for rule in self._policy.rules]

    def grant(self, grant: PermissionGrant) -> None:
        self._grants.append(grant)
        self._policy_revision += 1

    def grants(self) -> list[PermissionGrant]:
        return list(self._grants)

    def export_state(self) -> dict[str, Any]:
        return {
            "approval_mode": self.approval_mode.value,
            "policy": self._policy.to_dict(),
            "grants": [
                {"tool_name": grant.tool_name, "arguments": dict(grant.arguments),
                 "scope": grant.scope, "session_id": grant.session_id,
                 "metadata": dict(grant.metadata)}
                for grant in self._grants
            ],
            "policy_revision": self._policy_revision,
        }

    def restore_state(self, state: dict[str, Any] | None) -> None:
        payload = dict(state or {})
        self.approval_mode = ApprovalMode(payload.get("approval_mode", ApprovalMode.ASK_FOR_APPROVAL.value))
        self._policy = PolicySet.from_value(payload.get("policy"))
        # Accept the prior state shape only as serialized data, not as a public API.
        if not self._policy.rules and payload.get("rules"):
            self._policy.extend(payload["rules"])
        self._grants = [PermissionGrant(**item) for item in payload.get("grants", [])]
        self._policy_revision = int(payload.get("policy_revision", self._policy.revision))

    def authorize(self, request: PermissionRequest) -> PermissionDecision:
        if request.context.runtime_mode == "plan":
            allowed = set(request.context.plan_allowed_actions)
            if request.tool_name not in allowed and (
                request.descriptor.side_effect_level != "none" or request.descriptor.risk_categories
            ):
                return PermissionDecision(
                    PermissionBehavior.DENY,
                    "This tool is not allowed in plan mode.",
                    {"runtime_mode": "plan", "policy_revision": self._policy_revision},
                )
        matching: list[PolicyRule] = []
        for rule in self._policy.rules:
            if self._matcher.matches(rule, request):
                matching.append(rule)
        # Explicit deny rules always win, including over one-shot grants.
        for behavior in (PermissionBehavior.DENY,):
            for rule in matching:
                if rule.behavior == behavior:
                    return PermissionDecision(
                        behavior,
                        rule.description or f"Matched {behavior.value} permission rule.",
                        {"source": rule.source, "policy_revision": self._policy_revision},
                    )

        for index, grant in enumerate(self._grants):
            if grant.tool_name != request.tool_name:
                continue
            if dict(grant.arguments) != dict(request.arguments):
                continue
            if grant.session_id and grant.session_id != request.context.session_id:
                continue
            if grant.scope == "request":
                self._grants.pop(index)
                self._policy_revision += 1
            return PermissionDecision(
                PermissionBehavior.ALLOW,
                "Matched an explicit permission grant.",
                {"grant_id": grant.grant_id, "policy_revision": self._policy_revision},
            )

        # Claude Code precedence is behavior-first, not specificity-first.
        for behavior in (PermissionBehavior.ASK, PermissionBehavior.ALLOW):
            for rule in matching:
                if rule.behavior == behavior:
                    return PermissionDecision(
                        behavior,
                        rule.description or f"Matched {behavior.value} permission rule.",
                        {"source": rule.source, "policy_revision": self._policy_revision},
                    )

        sandbox = request.extensions.get("sandbox")
        if isinstance(sandbox, dict) and (
            sandbox.get("auto_allow") is True
            and sandbox.get("will_sandbox") is True
            and sandbox.get("bypassed") is not True
        ):
            return PermissionDecision(
                PermissionBehavior.ALLOW,
                "Auto-allowed because Bash will execute inside the configured sandbox.",
                {"sandbox_auto_allow": True, "policy_revision": self._policy_revision},
            )

        risks = set(request.descriptor.risk_categories)
        dangerous = request.descriptor.side_effect_level == "high" or bool(
            risks.intersection({"filesystem_write", "shell", "network", "process", "mcp", "side_effect"})
        )
        if self.approval_mode == ApprovalMode.AUTO_APPROVE_SAFE and risks.issubset({"filesystem_read", "filesystem_write"}):
            dangerous = False
        behavior = PermissionBehavior.ALLOW if not dangerous or self.approval_mode == ApprovalMode.FULL_ACCESS else PermissionBehavior.ASK
        return PermissionDecision(behavior, "Default permission policy.", {"policy_revision": self._policy_revision})


PermissionEngine = RuleBasedPermissionEngine
