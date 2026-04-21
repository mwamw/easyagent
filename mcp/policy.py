"""Policy helpers for MCP runtime authorization and behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.permissions import PermissionBehavior, PermissionRule, RiskCategory


MCPPolicyEffect = Literal["allow", "deny"]
MCPCapabilityKind = Literal[
    "tool",
    "tool_list",
    "resource_list",
    "resource_read",
    "prompt_list",
    "prompt_get",
]


@dataclass(slots=True)
class MCPPolicyRule:
    effect: MCPPolicyEffect = "allow"
    server_names: tuple[str, ...] = ()
    capability_kinds: tuple[str, ...] = ()
    capability_names: tuple[str, ...] = ()
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "MCPPolicyRule":
        if isinstance(value, MCPPolicyRule):
            return value
        payload = dict(value or {})
        return cls(
            effect=str(payload.get("effect") or "allow").strip().lower() or "allow",
            server_names=tuple(
                str(item).strip()
                for item in list(payload.get("server_names") or payload.get("serverNames") or [])
                if str(item).strip()
            ),
            capability_kinds=tuple(
                str(item).strip()
                for item in list(payload.get("capability_kinds") or payload.get("capabilityKinds") or [])
                if str(item).strip()
            ),
            capability_names=tuple(
                str(item).strip()
                for item in list(payload.get("capability_names") or payload.get("capabilityNames") or [])
                if str(item).strip()
            ),
            reason=str(payload.get("reason") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def matches(
        self,
        *,
        server_name: str,
        capability_kind: str,
        capability_name: Optional[str] = None,
    ) -> bool:
        if self.server_names and server_name not in self.server_names:
            return False
        if self.capability_kinds and capability_kind not in self.capability_kinds:
            return False
        if self.capability_names and str(capability_name or "") not in self.capability_names:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect,
            "serverNames": list(self.server_names),
            "capabilityKinds": list(self.capability_kinds),
            "capabilityNames": list(self.capability_names),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MCPPolicyDecision:
    allowed: bool
    reason: str
    matched_rule_index: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "matchedRuleIndex": self.matched_rule_index,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MCPPolicyContext:
    default_behavior: MCPPolicyEffect = "allow"
    allow_auto_connect: bool = True
    capability_cache_ttl_seconds: int = 300
    resource_cache_ttl_seconds: int = 0
    prompt_cache_ttl_seconds: int = 0
    max_retries: int = 1
    persist_connection: bool = False
    rules: list[MCPPolicyRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "MCPPolicyContext":
        if isinstance(value, MCPPolicyContext):
            return value
        if value is None:
            return cls()
        payload = dict(value)
        return cls(
            default_behavior=str(payload.get("default_behavior") or payload.get("defaultBehavior") or "allow").strip().lower() or "allow",
            allow_auto_connect=bool(payload.get("allow_auto_connect", payload.get("allowAutoConnect", True))),
            capability_cache_ttl_seconds=int(payload.get("capability_cache_ttl_seconds", payload.get("capabilityCacheTtlSeconds", 300)) or 0),
            resource_cache_ttl_seconds=int(payload.get("resource_cache_ttl_seconds", payload.get("resourceCacheTtlSeconds", 0)) or 0),
            prompt_cache_ttl_seconds=int(payload.get("prompt_cache_ttl_seconds", payload.get("promptCacheTtlSeconds", 0)) or 0),
            max_retries=int(payload.get("max_retries", payload.get("maxRetries", 1)) or 0),
            persist_connection=bool(payload.get("persist_connection", payload.get("persistConnection", False))),
            rules=[MCPPolicyRule.from_value(item) for item in list(payload.get("rules") or [])],
            metadata=dict(payload.get("metadata") or {}),
        )

    def authorize(
        self,
        *,
        server_name: str,
        capability_kind: str,
        capability_name: Optional[str] = None,
    ) -> MCPPolicyDecision:
        normalized_server = str(server_name or "").strip()
        normalized_kind = str(capability_kind or "").strip()
        normalized_name = str(capability_name or "").strip() or None

        for index, rule in enumerate(self.rules):
            if not rule.matches(
                server_name=normalized_server,
                capability_kind=normalized_kind,
                capability_name=normalized_name,
            ):
                continue
            if rule.effect == "deny":
                return MCPPolicyDecision(
                    allowed=False,
                    reason=rule.reason or f"MCP policy 拒绝访问 server `{normalized_server}` 的 `{normalized_kind}` 能力。",
                    matched_rule_index=index,
                    metadata=dict(rule.metadata),
                )
            return MCPPolicyDecision(
                allowed=True,
                reason=rule.reason or "命中 MCP allow policy。",
                matched_rule_index=index,
                metadata=dict(rule.metadata),
            )

        if self.default_behavior == "deny":
            return MCPPolicyDecision(
                allowed=False,
                reason=f"MCP policy 默认拒绝访问 server `{normalized_server}` 的 `{normalized_kind}` 能力。",
            )
        return MCPPolicyDecision(allowed=True, reason="MCP policy 默认允许访问。")

    def to_permission_rules(self) -> list[PermissionRule]:
        converted: list[PermissionRule] = []
        for rule in self.rules:
            behavior = (
                PermissionBehavior.ALLOW
                if rule.effect == "allow"
                else PermissionBehavior.DENY
            )
            converted.append(
                PermissionRule(
                    tool_name="*",
                    behavior=behavior,
                    matcher={
                        "risk_categories": [RiskCategory.MCP.value],
                        "mcp_servers": list(rule.server_names),
                    },
                    source="mcp_policy",
                    description=rule.reason or None,
                )
            )
        return converted

    def export_state(self) -> dict[str, Any]:
        return {
            "defaultBehavior": self.default_behavior,
            "allowAutoConnect": self.allow_auto_connect,
            "capabilityCacheTtlSeconds": self.capability_cache_ttl_seconds,
            "resourceCacheTtlSeconds": self.resource_cache_ttl_seconds,
            "promptCacheTtlSeconds": self.prompt_cache_ttl_seconds,
            "maxRetries": self.max_retries,
            "persistConnection": self.persist_connection,
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": dict(self.metadata),
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        restored = self.from_value(payload)
        self.default_behavior = restored.default_behavior
        self.allow_auto_connect = restored.allow_auto_connect
        self.capability_cache_ttl_seconds = restored.capability_cache_ttl_seconds
        self.resource_cache_ttl_seconds = restored.resource_cache_ttl_seconds
        self.prompt_cache_ttl_seconds = restored.prompt_cache_ttl_seconds
        self.max_retries = restored.max_retries
        self.persist_connection = restored.persist_connection
        self.rules = list(restored.rules)
        self.metadata = dict(restored.metadata)
