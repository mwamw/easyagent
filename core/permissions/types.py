"""Permission type definitions for EasyAgent runtime."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "accept_edits"
    DONT_ASK = "dont_ask"
    BYPASS = "bypass"


class RiskCategory(str, Enum):
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    SHELL = "shell"
    NETWORK = "network"
    PROCESS = "process"
    MCP = "mcp"
    SIDE_EFFECT = "side_effect"


class PermissionRule(BaseModel):
    tool_name: str = Field(description="Tool 名称，支持 '*' 匹配所有工具")
    behavior: PermissionBehavior = Field(description="命中规则后的行为")
    matcher: dict[str, Any] = Field(default_factory=dict, description="规则匹配器")
    source: str = Field(default="session", description="规则来源")
    description: str | None = Field(default=None, description="规则说明")


class PermissionDecision(BaseModel):
    behavior: PermissionBehavior
    tool_name: str
    reason: str
    matched_rule_source: str | None = None
    risk_categories: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        *,
        tool_name: str,
        reason: str,
        matched_rule_source: str | None = None,
        risk_categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PermissionDecision":
        return cls(
            behavior=PermissionBehavior.ALLOW,
            tool_name=tool_name,
            reason=reason,
            matched_rule_source=matched_rule_source,
            risk_categories=list(risk_categories or []),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def deny(
        cls,
        *,
        tool_name: str,
        reason: str,
        matched_rule_source: str | None = None,
        risk_categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PermissionDecision":
        return cls(
            behavior=PermissionBehavior.DENY,
            tool_name=tool_name,
            reason=reason,
            matched_rule_source=matched_rule_source,
            risk_categories=list(risk_categories or []),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def ask(
        cls,
        *,
        tool_name: str,
        reason: str,
        matched_rule_source: str | None = None,
        risk_categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PermissionDecision":
        return cls(
            behavior=PermissionBehavior.ASK,
            tool_name=tool_name,
            reason=reason,
            matched_rule_source=matched_rule_source,
            risk_categories=list(risk_categories or []),
            requires_confirmation=True,
            metadata=dict(metadata or {}),
        )

