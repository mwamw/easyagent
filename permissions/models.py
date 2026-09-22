"""Minimal stable permission models shared by framework and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ToolPermissionDescriptor:
    """Generic Tool security description; default engines may use extensions."""

    tool_name: str
    capabilities: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    risk_categories: tuple[str, ...] = ()
    side_effect_level: str = "none"
    requires_confirmation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorizationContext:
    actor_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    workspace_root: str | None = None
    runtime_mode: str = "execute"
    plan_allowed_actions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    arguments: Mapping[str, Any]
    descriptor: ToolPermissionDescriptor
    context: AuthorizationContext = field(default_factory=AuthorizationContext)
    extensions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PermissionDecision:
    behavior: PermissionBehavior
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PermissionGrant:
    """A user-issued approval for a normalized tool request."""

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    scope: str = "request"
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""
    grant_id: str = field(default_factory=lambda: uuid4().hex)
