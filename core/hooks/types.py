"""Core hook primitives for lifecycle interception."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


HookAction = Literal["allow", "modify", "block"]


@dataclass(slots=True)
class HookDecision:
    """Single hook decision returned by a hook method."""

    action: HookAction = "allow"
    message: str = ""
    updates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str = "hook_blocked"

    @classmethod
    def allow(cls, *, metadata: Optional[dict[str, Any]] = None) -> "HookDecision":
        return cls(action="allow", metadata=dict(metadata or {}))

    @classmethod
    def modify(
        cls,
        updates: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "HookDecision":
        return cls(
            action="modify",
            updates=dict(updates or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def block(
        cls,
        message: str,
        *,
        error_type: str = "hook_blocked",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "HookDecision":
        return cls(
            action="block",
            message=message,
            error_type=error_type,
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class HookExecutionResult:
    """Aggregated outcome for a hook stage."""

    payload: dict[str, Any]
    audit: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    message: str = ""
    error_type: str = "hook_blocked"

