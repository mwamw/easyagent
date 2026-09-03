"""Core models for runtime meta-message injection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import time
from typing import Any, Callable, Optional
from uuid import uuid4


class MetaMessageLifecycle(str, Enum):
    """How long an injected meta message remains in agent history."""

    PERMANENT = "permanent"
    INVOCATION = "invocation"
    CONDITIONAL = "conditional"
    REQUEST = "request"


@dataclass(slots=True)
class MetaMessageContext:
    """Read-only runtime state exposed to content factories and conditions."""

    event_name: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    execution_mode: str = "execute"
    permission_mode: str = "default"
    current_task_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


MetaMessageContent = str | Callable[[MetaMessageContext], str]
MetaMessageCondition = Callable[[MetaMessageContext], bool]


@dataclass(slots=True)
class MetaMessage:
    """A user-role runtime instruction managed independently from system prompt."""

    name: str
    content: MetaMessageContent
    lifecycle: MetaMessageLifecycle = MetaMessageLifecycle.PERMANENT
    condition: Optional[MetaMessageCondition] = None
    dedup_key: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: f"meta_{uuid4().hex}")
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not self.name:
            raise ValueError("MetaMessage.name must be a non-empty string")
        if not isinstance(self.lifecycle, MetaMessageLifecycle):
            self.lifecycle = MetaMessageLifecycle(str(self.lifecycle))
        if self.lifecycle == MetaMessageLifecycle.CONDITIONAL and self.condition is None:
            raise ValueError("CONDITIONAL MetaMessage requires a condition function")
        if self.lifecycle != MetaMessageLifecycle.CONDITIONAL and self.condition is not None:
            raise ValueError("Only CONDITIONAL MetaMessages may define a condition function")
        if self.dedup_key is not None:
            self.dedup_key = str(self.dedup_key).strip() or None
        self.metadata = dict(self.metadata or {})

    def should_activate(self, context: MetaMessageContext) -> bool:
        return True if self.condition is None else bool(self.condition(context))

    def render(self, context: MetaMessageContext) -> str:
        value = self.content(context) if callable(self.content) else self.content
        return str(value or "").strip()

    def clone(self, **changes: Any) -> "MetaMessage":
        defaults = {
            "message_id": f"meta_{uuid4().hex}",
            "created_at": time.time(),
            "metadata": dict(self.metadata),
        }
        defaults.update(changes)
        return replace(self, **defaults)

    def to_dict(self) -> dict[str, Any]:
        if callable(self.content) or self.condition is not None:
            raise TypeError("Callable MetaMessage definitions cannot be serialized")
        return {
            "name": self.name,
            "content": self.content,
            "lifecycle": self.lifecycle.value,
            "dedupKey": self.dedup_key,
            "metadata": dict(self.metadata),
            "messageId": self.message_id,
            "createdAt": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaMessage":
        data = dict(payload or {})
        return cls(
            name=str(data.get("name") or ""),
            content=str(data.get("content") or ""),
            lifecycle=MetaMessageLifecycle(str(data.get("lifecycle") or "permanent")),
            dedup_key=data.get("dedupKey"),
            metadata=dict(data.get("metadata") or {}),
            message_id=str(data.get("messageId") or f"meta_{uuid4().hex}"),
            created_at=float(data.get("createdAt") or time.time()),
        )


@dataclass(slots=True)
class MetaMessageInjection:
    """Tracks one concrete history insertion."""

    injection_id: str
    message: MetaMessage
    history_handle: str
    injected_at: float = field(default_factory=time.time)


__all__ = [
    "MetaMessage",
    "MetaMessageCondition",
    "MetaMessageContent",
    "MetaMessageContext",
    "MetaMessageInjection",
    "MetaMessageLifecycle",
]
