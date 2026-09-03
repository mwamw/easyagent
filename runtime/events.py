"""Typed runtime events shared by agents and optional modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Iterable
from uuid import uuid4


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="python"))
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


class RuntimeEventType(str, Enum):
    AGENT_INVOKE_STARTED = "agent.invoke.started"
    AGENT_INVOKE_COMPLETED = "agent.invoke.completed"
    AGENT_INVOKE_FAILED = "agent.invoke.failed"
    AGENT_INVOKE_INTERRUPTED = "agent.invoke.interrupted"
    LLM_INVOKE_STARTED = "llm.invoke.started"
    LLM_INVOKE_COMPLETED = "llm.invoke.completed"
    LLM_INVOKE_FAILED = "llm.invoke.failed"
    TOOL_INVOKE_STARTED = "tool.invoke.started"
    TOOL_INVOKE_COMPLETED = "tool.invoke.completed"
    TOOL_INVOKE_FAILED = "tool.invoke.failed"
    HISTORY_COMPACTED = "history.compacted"
    STREAM_EVENT = "agent.stream.event"


class AgentStreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class AgentStreamEvent:
    type: AgentStreamEventType
    invocation_id: str
    sequence: int
    content: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "invocationId": self.invocation_id,
            "sequence": self.sequence,
            "content": self.content,
            "data": _json_safe(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStreamEvent":
        return cls(
            type=AgentStreamEventType(str(data["type"])),
            invocation_id=str(data["invocationId"]),
            sequence=int(data.get("sequence") or 0),
            content=data.get("content"),
            data=dict(data.get("data") or {}),
        )


@dataclass(slots=True, frozen=True)
class RuntimeEvent:
    type: RuntimeEventType
    agent_id: str
    invocation_id: str
    sequence: int = 0
    event_id: str = field(default_factory=lambda: f"event_{uuid4().hex}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "type": self.type.value,
            "agentId": self.agent_id,
            "invocationId": self.invocation_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "data": _json_safe(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeEvent":
        raw_timestamp = data.get("timestamp")
        if isinstance(raw_timestamp, datetime):
            timestamp = raw_timestamp
        elif raw_timestamp:
            timestamp = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
        else:
            timestamp = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return cls(
            type=RuntimeEventType(str(data["type"])),
            agent_id=str(data["agentId"]),
            invocation_id=str(data["invocationId"]),
            sequence=int(data.get("sequence") or 0),
            event_id=str(data.get("eventId") or f"event_{uuid4().hex}"),
            timestamp=timestamp,
            data=dict(data.get("data") or {}),
        )


RuntimeEventHandler = Callable[[RuntimeEvent], None]


@dataclass(slots=True)
class _Subscription:
    handler: RuntimeEventHandler
    event_types: frozenset[RuntimeEventType] | None


class RuntimeEventBus:
    """Synchronous in-process event bus with deterministic event ordering."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._subscriptions: dict[str, _Subscription] = {}
        self._sequence_by_invocation: dict[str, int] = {}
        self._history: list[RuntimeEvent] = []
        self._subscriber_errors: list[dict[str, str]] = []

    def subscribe(
        self,
        handler: RuntimeEventHandler,
        *,
        event_types: Iterable[RuntimeEventType] | None = None,
    ) -> str:
        if not callable(handler):
            raise TypeError("handler must be callable")
        token = f"subscription_{uuid4().hex}"
        resolved_types = frozenset(event_types) if event_types is not None else None
        with self._lock:
            self._subscriptions[token] = _Subscription(handler, resolved_types)
        return token

    def unsubscribe(self, token: str) -> bool:
        with self._lock:
            return self._subscriptions.pop(str(token), None) is not None

    def emit(self, event: RuntimeEvent) -> RuntimeEvent:
        if not isinstance(event, RuntimeEvent):
            raise TypeError("event must be RuntimeEvent")
        with self._lock:
            sequence = self._sequence_by_invocation.get(event.invocation_id, 0) + 1
            self._sequence_by_invocation[event.invocation_id] = sequence
            resolved = replace(event, sequence=sequence)
            self._history.append(resolved)
            subscriptions = list(self._subscriptions.values())
        for subscription in subscriptions:
            if subscription.event_types is not None and resolved.type not in subscription.event_types:
                continue
            try:
                subscription.handler(resolved)
            except Exception as exc:
                with self._lock:
                    self._subscriber_errors.append(
                        {
                            "eventId": resolved.event_id,
                            "eventType": resolved.type.value,
                            "handler": getattr(subscription.handler, "__qualname__", type(subscription.handler).__name__),
                            "errorType": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )
        return resolved

    def publish(
        self,
        event_type: RuntimeEventType,
        *,
        agent_id: str,
        invocation_id: str,
        data: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return self.emit(
            RuntimeEvent(
                type=event_type,
                agent_id=str(agent_id),
                invocation_id=str(invocation_id),
                data=dict(data or {}),
            )
        )

    def history(self, *, invocation_id: str | None = None) -> list[RuntimeEvent]:
        with self._lock:
            events = list(self._history)
        if invocation_id is not None:
            events = [event for event in events if event.invocation_id == invocation_id]
        return events

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._sequence_by_invocation.clear()
            self._subscriber_errors.clear()

    def restore_history(self, events: Iterable[RuntimeEvent | dict[str, Any]]) -> None:
        """Replace trace history without notifying live subscribers."""
        restored = [
            event if isinstance(event, RuntimeEvent) else RuntimeEvent.from_dict(event)
            for event in events
        ]
        sequences: dict[str, int] = {}
        for event in restored:
            sequences[event.invocation_id] = max(
                sequences.get(event.invocation_id, 0),
                event.sequence,
            )
        with self._lock:
            self._history = restored
            self._sequence_by_invocation = sequences
            self._subscriber_errors.clear()

    def subscriber_errors(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self._subscriber_errors]


__all__ = [
    "AgentStreamEvent",
    "AgentStreamEventType",
    "RuntimeEvent",
    "RuntimeEventBus",
    "RuntimeEventHandler",
    "RuntimeEventType",
]
