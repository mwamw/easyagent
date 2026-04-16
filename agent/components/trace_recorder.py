"""Trace recorder interfaces and default implementations for agent execution history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
from typing import Any, Callable, Optional
from uuid import uuid4


def default_trace_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if isinstance(value, list):
        return [default_trace_safe(item) for item in value]
    if isinstance(value, tuple):
        return [default_trace_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: default_trace_safe(item) for key, item in value.items()}
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class BaseTraceRecorder(ABC):
    """Abstract trace recorder used by BasicAgent."""

    @property
    @abstractmethod
    def trace_history(self) -> list[dict[str, Any]]:
        pass

    @trace_history.setter
    @abstractmethod
    def trace_history(self, value: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def begin_turn(self, raw_query: str) -> tuple[str, str]:
        pass

    @abstractmethod
    def get_last_turn_event_id(
        self,
        turn_id: Optional[str],
        *,
        exclude_types: Optional[set[str]] = None,
    ) -> Optional[str]:
        pass

    @abstractmethod
    def set_round_reasoning(
        self,
        content: str,
        *,
        turn_id: Optional[str],
        round_number: Optional[int],
        mode: str,
        stream: bool,
    ) -> Optional[str]:
        pass

    @abstractmethod
    def record_assistant_message(
        self,
        turn_id: Optional[str],
        content: Optional[str],
        *,
        parent_id: Optional[str] = None,
        stage: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        allow_empty: bool = False,
    ) -> Optional[str]:
        pass

    @abstractmethod
    def record_tool_call(
        self,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: Any,
        tool_id: str,
        *,
        parent_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> str:
        pass

    @abstractmethod
    def record_tool_result(
        self,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: Any,
        tool_id: str,
        content: Any,
        *,
        parent_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        success: Optional[bool] = None,
    ) -> str:
        pass

    @abstractmethod
    def record_turn_end(
        self,
        turn_id: Optional[str],
        *,
        final_event_id: Optional[str] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        status: str = "completed",
    ) -> str:
        pass

    @abstractmethod
    def get_trace_history(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryTraceRecorder(BaseTraceRecorder):
    """Default in-memory trace recorder preserving BasicAgent's historical behavior."""

    def __init__(
        self,
        *,
        trace_safe: Optional[Callable[[Any], Any]] = None,
        session_id: Optional[str] = None,
    ):
        self._trace_safe = trace_safe or default_trace_safe
        self._trace_history: list[dict[str, Any]] = []
        self._trace_session_id = session_id or f"trace_{uuid4().hex}"
        self._trace_event_counter = 0
        self._trace_seq = 0
        self._trace_turn_counter = 0

    @property
    def trace_history(self) -> list[dict[str, Any]]:
        return self._trace_history

    @trace_history.setter
    def trace_history(self, value: list[dict[str, Any]]) -> None:
        self._trace_history = list(value or [])
        self._normalize_trace_history()

    def export_state(self) -> dict[str, Any]:
        self._normalize_trace_history()
        return {
            "trace_history": self.get_trace_history(),
            "trace_session_id": self._trace_session_id,
            "trace_event_counter": self._trace_event_counter,
            "trace_seq": self._trace_seq,
            "trace_turn_counter": self._trace_turn_counter,
        }

    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        state = state or {}
        self._trace_history = list(state.get("trace_history", []))
        self._trace_session_id = state.get("trace_session_id") or f"trace_{uuid4().hex}"
        self._trace_event_counter = int(state.get("trace_event_counter") or 0)
        self._trace_seq = int(state.get("trace_seq") or 0)
        self._trace_turn_counter = int(state.get("trace_turn_counter") or 0)
        self._normalize_trace_history()

        legacy_thinking = [str(item) for item in state.get("thinking_history", []) if item is not None]
        if not self._trace_history and legacy_thinking:
            for content in legacy_thinking:
                self.record_event(
                    "reasoning",
                    role="assistant",
                    content=content,
                    turn_id=None,
                    metadata={"mode": "legacy"},
                )

    def record_event(
        self,
        event_type: str,
        *,
        role: str,
        content: str = "",
        turn_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **payload,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "id": self._next_trace_event_id(),
            "session_id": self._trace_session_id,
            "turn_id": turn_id,
            "seq": self._next_trace_seq(),
            "type": event_type,
            "timestamp": timestamp or datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": self._trace_safe(metadata or {}),
        }
        if parent_id is not None:
            event["parent_id"] = parent_id
        for key, value in payload.items():
            if value is None:
                continue
            event[key] = self._trace_safe(value)
        self._trace_history.append(event)
        return event

    def begin_turn(self, raw_query: str) -> tuple[str, str]:
        turn_id = self._next_turn_id()
        user_event = self.record_event(
            "user_message",
            role="user",
            content=raw_query,
            turn_id=turn_id,
        )
        return turn_id, str(user_event["id"])

    def get_last_turn_event_id(
        self,
        turn_id: Optional[str],
        *,
        exclude_types: Optional[set[str]] = None,
    ) -> Optional[str]:
        exclude = exclude_types or set()
        for event in reversed(self._trace_history):
            if event.get("turn_id") != turn_id:
                continue
            if event.get("type") in exclude:
                continue
            return str(event.get("id"))
        return None

    def set_round_reasoning(
        self,
        content: str,
        *,
        turn_id: Optional[str],
        round_number: Optional[int],
        mode: str,
        stream: bool,
    ) -> Optional[str]:
        if not content:
            return None
        for event in reversed(self._trace_history):
            if (
                event.get("type") in {"reasoning", "thinking"}
                and event.get("turn_id") == turn_id
                and event.get("round") == round_number
                and (event.get("metadata") or {}).get("mode") == mode
                and (event.get("metadata") or {}).get("stream") == stream
            ):
                event["content"] = content
                return str(event.get("id"))
        reasoning_event = self.record_event(
            "reasoning",
            role="assistant",
            content=content,
            turn_id=turn_id,
            round=round_number,
            metadata={
                "mode": mode,
                "stream": stream,
                "visibility": "internal",
            },
        )
        return str(reasoning_event["id"])

    def record_assistant_message(
        self,
        turn_id: Optional[str],
        content: Optional[str],
        *,
        parent_id: Optional[str] = None,
        stage: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        allow_empty: bool = False,
    ) -> Optional[str]:
        if not allow_empty and not content:
            return None
        event = self.record_event(
            "assistant_message",
            role="assistant",
            content=content or "",
            turn_id=turn_id,
            parent_id=parent_id,
            round=round_number,
            metadata={
                "stage": stage,
                "mode": mode,
                "stream": stream,
            },
        )
        return str(event["id"])

    def record_tool_call(
        self,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: Any,
        tool_id: str,
        *,
        parent_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> str:
        event = self.record_event(
            "tool_call",
            role="assistant",
            content="",
            turn_id=turn_id,
            parent_id=parent_id,
            round=round_number,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_id,
            metadata={
                "mode": mode,
                "stream": stream,
            },
        )
        return str(event["id"])

    def record_tool_result(
        self,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: Any,
        tool_id: str,
        content: Any,
        *,
        parent_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        success: Optional[bool] = None,
    ) -> str:
        event = self.record_event(
            "tool_result",
            role="tool",
            content=str(content),
            turn_id=turn_id,
            parent_id=parent_id,
            round=round_number,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_id,
            metadata={
                "mode": mode,
                "stream": stream,
                "success": success,
            },
        )
        return str(event["id"])

    def record_turn_end(
        self,
        turn_id: Optional[str],
        *,
        final_event_id: Optional[str] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        status: str = "completed",
    ) -> str:
        event = self.record_event(
            "turn_end",
            role="assistant",
            content="",
            turn_id=turn_id,
            metadata={
                "mode": mode,
                "stream": stream,
                "status": status,
                "final_event_id": final_event_id,
            },
        )
        return str(event["id"])

    def get_trace_history(self) -> list[dict[str, Any]]:
        return list(self._trace_history)

    def clear(self) -> None:
        self._trace_history.clear()
        self._trace_event_counter = 0
        self._trace_seq = 0
        self._trace_turn_counter = 0

    def _next_trace_event_id(self) -> str:
        self._trace_event_counter += 1
        return f"evt_{self._trace_event_counter:06d}"

    def _next_trace_seq(self) -> int:
        self._trace_seq += 1
        return self._trace_seq

    def _next_turn_id(self) -> str:
        self._trace_turn_counter += 1
        return f"turn_{self._trace_turn_counter:04d}"

    def _normalize_trace_history(self) -> None:
        normalized: list[dict[str, Any]] = []
        max_seq = self._trace_seq
        max_event_counter = self._trace_event_counter
        max_turn_counter = self._trace_turn_counter

        for event in self._trace_history:
            if not isinstance(event, dict):
                continue
            normalized_event = dict(event)
            event_type = str(normalized_event.get("type") or "unknown")
            if "timestamp" not in normalized_event:
                normalized_event["timestamp"] = normalized_event.pop("time", datetime.now().isoformat())
            if "session_id" not in normalized_event:
                normalized_event["session_id"] = self._trace_session_id
            if "seq" not in normalized_event:
                max_seq += 1
                normalized_event["seq"] = max_seq
            else:
                try:
                    max_seq = max(max_seq, int(normalized_event["seq"]))
                except Exception:
                    max_seq += 1
                    normalized_event["seq"] = max_seq
            if "id" not in normalized_event:
                max_event_counter += 1
                normalized_event["id"] = f"evt_{max_event_counter:06d}"
            else:
                event_id = str(normalized_event["id"])
                if event_id.startswith("evt_"):
                    try:
                        max_event_counter = max(max_event_counter, int(event_id.split("_", 1)[1]))
                    except Exception:
                        pass
            if "turn_id" in normalized_event and normalized_event["turn_id"]:
                turn_id = str(normalized_event["turn_id"])
                if turn_id.startswith("turn_"):
                    try:
                        max_turn_counter = max(max_turn_counter, int(turn_id.split("_", 1)[1]))
                    except Exception:
                        pass
            if "metadata" not in normalized_event or normalized_event["metadata"] is None:
                normalized_event["metadata"] = {}
            if "role" not in normalized_event:
                if event_type in {"reasoning", "thinking", "assistant_message"}:
                    normalized_event["role"] = "assistant"
                elif event_type == "user_message":
                    normalized_event["role"] = "user"
                elif event_type == "tool_result":
                    normalized_event["role"] = "tool"
                else:
                    normalized_event["role"] = "assistant"
            if event_type == "thinking":
                normalized_event["type"] = "reasoning"
            normalized.append(normalized_event)

        normalized.sort(key=lambda item: (int(item.get("seq", 0) or 0), str(item.get("timestamp", ""))))
        self._trace_history = normalized
        self._trace_seq = max_seq
        self._trace_event_counter = max_event_counter
        self._trace_turn_counter = max_turn_counter


__all__ = ["BaseTraceRecorder", "InMemoryTraceRecorder", "default_trace_safe"]
