"""Provider-neutral conversation history owned by an Agent."""

from __future__ import annotations

from threading import RLock
from typing import Any, Iterable

from core.history import CanonicalMessage, _json_safe, coerce_canonical_message
from core.llm import EasyLLM


class ConversationHistory:
    """Keeps canonical records and current-provider replay data in sync."""

    def __init__(self, llm: EasyLLM):
        if not isinstance(llm, EasyLLM):
            raise TypeError(f"llm must be EasyLLM, got {type(llm).__name__}")
        self._llm = llm
        self._canonical: list[CanonicalMessage] = []
        self._replay: list[Any] = []
        self._provider_name = llm.provider_name
        self._lock = RLock()

    @property
    def provider_name(self) -> str | None:
        return self._provider_name

    @property
    def canonical(self) -> list[CanonicalMessage]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._canonical]

    @property
    def replay(self) -> list[Any]:
        with self._lock:
            return [_json_safe(item) for item in self._replay]

    def _extend(self, canonical: Iterable[CanonicalMessage], replay: Iterable[Any]) -> None:
        with self._lock:
            self._canonical.extend(item.model_copy(deep=True) for item in canonical)
            for entry in replay:
                self._llm.append_replay_entry(self._replay, _json_safe(entry))

    def append_query(self, query: str) -> None:
        self._extend(
            self._llm.query_to_canonical(str(query)),
            self._llm.query_to_replay(str(query)),
        )

    def append_response(self, response: Any, *, include_reasoning: bool = False) -> None:
        self._extend(
            self._llm.response_to_canonical(response, include_reasoning=include_reasoning),
            self._llm.response_to_replay(response, include_reasoning=include_reasoning),
        )

    def append_assistant(
        self,
        *,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
    ) -> None:
        self._extend(
            self._llm.assistant_message_to_canonical(
                content=content,
                tool_calls=tool_calls,
                thinking=thinking,
            ),
            self._llm.assistant_message_to_replay(
                content=content,
                tool_calls=tool_calls,
                thinking=thinking,
            ),
        )

    def append_tool_result(self, content: str, tool_id: str, tool_name: str) -> None:
        self._extend(
            self._llm.tool_result_to_canonical(content, tool_id, tool_name),
            self._llm.tool_result_to_replay(content, tool_id, tool_name),
        )

    def add(self, message: Any) -> None:
        canonical = self._llm.history_entry_to_canonical(message)
        if not canonical:
            value = coerce_canonical_message(message)
            canonical = [value] if value is not None else []
        if not canonical:
            raise TypeError(f"Unsupported history message: {type(message).__name__}")
        replay = self._llm.canonical_to_replay_history(canonical)
        self._extend(canonical, replay)

    def add_many(self, messages: Iterable[Any]) -> None:
        for message in messages:
            self.add(message)

    def replace(self, messages: Iterable[Any]) -> None:
        normalized: list[CanonicalMessage] = []
        for message in messages:
            value = coerce_canonical_message(message)
            if value is not None:
                normalized.append(value)
                continue
            normalized.extend(self._llm.history_entry_to_canonical(message))
        with self._lock:
            self._canonical = [item.model_copy(deep=True) for item in normalized]
        self.rebuild_replay()

    def remove_by_metadata(self, key: str, value: Any) -> bool:
        with self._lock:
            retained = [
                message for message in self._canonical
                if message.metadata.get(key) != value
            ]
            changed = len(retained) != len(self._canonical)
            if changed:
                self._canonical = retained
        if changed:
            self.rebuild_replay()
        return changed

    def contains_metadata(self, key: str, value: Any) -> bool:
        with self._lock:
            return any(message.metadata.get(key) == value for message in self._canonical)

    def rebuild_replay(self) -> list[Any]:
        with self._lock:
            self._replay = self._llm.canonical_to_replay_history(self._canonical)
            self._provider_name = self._llm.provider_name
            return [_json_safe(item) for item in self._replay]

    def change_llm(self, llm: EasyLLM) -> None:
        if not isinstance(llm, EasyLLM):
            raise TypeError(f"llm must be EasyLLM, got {type(llm).__name__}")
        self._llm = llm
        self.rebuild_replay()

    def clear(self) -> None:
        with self._lock:
            self._canonical.clear()
            self._replay.clear()
            self._provider_name = self._llm.provider_name

    def export_state(self) -> dict[str, Any]:
        return {
            "providerName": self.provider_name,
            "canonical": [item.to_dict() for item in self.canonical],
        }

    def restore_state(self, state: dict[str, Any] | None) -> None:
        payload = dict(state or {})
        self.replace(list(payload.get("canonical") or []))

    def __len__(self) -> int:
        with self._lock:
            return len(self._canonical)


__all__ = ["ConversationHistory"]
