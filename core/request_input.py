from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from .history import _json_safe
from .replay_converter import append_replay_entry


def _clone_payload(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            payload = value.to_dict()
            if isinstance(payload, dict):
                return _json_safe(payload)
        except Exception:
            pass
    if isinstance(value, dict):
        return _json_safe(value)
    return value


@dataclass
class ReplayRequestInput:
    """Current-provider request buffer.

    `replay_history` stores request-ready history entries for the active provider.
    `system_prompt` remains separate so providers do not need to scan history to
    extract it.
    """

    provider_name: Optional[str]
    replay_history: list[Any] = field(default_factory=list)
    system_prompt: Optional[str] = None
    message_converter: Optional[Callable[[list[Any]], list[Any]]] = None
    request_ready_checker: Optional[Callable[[Any], bool]] = None
    visible_messages: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.visible_messages:
            self.visible_messages = self._build_visible_messages()

    def _build_visible_messages(self) -> list[Any]:
        visible: list[Any] = []
        if self.system_prompt:
            visible.append({"role": "system", "content": self.system_prompt})
        visible.extend(_clone_payload(item) for item in self.replay_history)
        return visible

    def _is_request_ready(self, message: Any) -> bool:
        checker = self.request_ready_checker
        return bool(checker(message)) if checker is not None else isinstance(message, dict)

    def _convert_item(self, message: Any) -> list[Any]:
        if isinstance(message, ReplayRequestInput):
            return [_clone_payload(item) for item in message.replay_history]
        if isinstance(message, list):
            converted: list[Any] = []
            for item in message:
                converted.extend(self._convert_item(item))
            return converted
        if self._is_request_ready(message):
            return [_clone_payload(message)]
        if self.message_converter is None:
            return [_clone_payload(message)]
        return [_clone_payload(item) for item in self.message_converter([message])]

    def append(self, message: Any) -> None:
        self.extend([message])

    def extend(self, messages: Iterable[Any]) -> None:
        for message in messages:
            for entry in self._convert_item(message):
                entry_payload = _clone_payload(entry)
                append_replay_entry(self.replay_history, entry_payload, self.provider_name)
                append_replay_entry(self.visible_messages, _clone_payload(entry_payload), self.provider_name)

    def clone(self) -> "ReplayRequestInput":
        return ReplayRequestInput(
            provider_name=self.provider_name,
            replay_history=[_clone_payload(item) for item in self.replay_history],
            system_prompt=self.system_prompt,
            message_converter=self.message_converter,
            request_ready_checker=self.request_ready_checker,
        )

    def as_visible_messages(self) -> list[Any]:
        return [_clone_payload(item) for item in self.visible_messages]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.visible_messages)

    def __len__(self) -> int:
        return len(self.visible_messages)

    def __getitem__(self, index: int) -> Any:
        return self.visible_messages[index]

    def __bool__(self) -> bool:
        return bool(self.visible_messages)
