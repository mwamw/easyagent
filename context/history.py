"""Framework conversation history: canonical messages only."""

from __future__ import annotations

from threading import RLock
from typing import Iterable
from uuid import uuid4

from core.history import CanonicalMessage, coerce_canonical_message


class ConversationHistory:
    """Thread-safe canonical conversation state shared by Context implementations.

    Provider request formats, replay caches, token budgets and LLM conversion do
    not belong here. Context implementations perform those operations around
    this stable semantic history.
    """

    def __init__(self, messages: Iterable[CanonicalMessage] = ()):
        self._messages: list[CanonicalMessage] = []
        self._handles: dict[str, str] = {}
        self._lock = RLock()
        self._revision = 0
        if messages:
            self.replace(messages)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def messages(self) -> list[CanonicalMessage]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._messages]

    @property
    def canonical(self) -> list[CanonicalMessage]:
        """Compatibility name for the framework's canonical message view."""
        return self.messages

    def append(self, message: CanonicalMessage) -> str:
        value = coerce_canonical_message(message)
        if value is None:
            raise TypeError("ConversationHistory accepts CanonicalMessage values")
        handle = str(value.metadata.get("metaMessageInjectionId") or uuid4().hex)
        with self._lock:
            self._messages.append(value.model_copy(deep=True))
            self._handles[handle] = handle
            self._revision += 1
        return handle

    def extend(self, messages: Iterable[CanonicalMessage]) -> list[str]:
        return [self.append(message) for message in messages]

    def replace(self, messages: Iterable[CanonicalMessage]) -> None:
        normalized = []
        for message in messages:
            value = coerce_canonical_message(message)
            if value is None:
                raise TypeError("ConversationHistory accepts CanonicalMessage values")
            normalized.append(value)
        with self._lock:
            self._messages = [item.model_copy(deep=True) for item in normalized]
            self._handles = {
                str(item.metadata["metaMessageInjectionId"]): str(item.metadata["metaMessageInjectionId"])
                for item in self._messages
                if item.metadata.get("metaMessageInjectionId")
            }
            self._revision += 1

    def remove(self, handle: str) -> bool:
        return self.remove_by_metadata("metaMessageInjectionId", handle)

    def contains(self, handle: str) -> bool:
        return self.contains_metadata("metaMessageInjectionId", handle)

    def remove_by_metadata(self, key: str, value) -> bool:
        with self._lock:
            retained = [message for message in self._messages if message.metadata.get(key) != value]
            changed = len(retained) != len(self._messages)
            if changed:
                self._messages = retained
                self._handles = {
                    str(item.metadata["metaMessageInjectionId"]): str(item.metadata["metaMessageInjectionId"])
                    for item in self._messages
                    if item.metadata.get("metaMessageInjectionId")
                }
                self._revision += 1
            return changed

    def contains_metadata(self, key: str, value) -> bool:
        with self._lock:
            return any(message.metadata.get(key) == value for message in self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._handles.clear()
            self._revision += 1

    def export_state(self) -> dict:
        return {"canonical": [item.to_dict() for item in self.messages]}

    def validate_state(self, state) -> None:
        payload = dict(state or {})
        if not isinstance(payload.get("canonical"), list):
            raise ValueError("Conversation history state must contain canonical messages")
        for item in payload["canonical"]:
            CanonicalMessage.model_validate(item)

    def restore_state(self, state) -> None:
        self.validate_state(state)
        self.replace(CanonicalMessage.model_validate(item) for item in state["canonical"])

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)


__all__ = ["ConversationHistory"]
