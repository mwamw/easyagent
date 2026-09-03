"""History boundary used by the provider-neutral meta-message manager."""

from __future__ import annotations

from typing import Protocol

from core.history import CanonicalMessage


class MetaMessageHistoryPort(Protocol):
    def append(self, message: CanonicalMessage) -> str:
        """Append a meta user message and return a removable history handle."""

    def remove(self, history_handle: str) -> bool:
        """Remove a previously injected message from canonical and replay history."""

    def contains(self, history_handle: str) -> bool:
        """Return whether the history still contains this injected message."""


class AgentMetaMessageHistoryPort:
    """Adapter over the framework's fixed ConversationHistory boundary."""

    def __init__(self, history):
        from agent.components.conversation_history import ConversationHistory

        if not isinstance(history, ConversationHistory):
            raise TypeError("history must be ConversationHistory")
        self._history = history

    def append(self, message: CanonicalMessage) -> str:
        handle = str((message.metadata or {}).get("metaMessageInjectionId") or "").strip()
        if not handle:
            raise ValueError("Canonical meta message is missing metaMessageInjectionId")
        self._history.add(message)
        return handle

    def remove(self, history_handle: str) -> bool:
        return self._history.remove_by_metadata(
            "metaMessageInjectionId",
            history_handle,
        )

    def contains(self, history_handle: str) -> bool:
        return self._history.contains_metadata(
            "metaMessageInjectionId",
            history_handle,
        )


__all__ = ["AgentMetaMessageHistoryPort", "MetaMessageHistoryPort"]
