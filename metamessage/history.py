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


class ContextMetaMessageHistoryPort:
    """Inject/remove messages using the installed Context's public interface."""

    def __init__(self, history):
        from context.base import BaseContextManager

        if not isinstance(history, BaseContextManager):
            raise TypeError("history must be a Context module")
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


__all__ = ["ContextMetaMessageHistoryPort", "MetaMessageHistoryPort"]
