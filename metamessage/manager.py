"""Runtime-event-driven meta-message injection and cleanup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
import logging
from threading import RLock
from typing import Any, Callable, Iterable, Optional
from uuid import uuid4

from core.history import CanonicalBlock, CanonicalMessage

from .events import MetaMessageEvent
from .history import MetaMessageHistoryPort
from .models import MetaMessage, MetaMessageContext, MetaMessageInjection, MetaMessageLifecycle


logger = logging.getLogger(__name__)
MetaMessageFactory = Callable[[MetaMessageContext], Optional[MetaMessage | Iterable[MetaMessage]]]
MetaMessageContextProvider = Callable[[], MetaMessageContext]

_AGENT_STARTED = "agent.invoke.started"
_AGENT_TERMINAL = {
    "agent.invoke.completed",
    "agent.invoke.failed",
    "agent.invoke.interrupted",
}
_LLM_TERMINAL = {"llm.invoke.completed", "llm.invoke.failed"}


class BaseMetaMessageManager(ABC):
    """Extension boundary used by Agent modules and the request pipeline."""

    @abstractmethod
    def bind(
        self,
        *,
        history_port: MetaMessageHistoryPort,
        context_provider: MetaMessageContextProvider,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> list[MetaMessageInjection]:
        raise NotImplementedError

    @abstractmethod
    def register(self, message: MetaMessage) -> MetaMessage:
        raise NotImplementedError

    @abstractmethod
    def emit(self, message: MetaMessage) -> MetaMessage:
        raise NotImplementedError

    @abstractmethod
    def publish(
        self,
        event: str | MetaMessageEvent,
        payload: Optional[dict[str, Any]] = None,
    ) -> list[MetaMessage]:
        raise NotImplementedError


class MetaMessageManager(BaseMetaMessageManager):
    """Owns meta-message rules, safe history insertion, and scoped cleanup."""

    def __init__(
        self,
        *,
        history_port: Optional[MetaMessageHistoryPort] = None,
        context_provider: Optional[MetaMessageContextProvider] = None,
    ) -> None:
        self._history_port = history_port
        self._context_provider = context_provider
        self._lock = RLock()
        self._registered: "OrderedDict[str, MetaMessage]" = OrderedDict()
        self._pending: "OrderedDict[str, MetaMessage]" = OrderedDict()
        self._injections: "OrderedDict[str, MetaMessageInjection]" = OrderedDict()
        self._subscriptions: dict[str, list[MetaMessageFactory]] = defaultdict(list)
        self._condition_active: dict[str, bool] = {}
        self._seen_dedup_keys: set[str] = set()
        self._invoke_active = False
        self._query = ""

    @property
    def invocation_active(self) -> bool:
        return self._invoke_active

    def bind(
        self,
        *,
        history_port: MetaMessageHistoryPort,
        context_provider: MetaMessageContextProvider,
    ) -> None:
        with self._lock:
            self._history_port = history_port
            self._context_provider = context_provider

    def _context(
        self,
        *,
        event_name: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> MetaMessageContext:
        base = self._context_provider() if self._context_provider is not None else MetaMessageContext()
        return MetaMessageContext(
            event_name=event_name,
            payload={**dict(base.payload), **dict(payload or {})},
            query=self._query or base.query,
            execution_mode=base.execution_mode,
            permission_mode=base.permission_mode,
            current_task_id=base.current_task_id,
        )

    def register(self, message: MetaMessage) -> MetaMessage:
        """Register user-configured permanent content or a conditional rule."""

        if not isinstance(message, MetaMessage):
            raise TypeError("message must be a MetaMessage")
        if message.lifecycle not in {
            MetaMessageLifecycle.PERMANENT,
            MetaMessageLifecycle.CONDITIONAL,
        }:
            raise ValueError(
                "INVOCATION and REQUEST MetaMessages are runtime module messages; "
                "use emit() when the triggering event occurs"
            )
        with self._lock:
            self._registered[message.message_id] = message
            self._condition_active.setdefault(message.message_id, False)
            if message.lifecycle == MetaMessageLifecycle.PERMANENT:
                self._queue(message)
        return message

    def emit(self, message: MetaMessage) -> MetaMessage:
        """Queue a concrete module-generated message for the next safe checkpoint."""

        if not isinstance(message, MetaMessage):
            raise TypeError("message must be a MetaMessage")
        if message.lifecycle == MetaMessageLifecycle.CONDITIONAL:
            raise ValueError("CONDITIONAL MetaMessages must be registered, not emitted")
        with self._lock:
            self._queue(message)
        return message

    def _queue(self, message: MetaMessage) -> bool:
        if message.dedup_key and self._dedup_key_is_active(message.dedup_key):
            return False
        if message.message_id in self._pending:
            return False
        self._pending[message.message_id] = message
        return True

    def _dedup_key_is_active(self, dedup_key: str) -> bool:
        if dedup_key in self._seen_dedup_keys:
            return True
        if any(item.dedup_key == dedup_key for item in self._pending.values()):
            return True
        return any(item.message.dedup_key == dedup_key for item in self._injections.values())

    def unregister(self, name_or_id: str, *, remove_injected: bool = True) -> int:
        target = str(name_or_id or "").strip()
        if not target:
            return 0
        removed = 0
        with self._lock:
            definition_ids = [
                key for key, item in self._registered.items()
                if key == target or item.name == target
            ]
            for key in definition_ids:
                definition = self._registered.pop(key, None)
                if definition is not None and definition.dedup_key:
                    self._seen_dedup_keys.discard(definition.dedup_key)
                self._pending.pop(key, None)
                self._condition_active.pop(key, None)
                removed += 1
            pending_ids = [
                key for key, item in self._pending.items()
                if key == target or item.name == target
            ]
            for key in pending_ids:
                self._pending.pop(key, None)
                removed += 1
            if remove_injected:
                injection_ids = [
                    key for key, item in self._injections.items()
                    if key == target or item.message.message_id == target or item.message.name == target
                ]
                removed += self._remove_injections(injection_ids)
        return removed

    def subscribe(self, event_name: str, factory: MetaMessageFactory) -> None:
        name = str(event_name or "").strip()
        if not name:
            raise ValueError("event_name must be non-empty")
        if not callable(factory):
            raise TypeError("factory must be callable")
        with self._lock:
            self._subscriptions[name].append(factory)

    def unsubscribe(
        self,
        event_name: str,
        factory: Optional[MetaMessageFactory] = None,
    ) -> int:
        name = str(event_name or "").strip()
        if not name:
            return 0
        with self._lock:
            subscriptions = self._subscriptions.get(name)
            if not subscriptions:
                return 0
            if factory is None:
                removed = len(subscriptions)
                self._subscriptions.pop(name, None)
                return removed
            retained = [item for item in subscriptions if item is not factory]
            removed = len(subscriptions) - len(retained)
            if retained:
                self._subscriptions[name] = retained
            else:
                self._subscriptions.pop(name, None)
            return removed

    def publish(
        self,
        event: str | MetaMessageEvent,
        payload: Optional[dict[str, Any]] = None,
    ) -> list[MetaMessage]:
        """Consume runtime/module events and apply lifecycle transitions automatically."""

        resolved = event if isinstance(event, MetaMessageEvent) else MetaMessageEvent(str(event), dict(payload or {}))
        with self._lock:
            if resolved.name == _AGENT_STARTED:
                self._invoke_active = True
                self._query = str(resolved.payload.get("query") or "")
            factories = list(self._subscriptions.get(resolved.name, []))

        context = self._context(event_name=resolved.name, payload=resolved.payload)
        emitted: list[MetaMessage] = []
        for factory in factories:
            produced = factory(context)
            if produced is None:
                continue
            items = [produced] if isinstance(produced, MetaMessage) else list(produced)
            for message in items:
                self.emit(message)
                emitted.append(message)

        with self._lock:
            self._evaluate_conditions(context)
            if resolved.name in _LLM_TERMINAL:
                self._remove_injections_by_lifecycle({MetaMessageLifecycle.REQUEST})
            if resolved.name in _AGENT_TERMINAL:
                self._remove_pending_by_lifecycle(
                    {MetaMessageLifecycle.INVOCATION, MetaMessageLifecycle.REQUEST}
                )
                self._remove_injections_by_lifecycle(
                    {MetaMessageLifecycle.INVOCATION, MetaMessageLifecycle.REQUEST}
                )
                self._invoke_active = False
                self._query = ""
        return emitted

    def flush(self) -> list[MetaMessageInjection]:
        """Append pending messages at a protocol-safe request boundary."""

        with self._lock:
            if self._history_port is None:
                raise RuntimeError("MetaMessageManager is not bound to a history port")
            context = self._context(event_name="request.before")
            self._evaluate_conditions(context)
            pending = list(self._pending.values())
            self._pending.clear()

            injected: list[MetaMessageInjection] = []
            for message in pending:
                content = message.render(context)
                if not content:
                    continue
                injection_id = f"injection_{uuid4().hex}"
                metadata = {
                    **message.metadata,
                    "is_meta": True,
                    "metaMessageId": message.message_id,
                    "metaMessageName": message.name,
                    "metaMessageLifecycle": message.lifecycle.value,
                    "metaMessageInjectionId": injection_id,
                }
                history_handle = self._history_port.append(
                    CanonicalMessage(
                        role="user",
                        content=[CanonicalBlock(type="text", text=content)],
                        metadata=metadata,
                    )
                )
                injection = MetaMessageInjection(
                    injection_id=injection_id,
                    message=message,
                    history_handle=history_handle,
                )
                self._injections[injection_id] = injection
                if message.dedup_key and message.lifecycle == MetaMessageLifecycle.PERMANENT:
                    self._seen_dedup_keys.add(message.dedup_key)
                injected.append(injection)
            return injected

    def _evaluate_conditions(self, context: MetaMessageContext) -> None:
        for key, message in list(self._registered.items()):
            if message.lifecycle != MetaMessageLifecycle.CONDITIONAL:
                continue
            active = message.should_activate(context)
            was_active = self._condition_active.get(key, False)
            if active and not was_active:
                self._queue(message.clone(metadata={**message.metadata, "definitionId": key}))
            elif not active and was_active:
                injection_ids = [
                    injection_id
                    for injection_id, injection in self._injections.items()
                    if injection.message.metadata.get("definitionId") == key
                ]
                self._remove_injections(injection_ids)
                pending_ids = [
                    message_id
                    for message_id, pending in self._pending.items()
                    if pending.metadata.get("definitionId") == key
                ]
                for message_id in pending_ids:
                    self._pending.pop(message_id, None)
            self._condition_active[key] = active

    def _remove_pending_by_lifecycle(self, lifecycles: set[MetaMessageLifecycle]) -> int:
        message_ids = [
            message_id
            for message_id, message in self._pending.items()
            if message.lifecycle in lifecycles
        ]
        for message_id in message_ids:
            self._pending.pop(message_id, None)
        return len(message_ids)

    def _remove_injections_by_lifecycle(self, lifecycles: set[MetaMessageLifecycle]) -> int:
        injection_ids = [
            injection_id
            for injection_id, injection in self._injections.items()
            if injection.message.lifecycle in lifecycles
        ]
        return self._remove_injections(injection_ids)

    def _remove_injections(self, injection_ids: Iterable[str]) -> int:
        if self._history_port is None:
            return 0
        removed = 0
        for injection_id in list(injection_ids):
            injection = self._injections.pop(injection_id, None)
            if injection is None:
                continue
            if self._history_port.remove(injection.history_handle):
                removed += 1
        return removed

    def list_registered(self) -> list[MetaMessage]:
        with self._lock:
            return list(self._registered.values())

    def list_pending(self) -> list[MetaMessage]:
        with self._lock:
            return list(self._pending.values())

    def list_injections(self) -> list[MetaMessageInjection]:
        with self._lock:
            return list(self._injections.values())

    def reconcile_history(self) -> int:
        with self._lock:
            if self._history_port is None:
                return 0
            stale_ids = [
                injection_id
                for injection_id, injection in self._injections.items()
                if not self._history_port.contains(injection.history_handle)
            ]
            for injection_id in stale_ids:
                self._injections.pop(injection_id, None)
            return len(stale_ids)

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            serializable_registered: list[dict[str, Any]] = []
            for message in self._registered.values():
                try:
                    serializable_registered.append(message.to_dict())
                except TypeError:
                    continue
            serializable_pending: list[dict[str, Any]] = []
            for message in self._pending.values():
                if message.lifecycle != MetaMessageLifecycle.PERMANENT:
                    continue
                try:
                    serializable_pending.append(message.to_dict())
                except TypeError:
                    continue
            permanent_injections = [
                {
                    "injectionId": injection.injection_id,
                    "historyHandle": injection.history_handle,
                    "messageId": injection.message.message_id,
                    "name": injection.message.name,
                    "dedupKey": injection.message.dedup_key,
                    "metadata": dict(injection.message.metadata),
                    "injectedAt": injection.injected_at,
                }
                for injection in self._injections.values()
                if injection.message.lifecycle == MetaMessageLifecycle.PERMANENT
            ]
            return {
                "version": 2,
                "registered": serializable_registered,
                "pending": serializable_pending,
                "permanentInjections": permanent_injections,
                "seenDedupKeys": sorted(self._seen_dedup_keys),
            }

    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        data = dict(state or {})
        with self._lock:
            self._registered.clear()
            self._pending.clear()
            self._injections.clear()
            self._condition_active.clear()
            self._seen_dedup_keys = {
                str(item) for item in list(data.get("seenDedupKeys") or []) if item
            }
            for payload in list(data.get("registered") or []):
                message = MetaMessage.from_dict(payload)
                if message.lifecycle not in {
                    MetaMessageLifecycle.PERMANENT,
                    MetaMessageLifecycle.CONDITIONAL,
                }:
                    continue
                self._registered[message.message_id] = message
                self._condition_active[message.message_id] = False
            for payload in list(data.get("pending") or []):
                message = MetaMessage.from_dict(payload)
                self._pending[message.message_id] = message
            for payload in list(data.get("permanentInjections") or []):
                injection_id = str(payload.get("injectionId") or "").strip()
                history_handle = str(payload.get("historyHandle") or "").strip()
                if not injection_id or not history_handle:
                    continue
                message = MetaMessage(
                    name=str(payload.get("name") or "restored_metamessage"),
                    content="",
                    lifecycle=MetaMessageLifecycle.PERMANENT,
                    dedup_key=payload.get("dedupKey"),
                    metadata=dict(payload.get("metadata") or {}),
                    message_id=str(payload.get("messageId") or f"meta_{uuid4().hex}"),
                )
                self._injections[injection_id] = MetaMessageInjection(
                    injection_id=injection_id,
                    message=message,
                    history_handle=history_handle,
                    injected_at=float(payload.get("injectedAt") or 0.0),
                )


__all__ = [
    "BaseMetaMessageManager",
    "MetaMessageContextProvider",
    "MetaMessageFactory",
    "MetaMessageManager",
]
