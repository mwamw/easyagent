"""Public observability contract and child event subscription semantics."""
from abc import ABC, abstractmethod
from runtime.events import RuntimeEvent, RuntimeEventBus
from .models import AgentInvoke


class BaseObservabilityManager(ABC):
    @property
    def active_invoke_id(self) -> str | None:
        """Optional trace correlation for child invocations."""
        return None

    def create_child(self, *, parent_invoke_id: str | None = None) -> "BaseObservabilityManager":
        """Forward child events through the public sink without rebinding the parent.

        Implementations with per-agent aggregation can override this to create a
        fresh aggregator sharing only their external store.
        """
        return _ChildObservabilityManager(self)

    @abstractmethod
    def bind(self, *, agent_id: str, event_bus: RuntimeEventBus) -> "BaseObservabilityManager":
        raise NotImplementedError

    @abstractmethod
    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[AgentInvoke]:
        """Return completed AgentInvoke records in storage order."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class _ChildObservabilityManager(BaseObservabilityManager):
    def __init__(self, sink: BaseObservabilityManager) -> None:
        self._sink = sink
        self._bus: RuntimeEventBus | None = None
        self._subscription: str | None = None

    def bind(self, *, agent_id: str, event_bus: RuntimeEventBus) -> BaseObservabilityManager:
        if self._bus is not None and self._bus is not event_bus:
            raise RuntimeError("Child observability is already bound")
        self._bus = event_bus
        if self._subscription is None:
            self._subscription = event_bus.subscribe(self.handle_runtime_event)
        return self

    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        self._sink.handle_runtime_event(event)

    def list(self) -> list[AgentInvoke]:
        return self._sink.list()

    def close(self) -> None:
        if self._bus is not None and self._subscription is not None:
            self._bus.unsubscribe(self._subscription)
            self._subscription = None
