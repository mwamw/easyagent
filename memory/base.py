"""Contract for a complete memory mechanism, not a memory record store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TYPE_CHECKING
from core.scope import RuntimeScope


if TYPE_CHECKING:
    from core.hooks import BaseHook
    from prompt import PromptBlock
    from prompt.base import PromptBuildContext
    from Tool.BaseTool import Tool
    from runtime.metamessage.models import MetaMessage


class MetaMessageEmitterPort(Protocol):
    """The only dynamic-message capability exposed to a Memory module."""

    def emit(self, message: "MetaMessage") -> "MetaMessage": ...
    def register(self, message: "MetaMessage") -> "MetaMessage": ...
    def subscribe(self, event_name: str, factory: Callable) -> None: ...


@dataclass(frozen=True, slots=True)
class MemoryServices:
    """Borrowed public services; memory decides whether to inject content."""

    metamessages: MetaMessageEmitterPort
    scope_provider: Callable[[], RuntimeScope]

    @property
    def scope(self) -> RuntimeScope:
        return self.scope_provider()


class BaseMemory(ABC):
    """Agent-facing contract for a complete, replaceable memory mechanism.

    Concrete memory systems decide their storage model and retrieval algorithm.
    The framework only requires them to explain their mechanism to the model,
    expose retrieval/persistence capabilities, and participate in lifecycle.
    """

    def bind(self, services: MemoryServices) -> None:
        if getattr(self, "_memory_services", None) is not None:
            raise RuntimeError("A memory instance may only be bound to one Agent")
        self._memory_services = services

    @property
    def services(self) -> MemoryServices:
        services = getattr(self, "_memory_services", None)
        if services is None:
            raise RuntimeError("Memory is not bound to an Agent")
        return services

    def unbind(self) -> None:
        self._memory_services = None

    @abstractmethod
    def get_prompt_blocks(
        self,
        context: "PromptBuildContext",
    ) -> Sequence["PromptBlock"]:
        """Describe when and how the model should use this memory system."""
        raise NotImplementedError

    @abstractmethod
    def get_retrieval_tools(self) -> Sequence["Tool"]:
        """Return tools that read, search, or locate memory."""
        raise NotImplementedError

    @abstractmethod
    def get_persistence_tools(self) -> Sequence["Tool"]:
        """Return tools that save, update, remove, or maintain memory."""
        raise NotImplementedError

    def get_hooks(self) -> Sequence["BaseHook"]:
        """Optional automatic recall/extraction hooks."""
        return ()

    def export_state(self) -> dict[str, object] | None:
        """Return session-owned state; external long-term stores need not be copied."""
        return None

    def validate_state(self, state: dict[str, object] | None) -> None:
        if state is not None:
            raise ValueError(f"{type(self).__name__} does not accept session state")

    def restore_state(self, state: dict[str, object] | None) -> None:
        self.validate_state(state)

    def reset_session_state(self) -> None:
        """Reset only state scoped to the current Agent session."""

    def close(self) -> None:
        """Release resources owned by this memory implementation."""

    async def aclose(self) -> None:
        self.close()
