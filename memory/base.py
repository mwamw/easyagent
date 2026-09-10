"""Contract for a complete memory mechanism, not a memory record store."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING
from core.scope import RuntimeScope


if TYPE_CHECKING:
    from core.hooks import BaseHook
    from prompt import PromptBlock
    from Tool.BaseTool import Tool
    from metamessage.emitter import MetaMessageEmitter


@dataclass(frozen=True, slots=True)
class MemoryServices:
    """Borrowed public services; memory decides whether to inject content."""

    metamessages: MetaMessageEmitter
    scope_provider: Callable[[], RuntimeScope]

    @property
    def scope(self) -> RuntimeScope:
        return self.scope_provider()


class BaseMemory(ABC):
    """Optional instructions/tools/hooks; no required CRUD, taxonomy, or backend."""

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

    def get_prompt_blocks(self, context: Any) -> list[PromptBlock]:
        return []

    def get_tools(self) -> list[Tool]:
        return []

    def get_hooks(self) -> list[BaseHook]:
        return []

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        self.close()
