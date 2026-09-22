"""Public planning contract; tool policy belongs to implementations."""

from abc import ABC, abstractmethod
from typing import Sequence

from runtime.agent import AgentRuntime
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from prompt.base import PromptBuildContext
from prompt import PromptBlock
from .models import ExecutionMode


class BasePlanMode(ABC):
    def get_prompt_blocks(self, context: PromptBuildContext) -> Sequence[PromptBlock]:
        return ()

    def get_tools(self) -> Sequence[Tool]:
        """Return tools after binding; the Agent handles registration."""
        return ()

    def create_child(self) -> "BasePlanMode | None":
        """Optional independent child plan; never share parent planning state."""
        return None

    @property
    @abstractmethod
    def mode(self) -> ExecutionMode:
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        return self.mode == ExecutionMode.PLAN

    @abstractmethod
    def bind(self, *, runtime: AgentRuntime, tool_registry: ToolRegistry | None) -> "BasePlanMode":
        raise NotImplementedError

    @abstractmethod
    def enter(self, *, allowed_actions: list[str] | None = None) -> None:
        raise NotImplementedError

    def request_exit(self, *, allowed_actions: list[str] | None = None) -> None:
        """Request review without changing mode; override to record plan-specific state."""
        if not self.is_active:
            raise RuntimeError("Plan mode is not active")

    @abstractmethod
    def exit(self) -> None:
        raise NotImplementedError
