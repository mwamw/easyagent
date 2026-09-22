"""Public execution contract and per-invocation module references."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, AsyncIterator
from core.Config import Config
from core.llm import EasyLLM
from permissions.base import BasePermissionEngine
from permissions.models import AuthorizationContext
from context.base import BaseContextManager
from prompt.base import BaseSystemPromptComposer, PromptBuildContext
from runtime import AgentStreamEvent, AgentRuntime
from Tool.ToolRegistry import ToolRegistry
from Tool.BaseTool import ToolResult

from runtime.invocation import AgentInvocationPhase, AgentInvocationState


@dataclass(slots=True)
class AgentExecutionServices:
    llm: EasyLLM
    config: Config
    context_manager: BaseContextManager
    prompt_composer: BaseSystemPromptComposer
    prompt_context_factory: Callable[[str], PromptBuildContext]
    runtime: AgentRuntime
    permission_engine: BasePermissionEngine | None
    authorization_context_factory: Callable[[], AuthorizationContext]
    tool_registry: ToolRegistry | None = None
    before_context_build: Callable[[], None] = lambda: None


class BaseAgentExecutor(ABC):
    @abstractmethod
    def invoke(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[AgentStreamEvent]:
        raise NotImplementedError

    @abstractmethod
    def astream(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        raise NotImplementedError

    def execute_tool_call(self, services: AgentExecutionServices, state: AgentInvocationState,
                          tool_call: Any, index: int = 0) -> tuple[str, dict[str, Any], str, ToolResult]:
        """Execute one model-selected tool call inside the ReAct loop."""
        raise NotImplementedError("This executor does not support tool calls")

    async def aexecute_tool_call(self, services: AgentExecutionServices, state: AgentInvocationState,
                                 tool_call: Any, index: int = 0) -> tuple[str, dict[str, Any], str, ToolResult]:
        """Async counterpart; implementations should await native async modules."""
        import asyncio
        return await asyncio.to_thread(self.execute_tool_call, services, state, tool_call, index)
