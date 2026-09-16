"""Public execution contract and per-invocation module references."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, AsyncIterator
from core.Config import Config
from core.hooks import HookManager
from core.llm import EasyLLM
from core.permissions.base import BasePermissionEngine
from core.permissions.models import AuthorizationContext
from context.base import BaseContextManager
from prompt.base import BaseSystemPromptComposer, PromptBuildContext
from runtime import AgentStreamEvent, ExecutionContext, RuntimeEventBus
from runtime.interruptions import BaseToolInterruptController, InMemoryToolInterruptController
from metamessage import BaseMetaMessageManager
from Tool.ToolRegistry import ToolRegistry

class AgentInvocationPhase(str, Enum):
    CREATED = "created"
    STARTED = "started"
    COMPILING = "compiling"
    LLM = "llm"
    TOOL = "tool"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class AgentInvocationState:
    invocation_id: str
    query: str
    stream: bool
    phase: AgentInvocationPhase = AgentInvocationPhase.CREATED
    round_number: int = 0
    output: str = ""
    error: BaseException | None = None


@dataclass(slots=True)
class AgentExecutionServices:
    agent_id: str
    llm: EasyLLM
    config: Config
    context_manager: BaseContextManager
    prompt_composer: BaseSystemPromptComposer
    prompt_context_factory: Callable[[str], PromptBuildContext]
    execution_context: ExecutionContext
    event_bus: RuntimeEventBus
    metamessage_manager: BaseMetaMessageManager
    permission_engine: BasePermissionEngine | None
    authorization_context_factory: Callable[[], AuthorizationContext]
    hook_manager: HookManager | None
    tool_registry: ToolRegistry | None = None
    reasoning: dict[str, Any] | None = None
    interrupt_controller: BaseToolInterruptController = field(default_factory=InMemoryToolInterruptController)
    stop_checker: Callable[[], str | None] = lambda: None
    mailbox_sync: Callable[[], None] = lambda: None
    session_id: str = ""
    invocation_lock: Any = None
    owns_invocation_lock: bool = False


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

    def execute_tool_call(self, services, state, tool_call, index=0):
        """Execute one model-selected tool call inside the ReAct loop."""
        raise NotImplementedError("This executor does not support tool calls")

    async def aexecute_tool_call(self, services, state, tool_call, index=0):
        """Async counterpart; implementations should await native async modules."""
        import asyncio
        return await asyncio.to_thread(self.execute_tool_call, services, state, tool_call, index)
