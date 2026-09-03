"""The maintained general-purpose EasyAgent implementation."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from agent.components.executor import (
    AgentExecutionServices,
    BaseAgentExecutor,
    DefaultAgentExecutor,
)
from agent.components.tool_interrupt_controller import InMemoryToolInterruptController
from core.agent import BaseAgent
from core.Config import Config
from core.llm import EasyLLM
from runtime import AgentStreamEvent


class BasicAgent(BaseAgent):
    """Composable Agent with light defaults and optional heavy capabilities."""

    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: str | None = None,
        description: str | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            description=description,
            config=config,
        )
        self.executor = DefaultAgentExecutor()
        self.interrupt_controller = InMemoryToolInterruptController()

    def _services(self) -> AgentExecutionServices:
        if self._closed:
            raise RuntimeError("Agent is closed and cannot accept new invocations")
        executor = self.executor
        if not isinstance(executor, BaseAgentExecutor):
            raise RuntimeError("BasicAgent requires an installed BaseAgentExecutor")
        mailbox_sync = lambda: None
        if self.multi_agent is not None:
            mailbox_sync = lambda: self.multi_agent.sync_mailbox(
                execution_context=self.execution_context,
                metamessage_manager=self.metamessage_manager,
            )
        return AgentExecutionServices(
            agent_id=self.name,
            llm=self.llm,
            config=self.config,
            history=self.history_store,
            prompt_composer=self.prompt_composer,
            prompt_context_factory=self.build_prompt_context,
            execution_context=self.execution_context,
            event_bus=self.event_bus,
            metamessage_manager=self.metamessage_manager,
            permission_engine=self.permission_engine,
            permission_context=self.permission_context,
            hook_manager=self.hook_manager,
            tool_registry=self.tool_registry,
            context_manager=self.context_manager,
            reasoning=self.reasoning,
            interrupt_controller=self.interrupt_controller,
            stop_checker=self.stop_reason_if_requested,
            mailbox_sync=mailbox_sync,
        )

    def invoke(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        return self.executor.invoke(
            self._services(),
            query,
            max_iter=max_iter,
            temperature=self.config.temperature if temperature is None else temperature,
            **kwargs,
        )

    async def ainvoke(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        return await self.executor.ainvoke(
            self._services(),
            query,
            max_iter=max_iter,
            temperature=self.config.temperature if temperature is None else temperature,
            **kwargs,
        )

    def stream(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Iterator[AgentStreamEvent]:
        return self.executor.stream(
            self._services(),
            query,
            max_iter=max_iter,
            temperature=self.config.temperature if temperature is None else temperature,
            **kwargs,
        )

    async def astream(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        async for event in self.executor.astream(
            self._services(),
            query,
            max_iter=max_iter,
            temperature=self.config.temperature if temperature is None else temperature,
            **kwargs,
        ):
            yield event


__all__ = ["BasicAgent"]
