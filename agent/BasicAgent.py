"""The maintained general-purpose EasyAgent implementation."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from executor import (
    AgentExecutionServices,
    BaseAgentExecutor,
    DefaultAgentExecutor,
)
from core.agent import BaseAgent
from core.Config import Config
from core.llm import EasyLLM
from runtime import AgentRuntime, AgentStreamEvent


class BasicAgent(BaseAgent):
    """Composable Agent with light defaults and optional heavy capabilities."""

    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: str | None = None,
        description: str | None = None,
        config: Config | None = None,
        *,
        runtime: AgentRuntime | None = None,
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            description=description,
            config=config,
            runtime=runtime,
        )
        self.executor = DefaultAgentExecutor()

    def _services(self) -> AgentExecutionServices:
        if self._sessions.busy:
            raise RuntimeError("Cannot invoke while a session operation is in progress")
        if self.runtime.closed:
            raise RuntimeError("Agent is closed and cannot accept new invocations")
        executor = self.executor
        if not isinstance(executor, BaseAgentExecutor):
            raise RuntimeError("BasicAgent requires an installed BaseAgentExecutor")
        before_context_build = lambda: None
        if self.multi_agent is not None:
            before_context_build = self.multi_agent.before_context_build
        return AgentExecutionServices(
            llm=self.llm,
            config=self.config,
            prompt_composer=self.prompt_composer,
            prompt_context_factory=self.build_prompt_context,
            runtime=self.runtime,
            permission_engine=self.permission_engine,
            authorization_context_factory=self.build_authorization_context,
            tool_registry=self.tool_registry,
            context_manager=self.context_manager,
            before_context_build=before_context_build,
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
