from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent import BasicAgent
from agent.components.executor import AgentExecutionServices, BaseAgentExecutor
from agent.components.prompt_composer import BaseSystemPromptComposer, PromptBuildContext
from core.llm import EasyLLM
from runtime import AgentStreamEvent, AgentStreamEventType
from prompt import PromptBlock


class StubProvider:
    def close(self):
        return None


class StubLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.temperature = 0.1
        self.timeout = 60
        self.kwargs = {}
        self._provider = StubProvider()
        self.client = None


class CustomPrompt(BaseSystemPromptComposer):
    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return [PromptBlock("custom", f"Agent={context.agent_name}; query={context.query}")]


class CustomExecutor(BaseAgentExecutor):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def invoke(self, services: AgentExecutionServices, query: str, *, max_iter: int, temperature: float, **kwargs):
        self.calls.append(("invoke", query))
        return f"sync:{services.agent_id}:{query}"

    async def ainvoke(self, services: AgentExecutionServices, query: str, *, max_iter: int, temperature: float, **kwargs):
        self.calls.append(("ainvoke", query))
        return f"async:{services.agent_id}:{query}"

    def stream(self, services: AgentExecutionServices, query: str, *, max_iter: int, temperature: float, **kwargs):
        self.calls.append(("stream", query))
        yield AgentStreamEvent(
            type=AgentStreamEventType.FINAL,
            invocation_id="custom-sync",
            sequence=1,
            content=f"stream:{query}",
        )

    async def astream(self, services: AgentExecutionServices, query: str, *, max_iter: int, temperature: float, **kwargs):
        self.calls.append(("astream", query))
        yield AgentStreamEvent(
            type=AgentStreamEventType.FINAL,
            invocation_id="custom-async",
            sequence=1,
            content=f"astream:{query}",
        )


def test_prompt_composer_is_replaced_through_with_prompt():
    agent = BasicAgent("prompt-agent", StubLLM()).with_prompt(CustomPrompt())

    assert agent.get_enhanced_prompt("inspect") == "Agent=prompt-agent; query=inspect"


def test_executor_is_replaced_through_with_executor_for_all_public_paths():
    executor = CustomExecutor()
    agent = BasicAgent("executor-agent", StubLLM()).with_executor(executor)

    assert agent.invoke("one") == "sync:executor-agent:one"
    assert asyncio.run(agent.ainvoke("two")) == "async:executor-agent:two"
    assert list(agent.stream("three"))[-1].content == "stream:three"

    async def collect():
        return [event async for event in agent.astream("four")]

    assert asyncio.run(collect())[-1].content == "astream:four"
    assert executor.calls == [
        ("invoke", "one"),
        ("ainvoke", "two"),
        ("stream", "three"),
        ("astream", "four"),
    ]


def test_custom_executor_receives_the_current_composed_services():
    executor = CustomExecutor()
    agent = BasicAgent("services-agent", StubLLM()).with_executor(executor)
    services = agent._services()

    assert isinstance(services, AgentExecutionServices)
    assert services.agent_id == "services-agent"
    assert services.history is agent.history_store
    assert services.prompt_composer is agent.prompt_composer
    assert services.event_bus is agent.event_bus
