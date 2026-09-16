"""Explicit assembly; importing this example performs no model or database I/O."""

from easyagent import BasicAgent
from easyagent.context import ContextManager
from easyagent.memory import BaseMemory, LayeredMemory
from easyagent.prompting import PromptBlock, SystemPromptComposer
from easyagent.session import InMemorySessionStore


class ApplicationMemory(BaseMemory):
    """Applications supply their own tools; no CRUD or storage shape is imposed."""

    def __init__(self, instructions, retrieval_tools=(), persistence_tools=()):
        self.instructions = instructions
        self.retrieval_tools = list(retrieval_tools)
        self.persistence_tools = list(persistence_tools)

    def get_prompt_blocks(self, context):
        return [PromptBlock("memory.application", self.instructions)]

    def get_retrieval_tools(self):
        return self.retrieval_tools

    def get_persistence_tools(self):
        return self.persistence_tools


def build_agent(llm, *, memory=None, store=None):
    agent = BasicAgent("application", llm)
    agent.with_prompt(SystemPromptComposer(
        blocks=[PromptBlock("product", "You are a helpful engineering assistant.")],
        include_defaults=False,
    ))
    agent.with_context(ContextManager(max_tokens=24000))
    agent.with_memory(memory if memory is not None else LayeredMemory())
    agent.with_session(store if store is not None else InMemorySessionStore())
    return agent


async def demonstrate(llm, another_llm):
    store = InMemorySessionStore()
    agent = build_agent(llm, store=store)
    answer = await agent.ainvoke("Explain your available memory tools.")
    session_id = await agent.asave_session()
    resumed = build_agent(another_llm, store=store)
    await resumed.arestore_session(session_id)
    return answer, resumed
