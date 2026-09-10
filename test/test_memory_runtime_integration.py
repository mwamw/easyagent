import asyncio
import subprocess
import sys

import pytest

from easyagent import BasicAgent
from easyagent.context import ContextManager, ContextBudgetExceeded
from easyagent.hooks import BaseHook, HookDecision, HookManager
from easyagent.memory import BaseMemory, LayeredMemory
from easyagent.metamessages import MetaMessage, MetaMessageLifecycle
from easyagent.prompting import PromptBlock, SystemPromptComposer
from test_session_persistence import SessionLLM


def test_default_layered_memory_is_model_directed_and_lightweight():
    memory = LayeredMemory()
    memory.add_memory("private memory content", "working", 1.0)
    agent = BasicAgent("test", SessionLLM()).with_memory(memory)
    assert list(memory.memory_types) == ["working"]
    assert agent.context_manager is None
    assert agent.tool_registry.has_tool("search_memory_tool")
    assert "private memory content" not in agent.get_enhanced_prompt()
    assert memory.get_hooks() == []
    assert agent.invoke("hello") == "session-response"
    assert "private memory content" not in repr(agent.history)


def test_simple_memory_import_does_not_load_optional_clients():
    code = "from easyagent.memory import BaseMemory, LayeredMemory; m=LayeredMemory(); import sys; assert not {'sentence_transformers', 'qdrant_client', 'neo4j', 'chromadb'}.intersection(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_memory_tools_use_live_session_scope():
    agent = BasicAgent("test", SessionLLM()).with_memory(LayeredMemory())
    tool = agent.tool_registry.get_tool("add_memory_tool")
    agent.session_id = "first"
    assert tool.get_current_session_id() == "first"
    agent.session_id = "second"
    assert tool.get_current_session_id() == "second"


class ReadOnlyMemory(BaseMemory):
    def get_prompt_blocks(self, context):
        return [PromptBlock("memory.custom", "Custom read-only memory instructions")]


def test_custom_memory_needs_no_crud_and_owns_its_prompt():
    agent = BasicAgent("test", SessionLLM()).with_memory(ReadOnlyMemory())
    agent.with_prompt(SystemPromptComposer(include_defaults=False))
    assert "Custom read-only" in agent.get_enhanced_prompt()
    assert agent.tool_registry is None
    agent.with_context(ContextManager(max_tokens=10000))
    assert not hasattr(agent.context_manager, "add_source")


def test_replacing_memory_rebinds_tools_and_preserves_other_registrations():
    first, second = LayeredMemory(), LayeredMemory()
    agent = BasicAgent("test", SessionLLM()).with_memory(first)
    with pytest.raises(ValueError):
        agent.with_memory(second)
    agent.with_memory(second, replace=True)
    assert agent.tool_registry.get_tool("add_memory_tool").memory_manage is second
    with pytest.raises(RuntimeError):
        _ = first.services


class RecallHook(BaseHook):
    def __init__(self, memory):
        self.memory = memory

    async def abefore_context(self, payload):
        await asyncio.sleep(0)
        self.memory.services.metamessages.emit(MetaMessage(
            name="recall", content="async recalled context", lifecycle=MetaMessageLifecycle.REQUEST,
        ))


class AsyncMemory(ReadOnlyMemory):
    def get_hooks(self):
        return [RecallHook(self)]


@pytest.mark.asyncio
async def test_async_memory_hook_is_awaited_and_temporary_content_expires():
    agent = BasicAgent("test", SessionLLM()).with_memory(AsyncMemory())
    seen = []
    agent.event_bus.subscribe(lambda event: seen.append(event.data))
    assert await agent.ainvoke("hello") == "session-response"
    assert "async recalled context" in repr(seen)
    assert "async recalled context" not in repr(agent.history)
    with pytest.raises(RuntimeError, match="requires async"):
        agent.invoke("sync not supported")


def test_post_assembly_hook_cannot_bypass_total_budget():
    class Extra(BaseHook):
        def before_llm_request(self, payload):
            payload["request_input"].append_dynamic_tail_text("large content " * 1000)
    agent = BasicAgent("test", SessionLLM()).with_context(ContextManager(max_tokens=2000))
    agent.with_hooks(HookManager([Extra()]))
    with pytest.raises(ContextBudgetExceeded):
        agent.invoke("hello")


@pytest.mark.asyncio
async def test_cancel_during_memory_preparation_cleans_lifecycle_and_releases_agent():
    entered = asyncio.Event()
    pending = asyncio.Event()
    class WaitingMemory(ReadOnlyMemory):
        def get_hooks(self):
            memory = self
            class WaitingHook(BaseHook):
                async def abefore_context(self, payload):
                    memory.services.metamessages.emit(MetaMessage(name="temporary", content="pending memory", lifecycle=MetaMessageLifecycle.INVOCATION))
                    entered.set()
                    await pending.wait()
            return [WaitingHook()]
    agent = BasicAgent("test", SessionLLM()).with_memory(WaitingMemory())
    task = asyncio.create_task(agent.ainvoke("first"))
    await entered.wait()
    with pytest.raises(RuntimeError, match="one active invocation"):
        await agent.ainvoke("second must not enter history")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not agent.metamessage_manager.invocation_active
    assert not agent.metamessage_manager.list_pending()
    assert "second must not enter history" not in repr(agent.history)
    agent.with_memory(ReadOnlyMemory(), replace=True)
    assert await agent.ainvoke("third") == "session-response"


def test_temporary_metamessages_are_not_absorbed_by_compaction():
    from easyagent.context import BaseHistoryCompactor
    from core.history import CanonicalMessage, CanonicalBlock
    captured = []
    class Capture(BaseHistoryCompactor):
        recent_turns = 1
        def compact(self, history):
            captured.extend(history)
            return [CanonicalMessage(role="user", content=[CanonicalBlock(type="text", text="summary")])]
    agent = BasicAgent("test", SessionLLM())
    agent.add_user_message("old question")
    agent.emit_metamessage(MetaMessage(name="temporary", content="temporary private result", lifecycle=MetaMessageLifecycle.REQUEST))
    agent.metamessage_manager.flush()
    agent.add_assistant_message("old answer")
    agent.add_user_message("new question")
    manager = ContextManager(max_tokens=10000)
    manager.set_history_compactor(Capture())
    result = manager.compact_persistent_history(agent.history, agent.replay_history, provider_name="openai", token_counter=manager.counter, max_tokens=10000, force=True)
    assert "temporary private result" not in repr(captured)
    agent.history = result.canonical_history
    assert "temporary private result" in repr(agent.history)
    agent.metamessage_manager.publish("llm.invoke.completed")
    assert "temporary private result" not in repr(agent.history)


@pytest.mark.asyncio
async def test_owned_async_memory_is_closed_before_its_services_are_unbound():
    class OwnedMemory(ReadOnlyMemory):
        closed = False
        async def aclose(self):
            assert self.services.scope.session_id
            self.closed = True
    memory = OwnedMemory()
    agent = BasicAgent("test", SessionLLM()).with_memory(memory, owned=True)
    report = await agent.aclose()
    assert report["status"] == "closed"
    assert memory.closed
    with pytest.raises(RuntimeError, match="not bound"):
        _ = memory.services
