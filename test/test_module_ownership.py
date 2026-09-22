"""Public module replacement tests: no default Context/Prompt internals required."""

import ast
import asyncio
from pathlib import Path

import pytest

from agent import BasicAgent
from context.base import BaseContextManager
from context.default.manager import ContextManager
from context.models import ContextCompactionResult, ContextUsage
from core.request_compiler import compile_prompt_blocks
from core.request_input import ReplayRequestInput
from easyagent.executors import BaseAgentExecutor
from easyagent.memory import BaseMemory
from easyagent.session import InMemorySessionStore
from runtime.metamessage import MetaMessage, MetaMessageLifecycle
from prompt import PromptBlock
from prompt.base import BaseSystemPromptComposer
from prompt.composer import SystemPromptComposer
from Tool import ToolResult

from test_permission_boundaries import OfflineLLM, CountingWriteTool, run_path
from test_session_persistence import SessionLLM
from test_agent_interfaces import CustomExecutor


class IndependentContext(BaseContextManager):
    """A minimal implementation using framework history but no default algorithms."""
    def __init__(self):
        super().__init__()
        self.prepared = []
        self.feedback = []

    def bind(self, llm, *, metamessages=None, owner=None):
        return super().bind(llm, metamessages=metamessages, owner=owner)

    def build_input(self, prompt_blocks, *, tools, options, query,
                    cache_policy, replay_history=None):
        compiled = compile_prompt_blocks(prompt_blocks)
        request = ReplayRequestInput(
            provider_name=self._llm.provider_name,
            replay_history=self.replay if replay_history is None else replay_history,
            system_prompt_blocks=compiled.system_prompt_blocks,
            system_reminder_blocks=compiled.system_reminder_blocks,
        )
        request.apply_runtime_layers()
        self.prepared.append(request)
        return request

    def calculate_usage(self, request_input, *, tools, options):
        return ContextUsage(estimated_tokens=0, max_tokens=None,
                            local_estimated_tokens=0, source="custom")

    def should_compact(self, usage):
        return False

    def compact_context(self, request_input, usage, **kwargs):
        return ContextCompactionResult(
            canonical_history=self.canonical, replay_history=self.replay,
            was_compacted=False, compaction_possible=False,
            tokens_before=0, tokens_after=0, budget=0,
        )

    def validate_input(self, request_input, *, tools, options, history_epoch=None):
        return None

    def append_tool_context(self, tool_name, value):
        return None

    def observe_model_result(self, usage, response_replay, *, use_output):
        self.feedback.append((usage, use_output))

@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
def test_custom_context_owns_history_and_builds_every_model_input(path):
    context, tool = IndependentContext(), CountingWriteTool()
    agent = BasicAgent("independent", OfflineLLM()).with_context(context).add_tool(tool)
    try:
        for name in ("builder", "counter", "budget", "history_store"):
            assert not hasattr(context, name)
        assert agent.permission_engine is None
        assert agent.hook_manager is None
        run_path(agent, path)
        assert tool.calls == 1
        assert len(context.prepared) == len(context.feedback) == 2
        assert [message.role for message in context.canonical] == ["user", "assistant", "tool", "assistant"]
        assert agent.history == context.canonical
        assert not hasattr(agent, "history_store")
        assert not hasattr(agent, "_context_usage")
    finally:
        agent.close()


def test_default_context_pipeline_dispatches_through_public_stage_contracts():
    class TracedContext(ContextManager):
        def __init__(self):
            super().__init__(max_tokens=10000)
            self.stages = []

        def process_meta_messages(self):
            self.stages.append("dynamic")
            return super().process_meta_messages()

        def build_input(self, *args, **kwargs):
            self.stages.append("build")
            return super().build_input(*args, **kwargs)

        def calculate_usage(self, *args, **kwargs):
            self.stages.append("usage")
            return super().calculate_usage(*args, **kwargs)

        def should_compact(self, usage):
            self.stages.append("decide")
            return super().should_compact(usage)

        def validate_input(self, *args, **kwargs):
            self.stages.append("validate")
            return super().validate_input(*args, **kwargs)

        def observe_prepared_input(self, *args, **kwargs):
            self.stages.append("prepared")
            return super().observe_prepared_input(*args, **kwargs)

        def observe_model_result(self, *args, **kwargs):
            self.stages.append("result")
            return super().observe_model_result(*args, **kwargs)

    context = TracedContext()
    agent = BasicAgent("staged", OfflineLLM()).with_context(context).add_tool(CountingWriteTool())
    try:
        agent.invoke("exercise every request stage")
        assert context.stages == [
            "dynamic", "build", "usage", "decide", "validate", "prepared", "result",
            "dynamic", "build", "usage", "decide", "validate", "prepared", "result",
        ]
    finally:
        agent.close()


def test_async_context_pipeline_uses_native_async_stage_contracts():
    class NativeAsyncContext(ContextManager):
        def __init__(self):
            super().__init__(max_tokens=10000)
            self.async_stages = []

        def build_input(self, *args, **kwargs):
            pytest.fail("sync build_input used")

        async def abuild_input(self, *args, **kwargs):
            self.async_stages.append("build")
            return ContextManager.build_input(self, *args, **kwargs)

        def calculate_usage(self, *args, **kwargs):
            pytest.fail("sync calculate_usage used")

        async def acalculate_usage(self, *args, **kwargs):
            self.async_stages.append("usage")
            return ContextManager.calculate_usage(self, *args, **kwargs)

        def validate_input(self, *args, **kwargs):
            pytest.fail("sync validate_input used")

        async def avalidate_input(self, *args, **kwargs):
            self.async_stages.append("validate")
            return ContextManager.validate_input(self, *args, **kwargs)

        def observe_prepared_input(self, *args, **kwargs):
            pytest.fail("sync observe_prepared_input used")

        async def aobserve_prepared_input(self, *args, **kwargs):
            self.async_stages.append("prepared")
            return ContextManager.observe_prepared_input(self, *args, **kwargs)

    context = NativeAsyncContext()
    agent = BasicAgent("async-staged", OfflineLLM()).with_context(context).add_tool(CountingWriteTool())
    try:
        asyncio.run(agent.ainvoke("use native async Context stages"))
        assert context.async_stages == [
            "build", "usage", "validate", "prepared",
            "build", "usage", "validate", "prepared",
        ]
    finally:
        agent.close()


@pytest.mark.parametrize(
    ("context_type", "method"),
    [
        (type("BadBuildContext", (IndependentContext,), {
            "build_input": lambda self, *args, **kwargs: {},
        }), "build_input"),
        (type("BadUsageContext", (IndependentContext,), {
            "calculate_usage": lambda self, *args, **kwargs: {},
        }), "calculate_usage"),
        (type("BadDecisionContext", (IndependentContext,), {
            "should_compact": lambda self, usage: "no",
        }), "should_compact"),
        (type("BadCompactionContext", (IndependentContext,), {
            "calculate_usage": lambda self, *args, **kwargs: ContextUsage(
                estimated_tokens=2, max_tokens=1,
            ),
            "should_compact": lambda self, usage: True,
            "compact_context": lambda self, *args, **kwargs: {},
        }), "compact_context"),
    ],
)
def test_executor_rejects_invalid_context_contract_results(context_type, method):
    agent = BasicAgent("invalid-context", OfflineLLM()).with_context(context_type())
    try:
        with pytest.raises(TypeError, match=rf"Context\.{method}\(\) must return"):
            agent.invoke("validate Context contract")
    finally:
        agent.close()


def test_async_executor_rejects_invalid_context_contract_result():
    class BadAsyncBuildContext(IndependentContext):
        async def abuild_input(self, *args, **kwargs):
            return {}

    agent = BasicAgent("invalid-async-context", OfflineLLM()).with_context(BadAsyncBuildContext())
    try:
        with pytest.raises(TypeError, match=r"Context\.abuild_input\(\) must return"):
            asyncio.run(agent.ainvoke("validate async Context contract"))
    finally:
        agent.close()


def test_session_uses_context_state_contract_with_framework_history():
    store = InMemorySessionStore()
    first = BasicAgent("source", SessionLLM()).with_context(IndependentContext()).with_session(store)
    second = BasicAgent("target", SessionLLM()).with_context(IndependentContext()).with_session(store)
    try:
        first.invoke("persist my context")
        saved = first.save_session()
        state = store.get_session(saved)["snapshot"]
        assert "canonical" in state["history"]
        assert "context" not in state["modules"]  # no second copy of owned history
        second.restore_session(saved)
        assert second.history == first.history
        second.add_user_message("independent branch")
        assert len(second.history) == len(first.history) + 1
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("path", ["ainvoke", "astream"])
def test_executor_awaits_custom_context_async_interfaces(path):
    class AsyncContext(IndependentContext):
        def build_input(self, *args, **kwargs):
            pytest.fail("Synchronous Context input assembly used by async executor")
        async def abuild_input(self, *args, **kwargs):
            await asyncio.sleep(0)
            return IndependentContext.build_input(self, *args, **kwargs)
    context = AsyncContext()
    agent = BasicAgent("async-context", OfflineLLM()).with_context(context).add_tool(CountingWriteTool())
    try:
        run_path(agent, path)
        assert len(context.prepared) == 2
    finally:
        agent.close()


def test_deferred_tool_owns_its_context_message():
    from Tool.ToolRegistry import ToolRegistry
    from Tool.builtin.tool_schema_tool import ToolSchemaTool
    registry = ToolRegistry()
    registry.register_tool(CountingWriteTool())
    result = ToolSchemaTool(registry).run({"tool_names": ["WriteProbe"]})
    assert result.structured_data["expandedToolNames"] == ["WriteProbe"]
    assert "WriteProbe" in result.ephemeral_context


def test_metamessage_uses_custom_context_public_message_interface():
    context = IndependentContext()
    agent = BasicAgent("injected", SessionLLM()).with_context(context)
    try:
        agent.emit_metamessage(MetaMessage(name="temporary", content="private temporary context",
                                          lifecycle=MetaMessageLifecycle.INVOCATION))
        agent.invoke("hello")
        assert "private temporary context" in repr(context.prepared[0].replay_history)
        assert "private temporary context" not in repr(agent.history)
    finally:
        agent.close()


class InstructionMemory(BaseMemory):
    def __init__(self, text):
        self.text = text

    def get_prompt_blocks(self, context):
        return [PromptBlock("memory.custom", self.text)]

    def get_retrieval_tools(self):
        return ()

    def get_persistence_tools(self):
        return ()


class IndependentPrompt(BaseSystemPromptComposer):
    def build(self, context):
        return [PromptBlock("identity", "independent identity")]


def test_prompt_contributions_survive_composer_change_and_replace_cleanly(tmp_path):
    folder = tmp_path / "review"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: review\ndescription: Review code\n---\nprivate skill body\n")
    agent = BasicAgent("prompt", SessionLLM()).with_memory(InstructionMemory("first"))
    try:
        agent.with_skill(tmp_path).with_plan().with_context(ContextManager(max_tokens=10000))
        agent.enter_plan_mode()
        agent.with_prompt(IndependentPrompt())
        blocks = {block.name: block.content for block in agent.get_system_prompt_blocks()}
        assert set(blocks) >= {"identity", "memory.custom", "skill_listing", "plan_mode", "compaction"}
        assert "private skill body" not in repr(blocks)
        agent.with_memory(InstructionMemory("second"), replace=True)
        blocks = [block for block in agent.get_system_prompt_blocks() if block.name == "memory.custom"]
        assert len(blocks) == 1 and blocks[0].content == "second"
        agent.with_prompt(SystemPromptComposer(include_defaults=False, include_module_blocks=False))
        assert agent.get_system_prompt_blocks() == []
    finally:
        agent.close()


def test_context_replacement_transfers_history_without_dual_ownership():
    agent = BasicAgent("replace", SessionLLM())
    try:
        agent.add_user_message("keep this")
        custom = IndependentContext()
        agent.with_context(custom)
        assert agent.history[0].text_content() == "keep this"
        agent.add_assistant_message("new context owns this")
        assert len(custom.canonical) == 2
        assert agent._services().context_manager is custom
    finally:
        agent.close()


def test_failed_context_transfer_keeps_original_context_and_history():
    class RejectHistory(IndependentContext):
        def replace(self, messages):
            raise ValueError("Unsupported existing history")
    agent = BasicAgent("failed-transfer", SessionLLM())
    try:
        agent.add_user_message("must survive")
        original = agent.context_manager
        with pytest.raises(ValueError, match="Unsupported existing history"):
            agent.with_context(RejectHistory())
        assert agent.context_manager is original
        assert agent.history[0].text_content() == "must survive"
    finally:
        agent.close()


def test_agent_exposes_only_executor_driven_invocation_for_tool_execution():
    agent = BasicAgent("direct", SessionLLM()).add_tool(CountingWriteTool())
    try:
        assert not hasattr(agent, "execute_tool_result")
        assert not hasattr(agent, "execute_approved_tool")
        assert not hasattr(agent, "grant_pending_permission")
    finally:
        agent.close()


def test_custom_context_resources_are_closed_without_closing_borrowed_model():
    class OwnedContext(IndependentContext):
        closed = False
        def close(self):
            self.closed = True
    context, llm = OwnedContext(), SessionLLM()
    agent = BasicAgent("resources", llm).with_context(context)
    report = agent.close(close_llm=False)
    assert context.closed
    assert report["components"]["context"]["status"] == "closed"
    assert "llm" not in report["components"]


def test_minimal_agent_installs_only_required_functional_modules():
    agent = BasicAgent("minimal", SessionLLM())
    try:
        assert agent.context_manager is not None
        assert isinstance(agent.executor, BaseAgentExecutor)
        assert isinstance(agent.prompt_composer, BaseSystemPromptComposer)
        for name in ("memory", "skill_manager", "permission_engine", "hook_manager",
                     "callback_manager", "plan", "observability", "multi_agent", "tool_registry"):
            assert getattr(agent, name) is None
        assert agent.invoke("hello") == "session-response"
    finally:
        agent.close()


def test_module_boundaries_do_not_reintroduce_default_implementation_dependencies():
    root = Path(__file__).resolve().parents[1]
    for misplaced in ("hooks", "permissions", "session", "guardrails"):
        assert not (root / "core" / misplaced).exists()
    for path, forbidden in (
        (root / "executor/default.py", {"context.default", "context.history", "prompt.composer"}),
        (root / "prompt/composer.py", {"memory", "skill", "plan", "context.default.manager", "Tool"}),
        (root / "session/manager.py", {"context.history", "context.default", "core.agent"}),
    ):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not any((node.module or "").startswith(name) for name in forbidden), (path, node.module)
    source = (root / "core/agent.py").read_text()
    assert "DefaultAgentExecutor._" not in source
    assert "self.history_store" not in source
    assert "RequestPreparer" not in (root / "executor/default.py").read_text()
    assert 'tool_name != "tool_schema_tool"' not in (root / "executor/default.py").read_text()
    executor_source = (root / "executor/default.py").read_text()
    for obsolete in (
        "_prepare_tool", "_finalize_tool_result", "_execute_tool",
        "_aexecute_tool", "_apply_after_tool_hook", "_commit_tool_result",
    ):
        assert obsolete not in executor_source
    agent_source = (root / "core/agent.py").read_text()
    for obsolete in (
        "def execute_tool_result", "def execute_approved_tool",
        "def grant_pending_permission",
    ):
        assert obsolete not in agent_source
    assert not list((root / "agent/components").glob("*.py"))
    history_source = (root / "context/history.py").read_text()
    for forbidden in ("core.llm", "provider_name", "_replay", "change_llm"):
        assert forbidden not in history_source
    permission_models = (root / "permissions/models.py").read_text()
    assert "ApprovalMode" not in permission_models
    session_source = (root / "session/manager.py").read_text()
    assert "self.agent" not in session_source
    assert "from core.agent" not in session_source


def test_framework_history_is_concrete_and_provider_independent():
    from context.history import ConversationHistory
    from core.history import CanonicalBlock, CanonicalMessage

    history = ConversationHistory()
    handle = history.append(CanonicalMessage(
        role="user",
        content=[CanonicalBlock(type="text", text="hello")],
    ))
    assert handle
    assert history.messages[0].text_content() == "hello"
    assert history.revision == 1
    restored = ConversationHistory()
    restored.restore_state(history.export_state())
    assert restored.messages == history.messages
