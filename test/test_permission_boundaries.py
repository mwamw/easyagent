"""Regression gates for the Skill / Plan / authorization contract boundaries.

No real model, filesystem-writing tool, or external service is called here.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent import BasicAgent
from easyagent.executors import AgentInvocationState
from core.Exception import ToolConfirmationRequired
from core.llm import EasyLLM
from permissions import (
    ApprovalMode, BasePermissionEngine, PermissionBehavior, PermissionDecision,
    RuleBasedPermissionEngine,
)
from plan import ExecutionMode, PlanModeConfig
from runtime import RuntimeEventType
from skill import SkillManager, SkillTool
from skill.tool import SKILL_TOOL_PROMPT
from Tool import Tool


class EmptyInput(BaseModel):
    pass


class CountingWriteTool(Tool):
    def __init__(self):
        super().__init__("WriteProbe", "Simulate a write", EmptyInput,
                         destructive=True, risk_categories=["filesystem_write"])
        self.calls = 0

    def run(self, parameters):
        self.calls += 1
        return "written"


class ScriptedProvider:
    def __init__(self):
        self.calls = 0

    def build_request(self, messages, **kwargs):
        return {"messages": list(messages), **kwargs}

    def build_tool_payload(self, tools):
        return list(tools)

    def apply_cache_policy(self, request, request_input):
        return request

    def invoke_raw(self, request):
        self.calls += 1
        calls = [SimpleNamespace(id="write-1", function=SimpleNamespace(
            name="WriteProbe", arguments="{}"))] if self.calls == 1 else []
        return SimpleNamespace(content=None if calls else "done", tool_calls=calls,
                               reasoning_content=None, usage=None)

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    def stream_raw(self, request):
        response = self.invoke_raw(request)
        calls = [SimpleNamespace(index=0, id=call.id, function=call.function)
                 for call in response.tool_calls]
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=response.content, tool_calls=calls or None,
                                  reasoning_content=None, reasoning=None),
            finish_reason="tool_calls" if calls else "stop",
        )])

    async def async_stream_raw(self, request):
        async def chunks():
            for chunk in self.stream_raw(request):
                yield chunk
        return chunks()


class OfflineLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "openai"
        self.model = "offline"
        self.base_url = "http://unused.invalid/v1"
        self.api_key = "test"
        self.max_tokens = 256
        self.temperature = 0
        self.kwargs = {}
        self._provider = ScriptedProvider()
        self.client = None


class RecordingEngine(BasePermissionEngine):
    """Intentionally no approval_mode or rule-management API."""
    def __init__(self, behavior):
        self.behavior = behavior
        self.requests = []

    def authorize(self, request):
        self.requests.append(request)
        return PermissionDecision(self.behavior, "Custom policy decision")


class ContextAgent(BasicAgent):
    def build_authorization_context(self):
        return replace(super().build_authorization_context(),
                       metadata={"context_source": "agent-factory"})


def run_path(agent, path):
    if path == "invoke":
        return agent.invoke("run the probe")
    if path == "ainvoke":
        return asyncio.run(agent.ainvoke("run the probe"))
    if path == "stream":
        return list(agent.stream("run the probe"))

    async def collect():
        return [event async for event in agent.astream("run the probe")]
    return asyncio.run(collect())


@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.parametrize("behavior", list(PermissionBehavior))
def test_executor_defers_plan_decisions_to_custom_engine(path, behavior):
    tool = CountingWriteTool()
    engine = RecordingEngine(behavior)
    agent = ContextAgent("boundaries", OfflineLLM()).add_tool(tool).with_permissions(engine)
    agent.with_plan(config=PlanModeConfig(register_tools=False))
    agent.enter_plan_mode(allowed_actions=["Read"])
    try:
        if behavior == PermissionBehavior.ASK:
            with pytest.raises(ToolConfirmationRequired):
                run_path(agent, path)
            assert agent.get_pending_interruption() is not None
        else:
            run_path(agent, path)
        assert len(engine.requests) == 1
        request = engine.requests[0]
        assert request.context.runtime_mode == "plan"
        assert request.context.plan_allowed_actions == ("Read",)
        assert request.context.metadata == {"context_source": "agent-factory"}
        assert request.descriptor.side_effect_level == "high"
        assert tool.calls == int(behavior == PermissionBehavior.ALLOW)
    finally:
        agent.close()


@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
def test_default_engine_still_blocks_plan_writes(path):
    tool = CountingWriteTool()
    engine = RuleBasedPermissionEngine(approval_mode=ApprovalMode.FULL_ACCESS, load_policies=False)
    engine.add_rule(tool_name=tool.name, behavior="allow")
    agent = BasicAgent("default-plan", OfflineLLM()).add_tool(tool).with_permissions(engine)
    agent.with_plan(config=PlanModeConfig(register_tools=False))
    agent.enter_plan_mode()
    try:
        run_path(agent, path)
        assert tool.calls == 0
        assert any(event["data"].get("error_type") == "permission_denied"
                   for event in agent.get_trace_history() if event["type"] == "tool.invoke.failed")
    finally:
        agent.close()


@pytest.mark.parametrize("approval_mode", list(ApprovalMode))
@pytest.mark.parametrize("exit_via", ["agent", "plan"])
def test_plan_exit_preserves_approval_policy(approval_mode, exit_via):
    agent = BasicAgent("exit-plan", OfflineLLM()).with_permissions().with_plan(config=PlanModeConfig(register_tools=False))
    agent.set_permission_mode(approval_mode)
    before = agent.permission_engine.export_state()
    factory = agent._services().authorization_context_factory
    try:
        assert factory().runtime_mode == "execute"
        agent.enter_plan_mode(allowed_actions=["Read"])
        assert factory().runtime_mode == "plan"
        if exit_via == "agent":
            agent.exit_plan_mode()
        else:
            agent.plan.exit()
        assert factory().runtime_mode == "execute"
        assert factory().plan_allowed_actions == ()
        assert agent.permission_engine.approval_mode == approval_mode
        assert agent.get_execution_mode() == ExecutionMode.EXECUTE
        assert agent.permission_engine.export_state() == before
    finally:
        agent.close()


def test_mode_changed_during_plan_is_not_reset_on_exit():
    agent = BasicAgent("change-plan", OfflineLLM()).with_permissions().with_plan(config=PlanModeConfig(register_tools=False))
    try:
        agent.enter_plan_mode()
        agent.set_permission_mode(ApprovalMode.AUTO_APPROVE_SAFE)
        agent.exit_plan_mode()
        assert agent.permission_engine.approval_mode == ApprovalMode.AUTO_APPROVE_SAFE
    finally:
        agent.close()


def test_executor_uses_context_factory_without_reading_engine_mode():
    class OpaqueEngine(RecordingEngine):
        @property
        def approval_mode(self):
            raise AssertionError("Executor must obtain mode from its context factory")

    agent = BasicAgent("context-only", OfflineLLM()).add_tool(CountingWriteTool())
    try:
        services = agent._services()
        context = replace(agent.build_authorization_context(), runtime_mode="plan")
        engine = OpaqueEngine(PermissionBehavior.ALLOW)
        services.permission_engine = engine
        services.authorization_context_factory = lambda: context
        state = AgentInvocationState("invoke-test", "test", False, round_number=1)
        call = SimpleNamespace(
            id="call-test",
            function=SimpleNamespace(name="WriteProbe", arguments="{}"),
        )
        _, _, _, result = agent.executor.execute_tool_call(services, state, call)
        assert result.status == "success"
        assert engine.requests[0].context is context
    finally:
        agent.close()


@pytest.mark.parametrize("frontmatter", ["recommended-tools", "allowed-tools"])
@pytest.mark.parametrize("terminal", [RuntimeEventType.AGENT_INVOKE_COMPLETED,
                                     RuntimeEventType.AGENT_INVOKE_FAILED,
                                     RuntimeEventType.AGENT_INVOKE_INTERRUPTED])
def test_skill_never_mutates_authorization_or_tool_visibility(tmp_path, monkeypatch, frontmatter, terminal):
    folder = tmp_path / "review"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\nname: review\ndescription: Review code\n{frontmatter}: WriteProbe\n---\nInspect carefully.\n",
        encoding="utf-8",
    )
    agent = BasicAgent("skill-boundary", OfflineLLM()).add_tool(CountingWriteTool()).with_permissions()
    engine = agent.permission_engine
    engine.add_rule(tool_name="WriteProbe", behavior="deny")
    before = engine.export_state()
    expanded = agent.tool_registry.get_deferred_expanded_tool_names()

    def forbidden(*args, **kwargs):
        pytest.fail("Skill must not administer permissions or expose tool schemas")

    for name in ("add_rule", "clear_rules", "grant", "set_approval_mode"):
        monkeypatch.setattr(engine, name, forbidden)
    monkeypatch.setattr(agent.tool_registry, "expand_deferred_tools", forbidden)
    try:
        agent.with_skill(tmp_path)
        assert agent.tool_registry.execute_tool_result("skill_tool", {"skill": "review"}).status == "success"
        agent.event_bus.publish(terminal, agent_id=agent.name, invocation_id="skill-test")
        request = agent.tool_registry.build_permission_request(
            "WriteProbe", {}, context=agent.build_authorization_context()
        )
        assert engine.authorize(request).behavior == PermissionBehavior.DENY
        agent.skill_manager.restore_state(agent.skill_manager.export_state())
        agent.skill_manager.close()
        assert engine.export_state() == before
        assert agent.tool_registry.get_deferred_expanded_tool_names() == expanded
        assert agent.tool_registry.get_tool("WriteProbe").calls == 0
    finally:
        agent.close()


def test_skill_tool_prompt_and_descriptor_do_not_claim_permission_management():
    tool = SkillTool(SkillManager())
    assert "Skill-scoped tool permissions" not in SKILL_TOOL_PROMPT
    assert "do not grant permissions" in SKILL_TOOL_PROMPT
    assert "permissions" not in tool.get_permission_descriptor().resources
