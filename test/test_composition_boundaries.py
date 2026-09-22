"""Cross-module regressions using only local doubles, never external model calls."""
from __future__ import annotations

import asyncio
import inspect
from threading import Lock

import pytest

from agent import BasicAgent
from codeintel import BaseCodeIntel, CodeIntelAvailability, CodeIntelQueryResult, CodeIntelManager
from context.base import BaseContextManager
from core.Config import Config
from core.Exception import SessionSerializationError
from executor.base import BaseAgentExecutor
from multiagent.default.definitions import build_default_agent_definitions
from multiagent.default.factory import DefaultSubagentFactory
from multiagent.default.subagents import SubagentRequest
from multiagent.default.tool_binding import clone_tool_registry_for_workspace
from observability.base import BaseObservabilityManager
from permissions import (
    BasePermissionEngine, PermissionBehavior, PermissionDecision, PermissionGrant,
    RuleBasedPermissionEngine, ToolPermissionDescriptor,
)
from plan import BasePlanMode, ExecutionMode
from runtime import AgentRuntime, RuntimeEvent, RuntimeEventType
from session import ComponentRestoreReport, InMemorySessionStore, SessionState
from task import BaseTaskService, TaskRecord, TaskService, InMemoryTaskStore
from Tool import ToolRegistry, ToolResult
from worktree import BaseWorktreeManager, GitWorktreeInfo, GitWorktreeSession

from test_multiagent_module import DummyLLM, ProbeTool
from test_module_ownership import IndependentContext
from test_permission_boundaries import CountingWriteTool, OfflineLLM, run_path


def child_factory(parent: BasicAgent) -> DefaultSubagentFactory:
    return DefaultSubagentFactory(
        parent_name=parent.name, llm=parent.llm, config=parent.config,
        system_prompt=None, reasoning=None, runtime=parent.runtime,
        tool_registry=parent.tool_registry or ToolRegistry(),
        permission_engine=parent.permission_engine, observability=parent.observability,
        definitions=build_default_agent_definitions(), plan=parent.plan,
    )


class LockedPolicy(BasePermissionEngine):
    def __init__(self):
        self.lock = Lock()
        self.requests = []

    def authorize(self, request):
        with self.lock:
            self.requests.append(request)
        return PermissionDecision(PermissionBehavior.DENY, "external policy")

    async def aauthorize(self, request):
        return self.authorize(request)


class EventSink(BaseObservabilityManager):
    def __init__(self):
        self.events = []
        self.closed = False

    def bind(self, *, agent_id, event_bus):
        self.parent_bus = event_bus
        return self

    def handle_runtime_event(self, event):
        self.events.append(event)

    def list(self):
        return []

    def close(self):
        self.closed = True


def test_child_supports_minimal_permission_and_observability_contracts(tmp_path):
    parent = BasicAgent("parent", DummyLLM(), config=Config(workspace_root=str(tmp_path)))
    policy, sink = LockedPolicy(), EventSink()
    parent.with_permissions(policy).with_observability(sink)
    child = child_factory(parent)(SubagentRequest("worker", "no model", mode="bypassPermissions"))
    try:
        assert child.runtime is not parent.runtime
        assert child.permission_engine is not policy
        child.add_tool(ProbeTool())
        request = child.tool_registry.build_permission_request(
            "DelegateProbe", {"value": "check"}, context=child.build_authorization_context(),
        )
        assert child.permission_engine.authorize(request).behavior == PermissionBehavior.DENY
        assert asyncio.run(child.permission_engine.aauthorize(request)).behavior == PermissionBehavior.DENY
        assert policy.requests[-1].context.session_id == child.session_id
        assert parent.runtime.permission_mode == "ask_for_approval"
        assert child.invoke("hello") == "done"
        assert any(event.agent_id == child.runtime.agent_id for event in sink.events)
        assert sink.parent_bus is parent.event_bus
        child.close(close_llm=False)
        assert not sink.closed
        count = len(sink.events)
        child.event_bus.publish(RuntimeEventType.AGENT_INVOKE_STARTED, agent_id="closed", invocation_id="ignored")
        assert len(sink.events) == count
    finally:
        if not child.runtime.closed:
            child.close(close_llm=False)
        parent.close()


def test_default_permission_child_has_independent_rules_and_no_parent_grants(tmp_path):
    parent = BasicAgent("parent", DummyLLM(), config=Config(workspace_root=str(tmp_path))).with_permissions()
    parent.permission_engine.add_rule(tool_name="Forbidden", behavior="deny")
    parent.permission_engine.grant(PermissionGrant("WriteProbe", {}, session_id=parent.session_id))
    child = child_factory(parent)(SubagentRequest("worker", "no model"))
    try:
        assert child.permission_engine.grants() == []
        child.add_tool(CountingWriteTool())
        request = child.tool_registry.build_permission_request(
            "WriteProbe", {}, context=child.build_authorization_context(),
        )
        assert child.permission_engine.authorize(request).behavior == PermissionBehavior.ASK
        assert len(parent.permission_engine.grants()) == 1
        child.permission_engine.add_rule(tool_name="OnlyChild", behavior="deny")
        assert len(parent.permission_engine.rules()) == 1
        assert len(child.permission_engine.rules()) == 2
        child.set_permission_mode("full_access")
        assert parent.runtime.permission_mode == "ask_for_approval"
        assert child.runtime.permission_mode == "full_access"
        parent.runtime.permission_mode = "auto_approve_safe"
        parent.build_prompt_context()
        assert parent.permission_engine.approval_mode.value == "auto_approve_safe"
    finally:
        child.close(close_llm=False)
        parent.close()


def test_child_plan_tools_do_not_modify_parent_plan(tmp_path):
    parent = BasicAgent("parent", DummyLLM(), config=Config(workspace_root=str(tmp_path))).with_plan()
    parent.enter_plan_mode()
    child = child_factory(parent)(SubagentRequest("worker", "no model", mode="plan"))
    try:
        assert child.plan is not parent.plan
        tool = child.tool_registry.get_tool("ExitPlanMode")
        assert tool is not parent.tool_registry.get_tool("ExitPlanMode")
        tool.run({"allowedPrompts": []})
        assert child.plan.state.exit_requested
        assert not parent.plan.state.exit_requested
        child.exit_plan_mode()
        assert parent.runtime.execution_mode == "plan"
    finally:
        child.close(close_llm=False)
        parent.close()


def test_tool_bindings_belong_to_factory_and_preserve_descriptors(tmp_path):
    class ChildTool(ProbeTool):
        pass

    def bind_child(tool, runtime):
        child = ChildTool()
        child.workspace = runtime.workspace_root
        return child

    registry = ToolRegistry()
    unsafe = CountingWriteTool()
    registry.register_tool(unsafe)
    tool = ChildTool()
    descriptor = ToolPermissionDescriptor(tool.name, requires_confirmation=True, metadata={"rules": []})
    registry.register_tool(tool, permission_descriptor=descriptor)
    runtime = AgentRuntime(workspace_root=str(tmp_path))
    child = clone_tool_registry_for_workspace(
        registry, workspace_root=str(tmp_path), runtime=runtime,
        tool_factories={tool.name: bind_child, unsafe.name: lambda tool, runtime: None},
    )
    assert not child.has_tool(unsafe.name)
    assert child.get_tool(tool.name) is not tool
    assert child.get_tool(tool.name).workspace == str(tmp_path)
    copied = child.get_permission_descriptor(tool.name)
    assert copied.requires_confirmation
    copied.metadata["rules"].append("child")
    assert descriptor.metadata["rules"] == []


def test_ordinary_tool_needs_no_child_interface(tmp_path):
    registry = ToolRegistry()
    tool = ProbeTool()
    registry.register_tool(tool)
    child = clone_tool_registry_for_workspace(registry, workspace_root=str(tmp_path))
    assert child.get_tool(tool.name) is tool
    assert not hasattr(tool, "create_child")
    assert child.execute_tool(tool.name, {"value": "shared"}) == "shared"


class SimplePlan(BasePlanMode):
    def bind(self, *, runtime, tool_registry):
        self.runtime = runtime
        return self

    @property
    def mode(self):
        return ExecutionMode(self.runtime.execution_mode)

    def enter(self, *, allowed_actions=None):
        self.runtime.set_execution_mode("plan", allowed_actions=tuple(allowed_actions or ()))

    def exit(self):
        self.runtime.set_execution_mode("execute")

    def get_tools(self):
        return (ProbeTool(),)


def test_plan_assembly_needs_no_default_configuration_or_install_method():
    plan = SimplePlan()
    agent = BasicAgent("plan", DummyLLM()).with_plan(plan)
    try:
        assert not hasattr(plan, "config")
        assert agent.tool_registry.has_tool("DelegateProbe")
        agent.enter_plan_mode()
        agent.request_exit_plan_mode()
        assert agent.runtime.execution_mode == "plan"
        agent.exit_plan_mode()
        assert agent.runtime.execution_mode == "execute"
        assert agent.with_plan(plan) is agent
    finally:
        agent.close()


class SimpleCodeIntel(BaseCodeIntel):
    def bind(self, runtime):
        self.runtime = runtime

    def get_status(self, *, file_path=None):
        return CodeIntelAvailability(True, "custom", self.runtime.workspace_root)

    def find_definition(self, **kwargs):
        return CodeIntelQueryResult.ok(provider_name="custom", workspace_root=self.runtime.workspace_root)

    find_references = find_definition
    get_document_symbols = find_definition
    get_workspace_symbols = find_definition
    get_diagnostics = find_definition


def test_custom_codeintel_module_receives_runtime_without_agent(tmp_path):
    module = SimpleCodeIntel()
    agent = BasicAgent("intel", DummyLLM(), config=Config(workspace_root=str(tmp_path))).with_codeintel(module)
    try:
        assert module.runtime is agent.runtime
        assert not hasattr(module, "parent_agent")
        result = agent.tool_registry.execute_tool_result("CodeIntelStatus", {})
        assert result.status == "success"
        assert "parent_agent" not in inspect.signature(CodeIntelManager).parameters
    finally:
        agent.close()


class SimpleTasks(BaseTaskService):
    def __init__(self):
        self.backend = TaskService(InMemoryTaskStore())

    def create_task(self, **kwargs):
        return self.backend.create_task(**kwargs)

    def get_task(self, task_id):
        return self.backend.get_task(task_id)

    def update_task(self, task_id, **kwargs):
        return self.backend.update_task(task_id, **kwargs)

    def list_tasks(self, **kwargs):
        return self.backend.list_tasks(**kwargs)


def test_custom_task_module_is_not_required_to_inherit_default_service():
    module = SimpleTasks()
    agent = BasicAgent("tasks", DummyLLM()).with_task_service(module)
    try:
        assert not isinstance(module, TaskService)
        result = agent.tool_registry.execute_tool_result("TaskCreate", {"title": "test", "description": "local"})
        assert result.status == "success"
        assert len(module.list_tasks()) == 1
    finally:
        agent.close()


class SimpleWorktree(BaseWorktreeManager):
    def __init__(self, path):
        self.path, self.active = path, None

    def create_worktree(self, name, **kwargs):
        return GitWorktreeInfo(self.path, name, "test-head")

    def remove_worktree(self, path, *, force=False):
        pass

    def get_active_session(self):
        return self.active

    def enter_worktree(self, name=None, **kwargs):
        self.active = GitWorktreeSession(self.path, self.create_worktree(name or "test"), "test-head", 0)
        return self.active

    def exit_worktree(self, action, *, discard_changes=False):
        self.active = None
        return {"status": "kept"}

    def close(self, *, action="keep", discard_changes=False):
        return self.exit_worktree(action, discard_changes=discard_changes)


def test_custom_worktree_module_is_used_through_contract(tmp_path):
    module = SimpleWorktree(str(tmp_path))
    agent = BasicAgent("worktree", DummyLLM()).with_worktree(module)
    try:
        result = agent.tool_registry.execute_tool_result("EnterWorktree", {"name": "local"})
        assert result.status == "success"
        agent.build_prompt_context()
        assert agent.runtime.worktree_path == str(tmp_path)
        assert agent.runtime.worktree_branch == "local"
    finally:
        agent.close()


@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
def test_context_owns_validation_generation_on_all_execution_paths(path):
    class EpochContext(IndependentContext):
        def validation_epoch(self, compaction):
            return 731

        def validate_input(self, request_input, *, tools, options, history_epoch=None):
            assert history_epoch == 731

    agent = BasicAgent("context", OfflineLLM()).with_context(EpochContext()).add_tool(CountingWriteTool())
    try:
        run_path(agent, path)
        usage = agent.get_context_usage()
        assert usage["estimatedRequestTokens"] == 0
        assert usage["canonicalMessages"] == 4
    finally:
        agent.close()


def test_core_contract_public_methods_have_return_annotations():
    for cls in (BaseContextManager, BaseAgentExecutor):
        for name, member in vars(cls).items():
            if name.startswith("_"):
                continue
            function = member.fget if isinstance(member, property) else member
            if inspect.isfunction(function):
                assert inspect.signature(function).return_annotation is not inspect.Signature.empty, name


def test_session_accepts_structured_reports_and_rolls_back_invalid_results():
    class Counter:
        def __init__(self):
            self.value = 1

        def export_state(self):
            return {"value": self.value}

        def restore_state(self, state):
            self.value = state["value"]
            return ComponentRestoreReport("counter")

    module = Counter()
    assert isinstance(module, SessionState)
    agent = BasicAgent("session", DummyLLM()).with_session(InMemorySessionStore())
    agent.register_session_state("counter", module, required=True)
    try:
        session_id = agent.save_session()
        module.value = 2
        agent.restore_session(session_id)
        assert module.value == 1
        assert agent.get_last_restore_report()["components"]["counter"]["status"] == "restored"
        original = module.restore_state
        module.value = 3
        def invalid(state):
            original(state)
            return "invalid contract result"
        module.restore_state = invalid
        with pytest.raises(TypeError, match="Invalid restore result"):
            agent.restore_session(session_id)
        assert module.value == 3
    finally:
        agent.close()


def test_interruption_resolution_uses_installed_context_for_tool_context():
    class ToolContext(IndependentContext):
        def append_tool_context(self, tool_name, value):
            self.received = (tool_name, value)
    context = ToolContext()
    agent = BasicAgent("interrupt", DummyLLM()).with_context(context)
    try:
        agent.runtime.interruptions.create_interruption(
            tool_name="Probe", tool_args={}, tool_id="call-1", round_number=1,
            tool_result=ToolResult.needs_confirmation("confirm"),
        )
        agent.resolve_pending_interruption(content="approved", ephemeral_context={"source": "custom"})
        assert context.received == ("Probe", {"source": "custom"})
        assert agent.get_pending_interruption() is None
    finally:
        agent.close()


def test_permission_subclass_authorization_is_preserved_in_child():
    class StrictPolicy(RuleBasedPermissionEngine):
        def authorize(self, request):
            return PermissionDecision(PermissionBehavior.DENY, "strict subclass")
    parent = BasicAgent("strict", DummyLLM()).with_permissions(StrictPolicy(load_policies=False))
    child = child_factory(parent)(SubagentRequest("child", "no model"))
    try:
        child.add_tool(ProbeTool())
        request = child.tool_registry.build_permission_request(
            "DelegateProbe", {"value": "x"}, context=child.build_authorization_context(),
        )
        assert child.permission_engine.authorize(request).reason == "strict subclass"
    finally:
        child.close(close_llm=False)
        parent.close()


def test_agent_does_not_read_permission_implementation_mode_attribute():
    class OpaquePolicy(LockedPolicy):
        @property
        def approval_mode(self):
            raise AssertionError("Not part of BasePermissionEngine")
    agent = BasicAgent("opaque", DummyLLM()).with_permissions(OpaquePolicy())
    try:
        agent.build_prompt_context()
        assert agent.invoke("hello") == "done"
        with pytest.raises(NotImplementedError, match="policy modes"):
            agent.set_permission_mode("unsupported")
    finally:
        agent.close()
