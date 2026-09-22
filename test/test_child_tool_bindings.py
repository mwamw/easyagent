"""Child assembly regressions: no model/network calls or OS sandbox required."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from easyagent import BasicAgent, Config
from easyagent.tools import Tool, ToolRegistry, ToolResult
from easyagent.multiagent import SubagentRequest
from multiagent.default.tool_binding import clone_tool_registry_for_workspace
from multiagent.default.definitions import AgentDefinition
from permissions import BasePermissionEngine, PermissionBehavior, PermissionDecision, ToolPermissionDescriptor
from Tool.builtin import register_shell_tools
from Tool.builtin.filesystem import FileReadTool, ListTool
from Tool.builtin.notebook_edit import NotebookEditTool
from Tool.builtin.mcp_tool import MCPWrappedTool
from test_composition_boundaries import child_factory
from test_multiagent_module import DummyLLM, ProbeInput, ProbeTool
from test_permission_boundaries import CountingWriteTool, OfflineLLM, run_path


def test_shared_tool_registry_state_is_independent(tmp_path):
    tool = ProbeTool()
    parent = ToolRegistry()
    parent.register_tool(tool, expose_in_deferred=False)
    child = clone_tool_registry_for_workspace(parent, workspace_root=str(tmp_path))
    child.register_tool(tool, visibility="turn", expose_in_deferred=True)
    child.expand_deferred_tools([tool.name])
    assert child.get_tool(tool.name) is parent.get_tool(tool.name)
    assert parent.get_tool_spec(tool.name).visibility_scope == "resident"
    assert not parent.get_tool_spec(tool.name).expose_in_deferred
    assert child.get_tool_spec(tool.name).visibility_scope == "turn"
    assert child.get_tool_spec(tool.name).expose_in_deferred
    assert tool.spec.visibility_scope == "resident"
    assert parent.get_deferred_expanded_tool_names() == []
    assert parent.export_tools(mode="deferred") == []
    assert len(child.export_tools(mode="deferred")) == 1
    child.unregister_tool(tool.name)
    assert parent.has_tool(tool.name)
    assert child.get_deferred_expanded_tool_names() == []


def test_function_tool_is_inherited_without_multiagent_methods(tmp_path):
    registry = ToolRegistry()

    @registry.tool("echo", "Echo a value", ProbeInput)
    def echo(value):
        return value

    child = clone_tool_registry_for_workspace(registry, workspace_root=str(tmp_path))
    assert child.get_tool("echo") is registry.get_tool("echo")
    assert child.execute_tool("echo", {"value": "ok"}) == "ok"
    assert not hasattr(Tool, "create_child")
    assert not hasattr(BasePermissionEngine, "create_child")


def test_async_shared_tool_has_no_cross_call_argument_state(tmp_path):
    class AsyncEcho(ProbeTool):
        async def arun(self, parameters):
            await asyncio.sleep(0)
            return ToolResult.success(parameters["value"])

    registry = ToolRegistry()
    registry.register_tool(AsyncEcho())
    child = clone_tool_registry_for_workspace(registry, workspace_root=str(tmp_path))

    async def invoke_both():
        return await asyncio.gather(
            registry.get_tool("DelegateProbe").arun({"value": "parent"}),
            child.get_tool("DelegateProbe").arun({"value": "child"}),
        )

    assert [result.content for result in asyncio.run(invoke_both())] == ["parent", "child"]


def test_workspace_tools_and_process_managers_are_child_local(tmp_path):
    parent_root, child_root = tmp_path / "parent", tmp_path / "child"
    parent_root.mkdir()
    child_root.mkdir()
    (parent_root / "info.txt").write_text("parent content")
    (child_root / "info.txt").write_text("child content")
    registry = ToolRegistry()
    registry.register_tool(FileReadTool(workspace_root=str(parent_root)))
    registry.register_tool(ListTool(workspace_root=str(parent_root), max_results=7))
    registry.register_tool(NotebookEditTool(workspace_root=str(parent_root)))
    register_shell_tools(registry, workspace_root=str(parent_root))
    child = clone_tool_registry_for_workspace(registry, workspace_root=str(child_root))
    for name in ("FileRead", "List", "NotebookEdit", "Bash", "TaskOutput", "TaskStop"):
        assert child.get_tool(name) is not registry.get_tool(name)
    assert child.get_tool("List").max_results == 7
    assert child.get_tool("NotebookEdit").workspace_root == str(child_root)
    child_processes = child.get_tool("Bash").process_manager
    assert child_processes is not registry.get_tool("Bash").process_manager
    assert child.get_tool("TaskOutput").process_manager is child_processes
    assert child.get_tool("TaskStop").process_manager is child_processes
    with ThreadPoolExecutor(2) as pool:
        values = list(pool.map(
            lambda r: r.execute_tool("FileRead", {"file_path": "info.txt"}),
            [registry, child],
        ))
    assert "parent content" in values[0]
    assert "child content" in values[1]


def test_shell_helpers_not_added_if_missing_in_parent_and_override_called_once(tmp_path):
    registry = ToolRegistry()
    register_shell_tools(registry, workspace_root=str(tmp_path))
    registry.unregister_tool("TaskStop")
    calls = []

    def omit_output(tool, runtime):
        calls.append(tool.name)
        return None

    child = clone_tool_registry_for_workspace(
        registry, workspace_root=str(tmp_path), tool_factories={"TaskOutput": omit_output},
    )
    assert calls == ["TaskOutput"]
    assert child.has_tool("Bash")
    assert not child.has_tool("TaskOutput")
    assert not child.has_tool("TaskStop")


@pytest.mark.parametrize("bad", [lambda tool, runtime: object(), lambda tool, runtime: CountingWriteTool()])
def test_invalid_tool_factory_results_are_rejected(tmp_path, bad):
    registry = ToolRegistry()
    registry.register_tool(ProbeTool())
    with pytest.raises((TypeError, ValueError)):
        clone_tool_registry_for_workspace(
            registry, workspace_root=str(tmp_path), tool_factories={"DelegateProbe": bad},
        )


def test_mcp_tool_borrows_connection_but_closing_child_does_not_close_it(tmp_path):
    calls = []
    manager = SimpleNamespace(
        registry_server_name="local-test",
        execute_tool=lambda name, arguments: calls.append((name, arguments)) or "remote result",
    )
    tool = MCPWrappedTool(manager, {"name": "probe", "input_schema": {"type": "object", "properties": {}}})
    parent = BasicAgent("parent", DummyLLM(), config=Config(workspace_root=str(tmp_path))).add_tool(tool)
    child = child_factory(parent)(SubagentRequest("child", "probe"))
    try:
        assert child.tool_registry.get_tool(tool.name) is tool
        assert child.mcp_managers == []
        child.close(close_llm=False)
        assert parent.tool_registry.execute_tool(tool.name, {}) == "remote result"
        assert calls == [("probe", {})]
    finally:
        if not child.runtime.closed:
            child.close(close_llm=False)
        parent.close()


def test_public_multiagent_assembly_accepts_child_bindings(tmp_path):
    class BoundProbe(ProbeTool):
        def __init__(self, workspace):
            super().__init__()
            self.workspace = workspace

    class Policy(BasePermissionEngine):
        def __init__(self):
            self.requests = []
            self.runtime = None

        def bind(self, runtime):
            assert self.runtime is None
            self.runtime = runtime

        def authorize(self, request):
            self.requests.append(request)
            return PermissionDecision(PermissionBehavior.ALLOW)

    policy = Policy()
    parent = BasicAgent("parent", DummyLLM()).add_tool(BoundProbe("parent")).with_permissions(policy)
    parent.with_multi_agent(
        storage_dir=str(tmp_path / "agents"),
        child_tool_factories={"DelegateProbe": lambda tool, runtime: BoundProbe(runtime.workspace_root)},
        child_permission_factory=lambda engine, runtime: Policy(),
    )
    child = parent.multi_agent.agent_factory(SubagentRequest("child", "probe", workspace_root=str(tmp_path)))
    try:
        assert child.tool_registry.get_tool("DelegateProbe").workspace == str(tmp_path)
        assert parent.tool_registry.get_tool("DelegateProbe").workspace == "parent"
        assert child.permission_engine is not policy
        assert child.permission_engine.runtime is child.runtime
        assert policy.runtime is parent.runtime
    finally:
        child.close(close_llm=False)
        parent.close()


@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
def test_dontask_keeps_full_access_and_executor_still_checks_denies(tmp_path, path):
    parent = BasicAgent("parent", OfflineLLM(), config=Config(workspace_root=str(tmp_path)))
    parent.add_tool(CountingWriteTool()).with_permissions()
    factory = child_factory(parent)
    factory.child_tool_factories["WriteProbe"] = lambda tool, runtime: CountingWriteTool()
    child = factory(SubagentRequest("child", "write", mode="dontAsk"))
    try:
        assert child.runtime.permission_mode == "full_access"
        assert parent.runtime.permission_mode == "ask_for_approval"
        run_path(child, path)
        assert child.tool_registry.get_tool("WriteProbe").calls == 1
        assert parent.tool_registry.get_tool("WriteProbe").calls == 0
        child.permission_engine.add_rule(tool_name="WriteProbe", behavior="deny")
        request = child.tool_registry.build_permission_request(
            "WriteProbe", {}, context=child.build_authorization_context(),
        )
        assert child.permission_engine.authorize(request).behavior == PermissionBehavior.DENY
    finally:
        child.close(close_llm=False)
        parent.close()


def test_deferred_schema_tool_binds_child_registry(tmp_path):
    parent = BasicAgent("parent", DummyLLM(), config=Config(tool_schema_mode="deferred"))
    parent.add_tool(ProbeTool())
    child = child_factory(parent)(SubagentRequest("child", "probe", workspace_root=str(tmp_path)))
    try:
        child.tool_registry.execute_tool("tool_schema_tool", {"tool_names": ["DelegateProbe"]})
        assert child.tool_registry.get_deferred_expanded_tool_names() == ["DelegateProbe"]
        assert parent.tool_registry.get_deferred_expanded_tool_names() == []
    finally:
        child.close(close_llm=False)
        parent.close()


def test_registry_permission_overrides_do_not_mutate_shared_tool_or_leave_stale_rule():
    tool = ProbeTool()
    registry = ToolRegistry()
    registry.register_tool(tool, permission_descriptor=ToolPermissionDescriptor(
        tool.name, requires_confirmation=True,
    ))
    assert registry.list_tool_descriptors()[0]["requires_confirmation"]
    assert not tool.get_permission_descriptor().requires_confirmation
    registry.register_tool(tool, visibility="turn")
    assert registry.get_permission_descriptor(tool.name).requires_confirmation
    registry.register_tool(ProbeTool())
    assert not registry.get_permission_descriptor(tool.name).requires_confirmation


def test_child_module_tools_preserve_descriptors_and_final_filters(tmp_path):
    parent = BasicAgent("parent", DummyLLM()).with_plan()
    registry = parent.tool_registry
    tool = registry.get_tool("ExitPlanMode")
    registry.register_tool(tool, permission_descriptor=ToolPermissionDescriptor(
        tool.name, requires_confirmation=True, metadata={"rules": []},
    ))
    factory = child_factory(parent)
    factory.definitions.register(AgentDefinition(
        "restricted", "Test final filtering",
        tools=("ExitPlanMode", "EnterPlanMode"), disallowed_tools=("EnterPlanMode",),
    ))
    child = factory(SubagentRequest("child", "plan", agent_type="restricted", workspace_root=str(tmp_path)))
    try:
        assert child.tool_registry.get_tool_names() == ["ExitPlanMode"]
        descriptor = child.tool_registry.get_permission_descriptor("ExitPlanMode")
        assert descriptor.requires_confirmation
        descriptor.metadata["rules"].append("child")
        assert registry.get_permission_descriptor("ExitPlanMode").metadata["rules"] == []
        assert child.tool_registry.get_tool("ExitPlanMode") is not tool
    finally:
        child.close(close_llm=False)
        parent.close()


def test_bound_tools_can_be_explicitly_excluded_without_changing_tool_class(tmp_path):
    registry = ToolRegistry()
    tool = ProbeTool()
    registry.register_tool(tool)
    child = clone_tool_registry_for_workspace(
        registry, workspace_root=str(tmp_path), excluded_tools=[tool],
    )
    assert not child.has_tool(tool.name)
    assert registry.has_tool(tool.name)


def test_factory_omission_is_respected_after_child_module_installation(tmp_path):
    parent = BasicAgent("parent", DummyLLM()).with_plan()
    factory = child_factory(parent)
    calls = []

    def omit(tool, runtime):
        calls.append(tool.name)
        return None

    factory.child_tool_factories["ExitPlanMode"] = omit
    child = factory(SubagentRequest("child", "plan", workspace_root=str(tmp_path)))
    try:
        assert calls == ["ExitPlanMode"]
        assert not child.tool_registry.has_tool("ExitPlanMode")
        assert parent.tool_registry.has_tool("ExitPlanMode")
    finally:
        child.close(close_llm=False)
        parent.close()


def test_restricted_tool_factory_is_not_called(tmp_path):
    parent = BasicAgent("parent", DummyLLM()).add_tool(ProbeTool())
    factory = child_factory(parent)
    calls = []
    factory.child_tool_factories["DelegateProbe"] = lambda tool, runtime: calls.append(tool)
    child = factory(SubagentRequest("child", "read", agent_type="Explore", workspace_root=str(tmp_path)))
    try:
        assert calls == []
        assert not child.tool_registry.has_tool("DelegateProbe")
    finally:
        child.close(close_llm=False)
        parent.close()


@pytest.mark.parametrize("returns_parent", [False, True])
def test_child_permission_factory_cannot_disable_checks_or_rebind_parent(tmp_path, returns_parent):
    parent = BasicAgent("parent", DummyLLM()).with_permissions()
    factory = child_factory(parent)
    factory.child_permission_factory = lambda policy, runtime: policy if returns_parent else None
    try:
        with pytest.raises((TypeError, ValueError)):
            factory(SubagentRequest("child", "probe", mode="dontAsk", workspace_root=str(tmp_path)))
        assert parent.permission_engine._runtime is parent.runtime
    finally:
        parent.close()
