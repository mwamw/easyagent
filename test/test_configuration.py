"""Declarative assembly tests use offline model doubles, never paid APIs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent import BasicAgent
from easyagent.configuration import (
    AgentBuilder, ConfigurationError, ConfigurationRegistry, Options,
    default_registry, load_agent, read_configuration, validate_agent,
)
from permissions import BasePermissionEngine, PermissionBehavior, PermissionDecision
from Tool import ToolRegistry
from test_multiagent_module import DummyLLM, ProbeTool
from test_permission_boundaries import CountingWriteTool, OfflineLLM, run_path


def write_config(tmp_path, data):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def registry_with_model(model_type=DummyLLM):
    registry = default_registry()
    models = []

    def factory(options, paths):
        model = model_type()
        model.closed = False
        original = model.close

        def close():
            model.closed = True
            original()
        model.close = close
        models.append(model)
        return model

    registry.register_model("offline", config_model=Options, factory=factory)
    return registry, models


def document(**sections):
    return {"model": {"implementation": "offline"}, **sections}


def test_minimal_returns_ordinary_agent_and_can_continue_python_assembly(tmp_path):
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(agent={"name": "configured"}))
    agent = load_agent(path, registry=registry)
    try:
        assert type(agent) is BasicAgent
        assert agent.name == "configured"
        assert agent.runtime.workspace_root == str(tmp_path)
        assert agent.memory is None and agent.permission_engine is None
        assert agent.tool_registry is None
        agent.add_tool(ProbeTool())
        assert agent.invoke("hello") == "done"
    finally:
        agent.close()
    assert models[0].closed


@pytest.mark.parametrize("path_kind", ["invoke", "ainvoke", "stream", "astream"])
def test_four_invocation_paths_use_same_executor(tmp_path, path_kind):
    registry, models = registry_with_model(OfflineLLM)
    registry.register_tools("probe", config_model=Options,
                            register=lambda r, o, c: r.register_tool(CountingWriteTool()))
    path = write_config(tmp_path, document(
        tools=[{"name": "probe"}],
        permissions={"implementation": "rule_based", "config": {"approval_mode": "full_access"}},
        context={"config": {"max_tokens": 16000}},
    ))
    agent = load_agent(path, registry=registry)
    try:
        run_path(agent, path_kind)
        assert agent.tool_registry.get_tool("WriteProbe").calls == 1
    finally:
        agent.close()


@pytest.mark.parametrize("section", ["context", "executor", "prompt", "model"])
def test_required_components_cannot_be_disabled(tmp_path, section):
    registry, models = registry_with_model()
    data = document()
    data[section] = {"enabled": False}
    with pytest.raises(ConfigurationError):
        load_agent(write_config(tmp_path, data), registry=registry)
    assert models == []


@pytest.mark.parametrize("data", [
    {"unexpected": True},
    {"version": 2},
    {"context": {"config": {"max_token": 42}}},
    {"context": {"config": {"max_tokens": -1}}},
    {"permissions": {"implementation": "unknown"}},
    {"tools": [{"name": "unknown"}]},
    {"tools": [{"name": "calculator"}, {"name": "calculator"}]},
    {"skills": {"directories": ["missing"]}},
    {"agent": {"workspace": "missing"}},
    {"prompt": {"file": "missing"}},
    {"prompt": {"file": "missing", "text": "also"}},
    {"agent": {"config": {"workspace_root": "other"}}},
    {"mcp": {"servers": [{"name": "x", "source": "not-url", "source_type": "url"}]}},
])
def test_invalid_documents_fail_before_constructing_models(tmp_path, data):
    registry, models = registry_with_model()
    with pytest.raises(ConfigurationError):
        load_agent(write_config(tmp_path, document(**data)), registry=registry)
    assert models == []


def test_disabled_optional_component_does_not_load_or_resolve_credentials(tmp_path):
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(memory={
        "enabled": False, "implementation": "not-installed",
        "config": {"key": {"env": "EASYAGENT_TEST_DEFINITELY_MISSING"}},
    }))
    assert validate_agent(path, registry=registry) == ("model",)
    assert models == []
    agent = load_agent(path, registry=registry)
    assert agent.memory is None
    agent.close()


@pytest.mark.parametrize("content", [
    "model: {}\nmodel: {}",
    "model: !!python/object/apply:os.system ['false']",
    "model: &cycle {config: *cycle}",
    "[]",
    "model: [",
])
def test_yaml_is_safe_and_duplicate_keys_are_not_ignored(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        read_configuration(path)


def test_paths_are_config_relative_not_process_relative(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "prompt.md").write_text("Project instructions", encoding="utf-8")
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(
        agent={"workspace": "./workspace"},
        prompt={"file": "./prompt.md"},
        tools=[{"name": "filesystem"}, {"name": "file_write"}, {"name": "file_edit"}, {"name": "notebook_edit"}],
        session={"implementation": "sqlite", "config": {"path": "./state/sessions.db"}},
    ))
    monkeypatch.chdir(workspace)
    # Validation must not create SQLite files.
    validate_agent(path, registry=registry)
    assert not (tmp_path / "state").exists()
    agent = load_agent(path, registry=registry)
    try:
        assert agent.system_prompt == "Project instructions"
        assert agent.tool_registry.get_tool("FileRead").workspace_root == str(workspace)
        assert (tmp_path / "state/sessions.db").exists()
        agent.invoke("hello")
        agent.save_session()
        from session import SQLiteSessionStore
        assert SQLiteSessionStore(tmp_path / "state/sessions.db").get_session(agent.session_id)
    finally:
        agent.close()


def test_custom_permission_gets_typed_options_not_agent(tmp_path):
    class PolicyOptions(Options):
        reason: str

    class Policy(BasePermissionEngine):
        def __init__(self, reason):
            self.reason = reason

        def authorize(self, request):
            return PermissionDecision(PermissionBehavior.DENY, self.reason)

    registry, models = registry_with_model()
    seen = []

    def create(options, context):
        assert isinstance(options, PolicyOptions)
        assert not hasattr(context, "agent")
        seen.append(context.paths.workspace)
        return Policy(options.reason)

    registry.register_permission("company", config_model=PolicyOptions, factory=create)
    path = write_config(tmp_path, document(permissions={"implementation": "company", "config": {"reason": "company"}}))
    agent = load_agent(path, registry=registry)
    assert agent.permission_engine.reason == "company"
    assert seen == [tmp_path]
    agent.close()


def test_invalid_custom_return_is_rejected_not_replaced_with_default(tmp_path):
    registry, models = registry_with_model()
    registry.register_permission("broken", config_model=Options, factory=lambda o, c: None)
    with pytest.raises(ConfigurationError, match="permissions"):
        load_agent(write_config(tmp_path, document(permissions={"implementation": "broken"})), registry=registry)
    assert models[0].closed


def test_partial_failure_closes_model_and_hides_external_exception_secrets(tmp_path):
    registry, models = registry_with_model()
    secret = "private-key-DO-NOT-ECHO"

    def fail(options, context):
        raise RuntimeError(secret)

    registry.register_permission("broken", config_model=Options, factory=fail)
    with pytest.raises(ConfigurationError) as error:
        load_agent(write_config(tmp_path, document(permissions={"implementation": "broken"})), registry=registry)
    assert secret not in str(error.value)
    assert error.value.__suppress_context__
    assert models[0].closed


def test_later_failure_rolls_back_unmanaged_policy_resources(tmp_path):
    registry, models = registry_with_model()
    closed = []

    class Policy(BasePermissionEngine):
        def authorize(self, request):
            return PermissionDecision(PermissionBehavior.DENY)

        def close(self):
            closed.append("policy")

    def fail(agent, options, context):
        raise RuntimeError("later")

    registry.register_permission("resource", config_model=Options, factory=lambda o, c: Policy())
    registry.register_installer("multi_agent", "broken", config_model=Options, install=fail)
    with pytest.raises(ConfigurationError):
        load_agent(write_config(tmp_path, document(
            permissions={"implementation": "resource"},
            multi_agent={"implementation": "broken"},
        )), registry=registry)
    assert closed == ["policy"]
    assert models[0].closed


def test_environment_reference_and_error_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYAGENT_CONFIG_TEST_KEY", "private-key-123")
    path = write_config(tmp_path, {"model": {"api_key": {"env": "EASYAGENT_CONFIG_TEST_KEY"}}})
    config = read_configuration(path)
    assert config.model.config["api_key"] == "private-key-123"
    path = write_config(tmp_path, {"model": {"api_key": {"env": "EASYAGENT_MISSING_KEY"}}})
    with pytest.raises(ConfigurationError, match="EASYAGENT_MISSING_KEY"):
        read_configuration(path)
    path = write_config(tmp_path, {"model": {"max_tokens": "private-key-123"}})
    with pytest.raises(ConfigurationError) as error:
        validate_agent(path)
    assert "private-key-123" not in str(error.value)


def test_builtin_tool_groups_and_sandbox_options(tmp_path):
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(tools=[
        {"name": "calculator"}, {"name": "web_search"}, {"name": "web_fetch"},
        {"name": "ask_user_question"},
        {"name": "bash", "config": {"sandbox": {
            "enabled": True, "mode": "regular", "network": "deny",
            "deny_read": ["./secret"],
        }}},
    ]))
    agent = load_agent(path, registry=registry)
    try:
        tool = agent.tool_registry.get_tool("Bash")
        assert tool.sandbox_config.enabled
        assert tool.sandbox_config.deny_read == (str(tmp_path / "secret"),)
        assert agent.tool_registry.has_tool("TaskOutput")
        assert agent.tool_registry.execute_tool("calculator", {"expression": "2+3"}) == "5"
    finally:
        agent.close()


def test_tools_duplicate_names_across_groups_fail_instead_of_overwriting(tmp_path):
    registry, models = registry_with_model()
    for name in ("one", "two"):
        registry.register_tools(name, config_model=Options, register=lambda r, o, c: r.register_tool(ProbeTool()))
    with pytest.raises(ConfigurationError):
        load_agent(write_config(tmp_path, document(tools=[{"name": "one"}, {"name": "two"}])), registry=registry)
    assert models[0].closed


def test_plan_permission_ask_and_memory_use_normal_module_paths(tmp_path):
    from core.Exception import ToolConfirmationRequired
    registry, models = registry_with_model(OfflineLLM)
    registry.register_tools("probe", config_model=Options,
                            register=lambda r, o, c: r.register_tool(CountingWriteTool()))
    path = write_config(tmp_path, document(
        tools=[{"name": "probe"}], plan={}, permissions={}, memory={},
        tasks={}, hooks={}, callbacks={}, observability={},
    ))
    agent = load_agent(path, registry=registry)
    try:
        assert agent.memory is not None
        assert agent.task_service is not None
        assert agent.plan is not None
        with pytest.raises(ToolConfirmationRequired):
            agent.invoke("write")
        assert agent.get_pending_interruption() is not None
        assert agent.tool_registry.get_tool("WriteProbe").calls == 0
    finally:
        agent.close()


def test_cli_check_never_builds_or_invokes(tmp_path, monkeypatch, capsys):
    from configuration.__main__ import main
    registry, models = registry_with_model()
    path = write_config(tmp_path, document())
    monkeypatch.setattr("configuration.__main__.validate_agent", lambda p: validate_agent(p, registry=registry))
    assert main([str(path), "--check"]) == 0
    assert "valid" in capsys.readouterr().out
    assert models == []


def test_skill_and_multiagent_are_assembled_with_completed_parent(tmp_path):
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: review\ndescription: Review changes\n---\nRead before editing.")
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(
        tools=[{"name": "filesystem"}],
        skills={"directories": ["./skills"]},
        permissions={}, tasks={}, plan={},
        multi_agent={"config": {"storage_dir": "./workers"}},
    ))
    agent = load_agent(path, registry=registry)
    try:
        assert "review" in agent.skill_manager.skill_names
        assert agent.tool_registry.has_tool("skill_tool")
        factory = agent.multi_agent.agent_factory
        assert factory.permission_engine is agent.permission_engine
        assert factory.task_service is agent.task_service
        assert factory.plan is agent.plan
        assert factory.skill_directories == (str(tmp_path / "skills"),)
        assert factory.runtime is agent.runtime
        from multiagent.default.subagents import SubagentRequest
        child = factory(SubagentRequest("child", "read", workspace_root=str(tmp_path)))
        try:
            assert child.tool_registry.has_tool("FileRead")
            assert child.skill_manager is not agent.skill_manager
        finally:
            child.close(close_llm=False)
    finally:
        agent.close()


def test_mcp_uses_standard_integration_and_owns_connections(tmp_path, monkeypatch):
    from Tool.builtin.mcp_tool import MCPToolManager
    created = []

    class LocalMCP(MCPToolManager):
        def __init__(self, **kwargs):
            self.arguments = kwargs
            self._registered_server_name = "local"
            self.closed = False
            created.append(self)

        @property
        def registry_server_name(self):
            return self._registered_server_name

        def register_to_registry(self, registry, **kwargs):
            self._registered_server_name = kwargs["server_name"]
            registry.register_tool(ProbeTool())

        def close(self):
            self.closed = True
            return {"status": "closed"}

    monkeypatch.setattr("Tool.builtin.MCPToolManager", LocalMCP)
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(mcp={"servers": [{
        "name": "local", "source": "test-server", "args": ["unchanged"],
    }]}))
    validate_agent(path, registry=registry)
    assert created == []
    agent = load_agent(path, registry=registry)
    assert agent.mcp_managers == created
    assert created[0].arguments["auto_connect"] is False
    assert created[0].arguments["server_args"] == ["unchanged"]
    assert agent.tool_registry.has_tool("DelegateProbe")
    agent.close()
    assert created[0].closed


def test_disabled_mcp_does_not_require_environment(tmp_path):
    registry, models = registry_with_model()
    path = write_config(tmp_path, document(mcp={"enabled": False, "servers": [{
        "name": "local", "source": "test-server",
        "env": {"TOKEN": {"env": "MISSING_DISABLED_MCP_TOKEN"}},
    }]}))
    agent = load_agent(path, registry=registry)
    assert agent.mcp_managers == []
    agent.close()


def test_declared_group_conflicts_fail_before_model_creation(tmp_path):
    registry, models = registry_with_model()
    registry.register_tools(
        "override", config_model=Options, register=lambda r, o, c: None,
        tool_names=("FileRead",),
    )
    with pytest.raises(ConfigurationError, match="conflicting"):
        load_agent(write_config(tmp_path, document(tools=[
            {"name": "filesystem"}, {"name": "override"},
        ])), registry=registry)
    assert models == []


def test_prompt_without_default_rules_still_includes_explicit_identity(tmp_path):
    registry, models = registry_with_model()
    agent = load_agent(write_config(tmp_path, document(prompt={
        "text": "Custom identity only", "config": {"include_defaults": False},
    })), registry=registry)
    try:
        assert "Custom identity only" in agent.get_enhanced_prompt()
    finally:
        agent.close()


def test_policy_file_is_checked_before_model_and_used_by_engine(tmp_path):
    registry, models = registry_with_model()
    policy = tmp_path / "policy.json"
    policy.write_text('{"permissions": {"deny": ["DelegateProbe"]}}')
    path = write_config(tmp_path, document(permissions={"config": {"policy_paths": ["./policy.json"]}}))
    agent = load_agent(path, registry=registry)
    try:
        agent.add_tool(ProbeTool())
        request = agent.tool_registry.build_permission_request(
            "DelegateProbe", {"value": "hello"}, context=agent.build_authorization_context(),
        )
        assert agent.permission_engine.authorize(request).behavior == PermissionBehavior.DENY
    finally:
        agent.close()
    policy.write_text("invalid-secret-JSON")
    count = len(models)
    with pytest.raises(ConfigurationError) as error:
        load_agent(path, registry=registry)
    assert "invalid-secret-JSON" not in str(error.value)
    assert len(models) == count


def test_invalid_custom_prepare_is_redacted(tmp_path):
    registry, models = registry_with_model()

    def prepare(options, paths):
        raise ValueError("secret-prepare-exception")

    registry.register_permission("x", config_model=Options, factory=lambda o, c: None, prepare=prepare)
    with pytest.raises(ConfigurationError) as error:
        validate_agent(write_config(tmp_path, document(permissions={"implementation": "x"})), registry=registry)
    assert "secret-prepare-exception" not in str(error.value)
    assert models == []


def test_model_temperature_is_default_invocation_temperature(tmp_path):
    registry, models = registry_with_model()
    agent = load_agent(write_config(tmp_path, document()), registry=registry)
    assert agent.config.temperature == models[0].temperature
    agent.close()
    agent = load_agent(write_config(tmp_path, document(agent={"config": {"temperature": 0.1}})), registry=registry)
    assert agent.config.temperature == 0.1
    agent.close()


def test_shipped_configs_validate_and_coding_config_loads_offline(tmp_path, monkeypatch):
    import shutil
    directory = Path(__file__).resolve().parents[1] / "example" / "configuration"
    monkeypatch.setenv("OPENAI_API_KEY", "offline-validation-only")
    for path in directory.glob("*.yaml"):
        validate_agent(path)
    copied = tmp_path / "examples"
    shutil.copytree(directory, copied)
    data = yaml.safe_load((copied / "coding.yaml").read_text())
    data["model"] = {"implementation": "offline"}
    data["agent"]["workspace"] = "."
    registry, models = registry_with_model()
    agent = load_agent(write_config(copied, data), registry=registry)
    try:
        assert agent.plan is not None
        assert agent.multi_agent is not None
        assert "review" in agent.skill_manager.skill_names
        assert agent.invoke("hello") == "done"
    finally:
        agent.close()
