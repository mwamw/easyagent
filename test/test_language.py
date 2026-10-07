"""Language contracts: no paid models, imports cannot execute Python."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from easyagent import BasicAgent
from easyagent.language import LanguageError, compile_file, parse
from configuration import AgentBuilder, Options
from test_configuration import registry_with_model
from test_permission_boundaries import CountingWriteTool, OfflineLLM, run_path


def source(tmp_path: Path, text: str, name: str = "main.ea") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = 'model Main using offline {} agent Assistant { model = Main }'


def test_compilation_is_pure_build_uses_existing_agent(tmp_path):
    registry, models = registry_with_model()
    program = compile_file(source(tmp_path, MINIMAL), registry=registry)
    assert models == []
    assert program.select()[0] == "Assistant"
    assert program.validate_environment() == ("model",)
    assert models == []
    agent = program.build()
    try:
        assert type(agent) is BasicAgent
        assert agent.name == "Assistant"
        assert agent.runtime.workspace_root == str(tmp_path)
        assert agent.invoke("hello") == "done"
    finally:
        agent.close()
    assert models[0].closed


@pytest.mark.parametrize("invocation", ["invoke", "ainvoke", "stream", "astream"])
def test_all_invocation_paths_share_runtime(tmp_path, invocation):
    registry, models = registry_with_model(OfflineLLM)
    registry.register_tools("probe", config_model=Options,
                            register=lambda r, o, c: r.register_tool(CountingWriteTool()))
    path = source(tmp_path, '''
        model Main using offline {}
        agent Assistant {
            model = Main
            tools { probe() }
            permission using rule_based { approval_mode = "full_access" }
            context { max_tokens = 16000 }
        }
    ''')
    agent = compile_file(path, registry=registry).build()
    try:
        run_path(agent, invocation)
        assert agent.tool_registry.get_tool("WriteProbe").calls == 1
    finally:
        agent.close()


def test_templates_imports_aliases_defaults_and_path_ownership(tmp_path, monkeypatch):
    registry, models = registry_with_model()
    source(tmp_path, '''
        model Main using offline {}
        define Harness(workspace: Path, model: Model, budget: Integer = 24000) -> Agent {
            agent {
                workspace = workspace
                model = model
                context { max_tokens = budget }
                session using sqlite { path = path("./state/session.db") }
            }
        }
    ''', "shared/templates.ea")
    path = source(tmp_path, '''
        import { Main as ImportedModel, Harness } from "./shared/templates.ea"
        agent One = Harness(workspace = path("."), model = ImportedModel)
        agent Two = Harness(workspace = path("."), model = ImportedModel, budget = 32000)
        run Two with terminal { streaming = false }
    ''')
    monkeypatch.chdir(tmp_path.parent)
    program = compile_file(path, registry=registry)
    assert program.select()[0] == "Two"
    one = program.configuration("One", environ={})
    assert one.context.config["max_tokens"] == 24000
    assert one.session.config["path"] == str(tmp_path / "shared/state/session.db")
    assert not (tmp_path / "shared/state").exists()
    first, second = program.build("One"), program.build("Two")
    try:
        assert first.runtime is not second.runtime
        assert first.context_manager is not second.context_manager
        assert first.llm is not second.llm
        first.context_manager.append_query("first only")
        assert not second.context_manager.canonical
    finally:
        first.close()
        second.close()


def test_env_is_late_bound_and_explain_contains_no_values(tmp_path, monkeypatch):
    monkeypatch.delenv("EA_TEST_SECRET", raising=False)
    path = source(tmp_path, '''
        model Main { model = "local" api_key = env("EA_TEST_SECRET") }
        agent Assistant { model = Main }
    ''')
    program = compile_file(path)
    preview = json.dumps(program.explain())
    assert "EA_TEST_SECRET" not in preview and "api_key" not in preview
    with pytest.raises(LanguageError, match="EA_TEST_SECRET"):
        program.configuration(environ={})
    assert program.configuration(environ={"EA_TEST_SECRET": "secret1"}).model.config["api_key"] == "secret1"
    assert program.configuration(environ={"EA_TEST_SECRET": "secret2"}).model.config["api_key"] == "secret2"


def test_env_defaults_deferred_paths_and_strict_runtime_binding(tmp_path):
    path = source(tmp_path, '''
        model Main { model = env("EA_MODEL", default = "fallback") }
        permission P { approval_mode = env("EA_MODE") }
        agent A { model = Main permission = P workspace = path(env("EA_WORKSPACE")) }
    ''')
    program = compile_file(path)
    config = program.configuration(environ={"EA_MODE": "full_access", "EA_WORKSPACE": "child"})
    assert config.model.config["model"] == "fallback"
    assert config.agent.workspace == str(tmp_path / "child")
    with pytest.raises(LanguageError) as caught:
        program.configuration(environ={"EA_MODE": "private-invalid-value", "EA_WORKSPACE": "."})
    assert "private-invalid-value" not in str(caught.value)


@pytest.mark.parametrize("text, message", [
    ('model M {} agent A { model = Unknown }', "Unknown name"),
    ('permission P {} agent A { model = P }', "requires model"),
    ('model M {} model M {}', "Duplicate"),
    ('model M { model = "a" model = "b" }', "Duplicate field"),
    ('model M = N model N = M', "Cyclic"),
    ('model M {} agent A { model = M context { max_tokens = "12" } }', "Integer"),
    ('model M {} agent A { model = M context { max_tokens = true } }', "Integer"),
    ('model M {} agent A { model = M context { max_tokens = -1 } }', "invalid value"),
    ('model M {} agent A { model = M context { max_token = 12 } }', "Unknown configuration"),
    ('model M {} agent A { model = M context = none }', "requires context"),
    ('agent A {}', "requires a model"),
    ('model M {} agent A { model = M workspace = "." }', "workspace requires"),
    ('model M {} agent A { model = M invented = true }', "Unknown Agent field"),
    ('model M {} agent A { model = M tools { unknown() } }', "Unknown tool"),
    ('model M {} agent A { model = M tools { calculator() calculator() } }', "Duplicate"),
    ('model M {} agent A { model = M skills {} }', "at least one"),
    ('model M {} agent A { model = M skills { directory(".") } }', "Expected Path"),
    ('model M {} agent A { model = M } run A with imaginary {}', "Unknown UI"),
    ('model M {} agent A { model = M } run A with terminal { max_iter = "2" }', "Integer"),
    ('model M {} agent A { model = M } run A run A', "Only one run"),
    ('model M {} run M', "Expected agent"),
    ('define X(a: Strange) -> Agent { agent {} }', "Unknown parameter type"),
    ('define X() -> Model { X() }', "Recursive"),
    ('define X() -> Model { Y() } define Y() -> Model { X() }', "Recursive"),
    ('define X(a: Integer) -> Context { context { max_tokens = a } } context C = X(a = "wrong")', "Expected Integer"),
    ('define X(a: Integer) -> Context { context { max_tokens = a } } context C = X()', "Missing argument"),
    ('define X(a: Integer) -> Context { context { max_tokens = a } } context C = X(b = 2)', "Missing argument"),
    ('define X() -> Model { context {} }', "Expected model"),
    ('model M { api_key = env("MISSING") timeout = -1 }', "invalid value"),
    ('context C { max_tokens = env("NUMBER") }', "Integer"),
    ('define Bad(size: Boolean) -> Context { context { max_tokens = size } }', "Integer"),
    ('define Bad(flag: Integer) -> Prompt { prompt { include_defaults = flag } }', "Boolean"),
    ('model M {} agent A { model = M prompt { text = "x" file = path("x") } }', "text or file"),
    ('model M {} agent A { model = M mcp = {servers = [{name="x", source="x", surprise=true}]} }', "Unknown configuration field"),
    ('model M {} agent A {model=M config={cache_policy={modex="wrong"}}}', "Unknown configuration field"),
])
def test_semantic_errors_are_source_located(tmp_path, text, message):
    with pytest.raises(LanguageError, match=message) as caught:
        compile_file(source(tmp_path, text))
    assert caught.value.span.file == tmp_path / "main.ea"
    assert caught.value.span.line > 0 and caught.value.span.column > 0


@pytest.mark.parametrize("text", [
    'model M { key = __import__("os") }',
    'model M { key = "unterminated }',
    'model M { key = "\\q" }',
    'model M { key = "\\ud800" }',
    'model M { key = 1e999 }',
    'model M { key = [1, 2 }',
    'model M { key = os.system("true") }',
    'model M { key = "x" + "y" }',
])
def test_no_python_or_unsafe_literal_execution(tmp_path, text):
    with pytest.raises(LanguageError):
        compile_file(source(tmp_path, text))


def test_local_import_errors_cycles_and_no_hidden_execution(tmp_path):
    path = source(tmp_path, 'import { M } from "./second.ea" model N = M')
    source(tmp_path, 'import { N } from "./main.ea" model M = N', "second.ea")
    with pytest.raises(LanguageError, match="Cyclic source import"):
        compile_file(path)
    path = source(tmp_path, 'import { M } from "https://example.com/x.ea"')
    with pytest.raises(LanguageError, match="Network imports"):
        compile_file(path)
    path = source(tmp_path, 'import { M } from "./script.py"')
    with pytest.raises(LanguageError, match="local .ea"):
        compile_file(path)


def test_triple_string_comments_unicode_and_optional_separators(tmp_path):
    registry, _ = registry_with_model()
    path = source(tmp_path, '''
        # Source comments
        model Main using offline {}; // another comment
        agent A {
            model = Main,
            prompt { text = """你是研究助手。
保持清晰。"""; include_defaults = false }
        }
    ''')
    config = compile_file(path, registry=registry).configuration()
    assert config.prompt.text == "你是研究助手。\n保持清晰。"
    assert config.prompt.config["include_defaults"] is False


def test_disabled_optional_features_do_not_require_implementation_or_env(tmp_path):
    path = source(tmp_path, '''
        model M {}
        agent A {
            model = M memory = none permission = none skills = none mcp = none
            tools { disabled unavailable(api_key = env("EA_UNSET")) }
        }
    ''')
    config = compile_file(path).configuration(environ={})
    assert config.memory is None and config.permissions is None
    assert not config.tools[0].enabled


def test_every_declaration_and_unused_template_is_checked(tmp_path):
    with pytest.raises(LanguageError, match="Unknown"):
        compile_file(source(tmp_path, 'model M {} define Unused(x: Model) -> Agent { agent { model = x contxt = 1 } }'))


def test_select_requires_unambiguous_agent(tmp_path):
    program = compile_file(source(tmp_path, 'model M {} agent A {model=M} agent B {model=M}'))
    with pytest.raises(LanguageError, match="Select an Agent"):
        program.select()
    assert program.select("B")[0] == "B"


def test_builder_document_api_does_not_mutate_recipe(tmp_path):
    registry, _ = registry_with_model()
    config = compile_file(source(tmp_path, MINIMAL), registry=registry).configuration()
    before = config.model_dump()
    builder = AgentBuilder(registry)
    builder.validate_document(config, source=tmp_path / "main.ea")
    agent = builder.build_document(config, source=tmp_path / "main.ea")
    agent.close()
    assert config.model_dump() == before


def test_cli_check_explain_schema_and_sanitized_errors(tmp_path, capsys):
    from language.__main__ import main
    path = source(tmp_path, 'model M {api_key="never-display-this"} agent A {model=M}')
    assert main(["check", str(path)]) == 0
    assert main(["explain", str(path)]) == 0
    assert "never-display-this" not in capsys.readouterr().out
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["language_version"] == 1
    path = source(tmp_path, 'context C {max_tokens="never-display-this"}')
    assert main(["check", str(path)]) == 2
    assert "never-display-this" not in capsys.readouterr().err


def test_default_source_examples_compile_without_credentials(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "example/language"
    for key in ("EASYAGENT_MODEL", "EASYAGENT_API_KEY", "EASYAGENT_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    for name in ("minimal.ea", "coding.ea", "research.ea"):
        assert compile_file(directory / name).agents


def test_imported_run_does_not_select_entrypoint_or_start_ui(tmp_path):
    source(tmp_path, 'model M {} agent Imported {model=M} run Imported with terminal {}', "library.ea")
    path = source(tmp_path, 'import { M } from "./library.ea" agent Local {model=M}')
    program = compile_file(path)
    assert program.launch is None and program.select()[0] == "Local"


def test_custom_component_schema_and_factory_are_shared_with_python_api(tmp_path):
    from permissions import BasePermissionEngine, PermissionDecision, PermissionBehavior

    class PolicyOptions(Options):
        label: str

    class Policy(BasePermissionEngine):
        def __init__(self, label):
            self.label = label

        def authorize(self, request):
            return PermissionDecision(PermissionBehavior.ALLOW, reason=self.label)

    registry, models = registry_with_model()
    policies = []

    def factory(options, context):
        policy = Policy(options.label)
        policies.append(policy)
        return policy

    registry.register_permission("custom", config_model=PolicyOptions, factory=factory)
    path = source(tmp_path, MINIMAL.replace('model = Main', 'model = Main permission using custom {label="domain policy"}'))
    program = compile_file(path, registry=registry)
    assert not policies and not models
    agent = program.build()
    try:
        assert agent.permission_engine is policies[0]
        assert agent.permission_engine.label == "domain policy"
    finally:
        agent.close()


def test_disabled_mcp_does_not_bind_secrets(tmp_path):
    program = compile_file(source(tmp_path, '''
        model M {}
        agent A { model=M mcp={enabled=false servers=[{
            name="external", source="tool", env={API_KEY=env("ABSENT")}
        }]} }
    '''))
    assert not program.configuration(environ={}).mcp.enabled
    assert program.explain()["agents"]["A"]["mcp"] is False


def test_static_check_never_calls_custom_prepare_or_factory(tmp_path):
    from context import ContextManager
    registry, models = registry_with_model()
    calls = []

    def prepare(options, paths):
        calls.append("prepare")
        return options

    registry.register_context("custom", config_model=Options,
                              factory=lambda o, c: calls.append("factory") or ContextManager(), prepare=prepare)
    path = source(tmp_path, MINIMAL.replace('model = Main', 'model = Main context using custom {}'))
    program = compile_file(path, registry=registry)
    assert not calls and not models
    program.validate_environment()
    assert calls == ["prepare"] and not models
    agent = program.build()
    agent.close()
    assert calls == ["prepare", "prepare", "factory"]


def test_ui_run_uses_public_agent_and_cleans_owned_resources(tmp_path):
    from language.__main__ import run_program
    from ui import BaseAgentUI
    from ui.configuration import UIRegistry

    registry, models = registry_with_model()
    records = []

    class ProbeUI(BaseAgentUI):
        async def run(self, agent):
            records.append(agent)
            assert not agent.runtime.closed

        async def aclose(self):
            records.append("closed")

    uis = UIRegistry()
    uis.register("probe", config_model=Options, factory=lambda o: ProbeUI())
    program = compile_file(source(tmp_path, MINIMAL + " run Assistant with probe {}"), registry=registry, ui_registry=uis)
    assert not records
    asyncio.run(run_program(program, None, None))
    assert records[0].runtime.closed
    assert records[1] == "closed"
    assert models[0].closed


def test_invalid_environment_failure_precedes_model_factory(tmp_path):
    registry, models = registry_with_model()
    path = source(tmp_path, MINIMAL.replace('model = Main', 'model = Main workspace=path("missing")'))
    program = compile_file(path, registry=registry)
    with pytest.raises(LanguageError, match="Workspace directory does not exist"):
        program.build()
    assert not models


def test_parser_depth_and_import_budget_fail_cleanly(tmp_path):
    with pytest.raises(LanguageError):
        parse('model M { provider_options=' + '[' * 1200 + '0' + ']' * 1200 + '}')
    for index in range(34):
        text = f'import {{ M }} from "./f{index + 1}.ea"' if index < 33 else 'model M {}'
        source(tmp_path, text, f"f{index}.ea")
    with pytest.raises(LanguageError, match="limit"):
        compile_file(tmp_path / "f0.ea")


def test_builder_errors_point_to_imported_component_field(tmp_path):
    registry, _ = registry_with_model()
    source(tmp_path, 'prompt Instructions {\n file = path("./absent.md")\n}', "library.ea")
    path = source(tmp_path, 'import { Instructions } from "./library.ea" ' +
                  MINIMAL.replace("model = Main", "model = Main prompt = Instructions"))
    program = compile_file(path, registry=registry)
    with pytest.raises(LanguageError) as caught:
        program.build()
    assert caught.value.span.file == tmp_path / "library.ea"
    assert caught.value.span.line == 2


@pytest.mark.parametrize("body", [
    'import { X } from "\\u0000.ea"',
    'model M {} agent A { model=M workspace=path("\\u0000") }',
    'model M { api_key=env("invalid\\nname") }',
])
def test_invalid_paths_and_environment_names_have_safe_diagnostics(tmp_path, body):
    with pytest.raises(LanguageError) as caught:
        compile_file(source(tmp_path, body))
    assert "\x00" not in str(caught.value) and "\n" not in str(caught.value)


def test_lexer_limits_and_missing_import_names(tmp_path):
    with pytest.raises(LanguageError, match="million"):
        parse(" " * 2_000_001)
    source(tmp_path, 'model M {}', "library.ea")
    with pytest.raises(LanguageError, match="does not exist"):
        compile_file(source(tmp_path, 'import { Missing } from "./library.ea"'))


def test_editor_extension_assets_are_valid_and_offline():
    directory = Path(__file__).resolve().parents[1] / "tooling/vscode"
    package = json.loads((directory / "package.json").read_text())
    assert "main" not in package and "dependencies" not in package
    assert package["contributes"]["languages"][0]["extensions"] == [".ea"]
    for path in directory.rglob("*.json"):
        assert isinstance(json.loads(path.read_text()), dict)


def test_query_runner_uses_ainvoke_and_closes_agent(tmp_path, capsys):
    from language.__main__ import run_program
    from test_multiagent_module import DummyLLM

    class AsyncLLM(DummyLLM):
        async def ainvoke_raw(self, messages, **kwargs):
            return self.invoke_raw(messages, **kwargs)

    registry, models = registry_with_model(AsyncLLM)
    program = compile_file(source(tmp_path, MINIMAL), registry=registry)
    asyncio.run(run_program(program, None, "hello"))
    assert capsys.readouterr().out.strip() == "done"
    assert models[0].closed


def test_prompt_tool_skill_session_mcp_and_optional_modules_lower_correctly(tmp_path):
    program = compile_file(source(tmp_path, '''
        model M {}
        prompt Rules { text="Instructions" include_defaults=false }
        memory Memory { max_capacity=20 }
        agent A {
            model=M prompt=Rules memory=Memory
            hooks {} callbacks {} codeintel {} worktree {storage_dir=path("./worktrees")}
            tasks {} plan {register_tools=false}
            session using sqlite {path=path("./sessions.db")}
            observability using sqlite {path=path("./traces.db")}
            multi_agent {max_background_tasks=2}
            skills {directory(path("./skills"))}
            mcp={servers=[{name="server",source="python",args=["server.py"]}]}
            tools {bash(sandbox={enabled=true,network="deny"})}
        }
    '''))
    config = program.configuration(environ={})
    assert config.memory.config["max_capacity"] == 20
    assert config.prompt.text == "Instructions" and not config.prompt.config["include_defaults"]
    for name in ("hooks", "callbacks", "codeintel", "worktree", "tasks", "plan", "multi_agent"):
        assert getattr(config, name).enabled
    assert config.session.config["path"] == str(tmp_path / "sessions.db")
    assert config.mcp.servers[0].args == ["server.py"]
    assert config.skills.directories == [str(tmp_path / "skills")]
    assert config.tools[0].config["sandbox"]["enabled"] is True
    assert not (tmp_path / "sessions.db").exists()


def test_open_provider_mappings_and_nested_dataclass_schema(tmp_path):
    program = compile_file(source(tmp_path, '''
        model M {provider_options={custom_feature={foo=42}}}
        agent A {model=M config={cache_policy={mode="read_only", enabled=true}}}
    '''))
    config = program.configuration(environ={})
    assert config.agent.config["cache_policy"]["mode"] == "read_only"
    assert config.model.config["provider_options"]["custom_feature"]["foo"] == 42


def test_unresolved_env_never_hides_missing_required_fields(tmp_path):
    registry, _ = registry_with_model()

    class CustomOptions(Options):
        api_key: str
        mandatory: str

    registry.register_installer("memory", "custom", config_model=CustomOptions,
                                install=lambda a, o, c: None)
    with pytest.raises(LanguageError, match="mandatory: required field"):
        compile_file(source(tmp_path, 'memory M using custom {api_key=env("KEY")}'), registry=registry)


@pytest.mark.parametrize("field_name", ["size", "budget"])
def test_registered_configuration_aliases_and_paths_are_supported(tmp_path, field_name):
    from pydantic import ConfigDict, Field
    registry, _ = registry_with_model()

    class CustomOptions(Options):
        model_config = ConfigDict(populate_by_name=True, extra="forbid")
        budget: int = Field(alias="size", gt=0)
        directory: Path

    registry.register_installer("memory", "custom", config_model=CustomOptions,
                                install=lambda a, o, c: None)
    path = source(tmp_path, f'memory M using custom {{{field_name}=12 directory=path(".")}}')
    assert compile_file(path, registry=registry)


def test_compiled_agent_ask_uses_existing_interruption_and_resume(tmp_path):
    from core.Exception import ToolInterruption
    from easyagent.ui import InteractionReply

    registry, _ = registry_with_model(OfflineLLM)
    registry.register_tools("probe", config_model=Options,
                            register=lambda r, o, c: r.register_tool(CountingWriteTool()))
    path = source(tmp_path, MINIMAL.replace("model = Main", '''
        model = Main tools {probe()}
        permission using rule_based {approval_mode="ask_for_approval"}
    '''))
    agent = compile_file(path, registry=registry).build()

    async def run():
        try:
            with pytest.raises(ToolInterruption):
                await agent.ainvoke("do it")
            pending = agent.get_pending_interruption()
            assert pending is not None
            assert agent.tool_registry.get_tool("WriteProbe").calls == 0
            await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "approve"))
            events = [event async for event in agent.astream_continue()]
            assert any(event.type.value == "final" and event.content == "done" for event in events)
            assert agent.tool_registry.get_tool("WriteProbe").calls == 1
        finally:
            await agent.aclose()
    asyncio.run(run())


def test_source_ui_is_closed_if_agent_build_fails(tmp_path):
    from language.__main__ import run_program
    from ui import BaseAgentUI
    from ui.configuration import UIRegistry

    registry, models = registry_with_model()
    records = []

    class TestUI(BaseAgentUI):
        async def run(self, agent):
            raise AssertionError("A failed build must not enter the UI")

        async def aclose(self):
            records.append("closed")

    ui_registry = UIRegistry()
    ui_registry.register("test", config_model=Options, factory=lambda o: TestUI())
    path = source(tmp_path, MINIMAL.replace("model = Main", 'model = Main workspace=path("missing")') +
                  " run Assistant with test {}")
    program = compile_file(path, registry=registry, ui_registry=ui_registry)
    with pytest.raises(LanguageError):
        asyncio.run(run_program(program, None, None))
    assert not models and records == ["closed"]
