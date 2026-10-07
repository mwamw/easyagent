"""Generated apps use the ordinary framework; no paid API calls in this suite."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread

import pytest
import yaml

from application import build_application, plan_application, run_application
from application.__main__ import main
from configuration import ConfigurationError, read_configuration, read_configuration_source
from configuration.builtins import ModelOptions, default_registry
from test_multiagent_module import DummyLLM
from test_permission_boundaries import OfflineLLM, CountingWriteTool
from configuration import Options


def config(tmp_path, **sections):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({
        "model": {"provider": "openai", "model": "offline-model", "api_key": {"env": "APP_TEST_KEY"}},
        **sections,
    }), encoding="utf-8")
    return path


def test_generation_does_not_need_credentials_or_construct_agent(tmp_path, monkeypatch):
    from core.llm import EasyLLM
    monkeypatch.delenv("APP_TEST_KEY", raising=False)
    monkeypatch.setattr(EasyLLM, "__init__", lambda *a, **k: pytest.fail("Build must not create a model"))
    source = config(tmp_path)
    app = build_application(source, output=tmp_path / "app")
    assert (app / "app.py").is_file()
    assert os.access(app / "app.py", os.X_OK)
    assert (app / "state").is_dir()
    raw = read_configuration_source(app / "agent.yaml")
    assert raw["model"]["api_key"] == {"env": "APP_TEST_KEY"}
    assert raw["agent"]["workspace"] == str(tmp_path)
    assert "APP_TEST_KEY=\n" in (app / ".env.example").read_text()
    assert source.read_text().find("workspace") == -1


def test_common_inline_secrets_are_externalized_not_written(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_TEST_KEY", "environment-secret")
    source = config(tmp_path, model={
        "model": "offline", "api_key": "literal-secret",
        "base_url": "https://user:password@host.invalid/v1",
        "default_headers": {"Authorization": "Bearer header-secret"},
    }, mcp={"servers": [{"name": "demo", "source": "server", "env": {"APP_TOKEN": "mcp-secret"}}]})
    app = build_application(source, output=tmp_path / "app")
    written = b"\n".join(p.read_bytes() for p in app.rglob("*") if p.is_file())
    for secret in (b"literal-secret", b"environment-secret", b"user:password", b"header-secret", b"mcp-secret"):
        assert secret not in written
    variables = json.loads((app / "build.json").read_text())["environment_variables"]
    assert len(variables) == 4
    assert all(name + "=\n" in (app / ".env.example").read_text() for name in variables)


def test_prompt_skills_policies_and_fresh_state_are_packaged(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_TEST_KEY", "placeholder")
    (tmp_path / "system.md").write_text("Product instructions")
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: review\ndescription: Review code\n---\nRead first.\n")
    (skill / "check.sh").write_text("#!/bin/sh\ntrue\n")
    (skill / "check.sh").chmod(0o755)
    (skill / ".env").write_text("SECRET=do-not-copy")
    (skill / "history.db").write_text("do-not-copy-history")
    policy = tmp_path / "policies"
    policy.mkdir()
    (policy / "settings.json").write_text('{"permissions":{"deny":["Bash(rm *)"]}}')
    (policy / "unrelated.log").write_text("do-not-copy-log")
    (tmp_path / "old.db").write_text("do-not-copy-state")
    source = config(tmp_path, prompt={"file": "system.md"}, skills={"directories": ["skills"]},
                    permissions={"config": {"policy_directory": "policies"}},
                    session={"implementation": "sqlite", "config": {"path": "old.db"}}, multi_agent={})
    app = build_application(source, output=tmp_path / "app")
    doc = read_configuration(app / "agent.yaml")
    assert (app / doc.prompt.file).read_text() == "Product instructions"
    copied = app / doc.skills.directories[0] / "review"
    assert os.access(copied / "check.sh", os.X_OK)
    assert not (copied / ".env").exists() and not (copied / "history.db").exists()
    assert (app / doc.permissions.config["policy_directory"] / "settings.json").is_file()
    assert not list(app.rglob("unrelated.log"))
    assert doc.session.config["path"] == "state/session/old.db"
    assert not (app / doc.session.config["path"]).exists()
    assert doc.multi_agent.config["storage_dir"] == "state/multi_agent/data"


def test_empty_policy_directory_is_preserved(tmp_path):
    (tmp_path / "policies").mkdir()
    app = build_application(config(tmp_path, permissions={"config": {"policy_directory": "policies"}}), output=tmp_path / "app")
    assert (app / "assets/permissions").is_dir()


def test_paths_preserve_external_execution_boundaries(tmp_path):
    source = config(tmp_path, agent={"workspace": "project", "allowed_roots": ["data"]},
                    tools=[{"name": "bash", "config": {"cwd": "project/subdir", "sandbox": {
                        "enabled": True, "allow_write": ["project"], "deny_read": ["secrets"]}}}],
                    session={"implementation": "sqlite", "config": {"path": str(tmp_path / "external.db")}},
                    mcp={"servers": [{"name": "external", "source": "server.py", "source_type": "file"}]})
    app = build_application(source, output=tmp_path / "app")
    raw = read_configuration_source(app / "agent.yaml")
    assert raw["agent"]["workspace"] == str(tmp_path / "project")
    assert raw["agent"]["allowed_roots"] == [str(tmp_path / "data")]
    sandbox = raw["tools"][0]["config"]["sandbox"]
    assert sandbox["deny_read"] == [str(tmp_path / "secrets")]
    assert raw["session"]["config"]["path"] == str(tmp_path / "external.db")
    assert raw["mcp"]["servers"][0]["source"] == str(tmp_path / "server.py")
    assert not (app / "project").exists()


@pytest.mark.parametrize("changes", [
    {"context": {"config": {"max_tokens": -1}}},
    {"model": {"provider": "openai", "api_key": {"env": "ABSENT"}, "timeout": -1}},
    {"model": {"implementation": "custom"}},
    {"tools": [{"name": "unknown"}]},
    {"tools": [{"name": "calculator"}, {"name": "calculator"}]},
    {"skills": {"directories": ["missing"]}},
    {"prompt": {"file": "missing"}},
    {"prompt": {"file": {"env": "PROMPT_FILE"}}},
    {"context": {"enabled": False}},
    {"agent": {"name": {"env": "APP_NAME"}}},
    {"ui": {"implementation": "custom"}},
    {"ui": {"config": {"max_iter": -1}}},
])
def test_invalid_source_does_not_leave_output(tmp_path, changes):
    with pytest.raises(ConfigurationError):
        build_application(config(tmp_path, **changes), output=tmp_path / "app")
    assert not (tmp_path / "app").exists()


@pytest.mark.parametrize("text", [
    "model: {}\nmodel: {}", "model: &cycle {config: *cycle}",
    "model: !!python/object/apply:os.system ['false']", "[]",
])
def test_source_loader_keeps_yaml_safety(tmp_path, text):
    source = tmp_path / "bad.yaml"
    source.write_text(text)
    with pytest.raises(ConfigurationError):
        build_application(source, output=tmp_path / "app")
    assert not (tmp_path / "app").exists()


def test_optional_null_disabled_and_dynamic_config_values(tmp_path):
    source = config(tmp_path, skills=None, permissions={"config": {"approval_mode": {"env": "MODE"}}},
                    memory={"enabled": False, "implementation": "missing", "config": {"token": {"env": "MISSING"}}},
                    tools=[{"name": "missing", "enabled": False}], ui={"enabled": False})
    app = build_application(source, output=tmp_path / "app")
    assert "ui" not in (app / "requirements.txt").read_text().splitlines()[-1]
    assert read_configuration_source(app / "agent.yaml")["permissions"]["config"]["approval_mode"] == {"env": "MODE"}
    assert "MISSING=" not in (app / ".env.example").read_text()


def test_existing_outputs_and_symlink_assets_are_not_overwritten(tmp_path):
    source = config(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data")
    with pytest.raises(ConfigurationError, match="already exists"):
        build_application(source, output=output)
    assert marker.read_text() == "user data"
    (tmp_path / "prompt.md").symlink_to(marker)
    with pytest.raises(ConfigurationError, match="Symlink"):
        build_application(config(tmp_path, prompt={"file": "prompt.md"}), output=tmp_path / "app")


def test_batch_globs_check_and_collision_handling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config(tmp_path)
    (tmp_path / "other.yaml").write_text("model: {model: offline}\n")
    assert main(["--file", "*.yaml", "--output", "apps", "--check"]) == 0
    assert not (tmp_path / "apps").exists()
    assert main(["--file", "*.yaml", "--output", "apps"]) == 0
    assert (tmp_path / "apps/agent-app/app.py").is_file()
    assert (tmp_path / "apps/other-app/app.py").is_file()
    assert main(["--file", "*.yaml", "--output", "apps"]) == 2
    assert main(["--file", "missing*.yaml"]) == 2


def test_batch_preflight_failure_writes_nothing(tmp_path):
    first = config(tmp_path)
    second = tmp_path / "bad.yaml"
    second.write_text("model: {}\ntools: [{name: missing}]\n")
    assert main(["--file", str(first), str(second), "--output", str(tmp_path / "apps")]) == 2
    assert not (tmp_path / "apps").exists()


def test_packaging_failure_cleans_only_its_own_files(tmp_path, monkeypatch):
    from application.builder import write_application
    plan = plan_application(config(tmp_path))
    original = Path.open

    def broken(path, *args, **kwargs):
        if path.name == "build.json" and args == ("xb",):
            raise OSError("simulated disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken)
    with pytest.raises(OSError):
        write_application(plan, tmp_path / "app")
    assert not (tmp_path / "app").exists()


def test_application_runner_reuses_public_agent_and_closes_resources(tmp_path, monkeypatch, capsys):
    class AsyncLLM(DummyLLM):
        async def ainvoke_raw(self, messages, **kwargs):
            return self.invoke_raw(messages, **kwargs)

    models = []
    registry = default_registry()

    def factory(options, paths):
        model = AsyncLLM()
        original = model.close
        model.was_closed = False
        def close():
            model.was_closed = True
            original()
        model.close = close
        models.append(model)
        return model

    registry.models["default"] = (ModelOptions, factory)
    app = build_application(config(tmp_path), output=tmp_path / "app")
    monkeypatch.setenv("APP_TEST_KEY", "test")
    assert run_application(app, ["--check"], registry=registry) == 0
    assert not models
    assert run_application(app, ["--query", "hello"], registry=registry) == 0
    assert "done" in capsys.readouterr().out
    assert models[0].was_closed


def test_one_shot_ask_never_executes_unapproved_tool(tmp_path, monkeypatch, capsys):
    app = build_application(config(tmp_path, permissions={}), output=tmp_path / "app")
    # Developer launchers may explicitly register additional tools after generation.
    raw = read_configuration_source(app / "agent.yaml")
    raw["tools"] = [{"name": "probe"}]
    (app / "agent.yaml").write_text(yaml.safe_dump(raw))
    tool = CountingWriteTool()
    registry = default_registry()
    registry.models["default"] = (ModelOptions, lambda options, paths: OfflineLLM())
    registry.register_tools("probe", config_model=Options,
                            register=lambda tools, options, context: tools.register_tool(tool))
    monkeypatch.setenv("APP_TEST_KEY", "test")
    assert run_application(app, ["--query", "write"], registry=registry) == 3
    assert tool.calls == 0
    assert "User interaction required" in capsys.readouterr().err


def test_interactive_launcher_passes_agent_to_ui_and_closes_both(tmp_path, monkeypatch):
    from ui import BaseAgentUI
    from ui.configuration import UIRegistry

    class RecordingUI(BaseAgentUI):
        closed = False
        async def run(self, agent):
            self.agent = agent
            self.answer = await agent.ainvoke("hello")

        async def aclose(self):
            self.closed = True

    class Model(OfflineLLM):
        closed = False
        def close(self):
            self.closed = True
            return super().close()

    model = Model()
    model._provider.calls = 1  # Final response, without requesting a tool.
    registry = default_registry()
    registry.models["default"] = (ModelOptions, lambda options, paths: model)
    ui = RecordingUI()
    uis = UIRegistry()
    uis.register("terminal", config_model=Options, factory=lambda options: ui)
    app = build_application(config(tmp_path), output=tmp_path / "app")
    monkeypatch.setenv("APP_TEST_KEY", "test")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert run_application(app, [], registry=registry, ui_registry=uis) == 0
    assert ui.answer == "done"
    assert ui.closed and model.closed


def test_dotenv_is_app_local_process_env_wins_and_workspace_override(tmp_path, monkeypatch):
    from application.runner import application_configuration
    app = build_application(config(tmp_path), output=tmp_path / "app")
    (app / ".env").write_text("APP_TEST_KEY=app-file-value\n")
    (tmp_path / ".env").write_text("APP_TEST_KEY=wrong-parent-value\n")
    monkeypatch.delenv("APP_TEST_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    _, doc = application_configuration(app, workspace=tmp_path)
    assert doc.model.config["api_key"] == "app-file-value"
    assert "APP_TEST_KEY" not in os.environ
    monkeypatch.setenv("APP_TEST_KEY", "process-value")
    _, doc = application_configuration(app)
    assert doc.model.config["api_key"] == "process-value"


def test_multiple_apps_do_not_share_dotenv_values(tmp_path, monkeypatch):
    from application.runner import application_configuration
    source = config(tmp_path)
    first = build_application(source, output=tmp_path / "one")
    second = build_application(source, output=tmp_path / "two")
    (first / ".env").write_text("APP_TEST_KEY=one-value\n")
    (second / ".env").write_text("APP_TEST_KEY=two-value\n")
    monkeypatch.delenv("APP_TEST_KEY", raising=False)
    assert application_configuration(first)[1].model.config["api_key"] == "one-value"
    assert application_configuration(second)[1].model.config["api_key"] == "two-value"
    assert "APP_TEST_KEY" not in os.environ


def test_generated_cli_runs_after_app_move_from_unrelated_cwd(tmp_path):
    calls = []

    class ModelServer(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"id": "offline", "object": "chat.completion", "created": 1,
                               "model": "offline-model", "choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant", "content": "local generated app works"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = config(tmp_path, model={"model": "offline-model", "api_key": {"env": "APP_TEST_KEY"},
                                       "base_url": f"http://127.0.0.1:{server.server_port}/v1"})
        app = build_application(source, output=tmp_path / "app")
        moved = tmp_path / "moved"
        shutil.move(app, moved)
        env = {**os.environ, "APP_TEST_KEY": "local-test-only"}
        checked = subprocess.run([sys.executable, str(moved / "app.py"), "--check"],
                                 cwd=tmp_path.parent, env=env, capture_output=True, text=True, timeout=30)
        assert checked.returncode == 0, checked.stderr
        assert not calls
        queried = subprocess.run([sys.executable, str(moved / "app.py"), "--query", "hello"],
                                 cwd=tmp_path.parent, env=env, capture_output=True, text=True, timeout=30)
        assert queried.returncode == 0, queried.stderr
        assert "local generated app works" in queried.stdout
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
