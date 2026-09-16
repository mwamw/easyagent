"""Keep the README's public-API examples runnable without external services."""

import ast
import asyncio
import inspect
from pathlib import Path
import re
from types import ModuleType
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import pytest

import easyagent
from test_provider_usage_metrics import _openai_stream_chunk
from test_session_persistence import EchoTool, SessionLLM, SessionProvider

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SNIPPETS = dict(re.findall(
    r"<!-- readme-example: ([a-z-]+) -->\s*```python\n(.*?)\n```",
    README.read_text(), flags=re.S,
))


def load_example(name, monkeypatch):
    module = ModuleType(f"readme_example_{name.replace('-', '_')}")
    import sys
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(SNIPPETS[name], f"README:{name}", "exec"), module.__dict__)
    return module


class StreamingProvider(SessionProvider):
    def stream_raw(self, request):
        yield _openai_stream_chunk(content="session-response", finish_reason="stop")

    async def async_stream_raw(self, request):
        async def events():
            for event in self.stream_raw(request):
                yield event
        return events()


def fake_llm():
    llm = SessionLLM()
    llm._provider = StreamingProvider()
    return llm


def test_readme_links_examples_and_project_metadata():
    text = README.read_text()
    assert len(SNIPPETS) == 7
    assert len(re.findall(r"```python\n", text)) == len(SNIPPETS)
    for snippet in SNIPPETS.values():
        parsed = ast.parse(snippet)
        for node in ast.walk(parsed):
            if isinstance(node, ast.ImportFrom):
                assert node.module == "pathlib" or node.module.startswith("easyagent")
    anchors = {
        re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        for title in re.findall(r"^#{1,6} (.+)$", text, flags=re.M)
    }
    for link in re.findall(r"!?\[[^\]\n]+\]\(([^)\n]+)\)", text):
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path:
            path = ROOT / unquote(url.path)
            assert path.exists(), link
            if path.suffix == ".svg":
                assert ET.parse(path).getroot().get("viewBox")
        elif url.fragment:
            assert unquote(url.fragment) in anchors, link
    for removed in ("BaseContextSource", "ContextSource", "MemoryManage", "ConversationStore"):
        assert removed not in text
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["description"] == (
        "A composable Python framework for building custom Agent Harnesses "
        "with replaceable runtime components."
    )
    assert project["version"] == easyagent.__version__


def test_quick_start_uses_environment_and_closes_agent(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "EASYAGENT_BASE_URL": "http://127.0.0.1:9/v1",
        "EASYAGENT_API_KEY": "fake-readme-key",
        "EASYAGENT_MODEL": "readme-model",
    }.items():
        monkeypatch.setenv(name, value)
    signature = inspect.signature(easyagent.EasyLLM)
    llm = fake_llm()

    def construct(**kwargs):
        signature.bind(**kwargs)
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "readme-model"
        assert kwargs["api_key"] == "fake-readme-key"
        assert kwargs["base_url"] == "http://127.0.0.1:9/v1"
        return llm

    monkeypatch.setattr(easyagent, "EasyLLM", construct)
    example = load_example("quick-start", monkeypatch)
    assert "session-response" in capsys.readouterr().out
    assert llm._provider.closed
    for optional in ("memory", "skill_manager", "multi_agent", "observability", "tool_registry"):
        assert getattr(example.agent, optional) is None
    assert not (tmp_path / ".easyagent").exists()


def test_product_assembly_installs_context_and_observability(monkeypatch, tmp_path):
    llm = fake_llm()
    example = load_example("product-assembly", monkeypatch)
    agent = example.build_product_agent(llm, tmp_path)
    try:
        assert agent.context_manager.budget.max_tokens == 16000
        assert agent.context_manager.history_compactor is not None
        assert agent.tool_registry.has_tool("FileRead")
        assert agent.observability is not None
        assert agent.execution_context.workspace_root == str(tmp_path)
        assert agent.invoke("Explain the installed modules") == "session-response"
        assert agent.observability.list()
    finally:
        agent.close()


@pytest.mark.parametrize("mode", ["run_sync", "run_async", "print_stream", "print_async_stream"])
def test_invocation_examples(mode, monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    example = load_example("invocation", monkeypatch)
    agent = easyagent.BasicAgent("readme", fake_llm())
    try:
        function = getattr(example, mode)
        result = function(agent, "Hello")
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        if mode.startswith("run_"):
            assert result == "session-response"
        else:
            output = capsys.readouterr().out
            assert "text_delta" in output and "final" in output
    finally:
        agent.close()


def test_memory_and_prompt_examples_integrate_without_manual_registration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    memory = load_example("custom-memory", monkeypatch)
    prompt = load_example("custom-prompt", monkeypatch)
    agent = easyagent.BasicAgent("research", fake_llm())
    try:
        assert memory.install_research_memory(agent, [EchoTool()]) is agent
        assert prompt.install_research_prompt(agent) is agent
        assert agent.tool_registry.has_tool("Echo")
        assert agent.memory.services.scope.session_id == agent.session_id
        blocks = agent.get_system_prompt_blocks("Find prior evidence")
        assert {block.name for block in blocks} == {"identity", "memory.research"}
        assert agent.tool_registry.execute_tool_result("Echo", {"text": "evidence"}).content == "evidence"
        assert agent.invoke("Hello") == "session-response"
    finally:
        agent.close()


def test_code_agent_registers_deferred_shell_without_executing_it(monkeypatch, tmp_path):
    example = load_example("code-agent", monkeypatch)
    agent = example.build_code_agent(fake_llm(), tmp_path)
    try:
        assert agent.config.tool_schema_mode == "deferred"
        assert agent.tool_registry.has_tool("Bash")
        assert agent.tool_registry.has_tool("tool_schema_tool")
        assert agent.tool_registry.get_tool_spec("Bash").expose_in_deferred is False
        assert agent.tool_registry.get_tool_spec("FileRead").expose_in_deferred is True
    finally:
        agent.close()


def test_session_example_forks_without_closing_shared_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    example = load_example("session", monkeypatch)
    llm = fake_llm()
    try:
        session_id, branch_id, report = example.demonstrate_session(llm)
        assert session_id != branch_id
        assert report["status"] == "restored"
        assert llm._provider.closed is False
    finally:
        llm.close()
