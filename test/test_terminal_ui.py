"""Headless renderer and real prompt-toolkit pipe-input interaction tests."""
import asyncio
from io import StringIO
from pathlib import Path
import sys
from threading import Event

import pytest

pytest.importorskip("rich")
pytest.importorskip("prompt_toolkit")
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from agent import BasicAgent
from runtime import AgentStreamEvent, AgentStreamEventType
from ui import BaseAgentUI
from ui.configuration import UIRegistry, load_ui
from ui.terminal import TerminalUI
from ui.terminal.rendering import TerminalRenderer, SafeMarkdown, safe_text
from configuration import Options, load_agent
from test_configuration import registry_with_model, write_config, document
from test_permission_boundaries import OfflineLLM, CountingWriteTool


def event(kind, content="", **data):
    return AgentStreamEvent(AgentStreamEventType(kind), "test", 1, content=content, data=data)


def test_renderer_does_not_duplicate_final_or_tool_events_or_reveal_reasoning():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, width=80, color_system=None))
    renderer.start()
    renderer.event(event("reasoning_delta", "private reasoning"))
    renderer.event(event("text_delta", "Final answer"))
    renderer.event(event("final", "Final answer"))
    renderer.event(event("tool_call", tool_name="Probe", tool_call_id="1", arguments={}))
    renderer.event(event("tool_result", "complete result", tool_name="Probe", tool_call_id="1", status="success"))
    renderer.tool_result("1", "Probe", "success", "complete result")
    renderer.finish()
    text = output.getvalue()
    assert text.count("Final answer") == 1
    assert "private reasoning" not in text
    assert text.count("✓ Probe") == 1
    assert "complete result" not in text
    renderer.details()
    assert "complete result" in output.getvalue()


def test_final_without_deltas_and_error_events_are_not_lost_or_duplicated():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output))
    renderer.event(event("final", "answer without streaming chunks"))
    renderer.event(event("error", "secret provider error"))
    renderer.finish()
    assert output.getvalue().count("answer without streaming chunks") == 1
    assert "secret provider error" not in output.getvalue()


def test_terminal_control_sequences_are_filtered_but_multiline_text_survives():
    assert safe_text("a\x1b[2Jb\x1b]52;c;Yg==\x07\nc") == "ab\nc"
    assert safe_text("[bold]not markup[/bold]") == "[bold]not markup[/bold]"


def test_markdown_plugin_failure_degrades_to_text(monkeypatch):
    def fail(*args, **kwargs):
        raise ModuleNotFoundError("broken optional lexer plugin")
    monkeypatch.setattr("ui.terminal.rendering.Markdown", fail)
    output = StringIO()
    Console(file=output).print(SafeMarkdown("complete answer"))
    assert output.getvalue().count("complete answer") == 1


def test_stream_preview_does_not_parse_partial_markdown(monkeypatch):
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, force_terminal=True))
    renderer.start()
    try:
        def fail(*args, **kwargs):
            raise AssertionError("Markdown must not be parsed for live chunks")
        monkeypatch.setattr("ui.terminal.rendering.Markdown", fail)
        renderer.text = "Partial code fence"
        renderer.refresh()
    finally:
        renderer.finish()


async def wait_until(predicate):
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("UI did not reach the expected state")


@pytest.mark.parametrize("streaming", [True, False])
def test_ui_round_trip_borrows_agent_and_accepts_exit(streaming):
    async def run():
        output = StringIO()
        agent = BasicAgent("terminal", OfflineLLM())  # Provider asks for an absent tool; loop reports its error.
        agent.add_tool(CountingWriteTool())
        with create_pipe_input() as pipe:
            ui = TerminalUI(console=Console(file=output), input=pipe, output=DummyOutput(), streaming=streaming)
            task = asyncio.create_task(ui.run(agent))
            await asyncio.sleep(0.03)
            pipe.send_text("hello\r")
            await wait_until(lambda: agent.llm._provider.calls >= 2 and ui._task is None)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 3)
            assert "done" in output.getvalue()
            assert not agent.runtime.closed
            assert ui._subscription is None
        await agent.aclose()
    asyncio.run(run())


def test_permission_ui_approval_is_explicit_and_resumes_once():
    async def run():
        output = StringIO()
        tool = CountingWriteTool()
        agent = BasicAgent("terminal", OfflineLLM()).add_tool(tool).with_permissions()
        with create_pipe_input() as pipe:
            ui = TerminalUI(console=Console(file=output), input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.run(agent))
            await asyncio.sleep(0.03)
            pipe.send_text("hello\r")
            await wait_until(lambda: agent.get_pending_interruption() is not None and ui._task is None)
            assert tool.calls == 0
            pipe.send_text("\r")
            await asyncio.sleep(0.05)
            assert tool.calls == 0
            pipe.send_text("1\r")
            await wait_until(lambda: agent.llm._provider.calls >= 2 and ui._task is None)
            assert tool.calls == 1
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 3)
            assert output.getvalue().count("done") == 1
        await agent.aclose()
    asyncio.run(run())


def test_ctrl_c_requests_stop_without_detaching_running_sync_tool():
    entered, release = Event(), Event()

    class SlowTool(CountingWriteTool):
        def run(self, parameters):
            entered.set()
            assert release.wait(4)
            return super().run(parameters)

    async def run():
        agent = BasicAgent("terminal", OfflineLLM()).add_tool(SlowTool())
        output = StringIO()
        with create_pipe_input() as pipe:
            ui = TerminalUI(console=Console(file=output), input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.run(agent))
            try:
                await asyncio.sleep(0.03)
                pipe.send_text("hello\r")
                await wait_until(entered.is_set)
                pipe.send_bytes(b"\x03")
                await wait_until(agent.is_stop_requested)
                assert agent.runtime.invocation_lock.locked()
                assert ui._task is not None
                release.set()
                await wait_until(lambda: ui._task is None)
                pipe.send_text("/exit\r")
                await asyncio.wait_for(task, 3)
                assert "已停止后续执行" in output.getvalue()
            finally:
                release.set()
                await ui.aclose()
                await agent.aclose()
    asyncio.run(run())


def test_renderer_failure_waits_for_active_tool_cleanup(monkeypatch):
    entered, release = Event(), Event()

    class SlowTool(CountingWriteTool):
        def run(self, parameters):
            entered.set()
            assert release.wait(4)
            return super().run(parameters)

    async def run():
        agent = BasicAgent("terminal", OfflineLLM()).add_tool(SlowTool())
        output = StringIO()
        with create_pipe_input() as pipe:
            ui = TerminalUI(console=Console(file=output), input=pipe, output=DummyOutput())

            def fail_refresh():
                if entered.is_set():
                    raise RuntimeError("display failure")

            monkeypatch.setattr(ui.renderer, "refresh", fail_refresh)
            task = asyncio.create_task(ui.run(agent))
            try:
                await asyncio.sleep(0.03)
                pipe.send_text("hello\r")
                await wait_until(agent.is_stop_requested)
                assert agent.runtime.invocation_lock.locked()
                assert ui._task is not None
                release.set()
                await wait_until(lambda: ui._task is None)
                pipe.send_text("/exit\r")
                await asyncio.wait_for(task, 3)
                assert "RuntimeError" in output.getvalue()
                assert not agent.runtime.invocation_lock.locked()
            finally:
                release.set()
                await ui.aclose()
                await agent.aclose()
    asyncio.run(run())


def test_custom_ui_registry_and_agent_loading_stay_independent(tmp_path):
    class TestUI(BaseAgentUI):
        async def run(self, agent):
            pass

    class UIOptions(Options):
        label: str

    registry = UIRegistry()
    registry.register("custom", config_model=UIOptions, factory=lambda o: TestUI())
    path = write_config(tmp_path, document(ui={"implementation": "custom", "config": {"label": "demo"}}))
    assert isinstance(load_ui(path, registry=registry), TestUI)
    agent_registry, _ = registry_with_model()
    agent = load_agent(path, registry=agent_registry)
    assert type(agent) is BasicAgent
    agent.close()


def test_contract_import_has_no_terminal_dependency():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "import ui; import sys; assert 'rich' not in sys.modules; assert 'prompt_toolkit' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_clear_is_display_only():
    output = StringIO()
    ui = TerminalUI(console=Console(file=output))
    agent = BasicAgent("terminal", OfflineLLM())
    agent.context_manager.append_query("preserve this")
    before = list(agent.context_manager.canonical)
    ui._command("/clear")
    assert agent.context_manager.canonical == before
    agent.close()
