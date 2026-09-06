"""Offline checks for model-facing built-in instructions and reminders."""

import ast
import asyncio
import importlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent import BasicAgent
from agent.components.prompt_composer import SystemPromptComposer
from context.compressor.history import LLMHistoryCompactor
from core.history import coerce_canonical_message
from core.llm import EasyLLM
from Tool.builtin.filesystem import FileReadTool
from Tool.builtin import memorytool
from Tool.builtin.mcp_tool import _build_mcp_guidance


ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    f"Tool.builtin.{path.stem}"
    for path in (ROOT / "Tool/builtin").glob("*.py")
    if path.stem != "__init__"
) + ["Tool.claude_compat.models", "plan.tools", "skill.tool"]


def assert_english(text):
    assert not re.search(r"[\u4e00-\u9fff]", text), text


@pytest.mark.parametrize("module_name", MODULES)
def test_builtin_prompt_constants_and_parameter_schemas_are_english(module_name):
    module = importlib.import_module(module_name)
    for name, value in vars(module).items():
        if name.endswith("PROMPT") and isinstance(value, str):
            assert_english(value)
        if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module_name:
            assert_english(json.dumps(value.model_json_schema(), ensure_ascii=False))
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in {"description", "guidance", "prompt"}:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                assert_english(node.value.value)


@pytest.mark.parametrize("tool_class", [
    memorytool.AddMemoryTool, memorytool.SearchMemoryTool, memorytool.GetMemoryTool,
    memorytool.RemoveMemoryTool, memorytool.UpdateMemoryTool, memorytool.MemoryMaintenanceTool,
])
def test_generated_memory_tool_schemas_are_english(tool_class):
    tool = tool_class(SimpleNamespace(get_supported_type=lambda: ["working"]))
    assert_english(json.dumps(tool.get_openai_schema(), ensure_ascii=False))


@pytest.mark.parametrize("kind", ["tool", "resource_list", "resource_read", "prompt_get"])
def test_generated_mcp_guidance_is_english_but_remote_content_is_preserved(kind):
    guidance = _build_mcp_guidance("", {
        "readOnlyHint": True, "destructiveHint": True,
        "openWorldHint": True, "idempotentHint": False,
    }, capability_kind=kind)
    assert_english(guidance)
    remote = "远程服务的原始说明"
    assert _build_mcp_guidance(remote, {}, capability_kind=kind).startswith(remote)


def test_file_read_reminders_do_not_change_file_data(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("print('hello')\n", encoding="utf-8")
    tool = FileReadTool(workspace_root=str(tmp_path))
    result = tool.run({"file_path": str(path)})
    assert result.structured_data["content"] == "print('hello')"
    assert "<system-reminder>" in result.to_display_string()
    assert "provide analysis of malware" in result.to_display_string()
    assert path.read_text() == "print('hello')\n"
    shorter = tool.run({"file_path": str(path), "offset": 10})
    assert "shorter than the provided offset (10)" in shorter.content
    assert "empty" not in shorter.content
    empty = tmp_path / "empty"
    empty.touch()
    assert "contents are empty" in tool.run({"file_path": str(empty)}).content
    clipped = FileReadTool(workspace_root=str(tmp_path), max_output_chars=5)
    clipped_result = clipped.run({"file_path": str(path)})
    assert clipped_result.structured_data["truncated"]
    assert "Use FileRead with offset and limit" in clipped_result.content
    assert clipped_result.content.endswith("</system-reminder>")


class StubLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"


def test_plan_reminders_follow_state_and_custom_messages_survive():
    agent = BasicAgent("planner", StubLLM()).with_plan()
    try:
        agent.plan.enter(allowed_actions=["FileRead"])
        message = agent.plan._build_enter_message()
        assert "### Phase 5" in message
        assert message.endswith("Explicitly allowed actions: FileRead\n</system-reminder>")
        context = agent.build_prompt_context()
        composer = SystemPromptComposer(include_defaults=False)
        assert {block.name for block in composer.build_capability_blocks(context)} == {"plan_mode"}
        agent.clear_history()
        assert "plan_mode" in {block.name for block in composer.build_capability_blocks(agent.build_prompt_context())}
        agent.plan.exit()
        assert "plan_mode" not in {block.name for block in composer.build_capability_blocks(agent.build_prompt_context())}
        assert "## Exited Plan Mode" in agent.plan.config.exit_message
    finally:
        agent.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_english_compaction_prompt_and_continuation_keep_json_protocol(asynchronous):
    class SummaryLLM:
        def invoke(self, messages):
            self.messages = messages
            return '["Current work: update the prompt.", "Pending: run tests."]'

        async def ainvoke(self, messages):
            return self.invoke(messages)

    llm = SummaryLLM()
    compactor = LLMHistoryCompactor(llm=llm, language="en")
    history = [{
        "record_type": "canonical_message", "role": "user",
        "content": [{"type": "text", "text": "Update the prompt."}],
    }]
    result = asyncio.run(compactor.acompact(history)) if asynchronous else compactor.compact(history)
    assert len(result) == 2
    assert result[0]["metadata"]["compacted_summary"]
    first = coerce_canonical_message(result[0]).text_content()
    second = coerce_canonical_message(result[1]).text_content()
    assert "<system-reminder>" in first
    assert "Current work: update the prompt." in first
    assert second == "Pending: run tests."
    assert_english(llm.messages[0]["content"])
    assert "JSON array of strings" in llm.messages[0]["content"]
    assert "9. Optional Next Step" in llm.messages[0]["content"]
    assert "Do NOT call any tools" in llm.messages[0]["content"]
