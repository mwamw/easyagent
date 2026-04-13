import os
import sys

from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.BaseTool import Tool
from Tool.claude_compat import (
    CLAUDE_TOOL_MODELS,
    CLAUDE_TOOL_ORDER,
    ClaudeCompatDelegatingTool,
    get_claude_tool_definition,
)
from Tool.claude_compat.models import ClaudeBashInput


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="echo_tool",
            description="Echo text",
            parameters=EchoParams,
            guidance="Echo only.",
            read_only=True,
            source="builtin",
            tags=["echo"],
        )

    def run(self, parameters: dict):
        return f"echo:{parameters['text']}"


def test_claude_catalog_exposes_expected_models():
    assert CLAUDE_TOOL_ORDER[0] == "Agent"
    assert "Bash" in CLAUDE_TOOL_MODELS
    assert CLAUDE_TOOL_MODELS["Bash"] is ClaudeBashInput

    definition = get_claude_tool_definition("FileRead")
    assert definition.name == "FileRead"
    assert definition.read_only is True
    assert "filesystem" in definition.tags


def test_claude_compat_delegating_tool_preserves_delegate_behavior():
    delegate = EchoTool()
    compat_tool = ClaudeCompatDelegatingTool(
        claude_name="Bash",
        delegate=delegate,
        parameters=EchoParams,
        description="Compat wrapper",
    )

    result = compat_tool.run({"text": "hello"})
    spec = compat_tool.get_spec()

    assert result == "echo:hello"
    assert compat_tool.name == "Bash"
    assert spec.source == "claude_compat"
    assert spec.metadata["compat_layer"] == "claude_code"
    assert spec.metadata["delegate_tool_name"] == "echo_tool"
    assert "claude_compat" in spec.tags
