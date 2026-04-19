import json
import os
import sys
from typing import Literal, Optional

from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from core.llm import EasyLLM


class EchoParams(BaseModel):
    text: str


class IsolationParams(BaseModel):
    isolation: Optional[Literal["worktree"]] = None


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="echo_tool",
            description="回显输入。",
            parameters=EchoParams,
            guidance="仅在需要回显输入时调用。",
        )

    def run(self, parameters: dict):
        return parameters["text"]


class IsolationTool(Tool):
    def __init__(self):
        super().__init__(
            name="agent_like_tool",
            description="测试 Gemini schema 清洗。",
            parameters=IsolationParams,
        )

    def run(self, parameters: dict):
        return parameters.get("isolation") or ""


def test_registry_exports_provider_specific_tool_payloads():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())

    openai_tools = registry.export_tools("openai")
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "echo_tool"

    responses_tools = registry.export_tools("openai_responses")
    assert responses_tools[0]["type"] == "function"
    assert responses_tools[0]["name"] == "echo_tool"
    assert "function" not in responses_tools[0]

    anthropic_tools = registry.export_tools("anthropic_native")
    assert anthropic_tools[0]["name"] == "echo_tool"
    assert anthropic_tools[0]["input_schema"]["properties"]["text"]["type"] == "string"

    google_tools = registry.export_tools("google_native")
    assert len(google_tools) == 1
    assert "function_declarations" in google_tools[0]
    declaration = google_tools[0]["function_declarations"][0]
    assert declaration["name"] == "echo_tool"
    assert declaration["parameters"]["type"] == "OBJECT"
    assert declaration["parameters"]["properties"]["text"]["type"] == "STRING"


def test_google_native_adapter_sanitizes_const_to_enum():
    registry = ToolRegistry()
    registry.register_tool(IsolationTool())

    google_tools = registry.export_tools("google_native")
    declaration = google_tools[0]["function_declarations"][0]
    isolation_schema = declaration["parameters"]["properties"]["isolation"]

    assert isolation_schema["enum"] == ["worktree"]
    assert "const" not in isolation_schema
    assert "const" not in json.dumps(declaration, ensure_ascii=False)


def test_easyllm_exports_tools_through_provider_adapter():
    registry = ToolRegistry()
    registry.register_tool(IsolationTool())

    llm = EasyLLM(
        provider="google_native",
        model="gemini-3-flash",
        api_key="test-key",
        client=object(),
    )

    payload = llm.export_tools(registry)
    declaration = payload[0]["function_declarations"][0]

    assert declaration["name"] == "agent_like_tool"
    assert declaration["parameters"]["properties"]["isolation"]["enum"] == ["worktree"]
