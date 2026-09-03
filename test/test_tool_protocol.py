import os
import sys

from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry


class EchoParams(BaseModel):
    text: str


class EmptyParams(BaseModel):
    pass


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="echo_tool",
            description="返回输入内容",
            parameters=EchoParams,
            guidance="仅在需要回显文本时使用。",
            prompt="回显工具只适合验证文本传递，不适合做外部查询。",
            read_only=True,
            tags=["echo"],
        )

    def run(self, parameters: dict):
        return f"echo:{parameters['text']}"


class JsonTool(Tool):
    def __init__(self):
        super().__init__(
            name="json_tool",
            description="返回结构化 JSON",
            parameters=EmptyParams,
            output_mode="json",
            prompt="JSON 工具返回结构化对象；需要时先读取 structured_data。",
        )

    def run(self, parameters: dict):
        return {"ok": True, "value": 1}


class ConfirmTool(Tool):
    def __init__(self):
        super().__init__(
            name="confirm_tool",
            description="需要确认后才能执行",
            parameters=EmptyParams,
            requires_confirmation=True,
            destructive=True,
        )
        self.run_count = 0

    def run(self, parameters: dict):
        self.run_count += 1
        return "should not run"


def test_tool_spec_and_openai_schema():
    tool = EchoTool()
    spec = tool.get_spec()

    assert spec.name == "echo_tool"
    assert spec.guidance == "仅在需要回显文本时使用。"
    assert spec.prompt == "回显工具只适合验证文本传递，不适合做外部查询。"
    assert spec.read_only is True
    assert spec.tags == ["echo"]

    schema = tool.get_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo_tool"
    assert "仅在需要回显文本时使用。" in schema["function"]["description"]
    assert "回显工具只适合验证文本传递" in schema["function"]["description"]
    assert schema["function"]["parameters"]["properties"]["text"]["type"] == "string"


def test_registry_execute_tool_result_and_string_compat():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    registry.register_tool(JsonTool())

    echo_result = registry.execute_tool_result("echo_tool", {"text": "hello"})
    assert isinstance(echo_result, ToolResult)
    assert echo_result.status == "success"
    assert echo_result.to_display_string() == "echo:hello"

    json_result = registry.execute_tool_result("json_tool", {})
    assert json_result.status == "success"
    assert json_result.structured_data == {"ok": True, "value": 1}
    assert '"ok": true' in json_result.to_display_string()

    # 旧接口仍返回字符串
    assert registry.execute_tool("echo_tool", {"text": "legacy"}) == "echo:legacy"


def test_registry_tool_visibility_and_runtime_cleanup():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    registry.mount_runtime_tool(JsonTool())

    assert registry.get_tool_visibility("echo_tool") == "resident"
    assert registry.get_tool_visibility("json_tool") == "runtime"
    assert [spec.name for spec in registry.list_tool_specs(scope="resident")] == ["echo_tool"]
    assert [spec.name for spec in registry.list_tool_specs(scope="runtime")] == ["json_tool"]

    registry.clear_runtime_tools()
    assert registry.has_tool("echo_tool")
    assert not registry.has_tool("json_tool")


def test_registry_requires_confirmation_short_circuits_execution():
    registry = ToolRegistry()
    tool = ConfirmTool()
    registry.register_tool(tool)

    result = registry.execute_tool_result("confirm_tool", {})
    assert result.status == "needs_confirmation"
    assert "需要用户确认" in result.to_display_string()
    assert tool.run_count == 0
