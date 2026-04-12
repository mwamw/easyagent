"""MCP tool integration tests."""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.mcp_tool import MCPToolManager
from skill.builtin.mcp_skill import MCPSkill
from skill.manager import SkillManager
from skill.meta_tools import SkillTool
from skill.registry import SkillRegistry


class FakeMCPClient:
    def __init__(self):
        self.connected = False
        self.called = []

    def is_connected(self):
        return self.connected

    async def connect(self):
        self.connected = True

    async def disconnect(self, exc_type=None, exc_val=None, exc_tb=None):
        self.connected = False

    async def list_tools(self):
        return [
            {
                "name": "echo",
                "description": "Echo input text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "text to echo"}
                    },
                    "required": ["text"],
                },
                "annotations": {
                    "readOnlyHint": True,
                    "idempotentHint": True,
                },
            },
            {
                "name": "sum_two",
                "description": "Sum two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "first number"},
                        "b": {"type": "integer", "description": "second number"},
                    },
                    "required": ["a", "b"],
                },
                "annotations": {
                    "readOnlyHint": True,
                    "idempotentHint": True,
                },
            },
            {
                "name": "remote_write",
                "description": "Write remote state",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "payload": {"type": "string", "description": "write payload"}
                    },
                    "required": ["payload"],
                },
                "annotations": {
                    "destructiveHint": True,
                    "openWorldHint": True,
                    "idempotentHint": False,
                },
            },
        ]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        if tool_name == "echo":
            return arguments.get("text", "")
        if tool_name == "sum_two":
            return arguments.get("a", 0) + arguments.get("b", 0)
        if tool_name == "remote_write":
            return f"wrote:{arguments.get('payload', '')}"
        return None

    async def list_resources(self):
        return [
            {
                "uri": "memo://alpha",
                "name": "alpha",
                "description": "Alpha memo",
                "mime_type": "text/plain",
                "annotations": {},
            }
        ]

    async def read_resource(self, uri):
        self.called.append(("read_resource", {"uri": uri}))
        if uri == "memo://alpha":
            return "alpha body"
        return ""

    async def list_prompts(self):
        return [
            {
                "name": "review_notes",
                "description": "生成代码评审备注",
                "arguments": [
                    {"name": "language", "description": "语言", "required": False}
                ],
            }
        ]

    async def get_prompt(self, prompt_name, arguments):
        self.called.append(("get_prompt", {"prompt_name": prompt_name, "arguments": arguments}))
        return [
            {
                "role": "user",
                "content": f"请生成 {arguments.get('language', '中文')} 评审意见",
            }
        ]


class TestMCPToolIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.fake_client = FakeMCPClient()

    def test_register_remote_tools(self):
        manager = MCPToolManager(
            server_source="unused",
            client=self.fake_client,
            tool_prefix="mcp_",
        )

        wrapped = manager.register_to_registry(self.registry)

        self.assertEqual(len(wrapped), 3)
        self.assertIn("mcp_echo", self.registry.tools)
        self.assertIn("mcp_sum_two", self.registry.tools)
        self.assertIn("mcp_remote_write", self.registry.tools)

    def test_execute_wrapped_tool_via_registry(self):
        manager = MCPToolManager(server_source="unused", client=self.fake_client)
        manager.register_to_registry(self.registry)

        result = self.registry.executeTool("echo", {"text": "hello"})
        self.assertEqual(result, "hello")
        self.assertEqual(self.fake_client.called[-1][0], "echo")

        result2 = self.registry.executeTool("sum_two", {"a": 2, "b": 5})
        self.assertEqual(result2, "7")
        self.assertEqual(self.fake_client.called[-1], ("sum_two", {"a": 2, "b": 5}))

    def test_parameter_validation(self):
        manager = MCPToolManager(server_source="unused", client=self.fake_client)
        manager.register_to_registry(self.registry)

        with self.assertRaises(ValueError):
            self.registry.executeTool("sum_two", {"a": 2})

    def test_manual_connect_mode(self):
        manager = MCPToolManager(
            server_source="unused",
            client=self.fake_client,
            auto_connect=False,
        )

        with self.assertRaises(RuntimeError):
            manager.list_remote_tools()

        manager.connect()
        tools = manager.list_remote_tools()
        self.assertEqual(len(tools), 3)

        manager.close()
        self.assertFalse(self.fake_client.connected)

    def test_manual_connect_then_register_to_registry(self):
        manager = MCPToolManager(
            server_source="unused",
            client=self.fake_client,
            auto_connect=False,
            include_resources=True,
            resource_tool_prefix="mcp_",
        )

        manager.connect()
        snapshot = manager.snapshot(refresh=True)
        self.assertTrue(snapshot.tools)

        wrapped = manager.register_to_registry(self.registry)
        self.assertTrue(wrapped)
        self.assertIn("echo", self.registry.get_tool_names())
        self.assertIn("mcp_list_mcp_resources", self.registry.get_tool_names())

        manager.close()
        self.assertFalse(self.fake_client.connected)

    def test_register_resource_tools(self):
        manager = MCPToolManager(server_source="unused", client=self.fake_client)
        manager.register_to_registry(self.registry, include_resources=True, resource_tool_prefix="mcp_")

        self.assertIn("mcp_list_mcp_resources", self.registry.tools)
        self.assertIn("mcp_read_mcp_resource", self.registry.tools)

        listing = self.registry.executeTool("mcp_list_mcp_resources", {})
        self.assertIn("memo://alpha", listing)

        content = self.registry.executeTool("mcp_read_mcp_resource", {"uri": "memo://alpha"})
        self.assertEqual(content, "alpha body")

    def test_mcp_annotations_map_to_tool_spec(self):
        manager = MCPToolManager(server_source="unused", client=self.fake_client, tool_prefix="mcp_")
        manager.register_to_registry(self.registry)

        read_spec = self.registry.get_tool_spec("mcp_echo")
        assert read_spec is not None
        self.assertTrue(read_spec.read_only)
        self.assertTrue(read_spec.supports_parallel)
        self.assertFalse(read_spec.requires_confirmation)
        self.assertTrue(read_spec.metadata["mcp_read_only"])
        self.assertIn("只读", read_spec.build_schema_description())

        write_spec = self.registry.get_tool_spec("mcp_remote_write")
        assert write_spec is not None
        self.assertTrue(write_spec.destructive)
        self.assertTrue(write_spec.requires_confirmation)
        self.assertFalse(write_spec.supports_parallel)
        self.assertTrue(write_spec.metadata["mcp_open_world"])
        self.assertIn("远程副作用", write_spec.build_schema_description())
        self.assertIn("外部世界", write_spec.build_schema_description())

    def test_register_prompt_skills(self):
        manager = MCPToolManager(server_source="unused", client=self.fake_client)
        registry = SkillRegistry()

        names = manager.register_prompt_skills(registry, skill_prefix="mcp_unused_")
        self.assertEqual(names, ["mcp_unused_review_notes"])
        self.assertTrue(registry.has("mcp_unused_review_notes"))

        manifest = registry.get_manifest("mcp_unused_review_notes")
        self.assertEqual(manifest.source_type, "mcp_prompt")
        self.assertEqual(manifest.exposure_mode, "on_demand")

        skill = registry.create("mcp_unused_review_notes", prompt_arguments={"language": "英文"})
        body = skill.get_body_prompt()
        self.assertIn("MCP Prompt: review_notes", body)
        self.assertIn("英文", body)

    def test_skill_tool_passes_arguments_to_mcp_prompt_skill(self):
        registry = SkillRegistry()
        manager = MCPToolManager(server_source="unused", client=self.fake_client)
        manager.register_prompt_skills(registry, skill_prefix="mcp_unused_")

        skill_manager = SkillManager()
        tool = SkillTool(registry, skill_manager, set())
        result = tool.run(
            {
                "skill_name": "mcp_unused_review_notes",
                "skill_arguments": {"language": "英文"},
            }
        )

        self.assertEqual(result.status, "success")
        prompt_call = self.fake_client.called[-1]
        self.assertEqual(prompt_call[0], "get_prompt")
        self.assertEqual(prompt_call[1]["arguments"], {"language": "英文"})
        runtime_prompt = skill_manager.build_runtime_skill_context_prompt()
        self.assertIn("英文", runtime_prompt)

    def test_mcp_skill_registers_and_cleans_prompt_skills(self):
        registry = SkillRegistry()
        skill = MCPSkill(
            server_source="unused",
            auto_connect=True,
            include_resource_tools=False,
            prompt_registry=registry,
            register_prompt_skills=True,
        )
        skill._manager = MCPToolManager(server_source="unused", client=self.fake_client)

        skill.on_activate(agent=None)
        registered = registry.list_available_names()
        self.assertIn("mcp_unused_review_notes", registered)

        skill.on_deactivate(agent=None)
        self.assertNotIn("mcp_unused_review_notes", registry.list_available_names())


if __name__ == "__main__":
    unittest.main(verbosity=2)
