"""MCP tool integration tests."""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.mcp_tool import MCPToolManager


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
            },
        ]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        if tool_name == "echo":
            return arguments.get("text", "")
        if tool_name == "sum_two":
            return arguments.get("a", 0) + arguments.get("b", 0)
        return None


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

        self.assertEqual(len(wrapped), 2)
        self.assertIn("mcp_echo", self.registry.tools)
        self.assertIn("mcp_sum_two", self.registry.tools)

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
        self.assertEqual(len(tools), 2)

        manager.close()
        self.assertFalse(self.fake_client.connected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
