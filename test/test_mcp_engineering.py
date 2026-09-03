"""MCP engineering contract tests."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import BasicAgent
from core.llm import EasyLLM
from core.permissions import PermissionBehavior, PermissionContext, PermissionRule
from Emcp import MCPPolicyContext, MCPPolicyRule
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.mcp_tool import MCPToolManager, register_mcp_tools


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256

    def invoke(self, messages, temperature=None, **kwargs):
        return "mock-response"

    def prepare_messages_for_request(self, messages):
        return list(messages)


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
            }
        ]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, dict(arguments)))
        if tool_name == "echo":
            return arguments.get("text", "")
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
        return "alpha body"

    async def list_prompts(self):
        return [
            {
                "name": "review_notes",
                "description": "生成代码评审备注",
                "arguments": [],
            }
        ]

    async def get_prompt(self, prompt_name, arguments=None):
        self.called.append(("get_prompt", {"prompt_name": prompt_name, "arguments": arguments or {}}))
        return [{"role": "user", "content": "review"}]


class TestMCPEngineering(unittest.TestCase):
    def test_runtime_export_includes_policy_cache_and_connection_state(self):
        client = FakeMCPClient()
        policy = MCPPolicyContext(
            capability_cache_ttl_seconds=120,
            resource_cache_ttl_seconds=60,
            rules=[
                MCPPolicyRule(
                    effect="deny",
                    server_names=("alpha",),
                    capability_kinds=("prompt_get",),
                    reason="prompt fetch is blocked",
                )
            ],
        )
        manager = MCPToolManager(
            server_source="alpha",
            client=client,
            policy_context=policy,
            include_resources=True,
        )

        tools = manager.list_remote_tools()
        resources = manager.list_remote_resources()
        exported = manager.export_state()

        self.assertEqual(len(tools), 1)
        self.assertEqual(len(resources), 1)
        self.assertEqual(exported["policy"]["capabilityCacheTtlSeconds"], 120)
        self.assertEqual(exported["policy"]["resourceCacheTtlSeconds"], 60)
        self.assertTrue(exported["capabilitySnapshot"]["tools"])
        self.assertEqual(exported["connection"]["state"]["serverName"], "alpha")

    def test_policy_denial_is_classified(self):
        client = FakeMCPClient()
        manager = MCPToolManager(
            server_source="alpha",
            client=client,
            policy_context=MCPPolicyContext(
                rules=[
                    MCPPolicyRule(
                        effect="deny",
                        server_names=("alpha",),
                        capability_kinds=("prompt_get",),
                        reason="prompt fetch is blocked",
                    )
                ]
            ),
        )

        with self.assertRaises(Exception) as ctx:
            manager.get_remote_prompt("review_notes")

        detail = manager.describe_error(ctx.exception)
        self.assertEqual(detail["errorType"], "mcp_policy_denied")
        self.assertIn("prompt fetch is blocked", detail["message"])

    def test_registry_surfaces_and_source_identifiers_are_registered(self):
        registry = ToolRegistry()
        manager = register_mcp_tools(
            registry,
            server_source="alpha",
            client=FakeMCPClient(),
            include_resources=True,
        )

        surfaces = registry.list_runtime_surfaces("mcp_manager")
        self.assertIn(manager.registry_server_name, surfaces)
        self.assertIs(surfaces[manager.registry_server_name], manager)

        tool_spec = registry.get_tool_spec("echo")
        self.assertIsNotNone(tool_spec)
        self.assertEqual(tool_spec.metadata["source_identifier"], "mcp://alpha/tools/echo")
        self.assertEqual(tool_spec.source, "mcp")
        self.assertEqual(tool_spec.risk_categories, ["mcp"])

        resource_spec = registry.get_tool_spec("alpha_list_mcp_resources")
        self.assertIsNotNone(resource_spec)
        self.assertEqual(resource_spec.metadata["source_identifier"], "mcp://alpha/resources")

    def test_session_restore_rebuilds_mcp_runtime_from_snapshot(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = os.path.join(tempdir, "sessions.db")
            manager = MCPToolManager(
                server_source="alpha",
                client=FakeMCPClient(),
                include_resources=True,
            )
            agent = BasicAgent(
                name="assistant",
                llm=DummyLLM(),
            ).with_mcp(manager)
            session_id = "mcp-phase-f"
            agent.save_session(session_id, store=db_path)

            restored = BasicAgent.load_session(
                session_id,
                llm=DummyLLM(),
                store=db_path,
                mcp_managers=[
                    MCPToolManager(
                        server_source="alpha",
                        client=FakeMCPClient(),
                        include_resources=True,
                    )
                ],
            )

            self.assertIsNotNone(restored.tool_registry)
            self.assertTrue(restored.tool_registry.has_tool("echo"))
            self.assertTrue(restored.tool_registry.has_tool("alpha_list_mcp_resources"))
            result = restored.tool_registry.execute_tool("echo", {"text": "hello"})
            self.assertEqual(result, "hello")

            report = restored.get_last_restore_report()
            self.assertIsNotNone(report)
            self.assertEqual(report["components"]["mcp:alpha"]["status"], "restored")

    def test_permission_rules_can_target_mcp_server(self):
        registry = ToolRegistry()
        register_mcp_tools(
            registry,
            server_source="alpha",
            client=FakeMCPClient(),
        )
        permission_context = PermissionContext(
            rules=[
                PermissionRule(
                    tool_name="echo",
                    behavior=PermissionBehavior.DENY,
                    matcher={"mcp_servers": ["alpha"]},
                    description="alpha server is blocked",
                )
            ]
        )

        result = registry.execute_tool_result(
            "echo",
            {"text": "hello"},
            permission_context=permission_context,
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "permission_denied")
        self.assertIn("alpha server is blocked", result.to_display_string())

    def test_agent_close_reports_mcp_runtime_component(self):
        manager = MCPToolManager(
            server_source="alpha",
            client=FakeMCPClient(),
        )
        agent = BasicAgent(
            name="assistant",
            llm=DummyLLM(),
        ).with_mcp(manager)

        report = agent.close(close_llm=False)

        self.assertEqual(report["components"]["mcp[0]"]["status"], "closed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
