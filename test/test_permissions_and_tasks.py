import asyncio
import os
import sys
import unittest

from pydantic import BaseModel


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import BasicAgent
from core.permissions import (
    PermissionBehavior,
    PermissionContext,
    PermissionEngine,
    PermissionMode,
    PermissionRule,
)
from core.llm import EasyLLM
from plan import PlanModeConfig
from task import InMemoryTaskStore, TaskService, TaskStatus
from Tool import Tool, ToolRegistry
from Tool.builtin import register_task_tools


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


class NoopParams(BaseModel):
    pass


class UrlParams(BaseModel):
    url: str


class ConfirmingTool(Tool):
    def __init__(self):
        super().__init__(
            name="ConfirmingTool",
            description="需要确认的测试工具",
            parameters=NoopParams,
            requires_confirmation=True,
        )

    def run(self, parameters: dict):
        return "ok"


class MutatingTool(Tool):
    def __init__(self):
        super().__init__(
            name="MutatingTool",
            description="高风险写操作工具",
            parameters=NoopParams,
            destructive=True,
            risk_categories=["side_effect"],
        )

    def run(self, parameters: dict):
        return "mutated"


class EditableFileTool(Tool):
    def __init__(self):
        super().__init__(
            name="EditableFileTool",
            description="模拟需要确认的文件写入工具",
            parameters=NoopParams,
            requires_confirmation=True,
            destructive=True,
            risk_categories=["filesystem_write"],
        )

    def run(self, parameters: dict):
        return "edited"


class NetworkTool(Tool):
    def __init__(self):
        super().__init__(
            name="NetworkTool",
            description="模拟网络访问工具",
            parameters=UrlParams,
            requires_confirmation=True,
            risk_categories=["network"],
        )

    def run(self, parameters: dict):
        return parameters["url"]


class PermissionAndTaskTests(unittest.TestCase):
    def setUp(self):
        self.llm = DummyLLM()

    def test_permission_allow_rule_overrides_confirmation_requirement(self):
        registry = ToolRegistry()
        registry.register_tool(ConfirmingTool())
        context = PermissionContext(
            rules=[
                PermissionRule(
                    tool_name="ConfirmingTool",
                    behavior=PermissionBehavior.ALLOW,
                    source="test",
                    description="测试规则允许直接执行",
                )
            ]
        )

        result = registry.execute_tool_result(
            "ConfirmingTool",
            {},
            permission_context=context,
            permission_engine=PermissionEngine(),
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.content, "ok")

    def test_plan_mode_blocks_high_risk_tool(self):
        registry = ToolRegistry()
        registry.register_tool(MutatingTool())
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
        ).with_tool(registry)
        agent.with_plan(config=PlanModeConfig(register_tools=False))
        agent.enter_plan_mode(allowed_actions=["read"])

        result = agent.execute_tool_result("MutatingTool", {})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "permission_denied")
        self.assertIn("plan 模式", result.content)
        self.assertIn("side_effect", result.metadata["risk_categories"])

    def test_async_tool_execution_respects_permission_engine(self):
        registry = ToolRegistry()
        registry.register_tool(MutatingTool())
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
        ).with_tool(registry)
        agent.with_plan(config=PlanModeConfig(register_tools=False))
        agent.enter_plan_mode(allowed_actions=["read"])

        result = agent.execute_tool_result("MutatingTool", {})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "permission_denied")

    def test_permission_store_respects_source_priority(self):
        registry = ToolRegistry()
        registry.register_tool(ConfirmingTool())
        context = PermissionContext()
        context.set_source_rules(
            "session",
            [
                PermissionRule(
                    tool_name="ConfirmingTool",
                    behavior=PermissionBehavior.ALLOW,
                    description="会话层允许执行",
                )
            ],
            priority=50,
        )
        context.set_source_rules(
            "workspace",
            [
                PermissionRule(
                    tool_name="ConfirmingTool",
                    behavior=PermissionBehavior.DENY,
                    description="工作区策略禁止执行",
                )
            ],
            priority=10,
        )

        result = registry.execute_tool_result(
            "ConfirmingTool",
            {},
            permission_context=context,
            permission_engine=PermissionEngine(),
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "permission_denied")
        self.assertEqual(context.rules[0].source, "workspace")

    def test_accept_edits_mode_allows_file_write_tools(self):
        registry = ToolRegistry()
        registry.register_tool(EditableFileTool())
        context = PermissionContext(mode=PermissionMode.ACCEPT_EDITS)

        result = registry.execute_tool_result(
            "EditableFileTool",
            {},
            permission_context=context,
            permission_engine=PermissionEngine(),
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.content, "edited")

    def test_permission_rule_matches_url_hosts(self):
        registry = ToolRegistry()
        registry.register_tool(NetworkTool())
        context = PermissionContext(
            rules=[
                PermissionRule(
                    tool_name="NetworkTool",
                    behavior=PermissionBehavior.ALLOW,
                    matcher={"hosts": ["example.com"]},
                    source="test",
                    description="允许 example.com",
                )
            ]
        )

        allowed = registry.execute_tool_result(
            "NetworkTool",
            {"url": "https://docs.example.com/path"},
            permission_context=context,
            permission_engine=PermissionEngine(),
        )
        blocked = registry.execute_tool_result(
            "NetworkTool",
            {"url": "https://other.invalid/path"},
            permission_context=context,
            permission_engine=PermissionEngine(),
        )

        self.assertEqual(allowed.status, "success")
        self.assertEqual(blocked.status, "needs_confirmation")

    def test_task_tools_crud_flow(self):
        registry = ToolRegistry()
        service = TaskService(InMemoryTaskStore())
        register_task_tools(registry, service=service)

        create_result = registry.execute_tool_result(
            "TaskCreate",
            {
                "title": "Refactor runtime",
                "description": "拆出统一 runtime 内核",
                "owner": "alice",
                "metadata": {"phase": 1},
            },
        )
        task_id = create_result.metadata["task_id"]

        get_result = registry.execute_tool_result("TaskGet", {"task_id": task_id})
        update_result = registry.execute_tool_result(
            "TaskUpdate",
            {
                "task_id": task_id,
                "status": TaskStatus.IN_PROGRESS,
                "metadata": {"module": "runtime"},
            },
        )
        list_result = registry.execute_tool_result(
            "TaskList",
            {"status": TaskStatus.IN_PROGRESS, "owner": "alice"},
        )

        self.assertEqual(create_result.status, "success")
        self.assertIn('"title": "Refactor runtime"', create_result.to_display_string())
        self.assertEqual(get_result.structured_data["title"], "Refactor runtime")
        self.assertIn(f'"task_id": "{task_id}"', get_result.to_display_string())
        self.assertEqual(update_result.structured_data["status"], TaskStatus.IN_PROGRESS)
        self.assertEqual(update_result.structured_data["metadata"]["phase"], 1)
        self.assertEqual(update_result.structured_data["metadata"]["module"], "runtime")
        self.assertIn('"status": "in_progress"', update_result.to_display_string())
        self.assertEqual(list_result.status, "success")
        self.assertEqual(len(list_result.structured_data), 1)
        self.assertEqual(list_result.structured_data[0]["task_id"], task_id)

    def test_task_tools_return_structured_not_found_error(self):
        registry = ToolRegistry()
        service = TaskService(InMemoryTaskStore())
        register_task_tools(registry, service=service)

        result = registry.execute_tool_result("TaskGet", {"task_id": "task_missing"})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "task_not_found")
        self.assertEqual(result.metadata["task_id"], "task_missing")

    def test_agent_auto_registers_task_tools(self):
        registry = ToolRegistry()
        service = TaskService(InMemoryTaskStore())
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
        ).with_tool(registry).with_task_service(service)

        self.assertTrue(agent.tool_registry.has_tool("TaskCreate"))
        self.assertTrue(agent.tool_registry.has_tool("TaskGet"))
        self.assertTrue(agent.tool_registry.has_tool("TaskUpdate"))
        self.assertTrue(agent.tool_registry.has_tool("TaskList"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
