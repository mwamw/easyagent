import os
import sys
import tempfile
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    AgentTool,
    register_send_message_tool,
    register_team_create_tool,
    register_team_delete_tool,
)
from Tool.runtime import SubagentRequest
from runtime import AgentRuntimeManager, ExecutionContext, TeamManager


class FakeSubagent:
    def __init__(self, response_prefix: str = "handled", *, delay_s: float = 0.0):
        self.response_prefix = response_prefix
        self.delay_s = delay_s
        self.trace_history = [{"type": "tool_call", "tool_name": "WebSearch"}]

    def invoke(self, prompt: str) -> str:
        if self.delay_s:
            time.sleep(self.delay_s)
        return f"{self.response_prefix}:{prompt}"

    def get_context_usage(self) -> dict:
        return {"used_tokens": 21}

    def get_trace_history(self):
        return list(self.trace_history)


class TestAgentRuntimeManager(unittest.TestCase):
    def test_runtime_returns_agent_handle_with_execution_context(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            request = SubagentRequest(
                description="扫描仓库",
                prompt="只读分析 runtime 目录",
                workspace_root=tempdir,
                allowed_roots=(tempdir,),
            )
            context = ExecutionContext(
                workspace_root=tempdir,
                allowed_roots=(tempdir,),
                execution_mode="execute",
                permission_mode="default",
                metadata={"stage": "phase2"},
            )

            handle = runtime.run(request, execution_context=context)

            self.assertEqual(handle.status, "completed")
            self.assertEqual(handle.execution_context.workspace_root, os.path.abspath(tempdir))
            self.assertEqual(handle.execution_context.metadata["stage"], "phase2")
            self.assertEqual(handle.total_tool_use_count, 1)
            self.assertEqual(handle.total_tokens, 21)
            runtime.close()

    def test_runtime_team_membership_and_mailbox_delivery(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            teams = TeamManager(agent_runtime=runtime)
            runtime.bind_team_manager(teams)
            team = teams.create_team(name="reviewers")

            handle = runtime.run(
                SubagentRequest(
                    description="代码审查",
                    prompt="检查测试覆盖率",
                    team_name="reviewers",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                )
            )
            deliveries = runtime.send_message(
                recipient_type="team",
                recipient_id=team.team_id,
                content="保持只读分析，不要修改文件",
                sender_id="lead",
            )

            refreshed = runtime.get_handle(handle.agent_id)
            self.assertEqual(refreshed.team_id, team.team_id)
            self.assertEqual(len(deliveries), 1)
            self.assertEqual(len(refreshed.mailbox), 1)
            self.assertEqual(refreshed.mailbox[0].content, "保持只读分析，不要修改文件")
            runtime.close()


class TestCollaborationTools(unittest.TestCase):
    def test_team_and_message_tools_integrate_with_runtime(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            teams = TeamManager(agent_runtime=runtime)
            runtime.bind_team_manager(teams)

            registry = ToolRegistry()
            create_tool = register_team_create_tool(registry, team_manager=teams)
            delete_tool = register_team_delete_tool(registry, team_manager=teams)
            send_tool = register_send_message_tool(registry, agent_runtime=runtime)

            create_result = create_tool.run({"name": "analysis"})
            team_id = create_result.structured_data["teamId"]
            handle = runtime.run(
                SubagentRequest(
                    description="扫描代码",
                    prompt="查看 runtime/agents",
                    team_name="analysis",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                )
            )

            send_result = send_tool.run(
                {
                    "recipient_type": "team",
                    "recipient_id": team_id,
                    "content": "先整理结论，再补理由",
                    "sender_id": "manager",
                }
            )
            delete_result = delete_tool.run({"team_id": team_id})

            self.assertEqual(create_result.status, "success")
            self.assertEqual(send_result.status, "success")
            self.assertEqual(send_result.structured_data["deliveryCount"], 1)
            self.assertEqual(runtime.get_handle(handle.agent_id).mailbox[0].content, "先整理结论，再补理由")
            self.assertEqual(delete_result.status, "success")
            runtime.close()

    def test_agent_tool_uses_runtime_and_returns_execution_context(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            teams = TeamManager(agent_runtime=runtime)
            runtime.bind_team_manager(teams)
            teams.create_team(name="phase2")

            tool = AgentTool(
                agent_factory=lambda request: FakeSubagent(),
                agent_runtime=runtime,
                workspace_root=tempdir,
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            result = tool.run(
                {
                    "description": "运行时扫描",
                    "prompt": "分析 runtime/teams",
                    "team_name": "phase2",
                }
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(result.structured_data["executionContext"]["workspaceRoot"], os.path.abspath(tempdir))
            self.assertEqual(result.structured_data["teamId"], teams.get_team("phase2").team_id)
            runtime.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
