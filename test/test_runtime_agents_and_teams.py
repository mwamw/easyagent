import os
import sys
import tempfile
import threading
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    AgentTool,
    register_agent_runtime_tools,
    register_mailbox_tools,
    register_send_message_tool,
    register_team_create_tool,
    register_team_delete_tool,
)
from Tool.runtime import SubagentRequest
from agent import BasicAgent
from core.Exception import AgentStopRequested
from core.llm import EasyLLM
from runtime import AgentRuntimeManager, ExecutionContext, TeamManager
from task import InMemoryTaskStore, TaskService


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 128

    def invoke(self, messages, temperature=None, **kwargs):
        return "mock-response"

    def prepare_messages_for_request(self, messages):
        return list(messages)


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
        return {
            "version": 2,
            "requestEstimate": {"estimatedRequestTokens": 21, "source": "test", "metadata": {}},
            "budget": {},
            "compaction": {"last": {}, "estimatedRequestTokens": 21, "tokenSource": "test", "metadata": {}},
            "cache": {},
        }

    def get_trace_history(self):
        return list(self.trace_history)


class StoppableFakeSubagent(FakeSubagent):
    def __init__(self):
        super().__init__(response_prefix="stopped")
        self._stop_event = threading.Event()
        self._stop_reason = "manager requested stop"

    def request_stop(self, reason: str = "") -> None:
        self._stop_reason = str(reason or "").strip() or self._stop_reason
        self._stop_event.set()

    def invoke(self, prompt: str) -> str:
        while not self._stop_event.is_set():
            time.sleep(0.01)
        raise AgentStopRequested(self._stop_reason)


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

    def test_runtime_supports_task_scoped_mailbox_delivery(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            task_context = ExecutionContext(
                workspace_root=tempdir,
                allowed_roots=(tempdir,),
                execution_mode="execute",
                permission_mode="default",
                current_task_id="task_scope_1",
            )
            handle = runtime.run(
                SubagentRequest(
                    description="任务范围消息",
                    prompt="订阅 task mailbox",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                execution_context=task_context,
            )

            deliveries = runtime.send_message(
                recipient_type="task",
                recipient_id="task_scope_1",
                content="这是 task 级广播",
                sender_id="lead",
            )

            refreshed = runtime.get_handle(handle.agent_id)
            self.assertEqual(len(deliveries), 1)
            self.assertEqual(refreshed.mailbox[0].metadata["originalRecipientType"], "task")
            self.assertEqual(refreshed.mailbox[0].content, "这是 task 级广播")
            runtime.close()

    def test_runtime_mailbox_read_ack_and_completion_records(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(delay_s=0.05),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            handle = runtime.run(
                SubagentRequest(
                    description="后台 mailbox",
                    prompt="等待 manager 协作消息",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                run_in_background=True,
            )
            runtime.send_message(
                recipient_type="agent",
                recipient_id=handle.agent_id,
                content="先接收消息，再汇报结果",
                sender_id="manager",
                ttl_ms=10_000,
            )

            read_messages = runtime.read_mailbox(handle.agent_id)
            self.assertEqual(len(read_messages), 1)
            self.assertEqual(read_messages[0].status, "delivered")

            acked = runtime.ack_mailbox(handle.agent_id, message_ids=[read_messages[0].message_id], actor_id="manager")
            self.assertEqual(len(acked), 1)
            self.assertEqual(acked[0].status, "consumed")

            time.sleep(0.1)
            records = runtime.list_completion_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].agent_id, handle.agent_id)
            self.assertEqual(records[0].status, "completed")
            self.assertEqual(records[0].output_file, handle.output_file)

            waited = runtime.wait(handle.agent_id, timeout_ms=1000)
            self.assertEqual(waited.status, "completed")
            runtime.close()

    def test_runtime_export_and_restore_preserves_team_mailbox_and_task_binding(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            teams = TeamManager(agent_runtime=runtime)
            runtime.bind_team_manager(teams)
            team = teams.create_team(name="restorable-team", description="session restore target")
            context = ExecutionContext(
                workspace_root=tempdir,
                allowed_roots=(tempdir,),
                execution_mode="execute",
                permission_mode="default",
                current_task_id="task_restore",
                metadata={"phase": "phase3"},
            )

            handle = runtime.run(
                SubagentRequest(
                    description="状态导出",
                    prompt="整理当前运行时状态",
                    team_name="restorable-team",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                execution_context=context,
            )
            runtime.send_message(
                recipient_type="team",
                recipient_id=team.team_id,
                content="这条消息必须随 runtime state 一起恢复",
                sender_id="lead",
            )

            runtime_state = runtime.export_state()
            team_state = teams.export_state()
            runtime.close()

            restored_runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents-restored"),
            )
            restored_teams = TeamManager(agent_runtime=restored_runtime)
            restored_runtime.bind_team_manager(restored_teams)
            restored_teams.restore_state(team_state)
            restored_runtime.restore_state(runtime_state)

            restored_handle = restored_runtime.get_handle(handle.agent_id)
            self.assertEqual(restored_handle.team_id, team.team_id)
            self.assertEqual(restored_handle.execution_context.current_task_id, "task_restore")
            self.assertEqual(restored_handle.execution_context.metadata["phase"], "phase3")
            self.assertEqual(restored_handle.mailbox[0].content, "这条消息必须随 runtime state 一起恢复")
            self.assertEqual(restored_teams.get_team("restorable-team").member_agent_ids, (handle.agent_id,))
            restored_runtime.close()

    def test_runtime_wait_and_stop_complete_background_lifecycle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(delay_s=0.05),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            handle = runtime.run(
                SubagentRequest(
                    description="后台等待",
                    prompt="等待完成",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                run_in_background=True,
            )
            waited = runtime.wait(handle.agent_id, timeout_ms=1000)
            self.assertEqual(waited.status, "completed")
            runtime.close()

        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: StoppableFakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            handle = runtime.run(
                SubagentRequest(
                    description="后台停止",
                    prompt="等待停止",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                run_in_background=True,
            )
            stopped = runtime.stop(handle.agent_id, reason="manager requested stop", wait=True, timeout_ms=1000)
            self.assertEqual(stopped.status, "stopped")
            self.assertEqual(stopped.stop_reason, "manager requested stop")
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
            self.assertIn('"teamId"', create_result.to_display_string())
            self.assertEqual(send_result.status, "success")
            self.assertEqual(send_result.structured_data["deliveryCount"], 1)
            self.assertIn('"deliveries"', send_result.to_display_string())
            self.assertEqual(runtime.get_handle(handle.agent_id).mailbox[0].content, "先整理结论，再补理由")
            self.assertEqual(delete_result.status, "success")
            self.assertIn('"memberAgentIds"', delete_result.to_display_string())
            runtime.close()

    def test_mailbox_tools_and_prompt_injection_complete_message_lifecycle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            handle = runtime.run(
                SubagentRequest(
                    description="mailbox receiver",
                    prompt="监听协作消息",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                )
            )

            worker_registry = ToolRegistry()
            worker = BasicAgent(
                name="worker",
                llm=DummyLLM(),
                enable_tool=True,
                tool_registry=worker_registry,
            )
            register_mailbox_tools(
                worker_registry,
                agent_runtime=runtime,
                parent_agent=worker,
            )
            worker.bind_runtime(
                agent_runtime=runtime,
                execution_context=handle.execution_context,
            )

            runtime.send_message(
                recipient_type="agent",
                recipient_id=handle.agent_id,
                content="收到消息后先调整计划，再决定是否改文件",
                sender_id="manager",
            )

            prompt = worker.get_enhanced_prompt()
            self.assertIn("## 协作邮箱", prompt)
            self.assertIn("收到消息后先调整计划，再决定是否改文件", prompt)

            delivered = runtime.list_mailbox(handle.agent_id, include_consumed=False)
            self.assertEqual(delivered[0].status, "delivered")

            read_result = worker_registry.execute_tool_result("MailboxRead", {"limit": 20})
            self.assertEqual(read_result.status, "success")
            self.assertEqual(read_result.structured_data["count"], 1)
            self.assertIn('"messages"', read_result.to_display_string())

            message_id = read_result.structured_data["messages"][0]["messageId"]
            ack_result = worker_registry.execute_tool_result("MailboxAck", {"message_ids": [message_id]})
            self.assertEqual(ack_result.status, "success")
            self.assertEqual(ack_result.structured_data["messages"][0]["status"], "consumed")
            self.assertIn('"ackedAll"', ack_result.to_display_string())

            prompt_after_ack = worker.get_enhanced_prompt()
            self.assertNotIn("收到消息后先调整计划，再决定是否改文件", prompt_after_ack)
            runtime.close()

    def test_agent_tool_uses_runtime_and_returns_execution_context(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(delay_s=0.05),
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
                    "run_in_background": True,
                }
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(result.structured_data["executionContext"]["workspaceRoot"], os.path.abspath(tempdir))
            self.assertEqual(result.structured_data["teamId"], teams.get_team("phase2").team_id)
            display = result.to_display_string()
            self.assertIn('"outputFile"', display)
            self.assertIn('"executionContext"', display)
            self.assertIn(result.structured_data["agentId"], display)
            runtime.close()

    def test_agent_runtime_tools_query_wait_and_stop(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(delay_s=0.05),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            registry = ToolRegistry()
            get_tool, list_tool, wait_tool, stop_tool = register_agent_runtime_tools(
                registry,
                agent_runtime=runtime,
            )

            background_handle = runtime.run(
                SubagentRequest(
                    description="后台 agent",
                    prompt="等待结束",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                run_in_background=True,
            )
            get_result = get_tool.run({"agent_id": background_handle.agent_id})
            list_result = list_tool.run({"limit": 10})
            wait_result = wait_tool.run({"agent_id": background_handle.agent_id, "timeout_ms": 1000})

            self.assertEqual(get_result.status, "success")
            self.assertIn('"outputFile"', get_result.to_display_string())
            self.assertEqual(list_result.structured_data["count"], 1)
            self.assertIn('"agents"', list_result.to_display_string())
            self.assertEqual(wait_result.structured_data["status"], "completed")
            self.assertTrue(wait_result.structured_data["isBackground"])
            self.assertIn('"timedOut"', wait_result.to_display_string())

            runtime.close()

        with tempfile.TemporaryDirectory() as tempdir:
            runtime = AgentRuntimeManager(
                agent_factory=lambda request: StoppableFakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            registry = ToolRegistry()
            _, _, _, stop_tool = register_agent_runtime_tools(
                registry,
                agent_runtime=runtime,
            )
            background_handle = runtime.run(
                SubagentRequest(
                    description="可停止 agent",
                    prompt="等待 stop",
                    workspace_root=tempdir,
                    allowed_roots=(tempdir,),
                ),
                run_in_background=True,
            )
            stop_result = stop_tool.run(
                {
                    "agent_id": background_handle.agent_id,
                    "reason": "tool requested stop",
                    "wait": True,
                    "timeout_ms": 1000,
                }
            )
            self.assertEqual(stop_result.status, "success")
            self.assertEqual(stop_result.structured_data["status"], "stopped")
            self.assertEqual(stop_result.structured_data["stopReason"], "tool requested stop")
            self.assertIn('"requestedReason"', stop_result.to_display_string())
            runtime.close()

    def test_agent_tool_binds_subagent_task_team_and_runtime_state(self):
        with tempfile.TemporaryDirectory() as tempdir:
            llm = DummyLLM()
            service = TaskService(InMemoryTaskStore())
            registry = ToolRegistry()
            parent = BasicAgent(
                name="manager",
                llm=llm,
                enable_tool=True,
                tool_registry=registry,
                task_service=service,
            )
            root_task = service.create_task(title="Parent task", owner="manager")
            parent.set_current_task(root_task.task_id)

            runtime = AgentRuntimeManager(
                agent_factory=lambda request: FakeSubagent(),
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            teams = TeamManager(agent_runtime=runtime)
            runtime.bind_team_manager(teams)
            teams.create_team(name="task-team")

            tool = AgentTool(
                parent_agent=parent,
                agent_factory=lambda request: FakeSubagent(),
                agent_runtime=runtime,
                workspace_root=tempdir,
                storage_dir=os.path.join(tempdir, ".agents"),
            )
            result = tool.run(
                {
                    "description": "Task-bound audit",
                    "prompt": "检查 task/runtime 绑定",
                    "team_name": "task-team",
                }
            )

            handle = runtime.get_handle(result.structured_data["agentId"])
            child_tasks = service.list_tasks(parent_task_id=root_task.task_id)

            self.assertEqual(result.status, "success")
            self.assertEqual(len(child_tasks), 1)
            self.assertEqual(handle.execution_context.current_task_id, child_tasks[0].task_id)
            self.assertEqual(child_tasks[0].owner, handle.agent_id)
            self.assertEqual(child_tasks[0].metadata["runtime"]["teamId"], handle.team_id)
            self.assertEqual(child_tasks[0].metadata["runtime"]["status"], "completed")
            self.assertEqual(child_tasks[0].metadata["runtime"]["outputFile"], handle.output_file)
            runtime.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
