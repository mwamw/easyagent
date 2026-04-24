"""
会话持久化测试
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta

from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import BasicAgent, ConversationalAgent, PlanningAgent, ReactAgent
from core.Config import Config
from core.Message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from core.llm import EasyLLM
from core.permissions import PermissionBehavior, PermissionMode, PermissionRule
from core.request_input import ReplayRequestInput
from context.manager import ContextManager
from db import ConversationStore, SessionStore
from runtime import TeamManager
from skill import BaseSkill, SkillConfig, SkillRegistry
from task import InMemoryTaskStore, SQLiteTaskStore, TaskService
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    register_agent_tool,
    register_agent_runtime_tools,
    register_file_read_tool,
    register_enter_worktree_tool,
    register_exit_worktree_tool,
    register_mailbox_tools,
    register_send_message_tool,
    register_team_create_tool,
    register_team_delete_tool,
)
from Tool.runtime import WorktreeManager


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.last_messages = []

    def invoke(self, messages, temperature=None, **kwargs):
        self.last_messages = list(messages)
        return "mock-response"

    def prepare_messages_for_request(self, messages):
        return list(messages)


class ReplayAwareDummyLLM(DummyLLM):
    def __init__(self, provider_name: str = "mock", model: str = "mock-model"):
        self.provider_name = provider_name
        self.model = model
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.last_messages = []
        self.temperature = 0.7
        self.timeout = 60
        self.kwargs = {}


class ClosableDummyLLM(DummyLLM):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


class EchoParams(BaseModel):
    text: str


class FakeMemoryManage:
    def __init__(self):
        self.memory_types = {}


class SkillEchoParams(BaseModel):
    pass


class SkillEchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="skill_echo",
            description="Echo from restored skill",
            parameters=SkillEchoParams,
            read_only=True,
        )

    def run(self, parameters: dict):
        return "skill-echo"


class RestorableSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            SkillConfig(
                name="restorable_skill",
                description="Can be rebuilt from SkillRegistry",
                auto_activate=True,
            )
        )

    def get_tools(self):
        return [SkillEchoTool()]

    def get_prompt(self) -> str:
        return "Restored skill prompt"


class RuntimeSubagent:
    def __init__(self):
        self.trace_history = [{"type": "tool_call", "tool_name": "WebSearch"}]

    def invoke(self, prompt: str) -> str:
        return f"runtime:{prompt}"

    def get_context_usage(self) -> dict:
        return {"used_tokens": 9}


class SlowRuntimeSubagent(RuntimeSubagent):
    def invoke(self, prompt: str) -> str:
        time.sleep(0.5)
        return super().invoke(prompt)


def _init_git_repo(root: str) -> str:
    repo = os.path.join(root, "repo")
    os.makedirs(repo, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
    with open(os.path.join(repo, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)
    return repo

    def get_trace_history(self):
        return list(self.trace_history)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool("echo", "Echo tool", EchoParams)
    def echo(text: str) -> str:
        return text

    return registry


class SessionPersistenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "sessions.db")
        self.session_store = SessionStore(self.db_path)
        self.conversation_store = ConversationStore(self.db_path)
        self.llm = DummyLLM()

    def tearDown(self):
        SkillRegistry.reset()
        self.tempdir.cleanup()

    def test_session_store_crud_and_cleanup(self):
        self.session_store.create_or_update_session(
            session_id="active",
            agent_type="BasicAgent",
            agent_name="assistant",
            snapshot={"agent_type": "BasicAgent", "name": "assistant"},
            metadata={"tag": "current"},
        )
        self.session_store.create_or_update_session(
            session_id="expired",
            agent_type="BasicAgent",
            agent_name="assistant",
            snapshot={"agent_type": "BasicAgent", "name": "assistant"},
            expires_at=datetime.now() - timedelta(minutes=1),
        )

        listed = self.session_store.list_sessions()
        self.assertEqual([item["session_id"] for item in listed], ["active"])

        all_sessions = self.session_store.list_sessions(include_expired=True)
        self.assertEqual({item["session_id"] for item in all_sessions}, {"active", "expired"})

        removed = self.session_store.cleanup_expired_sessions()
        self.assertEqual(removed, 1)
        self.assertIsNone(self.session_store.get_session("expired"))
        self.assertTrue(self.session_store.delete_session("active"))

    def test_conversation_store_round_trip(self):
        self.session_store.create_or_update_session(
            session_id="conv-1",
            agent_type="BasicAgent",
            agent_name="assistant",
            snapshot={"agent_type": "BasicAgent", "name": "assistant"},
        )
        messages = [
            SystemMessage("system", metadata={"source": "test"}),
            UserMessage("hello"),
            AssistantMessage("world"),
            ToolMessage("done", tool_call_id="call-1", name="echo", metadata={"tool": True}),
        ]

        self.conversation_store.replace_messages("conv-1", messages)
        loaded = self.conversation_store.load_messages("conv-1")

        self.assertEqual([msg["role"] for msg in loaded], ["system", "user", "assistant", "tool"])
        self.assertEqual(loaded[0]["metadata"]["source"], "test")
        self.assertEqual(loaded[-1]["tool_call_id"], "call-1")
        self.assertEqual(loaded[-1]["name"], "echo")

    def test_conversation_store_round_trip_raw_provider_messages(self):
        self.session_store.create_or_update_session(
            session_id="conv-raw",
            agent_type="BasicAgent",
            agent_name="assistant",
            snapshot={"agent_type": "BasicAgent", "name": "assistant"},
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"},
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "hi",
            },
        ]

        self.conversation_store.replace_messages("conv-raw", messages)
        loaded = self.conversation_store.load_messages("conv-raw")

        self.assertEqual(loaded[0]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(loaded[1]["type"], "function_call_output")
        self.assertEqual(loaded[1]["call_id"], "call-1")

    def test_basic_agent_save_and_restore(self):
        registry = build_registry()
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            system_prompt="test prompt",
            enable_tool=True,
            tool_registry=registry,
            config=Config(max_history_length=9, temperature=0.1),
            verbose_thinking=True,
            history_via_context_manager=True,
        )
        agent.add_message(UserMessage("hello"))
        agent.add_message(AssistantMessage("world"))
        agent.trace_history = [
            {
                "id": "evt_000001",
                "session_id": "trace_test",
                "turn_id": "turn_0001",
                "seq": 1,
                "type": "reasoning",
                "timestamp": datetime.now().isoformat(),
                "role": "assistant",
                "content": "thought 1",
                "metadata": {},
            },
            {
                "id": "evt_000002",
                "session_id": "trace_test",
                "turn_id": "turn_0002",
                "seq": 2,
                "type": "reasoning",
                "timestamp": datetime.now().isoformat(),
                "role": "assistant",
                "content": "thought 2",
                "metadata": {},
            },
        ]
        agent.save_session("basic-1", store=self.session_store, metadata={"suite": "unit"})

        restored = BasicAgent.load_session(
            "basic-1",
            llm=self.llm,
            store=self.session_store,
            tool_registry=registry,
        )

        self.assertIsInstance(restored, BasicAgent)
        self.assertEqual(restored.name, "assistant")
        self.assertEqual(restored.system_prompt, "test prompt")
        self.assertTrue(restored.enable_tool)
        self.assertEqual(restored.get_history_length(), 2)
        self.assertEqual(restored.get_history()[0]["role"], "user")
        self.assertEqual(restored.get_history()[0]["content"], "hello")
        self.assertEqual(restored.get_canonical_history()[0].text_content(), "hello")
        self.assertEqual(
            [event["content"] for event in restored.get_trace_history() if event["type"] == "reasoning"],
            ["thought 1", "thought 2"],
        )
        self.assertEqual(
            [event["type"] for event in restored.get_trace_history()],
            ["reasoning", "reasoning"],
        )

    def test_replay_history_rebuilds_after_provider_change(self):
        llm = ReplayAwareDummyLLM()
        agent = BasicAgent(name="assistant", llm=llm)
        agent.add_message(UserMessage("hello"))
        agent.add_message(AssistantMessage("world"))

        self.assertEqual(agent.get_history()[0]["role"], "user")
        self.assertEqual(agent.get_history()[0]["content"], "hello")
        self.assertEqual(agent.replay_history_provider_name, "mock")

        native_llm = ReplayAwareDummyLLM(provider_name="google_native", model="gemini-2.5-pro")
        agent.change_model(llm=native_llm)
        rebuilt = agent._build_start_messages("next turn")

        self.assertEqual(agent.replay_history_provider_name, "google_native")
        self.assertEqual(agent.get_canonical_history()[0].role, "user")
        self.assertEqual(agent.get_canonical_history()[0].text_content(), "hello")
        self.assertEqual(agent.get_history()[0]["parts"][0]["text"], "hello")
        self.assertEqual(agent.get_history()[0]["role"], "user")
        self.assertIsInstance(rebuilt, ReplayRequestInput)
        self.assertTrue(rebuilt.system_prompt)
        self.assertEqual(rebuilt.replay_history[0]["role"], "user")
        self.assertEqual(rebuilt.replay_history[-1]["parts"][0]["text"], "next turn")

    def test_change_model_rebuilds_request_ready_replay_history(self):
        agent = BasicAgent(name="assistant", llm=ReplayAwareDummyLLM(provider_name="openai"))
        agent.add_message(UserMessage("hello"))
        agent.add_message(AssistantMessage("world"))

        self.assertEqual(agent.get_history()[0], {"role": "user", "content": "hello"})
        self.assertEqual(agent.get_history()[1]["role"], "assistant")

        agent.change_model(llm=ReplayAwareDummyLLM(provider_name="anthropic_native", model="claude-4.5-sonnet"))

        replay_history = agent.get_history()
        self.assertEqual(replay_history[0]["role"], "user")
        self.assertEqual(replay_history[0]["content"], "hello")
        self.assertEqual(replay_history[1]["role"], "assistant")
        self.assertEqual(replay_history[1]["content"], "world")
        self.assertEqual(agent.replay_history_provider_name, "anthropic_native")

    def test_change_model_preserves_cross_provider_thinking_but_filters_signatures(self):
        agent = BasicAgent(name="assistant", llm=ReplayAwareDummyLLM(provider_name="anthropic_native"))
        agent.add_message(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private plan", "signature": "anthropic-sig"},
                    {"type": "text", "text": "visible answer"},
                ],
            }
        )

        self.assertIn("anthropic-sig", repr(agent.get_history()))
        agent.change_model(llm=ReplayAwareDummyLLM(provider_name="google_native", model="gemini-3-flash"))

        google_replay = agent.get_history()
        google_replay_text = repr(google_replay)
        self.assertEqual(
            google_replay,
            [
                {
                    "role": "model",
                    "parts": [
                        {"text": "private plan", "thought": True},
                        {"text": "visible answer"},
                    ],
                }
            ],
        )
        self.assertNotIn("anthropic-sig", google_replay_text)
        self.assertNotIn("thought_signature", google_replay_text)
        self.assertIn("private plan", google_replay_text)
        self.assertEqual(agent.get_canonical_history()[0].content[0].signature, "anthropic-sig")

        agent.change_model(llm=ReplayAwareDummyLLM(provider_name="claude_native", model="claude-4.5-sonnet"))
        claude_replay = agent.get_history()
        self.assertEqual(claude_replay[0]["content"][0]["signature"], "anthropic-sig")
        self.assertEqual(agent.replay_history_provider_name, "claude_native")

    def test_direct_provider_mutation_requires_change_model(self):
        agent = BasicAgent(name="assistant", llm=ReplayAwareDummyLLM(provider_name="openai"))
        agent.add_message(UserMessage("hello"))
        agent.add_message(AssistantMessage("world"))

        agent.llm.provider_name = "google_native"

        with self.assertRaises(RuntimeError):
            agent._build_start_messages("next turn")

    def test_basic_agent_restore_without_tool_registry_downgrades_to_plain_mode(self):
        registry = build_registry()
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
        )
        agent.add_message(UserMessage("hello"))
        agent.save_session("basic-plain", store=self.session_store)

        restored = BasicAgent.load_session(
            "basic-plain",
            llm=self.llm,
            store=self.session_store,
        )

        self.assertFalse(restored.enable_tool)
        self.assertIsNone(restored.tool_registry)

    def test_basic_agent_context_usage_persists_across_session(self):
        manager = ContextManager(max_tokens=80, auto_history=True)
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            system_prompt="test prompt",
            context_manager=manager,
            history_via_context_manager=True,
        )
        agent.add_message(UserMessage("hello"))
        agent.add_message(AssistantMessage("world"))

        usage = agent.get_context_usage()

        self.assertEqual(usage["max_tokens"], 80)
        self.assertGreater(usage["used_tokens"], 0)
        self.assertGreaterEqual(usage["remaining_tokens"], 0)
        self.assertFalse(hasattr(manager, "last_usage"))

        agent.save_session("basic-context-usage", store=self.session_store)

        restored_manager = ContextManager(max_tokens=80, auto_history=True)
        restored = BasicAgent.load_session(
            "basic-context-usage",
            llm=self.llm,
            store=self.session_store,
            context_manager=restored_manager,
        )

        restored_usage = restored.get_context_usage()
        self.assertEqual(restored_usage["max_tokens"], usage["max_tokens"])
        self.assertEqual(restored_usage["history_tokens"], usage["history_tokens"])
        self.assertEqual(restored_usage["stable_context_tokens"], usage["stable_context_tokens"])
        self.assertEqual(restored_usage["canonical_history_messages"], usage["canonical_history_messages"])
        self.assertEqual(restored_usage["replay_history_messages"], usage["replay_history_messages"])
        self.assertFalse(hasattr(restored_manager, "last_usage"))

    def test_get_context_usage_does_not_mutate_context_manager(self):
        manager = ContextManager(max_tokens=128, auto_history=True)
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            system_prompt="test prompt",
            context_manager=manager,
        )
        agent.add_message(UserMessage("hello"))

        usage = agent.get_context_usage()

        self.assertIn("used_tokens", usage)
        self.assertFalse(hasattr(manager, "last_usage"))

    def test_get_context_usage_exposes_split_token_metrics(self):
        manager = ContextManager(max_tokens=128, auto_history=True)
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            system_prompt="test prompt",
            context_manager=manager,
        )
        agent.add_message(UserMessage("hello"))
        agent.add_message(AssistantMessage("world"))

        usage = agent.get_context_usage()

        self.assertIn("history_tokens", usage)
        self.assertIn("system_tokens", usage)
        self.assertIn("tool_tokens", usage)
        self.assertIn("reasoning_tokens", usage)
        self.assertIn("estimated_request_tokens", usage)
        self.assertEqual(usage["estimated_request_tokens"], usage["stable_context_tokens"])
        self.assertEqual(usage["used_tokens"], usage["estimated_request_tokens"])
        self.assertEqual(
            usage["estimated_request_tokens"],
            usage["history_tokens"] + usage["system_tokens"] + usage["tool_tokens"] + usage["reasoning_tokens"],
        )

    def test_reasoning_history_persistence_can_be_disabled(self):
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            config=Config(persist_reasoning_history=False),
        )

        agent._append_assistant_message_history(content="world", thinking="hidden chain")

        last_canonical = agent.history[-1]
        last_replay = agent.raw_history[-1]
        self.assertTrue(all(block.type != "reasoning" for block in last_canonical.content))
        self.assertEqual(last_replay["role"], "assistant")
        self.assertNotIn("reasoning_content", last_replay)

    def test_reasoning_history_persistence_defaults_to_enabled(self):
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
        )

        agent._append_assistant_message_history(content="world", thinking="visible chain")

        last_canonical = agent.history[-1]
        last_replay = agent.raw_history[-1]
        self.assertTrue(any(block.type == "reasoning" for block in last_canonical.content))
        self.assertEqual(last_replay.get("reasoning_content"), "visible chain")

    def test_basic_agent_restores_mode_permissions_and_current_task(self):
        registry = build_registry()
        task_service = TaskService(InMemoryTaskStore())
        task = task_service.create_task(title="Implement permissions", owner="agent")
        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            task_service=task_service,
        )
        agent.enter_plan_mode(allowed_actions=["read", "search"])
        agent.add_permission_rule(
            PermissionRule(
                tool_name="echo",
                behavior=PermissionBehavior.ALLOW,
                source="session",
                description="允许 echo 工具直接执行",
            )
        )
        agent.set_current_task(task.task_id)
        agent.add_message(UserMessage("resume later"))
        agent.save_session("basic-runtime-state", store=self.session_store)
        restore_registry = build_registry()

        restored = BasicAgent.load_session(
            "basic-runtime-state",
            llm=self.llm,
            store=self.session_store,
            tool_registry=restore_registry,
            task_service=task_service,
        )

        self.assertEqual(restored.get_execution_mode().value, "plan")
        self.assertEqual(restored.permission_context.mode, PermissionMode.PLAN)
        self.assertEqual(restored.permission_context.rules[0].tool_name, "echo")
        self.assertEqual(restored.mode_controller.state.allowed_actions, ["read", "search"])
        self.assertEqual(restored.current_task_id, task.task_id)
        self.assertTrue(restored.tool_registry.has_tool("TaskCreate"))
        self.assertTrue(restored.tool_registry.has_tool("TaskList"))

    def test_load_session_auto_restores_framework_dependencies(self):
        registry = ToolRegistry()
        register_file_read_tool(registry, workspace_root=self.tempdir.name)
        config = Config(
            workspace_root=self.tempdir.name,
            allowed_roots=[self.tempdir.name],
            command_timeout_ms=4321,
        )
        context_manager = ContextManager(max_tokens=80)
        task_db_path = os.path.join(self.tempdir.name, "tasks.sqlite3")
        task_service = TaskService(SQLiteTaskStore(task_db_path))
        skill_registry = SkillRegistry.instance()
        skill_registry.register_class(RestorableSkill, name="restorable_skill")

        agent = BasicAgent(
            name="assistant",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            config=config,
            context_manager=context_manager,
            task_service=task_service,
        )
        agent.with_skill(RestorableSkill())
        agent.save_session("auto-restore-deps", store=self.session_store)

        restored = BasicAgent.load_session(
            "auto-restore-deps",
            llm=self.llm,
            store=self.session_store,
        )

        self.assertTrue(restored.enable_tool)
        self.assertIsNotNone(restored.tool_registry)
        self.assertTrue(restored.tool_registry.has_tool("FileRead"))
        self.assertTrue(restored.tool_registry.has_tool("TaskCreate"))
        self.assertTrue(restored.tool_registry.has_tool("skill_echo"))
        self.assertIsNotNone(restored.context_manager)
        self.assertEqual(restored.context_manager.budget.max_tokens, 80)
        self.assertIsNotNone(restored.task_service)
        self.assertEqual(restored.task_service.store.db_path, task_db_path)
        self.assertTrue(restored.skill_manager.has_skill("restorable_skill"))
        self.assertTrue(restored.skill_manager.is_active("restorable_skill"))

    def test_basic_agent_session_restore_rebuilds_collaboration_runtime(self):
        registry = ToolRegistry()
        task_service = TaskService(InMemoryTaskStore())
        task = task_service.create_task(title="Restore collaboration runtime", owner="manager")
        agent = BasicAgent(
            name="manager",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            task_service=task_service,
            config=Config(
                workspace_root=self.tempdir.name,
                allowed_roots=[self.tempdir.name],
            ),
        )

        agent_tool = register_agent_tool(
            registry,
            parent_agent=agent,
            agent_factory=lambda request: RuntimeSubagent(),
            workspace_root=self.tempdir.name,
            allowed_roots=(self.tempdir.name,),
            storage_dir=os.path.join(self.tempdir.name, ".agents"),
        )
        team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
        agent_tool.agent_runtime.bind_team_manager(team_manager)
        register_send_message_tool(
            registry,
            agent_runtime=agent_tool.agent_runtime,
            parent_agent=agent,
        )
        register_mailbox_tools(
            registry,
            agent_runtime=agent_tool.agent_runtime,
            parent_agent=agent,
        )
        register_agent_runtime_tools(
            registry,
            agent_runtime=agent_tool.agent_runtime,
            parent_agent=agent,
        )
        register_team_create_tool(registry, team_manager=team_manager)
        register_team_delete_tool(registry, team_manager=team_manager)
        agent.bind_runtime(
            agent_runtime=agent_tool.agent_runtime,
            team_manager=team_manager,
        )
        agent.set_current_task(task.task_id)

        team = team_manager.create_team(name="session-team")
        delegation = registry.execute_tool_result(
            "Agent",
            {
                "description": "恢复 runtime",
                "prompt": "只读整理当前协作状态",
                "team_name": "session-team",
            },
        )
        agent_id = delegation.structured_data["agentId"]
        delegated_task = task_service.list_tasks(parent_task_id=task.task_id)[0]
        message_result = registry.execute_tool_result(
            "SendMessage",
            {
                "recipient_type": "team",
                "recipient_id": team.team_id,
                "content": "恢复后 mailbox 仍应保留这条消息",
            },
        )
        self.assertEqual(message_result.status, "success")

        agent.save_session("basic-collaboration-runtime", store=self.session_store)
        restored = BasicAgent.load_session(
            "basic-collaboration-runtime",
            llm=self.llm,
            store=self.session_store,
            task_service=task_service,
        )

        self.assertIsNotNone(restored.agent_runtime)
        self.assertIsNotNone(restored.team_manager)
        self.assertEqual(restored.current_task_id, task.task_id)
        self.assertTrue(restored.tool_registry.has_tool("Agent"))
        self.assertTrue(restored.tool_registry.has_tool("AgentGet"))
        self.assertTrue(restored.tool_registry.has_tool("AgentList"))
        self.assertTrue(restored.tool_registry.has_tool("AgentWait"))
        self.assertTrue(restored.tool_registry.has_tool("AgentStop"))
        self.assertTrue(restored.tool_registry.has_tool("SendMessage"))
        self.assertTrue(restored.tool_registry.has_tool("MailboxRead"))
        self.assertTrue(restored.tool_registry.has_tool("MailboxAck"))
        self.assertTrue(restored.tool_registry.has_tool("TeamCreate"))
        self.assertTrue(restored.tool_registry.has_tool("TeamDelete"))

        restored_team = restored.team_manager.get_team("session-team")
        restored_handle = restored.agent_runtime.get_handle(agent_id)
        self.assertEqual(restored_handle.team_id, restored_team.team_id)
        self.assertEqual(restored_handle.execution_context.current_task_id, delegated_task.task_id)
        self.assertEqual(restored_handle.mailbox[0].content, "恢复后 mailbox 仍应保留这条消息")
        report = restored.get_last_restore_report()
        self.assertIsNotNone(report)
        self.assertEqual(report["status"], "restored")
        self.assertEqual(report["components"]["agent_runtime"]["status"], "restored")
        self.assertEqual(report["components"]["team_runtime"]["status"], "restored")

    def test_basic_agent_session_restore_reports_degraded_background_runtime(self):
        registry = ToolRegistry()
        task_service = TaskService(InMemoryTaskStore())
        agent = BasicAgent(
            name="manager",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            task_service=task_service,
            config=Config(
                workspace_root=self.tempdir.name,
                allowed_roots=[self.tempdir.name],
            ),
        )
        agent_tool = register_agent_tool(
            registry,
            parent_agent=agent,
            agent_factory=lambda request: SlowRuntimeSubagent(),
            workspace_root=self.tempdir.name,
            allowed_roots=(self.tempdir.name,),
            storage_dir=os.path.join(self.tempdir.name, ".agents-background"),
        )
        team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
        agent_tool.agent_runtime.bind_team_manager(team_manager)
        register_agent_runtime_tools(
            registry,
            agent_runtime=agent_tool.agent_runtime,
            parent_agent=agent,
        )
        register_send_message_tool(
            registry,
            agent_runtime=agent_tool.agent_runtime,
            parent_agent=agent,
        )
        agent.bind_runtime(agent_runtime=agent_tool.agent_runtime, team_manager=team_manager)

        registry.execute_tool_result(
            "Agent",
            {
                "description": "后台恢复报告",
                "prompt": "只读整理协作状态",
                "run_in_background": True,
            },
        )

        agent.save_session("background-runtime-degraded", store=self.session_store)
        restored = BasicAgent.load_session(
            "background-runtime-degraded",
            llm=self.llm,
            store=self.session_store,
            task_service=task_service,
        )

        report = restored.get_last_restore_report()
        self.assertIsNotNone(report)
        self.assertEqual(report["status"], "degraded")
        self.assertEqual(report["components"]["agent_runtime"]["status"], "degraded")
        self.assertTrue(report["components"]["agent_runtime"]["degradedItems"])

    def test_basic_agent_session_restore_reports_worktree_runtime(self):
        repo = _init_git_repo(self.tempdir.name)
        storage = os.path.join(self.tempdir.name, "worktrees")
        manager = WorktreeManager(repo, storage_dir=storage, original_cwd=repo)

        registry = ToolRegistry()
        register_enter_worktree_tool(registry, worktree_manager=manager)
        register_exit_worktree_tool(registry, worktree_manager=manager)
        agent = BasicAgent(
            name="worktree-agent",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            config=Config(
                workspace_root=repo,
                allowed_roots=[repo],
                enable_worktree=True,
            ),
        )

        enter_result = registry.execute_tool_result("EnterWorktree", {"name": "restore-session"})
        worktree_path = enter_result.structured_data["worktreePath"]
        self.assertTrue(os.path.isdir(worktree_path))

        agent.save_session("worktree-runtime-session", store=self.session_store)
        restored = BasicAgent.load_session(
            "worktree-runtime-session",
            llm=self.llm,
            store=self.session_store,
        )

        report = restored.get_last_restore_report()
        self.assertIsNotNone(report)
        self.assertEqual(report["components"]["worktree_runtime"]["status"], "restored")
        self.assertIn(os.path.abspath(worktree_path), report["components"]["worktree_runtime"]["restoredItems"])

    def test_basic_agent_close_returns_cleanup_report(self):
        repo = _init_git_repo(self.tempdir.name)
        storage = os.path.join(self.tempdir.name, "worktrees")
        manager = WorktreeManager(repo, storage_dir=storage, original_cwd=repo)
        llm = ClosableDummyLLM()

        registry = ToolRegistry()
        register_enter_worktree_tool(registry, worktree_manager=manager)
        register_exit_worktree_tool(registry, worktree_manager=manager)
        agent = BasicAgent(
            name="close-agent",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
            config=Config(
                workspace_root=repo,
                allowed_roots=[repo],
                enable_worktree=True,
            ),
        )

        agent_tool = register_agent_tool(
            registry,
            parent_agent=agent,
            workspace_root=repo,
            allowed_roots=(repo,),
            storage_dir=os.path.join(self.tempdir.name, ".agents-close"),
        )
        agent.bind_runtime(agent_runtime=agent_tool.agent_runtime)

        registry.execute_tool_result("EnterWorktree", {"name": "close-check"})

        report = agent.close()

        self.assertEqual(report["status"], "closed")
        self.assertEqual(report["components"]["agent_runtime"]["status"], "closed")
        self.assertEqual(report["components"]["worktree_runtime"]["status"], "closed")
        self.assertEqual(report["components"]["worktree_runtime"]["metadata"]["action"], "keep")
        self.assertFalse(manager.get_active_session())
        self.assertTrue(llm.closed)
        self.assertEqual(report["components"]["llm"]["status"], "closed")
        self.assertEqual(agent.get_last_close_report()["status"], "closed")

    def test_basic_agent_close_reports_degraded_background_runtime(self):
        llm = ClosableDummyLLM()
        registry = ToolRegistry()
        agent = BasicAgent(
            name="close-background-agent",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
            config=Config(
                workspace_root=self.tempdir.name,
                allowed_roots=[self.tempdir.name],
            ),
        )
        agent_tool = register_agent_tool(
            registry,
            parent_agent=agent,
            agent_factory=lambda request: SlowRuntimeSubagent(),
            workspace_root=self.tempdir.name,
            allowed_roots=(self.tempdir.name,),
            storage_dir=os.path.join(self.tempdir.name, ".agents-close-background"),
        )
        team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
        agent_tool.agent_runtime.bind_team_manager(team_manager)
        agent.bind_runtime(
            agent_runtime=agent_tool.agent_runtime,
            team_manager=team_manager,
        )

        launch_result = registry.execute_tool_result(
            "Agent",
            {
                "description": "后台 close 报告",
                "prompt": "只读整理协作状态",
                "run_in_background": True,
            },
        )

        report = agent.close(close_worktree=False)

        self.assertEqual(report["status"], "degraded")
        self.assertEqual(report["components"]["agent_runtime"]["status"], "degraded")
        self.assertIn(
            launch_result.structured_data["agentId"],
            report["components"]["agent_runtime"]["degradedItems"],
        )
        self.assertTrue(llm.closed)
        self.assertEqual(agent.get_last_close_report()["status"], "degraded")

    def test_conversational_agent_restore_keeps_auto_save_flag(self):
        memory_manage = FakeMemoryManage()
        agent = ConversationalAgent(
            name="chatbot",
            llm=self.llm,
            memory_manage=memory_manage,
            auto_save_to_working=False,
        )
        agent.add_message(UserMessage("你好"))
        agent.add_message(AssistantMessage("你好，我在。"))
        agent.save_session("conv-agent", store=self.session_store)

        restored = ConversationalAgent.load_session(
            "conv-agent",
            llm=self.llm,
            store=self.session_store,
            memory_manage=memory_manage,
        )

        self.assertFalse(restored.auto_save_to_working)
        self.assertIs(restored.memory_manage, memory_manage)
        self.assertEqual(restored.get_history_length(), 2)

    def test_react_agent_restore_keeps_verbose_and_scratchpad(self):
        registry = build_registry()
        agent = ReactAgent(
            name="reactor",
            llm=self.llm,
            tool_registry=registry,
            enable_tool=True,
            verbose=False,
        )
        agent.scratchpad = ["Thought: inspect", "Observation: done"]
        agent.add_message(UserMessage("task"))
        agent.save_session("react-1", store=self.session_store)

        restored = ReactAgent.load_session(
            "react-1",
            llm=self.llm,
            store=self.session_store,
            tool_registry=registry,
        )

        self.assertFalse(restored.verbose)
        self.assertEqual(restored.scratchpad, ["Thought: inspect", "Observation: done"])

    def test_planning_agent_restore_keeps_planning_state(self):
        registry = build_registry()
        agent = PlanningAgent(
            name="planner",
            llm=self.llm,
            tool_registry=registry,
            enable_tool=True,
            max_steps=3,
            allow_replan=False,
        )
        agent.current_plan = ["step1", "step2"]
        agent.execution_log = [{"step": 1, "result": "ok"}]
        agent.add_message(UserMessage("plan this"))
        agent.save_session("plan-1", store=self.session_store)

        restored = PlanningAgent.load_session(
            "plan-1",
            llm=self.llm,
            store=self.session_store,
            tool_registry=registry,
        )

        self.assertEqual(restored.max_steps, 3)
        self.assertFalse(restored.allow_replan)
        self.assertEqual(restored.current_plan, ["step1", "step2"])
        self.assertEqual(restored.execution_log, [{"step": 1, "result": "ok"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
