"""
会话持久化测试
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import BasicAgent, ConversationalAgent, PlanningAgent, ReactAgent
from core.Config import Config
from core.Message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from core.llm import EasyLLM
from db import ConversationStore, SessionStore
from Tool.ToolRegistry import ToolRegistry


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"


class EchoParams(BaseModel):
    text: str


class FakeMemoryManage:
    def __init__(self):
        self.memory_types = {}


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

        self.assertEqual([msg.role for msg in loaded], ["system", "user", "assistant", "tool"])
        self.assertEqual(loaded[0].metadata["source"], "test")
        self.assertEqual(loaded[-1].tool_call_id, "call-1")
        self.assertEqual(loaded[-1].name, "echo")

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
        agent.thinking_history = ["thought 1", "thought 2"]

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
        self.assertEqual(restored.get_history()[0].content, "hello")
        self.assertEqual(restored.get_thinking_history(), ["thought 1", "thought 2"])
        self.assertTrue(restored.history_via_context_manager)

        listed = BasicAgent.list_sessions(store=self.session_store)
        self.assertEqual([item["session_id"] for item in listed], ["basic-1"])
        self.assertTrue(BasicAgent.delete_session("basic-1", store=self.session_store))

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
