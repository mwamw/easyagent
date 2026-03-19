"""
Agent + ContextManager 集成行为测试
"""
from mailbox import MMDF
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Optional

from regex import T
from sympy import false

from context.source import memory_source



from agent.BasicAgent import BasicAgent
from agent.ReactAgent import ReactAgent
from agent.ConversationalAgent import ConversationalAgent
from core.Message import UserMessage, AssistantMessage
from core.llm import EasyLLM
from context.manager import ContextManager
from Tool.ToolRegistry import ToolRegistry
from manual_test_runner import run_manual_tests, exit_with_status
from Tool.ToolRegistry import ToolRegistry
from memory.V2.MemoryManage import MemoryManage
from memory.V2.WorkingMemory import WorkingMemory
from memory import MemoryConfig
class SpyEasyLLM(EasyLLM):
    def __init__(self):
        self.provide = "mock"
        self.model = "mock-model"
        self.last_messages: list[Any] = []

    def invoke(self, messages, temperature: Optional[float] = None, **kwargs):
        self.last_messages = messages
        # 给 ReactAgent 一个可解析的完成信号
        text_blob = "\n".join(
            [str(getattr(m, "content", m)) for m in messages]
        )
        if "Final Answer" in text_blob or "Thought:" in text_blob:
            return "Final Answer: ok"
        return "ok"

    def think(self, messages, temperature: Optional[float] = None):
        yield "ok"


class TestAgentContextIntegration(unittest.TestCase):
    def setUp(self):
        self.llm = SpyEasyLLM()

    def test_default_no_context_history_in_prompt(self):
        toolRegistry=ToolRegistry()
        manager = ContextManager(max_tokens=2000, auto_history=True)
        agent = BasicAgent(
            name="a",
            llm=self.llm,
            context_manager=manager,
            history_via_context_manager=False,
            enable_tool=True,
            tool_registry=toolRegistry
        )
        agent.history = [
            UserMessage("u1"),
            AssistantMessage("a1"),
            UserMessage("u2"),
            AssistantMessage("a2"),
            UserMessage("u2"),
            AssistantMessage("a2"),
        ]
        wm=WorkingMemory(config=MemoryConfig())
        mm=MemoryManage(MemoryConfig(),working_memory=wm,enable_semantic=false,enable_episodic=false)
        mm.add_memory("test_working memory","working",importance=1.0,metadata={"source": "test"})
        agent.with_memory(mm)
        from context.source.memory_source import MemoryContextSource
        memory_source=MemoryContextSource(memory_manage=mm)
        manager.add_source(memory_source)
        prompt = agent.get_enhanced_prompt()
        
        print(manager.builder.source_names)
        print("prompt_without_ContextManager:", prompt)
        print("memory_prompt:", agent._build_memory_prompt())
        messages=agent._build_start_messages(query="测试查询")
        print("messages_without_ContextManager:", messages)
        self.assertNotIn("u1", prompt)


    def test_context_history_mode_includes_history_in_prompt(self):
        manager = ContextManager(max_tokens=2000, auto_history=True)
        agent = BasicAgent(
            name="a",
            llm=self.llm,
            context_manager=manager,
            history_via_context_manager=True,
        )
        agent.history = [
            UserMessage("u1"),
            AssistantMessage("a1"),
        ]
        agent._current_query = "q"

        prompt = agent.get_enhanced_prompt()
        self.assertNotIn("user: u1", prompt)
        self.assertNotIn("assistant: a1", prompt)

        agent.invoke("q")
        # ContextManager 接管完整起始消息：system + history(2) + current user
        self.assertEqual(len(self.llm.last_messages), 4)


class TestReactContextIntegration(unittest.TestCase):
    def setUp(self):
        self.llm = SpyEasyLLM()
        self.registry = ToolRegistry()

    def test_react_default_prompt_dedup(self):
        manager = ContextManager(max_tokens=2000, auto_history=True)
        agent = ReactAgent(
            name="react",
            llm=self.llm,
            tool_registry=self.registry,
            enable_tool=True,
            context_manager=manager,
            history_via_context_manager=False,
        )
        agent.history = [UserMessage("u1"), AssistantMessage("a1")]
        agent._current_query = "q"

        prompt = agent.get_enhanced_prompt()
        self.assertNotIn("user: u1", prompt)

    def test_react_context_history_mode_in_prompt(self):
        manager = ContextManager(max_tokens=2000, auto_history=True)
        agent = ReactAgent(
            name="react",
            llm=self.llm,
            tool_registry=self.registry,
            enable_tool=True,
            context_manager=manager,
            history_via_context_manager=True,
        )
        agent.history = [UserMessage("u1"), AssistantMessage("a1")]
        agent._current_query = "q"

        prompt = agent.get_enhanced_prompt()
        self.assertNotIn("user: u1", prompt)
        self.assertNotIn("assistant: a1", prompt)


class TestConversationalContextIntegration(unittest.TestCase):
    def setUp(self):
        self.llm = SpyEasyLLM()

    def test_conversational_default_dedup(self):
        manager = ContextManager(max_tokens=2000, auto_history=True)
        agent = ConversationalAgent(
            name="chat",
            llm=self.llm,
            context_manager=manager,
            history_via_context_manager=False,
            auto_save_to_working=False,
        )
        agent.history = [UserMessage("u1"), AssistantMessage("a1")]

        agent.invoke("q")
        self.assertEqual(len(self.llm.last_messages), 4)

    def test_conversational_context_history_mode(self):
        manager = ContextManager(max_tokens=2000, auto_history=True)
        agent = ConversationalAgent(
            name="chat",
            llm=self.llm,
            context_manager=manager,
            history_via_context_manager=True,
            auto_save_to_working=False,
        )
        agent.history = [UserMessage("u1"), AssistantMessage("a1")]

        agent.invoke("q")
        self.assertEqual(len(self.llm.last_messages), 4)


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestAgentContextIntegration,

        ],
        title="Context Builder/Manager Manual Test",
    )
    exit_with_status(ok)