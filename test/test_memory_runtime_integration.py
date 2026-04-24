import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from context.manager import ContextManager
from core.llm import EasyLLM
from memory import MemoryConfig, MemoryManage


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "openai"
        self.model = "mock-model"
        self.max_tokens = 4096
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.temperature = 0.7
        self.timeout = 60
        self.kwargs = {}

    def export_tools(self, tools):
        if hasattr(tools, "export_tools"):
            return tools.export_tools("openai")
        return tools


class MemoryRuntimeIntegrationTest(unittest.TestCase):
    def test_memory_manage_defaults_to_working_only(self):
        memory_manage = MemoryManage(MemoryConfig())

        self.assertEqual(list(memory_manage.memory_types.keys()), ["working"])

    def test_working_memory_uses_context_not_system_prompt_without_external_context_manager(self):
        memory_manage = MemoryManage(MemoryConfig())
        memory_manage.add_memory(
            "wm secret 42",
            memory_type="working",
            importance=1.0,
            metadata={"source": "test"},
        )
        agent = BasicAgent(name="memory-agent", llm=DummyLLM()).with_memory(memory_manage)

        prompt = agent.get_enhanced_prompt()
        request_input = agent._build_start_messages("what is the secret?")

        self.assertIsNotNone(agent.context_manager)
        self.assertEqual(agent.context_manager.builder.source_names.count("memory"), 1)
        self.assertNotIn("wm secret 42", prompt)
        self.assertIn("wm secret 42", repr(request_input.replay_history))
        self.assertTrue(agent.tool_registry.has_tool("add_memory_tool"))

    def test_with_context_after_memory_keeps_single_memory_source(self):
        memory_manage = MemoryManage(MemoryConfig())
        memory_manage.add_memory(
            "current task constraint",
            memory_type="working",
            importance=0.9,
            metadata={"source": "test"},
        )
        agent = BasicAgent(name="memory-agent", llm=DummyLLM()).with_memory(memory_manage)
        manager = ContextManager(max_tokens=20000)

        agent.with_context(manager)
        request_input = agent._build_start_messages("continue")

        self.assertIs(agent.context_manager, manager)
        self.assertEqual(manager.builder.source_names.count("memory"), 1)
        self.assertNotIn("current task constraint", agent.get_enhanced_prompt())
        self.assertIn("current task constraint", repr(request_input.replay_history))


if __name__ == "__main__":
    unittest.main()
