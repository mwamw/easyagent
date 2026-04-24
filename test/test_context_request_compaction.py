import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.builder import ContextBuilder
from context.compressor.history import LLMHistoryCompactor, RuleBasedHistoryCompactor
from context.manager import ContextManager
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
from core.Message import AssistantMessage, UserMessage
from core.llm import EasyLLM
from core.providers import create_codec
from core.request_input import ReplayRequestInput
from agent import BasicAgent


class MockHistoryLLM:
    def __init__(self, response: str):
        self.response = response
        self.sync_messages = None
        self.async_messages = None

    def invoke(self, messages, **kwargs):
        self.sync_messages = messages
        return self.response

    async def ainvoke(self, messages, **kwargs):
        self.async_messages = messages
        return self.response


class FailingHistoryLLM(MockHistoryLLM):
    def __init__(self):
        super().__init__(response="")

    def invoke(self, messages, **kwargs):
        raise RuntimeError("llm compaction failed")

    async def ainvoke(self, messages, **kwargs):
        raise RuntimeError("llm compaction failed")


class DummyAgentLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "openai"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256

    def invoke(self, messages, temperature=None, **kwargs):
        return "ok"

    def prepare_messages_for_request(self, messages):
        return list(messages)


class TestContextRequestCompaction(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.builder = ContextBuilder(
            budget=TokenBudget(max_tokens=120),
            counter=TokenCounter(chars_per_token=1.0),
        )
        self.manager = ContextManager(builder=self.builder)

    def test_build_request_input_returns_replay_buffer(self):
        request_input = self.manager.build_request_input(
            query="继续",
            system_prompt="sys",
            replay_history=[{"role": "assistant", "content": "历史回答"}],
            provider_name="openai",
            include_query=True,
        )
        self.assertIsInstance(request_input, ReplayRequestInput)
        self.assertEqual(request_input.system_prompt, "sys")
        self.assertEqual(request_input.replay_history[-1]["role"], "user")

    def test_compact_request_input_preserves_recent_turn(self):
        manager = ContextManager(
            builder=ContextBuilder(
                budget=TokenBudget(max_tokens=60),
                counter=TokenCounter(chars_per_token=1.0),
            )
        )
        manager.set_history_compactor(
            RuleBasedHistoryCompactor(
                token_counter=manager.counter,
                recent_turns=1,
            )
        )
        request_input = ReplayRequestInput(
            provider_name="openai",
            replay_history=[
                {"role": "user", "content": "第一轮问题非常长" * 4},
                {"role": "assistant", "content": "第一轮回答非常长" * 4},
                {"role": "user", "content": "最后一轮问题"},
                {"role": "assistant", "content": "最后一轮回答"},
            ],
            system_prompt="系统提示",
        )
        result = manager.compact_request_input(request_input, max_tokens=60)
        codec = create_codec("openai")
        request_tokens = codec.count_request_tokens(
            manager.counter,
            request_input.replay_history,
            system_prompt=request_input.system_prompt,
        )
        self.assertTrue(result.was_compacted)
        self.assertLessEqual(request_tokens, 60)
        self.assertEqual(request_input.replay_history[-1]["content"], "最后一轮回答")

    def test_usage_reports_negative_remaining_when_request_is_over_budget_but_not_compactable(self):
        manager = ContextManager(
            builder=ContextBuilder(
                budget=TokenBudget(max_tokens=120),
                counter=TokenCounter(chars_per_token=1.0),
            )
        )
        request_input = ReplayRequestInput(
            provider_name="openai",
            replay_history=[
                {"role": "user", "content": "单轮问题非常长" * 3},
                {"role": "assistant", "content": "单轮回答也非常长" * 3},
            ],
            system_prompt="系统提示也很长" * 3,
        )
        result = manager.compact_request_input(request_input, max_tokens=40)
        usage = manager.analyze_messages_usage(request_input)
        self.assertFalse(result.was_compacted)
        self.assertFalse(result.compaction_possible)
        self.assertLess(usage["remaining_tokens"], 0)
        self.assertGreater(usage["overflow_tokens"], 0)
        self.assertFalse(usage["request_compacted"])
        self.assertFalse(usage["request_compaction_possible"])

    async def test_llm_history_compactor_acompact_returns_canonical_summary(self):
        compactor = LLMHistoryCompactor(
            llm=MockHistoryLLM('["用户要求保留翻译结论。", "助手确认工具结果和数学结论。"]'),
            token_counter=TokenCounter(chars_per_token=1.0),
            max_summary_messages=2,
        )
        history = [
            {
                "record_type": "canonical_message",
                "role": "user",
                "provider": "openai",
                "provider_message_type": "user",
                "content": [{"type": "text", "text": "请翻译并计算 3^22"}],
                "metadata": {},
            },
            {
                "record_type": "canonical_message",
                "role": "assistant",
                "provider": "openai",
                "provider_message_type": "assistant",
                "content": [{"type": "text", "text": "我会调用工具处理"}],
                "metadata": {},
            },
        ]
        compacted = await compactor.acompact(history, max_tokens=20)
        self.assertGreaterEqual(len(compacted), 1)
        self.assertTrue(all(item["record_type"] == "canonical_message" for item in compacted))
        self.assertEqual(compacted[0]["content"][0]["type"], "text")

    def test_llm_history_compactor_fallback_metadata_propagates_to_agent_usage(self):
        manager = ContextManager(
            builder=ContextBuilder(
                budget=TokenBudget(max_tokens=40),
                counter=TokenCounter(chars_per_token=1.0),
            )
        )
        manager.set_history_compactor(
            LLMHistoryCompactor(
                llm=FailingHistoryLLM(),
                token_counter=manager.counter,
                recent_turns=1,
            )
        )
        agent = BasicAgent(
            name="assistant",
            llm=DummyAgentLLM(),
            context_manager=manager,
        )
        agent.get_enhanced_prompt = lambda: ""
        agent.add_message(UserMessage("第一轮问题非常长" * 4))
        agent.add_message(AssistantMessage("第一轮回答非常长" * 4))
        agent.add_message(UserMessage("最后一轮问题"))
        agent.add_message(AssistantMessage("最后一轮回答"))

        compacted = agent.compact_history(max_tokens=120)

        self.assertTrue(compacted)
        usage = agent.get_context_usage()
        compactor_info = usage["last_history_compaction"]["metadata"]["compactor"]
        self.assertTrue(compactor_info["fallback_used"])
        self.assertEqual(compactor_info["status"], "fallback")
        self.assertEqual(compactor_info["fallback_compactor"], "RuleBasedHistoryCompactor")
        self.assertEqual(compactor_info["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
