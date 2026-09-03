import unittest
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.builder import ContextBuilder
from context.compressor.history import LLMHistoryCompactor, RuleBasedHistoryCompactor
from context.manager import ContextManager
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
from core.llm import EasyLLM
from core.providers import create_codec
from core.request_input import ReplayRequestInput


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
        self._provider = SimpleNamespace(
            get_cache_capability=lambda: SimpleNamespace(
                to_dict=lambda: {
                    "supports_explicit_cache_control": False,
                    "supports_message_level_breakpoint": False,
                    "supports_tool_cache_marker": False,
                    "supports_usage_cache_fields": False,
                    "supports_cached_content_objects": False,
                    "supports_deferred_tools": True,
                    "usage_semantics": "openai_style",
                }
            )
        )

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

    def test_compact_persistent_history_preserves_recent_turn(self):
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
        codec = create_codec("openai")
        canonical_history = codec.replay_to_canonical(request_input.replay_history)
        result = manager.compact_persistent_history(
            canonical_history,
            request_input.replay_history,
            provider_name="openai",
            token_counter=manager.counter,
            system_prompt=request_input.system_prompt,
            max_tokens=60,
        )
        request_tokens = codec.count_request_tokens(
            manager.counter,
            result.replay_history,
            system_prompt=request_input.system_prompt,
        )
        self.assertTrue(result.was_compacted)
        self.assertTrue(result.compaction_possible)
        self.assertEqual(result.replay_history[-1]["content"], "最后一轮回答")
        self.assertGreater(request_tokens, 0)

    def test_request_over_budget_and_not_compactable_is_detectable(self):
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
        codec = create_codec("openai")
        canonical_history = codec.replay_to_canonical(request_input.replay_history)
        result = manager.compact_persistent_history(
            canonical_history,
            request_input.replay_history,
            provider_name="openai",
            token_counter=manager.counter,
            system_prompt=request_input.system_prompt,
            max_tokens=40,
        )
        request_tokens = codec.count_request_tokens(
            manager.counter,
            request_input.replay_history,
            system_prompt=request_input.system_prompt,
        )
        remaining_tokens = 40 - request_tokens
        self.assertFalse(result.was_compacted)
        self.assertFalse(result.compaction_possible)
        self.assertLess(remaining_tokens, 0)
        self.assertGreater(-remaining_tokens, 0)

    async def test_llm_history_compactor_acompact_returns_canonical_summary(self):
        compactor = LLMHistoryCompactor(
            llm=MockHistoryLLM('["用户要求保留翻译结论。", "助手确认工具结果和数学结论。"]'),
            token_counter=TokenCounter(chars_per_token=1.0),
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
        compacted = await compactor.acompact(history)
        self.assertGreaterEqual(len(compacted), 1)
        self.assertTrue(all(item["record_type"] == "canonical_message" for item in compacted))
        self.assertEqual(compacted[0]["content"][0]["type"], "text")

    def test_llm_history_compactor_fallback_metadata_is_returned(self):
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
        replay = [
            {"role": "user", "content": "第一轮问题非常长" * 4},
            {"role": "assistant", "content": "第一轮回答非常长" * 4},
            {"role": "user", "content": "最后一轮问题"},
            {"role": "assistant", "content": "最后一轮回答"},
        ]
        codec = create_codec("openai")
        compacted = manager.compact_persistent_history(
            codec.replay_to_canonical(replay),
            replay,
            provider_name="openai",
            token_counter=manager.counter,
            max_tokens=120,
            force=True,
        )

        self.assertTrue(compacted.was_compacted)
        compactor_info = compacted.metadata["compactor"]
        self.assertTrue(compactor_info["fallback_used"])
        self.assertEqual(compactor_info["status"], "fallback")
        self.assertEqual(compactor_info["fallback_compactor"], "RuleBasedHistoryCompactor")
        self.assertEqual(compactor_info["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
