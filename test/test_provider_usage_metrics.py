from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import BasicAgent
from core.llm import EasyLLM
from core.providers import AnthropicNativeProvider, GoogleNativeProvider, create_codec
from observability import InMemoryObservabilityStore
from runtime import AgentStreamEventType


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _openai_stream_chunk(*, content: str | None = None, finish_reason: str | None = None, usage=None):
    delta = _ns(content=content, reasoning_content=None, reasoning=None, tool_calls=None)
    if usage is not None:
        return _ns(choices=[], usage=usage)
    choice = _ns(delta=delta, finish_reason=finish_reason)
    return _ns(choices=[choice], usage=None)


class _FakeOpenAIChatCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return [
                _openai_stream_chunk(content="hello "),
                _openai_stream_chunk(content="world", finish_reason="stop"),
                _openai_stream_chunk(
                    usage=_ns(
                        prompt_tokens=11,
                        completion_tokens=7,
                        total_tokens=18,
                        prompt_tokens_details=_ns(cached_tokens=4),
                        completion_tokens_details=_ns(reasoning_tokens=3),
                    )
                ),
            ]
        return _ns(
            choices=[
                _ns(
                    message=_ns(
                        content="hello world",
                        reasoning_content=None,
                        tool_calls=[],
                    )
                )
            ],
            usage=_ns(
                prompt_tokens=13,
                completion_tokens=5,
                total_tokens=18,
                prompt_tokens_details=_ns(cached_tokens=2),
                completion_tokens_details=_ns(reasoning_tokens=1),
            ),
        )


class _FakeOpenAIChatClient:
    def __init__(self):
        self.chat = _ns(completions=_FakeOpenAIChatCompletions())


class _FakeResponsesAPI:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return [
                _ns(type="response.output_text.delta", delta="hello "),
                _ns(type="response.output_text.delta", delta="world"),
                _ns(
                    type="response.completed",
                    response=_ns(
                        usage=_ns(
                            input_tokens=17,
                            output_tokens=9,
                            total_tokens=26,
                            input_tokens_details=_ns(cached_tokens=5),
                            output_tokens_details=_ns(reasoning_tokens=4),
                        )
                    ),
                ),
            ]
        return _ns(
            usage=_ns(
                input_tokens=15,
                output_tokens=6,
                total_tokens=21,
                input_tokens_details=_ns(cached_tokens=3),
                output_tokens_details=_ns(reasoning_tokens=2),
            )
        )


class _FakeOpenAIResponsesClient:
    def __init__(self):
        self.responses = _FakeResponsesAPI()


class TestProviderUsageMetrics(unittest.TestCase):
    def test_openai_chat_non_stream_usage_comes_from_provider_response(self):
        client = _FakeOpenAIChatClient()
        llm = EasyLLM(
            provider="openai",
            model="gpt-4o-mini",
            api_key="key",
            base_url="http://mock.local/v1",
            client=client,
        )

        response = llm.invoke_raw([{"role": "user", "content": "hello"}])
        usage = llm.extract_usage_metrics(response)

        self.assertEqual(response.content, "hello world")
        self.assertEqual(usage["inputTokens"], 13)
        self.assertEqual(usage["outputTokens"], 5)
        self.assertEqual(usage["totalTokens"], 18)
        self.assertEqual(usage["cachedInputTokens"], 2)
        self.assertEqual(usage["reasoningTokens"], 1)
        self.assertEqual(usage["usageSource"], "provider")

    def test_openai_chat_stream_events_include_usage_and_stream_options(self):
        client = _FakeOpenAIChatClient()
        llm = EasyLLM(
            provider="openai",
            model="gpt-4o-mini",
            api_key="key",
            base_url="http://mock.local/v1",
            client=client,
        )

        events = list(llm.stream_events([{"role": "user", "content": "stream"}]))

        self.assertTrue(events)
        final_event = events[-1]
        self.assertEqual(final_event["type"], "final_response")
        self.assertEqual(final_event["content"], "hello world")
        usage = llm.extract_usage_metrics(final_event)
        self.assertEqual(usage["inputTokens"], 11)
        self.assertEqual(usage["outputTokens"], 7)
        self.assertEqual(usage["cachedInputTokens"], 4)
        self.assertEqual(usage["reasoningTokens"], 3)

        request = client.chat.completions.calls[-1]
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"]["include_usage"], True)

    def test_openai_responses_provider_and_stream_usage_are_extracted(self):
        client = _FakeOpenAIResponsesClient()
        llm = EasyLLM(
            provider="openai_responses",
            model="gpt-5.4-mini",
            api_key="key",
            base_url="http://mock.local/v1",
            client=client,
        )

        response = llm.invoke_raw([{"role": "user", "content": "hello"}])
        usage = llm.extract_usage_metrics(response)
        self.assertEqual(usage["inputTokens"], 15)
        self.assertEqual(usage["outputTokens"], 6)
        self.assertEqual(usage["cachedInputTokens"], 3)
        self.assertEqual(usage["reasoningTokens"], 2)

        events = list(llm.stream_events([{"role": "user", "content": "stream"}]))
        final_event = events[-1]
        self.assertEqual(final_event["type"], "final_response")
        stream_usage = llm.extract_usage_metrics(final_event)
        self.assertEqual(stream_usage["inputTokens"], 17)
        self.assertEqual(stream_usage["outputTokens"], 9)
        self.assertEqual(stream_usage["cachedInputTokens"], 5)
        self.assertEqual(stream_usage["reasoningTokens"], 4)

    def test_anthropic_native_provider_and_stream_usage_are_extracted(self):
        provider = AnthropicNativeProvider(
            model="claude-4.5-sonnet",
            api_key="key",
            base_url="http://mock.local",
            client=_ns(messages=_ns()),
        )
        response = _ns(
            content=[_ns(type="text", text="done")],
            usage=_ns(
                input_tokens=19,
                output_tokens=8,
                cache_read_input_tokens=6,
                cache_creation_input_tokens=2,
            ),
        )
        usage = provider.get_usage_from_response(response)
        self.assertEqual(usage["inputTokens"], 19)
        self.assertEqual(usage["outputTokens"], 8)
        self.assertEqual(usage["cacheReadTokens"], 6)
        self.assertEqual(usage["cacheCreationTokens"], 2)
        self.assertEqual(usage["cachedInputTokens"], 6)

        codec = create_codec("anthropic_native")
        stream_events = list(
            codec.stream_events(
                [
                    {"type": "message_start", "message": {"usage": {"input_tokens": 21, "output_tokens": 0}}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "hello"}},
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": None,
                            "usage": {
                                "input_tokens": 21,
                                "output_tokens": 9,
                                "cache_read_input_tokens": 7,
                            },
                        },
                    },
                    {"type": "message_stop"},
                ]
            )
        )
        final_event = stream_events[-1]
        self.assertEqual(final_event["type"], "final_response")
        stream_usage = provider.get_usage_from_response(final_event)
        self.assertEqual(stream_usage["inputTokens"], 21)
        self.assertEqual(stream_usage["outputTokens"], 9)
        self.assertEqual(stream_usage["cacheReadTokens"], 7)

    def test_google_native_provider_and_stream_usage_are_extracted(self):
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="key",
            base_url="",
            client=_ns(models=_ns()),
        )
        response = _ns(
            text="done",
            usage_metadata=_ns(
                prompt_token_count=23,
                candidates_token_count=11,
                total_token_count=34,
                cached_content_token_count=8,
                thoughts_token_count=5,
                tool_use_prompt_token_count=3,
            ),
        )
        usage = provider.get_usage_from_response(response)
        self.assertEqual(usage["inputTokens"], 23)
        self.assertEqual(usage["outputTokens"], 11)
        self.assertEqual(usage["totalTokens"], 34)
        self.assertEqual(usage["cachedInputTokens"], 8)
        self.assertEqual(usage["reasoningTokens"], 5)
        self.assertEqual(usage["toolUsePromptTokens"], 3)

        codec = create_codec("google_native")
        chunks = [
            _ns(
                text="hello",
                usage_metadata=_ns(
                    prompt_token_count=24,
                    candidates_token_count=12,
                    total_token_count=36,
                    cached_content_token_count=9,
                    thoughts_token_count=6,
                    tool_use_prompt_token_count=4,
                ),
                candidates=[_ns(content=_ns(role="model", parts=[_ns(text="hello")]))],
            )
        ]
        stream_events = list(codec.stream_events(chunks))
        final_event = stream_events[-1]
        self.assertEqual(final_event["type"], "final_response")
        stream_usage = provider.get_usage_from_response(final_event)
        self.assertEqual(stream_usage["inputTokens"], 24)
        self.assertEqual(stream_usage["outputTokens"], 12)
        self.assertEqual(stream_usage["cachedInputTokens"], 9)
        self.assertEqual(stream_usage["reasoningTokens"], 6)
        self.assertEqual(stream_usage["toolUsePromptTokens"], 4)

    def test_basic_agent_stream_observability_uses_provider_usage(self):
        client = _FakeOpenAIChatClient()
        llm = EasyLLM(
            provider="openai",
            model="gpt-4o-mini",
            api_key="key",
            base_url="http://mock.local/v1",
            client=client,
        )
        agent = BasicAgent(name="usage-observability", llm=llm).with_observability(
            store=InMemoryObservabilityStore()
        )

        events = list(agent.stream("stream with usage"))

        self.assertEqual(events[-1].type, AgentStreamEventType.FINAL)
        self.assertEqual(events[-1].content, "hello world")
        latest = agent.observability.latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.llm_invokes[-1].stats.input_tokens, 11)
        self.assertEqual(latest.llm_invokes[-1].stats.output_tokens, 7)
        self.assertEqual(latest.llm_invokes[-1].stats.cached_input_tokens, 4)
        self.assertEqual(latest.llm_invokes[-1].stats.reasoning_tokens, 3)

        summary = agent.observability.summary()
        self.assertEqual(summary["input_tokens"], 11)
        self.assertEqual(summary["output_tokens"], 7)
        self.assertEqual(summary["total_tokens"], 18)
        self.assertEqual(summary["cached_input_tokens"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
