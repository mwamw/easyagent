import os
import sys
import unittest
import asyncio
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.providers import create_provider, provider_requires_base_url, detect_provider_from_model
from core.providers.anthropic_native_provider import AnthropicNativeProvider
from core.providers.google_native_provider import GoogleNativeProvider


class _FakeGoogleModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text="hello",
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        role="model",
                        parts=[SimpleNamespace(text="hello")],
                    )
                )
            ],
        )


class _FakeAsyncGoogleModels:
    def __init__(self):
        self.calls = []

    async def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)

        async def _stream():
            yield SimpleNamespace(
                text="hello",
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            role="model",
                            parts=[SimpleNamespace(text="hello")],
                        )
                    )
                ],
            )

        return _stream()


class _FakeGoogleStreamModels:
    def __init__(self):
        self.calls = []

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            role="model",
                            parts=[
                                SimpleNamespace(text="plan", thought=True, thought_signature="sig-1"),
                                SimpleNamespace(
                                    function_call=SimpleNamespace(
                                        id="call_1",
                                        name="weather",
                                        args={"city": "Beijing"},
                                    )
                                ),
                            ],
                        )
                    )
                ],
            )
        ]


class _FakeGoogleClient:
    def __init__(self):
        self.models = _FakeGoogleModels()
        self.aio = None


class _FakeGoogleAsyncClient:
    def __init__(self):
        self.models = _FakeAsyncGoogleModels()


class _FakeGoogleStreamClient:
    def __init__(self):
        self.models = _FakeGoogleStreamModels()


class _FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
        )


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeAnthropicMessages()


class _FakeAnthropicAsyncStream:
    def __init__(self, events):
        self._events = list(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        async def _iter():
            for event in self._events:
                yield event
        return _iter()


class _FakeAnthropicAsyncMessages:
    def __init__(self, events):
        self._events = events
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeAnthropicAsyncStream(self._events)


class _FakeAnthropicAsyncClient:
    def __init__(self, events):
        self.messages = _FakeAnthropicAsyncMessages(events)


class TestNativeProviders(unittest.TestCase):
    def test_google_provider_invoke_with_tools_builds_native_contents(self):
        client = _FakeGoogleClient()
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=client,
        )
        assistant_message = provider.format_assistant_message(
            content="checking",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "weather",
                    "arguments": "{\"city\": \"Beijing\"}",
                }
            ],
        )
        tool_message = provider.format_tool_result("sunny", "call_1", "weather")

        provider.invoke_with_tools(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "what is the weather"},
                assistant_message,
                tool_message,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        call = client.models.calls[0]
        self.assertEqual(call["config"]["system_instruction"], "system prompt")
        self.assertEqual(call["config"]["automatic_function_calling"], {"disable": True})
        self.assertEqual(call["config"]["tools"][0]["function_declarations"][0]["name"], "weather")
        self.assertEqual(call["contents"][0]["role"], "user")
        self.assertEqual(call["contents"][1]["role"], "model")
        self.assertEqual(call["contents"][1]["parts"][1]["function_call"]["name"], "weather")
        self.assertEqual(call["contents"][2]["parts"][0]["function_response"]["response"], {"result": "sunny"})

    def test_google_provider_format_assistant_response_preserves_thinking_and_function_call(self):
        provider = GoogleNativeProvider(model="gemini-2.5-pro", api_key="k", base_url="", client=object())
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        role="model",
                        parts=[
                            SimpleNamespace(text="plan", thought=True, thought_signature="sig-1"),
                            SimpleNamespace(text="checking"),
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    id="call_1",
                                    name="weather",
                                    args={"city": "Beijing"},
                                )
                            ),
                        ],
                    )
                )
            ]
        )

        message = provider.format_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"][0]["type"], "thinking")
        self.assertEqual(message["content"][0]["thought_signature"], "sig-1")
        self.assertEqual(message["content"][1], {"type": "text", "text": "checking"})
        self.assertEqual(message["content"][2]["type"], "function_call")
        self.assertEqual(message["content"][2]["name"], "weather")
        self.assertEqual(provider.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_google_provider_preserves_function_call_thought_signature_for_replay(self):
        client = _FakeGoogleClient()
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=client,
        )
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        role="model",
                        parts=[
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    id="call_1",
                                    name="weather",
                                    args={"city": "Beijing"},
                                ),
                                thought_signature="sig-call-1",
                            )
                        ],
                    )
                )
            ]
        )

        assistant_message = provider.format_assistant_response(response, include_reasoning=True)
        self.assertEqual(assistant_message["content"][0]["thought_signature"], "sig-call-1")

        provider.invoke_with_tools(
            [
                {"role": "user", "content": "weather"},
                assistant_message,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        call = client.models.calls[0]
        self.assertEqual(call["contents"][1]["parts"][0]["thought_signature"], "sig-call-1")
        self.assertEqual(call["contents"][1]["parts"][0]["function_call"]["name"], "weather")

    def test_google_provider_handles_real_sdk_part_objects(self):
        try:
            from google.genai import types as gt
        except Exception as exc:
            self.skipTest(f"google-genai unavailable: {exc}")
        provider = GoogleNativeProvider(model="gemini-2.5-pro", api_key="k", base_url="", client=object())
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=gt.Content(
                        role="model",
                        parts=[
                            gt.Part(text="plan", thought=True, thought_signature=b"sig-1"),
                            gt.Part(text="checking"),
                            gt.Part(functionCall=gt.FunctionCall(id="call_1", name="weather", args={"city": "Beijing"})),
                        ],
                    )
                )
            ]
        )

        message = provider.format_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["content"][0]["type"], "thinking")
        self.assertEqual(message["content"][0]["text"], "plan")
        self.assertEqual(message["content"][0]["thought_signature"], b"sig-1")
        self.assertEqual(message["content"][2]["type"], "function_call")
        self.assertEqual(message["content"][2]["id"], "call_1")
        self.assertEqual(provider.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_google_provider_prepare_messages_merges_function_responses(self):
        provider = GoogleNativeProvider(model="gemini-2.5-pro", api_key="k", base_url="", client=object())
        prepared = provider.prepare_messages_for_request(
            [
                provider.format_assistant_message(
                    tool_calls=[
                        {"id": "call_1", "name": "weather", "arguments": "{\"city\": \"Beijing\"}"},
                        {"id": "call_2", "name": "time", "arguments": "{\"zone\": \"Asia/Shanghai\"}"},
                    ]
                ),
                provider.format_tool_result("sunny", "call_1", "weather"),
                provider.format_tool_result("08:00", "call_2", "time"),
            ]
        )

        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared[0]["role"], "assistant")
        self.assertEqual(prepared[1]["role"], "user")
        self.assertEqual(len(prepared[1]["content"]), 2)
        self.assertEqual(prepared[1]["content"][0]["id"], "call_1")
        self.assertEqual(prepared[1]["content"][1]["id"], "call_2")

    def test_google_provider_async_stream_events_awaits_stream_factory(self):
        async_client = _FakeGoogleAsyncClient()
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=object(),
            async_client=async_client,
        )

        async def _collect():
            events = []
            async for event in provider.async_stream_events(
                [{"role": "user", "content": "hello"}]
            ):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        self.assertEqual(async_client.models.calls[0]["model"], "gemini-2.5-pro")
        self.assertEqual(events[0], {"type": "text_delta", "delta": "hello"})
        self.assertEqual(events[-1]["type"], "final_response")
        self.assertEqual(events[-1]["content"], "hello")

    def test_google_provider_stream_with_tools_preserves_thought_signature(self):
        client = _FakeGoogleStreamClient()
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=client,
        )

        events = list(
            provider.stream_with_tools(
                [{"role": "user", "content": "weather"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "Get weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )

        tool_event = events[-1]
        self.assertEqual(tool_event["type"], "tool_calls")
        self.assertEqual(tool_event["assistant_items"]["content"][0]["type"], "thinking")
        self.assertEqual(tool_event["assistant_items"]["content"][0]["thought_signature"], "sig-1")
        self.assertEqual(tool_event["assistant_items"]["content"][1]["type"], "function_call")

    def test_anthropic_provider_invoke_with_tools_builds_native_messages(self):
        client = _FakeAnthropicClient()
        provider = AnthropicNativeProvider(
            model="claude-sonnet-4-5",
            api_key="k",
            base_url="",
            client=client,
        )
        assistant_message = provider.format_assistant_message(
            content="checking",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "weather",
                    "arguments": "{\"city\": \"Beijing\"}",
                }
            ],
        )
        tool_message = provider.format_tool_result("sunny", "call_1", "weather")

        provider.invoke_with_tools(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "what is the weather"},
                assistant_message,
                tool_message,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        call = client.messages.calls[0]
        self.assertEqual(call["system"], "system prompt")
        self.assertEqual(call["tools"][0]["name"], "weather")
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertEqual(call["messages"][1]["role"], "assistant")
        self.assertEqual(call["messages"][1]["content"][1]["type"], "tool_use")
        self.assertEqual(call["messages"][2]["content"][0]["type"], "tool_result")

    def test_anthropic_provider_format_assistant_response_preserves_thinking_blocks(self):
        provider = AnthropicNativeProvider(model="claude-sonnet-4-5", api_key="k", base_url="", client=object())
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="plan", signature="sig-1"),
                SimpleNamespace(type="text", text="checking"),
                SimpleNamespace(type="tool_use", id="call_1", name="weather", input={"city": "Beijing"}),
            ]
        )

        message = provider.format_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["content"][0], {"type": "thinking", "thinking": "plan", "signature": "sig-1"})
        self.assertEqual(message["content"][1], {"type": "text", "text": "checking"})
        self.assertEqual(message["content"][2]["type"], "tool_use")
        self.assertEqual(provider.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_anthropic_provider_handles_real_sdk_block_objects(self):
        try:
            from anthropic import types as at
        except Exception as exc:
            self.skipTest(f"anthropic unavailable: {exc}")
        provider = AnthropicNativeProvider(model="claude-sonnet-4-5", api_key="k", base_url="", client=object())
        response = SimpleNamespace(
            content=[
                at.ThinkingBlock(type="thinking", thinking="plan", signature="sig-1"),
                at.TextBlock(type="text", text="checking"),
                at.ToolUseBlock(type="tool_use", id="call_1", name="weather", input={"city": "Beijing"}),
            ]
        )

        message = provider.format_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["content"][0], {"type": "thinking", "thinking": "plan", "signature": "sig-1"})
        self.assertEqual(message["content"][1], {"type": "text", "text": "checking"})
        self.assertEqual(message["content"][2]["type"], "tool_use")
        self.assertEqual(message["content"][2]["name"], "weather")
        self.assertEqual(provider.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_anthropic_provider_stream_tool_calls_preserve_input_json_delta_arguments(self):
        provider = AnthropicNativeProvider(model="claude-sonnet-4-5", api_key="k", base_url="", client=object())
        state = provider._init_anthropic_stream_state()
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Beijing\"}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]

        emitted = []
        for event in events:
            emitted.extend(provider._extract_anthropic_stream_events(event, state))

        self.assertEqual(emitted[-1]["type"], "tool_calls")
        self.assertEqual(emitted[-1]["tool_calls"][0]["arguments"], {"city": "Beijing"})
        assistant_message = provider._build_stream_assistant_message(state)
        self.assertEqual(assistant_message["content"][0]["type"], "tool_use")
        self.assertEqual(assistant_message["content"][0]["input"], {"city": "Beijing"})

    def test_anthropic_provider_stream_tool_calls_fallback_to_start_block_input(self):
        provider = AnthropicNativeProvider(model="claude-sonnet-4-5", api_key="k", base_url="", client=object())
        state = provider._init_anthropic_stream_state()
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Beijing"}}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]

        emitted = []
        for event in events:
            emitted.extend(provider._extract_anthropic_stream_events(event, state))

        self.assertEqual(emitted[-1]["type"], "tool_calls")
        self.assertEqual(emitted[-1]["tool_calls"][0]["arguments"], {"city": "Beijing"})

    def test_anthropic_provider_stream_thinking_signature_is_not_emitted_as_text(self):
        provider = AnthropicNativeProvider(model="claude-sonnet-4-5", api_key="k", base_url="", client=object())
        state = provider._init_anthropic_stream_state()

        thinking_events = provider._extract_anthropic_stream_events(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "plan"}},
            state,
        )
        signature_events = provider._extract_anthropic_stream_events(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
            state,
        )

        self.assertEqual(thinking_events, [{"type": "thinking_delta", "delta": "plan"}])
        self.assertEqual(signature_events, [])
        self.assertEqual(state["thinking_parts"], ["plan"])
        assistant_message = provider._build_stream_assistant_message(state)
        self.assertEqual(
            assistant_message["content"][0],
            {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
        )
        self.assertEqual(assistant_message["reasoning_content"], "plan")

    def test_anthropic_provider_stream_builds_thinking_block_from_accumulated_text(self):
        provider = AnthropicNativeProvider(model="claude-sonnet-4-5", api_key="k", base_url="", client=object())
        state = provider._init_anthropic_stream_state()
        state["thinking_parts"] = ["plan"]
        state["assistant_blocks"] = {
            1: {"type": "text", "text": "先处理工具"},
            2: {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}},
        }
        state["tool_calls"] = {
            2: {"id": "call_1", "name": "weather", "input_json": "", "input": {"city": "Beijing"}}
        }

        assistant_message = provider._build_stream_assistant_message(state)

        self.assertEqual(
            assistant_message["content"][0],
            {"type": "thinking", "thinking": "plan"},
        )
        self.assertEqual(assistant_message["content"][1], {"type": "text", "text": "先处理工具"})
        self.assertEqual(assistant_message["content"][2]["type"], "tool_use")
        self.assertEqual(assistant_message["content"][2]["input"], {"city": "Beijing"})
        self.assertEqual(assistant_message["reasoning_content"], "plan")

    def test_provider_factory_keeps_compat_and_native_tracks_separate(self):
        google_compat = create_provider("google", model="gemini-2.5-pro", api_key="k", base_url="http://localhost", client=object())
        google_native = create_provider("google_native", model="gemini-2.5-pro", api_key="k", base_url="", client=object())
        anthropic_compat = create_provider("anthropic", model="claude-sonnet-4-5", api_key="k", base_url="http://localhost", client=object())
        anthropic_native = create_provider("anthropic_native", model="claude-sonnet-4-5", api_key="k", base_url="", client=object())

        from core.providers.google_provider import GoogleProvider
        from core.providers.anthropic_provider import AnthropicProvider

        self.assertIsInstance(google_compat, GoogleProvider)
        self.assertIsInstance(google_native, GoogleNativeProvider)
        self.assertIsInstance(anthropic_compat, AnthropicProvider)
        self.assertIsInstance(anthropic_native, AnthropicNativeProvider)

    def test_provider_detection_prefers_native_for_official_models(self):
        self.assertEqual(detect_provider_from_model("gemini-2.5-pro"), "google_native")
        self.assertEqual(detect_provider_from_model("claude-4.5-sonnet"), "anthropic_native")
        self.assertFalse(provider_requires_base_url("google_native"))
        self.assertFalse(provider_requires_base_url("anthropic_native"))
        self.assertTrue(provider_requires_base_url("google"))
        self.assertTrue(provider_requires_base_url("anthropic"))


if __name__ == "__main__":
    unittest.main()
