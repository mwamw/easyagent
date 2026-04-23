import os
import sys
import unittest
import asyncio
import base64
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.providers import (
    AnthropicNativeProvider,
    GoogleNativeProvider,
    create_codec,
    create_provider,
    detect_provider_from_model,
    provider_requires_base_url,
)
from core.request_input import ReplayRequestInput


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


class _RetryingGoogleAsyncModels:
    def __init__(self):
        self.calls = []

    async def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("Unable to submit request because Thought signature is not valid.")

        async def _stream():
            yield SimpleNamespace(
                text="fallback-ok",
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            role="model",
                            parts=[SimpleNamespace(text="fallback-ok")],
                        )
                    )
                ],
            )

        return _stream()


class _RetryingGoogleAsyncClient:
    def __init__(self):
        self.models = _RetryingGoogleAsyncModels()


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
        codec = create_codec("google_native")
        assistant_message = codec.build_assistant_message(
            content="checking",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "weather",
                    "arguments": "{\"city\": \"Beijing\"}",
                }
            ],
        )
        tool_message = codec.build_tool_result("sunny", "call_1", "weather")
        replay_history = codec.prepare_messages(
            [
                {"role": "user", "content": "what is the weather"},
                assistant_message,
                tool_message,
            ]
        )

        request = provider.build_request(
            replay_history,
            system_prompt="system prompt",
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
        provider.invoke_raw(request)

        call = client.models.calls[0]
        self.assertEqual(call["config"]["system_instruction"], "system prompt")
        self.assertEqual(call["config"]["automatic_function_calling"], {"disable": True})
        self.assertEqual(call["config"]["tools"][0]["function_declarations"][0]["name"], "weather")
        self.assertEqual(call["contents"][0]["role"], "user")
        self.assertEqual(call["contents"][1]["role"], "model")
        self.assertEqual(call["contents"][1]["parts"][1]["function_call"]["name"], "weather")
        self.assertEqual(call["contents"][2]["parts"][0]["function_response"]["response"], {"result": "sunny"})

    def test_google_provider_format_assistant_response_preserves_thinking_and_function_call(self):
        codec = create_codec("google_native")
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

        message = codec.build_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["role"], "model")
        self.assertTrue(message["parts"][0]["thought"])
        self.assertEqual(message["parts"][0]["thought_signature"], "sig-1")
        self.assertEqual(message["parts"][1], {"text": "checking"})
        self.assertEqual(message["parts"][2]["function_call"]["name"], "weather")
        self.assertEqual(codec.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

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

        codec = create_codec("google_native")
        assistant_message = codec.build_assistant_response(response, include_reasoning=True)
        self.assertEqual(assistant_message["parts"][0]["thought_signature"], "sig-call-1")

        request = provider.build_request(
            codec.prepare_messages([{"role": "user", "content": "weather"}, assistant_message]),
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
        provider.invoke_raw(request)

        call = client.models.calls[0]
        self.assertEqual(call["contents"][1]["parts"][0]["thought_signature"], "sig-call-1")
        self.assertEqual(call["contents"][1]["parts"][0]["function_call"]["name"], "weather")

    def test_google_provider_handles_real_sdk_part_objects(self):
        try:
            from google.genai import types as gt
        except Exception as exc:
            self.skipTest(f"google-genai unavailable: {exc}")
        codec = create_codec("google_native")
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

        message = codec.build_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["content"][0]["type"], "thinking")
        self.assertEqual(message["content"][0]["text"], "plan")
        self.assertEqual(message["content"][0]["thought_signature"], b"sig-1")
        self.assertEqual(message["content"][2]["type"], "function_call")
        self.assertEqual(message["content"][2]["id"], "call_1")
        self.assertEqual(codec.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_google_provider_prepare_messages_merges_function_responses(self):
        codec = create_codec("google_native")
        prepared = codec.prepare_messages(
            [
                codec.build_assistant_message(
                    tool_calls=[
                        {"id": "call_1", "name": "weather", "arguments": "{\"city\": \"Beijing\"}"},
                        {"id": "call_2", "name": "time", "arguments": "{\"zone\": \"Asia/Shanghai\"}"},
                    ]
                ),
                codec.build_tool_result("sunny", "call_1", "weather"),
                codec.build_tool_result("08:00", "call_2", "time"),
            ]
        )

        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared[0]["role"], "model")
        self.assertEqual(prepared[1]["role"], "user")
        self.assertEqual(len(prepared[1]["parts"]), 2)
        self.assertEqual(prepared[1]["parts"][0]["function_response"]["id"], "call_1")
        self.assertEqual(prepared[1]["parts"][1]["function_response"]["id"], "call_2")

    def test_google_provider_async_stream_events_awaits_stream_factory(self):
        async_client = _FakeGoogleAsyncClient()
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=object(),
            async_client=async_client,
        )

        codec = create_codec("google_native")

        async def _collect():
            request = provider.build_request(
                codec.prepare_messages([{"role": "user", "content": "hello"}]),
                stream=True,
            )
            raw_stream = await provider.async_stream_raw(request)
            events = []
            async for event in codec.astream_events(raw_stream):
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

        codec = create_codec("google_native")
        request = provider.build_request(
            codec.prepare_messages([{"role": "user", "content": "weather"}]),
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
            stream=True,
        )
        events = list(codec.stream_events(provider.stream_raw(request), tools=True))

        tool_event = events[-1]
        self.assertEqual(tool_event["type"], "tool_calls")
        assistant_message = tool_event["assistant_items"][0]
        self.assertTrue(assistant_message["parts"][0]["thought"])
        self.assertEqual(assistant_message["parts"][0]["thought_signature"], "sig-1")
        self.assertIn("function_call", assistant_message["parts"][1])

    def test_google_provider_request_buffer_encodes_binary_thought_signature(self):
        request_input = ReplayRequestInput(
            provider_name="google_native",
            replay_history=[{"role": "user", "parts": [{"text": "weather"}]}],
        )
        raw_signature = b"\n$\x01\x8f=k"
        request_input.append_replay(
            {
                "role": "model",
                "parts": [
                    {
                        "text": "plan",
                        "thought": True,
                        "thought_signature": raw_signature,
                    }
                ],
            }
        )
        signature = request_input.replay_history[-1]["parts"][0]["thought_signature"]
        self.assertEqual(signature, base64.b64encode(raw_signature).decode("ascii"))

    def test_google_provider_builds_fallback_request_without_signed_model_turn(self):
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=_FakeGoogleClient(),
        )
        request = {
            "model": "gemini-2.5-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "使用工具计算3^22"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": "call_1",
                                "name": "calculator",
                                "args": {"expression": "3**22"},
                            },
                            "thought_signature": "sig-1",
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "id": "call_1",
                                "name": "calculator",
                                "response": {"result": "31381059609"},
                            }
                        }
                    ],
                },
            ],
            "config": {"tools": []},
        }

        fallback = provider._build_invalid_signature_fallback_request(request)
        self.assertIsNotNone(fallback)
        self.assertEqual([item["role"] for item in fallback["contents"]], ["user", "user"])
        self.assertEqual(
            fallback["contents"][1]["parts"][0]["function_response"]["response"],
            {"result": "31381059609"},
        )

    def test_google_provider_builds_fallback_request_by_stripping_stale_thought_signatures(self):
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=_FakeGoogleClient(),
        )
        request = {
            "model": "gemini-2.5-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "hello"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": "private plan",
                            "thought": True,
                            "thought_signature": "stale-anthropic-signature",
                        },
                        {"text": "visible answer"},
                    ],
                },
            ],
            "config": {},
        }

        fallback = provider._build_invalid_signature_fallback_request(request)
        self.assertIsNotNone(fallback)
        self.assertNotIn("thought_signature", repr(fallback))
        self.assertIn("thought_signature", repr(request))
        self.assertEqual(fallback["contents"][1]["parts"][0]["text"], "private plan")

    def test_google_provider_async_stream_raw_retries_invalid_thought_signature(self):
        async_client = _RetryingGoogleAsyncClient()
        provider = GoogleNativeProvider(
            model="gemini-2.5-pro",
            api_key="k",
            base_url="",
            client=object(),
            async_client=async_client,
        )
        request = {
            "model": "gemini-2.5-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "使用工具计算3^22"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": "call_1",
                                "name": "calculator",
                                "args": {"expression": "3**22"},
                            },
                            "thought_signature": "sig-1",
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "id": "call_1",
                                "name": "calculator",
                                "response": {"result": "31381059609"},
                            }
                        }
                    ],
                },
            ],
            "config": {"tools": []},
        }

        async def _collect():
            stream = await provider.async_stream_raw(request)
            chunks = []
            async for chunk in stream:
                chunks.append(chunk.text)
            return chunks

        chunks = asyncio.run(_collect())
        self.assertEqual(chunks, ["fallback-ok"])
        self.assertEqual(len(async_client.models.calls), 2)
        self.assertEqual(
            [item["role"] for item in async_client.models.calls[1]["contents"]],
            ["user", "user"],
        )

    def test_google_provider_canonical_replay_preserves_function_call_signature(self):
        codec = create_codec("google_native")
        canonical = codec.history_entry_to_canonical(
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call_1",
                            "name": "weather",
                            "args": {"city": "Beijing"},
                        },
                        "thought_signature": b"sig-call-1",
                    }
                ],
            }
        )
        replay = codec.prepare_messages(canonical)
        self.assertEqual(
            replay[0]["parts"][0]["thought_signature"],
            base64.b64encode(b"sig-call-1").decode("ascii"),
        )

    def test_google_replay_preserves_anthropic_thinking_without_cross_provider_signature(self):
        anthropic_codec = create_codec("anthropic_native")
        google_codec = create_codec("google_native")
        claude_alias_codec = create_codec("claude_native")
        canonical = anthropic_codec.history_entry_to_canonical(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private plan", "signature": "anthropic-sig"},
                    {"type": "text", "text": "visible answer"},
                ],
            }
        )

        google_replay = google_codec.canonical_to_replay(canonical)
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

        claude_replay = claude_alias_codec.canonical_to_replay(canonical)
        self.assertEqual(claude_replay[0]["content"][0]["signature"], "anthropic-sig")
        self.assertEqual(canonical[0].content[0].signature, "anthropic-sig")

    def test_anthropic_replay_preserves_google_thinking_without_cross_provider_signature(self):
        google_codec = create_codec("google_native")
        anthropic_codec = create_codec("anthropic_native")
        gemini_alias_codec = create_codec("gemini_native")
        canonical = google_codec.history_entry_to_canonical(
            {
                "role": "model",
                "parts": [
                    {"text": "private plan", "thought": True, "thought_signature": "google-sig"},
                    {"text": "visible answer"},
                ],
            }
        )

        anthropic_replay = anthropic_codec.canonical_to_replay(canonical)
        anthropic_replay_text = repr(anthropic_replay)
        self.assertEqual(
            anthropic_replay,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private plan"},
                        {"type": "text", "text": "visible answer"},
                    ],
                }
            ],
        )
        self.assertNotIn("google-sig", anthropic_replay_text)
        self.assertIn("private plan", anthropic_replay_text)

        gemini_replay = gemini_alias_codec.canonical_to_replay(canonical)
        self.assertEqual(gemini_replay[0]["parts"][0]["thought_signature"], "google-sig")
        self.assertEqual(canonical[0].content[0].signature, "google-sig")

    def test_anthropic_provider_invoke_with_tools_builds_native_messages(self):
        client = _FakeAnthropicClient()
        provider = AnthropicNativeProvider(
            model="claude-sonnet-4-5",
            api_key="k",
            base_url="",
            client=client,
        )
        codec = create_codec("anthropic_native")
        assistant_message = codec.build_assistant_message(
            content="checking",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "weather",
                    "arguments": "{\"city\": \"Beijing\"}",
                }
            ],
        )
        tool_message = codec.build_tool_result("sunny", "call_1", "weather")
        replay_history = codec.prepare_messages(
            [
                {"role": "user", "content": "what is the weather"},
                assistant_message,
                tool_message,
            ]
        )

        request = provider.build_request(
            replay_history,
            system_prompt="system prompt",
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
        provider.invoke_raw(request)

        call = client.messages.calls[0]
        self.assertEqual(call["system"], "system prompt")
        self.assertEqual(call["tools"][0]["name"], "weather")
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertEqual(call["messages"][1]["role"], "assistant")
        self.assertEqual(call["messages"][1]["content"][1]["type"], "tool_use")
        self.assertEqual(call["messages"][2]["content"][0]["type"], "tool_result")

    def test_anthropic_provider_format_assistant_response_preserves_thinking_blocks(self):
        codec = create_codec("anthropic_native")
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="plan", signature="sig-1"),
                SimpleNamespace(type="text", text="checking"),
                SimpleNamespace(type="tool_use", id="call_1", name="weather", input={"city": "Beijing"}),
            ]
        )

        message = codec.build_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["content"][0], {"type": "thinking", "thinking": "plan", "signature": "sig-1"})
        self.assertEqual(message["content"][1], {"type": "text", "text": "checking"})
        self.assertEqual(message["content"][2]["type"], "tool_use")
        self.assertEqual(codec.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_anthropic_provider_handles_real_sdk_block_objects(self):
        try:
            from anthropic import types as at
        except Exception as exc:
            self.skipTest(f"anthropic unavailable: {exc}")
        codec = create_codec("anthropic_native")
        response = SimpleNamespace(
            content=[
                at.ThinkingBlock(type="thinking", thinking="plan", signature="sig-1"),
                at.TextBlock(type="text", text="checking"),
                at.ToolUseBlock(type="tool_use", id="call_1", name="weather", input={"city": "Beijing"}),
            ]
        )

        message = codec.build_assistant_response(response, include_reasoning=True)

        self.assertEqual(message["content"][0], {"type": "thinking", "thinking": "plan", "signature": "sig-1"})
        self.assertEqual(message["content"][1], {"type": "text", "text": "checking"})
        self.assertEqual(message["content"][2]["type"], "tool_use")
        self.assertEqual(message["content"][2]["name"], "weather")
        self.assertEqual(codec.get_tool_calls(response)[0]["arguments"], {"city": "Beijing"})

    def test_anthropic_provider_stream_tool_calls_preserve_input_json_delta_arguments(self):
        codec = create_codec("anthropic_native")
        state = codec._init_anthropic_stream_state()
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Beijing\"}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]

        emitted = []
        for event in events:
            emitted.extend(codec._extract_anthropic_stream_events(event, state))

        self.assertEqual(emitted[-1]["type"], "tool_calls")
        self.assertEqual(emitted[-1]["tool_calls"][0]["arguments"], {"city": "Beijing"})
        assistant_message = codec._build_stream_assistant_message(state)
        self.assertEqual(assistant_message["content"][0]["type"], "tool_use")
        self.assertEqual(assistant_message["content"][0]["input"], {"city": "Beijing"})

    def test_anthropic_provider_stream_tool_calls_fallback_to_start_block_input(self):
        codec = create_codec("anthropic_native")
        state = codec._init_anthropic_stream_state()
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Beijing"}}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]

        emitted = []
        for event in events:
            emitted.extend(codec._extract_anthropic_stream_events(event, state))

        self.assertEqual(emitted[-1]["type"], "tool_calls")
        self.assertEqual(emitted[-1]["tool_calls"][0]["arguments"], {"city": "Beijing"})

    def test_anthropic_provider_stream_thinking_signature_is_not_emitted_as_text(self):
        codec = create_codec("anthropic_native")
        state = codec._init_anthropic_stream_state()

        thinking_events = codec._extract_anthropic_stream_events(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "plan"}},
            state,
        )
        signature_events = codec._extract_anthropic_stream_events(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
            state,
        )

        self.assertEqual(thinking_events, [{"type": "thinking_delta", "delta": "plan"}])
        self.assertEqual(signature_events, [])
        self.assertEqual(state["thinking_parts"], ["plan"])
        assistant_message = codec._build_stream_assistant_message(state)
        self.assertEqual(
            assistant_message["content"][0],
            {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
        )
        self.assertEqual(assistant_message["reasoning_content"], "plan")

    def test_anthropic_provider_stream_builds_thinking_block_from_accumulated_text(self):
        codec = create_codec("anthropic_native")
        state = codec._init_anthropic_stream_state()
        state["thinking_parts"] = ["plan"]
        state["assistant_blocks"] = {
            1: {"type": "text", "text": "先处理工具"},
            2: {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}},
        }
        state["tool_calls"] = {
            2: {"id": "call_1", "name": "weather", "input_json": "", "input": {"city": "Beijing"}}
        }

        assistant_message = codec._build_stream_assistant_message(state)

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

        from core.providers import AnthropicProvider, GoogleProvider

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
