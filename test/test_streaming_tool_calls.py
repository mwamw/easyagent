"""
流式工具调用测试
"""
import os
import sys
import unittest
import io
from contextlib import redirect_stdout
from types import SimpleNamespace

from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import EasyLLM
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.google_provider import GoogleProvider
from core.providers.openai_provider import OpenAIProvider
from core.providers.openai_responses_provider import OpenAIResponsesProvider
from Tool.ToolRegistry import ToolRegistry


class EchoParams(BaseModel):
    text: str
 

class ScriptedStreamingProvider:
    def __init__(self):
        self.round = 0

    async def async_stream_with_tools(self, messages, tools, temperature=None, **kwargs):
        if self.round == 0:
            self.round += 1
            yield {"type": "thinking_delta", "delta": "need tool"}
            yield {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "echo",
                        "arguments": "{\"text\": \"ping\"}",
                    }
                ],
                "content": "",
                "thinking": "need tool",
            }
            return

        yield {"type": "text_delta", "delta": "pong"}
        yield {"type": "final_response", "content": "pong", "thinking": ""}

    def format_tool_result(self, content, tool_id, tool_name):
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_id,
        }

    def format_assistant_message(self, content=None, tool_calls=None):
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
                for tool_call in (tool_calls or [])
            ],
        }


class ScriptedStreamingProviderNoThinking:
    def __init__(self):
        self.round = 0

    async def async_stream_with_tools(self, messages, tools, temperature=None, **kwargs):
        if self.round == 0:
            self.round += 1
            yield {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "echo",
                        "arguments": "{\"text\": \"ping\"}",
                    }
                ],
                "content": "准备调用工具",
                "thinking": "",
            }
            return

        yield {"type": "final_response", "content": "pong", "thinking": ""}

    def format_tool_result(self, content, tool_id, tool_name):
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_id,
        }

    def format_assistant_message(self, content=None, tool_calls=None):
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
                for tool_call in (tool_calls or [])
            ],
        }


class ScriptedStreamingProviderWithAssistantItems:
    async def async_stream_with_tools(self, messages, tools, temperature=None, **kwargs):
        yield {
            "type": "round_start",
            "round": 1,
        }
        yield {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "echo",
                    "arguments": "{\"text\": \"ping\"}",
                }
            ],
            "content": "我先计算",
            "thinking": "",
            "assistant_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "我先计算"},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": "{\"text\": \"ping\"}",
                },
            ],
        }
        yield {
            "type": "round_start",
            "round": 2,
        }
        yield {"type": "final_response", "content": "pong", "thinking": ""}

    def format_tool_result(self, content, tool_id, tool_name):
        return {
            "type": "function_call_output",
            "call_id": tool_id,
            "output": content,
        }

    def format_assistant_message(self, content=None, tool_calls=None):
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls or [],
        }


class DummyLLM(EasyLLM):
    def __init__(self, provider):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self._provider = provider
        self.client = None


class PlainStreamingProvider:
    def stream(self, messages, temperature=None, **kwargs):
        yield "hello "
        yield "world"

    async def async_stream(self, messages, temperature=None, **kwargs):
        yield "hello "
        yield "world"


class TestStreamingToolCalls(unittest.IsolatedAsyncioTestCase):
    async def test_basic_agent_streaming_tool_loop(self):
        from agent.BasicAgent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return f"tool:{text}"

        llm = DummyLLM(ScriptedStreamingProvider())
        agent = BasicAgent(
            name="streamer",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
            verbose_thinking=True,
        )

        events = []
        async for event in agent.astream_invoke_with_tool("say hi"):
            events.append(event)

        self.assertEqual(
            [event["type"] for event in events],
            ["round_start", "thinking_delta", "tool_call", "tool_result", "round_start", "text_delta", "final"],
        )
        self.assertEqual(events[0]["round"], 1)
        self.assertEqual(events[2]["tool_name"], "echo")
        self.assertEqual(events[3]["content"], "tool:ping")
        self.assertEqual(events[4]["round"], 2)
        self.assertEqual(events[-1]["content"], "pong")
        self.assertEqual(agent.get_history_length(), 4)
        self.assertEqual(agent.get_thinking_history(), ["need tool"])
        trace_types = [event["type"] for event in agent.get_trace_history()]
        self.assertIn("llm_input", trace_types)
        self.assertIn("tool_call", trace_types)
        self.assertIn("tool_result", trace_types)
        self.assertIn("final", trace_types)
        history = agent.get_history()
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1]["tool_calls"][0]["function"]["name"], "echo")
        self.assertEqual(history[2]["tool_call_id"], "call_1")
        self.assertEqual(history[3].role, "assistant")

    async def test_astream_invoke_tool_mode_returns_final_text(self):
        from agent.BasicAgent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return text

        llm = DummyLLM(ScriptedStreamingProvider())
        agent = BasicAgent(
            name="streamer",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
        )

        result = await agent.astream_invoke("say hi")
        self.assertEqual(result, "pong")

    async def test_astream_invoke_tool_mode_displays_thinking_and_final(self):
        from agent.BasicAgent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return text

        llm = DummyLLM(ScriptedStreamingProvider())
        agent = BasicAgent(
            name="streamer",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
            verbose_thinking=True,
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = await agent.astream_invoke("say hi")

        self.assertEqual(result, "pong")
        self.assertEqual(
            stdout.getvalue(),
            "round 1\n\nthinking content:\nneed tool\ntool_calls:\necho : {'text': 'ping'}\n\nround 2\n\ncontent:\npong\nfinal res:\npong\n",
        )
        self.assertEqual(agent.get_thinking_history(), ["need tool"])

    async def test_astream_invoke_with_tool_preserves_assistant_items_order(self):
        from agent.BasicAgent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return text

        llm = DummyLLM(ScriptedStreamingProviderWithAssistantItems())
        agent = BasicAgent(
            name="streamer",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
        )

        async for _ in agent.astream_invoke_with_tool("say hi"):
            pass

        history = agent.get_history()
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1]["type"], "message")
        self.assertEqual(history[2]["type"], "function_call")
        self.assertEqual(history[3]["type"], "function_call_output")
        self.assertEqual(history[-1].role, "assistant")

    def test_stream_invoke_tool_mode_returns_final_text(self):
        from agent.BasicAgent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return text

        llm = DummyLLM(ScriptedStreamingProvider())
        agent = BasicAgent(
            name="streamer",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
        )

        result = agent.stream_invoke("say hi")
        self.assertEqual(result, "pong")

    def test_stream_invoke_tool_mode_skips_empty_thinking_header(self):
        from agent.BasicAgent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return text

        llm = DummyLLM(ScriptedStreamingProviderNoThinking())
        agent = BasicAgent(
            name="streamer",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
            verbose_thinking=True,
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = agent.stream_invoke("say hi")

        self.assertEqual(result, "pong")
        self.assertEqual(
            stdout.getvalue(),
            "round 1\n\ncontent:\n准备调用工具\ntool_calls:\necho : {'text': 'ping'}\n\nround 2\n\ncontent:\npong\nfinal res:\npong\n",
        )
        self.assertEqual(agent.get_thinking_history(), [])

    def test_stream_invoke_plain_mode_displays_content_and_final(self):
        from agent.BasicAgent import BasicAgent

        llm = DummyLLM(PlainStreamingProvider())
        agent = BasicAgent(
            name="plain-streamer",
            llm=llm,
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = agent.stream_invoke("say hi")

        self.assertEqual(result, "hello world")
        self.assertEqual(
            stdout.getvalue(),
            "content:\nhello world\nfinal res:\nhello world\n",
        )
        trace = agent.get_trace_history()
        self.assertEqual(trace[0]["type"], "round_start")
        self.assertEqual(trace[1]["type"], "llm_input")
        self.assertEqual(trace[-1]["type"], "final")


class TestProviderStreamingHelpers(unittest.TestCase):
    def test_openai_compatible_tool_call_aggregation(self):
        provider = OpenAIProvider(model="mock", api_key="k", base_url="http://localhost")
        state = provider._init_chat_tool_stream_state()

        chunk1 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content=None,
                        reasoning=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(name="search", arguments="{\"q\":"),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        )
        chunk2 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        reasoning_content=None,
                        reasoning=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments=" \"ai\"}"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )

        events1 = provider._extract_chat_stream_events(chunk1, state)
        events2 = provider._extract_chat_stream_events(chunk2, state)

        self.assertEqual(events1, [])
        self.assertEqual(events2[0]["type"], "tool_calls")
        self.assertEqual(events2[0]["tool_calls"][0]["name"], "search")
        self.assertEqual(events2[0]["tool_calls"][0]["arguments"], "{\"q\": \"ai\"}")

    def test_base_provider_thinking_does_not_fallback_to_content(self):
        provider = OpenAIProvider(model="mock", api_key="k", base_url="http://localhost")
        response = SimpleNamespace(
            reasoning_content=None,
            content="normal content",
        )

        self.assertIsNone(provider.get_thinking_content(response))

    def test_base_provider_has_tool_calls_returns_bool(self):
        provider = OpenAIProvider(model="mock", api_key="k", base_url="http://localhost")
        response = SimpleNamespace(
            tool_calls=[SimpleNamespace(id="call_1")],
        )

        self.assertIs(provider.has_tool_calls(response), True)

    def test_responses_provider_event_aggregation(self):
        provider = OpenAIResponsesProvider(model="mock", api_key="k", base_url="http://localhost")
        state = provider._init_responses_tool_stream_state()

        delta_event = SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            call_id="call_1",
            output_index=0,
            delta="{\"city\": \"Bei",
        )
        done_event = SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="weather",
                arguments="{\"city\": \"Beijing\"}",
            ),
        )
        complete_event = SimpleNamespace(
            type="response.completed",
            response=None,
        )

        self.assertEqual(provider._extract_responses_stream_events(delta_event, state), [])
        self.assertEqual(provider._extract_responses_stream_events(done_event, state), [])
        events = provider._extract_responses_stream_events(complete_event, state)

        self.assertEqual(events[0]["type"], "tool_calls")
        self.assertEqual(events[0]["tool_calls"][0]["name"], "weather")
        self.assertEqual(events[0]["tool_calls"][0]["arguments"], "{\"city\": \"Beijing\"}")

    def test_responses_provider_event_aggregation_preserves_assistant_item_order(self):
        provider = OpenAIResponsesProvider(model="mock", api_key="k", base_url="http://localhost")
        state = provider._init_responses_tool_stream_state()

        message_event = SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                id="msg_1",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="我先计算")],
            ),
        )
        function_call_event = SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="calculator",
                arguments="{\"expression\":\"3**22\"}",
            ),
        )
        complete_event = SimpleNamespace(
            type="response.completed",
            response=None,
        )

        self.assertEqual(provider._extract_responses_stream_events(message_event, state), [])
        self.assertEqual(provider._extract_responses_stream_events(function_call_event, state), [])
        events = provider._extract_responses_stream_events(complete_event, state)

        self.assertEqual(events[0]["assistant_items"][0]["type"], "message")
        self.assertEqual(events[0]["assistant_items"][1]["type"], "function_call")

    def test_responses_provider_format_assistant_response_preserves_reasoning_summary(self):
        provider = OpenAIResponsesProvider(model="mock", api_key="k", base_url="http://localhost")
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    id="rs_1",
                    summary=[SimpleNamespace(type="summary_text", text="先分析一下")],
                ),
                SimpleNamespace(
                    type="message",
                    id="msg_1",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="我先计算")],
                ),
            ]
        )

        items = provider.format_assistant_response(response)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["content"][0]["text"], "我先计算")

    def test_responses_provider_get_response_content_prefers_final_answer(self):
        provider = OpenAIResponsesProvider(model="mock", api_key="k", base_url="http://localhost")
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    phase="commentary",
                    content=[SimpleNamespace(type="output_text", text="我先计算")],
                ),
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    phase="final_answer",
                    content=[SimpleNamespace(type="output_text", text="最终结果")],
                ),
            ],
            output_text="最终结果",
        )

        self.assertEqual(provider.get_response_content(response), "最终结果")

    def test_responses_provider_convert_input_strips_noise_fields(self):
        converted = OpenAIResponsesProvider._convert_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"text": "先分析一下"}],
                    "encrypted_content": "secret",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "phase": "commentary",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "我先计算"}],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "calculator",
                    "arguments": "{\"expression\":\"3**22\"}",
                    "status": "completed",
                },
            ]
        )

        self.assertEqual(
            converted,
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我先计算"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "calculator",
                    "arguments": "{\"expression\":\"3**22\"}",
                },
            ],
        )

    def test_responses_provider_deduplicates_message_done_after_text_delta(self):
        provider = OpenAIResponsesProvider(model="mock", api_key="k", base_url="http://localhost")
        state = provider._init_responses_tool_stream_state()

        delta_event = SimpleNamespace(
            type="response.output_text.delta",
            delta="我先计算",
        )
        message_done_event = SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="我先计算")],
            ),
        )
        complete_event = SimpleNamespace(
            type="response.completed",
            response=None,
        )

        self.assertEqual(
            provider._extract_responses_stream_events(delta_event, state),
            [{"type": "text_delta", "delta": "我先计算"}],
        )
        self.assertEqual(provider._extract_responses_stream_events(message_done_event, state), [])
        events = provider._extract_responses_stream_events(complete_event, state)
        self.assertEqual(events[0]["content"], "我先计算")

    def test_google_provider_format_tool_result(self):
        provider = GoogleProvider(model="mock", api_key="k", base_url="http://localhost")
        message = provider.format_tool_result("sunny", "call_1", "weather")

        self.assertEqual(
            message,
            {
                "role": "function",
                "content": "sunny",
                "tool_call_id": "call_1",
                "name": "weather",
            },
        )

    def test_anthropic_provider_format_assistant_message(self):
        provider = AnthropicProvider(model="mock", api_key="k", base_url="http://localhost")
        message = provider.format_assistant_message(
            content="checking",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "weather",
                    "arguments": "{\"city\": \"Beijing\"}",
                }
            ],
        )

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(
            message["content"],
            [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "weather",
                    "input": {"city": "Beijing"},
                },
            ],
        )

    def test_anthropic_provider_format_assistant_response_preserves_text_and_tool_use(self):
        provider = AnthropicProvider(model="mock", api_key="k", base_url="http://localhost")
        response = SimpleNamespace(
            content="checking",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="weather",
                        arguments="{\"city\": \"Beijing\"}",
                    ),
                )
            ],
        )

        message = provider.format_assistant_response(response)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(
            message["content"],
            [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "weather",
                    "input": {"city": "Beijing"},
                },
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
