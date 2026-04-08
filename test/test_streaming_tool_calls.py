"""
流式工具调用测试
"""
import os
import sys
import unittest
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


class DummyLLM(EasyLLM):
    def __init__(self, provider):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self._provider = provider
        self.client = None


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
            ["thinking_delta", "tool_call", "tool_result", "text_delta", "final"],
        )
        self.assertEqual(events[1]["tool_name"], "echo")
        self.assertEqual(events[2]["content"], "tool:ping")
        self.assertEqual(events[-1]["content"], "pong")
        self.assertEqual(agent.get_history_length(), 2)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
