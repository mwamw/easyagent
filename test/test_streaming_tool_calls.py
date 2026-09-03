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
from core.providers import AnthropicProvider, GoogleProvider, OpenAIProvider, OpenAIResponsesProvider, create_codec
from runtime import AgentStreamEvent, AgentStreamEventType
from Tool.ToolRegistry import ToolRegistry


class EchoParams(BaseModel):
    text: str


def _openai_text_chunk(text: str = "", *, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, reasoning_content=None, reasoning=None, tool_calls=None),
                finish_reason=finish_reason,
            )
        ]
    )


def _openai_reasoning_chunk(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="", reasoning_content=text, reasoning=None, tool_calls=None),
                finish_reason=None,
            )
        ]
    )


def _openai_tool_call_chunk(tool_id: str, name: str, arguments: str, *, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="",
                    reasoning_content=None,
                    reasoning=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=tool_id,
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason=finish_reason,
            )
        ]
    )
 

class ScriptedStreamingProvider:
    def __init__(self):
        self.round = 0

    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    def _round_chunks(self):
        if self.round == 0:
            self.round += 1
            return [
                _openai_reasoning_chunk("need tool"),
                _openai_tool_call_chunk("call_1", "echo", "{\"text\": \"ping\"}"),
            ]
        return [
            _openai_text_chunk("pong", finish_reason="stop"),
        ]

    def stream_raw(self, request):
        return list(self._round_chunks())

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self._round_chunks():
                yield item
        return _stream()


class ScriptedStreamingProviderNoThinking:
    def __init__(self):
        self.round = 0

    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    def _round_chunks(self):
        if self.round == 0:
            self.round += 1
            return [
                _openai_text_chunk("准备调用工具"),
                _openai_tool_call_chunk("call_1", "echo", "{\"text\": \"ping\"}"),
            ]
        return [_openai_text_chunk("pong", finish_reason="stop")]

    def stream_raw(self, request):
        return list(self._round_chunks())

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self._round_chunks():
                yield item
        return _stream()


class ScriptedStreamingProviderWithAssistantItems:
    def __init__(self):
        self.round = 0

    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    def _round_events(self):
        if self.round == 0:
            self.round += 1
            return [
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="message",
                        id="msg_1",
                        role="assistant",
                        content=[SimpleNamespace(type="output_text", text="我先计算")],
                    ),
                ),
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="function_call",
                        id="fc_1",
                        call_id="call_1",
                        name="echo",
                        arguments="{\"text\": \"ping\"}",
                    ),
                ),
                SimpleNamespace(type="response.completed", response=None),
            ]
        return [
            SimpleNamespace(type="response.output_text.delta", delta="pong"),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    id="msg_2",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="pong")],
                    phase="final_answer",
                ),
            ),
            SimpleNamespace(type="response.completed", response=None),
        ]

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self._round_events():
                yield item
        return _stream()


class ScriptedStreamingProviderMultiToolRounds:
    def __init__(self):
        self.round = 0

    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    def _round_chunks(self):
        if self.round == 0:
            self.round += 1
            return [
                _openai_reasoning_chunk("先翻译"),
                _openai_tool_call_chunk(
                    "call_translate",
                    "translate_tool",
                    "{\"text\": \"你是谁，在哪里\", \"target_lang\": \"en\"}",
                ),
            ]
        if self.round == 1:
            self.round += 1
            return [
                _openai_tool_call_chunk(
                    "call_calculator",
                    "calculator",
                    "{\"expression\": \"3**22\"}",
                ),
            ]
        return [_openai_text_chunk("全部完成", finish_reason="stop")]

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self._round_chunks():
                yield item
        return _stream()


class ScriptedStreamingProviderEmptyFinal:
    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    async def async_stream_raw(self, request):
        async def _stream():
            yield _openai_reasoning_chunk("only thinking")
            yield _openai_text_chunk("", finish_reason="stop")
        return _stream()


class DummyLLM(EasyLLM):
    def __init__(self, provider):
        self.provider_name = "openai"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        if not hasattr(provider, "apply_cache_policy"):
            provider.apply_cache_policy = lambda request, request_input: request
        if not hasattr(provider, "build_tool_payload"):
            provider.build_tool_payload = lambda tools: list(tools or [])
        self._provider = provider
        self.client = None


class PlainStreamingProvider:
    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    def stream_raw(self, request):
        return [
            _openai_text_chunk("hello "),
            _openai_text_chunk("world", finish_reason="stop"),
        ]

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self.stream_raw(request):
                yield item
        return _stream()


class PlainThinkingProvider:
    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "stream": stream}

    def invoke_raw(self, request):
        return SimpleNamespace(content="hello", reasoning_content="先想一想")


class TestAgentStreamingContract(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _tool_agent(provider):
        from agent import BasicAgent

        registry = ToolRegistry()

        @registry.tool("echo", "Echo tool", EchoParams)
        def echo(text: str) -> str:
            return f"tool:{text}"

        return BasicAgent(name="streamer", llm=DummyLLM(provider)).with_tool(registry)

    async def test_astream_emits_structured_tool_loop_events(self):
        agent = self._tool_agent(ScriptedStreamingProvider())

        events = [event async for event in agent.astream("say hi")]

        self.assertTrue(all(isinstance(event, AgentStreamEvent) for event in events))
        self.assertEqual(
            [event.type for event in events],
            [
                AgentStreamEventType.REASONING_DELTA,
                AgentStreamEventType.TOOL_CALL,
                AgentStreamEventType.TOOL_RESULT,
                AgentStreamEventType.TEXT_DELTA,
                AgentStreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[1].data["tool_name"], "echo")
        self.assertEqual(events[2].data["status"], "success")
        self.assertIn("tool:ping", events[2].content)
        self.assertEqual(events[-1].content, "pong")
        self.assertEqual([message.role for message in agent.history], ["user", "assistant", "tool", "assistant"])

    def test_stream_uses_the_same_structured_contract(self):
        agent = self._tool_agent(ScriptedStreamingProvider())

        events = list(agent.stream("say hi"))

        self.assertEqual(events[-1].type, AgentStreamEventType.FINAL)
        self.assertEqual(events[-1].content, "pong")
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        serialized = events[0].to_dict()
        self.assertEqual(AgentStreamEvent.from_dict(serialized), events[0])

    async def test_empty_final_response_keeps_only_the_user_turn(self):
        from agent import BasicAgent

        agent = BasicAgent(
            name="empty-final",
            llm=DummyLLM(ScriptedStreamingProviderEmptyFinal()),
        ).with_tool(ToolRegistry())

        events = [event async for event in agent.astream("think only")]

        self.assertEqual(events[-1].type, AgentStreamEventType.FINAL)
        self.assertEqual(events[-1].content, "")
        self.assertEqual([message.role for message in agent.history], ["user"])

    def test_plain_stream_emits_text_and_final_without_printing(self):
        from agent import BasicAgent

        agent = BasicAgent(name="plain-streamer", llm=DummyLLM(PlainStreamingProvider()))

        events = list(agent.stream("say hi"))

        self.assertEqual(
            [event.type for event in events],
            [AgentStreamEventType.TEXT_DELTA, AgentStreamEventType.TEXT_DELTA, AgentStreamEventType.FINAL],
        )
        self.assertEqual("".join(event.content or "" for event in events[:-1]), "hello world")
        self.assertEqual(events[-1].content, "hello world")

    def test_invoke_preserves_reasoning_in_canonical_history(self):
        from agent import BasicAgent

        agent = BasicAgent(name="plain-thinking", llm=DummyLLM(PlainThinkingProvider()))

        self.assertEqual(agent.invoke("say hi"), "hello")
        self.assertEqual([message.role for message in agent.history], ["user", "assistant"])
        reasoning_blocks = [
            block
            for block in agent.history[-1].content
            if block.type == "reasoning"
        ]
        self.assertEqual([block.text for block in reasoning_blocks], ["先想一想"])


class TestProviderStreamingHelpers(unittest.TestCase):
    def test_openai_compatible_tool_call_aggregation(self):
        codec = create_codec("openai")

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
                    finish_reason="stop",
                )
            ]
        )

        events = list(codec.stream_events([chunk1, chunk2], tools=True))

        self.assertEqual(events[0]["type"], "tool_calls")
        self.assertEqual(events[0]["tool_calls"][0]["name"], "search")
        self.assertEqual(events[0]["tool_calls"][0]["arguments"], "{\"q\": \"ai\"}")

    def test_base_provider_thinking_does_not_fallback_to_content(self):
        codec = create_codec("openai")
        response = SimpleNamespace(
            reasoning_content=None,
            content="normal content",
        )

        self.assertIsNone(codec.get_thinking_content(response))

    def test_base_provider_has_tool_calls_returns_bool(self):
        codec = create_codec("openai")
        response = SimpleNamespace(
            tool_calls=[SimpleNamespace(id="call_1")],
        )

        self.assertIs(codec.has_tool_calls(response), True)

    def test_responses_provider_event_aggregation(self):
        codec = create_codec("openai_responses")
        state = codec._init_responses_tool_stream_state()

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

        self.assertEqual(codec._extract_responses_stream_events(delta_event, state), [])
        self.assertEqual(codec._extract_responses_stream_events(done_event, state), [])
        events = codec._extract_responses_stream_events(complete_event, state)

        self.assertEqual(events[0]["type"], "tool_calls")
        self.assertEqual(events[0]["tool_calls"][0]["name"], "weather")
        self.assertEqual(events[0]["tool_calls"][0]["arguments"], "{\"city\": \"Beijing\"}")

    def test_responses_provider_event_aggregation_preserves_assistant_item_order(self):
        codec = create_codec("openai_responses")
        state = codec._init_responses_tool_stream_state()

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

        self.assertEqual(codec._extract_responses_stream_events(message_event, state), [])
        self.assertEqual(codec._extract_responses_stream_events(function_call_event, state), [])
        events = codec._extract_responses_stream_events(complete_event, state)

        self.assertEqual(events[0]["assistant_items"][0]["type"], "message")
        self.assertEqual(events[0]["assistant_items"][1]["type"], "function_call")

    def test_responses_provider_format_assistant_response_preserves_reasoning_payload(self):
        codec = create_codec("openai_responses")
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="reasoning",
                    id="rs_1",
                    summary=[SimpleNamespace(type="summary_text", text="先分析一下")],
                    content=[SimpleNamespace(type="reasoning_text", text="完整推理")],
                    encrypted_content="secret",
                    status="completed",
                ),
                SimpleNamespace(
                    type="message",
                    id="msg_1",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="我先计算")],
                ),
            ]
        )

        items = codec.build_assistant_response(response)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["content"][0]["text"], "我先计算")

        items_with_reasoning = codec.build_assistant_response(response, include_reasoning=True)
        self.assertEqual(len(items_with_reasoning), 2)
        self.assertEqual(items_with_reasoning[0]["type"], "reasoning")
        self.assertEqual(items_with_reasoning[0]["id"], "rs_1")
        self.assertEqual(items_with_reasoning[0]["summary"][0]["text"], "先分析一下")
        self.assertEqual(items_with_reasoning[0]["content"][0]["text"], "完整推理")
        self.assertEqual(items_with_reasoning[0]["encrypted_content"], "secret")
        self.assertEqual(items_with_reasoning[0]["status"], "completed")
        self.assertEqual(items_with_reasoning[1]["type"], "message")
        self.assertEqual(items_with_reasoning[1]["content"][0]["text"], "我先计算")

    def test_responses_provider_get_response_content_prefers_final_answer(self):
        codec = create_codec("openai_responses")
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

        self.assertEqual(codec.get_response_content(response), "最终结果")

    def test_responses_codec_prepare_messages_keeps_request_ready_items(self):
        codec = create_codec("openai_responses")
        source = [
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

        prepared = codec.prepare_messages(source)
        self.assertEqual(prepared, source)

    def test_responses_provider_deduplicates_message_done_after_text_delta(self):
        codec = create_codec("openai_responses")
        state = codec._init_responses_tool_stream_state()

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
            codec._extract_responses_stream_events(delta_event, state),
            [{"type": "text_delta", "delta": "我先计算"}],
        )
        self.assertEqual(codec._extract_responses_stream_events(message_done_event, state), [])
        events = codec._extract_responses_stream_events(complete_event, state)
        self.assertEqual(events[0]["content"], "我先计算")

    def test_google_provider_format_tool_result(self):
        codec = create_codec("google")
        message = codec.build_tool_result("sunny", "call_1", "weather")

        self.assertEqual(
            message,
            {
                "role": "function",
                "content": "sunny",
                "tool_call_id": "call_1",
                "name": "weather",
            },
        )

    def test_openai_provider_prepare_message_for_request_keeps_reasoning_content(self):
        codec = create_codec("openai")
        payload = codec.prepare_messages(
            [
                {
                    "role": "assistant",
                    "content": "我先计算",
                    "reasoning_content": "先想一想",
                }
            ]
        )[0]

        self.assertEqual(
            payload,
            {
                "role": "assistant",
                "content": "我先计算",
                "reasoning_content": "先想一想",
            },
        )

    def test_anthropic_provider_format_assistant_message(self):
        codec = create_codec("anthropic")
        message = codec.build_assistant_message(
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
        codec = create_codec("anthropic")
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="checking"),
                SimpleNamespace(
                    type="tool_use",
                    id="call_1",
                    name="weather",
                    input={"city": "Beijing"},
                )
            ],
        )

        message = codec.build_assistant_response(response)

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
