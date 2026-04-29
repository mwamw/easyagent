import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from agent import DefaultHistoryMessageAssembler
from agent import BaseInvocationRunner
from agent import DefaultPromptComposer
from agent import DefaultRuntimeSkillContextBridge
from agent import BaseStreamDisplayRenderer
from agent import BaseToolLoopEngine
from agent import InMemoryTraceRecorder
from core.Message import MetaUserMessage, SystemMessage, UserMessage
from core.llm import EasyLLM
from core.providers import OpenAIChatCodec
from core.request_input import ReplayRequestInput
from prompt import PromptBlock


class PlainProvider:
    def __init__(self):
        self.last_messages = []

    def build_request(self, messages, *, system_prompt=None, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(list(messages))
        return {"messages": request_messages, "stream": stream}

    def invoke_raw(self, request):
        self.last_messages = list(request["messages"])
        return SimpleNamespace(content="hello world")

    def stream_raw(self, request):
        self.last_messages = list(request["messages"])
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hello ", reasoning_content=None, reasoning=None, tool_calls=None), finish_reason=None)]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="world", reasoning_content=None, reasoning=None, tool_calls=None), finish_reason="stop")]
            ),
        ]

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self.stream_raw(request):
                yield item
        return _stream()


class DummyLLM(EasyLLM):
    def __init__(self, provider):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self._provider = provider
        self.client = None


class RecordingProvider:
    def __init__(self):
        self.last_build = None

    def build_request(self, replay_history, *, system_prompt=None, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        self.last_build = {
            "replay_history": list(replay_history),
            "system_prompt": system_prompt,
            "stream": stream,
        }
        return dict(self.last_build)

    def invoke_raw(self, request):
        return SimpleNamespace(content="ok")

    def stream_raw(self, request):
        return []

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    async def async_stream_raw(self, request):
        async def _stream():
            if False:
                yield None
        return _stream()


class FailingPrepareCodec(OpenAIChatCodec):
    def prepare_messages(self, messages):
        raise AssertionError("prepare_messages should not be called for ReplayRequestInput")


class RecordingTraceRecorder(InMemoryTraceRecorder):
    def __init__(self):
        super().__init__()
        self.event_types: list[str] = []

    def record_event(self, event_type, **kwargs):
        self.event_types.append(event_type)
        return super().record_event(event_type, **kwargs)


class RecordingRenderer(BaseStreamDisplayRenderer):
    def __init__(self):
        self.events: list[dict] = []
        self.final_text: str | None = None

    def create_state(self):
        return {"seen": 0}

    def render_event(self, state, event: dict):
        state["seen"] += 1
        self.events.append(dict(event))

    def render_final(self, state, final_text: str):
        self.final_text = final_text


class RecordingPromptComposer(DefaultPromptComposer):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def get_system_prompt_blocks(self, agent):
        self.calls.append("get_system_prompt_blocks")
        return [
            PromptBlock(
                name="custom",
                content="## Custom Prompt\nUse the custom composer.",
                order=0,
            )
        ]


class RecordingRuntimeSkillContextBridge(DefaultRuntimeSkillContextBridge):
    def __init__(self):
        self.calls: list[str] = []

    def append_runtime_skill_context_message(self, agent, messages):
        self.calls.append("append_runtime_skill_context_message")
        messages.append(
            MetaUserMessage(
                "## Bridge Runtime Context\nInjected by custom bridge.",
                metadata={"source": "bridge"},
            )
        )

    def clear_ephemeral_skill_state(self, agent):
        self.calls.append("clear_ephemeral_skill_state")


class RecordingHistoryMessageAssembler(DefaultHistoryMessageAssembler):
    def __init__(self):
        self.calls: list[str] = []

    def build_start_messages(self, agent, query: str):
        self.calls.append(f"build_start_messages:{query}")
        return [
            SystemMessage("assembler system prompt"),
            UserMessage(f"assembled:{query}"),
        ]


class RecordingInvocationRunner(BaseInvocationRunner):
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def invoke(self, agent, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs):
        self.calls.append(("invoke", query))
        return "runner-result"

    def stream_invoke(self, agent, query: str, temperature: float = 0.7, **kwargs):
        self.calls.append(("stream_invoke", query))
        return "runner-stream-result"

    def stream_invoke_with_tool(self, agent, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs):
        self.calls.append(("stream_invoke_with_tool", query))
        yield {"type": "final", "content": "runner-tool-stream"}

    async def ainvoke(self, agent, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs):
        self.calls.append(("ainvoke", query))
        return "runner-async-result"

    async def astream_invoke(self, agent, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs):
        self.calls.append(("astream_invoke", query))
        return "runner-async-stream-result"


class RecordingToolLoopEngine(BaseToolLoopEngine):
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def invoke(self, agent, query: str, messages, max_iter: int = 10, temperature: float = 0.7, trace_query=None, **kwargs):
        self.calls.append(("invoke", query))
        return "engine-result"

    async def ainvoke(self, agent, query: str, messages, max_iter: int = 10, temperature: float = 0.7, trace_query=None, **kwargs):
        self.calls.append(("ainvoke", query))
        return "engine-async-result"

    async def astream_invoke(self, agent, query: str, max_iter: int = 10, temperature: float = 0.7, trace_query=None, **kwargs):
        self.calls.append(("astream_invoke", query))
        yield {"type": "final", "content": "engine-stream-result"}


class TestAgentInterfaces(unittest.TestCase):
    def test_basic_agent_normalizes_string_and_xhigh_reasoning(self):
        agent = BasicAgent(
            name="reasoning-agent",
            llm=DummyLLM(PlainProvider()),
            reasoning="max",
        )

        self.assertEqual(agent.reasoning, {"effort": "xhigh"})

        detailed = BasicAgent(
            name="reasoning-agent-detail",
            llm=DummyLLM(PlainProvider()),
            reasoning={"effort": "extra-high", "summary": "detailed"},
        )

        self.assertEqual(detailed.reasoning["effort"], "xhigh")
        self.assertEqual(detailed.reasoning["summary"], "detailed")

    def test_llm_replay_request_input_skips_full_message_preparation(self):
        provider = RecordingProvider()
        llm = DummyLLM(provider)
        llm.codec = FailingPrepareCodec("mock")
        request_input = ReplayRequestInput(
            provider_name="mock",
            replay_history=[{"role": "user", "content": "hello"}],
            system_prompt="system prompt",
        )

        response = llm.invoke_raw(request_input)

        self.assertEqual(response.content, "ok")
        self.assertEqual(
            provider.last_build,
            {
                "replay_history": [{"role": "user", "content": "hello"}],
                "system_prompt": "system prompt",
                "stream": False,
            },
        )

    def test_basic_agent_accepts_custom_trace_recorder_and_stream_renderer(self):
        recorder = RecordingTraceRecorder()
        renderer = RecordingRenderer()
        agent = BasicAgent(
            name="interface-agent",
            llm=DummyLLM(PlainProvider()),
            trace_recorder=recorder,
            stream_renderer=renderer,
        )

        result = agent.stream_invoke("say hi")

        self.assertEqual(result, "hello world")
        self.assertEqual(
            [event["type"] for event in renderer.events],
            ["text_delta", "text_delta"],
        )
        self.assertEqual(renderer.final_text, "hello world")
        self.assertEqual(
            [event["type"] for event in agent.get_trace_history()],
            ["user_message", "assistant_message", "turn_end"],
        )
        self.assertEqual(
            recorder.event_types,
            ["user_message", "assistant_message", "turn_end"],
        )

    def test_basic_agent_accepts_custom_prompt_composer(self):
        provider = PlainProvider()
        composer = RecordingPromptComposer()
        agent = BasicAgent(
            name="prompt-agent",
            llm=DummyLLM(provider),
            prompt_composer=composer,
        )

        prompt = agent.get_enhanced_prompt()
        result = agent.invoke("say hi")

        self.assertEqual(prompt, "## Custom Prompt\nUse the custom composer.")
        self.assertEqual(result, "hello world")
        self.assertEqual(composer.calls, ["get_system_prompt_blocks", "get_system_prompt_blocks"])
        first_message = provider.last_messages[0]
        first_content = first_message["content"] if isinstance(first_message, dict) else first_message.content
        self.assertEqual(first_content, "## Custom Prompt\nUse the custom composer.")

    def test_basic_agent_accepts_custom_runtime_skill_context_bridge(self):
        bridge = RecordingRuntimeSkillContextBridge()
        agent = BasicAgent(
            name="bridge-agent",
            llm=DummyLLM(PlainProvider()),
            runtime_skill_context_bridge=bridge,
        )

        messages = []
        agent._append_runtime_skill_context_message(messages)
        agent._clear_ephemeral_skill_state()

        self.assertEqual(
            bridge.calls,
            ["append_runtime_skill_context_message", "clear_ephemeral_skill_state"],
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "## Bridge Runtime Context\nInjected by custom bridge.")
        self.assertEqual(messages[0].metadata["source"], "bridge")

    def test_basic_agent_accepts_custom_history_message_assembler(self):
        provider = PlainProvider()
        assembler = RecordingHistoryMessageAssembler()
        agent = BasicAgent(
            name="assembler-agent",
            llm=DummyLLM(provider),
            history_message_assembler=assembler,
        )

        result = agent.invoke("say hi")

        self.assertEqual(result, "hello world")
        self.assertEqual(assembler.calls, ["build_start_messages:say hi"])
        self.assertEqual(provider.last_messages[0]["content"], "assembler system prompt")
        self.assertEqual(provider.last_messages[1]["content"], "assembled:say hi")

    def test_basic_agent_accepts_custom_invocation_runner(self):
        runner = RecordingInvocationRunner()
        agent = BasicAgent(
            name="runner-agent",
            llm=DummyLLM(PlainProvider()),
            invocation_runner=runner,
        )

        result = agent.invoke("say hi")

        self.assertEqual(result, "runner-result")
        self.assertEqual(runner.calls, [("invoke", "say hi")])

    def test_basic_agent_accepts_custom_tool_loop_engine(self):
        engine = RecordingToolLoopEngine()
        agent = BasicAgent(
            name="engine-agent",
            llm=DummyLLM(PlainProvider()),
            tool_loop_engine=engine,
        )

        result = agent.invoke_with_tool("say hi", [])

        self.assertEqual(result, "engine-result")
        self.assertEqual(engine.calls, [("invoke", "say hi")])


if __name__ == "__main__":
    unittest.main()
