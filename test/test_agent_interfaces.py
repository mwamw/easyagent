import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from agent.stream_renderer import BaseStreamDisplayRenderer
from agent.trace_recorder import InMemoryTraceRecorder
from core.llm import EasyLLM


class PlainProvider:
    def invoke(self, messages, temperature=None, **kwargs):
        return "hello world"

    def stream(self, messages, temperature=None, **kwargs):
        yield "hello "
        yield "world"


class DummyLLM(EasyLLM):
    def __init__(self, provider):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self._provider = provider
        self.client = None


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


class TestAgentInterfaces(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
