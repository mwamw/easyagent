import os
import sys
import unittest
from types import SimpleNamespace

from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from agent.tool_interrupt_controller import InMemoryToolInterruptController
from core.Exception import ToolConfirmationRequired
from core.llm import EasyLLM
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry


class EmptyParams(BaseModel):
    pass


class DangerousTool(Tool):
    def __init__(self):
        super().__init__(
            name="dangerous",
            description="Dangerous operation",
            parameters=EmptyParams,
            requires_confirmation=True,
            destructive=True,
        )
        self.run_count = 0

    def run(self, parameters: dict):
        self.run_count += 1
        return "should not run"


class ConfirmationProvider:
    def invoke_with_tools(self, messages, tools, temperature=None, **kwargs):
        return SimpleNamespace(
            content="需要确认",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="dangerous", arguments="{}"),
                )
            ],
        )

    async def async_invoke_with_tools(self, messages, tools, temperature=None, **kwargs):
        return self.invoke_with_tools(messages, tools, temperature=temperature, **kwargs)

    async def async_stream_with_tools(self, messages, tools, temperature=None, **kwargs):
        yield {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "dangerous",
                    "arguments": "{}",
                }
            ],
            "content": "需要确认",
            "thinking": "",
        }

    def format_tool_result(self, content, tool_id, tool_name):
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_id,
        }

    def format_assistant_response(self, response):
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": item.id,
                    "type": "function",
                    "function": {
                        "name": item.function.name,
                        "arguments": item.function.arguments,
                    },
                }
                for item in response.tool_calls
            ],
        }

    def format_assistant_message(self, content=None, tool_calls=None):
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": item["id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                }
                for item in (tool_calls or [])
            ],
        }

    def get_thinking_content(self, response):
        return None

    def get_response_content(self, response):
        return getattr(response, "content", "")

    def has_tool_calls(self, response):
        return bool(getattr(response, "tool_calls", None))

    def get_tool_calls(self, response):
        return getattr(response, "tool_calls", [])


class DummyLLM(EasyLLM):
    def __init__(self, provider):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self._provider = provider
        self.client = None


class RecordingInterruptController(InMemoryToolInterruptController):
    def __init__(self):
        super().__init__()
        self.payloads: list[dict] = []
        self.created_statuses: list[str] = []

    def build_payload(self, **kwargs):
        payload = super().build_payload(**kwargs)
        self.payloads.append(dict(payload))
        return payload

    def create_interruption(self, **kwargs):
        interruption = super().create_interruption(**kwargs)
        self.created_statuses.append(interruption.status)
        return interruption


class ToolInterruptionTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.tool = DangerousTool()
        self.registry.register_tool(self.tool)
        self.agent = BasicAgent(
            name="interruptible",
            llm=DummyLLM(ConfirmationProvider()),
            enable_tool=True,
            tool_registry=self.registry,
        )

    def test_invoke_raises_confirmation_required(self):
        with self.assertRaises(ToolConfirmationRequired) as ctx:
            self.agent.invoke("执行危险操作")

        exc = ctx.exception
        interrupt = self.agent.get_last_tool_interrupt()

        self.assertEqual(exc.tool_name, "dangerous")
        self.assertEqual(exc.status, "needs_confirmation")
        self.assertEqual(self.tool.run_count, 0)
        self.assertIsNotNone(interrupt)
        assert interrupt is not None
        self.assertEqual(interrupt["status"], "needs_confirmation")
        self.assertEqual(self.agent.get_history_length(), 3)

    async def test_astream_invoke_with_tool_emits_interruption_event(self):
        events = []
        async for event in self.agent.astream_invoke_with_tool("执行危险操作"):
            events.append(event)

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types, ["round_start", "text_delta", "tool_call", "tool_result", "interruption"])
        interruption = events[-1]
        self.assertEqual(interruption["reason"], "needs_confirmation")
        self.assertEqual(interruption["tool_name"], "dangerous")
        self.assertEqual(self.agent.get_last_tool_interrupt()["tool_name"], "dangerous")

    def test_basic_agent_uses_custom_interrupt_controller(self):
        controller = RecordingInterruptController()
        agent = BasicAgent(
            name="interruptible-custom",
            llm=DummyLLM(ConfirmationProvider()),
            enable_tool=True,
            tool_registry=self.registry,
            tool_interrupt_controller=controller,
        )

        with self.assertRaises(ToolConfirmationRequired):
            agent.invoke("执行危险操作")

        self.assertEqual(controller.created_statuses, ["needs_confirmation"])
        self.assertEqual(controller.payloads[0]["tool_name"], "dangerous")
        self.assertEqual(agent.get_last_tool_interrupt()["tool_name"], "dangerous")
