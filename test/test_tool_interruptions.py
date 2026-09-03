import os
import sys
import unittest
from types import SimpleNamespace

from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from agent.components.tool_interrupt_controller import InMemoryToolInterruptController
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
    def build_request(self, messages, *, tools=None, temperature=None, reasoning=None, stream=False, **kwargs):
        return {"messages": list(messages), "tools": tools or [], "stream": stream}

    def invoke_raw(self, request):
        return SimpleNamespace(
            content="需要确认",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="dangerous", arguments="{}"),
                )
            ],
        )

    def stream_raw(self, request):
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="需要确认", reasoning_content=None, reasoning=None, tool_calls=None),
                        finish_reason=None,
                    )
                ]
            ),
            SimpleNamespace(
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
                                    function=SimpleNamespace(name="dangerous", arguments="{}"),
                                )
                            ],
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
        ]

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    async def async_stream_raw(self, request):
        async def _stream():
            for item in self.stream_raw(request):
                yield item
        return _stream()

    def apply_cache_policy(self, request, request_input):
        return request


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
        ).with_tool(self.registry)

    def test_invoke_raises_confirmation_required(self):
        with self.assertRaises(ToolConfirmationRequired) as ctx:
            self.agent.invoke("执行危险操作")

        exc = ctx.exception
        interrupt = self.agent.get_pending_interruption()

        self.assertEqual(exc.tool_name, "dangerous")
        self.assertEqual(exc.status, "needs_confirmation")
        self.assertEqual(self.tool.run_count, 0)
        self.assertIsNotNone(interrupt)
        assert interrupt is not None
        self.assertEqual(interrupt["status"], "needs_confirmation")
        self.assertEqual(self.agent.get_history_length(), 2)

    def test_execute_confirmed_tool_result_bypasses_confirmation_short_circuit(self):
        result = self.registry.execute_confirmed_tool_result("dangerous", {})
        self.assertEqual(result.status, "success")
        self.assertEqual(result.to_display_string(), "should not run")
        self.assertEqual(self.tool.run_count, 1)

    async def test_astream_emits_structured_error_before_raising_interruption(self):
        events = []
        with self.assertRaises(ToolConfirmationRequired):
            async for event in self.agent.astream("执行危险操作"):
                events.append(event)

        event_types = [event.type.value for event in events]
        self.assertEqual(event_types, ["text_delta", "tool_call", "error"])
        interruption = events[-1]
        self.assertTrue(interruption.data["interrupted"])
        self.assertEqual(interruption.data["error_type"], "ToolConfirmationRequired")
        self.assertEqual(self.agent.get_pending_interruption()["tool_name"], "dangerous")

    def test_basic_agent_uses_custom_interrupt_controller(self):
        controller = RecordingInterruptController()
        agent = BasicAgent(
            name="interruptible-custom",
            llm=DummyLLM(ConfirmationProvider()),
        ).with_tool(self.registry)
        agent.with_interruptions(controller)

        with self.assertRaises(ToolConfirmationRequired):
            agent.invoke("执行危险操作")

        self.assertEqual(controller.created_statuses, ["needs_confirmation"])
        self.assertEqual(controller.payloads[0]["tool_name"], "dangerous")
        self.assertEqual(agent.get_pending_interruption()["tool_name"], "dangerous")

    def test_resolve_pending_interruption_commits_external_tool_result(self):
        with self.assertRaises(ToolConfirmationRequired):
            self.agent.invoke("执行危险操作")

        resolved = self.agent.resolve_pending_interruption(
            content="用户确认后由宿主执行完成。",
            ephemeral_context={"confirmation": "approved"},
        )

        self.assertEqual(resolved["status"], "resolved")
        self.assertIsNone(self.agent.get_pending_interruption())
        self.assertEqual(self.agent.get_canonical_history()[-1].role, "tool")
        self.assertEqual(len(self.agent.metamessage_manager.list_pending()), 1)
