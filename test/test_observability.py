from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

from pydantic import BaseModel


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import BasicAgent
from core.llm import EasyLLM
from db import SessionStore
from Tool import Tool, ToolRegistry


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="EchoTool",
            description="Echo input text for observability tests.",
            parameters=EchoParams,
            read_only=True,
            side_effect_level="none",
            resource_scope=["runtime"],
        )

    def run(self, parameters: dict):
        return f"echo:{parameters['text']}"


def _chat_response(
    *,
    content: str | None = None,
    thinking: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    return SimpleNamespace(
        content=content,
        reasoning_content=thinking,
        tool_calls=tool_calls or [],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _stream_chunk(*, content: str | None = None, thinking: str | None = None, finish_reason: str | None = None):
    delta = SimpleNamespace(content=content, reasoning_content=thinking, reasoning=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class ObservabilityProvider:
    def __init__(self):
        self.requests: list[dict] = []

    def build_tool_payload(self, tools):
        return list(tools)

    def build_request(
        self,
        messages,
        *,
        system_prompt=None,
        tools=None,
        temperature=None,
        reasoning=None,
        stream=False,
        **kwargs,
    ):
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(list(messages))
        request = {
            "messages": request_messages,
            "tools": tools,
            "stream": stream,
            "temperature": temperature,
            "reasoning": reasoning,
        }
        self.requests.append(request)
        return request

    def invoke_raw(self, request):
        has_tools = bool(request.get("tools"))
        if has_tools:
            if any(isinstance(item, dict) and item.get("role") == "tool" for item in request["messages"]):
                return _chat_response(
                    content="tool flow complete",
                    thinking="final answer",
                    prompt_tokens=14,
                    completion_tokens=6,
                )
            return _chat_response(
                content=None,
                thinking="need to call echo tool",
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(name="EchoTool", arguments='{"text":"observability"}'),
                    )
                ],
                prompt_tokens=12,
                completion_tokens=7,
            )
        return _chat_response(
            content="plain response",
            thinking="plain reasoning",
            prompt_tokens=11,
            completion_tokens=4,
        )

    def stream_raw(self, request):
        yield _stream_chunk(thinking="stream reasoning ")
        yield _stream_chunk(content="stream ", finish_reason=None)
        yield _stream_chunk(content="response", finish_reason="stop")

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    async def async_stream_raw(self, request):
        async def _stream():
            for chunk in self.stream_raw(request):
                yield chunk
        return _stream()


class DummyLLM(EasyLLM):
    def __init__(self, provider=None):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.temperature = 0.7
        self.timeout = 60
        self.kwargs = {}
        self._provider = provider or ObservabilityProvider()
        self.client = None


class TestObservability(unittest.TestCase):
    def test_plain_and_stream_invocations_record_summary_and_recent_events(self):
        provider = ObservabilityProvider()
        agent = BasicAgent(
            name="observability-plain",
            llm=DummyLLM(provider),
        )

        plain_result = agent.invoke("summarize current runtime state")
        stream_result = agent.stream_invoke("stream the final answer")

        self.assertEqual(plain_result, "plain response")
        self.assertEqual(stream_result, "stream response")

        summary = agent.get_observability_summary()
        self.assertEqual(summary["agentRuns"], 2)
        self.assertEqual(summary["successfulAgentRuns"], 2)
        self.assertEqual(summary["llmRequests"], 2)
        self.assertEqual(summary["toolCalls"], 0)
        self.assertGreater(summary["inputTokens"], 0)
        self.assertGreater(summary["outputTokens"], 0)
        self.assertIn("plain_invoke", summary["requestKinds"])
        self.assertIn("plain_stream_invoke", summary["requestKinds"])

        recent = agent.get_recent_observability_events(limit=5)
        self.assertTrue(any(item["eventType"] == "agent" for item in recent))
        self.assertTrue(any(item["eventType"] == "llm" for item in recent))

        trace_summary = agent.get_trace_summary(limit_turns=2)
        self.assertEqual(len(trace_summary), 2)
        self.assertEqual(trace_summary[0]["llmRequests"], 1)

    def test_async_plain_invoke_records_llm_request(self):
        agent = BasicAgent(
            name="observability-async",
            llm=DummyLLM(ObservabilityProvider()),
        )

        result = asyncio.run(agent.ainvoke("run asynchronously"))

        self.assertEqual(result, "plain response")
        summary = agent.get_observability_summary()
        self.assertEqual(summary["agentRuns"], 1)
        self.assertEqual(summary["llmRequests"], 1)
        self.assertEqual(summary["requestKinds"]["plain_ainvoke"], 1)

    def test_tool_invoke_records_tool_metrics_and_trace_summary(self):
        registry = ToolRegistry()
        registry.register_tool(EchoTool())
        agent = BasicAgent(
            name="observability-tool",
            llm=DummyLLM(ObservabilityProvider()),
            enable_tool=True,
            tool_registry=registry,
        )

        result = agent.invoke("use the echo tool and finish the answer")

        self.assertEqual(result, "tool flow complete")
        summary = agent.get_observability_summary()
        self.assertEqual(summary["agentRuns"], 1)
        self.assertEqual(summary["llmRequests"], 2)
        self.assertEqual(summary["toolCalls"], 1)
        self.assertEqual(summary["toolsUsed"]["EchoTool"], 1)
        self.assertEqual(summary["requestKinds"]["tool_invoke"], 2)

        tool_events = agent.get_recent_observability_events(limit=5, event_type="tool")
        self.assertEqual(tool_events[0]["toolName"], "EchoTool")
        self.assertEqual(tool_events[0]["status"], "success")

        trace_summary = agent.get_trace_summary(limit_turns=1)
        self.assertEqual(trace_summary[0]["toolCalls"], 1)
        self.assertEqual(trace_summary[0]["llmRequests"], 2)
        self.assertEqual(trace_summary[0]["toolsUsed"], ["EchoTool"])

    def test_session_restore_preserves_observability_state_and_clear_resets_it(self):
        with tempfile.TemporaryDirectory() as tempdir:
            store = SessionStore(os.path.join(tempdir, "observability.db"))
            agent = BasicAgent(
                name="observability-session",
                llm=DummyLLM(ObservabilityProvider()),
            )
            agent.invoke("persist observability")
            agent.save_session("obs-session", store=store)

            restored = BasicAgent.load_session(
                "obs-session",
                llm=DummyLLM(ObservabilityProvider()),
                store=store,
            )
            try:
                summary = restored.get_observability_summary()
                self.assertEqual(summary["agentRuns"], 1)
                self.assertEqual(summary["llmRequests"], 1)
                self.assertEqual(summary["requestKinds"]["plain_invoke"], 1)

                restored.clear_observability()
                cleared = restored.get_observability_summary()
                self.assertEqual(cleared["agentRuns"], 0)
                self.assertEqual(cleared["llmRequests"], 0)
                self.assertEqual(cleared["toolCalls"], 0)
            finally:
                restored.close(close_worktree=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
