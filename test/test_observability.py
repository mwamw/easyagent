from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
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
from observability import (
    AgentInvoke,
    AgentInvokeStats,
    BaseObservabilityManager,
    InMemoryObservabilityStore,
    LLMInvoke,
    LLMInvokeStats,
    SQLiteObservabilityStore,
)
from core.history import CanonicalBlock, CanonicalMessage
from runtime.events import RuntimeEvent, RuntimeEventBus
from Tool import Tool, ToolRegistry
from Tool.builtin.agent_tool import AgentTool
from training import (
    TrainingDataFilter,
    TrainingDataFormat,
    TrainingExporter,
)


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


class ObservabilityProvider:
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
        return {
            "messages": request_messages,
            "tools": tools,
            "stream": stream,
            "temperature": temperature,
            "reasoning": reasoning,
        }

    def apply_cache_policy(self, request, request_input):
        return request

    def invoke_raw(self, request):
        if request.get("tools"):
            if any(
                isinstance(item, dict) and item.get("role") == "tool"
                for item in request["messages"]
            ):
                return _chat_response(
                    content="tool flow complete",
                    thinking="final answer",
                    prompt_tokens=14,
                    completion_tokens=6,
                )
            return _chat_response(
                thinking="need to call echo tool",
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(
                            name="EchoTool",
                            arguments='{"text":"observability"}',
                        ),
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
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content="stream reasoning",
                        reasoning=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="stream response",
                        reasoning_content=None,
                        reasoning=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ]
        )

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)


class FailingProvider(ObservabilityProvider):
    def invoke_raw(self, request):
        raise RuntimeError("provider failed")


class DummyLLM(EasyLLM):
    def __init__(self, provider=None):
        self.provider_name = "openai"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.temperature = 0.7
        self.timeout = 60
        self.kwargs = {}
        self._provider = provider or ObservabilityProvider()
        self.client = None


class QueryFilter(TrainingDataFilter):
    def accept(self, invoke):
        return invoke.query.startswith("export")

    def transform(self, invoke):
        transformed = invoke.model_copy(deep=True)
        transformed.metadata = {"cleaned": True}
        return transformed


class CustomObservabilityManager(BaseObservabilityManager):
    def __init__(self, records: list[AgentInvoke]):
        self.records = records

    def bind(
        self,
        *,
        agent_id: str,
        event_bus: RuntimeEventBus,
    ) -> "CustomObservabilityManager":
        return self

    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        return None

    def list(self) -> list[AgentInvoke]:
        return list(self.records)

    def close(self) -> None:
        return None


class TestObservability(unittest.TestCase):
    def test_observability_is_opt_in(self):
        agent = BasicAgent(name="disabled", llm=DummyLLM())

        self.assertEqual(agent.invoke("plain query"), "plain response")
        self.assertIsNone(agent.observability)

    def test_plain_invoke_records_complete_provider_neutral_call(self):
        store = InMemoryObservabilityStore()
        agent = BasicAgent(
            name="plain",
            llm=DummyLLM(),
            system_prompt="You are a test agent.",
        ).with_observability(store=store)

        self.assertEqual(agent.invoke("record this invocation"), "plain response")
        invoke = agent.observability.latest()

        self.assertIsNotNone(invoke)
        assert invoke is not None
        self.assertEqual(invoke.query, "record this invocation")
        self.assertEqual(invoke.schema_version, "easyagent.observability.v1")
        self.assertEqual(invoke.record_type, "agent_invoke")
        self.assertTrue(invoke.stats.success)
        self.assertEqual(invoke.stats.llm_calls, 1)
        self.assertEqual(invoke.stats.total_tokens, 15)
        self.assertEqual(len(invoke.llm_invokes), 1)
        llm_invoke = invoke.llm_invokes[0]
        self.assertEqual(llm_invoke.record_type, "llm_invoke")
        self.assertTrue(llm_invoke.stats.success)
        self.assertEqual(llm_invoke.options["provider"], "openai")
        self.assertEqual(llm_invoke.options["model"], "mock-model")
        self.assertEqual([item.role for item in llm_invoke.input], ["system", "user"])
        self.assertEqual(llm_invoke.output[-1].role, "assistant")
        self.assertIn("plain response", llm_invoke.output[-1].text_content())
        self.assertEqual(invoke.output[-1].text_content(), "plain response")
        self.assertEqual([message.role for message in invoke.trace], ["user", "assistant"])

    def test_tool_invoke_records_llm_steps_and_complete_trace(self):
        registry = ToolRegistry()
        registry.register_tool(EchoTool())
        agent = BasicAgent(
            name="tool-agent",
            llm=DummyLLM(),
        ).with_tool(registry).with_observability(store=InMemoryObservabilityStore())

        self.assertEqual(agent.invoke("use echo"), "tool flow complete")
        invoke = agent.observability.latest()

        assert invoke is not None
        self.assertEqual(invoke.stats.llm_calls, 2)
        self.assertEqual(invoke.stats.tool_calls, 1)
        self.assertEqual(len(invoke.llm_invokes), 2)
        self.assertTrue(all(item.tools for item in invoke.llm_invokes))
        block_types = [
            block.type
            for message in invoke.trace
            for block in message.content
        ]
        self.assertIn("function_call", block_types)
        self.assertIn("function_response", block_types)
        self.assertEqual(invoke.output[-1].text_content(), "tool flow complete")

    def test_stream_and_async_invocations_use_the_same_records(self):
        agent = BasicAgent(name="mixed", llm=DummyLLM()).with_observability(
            store=InMemoryObservabilityStore()
        )

        stream_events = list(agent.stream("stream query"))
        self.assertEqual(stream_events[-1].content, "stream response")
        self.assertEqual(asyncio.run(agent.ainvoke("async query")), "plain response")

        records = agent.observability.list()
        self.assertEqual(len(records), 2)
        self.assertEqual([item.query for item in records], ["stream query", "async query"])
        self.assertTrue(all(item.stats.llm_calls == 1 for item in records))
        self.assertTrue(all(item.stats.success for item in records))

    def test_sqlite_store_persists_finalized_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "observability.sqlite3")
            agent = BasicAgent(name="sqlite", llm=DummyLLM()).with_observability(path=path)
            agent.invoke("persist this")
            invoke_id = agent.observability.latest().invoke_id
            agent.close()

            reopened = SQLiteObservabilityStore(path)
            try:
                restored = reopened.get(invoke_id)
                self.assertIsNotNone(restored)
                self.assertEqual(restored.query, "persist this")
                self.assertEqual(restored.llm_invokes[0].stats.total_tokens, 15)
            finally:
                reopened.close()

    def test_observability_records_follow_session_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            session_store = SessionStore(os.path.join(directory, "sessions.sqlite3"))
            agent = BasicAgent(name="session", llm=DummyLLM()).with_observability(
                store=InMemoryObservabilityStore()
            )
            agent.invoke("persist through session")
            invoke_id = agent.observability.latest().invoke_id
            agent.save_session("observability-session", store=session_store)

            restored = BasicAgent.load_session(
                "observability-session",
                llm=DummyLLM(),
                store=session_store,
            )
            self.assertIsNotNone(restored.observability)
            self.assertEqual(restored.observability.latest().invoke_id, invoke_id)
            self.assertEqual(
                restored.observability.latest().query,
                "persist through session",
            )

    def test_subagent_records_share_store_and_form_one_rollout(self):
        store = InMemoryObservabilityStore()
        parent = BasicAgent(name="parent", llm=DummyLLM()).with_observability(
            store=store
        )
        parent_invoke_id = parent.observability.begin_agent_invoke(
            query="delegate work",
            mode="tool",
            stream=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            tool = AgentTool(
                parent_agent=parent,
                agent_factory=lambda request: BasicAgent(
                    name=request.name or "child",
                    llm=DummyLLM(),
                ),
                storage_dir=directory,
            )
            try:
                result = tool.run(
                    {
                        "description": "child task",
                        "prompt": "complete child task",
                        "name": "child",
                    }
                )
                self.assertEqual(result.status, "success")
            finally:
                tool.agent_runtime.close()

        parent.observability.end_agent_invoke(
            parent_invoke_id,
            output=parent.llm.assistant_message_to_canonical(content="delegated"),
            success=True,
        )
        records = store.list()
        self.assertEqual(len(records), 2)
        child = next(item for item in records if item.agent_id == "child")
        self.assertEqual(child.parent_invoke_id, parent_invoke_id)

        rollouts = TrainingExporter.from_agent(parent).build("agentic_rollout")
        self.assertEqual(len(rollouts), 1)
        self.assertEqual(len(rollouts[0]["agent_invokes"]), 2)

    def test_training_export_uses_success_filter_and_supports_custom_cleaning(self):
        store = InMemoryObservabilityStore()
        successful = BasicAgent(name="success", llm=DummyLLM()).with_observability(
            store=store
        )
        failing = BasicAgent(
            name="failure",
            llm=DummyLLM(FailingProvider()),
        ).with_observability(store=store)
        successful.invoke("export this successful invoke")
        successful.observability.annotate({"reward": 1.0})
        with self.assertRaises(Exception):
            failing.invoke("do not export this failure")

        exporter = TrainingExporter(store)
        step_records = exporter.build(TrainingDataFormat.STEP_SFT)
        trace_records = exporter.build(TrainingDataFormat.TRACE_SFT)
        rollout_records = exporter.build(TrainingDataFormat.AGENTIC_ROLLOUT)

        self.assertEqual(len(step_records), 1)
        self.assertEqual(len(trace_records), 1)
        self.assertEqual(len(rollout_records), 1)
        self.assertEqual(step_records[0]["format"], "step_sft")
        self.assertEqual(trace_records[0]["format"], "trace_sft")
        self.assertEqual(trace_records[0]["metadata"]["reward"], 1.0)
        self.assertEqual(rollout_records[0]["format"], "agentic_rollout")

        cleaned = exporter.build("trace_sft", data_filter=QueryFilter())
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["metadata"], {"cleaned": True})

        with tempfile.TemporaryDirectory() as directory:
            report = exporter.export(directory)
            self.assertEqual(report.source_records, 2)
            self.assertEqual(report.accepted_records, 1)
            self.assertEqual(set(report.files), {
                "step_sft",
                "trace_sft",
                "agentic_rollout",
            })
            for path in report.files.values():
                with open(path, encoding="utf-8") as handle:
                    self.assertTrue(json.loads(handle.readline()))

    def test_training_export_accepts_custom_observability_manager(self):
        ended_at = datetime.now(timezone.utc)
        user_message = CanonicalMessage(
            role="user",
            content=[CanonicalBlock(type="text", text="custom input")],
        )
        assistant_message = CanonicalMessage(
            role="assistant",
            content=[CanonicalBlock(type="text", text="custom output")],
        )
        record = AgentInvoke(
            invoke_id="agent-invoke-custom",
            agent_id="custom-agent",
            query="custom input",
            trace=[user_message, assistant_message],
            output=[assistant_message],
            llm_invokes=[
                LLMInvoke(
                    invoke_id="llm-invoke-custom",
                    input=[user_message],
                    output=[assistant_message],
                    stats=LLMInvokeStats(
                        success=True,
                        status="success",
                        ended_at=ended_at,
                    ),
                )
            ],
            stats=AgentInvokeStats(
                success=True,
                status="success",
                ended_at=ended_at,
                llm_calls=1,
            ),
        )

        records = TrainingExporter(CustomObservabilityManager([record])).build(
            TrainingDataFormat.TRACE_SFT
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_agent_invoke_id"], record.invoke_id)
        self.assertEqual(records[0]["trace"][-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
