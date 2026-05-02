from __future__ import annotations

import asyncio
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
from observability import EvalCase, OfflineEvalHarness
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

    def apply_cache_policy(self, request, request_input):
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


class SequencedPlainProvider(ObservabilityProvider):
    def __init__(self, responses: list[str]):
        super().__init__()
        self._responses = list(responses)
        self._index = 0

    def invoke_raw(self, request):
        if self._index >= len(self._responses):
            content = self._responses[-1]
        else:
            content = self._responses[self._index]
        self._index += 1
        return _chat_response(
            content=content,
            thinking=f"reasoning for {content}",
            prompt_tokens=9,
            completion_tokens=4,
        )


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

        run_record = agent.export_run_record()
        self.assertIsNotNone(run_record)
        assert run_record is not None
        self.assertEqual(run_record["query"], "use the echo tool and finish the answer")
        self.assertEqual(run_record["summary"]["toolCalls"], 1)
        self.assertEqual(run_record["summary"]["llmRequests"], 2)
        self.assertEqual(run_record["summary"]["toolsUsed"], ["EchoTool"])
        self.assertTrue(any(item["type"] == "tool_call" for item in run_record["trace"]))
        self.assertTrue(any(item["type"] == "tool_result" for item in run_record["trace"]))

        eval_trace = agent.export_eval_trace()
        self.assertIsNotNone(eval_trace)
        assert eval_trace is not None
        self.assertEqual(eval_trace["tool_calls"], 1)
        self.assertEqual(eval_trace["llm_requests"], 2)
        self.assertEqual(eval_trace["status"], "success")

        examples = agent.export_training_examples()
        self.assertTrue(any(item["example_type"] == "tool_selection" for item in examples))
        self.assertTrue(any(item["example_type"] == "planning" for item in examples))
        self.assertTrue(any(item["example_type"] == "final_response" for item in examples))

        run_record_jsonl = agent.export_run_record_jsonl()
        eval_trace_jsonl = agent.export_eval_trace_jsonl()
        examples_jsonl = agent.export_training_examples_jsonl()
        sft_jsonl = agent.export_sft_dataset_jsonl()
        self.assertEqual(json.loads(run_record_jsonl)["query"], "use the echo tool and finish the answer")
        self.assertEqual(json.loads(eval_trace_jsonl)["tool_calls"], 1)
        self.assertGreaterEqual(len(examples_jsonl.splitlines()), 3)
        first_sft = json.loads(sft_jsonl.splitlines()[0])
        self.assertIn("messages", first_sft)
        self.assertIn("prompt", first_sft)
        self.assertIn("completion", first_sft)

    def test_run_outcome_label_is_exported_and_redaction_is_supported(self):
        provider = ObservabilityProvider()
        agent = BasicAgent(
            name="observability-outcome",
            llm=DummyLLM(provider),
        )

        result = agent.invoke("summarize current runtime state")
        self.assertEqual(result, "plain response")

        labeled = agent.label_run_outcome(
            status="success",
            success=True,
            changed_files=["src/app.py"],
            tools_used=["EchoTool"],
            root_cause_tags=["verified"],
            tests_attempted=["pytest -q"],
            tests_passed=["pytest -q"],
            notes="accepted by evaluator",
        )
        self.assertEqual(labeled["status"], "success")
        self.assertEqual(labeled["changed_files"], ["src/app.py"])

        run_record = agent.export_run_record()
        self.assertIsNotNone(run_record)
        assert run_record is not None
        self.assertEqual(run_record["outcome"]["changed_files"], ["src/app.py"])
        self.assertEqual(run_record["outcome"]["tests_passed"], ["pytest -q"])
        verification_examples = agent.export_training_examples()
        self.assertTrue(any(item["example_type"] == "verification_summary" for item in verification_examples))

        redacted = agent.export_run_record(redact=True)
        self.assertIsNotNone(redacted)
        assert redacted is not None
        self.assertEqual(redacted["query"], "[redacted]")
        self.assertEqual(redacted["final_output"], "[redacted]")

    def test_run_record_links_cache_breaks_to_latest_run(self):
        agent = BasicAgent(
            name="observability-cache-break",
            llm=DummyLLM(ObservabilityProvider()),
        )

        agent.invoke("cache break linkage query")
        latest_run = agent.export_run_record()
        self.assertIsNotNone(latest_run)
        assert latest_run is not None
        cache_signature = latest_run["llm_requests"][-1]["metadata"]["cacheSignature"]
        agent.observability_recorder.record_cache_break(
            reason="cache_signature_changed",
            changed_fields=["messages.query"],
            previous_signature={"messages": "before"},
            current_signature=cache_signature,
            metadata={"turnId": latest_run["turn_id"]},
        )

        summary = agent.get_observability_summary()
        self.assertGreaterEqual(summary["cacheBreaks"], 1)

        run_record = agent.export_run_record()
        self.assertIsNotNone(run_record)
        assert run_record is not None
        self.assertGreaterEqual(run_record["summary"]["cacheBreaks"], 1)
        self.assertTrue(run_record["summary"]["cacheBreakReasons"])
        self.assertTrue(run_record["cache_breaks"])

        eval_trace = agent.export_eval_trace()
        self.assertIsNotNone(eval_trace)
        assert eval_trace is not None
        self.assertGreaterEqual(len(eval_trace["cache_break_reasons"]), 1)

    def test_offline_eval_harness_scores_cases(self):
        harness = OfflineEvalHarness()

        def build_agent():
            registry = ToolRegistry()
            registry.register_tool(EchoTool())
            return BasicAgent(
                name="observability-eval",
                llm=DummyLLM(ObservabilityProvider()),
                enable_tool=True,
                tool_registry=registry,
            )

        cases = [
            EvalCase(
                case_id="tool-flow",
                query="use the echo tool and finish the answer",
                expected_output_contains=["tool flow complete"],
                expected_tools=["EchoTool"],
                max_tool_calls=1,
            ),
            EvalCase(
                case_id="too-strict",
                query="use the echo tool and finish the answer",
                expected_output_contains=["missing snippet"],
            ),
        ]

        results = harness.run_cases(build_agent, cases)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["success"])
        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertFalse(results[1]["success"])
        self.assertTrue(results[1]["failure_reasons"])

        summary = harness.summarize(results)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["successes"], 1)

    def test_sft_dataset_and_preference_pair_exports(self):
        provider = SequencedPlainProvider(["weak answer", "strong answer"])
        agent = BasicAgent(
            name="observability-datasets",
            llm=DummyLLM(provider),
        )

        query = "compare two candidate answers"
        first_result = agent.invoke(query)
        second_result = agent.invoke(query)
        self.assertEqual(first_result, "weak answer")
        self.assertEqual(second_result, "strong answer")

        runs = agent.list_agent_runs()
        self.assertEqual(len(runs), 2)
        first_run_id = runs[0]["id"]
        second_run_id = runs[1]["id"]

        agent.label_run_outcome(
            run_id=first_run_id,
            status="failed_verification",
            success=False,
            root_cause_tags=["insufficient_answer"],
            notes="rejected by evaluator",
            metadata={"preferenceGroup": "compare-query", "preferenceScore": 0.1},
        )
        agent.label_run_outcome(
            run_id=second_run_id,
            status="success",
            success=True,
            root_cause_tags=["accepted"],
            notes="accepted by evaluator",
            metadata={"preferenceGroup": "compare-query", "preferenceScore": 1.0},
        )

        sft_dataset = agent.export_sft_dataset(run_ids=[first_run_id, second_run_id])
        self.assertGreaterEqual(len(sft_dataset), 4)
        self.assertTrue(all("messages" in item for item in sft_dataset))

        filtered_sft = agent.export_sft_dataset(example_types=["final_response"])
        self.assertEqual(len(filtered_sft), 2)
        self.assertTrue(all(item["example_type"] == "final_response" for item in filtered_sft))

        preference_pairs = agent.export_preference_pairs()
        self.assertEqual(len(preference_pairs), 1)
        self.assertEqual(preference_pairs[0]["chosen"], "strong answer")
        self.assertEqual(preference_pairs[0]["rejected"], "weak answer")
        self.assertEqual(preference_pairs[0]["chosen_run_id"], second_run_id)
        self.assertEqual(preference_pairs[0]["rejected_run_id"], first_run_id)

        explicit_pairs = agent.export_preference_pairs(
            chosen_run_ids=[second_run_id],
            rejected_run_ids=[first_run_id],
        )
        self.assertEqual(len(explicit_pairs), 1)
        self.assertEqual(explicit_pairs[0]["chosen_score"], 1.0)
        self.assertEqual(explicit_pairs[0]["rejected_score"], 0.1)

        preference_jsonl = agent.export_preference_pairs_jsonl()
        self.assertEqual(json.loads(preference_jsonl.splitlines()[0])["chosen"], "strong answer")

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

                restored.label_run_outcome(
                    status="partial_success",
                    success=True,
                    failure_stage=None,
                    root_cause_tags=["user_review_pending"],
                    notes="restored outcome persisted",
                )
                restored.save_session("obs-session", store=store)

                restored_again = BasicAgent.load_session(
                    "obs-session",
                    llm=DummyLLM(ObservabilityProvider()),
                    store=store,
                )
                try:
                    run_record = restored_again.export_run_record()
                    self.assertIsNotNone(run_record)
                    assert run_record is not None
                    self.assertEqual(run_record["outcome"]["status"], "partial_success")
                    self.assertEqual(run_record["outcome"]["root_cause_tags"], ["user_review_pending"])
                finally:
                    restored_again.close(close_worktree=False)

                restored.clear_observability()
                cleared = restored.get_observability_summary()
                self.assertEqual(cleared["agentRuns"], 0)
                self.assertEqual(cleared["llmRequests"], 0)
                self.assertEqual(cleared["toolCalls"], 0)
            finally:
                restored.close(close_worktree=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
