from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from core import enable_logging
from core.callbacks import BaseCallback, CallbackManager
from core.history import _json_safe
from core.llm import EasyLLM
from skill.builtin.calculator_skill import CalculatorSkill


ARTIFACT_DIR = PROJECT_ROOT / "example" / "_artifacts"
QUERY = "必须使用计算器工具计算 3^22，并说明你使用了工具。"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _summarize_parts(parts: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not isinstance(parts, list):
        return summary
    for part in parts:
        if not isinstance(part, dict):
            summary.append({"kind": type(part).__name__})
            continue
        if "text" in part:
            item: dict[str, Any] = {
                "kind": "thinking" if part.get("thought") else "text",
                "text": part.get("text"),
            }
            if part.get("thought_signature") is not None:
                item["thought_signature"] = part.get("thought_signature")
            summary.append(item)
            continue
        function_call = part.get("function_call")
        if isinstance(function_call, dict):
            item = {
                "kind": "function_call",
                "id": function_call.get("id"),
                "name": function_call.get("name"),
                "args": function_call.get("args"),
            }
            if part.get("thought_signature") is not None:
                item["thought_signature"] = part.get("thought_signature")
            summary.append(item)
            continue
        function_response = part.get("function_response")
        if isinstance(function_response, dict):
            summary.append(
                {
                    "kind": "function_response",
                    "id": function_response.get("id"),
                    "name": function_response.get("name"),
                    "response": function_response.get("response"),
                }
            )
            continue
        summary.append({"kind": "raw", "payload": _json_safe(part)})
    return summary


def _summarize_contents(contents: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(contents, list):
        return rows
    for index, message in enumerate(contents):
        if not isinstance(message, dict):
            rows.append({"index": index, "kind": type(message).__name__})
            continue
        rows.append(
            {
                "index": index,
                "role": message.get("role"),
                "parts": _summarize_parts(message.get("parts")),
            }
        )
    return rows


class DebugCallback(BaseCallback):
    def __init__(self, state: dict[str, Any]):
        self.state = state

    def on_llm_start(self, messages, **kwargs) -> None:
        replay_history = getattr(messages, "replay_history", messages)
        payload = _json_safe(replay_history)
        self.state.setdefault("llm_starts", []).append(payload)
        print("\n[callback] on_llm_start")
        print(json.dumps(_summarize_contents(payload), ensure_ascii=False, indent=2))

    def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs) -> None:
        item = {"tool_name": tool_name, "tool_input": _json_safe(tool_input)}
        self.state.setdefault("tool_starts", []).append(item)
        print(f"\n[callback] on_tool_start {tool_name}: {json.dumps(item['tool_input'], ensure_ascii=False)}")

    def on_tool_end(self, tool_name: str, tool_output: str, success: bool = True, error=None, **kwargs) -> None:
        item = {
            "tool_name": tool_name,
            "tool_output": tool_output,
            "success": success,
            "error": repr(error) if error else None,
        }
        self.state.setdefault("tool_ends", []).append(item)
        print(f"\n[callback] on_tool_end {tool_name}: success={success} output={tool_output!r}")


@dataclass
class RunState:
    requests: list[dict[str, Any]] = field(default_factory=list)
    raw_chunks: list[list[Any]] = field(default_factory=list)
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    llm_starts: list[Any] = field(default_factory=list)
    tool_starts: list[Any] = field(default_factory=list)
    tool_ends: list[Any] = field(default_factory=list)
    final_result: str | None = None
    error: str | None = None
    traceback_text: str | None = None
    raw_history: Any = None
    canonical_history: Any = None
    trace_history: Any = None
    pending_step_state: Any = None


def _serialize_raw_chunk(chunk: Any) -> Any:
    if hasattr(chunk, "model_dump"):
        try:
            return _json_safe(chunk.model_dump())
        except Exception:
            pass
    if hasattr(chunk, "to_dict"):
        try:
            return _json_safe(chunk.to_dict())
        except Exception:
            pass
    return _json_safe(chunk)


def _wrap_async_stream(stream: Any, bucket: list[Any]):
    async def _iter():
        async for chunk in stream:
            bucket.append(_serialize_raw_chunk(chunk))
            yield chunk

    return _iter()


def _install_request_probe(llm: EasyLLM, state: RunState) -> None:
    provider = llm._provider
    original_async_stream_raw = provider.async_stream_raw

    async def debug_async_stream_raw(request: Any) -> Any:
        safe_request = _json_safe(request)
        request_index = len(state.requests) + 1
        state.requests.append(safe_request)
        raw_chunk_bucket: list[Any] = []
        state.raw_chunks.append(raw_chunk_bucket)
        print(f"\n===== outbound request #{request_index} =====")
        print(json.dumps(_summarize_contents(safe_request.get('contents')), ensure_ascii=False, indent=2))
        try:
            stream = await original_async_stream_raw(request)
            return _wrap_async_stream(stream, raw_chunk_bucket)
        except Exception:
            print(f"\n===== outbound request #{request_index} failed =====")
            print(json.dumps(safe_request, ensure_ascii=False, indent=2))
            raise

    provider.async_stream_raw = debug_async_stream_raw


def _drop_function_ids(contents: Any) -> Any:
    payload = json.loads(json.dumps(_json_safe(contents)))
    if not isinstance(payload, list):
        return payload
    for message in payload:
        if not isinstance(message, dict):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("function_call")
            if isinstance(function_call, dict):
                function_call.pop("id", None)
            function_response = part.get("function_response")
            if isinstance(function_response, dict):
                function_response.pop("id", None)
    return payload


def _drop_thought_signatures(contents: Any) -> Any:
    payload = json.loads(json.dumps(_json_safe(contents)))
    if not isinstance(payload, list):
        return payload
    for message in payload:
        if not isinstance(message, dict):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                part.pop("thought_signature", None)
    return payload


def _drop_model_turn(contents: Any) -> Any:
    payload = json.loads(json.dumps(_json_safe(contents)))
    if not isinstance(payload, list):
        return payload
    return [
        message
        for message in payload
        if not (
            isinstance(message, dict)
            and message.get("role") == "model"
            and isinstance(message.get("parts"), list)
            and any(isinstance(part, dict) and part.get("function_call") for part in message["parts"])
        )
    ]


async def replay_attempt_request(
    attempt: int,
    request_index: int,
    *,
    strip_ids: bool = False,
    strip_signatures: bool = False,
    drop_model_turn: bool = False,
) -> int:
    artifact_path = ARTIFACT_DIR / f"google_native_tool_debug_attempt_{attempt}.json"
    if not artifact_path.exists():
        print(f"artifact not found: {artifact_path}")
        return 2

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    requests = artifact.get("requests")
    if not isinstance(requests, list) or request_index < 1 or request_index > len(requests):
        print(f"invalid request index {request_index}; available={len(requests) if isinstance(requests, list) else 0}")
        return 2

    request_payload = json.loads(json.dumps(_json_safe(requests[request_index - 1])))
    if strip_ids:
        request_payload["contents"] = _drop_function_ids(request_payload.get("contents"))
    if strip_signatures:
        request_payload["contents"] = _drop_thought_signatures(request_payload.get("contents"))
    if drop_model_turn:
        request_payload["contents"] = _drop_model_turn(request_payload.get("contents"))

    llm = EasyLLM(provider="google_native", model=request_payload.get("model"))
    state = RunState()
    _install_request_probe(llm, state)
    raw_chunk_bucket: list[Any] = []
    state.raw_chunks.append(raw_chunk_bucket)

    print("\n================ replay ================")
    print(
        "artifact="
        f"{artifact_path.name} request_index={request_index} strip_ids={strip_ids} "
        f"strip_signatures={strip_signatures} drop_model_turn={drop_model_turn}"
    )
    print(json.dumps(_summarize_contents(request_payload.get("contents")), ensure_ascii=False, indent=2))

    final_chunks: list[str] = []
    try:
        stream = await llm.provider.async_stream_raw(request_payload)
        async for chunk in _wrap_async_stream(stream, raw_chunk_bucket):
            text = getattr(chunk, "text", None)
            if text:
                print(f"[replay:text] {text!r}")
                final_chunks.append(text)
        print(f"[replay:final] {''.join(final_chunks)!r}")
        return 0
    except Exception as exc:
        print(f"[replay:error] {exc!r}")
        return 1


async def run_attempt(attempt: int, query: str) -> tuple[bool, Path]:
    llm = EasyLLM(provider="google_native")
    state = RunState()
    callback_manager = CallbackManager([DebugCallback(state.__dict__)])
    agent = BasicAgent(
        name=f"google-native-debug-{attempt}",
        llm=llm,
        enable_tool=True,
        verbose_thinking=True,
        callback_manager=callback_manager,
        tool_registry=ToolRegistry(),
    )
    agent.with_skill(CalculatorSkill())
    _install_request_probe(llm, state)

    print(f"\n================ attempt {attempt} ================")
    print(f"model={llm.model} provider={llm.provider_name}")

    final_chunks: list[str] = []
    try:
        async for event in agent.astream_invoke_with_tool(query, max_iter=6):
            safe_event = _json_safe(event)
            state.stream_events.append(safe_event)
            event_type = event.get("type")
            if event_type in {"text_delta", "thinking_delta"}:
                delta = event.get("delta", "")
                prefix = "thinking" if event_type == "thinking_delta" else "text"
                print(f"[event:{prefix}] {delta!r}")
                continue
            if event_type == "tool_call":
                print(
                    "[event:tool_call]",
                    json.dumps(
                        {
                            "tool_name": event.get("tool_name"),
                            "tool_id": event.get("tool_id"),
                            "tool_args": event.get("tool_args"),
                        },
                        ensure_ascii=False,
                    ),
                )
                continue
            if event_type == "tool_result":
                print(
                    "[event:tool_result]",
                    json.dumps(
                        {
                            "tool_name": event.get("tool_name"),
                            "tool_id": event.get("tool_id"),
                            "status": event.get("status"),
                            "content": event.get("content"),
                        },
                        ensure_ascii=False,
                    ),
                )
                continue
            if event_type == "final":
                final_chunks.append(event.get("content", "") or "")
            print(f"[event:{event_type}] {json.dumps(safe_event, ensure_ascii=False)}")
        state.final_result = "".join(final_chunks) or None
        success = True
    except Exception as exc:
        success = False
        state.error = repr(exc)
        state.traceback_text = traceback.format_exc()
        print("\n===== run failed =====")
        print(state.traceback_text)
    finally:
        state.raw_history = _json_safe(agent.get_raw_history())
        state.canonical_history = _json_safe(agent.get_canonical_history())
        state.trace_history = _json_safe(agent.get_trace_history())
        state.pending_step_state = _json_safe(agent.get_pending_step_state())
        report_path = ARTIFACT_DIR / f"google_native_tool_debug_attempt_{attempt}.json"
        _json_dump(
            report_path,
            {
                "attempt": attempt,
                "query": query,
                "success": success,
                "requests": state.requests,
                "raw_chunks": state.raw_chunks,
                "stream_events": state.stream_events,
                "llm_starts": state.llm_starts,
                "tool_starts": state.tool_starts,
                "tool_ends": state.tool_ends,
                "final_result": state.final_result,
                "error": state.error,
                "traceback": state.traceback_text,
                "raw_history": state.raw_history,
                "canonical_history": state.canonical_history,
                "trace_history": state.trace_history,
                "pending_step_state": state.pending_step_state,
            },
        )
        print(f"\nreport saved to: {report_path}")
    return success, report_path


async def main() -> int:
    parser = argparse.ArgumentParser(description="Debug google_native tool calling with real API")
    parser.add_argument("--attempts", type=int, default=10, help="How many attempts to run before stopping")
    parser.add_argument("--query", type=str, default=QUERY, help="Prompt to send to the agent")
    parser.add_argument("--replay-attempt", type=int, help="Replay one captured request from a saved artifact")
    parser.add_argument("--replay-request-index", type=int, default=2, help="Which request from the artifact to replay")
    parser.add_argument("--strip-ids", action="store_true", help="Drop function_call/function_response ids before replay")
    parser.add_argument("--strip-signatures", action="store_true", help="Drop thought_signature fields before replay")
    parser.add_argument("--drop-model-turn", action="store_true", help="Drop the model function_call turn before replay")
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Stop once a failure is observed",
    )
    args = parser.parse_args()

    load_dotenv()
    enable_logging()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if args.replay_attempt is not None:
        return await replay_attempt_request(
            args.replay_attempt,
            args.replay_request_index,
            strip_ids=args.strip_ids,
            strip_signatures=args.strip_signatures,
            drop_model_turn=args.drop_model_turn,
        )

    had_failure = False
    last_report: Path | None = None
    for attempt in range(1, args.attempts + 1):
        success, report_path = await run_attempt(attempt, args.query)
        last_report = report_path
        if not success:
            had_failure = True
            if args.stop_on_first_failure:
                break

    if had_failure:
        print(f"\nfinished with at least one failure. last report: {last_report}")
        return 1
    print(f"\nall attempts succeeded. last report: {last_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
