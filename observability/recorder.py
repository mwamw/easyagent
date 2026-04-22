"""In-memory observability recorder for agent/runtime execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime, timezone
import time
from typing import Any, Optional
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    return str(value)


def _avg_ms(items: list[dict[str, Any]]) -> Optional[float]:
    values = [float(item["durationMs"]) for item in items if item.get("durationMs") is not None]
    if not values:
        return None
    return sum(values) / len(values)


class BaseObservabilityRecorder(ABC):
    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class InMemoryObservabilityRecorder(BaseObservabilityRecorder):
    """Collects agent, LLM, and tool execution telemetry in memory."""

    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        self._session_id = session_id or f"obs_{uuid4().hex}"
        self._agent_name = agent_name
        self._event_counter = 0
        self._agent_runs: list[dict[str, Any]] = []
        self._llm_requests: list[dict[str, Any]] = []
        self._tool_executions: list[dict[str, Any]] = []
        self._open_agent_runs: dict[str, dict[str, Any]] = {}
        self._open_llm_requests: dict[str, dict[str, Any]] = {}
        self._open_tool_executions: dict[str, dict[str, Any]] = {}

    def export_state(self) -> dict[str, Any]:
        return {
            "sessionId": self._session_id,
            "agentName": self._agent_name,
            "eventCounter": self._event_counter,
            "agentRuns": [_json_safe(item) for item in self._agent_runs],
            "llmRequests": [_json_safe(item) for item in self._llm_requests],
            "toolExecutions": [_json_safe(item) for item in self._tool_executions],
        }

    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        payload = dict(state or {})
        self._session_id = str(payload.get("sessionId") or f"obs_{uuid4().hex}")
        self._agent_name = str(payload.get("agentName") or "") or None
        self._event_counter = int(payload.get("eventCounter") or 0)
        self._agent_runs = [dict(item) for item in list(payload.get("agentRuns") or []) if isinstance(item, dict)]
        self._llm_requests = [dict(item) for item in list(payload.get("llmRequests") or []) if isinstance(item, dict)]
        self._tool_executions = [dict(item) for item in list(payload.get("toolExecutions") or []) if isinstance(item, dict)]
        self._open_agent_runs = {}
        self._open_llm_requests = {}
        self._open_tool_executions = {}

    def clear(self) -> None:
        self._event_counter = 0
        self._agent_runs.clear()
        self._llm_requests.clear()
        self._tool_executions.clear()
        self._open_agent_runs.clear()
        self._open_llm_requests.clear()
        self._open_tool_executions.clear()

    def set_agent_name(self, agent_name: Optional[str]) -> None:
        self._agent_name = str(agent_name or "") or None

    def begin_agent_run(
        self,
        *,
        query: str,
        mode: str,
        stream: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = self._next_event_id("agent")
        self._open_agent_runs[event_id] = {
            "id": event_id,
            "sessionId": self._session_id,
            "agentName": self._agent_name,
            "query": str(query or ""),
            "mode": str(mode or "unknown"),
            "stream": bool(stream),
            "metadata": dict(metadata or {}),
            "startedAt": _now_iso(),
            "startedPerf": time.perf_counter(),
        }
        return event_id

    def end_agent_run(
        self,
        event_id: str,
        *,
        output: str,
        success: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        turn_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        started = self._open_agent_runs.pop(event_id, None)
        if started is None:
            started = {
                "id": event_id,
                "sessionId": self._session_id,
                "agentName": self._agent_name,
                "query": "",
                "mode": "unknown",
                "stream": False,
                "metadata": {},
                "startedAt": _now_iso(),
                "startedPerf": time.perf_counter(),
            }
        duration_ms = max(0.0, (time.perf_counter() - float(started.pop("startedPerf", time.perf_counter()))) * 1000.0)
        payload = {
            **started,
            "turnId": turn_id,
            "success": bool(success),
            "status": "success" if success else "error",
            "outputPreview": str(output or "")[:500],
            "errorType": error_type,
            "errorMessage": error_message,
            "durationMs": duration_ms,
            "endedAt": _now_iso(),
        }
        merged_metadata = dict(started.get("metadata") or {})
        merged_metadata.update(dict(metadata or {}))
        payload["metadata"] = merged_metadata
        self._agent_runs.append(payload)
        return payload

    def begin_llm_request(
        self,
        *,
        turn_id: Optional[str],
        request_kind: str,
        stream: bool,
        tools_enabled: bool,
        provider_name: Optional[str],
        model: Optional[str],
        input_tokens: Optional[int],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = self._next_event_id("llm")
        self._open_llm_requests[event_id] = {
            "id": event_id,
            "sessionId": self._session_id,
            "agentName": self._agent_name,
            "turnId": turn_id,
            "requestKind": str(request_kind or "unknown"),
            "stream": bool(stream),
            "toolsEnabled": bool(tools_enabled),
            "providerName": provider_name,
            "model": model,
            "inputTokens": input_tokens,
            "metadata": dict(metadata or {}),
            "startedAt": _now_iso(),
            "startedPerf": time.perf_counter(),
        }
        return event_id

    def end_llm_request(
        self,
        event_id: str,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        cached_input_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        tool_use_prompt_tokens: Optional[int] = None,
        usage_source: Optional[str],
        success: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        started = self._open_llm_requests.pop(event_id, None)
        if started is None:
            started = {
                "id": event_id,
                "sessionId": self._session_id,
                "agentName": self._agent_name,
                "turnId": None,
                "requestKind": "unknown",
                "stream": False,
                "toolsEnabled": False,
                "providerName": None,
                "model": None,
                "inputTokens": None,
                "metadata": {},
                "startedAt": _now_iso(),
                "startedPerf": time.perf_counter(),
            }
        duration_ms = max(0.0, (time.perf_counter() - float(started.pop("startedPerf", time.perf_counter()))) * 1000.0)
        if input_tokens is None:
            input_tokens = started.get("inputTokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)
        payload = {
            **started,
            "inputTokens": input_tokens,
            "success": bool(success),
            "status": "success" if success else "error",
            "outputTokens": output_tokens,
            "totalTokens": total_tokens,
            "cachedInputTokens": cached_input_tokens,
            "reasoningTokens": reasoning_tokens,
            "cacheReadTokens": cache_read_tokens,
            "cacheCreationTokens": cache_creation_tokens,
            "toolUsePromptTokens": tool_use_prompt_tokens,
            "usageSource": usage_source,
            "errorType": error_type,
            "errorMessage": error_message,
            "costUsd": cost_usd,
            "durationMs": duration_ms,
            "endedAt": _now_iso(),
        }
        merged_metadata = dict(started.get("metadata") or {})
        merged_metadata.update(dict(metadata or {}))
        payload["metadata"] = merged_metadata
        self._llm_requests.append(payload)
        return payload

    def begin_tool_execution(
        self,
        *,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: dict[str, Any],
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        round_number: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = self._next_event_id("tool")
        self._open_tool_executions[event_id] = {
            "id": event_id,
            "sessionId": self._session_id,
            "agentName": self._agent_name,
            "turnId": turn_id,
            "toolName": str(tool_name or ""),
            "toolArgs": _json_safe(tool_args),
            "mode": mode,
            "stream": stream,
            "round": round_number,
            "metadata": dict(metadata or {}),
            "startedAt": _now_iso(),
            "startedPerf": time.perf_counter(),
        }
        return event_id

    def end_tool_execution(
        self,
        event_id: str,
        *,
        success: bool,
        result_status: Optional[str],
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        started = self._open_tool_executions.pop(event_id, None)
        if started is None:
            started = {
                "id": event_id,
                "sessionId": self._session_id,
                "agentName": self._agent_name,
                "turnId": None,
                "toolName": "",
                "toolArgs": {},
                "mode": None,
                "stream": None,
                "round": None,
                "metadata": {},
                "startedAt": _now_iso(),
                "startedPerf": time.perf_counter(),
            }
        duration_ms = max(0.0, (time.perf_counter() - float(started.pop("startedPerf", time.perf_counter()))) * 1000.0)
        payload = {
            **started,
            "success": bool(success),
            "status": result_status or ("success" if success else "error"),
            "errorType": error_type,
            "errorMessage": error_message,
            "durationMs": duration_ms,
            "endedAt": _now_iso(),
        }
        merged_metadata = dict(started.get("metadata") or {})
        merged_metadata.update(dict(metadata or {}))
        payload["metadata"] = merged_metadata
        self._tool_executions.append(payload)
        return payload

    def get_summary(self) -> dict[str, Any]:
        agent_runs = list(self._agent_runs)
        llm_requests = list(self._llm_requests)
        tool_executions = list(self._tool_executions)
        error_types = Counter()
        request_kinds = Counter()
        tools_used = Counter()
        last_error = None
        costs: list[float] = []

        for item in llm_requests:
            request_kinds[str(item.get("requestKind") or "unknown")] += 1
            if item.get("errorType"):
                error_types[str(item["errorType"])] += 1
                last_error = item
            if item.get("costUsd") is not None:
                try:
                    costs.append(float(item["costUsd"]))
                except Exception:
                    pass

        for item in tool_executions:
            tools_used[str(item.get("toolName") or "unknown")] += 1
            if item.get("errorType"):
                error_types[str(item["errorType"])] += 1
                last_error = item

        for item in agent_runs:
            if item.get("errorType"):
                error_types[str(item["errorType"])] += 1
                last_error = item

        input_tokens = sum(int(item.get("inputTokens") or 0) for item in llm_requests)
        output_tokens = sum(int(item.get("outputTokens") or 0) for item in llm_requests)
        total_tokens = sum(int(item.get("totalTokens") or 0) for item in llm_requests)
        cached_input_tokens = sum(int(item.get("cachedInputTokens") or 0) for item in llm_requests)
        reasoning_tokens = sum(int(item.get("reasoningTokens") or 0) for item in llm_requests)
        cache_read_tokens = sum(int(item.get("cacheReadTokens") or 0) for item in llm_requests)
        cache_creation_tokens = sum(int(item.get("cacheCreationTokens") or 0) for item in llm_requests)
        tool_use_prompt_tokens = sum(int(item.get("toolUsePromptTokens") or 0) for item in llm_requests)

        summary = {
            "sessionId": self._session_id,
            "agentName": self._agent_name,
            "agentRuns": len(agent_runs),
            "successfulAgentRuns": sum(1 for item in agent_runs if item.get("success")),
            "failedAgentRuns": sum(1 for item in agent_runs if not item.get("success")),
            "llmRequests": len(llm_requests),
            "llmErrors": sum(1 for item in llm_requests if not item.get("success")),
            "toolCalls": len(tool_executions),
            "toolErrors": sum(1 for item in tool_executions if not item.get("success")),
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens,
            "cachedInputTokens": cached_input_tokens,
            "reasoningTokens": reasoning_tokens,
            "cacheReadTokens": cache_read_tokens,
            "cacheCreationTokens": cache_creation_tokens,
            "toolUsePromptTokens": tool_use_prompt_tokens,
            "estimatedCostUsd": (sum(costs) if costs else None),
            "avgAgentDurationMs": _avg_ms(agent_runs),
            "avgLlmDurationMs": _avg_ms(llm_requests),
            "avgToolDurationMs": _avg_ms(tool_executions),
            "requestKinds": dict(request_kinds),
            "toolsUsed": dict(tools_used),
            "errorTypes": dict(error_types),
            "openRequests": {
                "agentRuns": len(self._open_agent_runs),
                "llmRequests": len(self._open_llm_requests),
                "toolExecutions": len(self._open_tool_executions),
            },
            "updatedAt": _now_iso(),
        }
        if last_error is not None:
            summary["lastError"] = {
                "type": last_error.get("errorType"),
                "message": last_error.get("errorMessage"),
                "source": last_error.get("requestKind") or last_error.get("toolName") or last_error.get("agentName"),
                "timestamp": last_error.get("endedAt"),
            }
        return summary

    def get_recent_events(
        self,
        *,
        limit: int = 20,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if event_type in {None, "agent"}:
            events.extend({**item, "eventType": "agent"} for item in self._agent_runs)
        if event_type in {None, "llm"}:
            events.extend({**item, "eventType": "llm"} for item in self._llm_requests)
        if event_type in {None, "tool"}:
            events.extend({**item, "eventType": "tool"} for item in self._tool_executions)
        events.sort(key=lambda item: str(item.get("endedAt") or item.get("startedAt") or ""), reverse=True)
        return [_json_safe(item) for item in events[: max(1, int(limit))]]

    def get_trace_summary(
        self,
        trace_history: list[dict[str, Any]],
        *,
        limit_turns: int = 5,
    ) -> list[dict[str, Any]]:
        normalized_trace = [dict(item) for item in list(trace_history or []) if isinstance(item, dict)]
        turn_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ordered_turns: list[str] = []
        seen_turns: set[str] = set()
        for event in normalized_trace:
            turn_id = str(event.get("turn_id") or event.get("turnId") or "").strip()
            if not turn_id:
                continue
            turn_events[turn_id].append(event)
            if turn_id not in seen_turns:
                seen_turns.add(turn_id)
                ordered_turns.append(turn_id)

        llm_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self._llm_requests:
            turn_id = str(item.get("turnId") or "").strip()
            if turn_id:
                llm_by_turn[turn_id].append(item)

        tool_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self._tool_executions:
            turn_id = str(item.get("turnId") or "").strip()
            if turn_id:
                tool_by_turn[turn_id].append(item)

        summaries: list[dict[str, Any]] = []
        for turn_id in reversed(ordered_turns[-max(1, int(limit_turns)):]):
            events = turn_events.get(turn_id, [])
            user_message = next((item for item in events if item.get("type") == "user_message"), {})
            turn_end = next((item for item in reversed(events) if item.get("type") == "turn_end"), {})
            llm_items = llm_by_turn.get(turn_id, [])
            tool_items = tool_by_turn.get(turn_id, [])
            summaries.append(
                {
                    "turnId": turn_id,
                    "query": str(user_message.get("content") or ""),
                    "status": ((turn_end.get("metadata") or {}).get("status") or "completed"),
                    "assistantMessages": sum(1 for item in events if item.get("type") == "assistant_message"),
                    "reasoningEvents": sum(1 for item in events if item.get("type") == "reasoning"),
                    "traceToolCalls": sum(1 for item in events if item.get("type") == "tool_call"),
                    "llmRequests": len(llm_items),
                    "toolCalls": len(tool_items),
                    "inputTokens": sum(int(item.get("inputTokens") or 0) for item in llm_items),
                    "outputTokens": sum(int(item.get("outputTokens") or 0) for item in llm_items),
                    "totalTokens": sum(int(item.get("totalTokens") or 0) for item in llm_items),
                    "cachedInputTokens": sum(int(item.get("cachedInputTokens") or 0) for item in llm_items),
                    "reasoningTokens": sum(int(item.get("reasoningTokens") or 0) for item in llm_items),
                    "cacheReadTokens": sum(int(item.get("cacheReadTokens") or 0) for item in llm_items),
                    "cacheCreationTokens": sum(int(item.get("cacheCreationTokens") or 0) for item in llm_items),
                    "toolUsePromptTokens": sum(int(item.get("toolUsePromptTokens") or 0) for item in llm_items),
                    "llmDurationMs": sum(float(item.get("durationMs") or 0.0) for item in llm_items),
                    "toolDurationMs": sum(float(item.get("durationMs") or 0.0) for item in tool_items),
                    "toolsUsed": sorted({str(item.get("toolName") or "") for item in tool_items if item.get("toolName")}),
                    "startedAt": str(user_message.get("timestamp") or ""),
                    "endedAt": str(turn_end.get("timestamp") or ""),
                }
            )
        return summaries

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{prefix}_{self._event_counter:06d}"


__all__ = ["BaseObservabilityRecorder", "InMemoryObservabilityRecorder"]
