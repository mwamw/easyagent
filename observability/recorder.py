"""In-memory observability recorder for agent/runtime execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import json as _stdlib_json
import time
from typing import Any, Optional
from uuid import uuid4

from .run_data import (
    AgentRunRecord,
    EvalTrace,
    PreferencePairRecord,
    RunOutcomeReport,
    SFTDatasetRecord,
    TrainingExample,
)


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


def _normalize_cache_accounting(item: dict[str, Any]) -> tuple[int, int, int]:
    semantics = str(item.get("cacheUsageSemantics") or "unknown")
    input_tokens = int(item.get("inputTokens") or 0)
    cached_input_tokens = int(item.get("cachedInputTokens") or 0)
    cache_read_tokens = int(item.get("cacheReadTokens") or 0)

    if semantics == "anthropic_style":
        prompt_cached = cache_read_tokens
        prompt_uncached = input_tokens
        prompt_total = prompt_cached + prompt_uncached
        return prompt_total, prompt_uncached, prompt_cached

    if semantics in {"openai_style", "google_style"}:
        prompt_total = input_tokens
        prompt_cached = cached_input_tokens
        prompt_uncached = max(0, prompt_total - prompt_cached)
        return prompt_total, prompt_uncached, prompt_cached

    prompt_cached = cache_read_tokens or cached_input_tokens
    prompt_total = input_tokens if input_tokens > 0 else prompt_cached
    prompt_uncached = max(0, prompt_total - prompt_cached)
    return prompt_total, prompt_uncached, prompt_cached


def _cache_layer_name(field: str) -> str:
    value = str(field or "")
    if value in {"system_hash", "cache_policy_hash"} or value.startswith("system."):
        return "system"
    if value in {"tools_hash"} or value.startswith("tools.") or value.startswith("expanded_tool"):
        return "tools"
    if value in {"reasoning_hash"} or value.startswith("reasoning."):
        return "reasoning"
    if value.startswith("runtime_reminder") or value.startswith("reminder."):
        return "runtime_reminders"
    if value.startswith("skill") or "skill" in value:
        return "skills"
    if value.startswith("message") or value.startswith("history") or value == "replay_history":
        return "messages"
    if value.startswith("provider") or value.startswith("model"):
        return "provider"
    return "other"


def _signature_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(_json_safe(value))


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

    @abstractmethod
    def export_run_record(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def export_eval_trace(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def export_training_examples(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def export_run_record_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def export_eval_trace_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def export_training_examples_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def export_sft_dataset(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        redact: bool = False,
        example_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def export_sft_dataset_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        redact: bool = False,
        example_types: Optional[list[str]] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def export_preference_pairs(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        chosen_run_ids: Optional[list[str]] = None,
        rejected_run_ids: Optional[list[str]] = None,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def export_preference_pairs_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        chosen_run_ids: Optional[list[str]] = None,
        rejected_run_ids: Optional[list[str]] = None,
        redact: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def label_run_outcome(
        self,
        *,
        run_id: Optional[str] = None,
        status: str,
        success: bool,
        failure_stage: Optional[str] = None,
        root_cause_tags: Optional[list[str]] = None,
        changed_files: Optional[list[str]] = None,
        tools_used: Optional[list[str]] = None,
        tests_attempted: Optional[list[str]] = None,
        tests_passed: Optional[list[str]] = None,
        tests_failed: Optional[list[str]] = None,
        user_approval_count: Optional[int] = None,
        user_deny_count: Optional[int] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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
        self._cache_breaks: list[dict[str, Any]] = []
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
            "cacheBreaks": [_json_safe(item) for item in self._cache_breaks],
        }

    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        payload = dict(state or {})
        self._session_id = str(payload.get("sessionId") or f"obs_{uuid4().hex}")
        self._agent_name = str(payload.get("agentName") or "") or None
        self._event_counter = int(payload.get("eventCounter") or 0)
        self._agent_runs = [dict(item) for item in list(payload.get("agentRuns") or []) if isinstance(item, dict)]
        self._llm_requests = [dict(item) for item in list(payload.get("llmRequests") or []) if isinstance(item, dict)]
        self._tool_executions = [dict(item) for item in list(payload.get("toolExecutions") or []) if isinstance(item, dict)]
        self._cache_breaks = [dict(item) for item in list(payload.get("cacheBreaks") or []) if isinstance(item, dict)]
        self._open_agent_runs = {}
        self._open_llm_requests = {}
        self._open_tool_executions = {}

    def clear(self) -> None:
        self._event_counter = 0
        self._agent_runs.clear()
        self._llm_requests.clear()
        self._tool_executions.clear()
        self._cache_breaks.clear()
        self._open_agent_runs.clear()
        self._open_llm_requests.clear()
        self._open_tool_executions.clear()

    def set_agent_name(self, agent_name: Optional[str]) -> None:
        self._agent_name = str(agent_name or "") or None

    def list_agent_runs(self) -> list[dict[str, Any]]:
        return [_json_safe(item) for item in self._agent_runs]

    def get_agent_run(self, run_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        run = self._resolve_agent_run(run_id)
        if run is None:
            return None
        return _json_safe(run)

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
            "cacheUsageSemantics": (metadata or {}).get("cacheUsageSemantics"),
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

    def record_cache_break(
        self,
        *,
        reason: str,
        changed_fields: Optional[list[str]] = None,
        previous_signature: Optional[dict[str, Any]] = None,
        current_signature: Optional[dict[str, Any]] = None,
        previous_cache_read_tokens: Optional[int] = None,
        current_cache_read_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = {
            "id": self._next_event_id("cache_break"),
            "sessionId": self._session_id,
            "agentName": self._agent_name,
            "reason": str(reason or "unknown"),
            "changedFields": list(changed_fields or []),
            "previousSignature": _json_safe(previous_signature),
            "currentSignature": _json_safe(current_signature),
            "previousCacheReadTokens": previous_cache_read_tokens,
            "currentCacheReadTokens": current_cache_read_tokens,
            "metadata": dict(metadata or {}),
            "createdAt": _now_iso(),
        }
        self._cache_breaks.append(payload)
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
        prompt_tokens_total = 0
        prompt_tokens_uncached = 0
        prompt_tokens_cached = 0
        cache_layer_breaks = Counter()
        for item in llm_requests:
            total_prompt, uncached_prompt, cached_prompt = _normalize_cache_accounting(item)
            prompt_tokens_total += total_prompt
            prompt_tokens_uncached += uncached_prompt
            prompt_tokens_cached += cached_prompt
        for item in self._cache_breaks:
            for field in list(item.get("changedFields") or []):
                cache_layer_breaks[_cache_layer_name(str(field))] += 1
        cache_hit_tokens = prompt_tokens_cached
        cache_hit_token_ratio = (
            float(prompt_tokens_cached) / float(prompt_tokens_total)
            if prompt_tokens_total > 0
            else None
        )
        request_prefix_signature = None
        if llm_requests:
            request_prefix_signature = (
                (llm_requests[-1].get("metadata") or {}).get("cacheSignature")
            )

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
            "promptTokensTotal": prompt_tokens_total,
            "promptTokensUncached": prompt_tokens_uncached,
            "promptTokensCached": prompt_tokens_cached,
            "cacheHitTokens": cache_hit_tokens,
            "cacheHitTokenRatio": cache_hit_token_ratio,
            "cacheHitTokenRatioNormalized": cache_hit_token_ratio,
            "cacheBreaks": len(self._cache_breaks),
            "cacheLayerBreaks": dict(cache_layer_breaks),
            "lastCacheBreak": self._cache_breaks[-1] if self._cache_breaks else None,
            "requestPrefixSignature": _json_safe(request_prefix_signature),
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
        if event_type in {None, "cache_break"}:
            events.extend({**item, "eventType": "cache_break"} for item in self._cache_breaks)
        events.sort(key=lambda item: str(item.get("endedAt") or item.get("startedAt") or item.get("createdAt") or ""), reverse=True)
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

    def export_run_record(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> Optional[dict[str, Any]]:
        run = self._resolve_agent_run(run_id)
        if run is None:
            return None
        record = self._build_run_record(run, trace_history)
        payload = record.to_dict()
        return self._apply_redaction(payload) if redact else payload

    def export_eval_trace(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> Optional[dict[str, Any]]:
        run = self._resolve_agent_run(run_id)
        if run is None:
            return None
        record = self._build_run_record(run, trace_history)
        outcome = record.outcome
        eval_trace = EvalTrace(
            run_id=record.run_id,
            query=record.query,
            success=record.success,
            status=record.status,
            final_output=record.final_output or record.output_preview,
            tool_calls=len(record.tool_executions),
            llm_requests=len(record.llm_requests),
            input_tokens=int(record.summary.get("inputTokens") or 0),
            output_tokens=int(record.summary.get("outputTokens") or 0),
            total_tokens=int(record.summary.get("totalTokens") or 0),
            duration_ms=record.duration_ms,
            prompt_tokens_cached=int(record.summary.get("promptTokensCached") or 0),
            prompt_tokens_uncached=int(record.summary.get("promptTokensUncached") or 0),
            cache_hit_ratio=record.summary.get("cacheHitTokenRatioNormalized"),
            tools_used=list(record.summary.get("toolsUsed") or []),
            changed_files=list((outcome.changed_files if outcome is not None else []) or []),
            cache_break_reasons=[
                str(item.get("reason") or "")
                for item in record.cache_breaks
                if str(item.get("reason") or "").strip()
            ],
            failure_stage=outcome.failure_stage if outcome is not None else None,
            root_cause_tags=list((outcome.root_cause_tags if outcome is not None else []) or []),
            metadata={
                "sessionId": record.session_id,
                "agentName": record.agent_name,
                "mode": record.mode,
                "stream": record.stream,
                "providerName": record.provider_name,
                "model": record.model,
            },
        )
        payload = eval_trace.to_dict()
        return self._apply_redaction(payload) if redact else payload

    def export_training_examples(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        run = self._resolve_agent_run(run_id)
        if run is None:
            return []
        record = self._build_run_record(run, trace_history)
        examples: list[TrainingExample] = []
        prior_messages: list[dict[str, Any]] = []
        tool_index = 0
        reasoning_index = 0
        for event in record.trace:
            event_type = str(event.get("type") or "")
            if event_type == "user_message":
                prior_messages.append({"role": "user", "content": str(event.get("content") or "")})
                continue
            if event_type in {"reasoning", "thinking"}:
                reasoning_content = str(event.get("content") or "")
                if reasoning_content:
                    reasoning_index += 1
                    examples.append(
                        TrainingExample(
                            example_id=f"{record.run_id}:planning:{reasoning_index}",
                            run_id=record.run_id,
                            example_type="planning",
                            input={
                                "query": record.query,
                                "turnId": record.turn_id,
                                "priorMessages": _json_safe(prior_messages[-6:]),
                            },
                            target={"content": reasoning_content},
                            metadata={
                                "mode": record.mode,
                                "stream": record.stream,
                                "providerName": record.provider_name,
                                "model": record.model,
                                "round": (event.get("metadata") or {}).get("round"),
                            },
                        )
                    )
                continue
            if event_type == "assistant_message":
                prior_messages.append({"role": "assistant", "content": str(event.get("content") or "")})
                continue
            if event_type != "tool_call":
                continue
            tool_index += 1
            examples.append(
                TrainingExample(
                    example_id=f"{record.run_id}:tool:{tool_index}",
                    run_id=record.run_id,
                    example_type="tool_selection",
                    input={
                        "query": record.query,
                        "turnId": record.turn_id,
                        "priorMessages": _json_safe(prior_messages[-6:]),
                    },
                    target={
                        "toolName": str(event.get("tool_name") or ""),
                        "toolArgs": _json_safe(event.get("tool_args") or {}),
                    },
                    metadata={
                        "mode": record.mode,
                        "stream": record.stream,
                        "providerName": record.provider_name,
                        "model": record.model,
                    },
                )
            )
        final_output = record.final_output or record.output_preview
        if final_output:
            examples.append(
                TrainingExample(
                    example_id=f"{record.run_id}:final:1",
                    run_id=record.run_id,
                    example_type="final_response",
                    input={
                        "query": record.query,
                        "turnId": record.turn_id,
                        "toolsUsed": list(record.summary.get("toolsUsed") or []),
                        "changedFiles": list(record.outcome.changed_files if record.outcome is not None else []),
                    },
                    target={"content": final_output},
                    metadata={
                        "success": record.success,
                        "status": record.status,
                        "providerName": record.provider_name,
                        "model": record.model,
                    },
                )
            )
        if record.outcome is not None:
            verification_target = {
                "status": record.outcome.status,
                "success": record.outcome.success,
                "failure_stage": record.outcome.failure_stage,
                "rootCauseTags": list(record.outcome.root_cause_tags),
                "changedFiles": list(record.outcome.changed_files),
                "testsAttempted": list(record.outcome.tests_attempted),
                "testsPassed": list(record.outcome.tests_passed),
                "testsFailed": list(record.outcome.tests_failed),
            }
            if any(
                verification_target[key]
                for key in (
                    "failure_stage",
                    "rootCauseTags",
                    "changedFiles",
                    "testsAttempted",
                    "testsPassed",
                    "testsFailed",
                )
            ):
                examples.append(
                    TrainingExample(
                        example_id=f"{record.run_id}:verification:1",
                        run_id=record.run_id,
                        example_type="verification_summary",
                        input={
                            "query": record.query,
                            "turnId": record.turn_id,
                            "toolsUsed": list(record.summary.get("toolsUsed") or []),
                            "finalOutput": final_output,
                        },
                        target=verification_target,
                        metadata={
                            "providerName": record.provider_name,
                            "model": record.model,
                        },
                    )
                )
        payload = [item.to_dict() for item in examples]
        return [self._apply_redaction(item) for item in payload] if redact else payload

    def export_run_record_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        payload = self.export_run_record(trace_history, run_id=run_id, redact=redact)
        return self._to_jsonl(payload)

    def export_eval_trace_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        payload = self.export_eval_trace(trace_history, run_id=run_id, redact=redact)
        return self._to_jsonl(payload)

    def export_training_examples_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        payload = self.export_training_examples(trace_history, run_id=run_id, redact=redact)
        return self._to_jsonl(payload)

    def export_sft_dataset(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        redact: bool = False,
        example_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        selected_types = {
            str(item).strip()
            for item in list(example_types or [])
            if str(item).strip()
        }
        dataset: list[dict[str, Any]] = []
        for run in self._resolve_agent_runs(run_id=run_id, run_ids=run_ids):
            examples = self.export_training_examples(
                trace_history,
                run_id=str(run.get("id") or ""),
                redact=False,
            )
            for example in examples:
                example_type = str(example.get("example_type") or "")
                if selected_types and example_type not in selected_types:
                    continue
                prompt = self._serialize_training_payload(example.get("input") or {})
                completion = self._serialize_training_payload(example.get("target") or {})
                record = SFTDatasetRecord(
                    record_id=f"sft:{example.get('example_id') or example.get('run_id')}",
                    run_id=str(example.get("run_id") or ""),
                    source_example_id=str(example.get("example_id") or ""),
                    example_type=example_type,
                    prompt=prompt,
                    completion=completion,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    input=_json_safe(example.get("input") or {}),
                    target=_json_safe(example.get("target") or {}),
                    metadata=_json_safe(example.get("metadata") or {}),
                )
                dataset.append(record.to_dict())
        return [self._apply_redaction(item) for item in dataset] if redact else dataset

    def export_sft_dataset_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        redact: bool = False,
        example_types: Optional[list[str]] = None,
    ) -> str:
        payload = self.export_sft_dataset(
            trace_history,
            run_id=run_id,
            run_ids=run_ids,
            redact=redact,
            example_types=example_types,
        )
        return self._to_jsonl(payload)

    def export_preference_pairs(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        chosen_run_ids: Optional[list[str]] = None,
        rejected_run_ids: Optional[list[str]] = None,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        records = [
            self._build_run_record(run, trace_history)
            for run in self._resolve_agent_runs(run_id=run_id, run_ids=run_ids)
        ]
        records = [record for record in records if record.final_output or record.output_preview]

        chosen_filter = {str(item) for item in list(chosen_run_ids or []) if str(item).strip()}
        rejected_filter = {str(item) for item in list(rejected_run_ids or []) if str(item).strip()}

        if chosen_filter or rejected_filter:
            chosen_records = [record for record in records if record.run_id in chosen_filter] if chosen_filter else records
            rejected_records = [record for record in records if record.run_id in rejected_filter] if rejected_filter else records
            pairs = self._build_explicit_preference_pairs(chosen_records, rejected_records)
        else:
            pairs = self._build_automatic_preference_pairs(records)

        payload = [item.to_dict() for item in pairs]
        return [self._apply_redaction(item) for item in payload] if redact else payload

    def export_preference_pairs_jsonl(
        self,
        trace_history: list[dict[str, Any]],
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        chosen_run_ids: Optional[list[str]] = None,
        rejected_run_ids: Optional[list[str]] = None,
        redact: bool = False,
    ) -> str:
        payload = self.export_preference_pairs(
            trace_history,
            run_id=run_id,
            run_ids=run_ids,
            chosen_run_ids=chosen_run_ids,
            rejected_run_ids=rejected_run_ids,
            redact=redact,
        )
        return self._to_jsonl(payload)

    def label_run_outcome(
        self,
        *,
        run_id: Optional[str] = None,
        status: str,
        success: bool,
        failure_stage: Optional[str] = None,
        root_cause_tags: Optional[list[str]] = None,
        changed_files: Optional[list[str]] = None,
        tools_used: Optional[list[str]] = None,
        tests_attempted: Optional[list[str]] = None,
        tests_passed: Optional[list[str]] = None,
        tests_failed: Optional[list[str]] = None,
        user_approval_count: Optional[int] = None,
        user_deny_count: Optional[int] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        run = self._resolve_agent_run(run_id)
        if run is None:
            raise ValueError("No completed agent run is available to label.")
        report = RunOutcomeReport(
            run_id=str(run.get("id") or ""),
            status=str(status or ("success" if success else "failed")),
            success=bool(success),
            failure_stage=str(failure_stage or "").strip() or None,
            root_cause_tags=[str(item) for item in list(root_cause_tags or []) if str(item).strip()],
            changed_files=[str(item) for item in list(changed_files or []) if str(item).strip()],
            tools_used=[str(item) for item in list(tools_used or []) if str(item).strip()],
            tests_attempted=[str(item) for item in list(tests_attempted or []) if str(item).strip()],
            tests_passed=[str(item) for item in list(tests_passed or []) if str(item).strip()],
            tests_failed=[str(item) for item in list(tests_failed or []) if str(item).strip()],
            user_approval_count=max(int(user_approval_count or 0), 0),
            user_deny_count=max(int(user_deny_count or 0), 0),
            notes=str(notes or "").strip() or None,
            metadata=dict(metadata or {}),
        )
        run_metadata = dict(run.get("metadata") or {})
        run_metadata["outcomeReport"] = report.to_dict()
        run["metadata"] = run_metadata
        return _json_safe(report.to_dict())

    def _resolve_agent_run(self, run_id: Optional[str]) -> Optional[dict[str, Any]]:
        if run_id:
            for item in self._agent_runs:
                if str(item.get("id") or "") == str(run_id):
                    return item
            return None
        if not self._agent_runs:
            return None
        return self._agent_runs[-1]

    def _resolve_agent_runs(
        self,
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if run_ids:
            selected = {str(item) for item in run_ids if str(item).strip()}
            return [item for item in self._agent_runs if str(item.get("id") or "") in selected]
        if run_id:
            run = self._resolve_agent_run(run_id)
            return [run] if run is not None else []
        return list(self._agent_runs)

    def _build_run_record(
        self,
        run: dict[str, Any],
        trace_history: list[dict[str, Any]],
    ) -> AgentRunRecord:
        turn_id = str(run.get("turnId") or "").strip() or None
        trace = self._filter_trace_for_turn(trace_history, turn_id=turn_id)
        llm_requests = self._filter_events_by_turn(self._llm_requests, turn_id=turn_id)
        tool_executions = self._filter_events_by_turn(self._tool_executions, turn_id=turn_id)
        cache_breaks = self._filter_cache_breaks_for_run(llm_requests)
        provider_name = next((str(item.get("providerName") or "") for item in llm_requests if item.get("providerName")), None) or None
        model = next((str(item.get("model") or "") for item in llm_requests if item.get("model")), None) or None
        final_output = self._extract_final_output(trace, fallback=str(run.get("outputPreview") or ""))
        tools_used = sorted({str(item.get("toolName") or "") for item in tool_executions if item.get("toolName")})
        outcome_payload = dict((run.get("metadata") or {}).get("outcomeReport") or {})
        outcome = RunOutcomeReport(
            run_id=str(outcome_payload.get("run_id") or run.get("id") or ""),
            status=str(outcome_payload.get("status") or ("success" if run.get("success") else "failed")),
            success=bool(outcome_payload.get("success") if "success" in outcome_payload else run.get("success")),
            failure_stage=str(outcome_payload.get("failure_stage") or "").strip() or None,
            root_cause_tags=[str(item) for item in list(outcome_payload.get("root_cause_tags") or []) if str(item).strip()],
            changed_files=[str(item) for item in list(outcome_payload.get("changed_files") or []) if str(item).strip()],
            tools_used=[str(item) for item in list(outcome_payload.get("tools_used") or tools_used) if str(item).strip()],
            tests_attempted=[str(item) for item in list(outcome_payload.get("tests_attempted") or []) if str(item).strip()],
            tests_passed=[str(item) for item in list(outcome_payload.get("tests_passed") or []) if str(item).strip()],
            tests_failed=[str(item) for item in list(outcome_payload.get("tests_failed") or []) if str(item).strip()],
            user_approval_count=int(outcome_payload.get("user_approval_count") or 0),
            user_deny_count=int(outcome_payload.get("user_deny_count") or 0),
            notes=str(outcome_payload.get("notes") or "").strip() or None,
            metadata=dict(outcome_payload.get("metadata") or {}),
        ) if outcome_payload else None
        summary = {
            "llmRequests": len(llm_requests),
            "toolCalls": len(tool_executions),
            "inputTokens": sum(int(item.get("inputTokens") or 0) for item in llm_requests),
            "outputTokens": sum(int(item.get("outputTokens") or 0) for item in llm_requests),
            "totalTokens": sum(int(item.get("totalTokens") or 0) for item in llm_requests),
            "cachedInputTokens": sum(int(item.get("cachedInputTokens") or 0) for item in llm_requests),
            "cacheReadTokens": sum(int(item.get("cacheReadTokens") or 0) for item in llm_requests),
            "reasoningTokens": sum(int(item.get("reasoningTokens") or 0) for item in llm_requests),
            "toolUsePromptTokens": sum(int(item.get("toolUsePromptTokens") or 0) for item in llm_requests),
            "toolsUsed": tools_used,
            "cacheBreaks": len(cache_breaks),
        }
        prompt_tokens_total = 0
        prompt_tokens_uncached = 0
        prompt_tokens_cached = 0
        cache_layer_breaks = Counter()
        for item in llm_requests:
            total_prompt, uncached_prompt, cached_prompt = _normalize_cache_accounting(item)
            prompt_tokens_total += total_prompt
            prompt_tokens_uncached += uncached_prompt
            prompt_tokens_cached += cached_prompt
        for item in cache_breaks:
            for field in list(item.get("changedFields") or []):
                cache_layer_breaks[_cache_layer_name(str(field))] += 1
        summary.update(
            {
                "promptTokensTotal": prompt_tokens_total,
                "promptTokensUncached": prompt_tokens_uncached,
                "promptTokensCached": prompt_tokens_cached,
                "cacheHitTokenRatioNormalized": (
                    float(prompt_tokens_cached) / float(prompt_tokens_total)
                    if prompt_tokens_total > 0
                    else None
                ),
                "cacheLayerBreaks": dict(cache_layer_breaks),
                "cacheBreakReasons": [
                    str(item.get("reason") or "")
                    for item in cache_breaks
                    if str(item.get("reason") or "").strip()
                ],
            }
        )
        return AgentRunRecord(
            run_id=str(run.get("id") or ""),
            session_id=str(run.get("sessionId") or "") or None,
            agent_name=str(run.get("agentName") or "") or None,
            turn_id=turn_id,
            query=str(run.get("query") or ""),
            mode=str(run.get("mode") or "unknown"),
            stream=bool(run.get("stream")),
            success=bool(run.get("success")),
            status=str(run.get("status") or ("success" if run.get("success") else "error")),
            started_at=str(run.get("startedAt") or "") or None,
            ended_at=str(run.get("endedAt") or "") or None,
            duration_ms=float(run.get("durationMs") or 0.0) if run.get("durationMs") is not None else None,
            provider_name=provider_name,
            model=model,
            output_preview=str(run.get("outputPreview") or ""),
            final_output=final_output,
            trace=[_json_safe(item) for item in trace],
            llm_requests=[_json_safe(item) for item in llm_requests],
            tool_executions=[_json_safe(item) for item in tool_executions],
            cache_breaks=[_json_safe(item) for item in cache_breaks],
            summary=_json_safe(summary),
            outcome=outcome,
            metadata=_json_safe(run.get("metadata") or {}),
        )

    @staticmethod
    def _filter_trace_for_turn(
        trace_history: list[dict[str, Any]],
        *,
        turn_id: Optional[str],
    ) -> list[dict[str, Any]]:
        if not turn_id:
            return []
        return [dict(item) for item in list(trace_history or []) if str(item.get("turn_id") or item.get("turnId") or "").strip() == turn_id]

    @staticmethod
    def _filter_events_by_turn(
        events: list[dict[str, Any]],
        *,
        turn_id: Optional[str],
    ) -> list[dict[str, Any]]:
        if not turn_id:
            return []
        return [dict(item) for item in events if str(item.get("turnId") or "").strip() == turn_id]

    def _filter_cache_breaks_for_run(self, llm_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        turn_ids = {
            str(item.get("turnId") or "").strip()
            for item in llm_requests
            if str(item.get("turnId") or "").strip()
        }
        signatures = {
            _signature_key((item.get("metadata") or {}).get("cacheSignature"))
            for item in llm_requests
            if (item.get("metadata") or {}).get("cacheSignature") is not None
        }
        matched: list[dict[str, Any]] = []
        for item in self._cache_breaks:
            metadata = dict(item.get("metadata") or {})
            item_turn_id = str(metadata.get("turnId") or "").strip()
            previous_signature = _signature_key(item.get("previousSignature"))
            current_signature = _signature_key(item.get("currentSignature"))
            if item_turn_id and item_turn_id in turn_ids:
                matched.append(dict(item))
                continue
            if previous_signature and previous_signature in signatures:
                matched.append(dict(item))
                continue
            if current_signature and current_signature in signatures:
                matched.append(dict(item))
                continue
        return matched

    @staticmethod
    def _extract_final_output(trace: list[dict[str, Any]], *, fallback: str) -> str:
        for event in reversed(trace):
            if str(event.get("type") or "") == "assistant_message":
                content = str(event.get("content") or "")
                if content:
                    return content
        return fallback

    @staticmethod
    def _apply_redaction(payload: Any) -> Any:
        if payload is None or isinstance(payload, (int, float, bool)):
            return payload
        if isinstance(payload, str):
            return "[redacted]" if payload else payload
        if isinstance(payload, list):
            return [InMemoryObservabilityRecorder._apply_redaction(item) for item in payload]
        if isinstance(payload, dict):
            redacted: dict[str, Any] = {}
            for key, value in payload.items():
                if key in {
                    "query",
                    "content",
                    "final_output",
                    "output_preview",
                    "toolArgs",
                    "tool_args",
                    "priorMessages",
                    "target",
                    "input",
                    "prompt",
                    "completion",
                    "chosen",
                    "rejected",
                    "messages",
                }:
                    redacted[key] = "[redacted]" if value not in (None, "", [], {}) else value
                else:
                    redacted[key] = InMemoryObservabilityRecorder._apply_redaction(value)
            return redacted
        return str(payload)

    @staticmethod
    def _serialize_training_payload(payload: dict[str, Any]) -> str:
        return _stdlib_json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)

    def _build_explicit_preference_pairs(
        self,
        chosen_records: list[AgentRunRecord],
        rejected_records: list[AgentRunRecord],
    ) -> list[PreferencePairRecord]:
        pairs: list[PreferencePairRecord] = []
        if not chosen_records or not rejected_records:
            return pairs
        grouped_rejected: dict[str, list[AgentRunRecord]] = defaultdict(list)
        for record in rejected_records:
            grouped_rejected[self._preference_group_key(record)].append(record)
        for chosen in chosen_records:
            group_key = self._preference_group_key(chosen)
            candidates = grouped_rejected.get(group_key) or rejected_records
            for rejected in candidates:
                if rejected.run_id == chosen.run_id:
                    continue
                pair = self._build_preference_pair(chosen, rejected, group_key=group_key)
                if pair is not None:
                    pairs.append(pair)
                    break
        return pairs

    def _build_automatic_preference_pairs(
        self,
        records: list[AgentRunRecord],
    ) -> list[PreferencePairRecord]:
        grouped: dict[str, list[AgentRunRecord]] = defaultdict(list)
        for record in records:
            grouped[self._preference_group_key(record)].append(record)
        pairs: list[PreferencePairRecord] = []
        for group_key, group in grouped.items():
            if len(group) < 2:
                continue
            scored = sorted(
                ((self._preference_score(record), record) for record in group),
                key=lambda item: item[0],
            )
            rejected_score, rejected = scored[0]
            chosen_score, chosen = scored[-1]
            if chosen.run_id == rejected.run_id:
                continue
            if chosen_score <= rejected_score:
                continue
            pair = self._build_preference_pair(chosen, rejected, group_key=group_key)
            if pair is not None:
                pairs.append(pair)
        return pairs

    @staticmethod
    def _preference_group_key(record: AgentRunRecord) -> str:
        outcome_metadata = dict(record.outcome.metadata if record.outcome is not None else {})
        record_metadata = dict(record.metadata or {})
        for source in (outcome_metadata, record_metadata):
            value = source.get("preferenceGroup")
            if value is not None and str(value).strip():
                return str(value).strip()
        return record.query

    @staticmethod
    def _preference_score(record: AgentRunRecord) -> float:
        outcome = record.outcome
        metadata_sources = [
            dict(record.metadata or {}),
            dict(outcome.metadata if outcome is not None else {}),
        ]
        for source in metadata_sources:
            if "preferenceScore" in source:
                try:
                    return float(source["preferenceScore"])
                except Exception:
                    pass

        score = 0.0
        if record.success:
            score += 1.0
        if outcome is not None:
            if outcome.success:
                score += 1.0
            status = str(outcome.status or "")
            if status == "success":
                score += 1.0
            elif status == "partial_success":
                score += 0.5
            score += 0.1 * len(outcome.tests_passed)
            score -= 0.1 * len(outcome.tests_failed)
            score += 0.05 * int(outcome.user_approval_count or 0)
            score -= 0.05 * int(outcome.user_deny_count or 0)
            score -= 0.05 * len(outcome.root_cause_tags)
        return score

    def _build_preference_pair(
        self,
        chosen: AgentRunRecord,
        rejected: AgentRunRecord,
        *,
        group_key: str,
    ) -> Optional[PreferencePairRecord]:
        chosen_text = chosen.final_output or chosen.output_preview
        rejected_text = rejected.final_output or rejected.output_preview
        if not chosen_text or not rejected_text:
            return None
        if chosen_text == rejected_text:
            return None
        prompt = chosen.query or rejected.query
        chosen_score = self._preference_score(chosen)
        rejected_score = self._preference_score(rejected)
        return PreferencePairRecord(
            pair_id=f"pref:{chosen.run_id}:{rejected.run_id}",
            prompt=prompt,
            chosen=chosen_text,
            rejected=rejected_text,
            chosen_run_id=chosen.run_id,
            rejected_run_id=rejected.run_id,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            messages=[{"role": "user", "content": prompt}],
            metadata={
                "groupKey": group_key,
                "chosen": {
                    "status": chosen.status,
                    "success": chosen.success,
                    "summary": _json_safe(chosen.summary),
                    "outcome": _json_safe(chosen.outcome.to_dict() if chosen.outcome is not None else None),
                },
                "rejected": {
                    "status": rejected.status,
                    "success": rejected.success,
                    "summary": _json_safe(rejected.summary),
                    "outcome": _json_safe(rejected.outcome.to_dict() if rejected.outcome is not None else None),
                },
            },
        )

    @staticmethod
    def _to_jsonl(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, list):
            lines = [
                _stdlib_json.dumps(_json_safe(item), ensure_ascii=False, sort_keys=True)
                for item in payload
            ]
            return "\n".join(lines)
        return _stdlib_json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{prefix}_{self._event_counter:06d}"


__all__ = ["BaseObservabilityRecorder", "InMemoryObservabilityRecorder"]
