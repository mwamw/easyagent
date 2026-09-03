"""Agent-bound observability manager for AgentInvoke and LLMInvoke records."""

from __future__ import annotations

from contextvars import ContextVar
from abc import ABC, abstractmethod
from datetime import datetime
import threading
import time
from typing import Any
from uuid import uuid4

from core.history import CanonicalBlock, CanonicalMessage
from runtime.events import RuntimeEvent, RuntimeEventBus, RuntimeEventType

from .models import AgentInvoke, LLMInvoke, utc_now
from .store import BaseObservabilityStore, InMemoryObservabilityStore


def _duration_ms(started: datetime, ended: datetime) -> float:
    return max(0.0, (ended - started).total_seconds() * 1000.0)


def _canonical_messages(messages: list[Any] | None) -> list[CanonicalMessage]:
    normalized: list[CanonicalMessage] = []
    for message in list(messages or []):
        if isinstance(message, CanonicalMessage):
            normalized.append(message.model_copy(deep=True))
        else:
            normalized.append(CanonicalMessage.model_validate(message))
    return normalized


def _same_message(left: CanonicalMessage, right: CanonicalMessage) -> bool:
    if left.role != right.role:
        return False
    left_visible = [block for block in left.content if block.type != "reasoning"]
    right_visible = [block for block in right.content if block.type != "reasoning"]
    return left_visible == right_visible


class BaseObservabilityManager(ABC):
    @abstractmethod
    def bind(self, *, agent_id: str, event_bus: RuntimeEventBus) -> "BaseObservabilityManager":
        raise NotImplementedError

    @abstractmethod
    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[AgentInvoke]:
        """Return completed AgentInvoke records in storage order."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ObservabilityManager(BaseObservabilityManager):
    def __init__(
        self,
        store: BaseObservabilityStore | None = None,
        *,
        parent_invoke_id: str | None = None,
        owns_store: bool = True,
    ):
        self.store = store or InMemoryObservabilityStore()
        self.parent_invoke_id = parent_invoke_id
        self._owns_store = owns_store
        self._agent_id_value: str | None = None
        self._event_bus: RuntimeEventBus | None = None
        self._subscription_token: str | None = None
        self._lock = threading.RLock()
        self._open_agent_invokes: dict[str, AgentInvoke] = {}
        self._open_llm_invokes: dict[str, tuple[str, LLMInvoke]] = {}
        self._open_tool_invokes: dict[str, tuple[str, float]] = {}
        self._active_parent: dict[str, str | None] = {}
        self._runtime_agent_invokes: dict[str, str] = {}
        self._runtime_llm_invokes: dict[tuple[str, str], str] = {}
        self._runtime_tool_invokes: dict[tuple[str, str], str] = {}
        self._active_invoke: ContextVar[str | None] = ContextVar(
            f"easyagent_observability_{uuid4().hex}",
            default=None,
        )

    @property
    def active_invoke_id(self) -> str | None:
        return self._active_invoke.get()

    def bind(self, *, agent_id: str, event_bus: RuntimeEventBus) -> "ObservabilityManager":
        if not isinstance(event_bus, RuntimeEventBus):
            raise TypeError("event_bus must be RuntimeEventBus")
        if self._event_bus is not None and self._event_bus is not event_bus:
            raise RuntimeError("ObservabilityManager 已绑定到另一个 RuntimeEventBus。")
        self._agent_id_value = str(agent_id)
        self._event_bus = event_bus
        if self._subscription_token is None:
            self._subscription_token = event_bus.subscribe(self.handle_runtime_event)
        return self

    def create_child(self, *, parent_invoke_id: str | None = None) -> "ObservabilityManager":
        return ObservabilityManager(
            self.store,
            parent_invoke_id=parent_invoke_id or self.active_invoke_id,
            owns_store=False,
        )

    def _agent_id(self) -> str:
        return self._agent_id_value or "agent"

    @staticmethod
    def _query_message(query: str) -> list[CanonicalMessage]:
        return [
            CanonicalMessage(
                role="user",
                content=[CanonicalBlock(type="text", text=str(query or ""))],
            )
        ]

    def begin_agent_invoke(
        self,
        *,
        query: str,
        mode: str,
        stream: bool,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        invoke_id = f"agent_invoke_{uuid4().hex}"
        previous = self.active_invoke_id
        parent_invoke_id = previous or self.parent_invoke_id
        record = AgentInvoke(
            invoke_id=invoke_id,
            parent_invoke_id=parent_invoke_id,
            agent_id=self._agent_id(),
            query=str(query or ""),
            trace=self._query_message(str(query or "")),
            metadata={
                "agent_name": self._agent_id(),
                "mode": str(mode or "unknown"),
                "stream": bool(stream),
                **dict(metadata or {}),
            },
        )
        with self._lock:
            self._open_agent_invokes[invoke_id] = record
            self._active_parent[invoke_id] = previous
        self._active_invoke.set(invoke_id)
        return invoke_id

    def end_agent_invoke(
        self,
        invoke_id: str,
        *,
        output: list[CanonicalMessage] | None,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInvoke | None:
        with self._lock:
            record = self._open_agent_invokes.pop(str(invoke_id), None)
            previous = self._active_parent.pop(str(invoke_id), None)
        if record is None:
            return None

        record.output = _canonical_messages(output)
        if record.output:
            if not record.trace or not _same_message(record.trace[-1], record.output[-1]):
                record.trace.extend(item.model_copy(deep=True) for item in record.output)

        ended_at = utc_now()
        llm_stats = [item.stats for item in record.llm_invokes]
        costs = [item.estimated_cost_usd for item in llm_stats if item.estimated_cost_usd is not None]
        record.stats.success = bool(success)
        record.stats.status = "success" if success else "error"
        record.stats.ended_at = ended_at
        record.stats.duration_ms = _duration_ms(record.stats.started_at, ended_at)
        record.stats.llm_calls = len(record.llm_invokes)
        record.stats.llm_errors = sum(1 for item in llm_stats if not item.success)
        record.stats.input_tokens = sum(item.input_tokens for item in llm_stats)
        record.stats.output_tokens = sum(item.output_tokens for item in llm_stats)
        record.stats.total_tokens = sum(item.total_tokens for item in llm_stats)
        record.stats.cached_input_tokens = sum(item.cached_input_tokens for item in llm_stats)
        record.stats.reasoning_tokens = sum(item.reasoning_tokens for item in llm_stats)
        record.stats.estimated_cost_usd = sum(costs) if costs else None
        record.stats.error_type = error_type
        record.stats.error_message = error_message
        record.metadata.update(dict(metadata or {}))
        try:
            self.store.save(record)
        finally:
            self._active_invoke.set(previous)
        return record.model_copy(deep=True)

    def begin_llm_invoke(
        self,
        *,
        input_messages: list[CanonicalMessage],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any] | None,
        estimated_input_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        agent_invoke_id = self.active_invoke_id
        if agent_invoke_id is None:
            return None
        invoke_id = f"llm_invoke_{uuid4().hex}"
        record = LLMInvoke(
            invoke_id=invoke_id,
            input=_canonical_messages(input_messages),
            tools=[dict(item) for item in list(tools or [])],
            options=dict(options or {}),
            metadata=dict(metadata or {}),
        )
        record.stats.input_tokens = int(estimated_input_tokens or 0)
        with self._lock:
            agent_record = self._open_agent_invokes.get(agent_invoke_id)
            if agent_record is None:
                return None
            agent_record.llm_invokes.append(record)
            self._open_llm_invokes[invoke_id] = (agent_invoke_id, record)
        return invoke_id

    def end_llm_invoke(
        self,
        invoke_id: str,
        *,
        output: list[CanonicalMessage] | None,
        usage: dict[str, Any] | None,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMInvoke | None:
        with self._lock:
            opened = self._open_llm_invokes.pop(str(invoke_id), None)
        if opened is None:
            return None
        agent_invoke_id, record = opened
        record.output = _canonical_messages(output)
        ended_at = utc_now()
        stats = record.stats
        stats.success = bool(success)
        stats.status = "success" if success else "error"
        stats.ended_at = ended_at
        stats.duration_ms = _duration_ms(stats.started_at, ended_at)
        values = dict(usage or {})
        if values.get("inputTokens") is not None:
            stats.input_tokens = int(values["inputTokens"])
        stats.output_tokens = int(values.get("outputTokens") or 0)
        stats.total_tokens = int(
            values.get("totalTokens")
            if values.get("totalTokens") is not None
            else stats.input_tokens + stats.output_tokens
        )
        stats.cached_input_tokens = int(values.get("cachedInputTokens") or 0)
        stats.reasoning_tokens = int(values.get("reasoningTokens") or 0)
        stats.cache_read_tokens = int(values.get("cacheReadTokens") or 0)
        stats.cache_creation_tokens = int(values.get("cacheCreationTokens") or 0)
        stats.tool_use_prompt_tokens = int(values.get("toolUsePromptTokens") or 0)
        cost = values.get("costUsd")
        stats.estimated_cost_usd = float(cost) if cost is not None else None
        stats.error_type = error_type
        stats.error_message = error_message
        record.metadata.update(dict(metadata or {}))

        if record.output:
            with self._lock:
                agent_record = self._open_agent_invokes.get(agent_invoke_id)
                if agent_record is not None:
                    agent_record.trace.extend(
                        item.model_copy(deep=True) for item in record.output
                    )
        return record.model_copy(deep=True)

    def record_tool_start(self) -> str | None:
        agent_invoke_id = self.active_invoke_id
        if agent_invoke_id is None:
            return None
        tool_invoke_id = f"tool_invoke_{uuid4().hex}"
        with self._lock:
            record = self._open_agent_invokes.get(agent_invoke_id)
            if record is None:
                return None
            record.stats.tool_calls += 1
            self._open_tool_invokes[tool_invoke_id] = (
                agent_invoke_id,
                time.perf_counter(),
            )
        return tool_invoke_id

    def record_tool_end(self, tool_invoke_id: str, *, success: bool) -> None:
        with self._lock:
            opened = self._open_tool_invokes.pop(str(tool_invoke_id), None)
            if opened is None:
                return
            agent_invoke_id, started = opened
            record = self._open_agent_invokes.get(agent_invoke_id)
            if record is None:
                return
            record.stats.tool_duration_ms += max(
                0.0,
                (time.perf_counter() - started) * 1000.0,
            )
            if not success:
                record.stats.tool_errors += 1

    def append_trace(self, messages: list[CanonicalMessage] | None) -> None:
        invoke_id = self.active_invoke_id
        if invoke_id is None:
            return
        normalized = _canonical_messages(messages)
        if not normalized:
            return
        with self._lock:
            record = self._open_agent_invokes.get(invoke_id)
            if record is not None:
                record.trace.extend(normalized)

    def record_cache_break(self, **payload: Any) -> dict[str, Any]:
        record = {
            "reason": str(payload.get("reason") or "unknown"),
            "changed_fields": list(payload.get("changed_fields") or []),
            "previous_signature": payload.get("previous_signature"),
            "current_signature": payload.get("current_signature"),
            "previous_cache_read_tokens": payload.get("previous_cache_read_tokens"),
            "current_cache_read_tokens": payload.get("current_cache_read_tokens"),
            "metadata": dict(payload.get("metadata") or {}),
            "created_at": utc_now().isoformat(),
        }
        invoke_id = self.active_invoke_id
        if invoke_id is not None:
            with self._lock:
                invoke = self._open_agent_invokes.get(invoke_id)
                if invoke is not None:
                    invoke.metadata.setdefault("cache_breaks", []).append(record)
        return record

    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        """Consume the framework event model without requiring Agent internals."""

        data = event.data
        runtime_id = event.invocation_id
        if event.type == RuntimeEventType.HISTORY_COMPACTED:
            self.record_cache_break(
                reason="history_compacted",
                changed_fields=["history.compacted"],
                metadata={
                    "tokensBefore": data.get("tokens_before"),
                    "tokensAfter": data.get("tokens_after"),
                    **dict(data.get("metadata") or {}),
                },
            )
            return
        if event.type == RuntimeEventType.AGENT_INVOKE_STARTED:
            observe_id = self.begin_agent_invoke(
                query=str(data.get("query") or ""),
                mode=str(data.get("mode") or "execute"),
                stream=bool(data.get("stream")),
                metadata=dict(data.get("metadata") or {}),
            )
            self._runtime_agent_invokes[runtime_id] = observe_id
            return

        if event.type == RuntimeEventType.LLM_INVOKE_STARTED:
            round_id = str(data.get("llm_invoke_id") or event.sequence)
            observe_id = self.begin_llm_invoke(
                input_messages=_canonical_messages(data.get("input")),
                tools=list(data.get("tools") or []),
                options=dict(data.get("options") or {}),
                estimated_input_tokens=data.get("estimated_input_tokens"),
                metadata=dict(data.get("metadata") or {}),
            )
            if observe_id is not None:
                self._runtime_llm_invokes[(runtime_id, round_id)] = observe_id
            return

        if event.type in {RuntimeEventType.LLM_INVOKE_COMPLETED, RuntimeEventType.LLM_INVOKE_FAILED}:
            round_id = str(data.get("llm_invoke_id") or "")
            observe_id = self._runtime_llm_invokes.pop((runtime_id, round_id), None)
            if observe_id is not None:
                self.end_llm_invoke(
                    observe_id,
                    output=_canonical_messages(data.get("output")),
                    usage=dict(data.get("usage") or {}),
                    success=event.type == RuntimeEventType.LLM_INVOKE_COMPLETED,
                    error_type=data.get("error_type"),
                    error_message=str(data.get("error_message") or "") or None,
                    metadata=dict(data.get("metadata") or {}),
                )
            return

        if event.type == RuntimeEventType.TOOL_INVOKE_STARTED:
            tool_call_id = str(data.get("tool_call_id") or event.sequence)
            observe_id = self.record_tool_start()
            if observe_id is not None:
                self._runtime_tool_invokes[(runtime_id, tool_call_id)] = observe_id
            return

        if event.type in {RuntimeEventType.TOOL_INVOKE_COMPLETED, RuntimeEventType.TOOL_INVOKE_FAILED}:
            tool_call_id = str(data.get("tool_call_id") or "")
            observe_id = self._runtime_tool_invokes.pop((runtime_id, tool_call_id), None)
            if observe_id is not None:
                self.record_tool_end(
                    observe_id,
                    success=event.type == RuntimeEventType.TOOL_INVOKE_COMPLETED,
                )
            self.append_trace(_canonical_messages(data.get("trace")))
            return

        if event.type not in {
            RuntimeEventType.AGENT_INVOKE_COMPLETED,
            RuntimeEventType.AGENT_INVOKE_FAILED,
            RuntimeEventType.AGENT_INVOKE_INTERRUPTED,
        }:
            return
        observe_id = self._runtime_agent_invokes.pop(runtime_id, None)
        if observe_id is None:
            return
        success = event.type == RuntimeEventType.AGENT_INVOKE_COMPLETED
        self.end_agent_invoke(
            observe_id,
            output=_canonical_messages(data.get("output_messages")),
            success=success,
            error_type=data.get("error_type"),
            error_message=str(data.get("error_message") or "") or None,
            metadata=dict(data.get("metadata") or {}),
        )

    def get(self, invoke_id: str) -> AgentInvoke | None:
        return self.store.get(invoke_id)

    def list(self) -> list[AgentInvoke]:
        return self.store.list()

    def latest(self) -> AgentInvoke | None:
        records = self.list()
        return records[-1] if records else None

    def annotate(
        self,
        metadata: dict[str, Any],
        *,
        invoke_id: str | None = None,
    ) -> AgentInvoke:
        target_id = invoke_id or self.active_invoke_id
        if target_id is None:
            latest = self.latest()
            if latest is None:
                raise KeyError("当前没有可标注的 AgentInvoke。")
            target_id = latest.invoke_id

        with self._lock:
            opened = self._open_agent_invokes.get(target_id)
            if opened is not None:
                opened.metadata.update(dict(metadata))
                return opened.model_copy(deep=True)

        finalized = self.store.get(target_id)
        if finalized is None:
            raise KeyError(f"AgentInvoke 不存在: {target_id}")
        finalized.metadata.update(dict(metadata))
        self.store.save(finalized)
        return finalized.model_copy(deep=True)

    def summary(self) -> dict[str, Any]:
        records = self.list()
        return {
            "agent_invokes": len(records),
            "successful_agent_invokes": sum(1 for item in records if item.stats.success),
            "failed_agent_invokes": sum(1 for item in records if not item.stats.success),
            "llm_invokes": sum(item.stats.llm_calls for item in records),
            "tool_calls": sum(item.stats.tool_calls for item in records),
            "input_tokens": sum(item.stats.input_tokens for item in records),
            "output_tokens": sum(item.stats.output_tokens for item in records),
            "total_tokens": sum(item.stats.total_tokens for item in records),
            "cached_input_tokens": sum(item.stats.cached_input_tokens for item in records),
            "duration_ms": sum(item.stats.duration_ms for item in records),
        }

    def export_state(self) -> dict[str, Any]:
        return {"records": [item.to_dict() for item in self.list()]}

    def restore_state(self, state: dict[str, Any] | None) -> None:
        for payload in list((state or {}).get("records") or []):
            self.store.save(AgentInvoke.model_validate(payload))

    def clear(self) -> None:
        self.store.clear()

    def close(self) -> None:
        if self._event_bus is not None and self._subscription_token is not None:
            self._event_bus.unsubscribe(self._subscription_token)
        self._subscription_token = None
        if self._owns_store:
            self.store.close()


__all__ = ["BaseObservabilityManager", "ObservabilityManager"]
