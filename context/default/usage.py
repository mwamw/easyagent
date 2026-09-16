"""Per-Agent provider-usage anchors for context sizing, independent of tracing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import RLock
from typing import Any

from core.history import _json_safe


def _tokens(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if result >= 0 else None


@dataclass(frozen=True)
class ContextTokenEstimate:
    tokens: int
    local_tokens: int
    source: str
    metadata: dict[str, Any]


@dataclass
class PreparedUsageRequest:
    request: Any
    tools: Any
    reasoning: Any
    signature: str
    history_epoch: int
    local_tokens: int


@dataclass(frozen=True)
class _UsageAnchor:
    signature: str
    history_epoch: int
    provider_tokens: int
    local_tokens: int
    kind: str
    usage: dict[str, Any]


class ContextUsageTracker:
    """Use an API-measured base plus the local delta, never max(API, local).

    The base belongs to one Agent and one history generation. Only the executor's
    successful main-model responses install it; summarizers and child Agents do
    not overwrite it. It is deliberately not restored from a session snapshot.
    """

    def __init__(self):
        self._anchor: _UsageAnchor | None = None
        self._counter = None
        self._lock = RLock()
        self.last_query = ""
        self.last_options: dict[str, Any] = {}
        self.revision = 0

    @property
    def counter(self):
        with self._lock:
            if self._counter is None:
                from .token.counter import TokenCounter
                self._counter = TokenCounter()
            return self._counter

    @staticmethod
    def _input_options(options):
        # Output/sampling knobs do not change the already measured input prefix.
        ignored = {"temperature", "max_tokens", "max_output_tokens", "max_completion_tokens", "stream", "reasoning"}
        return {key: value for key, value in (options or {}).items() if key not in ignored}

    def _signature(self, manager, request, llm, tools, reasoning, options):
        counter = manager.counter
        payload = {
            "provider": request.provider_name, "model": getattr(llm, "model", None),
            "llm_instance": id(llm), "endpoint": getattr(llm, "base_url", None),
            "system": request.render_system_prompt(), "tools": tools, "reasoning": reasoning,
            "stable_reminders": [(block.name, block.render()) for block in request.system_reminder_blocks
                                 if block.partition != "dynamic"],
            "options": self._input_options(options), "cache_policy": request.cache_policy.to_dict(),
            "counter": [id(counter), getattr(counter, "chars_per_token", None),
                        getattr(getattr(counter, "_encoder", None), "name", None)],
        }
        return sha256(json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def estimate(self, manager, request, *, llm, history_epoch, tools=None, reasoning=None, options=None):
        local = manager.count_request(request, tools=tools, reasoning=reasoning)
        signature = self._signature(manager, request, llm, tools, reasoning, options)
        with self._lock:
            anchor = self._anchor
        reason = "no_provider_usage"
        if anchor is not None:
            if anchor.history_epoch != history_epoch:
                reason = "history_replaced"
            elif anchor.signature != signature:
                reason = "request_context_changed"
            else:
                delta = local - anchor.local_tokens
                return ContextTokenEstimate(max(0, anchor.provider_tokens + delta), local,
                    "provider_usage_plus_delta_estimate", {
                        "provider_usage_tokens": anchor.provider_tokens,
                        "delta_tokens": delta, "anchor_kind": anchor.kind,
                        "usage": deepcopy(anchor.usage),
                    })
        return ContextTokenEstimate(local, local, "local_request_estimate", {"anchor_invalidated": reason})

    def prepare(self, manager, request, *, llm, history_epoch, tools=None, reasoning=None, options=None, query=""):
        with self._lock:
            self.last_query = query
            self.last_options = deepcopy(options or {})
        return PreparedUsageRequest(deepcopy(request), deepcopy(tools), deepcopy(reasoning),
            self._signature(manager, request, llm, tools, reasoning, options), history_epoch,
            manager.count_request(request, tools=tools, reasoning=reasoning))

    def observe(self, manager, prepared, usage, *, response_replay=(), use_output=True):
        if prepared is None or usage.get("usageSource") != "provider":
            return
        for key in ("inputTokens", "outputTokens", "totalTokens", "contextInputTokens", "contextTotalTokens"):
            if usage.get(key) is not None and _tokens(usage[key]) is None:
                return
        input_tokens = _tokens(usage.get("contextInputTokens"))
        if input_tokens is None:
            input_tokens = _tokens(usage.get("inputTokens"))
        output_tokens = _tokens(usage.get("outputTokens"))
        total_tokens = _tokens(usage.get("contextTotalTokens"))
        if total_tokens is None:
            total_tokens = _tokens(usage.get("totalTokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if input_tokens is None and total_tokens is not None and output_tokens is not None:
            input_tokens = _tokens(total_tokens - output_tokens)
        # Hidden/non-replayed reasoning is not part of the next request. Use the
        # measured input base and estimate only the actual retained output then.
        if _tokens(usage.get("reasoningTokens")):
            use_output = False
        base = input_tokens
        local = prepared.local_tokens
        kind = "request_input"
        if use_output and response_replay and total_tokens is not None:
            post_response = deepcopy(prepared.request)
            post_response.extend_replay(deepcopy(list(response_replay)))
            local = manager.count_request(post_response, tools=prepared.tools, reasoning=prepared.reasoning)
            base = total_tokens
            kind = "request_and_response"
        if base is None:
            return
        with self._lock:
            self._anchor = _UsageAnchor(prepared.signature, prepared.history_epoch, base, local, kind, deepcopy(usage))
            self.revision += 1

    def invalidate(self):
        with self._lock:
            self._anchor = None
            self.revision += 1
