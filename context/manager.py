"""
上下文管理器

当前主路径只保留 request-input 构建、history 压缩和 usage 分析。
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from context.builder import ContextBuilder
from context.compressor.base import BaseCompressor
from context.compressor.history import BaseHistoryCompactor
from context.formatter.base import BaseFormatter
from context.history_compaction import (
    HistoryCompactionResult,
    acompact_persistent_history,
    acompact_request_input,
    compact_persistent_history,
    compact_request_input,
)
from context.source.base import BaseContextSource
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
from core.history import _json_safe
from core.providers import create_codec
from core.request_input import ReplayRequestInput

logger = logging.getLogger(__name__)


def _build_usage_payload(
    *,
    label: str,
    request_tokens: int,
    max_tokens: int,
    compaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compaction = dict(compaction or {})
    remaining_tokens = max_tokens - request_tokens
    overflow_tokens = max(0, request_tokens - max_tokens)
    tokens_before = int(compaction.get("tokens_before", request_tokens) or request_tokens)
    tokens_after = int(compaction.get("tokens_after", request_tokens) or request_tokens)
    overflow_before = max(0, tokens_before - max_tokens)
    overflow_after = max(0, tokens_after - max_tokens)
    return {
        "label": label,
        "request_tokens": request_tokens,
        "used_tokens": request_tokens,
        "remaining_tokens": remaining_tokens,
        "overflow_tokens": overflow_tokens,
        "max_tokens": max_tokens,
        "request_compacted": bool(compaction.get("was_compacted", False)),
        "request_compaction_possible": bool(compaction.get("compaction_possible", False)),
        "request_tokens_before_compaction": tokens_before,
        "request_tokens_after_compaction": tokens_after,
        "overflow_tokens_before_compaction": overflow_before,
        "overflow_tokens_after_compaction": overflow_after,
        "tracked_at": datetime.now().isoformat(),
    }


class ContextManager:
    def __init__(
        self,
        builder: Optional[ContextBuilder] = None,
        max_tokens: int = 8000,
        formatter: Optional[BaseFormatter] = None,
        budget: Optional[TokenBudget] = None,
        auto_history: bool = False,
        history_max_turns: int = 50,
    ):
        if budget is None:
            budget = TokenBudget(max_tokens=max_tokens)

        if builder is not None:
            self._builder = builder
        else:
            self._builder = ContextBuilder(budget=budget)

        if formatter:
            self._builder.set_formatter(formatter)

        self._last_usage: Dict[str, Any] = {}
        self._last_request_compaction: Dict[str, Any] = {}
        self._last_persistent_compaction: Dict[str, Any] = {}

    def add_source(
        self,
        source: BaseContextSource,
        weight: float = 1.0,
        compressor: Optional[BaseCompressor] = None,
    ) -> "ContextManager":
        self._builder.add_source(source, weight=weight, compressor=compressor)
        return self

    def set_compressor(self, compressor: BaseCompressor) -> "ContextManager":
        self._builder.set_compressor(compressor)
        return self

    def set_formatter(self, formatter: BaseFormatter) -> "ContextManager":
        self._builder.set_formatter(formatter)
        return self

    def set_history_compactor(self, compactor: BaseHistoryCompactor) -> "ContextManager":
        self._builder.set_history_compactor(compactor)
        return self

    def build_request_input(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        replay_history: Optional[List[Any]] = None,
        provider_name: Optional[str] = None,
        include_query: bool = True,
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
        # max_turns: Optional[int] = None,
        **kwargs,
    ) -> ReplayRequestInput:
        # effective_max_turns = max_turns
        return self._builder.build_request_input(
            query=query,
            replay_history=replay_history,
            provider_name=provider_name,
            system_prompt=system_prompt,
            include_query=include_query,
            tools=tools,
            reasoning=reasoning,
            # max_turns=effective_max_turns,
            **kwargs,
        )

    def compact_history(
        self,
        history: Optional[List[Any]],
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        target_budget = max_tokens if max_tokens is not None else self.budget.max_tokens
        compacted_history = self._builder.compact_history(
            history=history,
            max_tokens=target_budget,
            max_turns=max_turns,
        )
        self._last_persistent_compaction = {
            "was_compacted": bool(compacted_history != list(history or [])),
            "budget": target_budget,
            "tracked_at": datetime.now().isoformat(),
        }
        return compacted_history

    def compact_request_input(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> HistoryCompactionResult:
        result = compact_request_input(
            request_input,
            token_counter=self._builder.counter,
            history_compactor=self._builder._history_compactor,
            max_tokens=max_tokens if max_tokens is not None else self.budget.max_tokens,
            tools=tools,
            reasoning=reasoning,
        )
        self._last_request_compaction = {
            "was_compacted": result.was_compacted,
            "compaction_possible": result.compaction_possible,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "budget": result.budget,
            "tracked_at": datetime.now().isoformat(),
        }
        return result

    async def acompact_request_input(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> HistoryCompactionResult:
        result = await acompact_request_input(
            request_input,
            token_counter=self._builder.counter,
            history_compactor=self._builder._history_compactor,
            max_tokens=max_tokens if max_tokens is not None else self.budget.max_tokens,
            tools=tools,
            reasoning=reasoning,
        )
        self._last_request_compaction = {
            "was_compacted": result.was_compacted,
            "compaction_possible": result.compaction_possible,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "budget": result.budget,
            "tracked_at": datetime.now().isoformat(),
        }
        return result

    def compact_persistent_history(
        self,
        canonical_history: List[Any],
        replay_history: List[Any],
        *,
        provider_name: Optional[str],
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> HistoryCompactionResult:
        result = compact_persistent_history(
            canonical_history,
            replay_history,
            provider_name=provider_name,
            token_counter=self._builder.counter,
            history_compactor=self._builder._history_compactor,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=reasoning,
            max_tokens=max_tokens if max_tokens is not None else self.budget.max_tokens,
        )
        self._last_persistent_compaction = {
            "was_compacted": result.was_compacted,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "budget": result.budget,
            "tracked_at": datetime.now().isoformat(),
        }
        return result

    async def acompact_persistent_history(
        self,
        canonical_history: List[Any],
        replay_history: List[Any],
        *,
        provider_name: Optional[str],
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> HistoryCompactionResult:
        result = await acompact_persistent_history(
            canonical_history,
            replay_history,
            provider_name=provider_name,
            token_counter=self._builder.counter,
            history_compactor=self._builder._history_compactor,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=reasoning,
            max_tokens=max_tokens if max_tokens is not None else self.budget.max_tokens,
        )
        self._last_persistent_compaction = {
            "was_compacted": result.was_compacted,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "budget": result.budget,
            "tracked_at": datetime.now().isoformat(),
        }
        return result

    def analyze_messages_usage(
        self,
        messages: Optional[List[Any] | ReplayRequestInput],
        *,
        max_tokens: Optional[int] = None,
        label: str = "messages",
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        budget_max_tokens = max_tokens if max_tokens is not None else self.budget.max_tokens
        if isinstance(messages, ReplayRequestInput):
            codec = create_codec(messages.provider_name)
            request_tokens = codec.count_request_tokens(
                self._builder.counter,
                messages.replay_history,
                system_prompt=messages.system_prompt,
                tools=tools,
                reasoning=reasoning,
            )
            return _build_usage_payload(
                label=label,
                request_tokens=request_tokens,
                max_tokens=budget_max_tokens,
                compaction=self._last_request_compaction,
            )

        normalized = self._normalize_messages(messages)
        used_tokens = self._builder.counter.count_messages(normalized)
        return _build_usage_payload(
            label=label,
            request_tokens=used_tokens,
            max_tokens=budget_max_tokens,
            compaction=self._last_request_compaction,
        )

    def update_last_usage(
        self,
        messages: Optional[List[Any] | ReplayRequestInput],
        *,
        max_tokens: Optional[int] = None,
        label: str = "messages",
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._last_usage = self.analyze_messages_usage(
            messages,
            max_tokens=max_tokens,
            label=label,
            tools=tools,
            reasoning=reasoning,
        )
        return dict(self._last_usage)

    def set_last_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        self._last_usage = dict(usage or {})

    @property
    def builder(self) -> ContextBuilder:
        return self._builder

    @property
    def budget(self) -> TokenBudget:
        return self._builder.budget

    @property
    def counter(self) -> TokenCounter:
        return self._builder.counter

    @property
    def last_usage(self) -> Dict[str, Any]:
        return dict(self._last_usage)

    @property
    def last_request_compaction(self) -> Dict[str, Any]:
        return dict(self._last_request_compaction)

    @property
    def last_persistent_compaction(self) -> Dict[str, Any]:
        return dict(self._last_persistent_compaction)

    @staticmethod
    def _normalize_messages(messages: Optional[List[Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages or []:
            if hasattr(message, "to_dict"):
                payload = message.to_dict()
            elif isinstance(message, dict):
                payload = _json_safe(message)
            elif hasattr(message, "role") and hasattr(message, "content"):
                payload = {
                    "role": str(getattr(message, "role", "user")),
                    "content": getattr(message, "content", ""),
                }
            else:
                payload = {"role": "user", "content": str(message)}

            if not isinstance(payload, dict):
                payload = {"role": "user", "content": str(payload)}
            normalized.append(payload)
        return normalized
