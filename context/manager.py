"""
上下文管理器

当前职责只保留：
1. 管理外部 context source 的收集与压缩
2. 构建本次请求使用的额外上下文消息
3. 暴露 token budget / history compactor 配置
"""

from __future__ import annotations

from typing import Optional,Any
from dataclasses import dataclass
from venv import logger
from context.builder import ContextBuilder
from context.compressor.base import BaseCompressor
from context.compressor.history import BaseHistoryCompactor
from context.formatter.base import BaseFormatter
from context.source.base import BaseContextSource
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
from core.request_input import ReplayRequestInput
from core.providers import create_codec
from core.history import CanonicalMessage, _json_safe, coerce_canonical_message
@dataclass
class HistoryCompactionResult:
    canonical_history: list[Any]
    replay_history: list[Any]
    was_compacted: bool
    compaction_possible: bool
    tokens_before: int
    tokens_after: int
    budget: int
    metadata: dict[str, Any]
def _copy_entry(value: Any) -> Any:
    if isinstance(value, dict):
        return _json_safe(value)
    if hasattr(value, "to_dict"):
        try:
            payload = value.to_dict()
            if isinstance(payload, dict):
                return _json_safe(payload)
        except Exception:
            pass
    return value
def _group_turns(history: list[Any]) -> list[list[Any]]:
    turns: list[list[Any]] = []
    current: list[Any] = []
    for message in history:
        canonical = coerce_canonical_message(message)
        role = canonical.role if canonical is not None else None
        if role == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns
def _split_turn_tail(history: list[Any], preserve_tail_turns: int) -> tuple[list[Any], list[Any]]:
    if not history:
        return [], []
    turns = _group_turns(history)
    if preserve_tail_turns <= 0 or preserve_tail_turns >= len(turns):
        return [], [_copy_entry(message) for message in history]
    prefix_turns = turns[:-preserve_tail_turns]
    tail_turns = turns[-preserve_tail_turns:]
    prefix = [_copy_entry(message) for turn in prefix_turns for message in turn]
    tail = [_copy_entry(message) for turn in tail_turns for message in turn]
    return prefix, tail
def _compactor_recent_turns(history_compactor: Any) -> int:
    return max(1, int(getattr(history_compactor, "recent_turns", 1) or 1))


def _result(
    *,
    canonical_history: list[Any],
    replay_history: list[Any],
    was_compacted: bool,
    compaction_possible: bool,
    tokens_before: int,
    tokens_after: int,
    budget: int,
    metadata: Optional[dict[str, Any]] = None,
) -> HistoryCompactionResult:
    return HistoryCompactionResult(
        canonical_history=canonical_history,
        replay_history=replay_history,
        was_compacted=was_compacted,
        compaction_possible=compaction_possible,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=budget,
        metadata=dict(metadata or {}),
    )


class ContextManager:
    def _history_compactor_metadata(self) -> dict[str, Any]:
        info = {}
        if hasattr(self.history_compactor, "get_last_run_info"):
            try:
                info = self.history_compactor.get_last_run_info()
            except Exception:
                info = {}
        return {"compactor": _json_safe(info)} if info else {}

    @staticmethod
    def _merge_compactor_metadata(current: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
        if not latest:
            return current
        if not current:
            return latest
        latest_compactor = latest.get("compactor") if isinstance(latest, dict) else None
        if isinstance(latest_compactor, dict) and latest_compactor.get("status") == "skipped":
            return current
        return {**current, **latest}

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
        system_prompt_blocks: Optional[list[Any]] = None,
        runtime_reminder_blocks: Optional[list[Any]] = None,
        dynamic_tail_blocks: Optional[list[Any]] = None,
        on_demand_expansion_blocks: Optional[list[Any]] = None,
        cache_policy: Optional[Any] = None,
        cache_metadata: Optional[dict[str, Any]] = None,
        replay_history: Optional[list[object]] = None,
        provider_name: Optional[str] = None,
        include_query: bool = True,
        extra_replay_entries: Optional[list[object]] = None,
        tools: Optional[Any] = None,
        reasoning: Optional[dict[str, object]] = None,
        **kwargs,
    ) -> ReplayRequestInput:
        return self._builder.build_request_input(
            query=query,
            replay_history=replay_history,
            provider_name=provider_name,
            system_prompt=system_prompt,
            system_prompt_blocks=system_prompt_blocks,
            runtime_reminder_blocks=runtime_reminder_blocks,
            dynamic_tail_blocks=dynamic_tail_blocks,
            on_demand_expansion_blocks=on_demand_expansion_blocks,
            cache_policy=cache_policy,
            cache_metadata=cache_metadata,
            include_query=include_query,
            extra_replay_entries=extra_replay_entries,
            tools=tools,
            reasoning=reasoning,
            **kwargs,
        )

    def compact_persistent_history(
        self,
        canonical_history: list[Any],
        replay_history: list[Any],
        *,
        provider_name: Optional[str],
        token_counter: Any,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        reasoning: Optional[dict[str, Any]] = None,
        max_tokens: int,
        force: bool = False,
        tokens_before_override: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> HistoryCompactionResult:
        codec = create_codec(provider_name)
        base_metadata = dict(metadata or {})
        if tokens_before_override is None:
            tokens_before = codec.count_request_tokens(
                token_counter,
                replay_history,
                system_prompt=system_prompt,
                tools=tools,
                reasoning=reasoning,
            )
        else:
            tokens_before = int(tokens_before_override)
        if not force and tokens_before <= max_tokens:
            return _result(
                canonical_history=canonical_history,
                replay_history=replay_history,
                was_compacted=False,
                compaction_possible=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                budget=max_tokens,
                metadata=base_metadata,
            )

        preserve_tail_turns = _compactor_recent_turns(self.history_compactor)
        canonical_prefix, canonical_tail = _split_turn_tail(canonical_history, preserve_tail_turns)
        if not canonical_prefix:
            if force:
                logger.warning("无法压缩 history，因为 prefix messages 为空")
            return _result(
                canonical_history=canonical_history,
                replay_history=replay_history,
                was_compacted=False,
                compaction_possible=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                budget=max_tokens,
                metadata=base_metadata,
            )

        try:
            compacted_prefix = self.history_compactor.compact(canonical_prefix)
            compactor_metadata = self._merge_compactor_metadata(base_metadata, self._history_compactor_metadata())
        except Exception as exc:
            logger.warning("历史压缩器运行失败，回退到规则截断: %s", exc)
            from context.compressor.history import RuleBasedHistoryCompactor
            fallback_compactor = RuleBasedHistoryCompactor(token_counter=token_counter)
            compacted_prefix = fallback_compactor.compact(canonical_prefix)
            compactor_metadata = self._merge_compactor_metadata(
                base_metadata,
                {
                    "compactor": {
                        "compactor": self.history_compactor.__class__.__name__,
                        "status": "fallback",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "fallback_used": True,
                        "fallback_compactor": "RuleBasedHistoryCompactor",
                    }
                }
            )
        compacted_canonical = [*compacted_prefix, *canonical_tail]
        compacted_replay = codec.canonical_to_replay(compacted_canonical)
        tokens_after = codec.count_request_tokens(
            token_counter,
            compacted_replay,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=reasoning,
        )
        return _result(
            canonical_history=compacted_canonical,
            replay_history=compacted_replay,
            was_compacted=compacted_canonical != canonical_history,
            compaction_possible=True,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            budget=max_tokens,
            metadata=compactor_metadata,
        )


    async def acompact_persistent_history(
        self,
        canonical_history: list[Any],
        replay_history: list[Any],
        *,
        provider_name: Optional[str],
        token_counter: Any,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        reasoning: Optional[dict[str, Any]] = None,
        max_tokens: int,
        force: bool = False,
        tokens_before_override: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> HistoryCompactionResult:
        codec = create_codec(provider_name)
        base_metadata = dict(metadata or {})
        if tokens_before_override is None:
            tokens_before = codec.count_request_tokens(
                token_counter,
                replay_history,
                system_prompt=system_prompt,
                tools=tools,
                reasoning=reasoning,
            )
        else:
            tokens_before = int(tokens_before_override)
        if not force and tokens_before <= max_tokens:
            return _result(
                canonical_history=canonical_history,
                replay_history=replay_history,
                was_compacted=False,
                compaction_possible=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                budget=max_tokens,
                metadata=base_metadata,
            )

        preserve_tail_turns = _compactor_recent_turns(self.history_compactor)
        canonical_prefix, canonical_tail = _split_turn_tail(canonical_history, preserve_tail_turns)
        if not canonical_prefix:
            return _result(
                canonical_history=canonical_history,
                replay_history=replay_history,
                was_compacted=False,
                compaction_possible=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                budget=max_tokens,
                metadata=base_metadata,
            )

        codec = create_codec(provider_name)
        try:
            compacted_prefix = await self.history_compactor.acompact(canonical_prefix)
            compactor_metadata = self._merge_compactor_metadata(base_metadata, self._history_compactor_metadata())
        except Exception as exc:
            logger.warning("历史异步压缩器运行失败，回退到规则截断: %s", exc)
            from context.compressor.history import RuleBasedHistoryCompactor
            fallback_compactor = RuleBasedHistoryCompactor(token_counter=token_counter)
            compacted_prefix = await fallback_compactor.acompact(canonical_prefix)
            compactor_metadata = self._merge_compactor_metadata(
                base_metadata,
                {
                    "compactor": {
                        "compactor": self.history_compactor.__class__.__name__,
                        "status": "fallback",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "fallback_used": True,
                        "fallback_compactor": "RuleBasedHistoryCompactor",
                    }
                }
            )
        compacted_canonical = [*compacted_prefix, *canonical_tail]
        compacted_replay = codec.canonical_to_replay(compacted_canonical)
        tokens_after = codec.count_request_tokens(
            token_counter,
            compacted_replay,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=reasoning,
        )

        return _result(
            canonical_history=compacted_canonical,
            replay_history=compacted_replay,
            was_compacted=compacted_canonical != canonical_history,
            compaction_possible=True,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            budget=max_tokens,
            metadata=compactor_metadata,
        )

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
    def history_compactor(self) -> BaseHistoryCompactor:
        return self._builder.history_compactor
