"""Assemble supplied context, control total input budget, and compact history.

Retrieval, dynamic injection, and persistence belong to their own modules.
"""

from __future__ import annotations

import logging
import json
from typing import Optional, Any
from context.default.builder import ContextBuilder
from context.default.compressor.base import BaseCompressor
from context.default.compressor.history import BaseHistoryCompactor
from context.default.formatter.base import BaseFormatter
from context.base import BaseContextManager
from context.models import (
    ContextCompactionResult,
    ContextUsage,
    ContextUsageReport,
)
from context.default.estimation import RequestEstimation
from context.default.usage import ContextUsageTracker
from context.default.token.budget import TokenBudget
from context.default.token.counter import TokenCounter
from core.request_input import ReplayRequestInput
from core.providers import create_codec
from core.history import _json_safe, coerce_canonical_message

logger = logging.getLogger(__name__)
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
        if role == "user" and not canonical.metadata.get("is_meta") and current:
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
    # Temporary injections must survive until their own lifecycle ends, but must
    # not be absorbed into a persistent summary where removal cannot reach them.
    temporary = []
    persistent = []
    for message in prefix:
        canonical = coerce_canonical_message(message)
        metadata = canonical.metadata if canonical is not None else {}
        if metadata.get("is_meta") and metadata.get("metaMessageLifecycle") != "permanent":
            temporary.append(message)
        else:
            persistent.append(message)
    return persistent, [*temporary, *tail]
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
) -> ContextCompactionResult:
    return ContextCompactionResult(
        canonical_history=canonical_history,
        replay_history=replay_history,
        was_compacted=was_compacted,
        compaction_possible=compaction_possible,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=budget,
        metadata=dict(metadata or {}),
    )


class ContextManager(BaseContextManager, RequestEstimation):
    def append_tool_context(self, tool_name: str, value: Any) -> None:
        if value is None or self._metamessages is None:
            return
        from runtime.metamessage import MetaMessage, MetaMessageLifecycle

        content = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, default=str, indent=2,
        )
        if content.strip():
            self._metamessages.emit(MetaMessage(
                name=f"tool_context:{tool_name}",
                content=(
                    f"<system-reminder>\nRuntime context produced by tool `{tool_name}`:\n"
                    f"{content}\n</system-reminder>"
                ),
                lifecycle=MetaMessageLifecycle.INVOCATION,
                metadata={"source": "tool_result", "toolName": tool_name},
            ))

    def get_prompt_blocks(self, context):
        from prompt import PromptBlock
        if self.budget.max_tokens is None:
            return []
        return [PromptBlock(
            "compaction",
            "Auto-compact is enabled. When the context window is nearly full, older messages "
            "may be compacted. Continue from retained context and re-read sources for exact details.",
            placement="system_reminder", order=200,
        )]

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
        max_tokens: int | None = None,
        formatter: Optional[BaseFormatter] = None,
        budget: Optional[TokenBudget] = None,
    ):
        super().__init__()
        self._context_epoch = 0
        self._usage = ContextUsageTracker()
        self._prepared_usage = None
        if budget is None:
            budget = TokenBudget(max_tokens=max_tokens)
        self._builder = builder or ContextBuilder(budget=budget)
        if formatter:
            self._builder.set_formatter(formatter)

    def bind(self, llm, *, metamessages=None, owner=None):
        previous = self._llm
        super().bind(
            llm,
            metamessages=metamessages,
            owner=owner,
        )
        if previous is not None and previous is not llm:
            self._context_epoch += 1
            self._usage.invalidate()
        return self

    @property
    def context_epoch(self):
        return self._context_epoch

    @property
    def history_epoch(self):
        return self._context_epoch

    def validation_epoch(self, compaction: ContextCompactionResult | None) -> int:
        return self.history_epoch + int(compaction is not None and compaction.was_compacted)

    @property
    def provider_name(self):
        return self._llm.provider_name if self._llm is not None else None

    @property
    def usage_revision(self):
        return self._usage.revision

    @property
    def last_query(self):
        return self._usage.last_query

    def replace(self, messages):
        super().replace(messages)
        self._context_epoch += 1
        self._usage.invalidate()

    def rebuild_replay(self):
        self._usage.invalidate()
        return self.replay

    def change_llm(self, llm):
        self.bind(
            llm,
            metamessages=self._metamessages,
            owner=self._owner,
        )

    def clear(self):
        super().clear()
        self._context_epoch += 1
        self._usage.invalidate()

    def restore_state(self, state):
        super().restore_state(state)
        self._context_epoch += 1
        self._usage.invalidate()

    def build_input(self, prompt_blocks, *, tools, options, query,
                    cache_policy, replay_history=None) -> ReplayRequestInput:
        from core.request_compiler import compile_prompt_blocks

        compiled = compile_prompt_blocks(prompt_blocks, cache_policy=cache_policy)
        return self.build_request_input(
            query=query, include_query=False,
            replay_history=self.replay if replay_history is None else replay_history,
            provider_name=self._require_llm().provider_name,
            system_prompt=compiled.system_prompt,
            system_prompt_blocks=compiled.system_prompt_blocks,
            system_reminder_blocks=compiled.system_reminder_blocks,
            dynamic_tail_blocks=compiled.dynamic_tail_blocks,
            on_demand_expansion_blocks=compiled.on_demand_expansion_blocks,
            cache_policy=compiled.cache_policy, tools=tools,
            reasoning=options.get("reasoning"),
        )

    def calculate_usage(self, request_input, *, tools, options) -> ContextUsage:
        estimate = self.estimate_request(
            request_input, usage_tracker=self._usage, llm=self._llm,
            history_epoch=self.context_epoch, tools=tools,
            reasoning=options.get("reasoning"), options=options,
        )
        return ContextUsage(
            estimated_tokens=estimate.tokens, max_tokens=self.budget.max_tokens,
            local_estimated_tokens=estimate.local_tokens, source=estimate.source,
            metadata=dict(estimate.metadata),
        )

    def should_compact(self, usage: ContextUsage) -> bool:
        return usage.over_budget

    def compact_context(self, request_input, usage, *, canonical_history,
                        replay_history, tools, options, max_tokens=None):
        base_tokens = self._llm.count_request_tokens(
            self.counter, replay_history,
            system_prompt=request_input.render_system_prompt(), tools=tools,
            reasoning=options.get("reasoning"),
        )
        overhead = max(0, (usage.local_estimated_tokens or usage.estimated_tokens) - base_tokens)
        limit = usage.max_tokens if max_tokens is None else max_tokens
        return self.compact_persistent_history(
            canonical_history, replay_history, provider_name=self._llm.provider_name,
            token_counter=self.counter, system_prompt=request_input.render_system_prompt(),
            tools=tools, reasoning=options.get("reasoning"),
            max_tokens=max(0, min(usage.max_tokens, int(limit)) - overhead),
            tokens_before_override=max(0, usage.estimated_tokens - overhead),
            metadata={"token_estimate": {"source": usage.source, **usage.metadata}},
        )

    async def acompact_context(self, request_input, usage, *, canonical_history,
                               replay_history, tools, options, max_tokens=None):
        # Preserve the default compactor's native asynchronous implementation.
        base_tokens = self._llm.count_request_tokens(
            self.counter, replay_history,
            system_prompt=request_input.render_system_prompt(), tools=tools,
            reasoning=options.get("reasoning"),
        )
        overhead = max(0, (usage.local_estimated_tokens or usage.estimated_tokens) - base_tokens)
        limit = usage.max_tokens if max_tokens is None else max_tokens
        return await self.acompact_persistent_history(
            canonical_history, replay_history, provider_name=self._llm.provider_name,
            token_counter=self.counter, system_prompt=request_input.render_system_prompt(),
            tools=tools, reasoning=options.get("reasoning"),
            max_tokens=max(0, min(usage.max_tokens, int(limit)) - overhead),
            tokens_before_override=max(0, usage.estimated_tokens - overhead),
            metadata={"token_estimate": {"source": usage.source, **usage.metadata}},
        )

    def validate_input(self, request_input, *, tools, options, history_epoch=None):
        self.validate_request(
            request_input, tools=tools, reasoning=options.get("reasoning"),
            usage_tracker=self._usage, llm=self._llm,
            history_epoch=self.context_epoch if history_epoch is None else history_epoch,
            options=options,
        )

    def observe_prepared_input(self, request_input, *, tools, options, query):
        self._prepared_usage = self._usage.prepare(
            self, request_input, llm=self._llm, history_epoch=self.context_epoch,
            tools=tools, reasoning=options.get("reasoning"), options=options,
            query=query,
        )

    def observe_model_result(self, usage, response_replay, *, use_output):
        self._usage.observe(self, self._prepared_usage, usage, response_replay=response_replay, use_output=use_output)

    async def acompact_persistent_history(self, *args, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.compact_persistent_history, *args, **kwargs)

    def get_usage(self, prompt_blocks, *, tools=None, reasoning=None, cache_policy=None) -> ContextUsageReport:
        from core.request_compiler import compile_prompt_blocks

        tracker = self._usage
        manager = self
        compiled = compile_prompt_blocks(
            prompt_blocks, cache_policy=cache_policy,
        )
        request = manager.build_request_input(
            query="", include_query=False, replay_history=self.replay,
            provider_name=self._llm.provider_name, system_prompt=compiled.system_prompt,
            system_prompt_blocks=compiled.system_prompt_blocks,
            system_reminder_blocks=compiled.system_reminder_blocks,
            dynamic_tail_blocks=compiled.dynamic_tail_blocks,
            on_demand_expansion_blocks=compiled.on_demand_expansion_blocks,
            cache_policy=compiled.cache_policy, tools=tools, reasoning=reasoning,
        )
        estimate = manager.estimate_request(
            request, usage_tracker=tracker, llm=self._llm,
            history_epoch=self.context_epoch,
            tools=tools, reasoning=reasoning, options=tracker.last_options,
        )
        limit = manager.budget.max_tokens
        return {
            "estimatedRequestTokens": estimate.tokens,
            "localEstimatedRequestTokens": estimate.local_tokens,
            "request_estimate_source": estimate.source,
            "requestEstimate": {
                "estimatedRequestTokens": estimate.tokens, "source": estimate.source,
                "localEstimatedRequestTokens": estimate.local_tokens, "metadata": estimate.metadata,
            },
            "maxTokens": limit,
            "budget": {"maxTokens": limit, "remainingTokens": None if limit is None else max(0, limit - estimate.tokens)},
            "canonicalMessages": len(self.canonical),
            "replayMessages": len(self.replay),
            "provider": self._llm.provider_name,
        }

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
        system_reminder_blocks: Optional[list[Any]] = None,
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
            system_reminder_blocks=system_reminder_blocks,
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
    ) -> ContextCompactionResult:
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
            from context.default.compressor.history import RuleBasedHistoryCompactor
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
    ) -> ContextCompactionResult:
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
            from context.default.compressor.history import RuleBasedHistoryCompactor
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
