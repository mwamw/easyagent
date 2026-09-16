"""Assemble already available context. No I/O, sources, or retrieval policies."""

from copy import deepcopy
from typing import Any, Iterable

from context.default.window import ContextItem, ContextWindow
from context.default.compressor.history import RuleBasedHistoryCompactor
from context.default.formatter.plain import PlainFormatter
from context.default.token.counter import TokenCounter
from context.default.token.budget import ContextBudgetExceeded, TokenBudget
from core.providers import create_codec
from core.request_input import ReplayRequestInput


class ContextBuilder:
    def __init__(self, budget=None, counter=None):
        self._budget = budget or TokenBudget()
        self._counter = counter or TokenCounter()
        self._formatter = PlainFormatter()
        self._global_compressor = None
        self._history_compactor = RuleBasedHistoryCompactor(token_counter=self._counter)

    def set_compressor(self, compressor):
        self._global_compressor = compressor
        return self

    def set_formatter(self, formatter):
        self._formatter = formatter
        return self

    def set_budget(self, budget):
        self._budget = budget
        return self

    def set_history_compactor(self, compactor):
        compactor.set_token_counter(self._counter)
        self._history_compactor = compactor
        return self

    def build(self, items: Iterable[ContextItem], *, reserved_tokens: int = 0) -> ContextWindow:
        """Build from supplied items, preserving input order and caller objects."""
        selected = deepcopy(list(items))
        for item in selected:
            if not isinstance(item, ContextItem):
                raise TypeError("ContextBuilder.build expects ContextItem values, not a query or source")
            item.token_count = self._counter.count(item.content)
        available = self._budget.remaining(max(0, reserved_tokens))
        if available is None:
            available = sum(item.token_count for item in selected)
        if self._global_compressor is not None:
            selected = self._global_compressor.compress(selected, max_tokens=available)
        window = ContextWindow(max_tokens=available, token_counter=self._counter)
        for item in selected:
            if not window.add(item):
                raise ContextBudgetExceeded(sum(it.token_count for it in selected), available)
        return window

    def build_text(self, items: Iterable[ContextItem], **kwargs) -> str:
        window = self.build(items, **kwargs)
        # Provenance is presentation metadata, never a source registry or quota.
        return "\n\n".join(self._formatter.format([item], item.source) for item in window.items)

    def build_request_input(
        self, query: str = "", *, system_prompt=None, system_prompt_blocks=None,
        system_reminder_blocks=None, dynamic_tail_blocks=None,
        on_demand_expansion_blocks=None, cache_policy=None, cache_metadata=None,
        replay_history=None, provider_name=None, include_query=True,
        extra_replay_entries=None, tools=None, reasoning=None,
        context_items: Iterable[ContextItem] = (),
    ) -> ReplayRequestInput:
        request = ReplayRequestInput(
            provider_name=provider_name,
            replay_history=deepcopy(list(replay_history or [])),
            system_prompt=system_prompt,
            system_prompt_blocks=list(system_prompt_blocks or []),
            system_reminder_blocks=list(system_reminder_blocks or []),
            dynamic_tail_blocks=list(dynamic_tail_blocks or []),
            on_demand_expansion_blocks=list(on_demand_expansion_blocks or []),
            cache_policy=cache_policy,
            cache_metadata=dict(cache_metadata or {}),
        )
        request.apply_runtime_layers()
        if extra_replay_entries:
            request.extend_replay(deepcopy(extra_replay_entries))
        for item in context_items:
            text = self._formatter.format([item], item.source)
            if text:
                request.append_dynamic_tail_text(text)
        if include_query and query:
            request.extend_replay(create_codec(provider_name).query_to_replay(query))
        return request

    @property
    def formatter(self):
        return self._formatter

    @property
    def budget(self):
        return self._budget

    @property
    def counter(self):
        return self._counter

    @property
    def history_compactor(self):
        return self._history_compactor
