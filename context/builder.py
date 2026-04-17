"""
上下文构建器

负责从多个来源收集、压缩、组装上下文。
这是上下文工程的核心编排层。
"""
from typing import List, Dict, Optional, Tuple, Any
from context.window import ContextItem, ContextWindow
from context.source.base import BaseContextSource
from context.compressor.base import BaseCompressor
from context.compressor.token_budget import TokenBudgetCompressor
from context.compressor.history import BaseHistoryCompactor, RuleBasedHistoryCompactor
from context.formatter.base import BaseFormatter
from context.formatter.plain import PlainFormatter
from context.token.counter import TokenCounter
from context.token.budget import TokenBudget
import logging
import json
from core.history import is_canonical_message
from core.request_input import ReplayRequestInput
from core.Message import UserMessage

logger = logging.getLogger(__name__)


class ContextBuilder:
    """多源上下文构建器

    典型使用：
        builder = ContextBuilder()
        builder.add_source(RAGContextSource(pipeline), weight=0.8)
        builder.add_source(HistoryContextSource(history), weight=0.6)
        builder.set_compressor(TokenBudgetCompressor(max_tokens=3000))
        builder.set_formatter(XMLFormatter())

        window = builder.build(query="什么是RAG？")
    """

    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        counter: Optional[TokenCounter] = None,
    ):
        self._sources: List[Tuple[BaseContextSource, float]] = []  # (source, weight)
        self._compressors: Dict[str, BaseCompressor] = {}  # source_name -> compressor
        self._global_compressor: Optional[BaseCompressor] = None
        self._formatter: BaseFormatter = PlainFormatter()
        self._budget = budget or TokenBudget()
        self._counter = counter or TokenCounter()
        self._history_compactor: BaseHistoryCompactor = RuleBasedHistoryCompactor(token_counter=self._counter)
        self._last_compacted_history: List[Dict[str, Any]] = []
        self._last_history_was_compacted: bool = False

    # ---- 配置 API ----

    def add_source(
        self,
        source: BaseContextSource,
        weight: float = 1.0,
        compressor: Optional[BaseCompressor] = None,
    ) -> "ContextBuilder":
        """添加上下文来源

        Args:
            source: 来源适配器
            weight: 权重（影响优先级加成）
            compressor: 专用于此来源的压缩器（可选）

        Returns:
            self（链式调用）
        """
        self._sources.append((source, weight))
        if compressor:
            self._compressors[source.source_name] = compressor
        return self

    def set_compressor(self, compressor: BaseCompressor) -> "ContextBuilder":
        """设置全局压缩器（在所有来源收集完后统一压缩）"""
        self._global_compressor = compressor
        return self

    def set_formatter(self, formatter: BaseFormatter) -> "ContextBuilder":
        """设置格式化器"""
        self._formatter = formatter
        return self

    def set_budget(self, budget: TokenBudget) -> "ContextBuilder":
        """设置 token 预算"""
        self._budget = budget
        return self

    def set_history_compactor(self, compactor: BaseHistoryCompactor) -> "ContextBuilder":
        """设置 history 专用压缩器。"""
        compactor.set_token_counter(self._counter)
        self._history_compactor = compactor
        return self

    # ---- 构建 ----

    def _collect_source_items(
        self,
        query: str,
        budget: TokenBudget,
        exclude_sources: Optional[set[str]] = None,
        **kwargs,
    ) -> List[ContextItem]:
        exclude_sources = exclude_sources or set()
        collected: List[Tuple[str, List[ContextItem]]] = []
        used: Dict[str, int] = {}

        for source, weight in self._sources:
            source_name = source.source_name
            if source_name in exclude_sources:
                continue
            source_budget = budget.get_budget(source_name)

            try:
                items = source.fetch(query, max_tokens=source_budget, **kwargs)
            except Exception as e:
                logger.warning("来源 %s 获取失败: %s", source_name, e)
                continue

            for item in items:
                if item.token_count == 0:
                    item.token_count = self._counter.count(item.content)
                item.priority = min(item.priority * weight, 1.0)

            collected.append((source_name, items))
            used[source_name] = sum(item.token_count for item in items)

        adjusted_budgets = budget.redistribute(used)
        all_items: List[ContextItem] = []
        for source_name, items in collected:
            src_compressor = self._compressors.get(source_name)
            source_budget = adjusted_budgets.get(source_name, budget.get_budget(source_name))
            if src_compressor:
                items = src_compressor.compress(items, max_tokens=source_budget)
            all_items.extend(items)

        return all_items

    def _build_window_from_items(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> ContextWindow:
        window = ContextWindow(
            max_tokens=max_tokens,
            token_counter=self._counter,
        )
        if self._global_compressor:
            compressor_budget = max_tokens
            configured_budget = getattr(self._global_compressor, "max_tokens", 0) or 0
            if configured_budget > 0:
                compressor_budget = min(max_tokens, configured_budget)
            items = self._global_compressor.compress(items, max_tokens=compressor_budget)

        items.sort(key=lambda it: it.priority, reverse=True)
        for item in items:
            if not window.add(item):
                continue

        return window

    def build(
        self,
        query: str,
        reserved_tokens: int = 0,
        exclude_sources: Optional[set[str]] = None,
        **kwargs,
    ) -> ContextWindow:
        """核心构建方法：获取 → 压缩 → 组装"""
        effective_max_tokens = max(0, self._budget.max_tokens - max(0, reserved_tokens))
        effective_budget = TokenBudget(
            max_tokens=effective_max_tokens,
            default_allocation=self._budget.default_allocation,
            allocations=dict(self._budget.allocations),
        )
        all_items = self._collect_source_items(
            query,
            effective_budget,
            exclude_sources=exclude_sources,
            **kwargs,
        )
        return self._build_window_from_items(all_items, effective_max_tokens)

    def build_text(self, query: str, **kwargs) -> str:
        """构建并格式化为文本字符串

        Args:
            query: 当前查询

        Returns:
            格式化后的上下文字符串
        """
        window = self.build(query, **kwargs)

        # 按来源分组
        groups: Dict[str, List[ContextItem]] = {}
        for item in window.items:
            groups.setdefault(item.source, []).append(item)

        return self._formatter.format_all(groups)

    def build_messages(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Any]] = None,
        replay_history: Optional[List[Any]] = None,
        history_converter: Optional[Any] = None,
        include_history: bool = True,
        include_query: bool = True,
        max_turns: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """构建多轮 messages。

        规则：
        1. history 以多轮对话消息格式保留。
        2. 除 history 外的其他来源合并为一条 system 消息。
        3. 当前 query 作为最后一条 user 消息。
        """
        normalized_history = self._copy_history_entries(
            history,
            max_turns=max_turns,
            newest_first=False,
        ) if include_history and history is not None else []
        normalized_replay_history = self._copy_history_entries(
            replay_history,
            max_turns=max_turns,
            newest_first=False,
        ) if include_history and replay_history is not None else []

        base_reserved = self._counter.count(system_prompt or "")
        if include_query and query:
            base_reserved += self._counter.count(query)
        base_reserved += 8

        available_after_base = max(0, self._budget.max_tokens - base_reserved)
        history_reserve = 0
        if include_history:
            history_reserve = int(
                available_after_base * self._budget.allocations.get("history", self._budget.default_allocation)
            )

        non_history_window = self.build(
            query,
            reserved_tokens=base_reserved + history_reserve,
            exclude_sources={"history"},
            **kwargs,
        )
        non_history_groups: Dict[str, List[ContextItem]] = {}
        for item in non_history_window.items:
            non_history_groups.setdefault(item.source, []).append(item)
        context_text = self._formatter.format_all(non_history_groups)

        base_messages: List[Dict[str, Any]] = []
        base_system_parts = [part for part in [system_prompt, context_text] if part]
        if base_system_parts:
            base_messages.append({
                "role": "system",
                "content": "\n\n".join(base_system_parts),
            })
        if include_query and query:
            base_messages.append({"role": "user", "content": query})

        history_budget = max(0, self._budget.max_tokens - self._counter.count_messages(base_messages))
        compacted_history, compacted_canonical_history, history_was_compacted = self._prepare_history_messages(
            normalized_history,
            normalized_replay_history,
            history_budget,
            history_converter=history_converter,
        ) if include_history and normalized_history else ([], [], False)

        messages = self._assemble_messages(
            system_prompt=system_prompt,
            context_text=context_text,
            history_messages=compacted_history if include_history else [],
            query=query if include_query else "",
        )

        total_tokens = self._counter.count_messages(messages)
        if total_tokens > self._budget.max_tokens and include_history and normalized_history:
            overflow = total_tokens - self._budget.max_tokens
            compacted_history, compacted_canonical_history, history_was_compacted = self._prepare_history_messages(
                normalized_history,
                normalized_replay_history,
                max(0, history_budget - overflow),
                history_converter=history_converter,
            )
            messages = self._assemble_messages(
                system_prompt=system_prompt,
                context_text=context_text,
                history_messages=compacted_history,
                query=query if include_query else "",
            )
            total_tokens = self._counter.count_messages(messages)

        if total_tokens > self._budget.max_tokens and non_history_window.items:
            overflow = total_tokens - self._budget.max_tokens
            reduced_items = TokenBudgetCompressor(
                max_tokens=max(0, non_history_window.total_tokens - overflow)
            ).compress(
                non_history_window.items,
                max_tokens=max(0, non_history_window.total_tokens - overflow),
            )
            reduced_groups: Dict[str, List[ContextItem]] = {}
            for item in reduced_items:
                reduced_groups.setdefault(item.source, []).append(item)
            context_text = self._formatter.format_all(reduced_groups)
            messages = self._assemble_messages(
                system_prompt=system_prompt,
                context_text=context_text,
                history_messages=compacted_history if include_history else [],
                query=query if include_query else "",
            )

        self._last_compacted_history = self._copy_history_entries(
            compacted_canonical_history if include_history else [],
            newest_first=False,
        )
        self._last_history_was_compacted = history_was_compacted
        return messages

    def build_request_input(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Any]] = None,
        replay_history: Optional[List[Any]] = None,
        history_converter: Optional[Any] = None,
        message_converter: Optional[Any] = None,
        request_ready_checker: Optional[Any] = None,
        provider_name: Optional[str] = None,
        include_history: bool = True,
        include_query: bool = True,
        max_turns: Optional[int] = None,
        **kwargs,
    ) -> ReplayRequestInput:
        normalized_history = self._copy_history_entries(
            history,
            max_turns=max_turns,
            newest_first=False,
        ) if include_history and history is not None else []
        normalized_replay_history = self._copy_history_entries(
            replay_history,
            max_turns=max_turns,
            newest_first=False,
        ) if include_history and replay_history is not None else []

        base_reserved = self._counter.count(system_prompt or "")
        if include_query and query:
            base_reserved += self._counter.count(query)
        base_reserved += 8

        available_after_base = max(0, self._budget.max_tokens - base_reserved)
        history_reserve = 0
        if include_history:
            history_reserve = int(
                available_after_base * self._budget.allocations.get("history", self._budget.default_allocation)
            )

        non_history_window = self.build(
            query,
            reserved_tokens=base_reserved + history_reserve,
            exclude_sources={"history"},
            **kwargs,
        )
        non_history_groups: Dict[str, List[ContextItem]] = {}
        for item in non_history_window.items:
            non_history_groups.setdefault(item.source, []).append(item)
        context_text = self._formatter.format_all(non_history_groups)

        history_budget = max(
            0,
            self._budget.max_tokens - self._counter.count(system_prompt or "") - self._counter.count(context_text or "") - self._counter.count(query or ""),
        )
        compacted_history, compacted_canonical_history, history_was_compacted = self._prepare_history_messages(
            normalized_history,
            normalized_replay_history,
            history_budget,
            history_converter=history_converter,
        ) if include_history and normalized_history else ([], [], False)

        combined_system = "\n\n".join(part for part in [system_prompt, context_text] if part) or None
        request_input = ReplayRequestInput(
            provider_name=provider_name,
            replay_history=self._copy_history_entries(compacted_history if include_history else [], newest_first=False),
            system_prompt=combined_system,
            message_converter=message_converter,
            request_ready_checker=request_ready_checker,
        )
        if include_query and query:
            request_input.append(UserMessage(query))

        self._last_compacted_history = self._copy_history_entries(
            compacted_canonical_history if include_history else [],
            newest_first=False,
        )
        self._last_history_was_compacted = history_was_compacted
        return request_input

    def compact_history(
        self,
        history: Optional[List[Any]],
        max_tokens: int,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        copied_history = self._copy_history_entries(
            history,
            max_turns=max_turns,
            newest_first=False,
        )
        compacted_history = self._history_compactor.compact(copied_history, max_tokens=max_tokens)
        self._last_compacted_history = self._copy_history_entries(compacted_history, newest_first=False)
        self._last_history_was_compacted = compacted_history != copied_history
        return compacted_history

    def _prepare_history_messages(
        self,
        canonical_history: List[Dict[str, Any]],
        replay_history: List[Dict[str, Any]],
        history_budget: int,
        *,
        history_converter: Optional[Any] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        if not canonical_history:
            return [], [], False

        if not any(is_canonical_message(message) for message in canonical_history):
            compacted = self._history_compactor.compact(canonical_history, max_tokens=history_budget)
            return compacted, self._copy_history_entries(compacted, newest_first=False), compacted != canonical_history

        if not replay_history:
            compacted = self._history_compactor.compact(canonical_history, max_tokens=history_budget)
            return compacted, self._copy_history_entries(compacted, newest_first=False), compacted != canonical_history

        preserve_tail_turns = max(1, int(getattr(self._history_compactor, "recent_turns", 1) or 1))
        canonical_prefix, canonical_tail = self._split_turn_tail(canonical_history, preserve_tail_turns)
        replay_prefix, replay_tail = self._split_turn_tail(replay_history or canonical_history, preserve_tail_turns)

        tail_tokens = self._counter.count_messages(replay_tail) if replay_tail else 0
        prefix_budget = max(0, history_budget - tail_tokens)
        compacted_prefix = self._history_compactor.compact(canonical_prefix, max_tokens=prefix_budget)
        compacted_canonical = [*compacted_prefix, *canonical_tail]
        history_was_compacted = compacted_canonical != canonical_history

        if history_converter is None:
            replay_prefix_messages = self._copy_history_entries(compacted_prefix, newest_first=False)
        else:
            replay_prefix_messages = self._copy_history_entries(
                history_converter(compacted_prefix),
                newest_first=False,
            )
        replay_messages = [*replay_prefix_messages, *replay_tail]

        if self._counter.count_messages(replay_messages) > history_budget:
            overflow = self._counter.count_messages(replay_messages) - history_budget
            if replay_prefix_messages:
                reduced_budget = max(0, prefix_budget - overflow)
                compacted_prefix = self._history_compactor.compact(canonical_prefix, max_tokens=reduced_budget)
                compacted_canonical = [*compacted_prefix, *canonical_tail]
                if history_converter is None:
                    replay_prefix_messages = self._copy_history_entries(compacted_prefix, newest_first=False)
                else:
                    replay_prefix_messages = self._copy_history_entries(
                        history_converter(compacted_prefix),
                        newest_first=False,
                    )
                replay_messages = [*replay_prefix_messages, *replay_tail]

        return replay_messages, compacted_canonical, history_was_compacted

    def _assemble_messages(
        self,
        *,
        system_prompt: Optional[str],
        context_text: str,
        history_messages: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        system_parts = [part for part in [system_prompt, context_text] if part]
        if system_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })
        messages.extend(history_messages)
        if query:
            messages.append({"role": "user", "content": query})
        return messages

    def _copy_history_entries(
        self,
        history: Optional[List[Any]],
        max_turns: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """复制 history 条目，保留 provider-specific 结构。"""
        if not history:
            return []

        selected = history[-max_turns:] if (max_turns and max_turns > 0) else history
        if newest_first:
            selected = list(reversed(selected))
        normalized: List[Dict[str, Any]] = []

        for msg in selected:
            if hasattr(msg, "to_dict"):
                normalized.append(msg.to_dict())
                continue
            if isinstance(msg, dict):
                normalized.append(self._make_json_copy(msg))
                continue
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
            else:
                role = "user"
                content = str(msg)

            normalized.append({"role": str(role), "content": content})

        return normalized

    def _group_turns(self, history: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        turns: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for message in history:
            role = message.get("role")
            if role == "user" and current:
                turns.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            turns.append(current)
        return turns

    def _split_turn_tail(
        self,
        history: List[Dict[str, Any]],
        preserve_tail_turns: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not history:
            return [], []
        turns = self._group_turns(history)
        if preserve_tail_turns <= 0 or preserve_tail_turns >= len(turns):
            return [], self._copy_history_entries(history, newest_first=False)
        prefix_turns = turns[:-preserve_tail_turns]
        tail_turns = turns[-preserve_tail_turns:]
        prefix = [message for turn in prefix_turns for message in turn]
        tail = [message for turn in tail_turns for message in turn]
        return self._copy_history_entries(prefix, newest_first=False), self._copy_history_entries(tail, newest_first=False)

    @staticmethod
    def _make_json_copy(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    
    @property
    def formatter(self) -> BaseFormatter:
        return self._formatter

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    @property
    def counter(self) -> TokenCounter:
        return self._counter

    @property
    def source_names(self) -> List[str]:
        """返回当前已注册来源名列表"""
        return [source.source_name for source, _ in self._sources]
