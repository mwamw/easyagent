"""
上下文构建器

负责从多个来源收集、压缩、组装上下文。
这是上下文工程的核心编排层。
"""
from typing import List, Dict, Optional, Tuple, Any
from context.window import ContextItem, ContextWindow
from context.source.base import BaseContextSource
from context.compressor.base import BaseCompressor
from context.compressor.history import BaseHistoryCompactor, RuleBasedHistoryCompactor
from context.formatter.base import BaseFormatter
from context.formatter.plain import PlainFormatter
from context.token.counter import TokenCounter
from context.token.budget import TokenBudget
import logging
from core.history import _json_safe
from core.providers import create_codec
from core.request_input import ReplayRequestInput

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

    def build_request_input(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        replay_history: Optional[List[Any]] = None,
        provider_name: Optional[str] = None,
        include_query: bool = True,
        extra_replay_entries: Optional[List[Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> ReplayRequestInput:
        normalized_replay_history = self._copy_history_entries(replay_history) if replay_history is not None else []
        codec = create_codec(provider_name)

        request_input = ReplayRequestInput(
            provider_name=provider_name,
            replay_history=normalized_replay_history,
            system_prompt=system_prompt,
        )
        if extra_replay_entries:
            request_input.extend_replay(extra_replay_entries)

        pending_query = codec.query_to_replay(query) if include_query and query else []
        base_request_tokens = codec.count_request_tokens(
            self._counter,
            request_input.replay_history,
            system_prompt=request_input.system_prompt,
            tools=tools,
            pending_messages=pending_query,
            reasoning=reasoning,
        )
        if base_request_tokens < self._budget.max_tokens:
            non_history_window = self.build(
                query,
                reserved_tokens=base_request_tokens,
                exclude_sources={"history"},
                **kwargs,
            )
            non_history_groups: Dict[str, List[ContextItem]] = {}
            for item in non_history_window.items:
                non_history_groups.setdefault(item.source, []).append(item)
            for source, items in non_history_groups.items():
                formatted = self._formatter.format(items, source)
                if formatted:
                    request_input.extend_replay(codec.query_to_replay(formatted))

        if pending_query:
            request_input.extend_replay(pending_query)
        return request_input

    def _copy_history_entries(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """复制 history 条目，保留 provider-specific 结构。"""
        if not history:
            return []

        selected = history
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

    @staticmethod
    def _make_json_copy(value: Any) -> Any:
        return _json_safe(value)
    
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
    def history_compactor(self) -> BaseHistoryCompactor:
        return self._history_compactor

    @property
    def source_names(self) -> List[str]:
        """返回当前已注册来源名列表"""
        return [source.source_name for source, _ in self._sources]
