"""
上下文管理器

顶层协调器，供 Agent 直接使用。
封装 ContextBuilder，提供简化的 API。
"""
from typing import List, Optional, Any, Dict
from datetime import datetime
import json
from context.window import ContextItem, ContextWindow
from context.builder import ContextBuilder
from context.source.base import BaseContextSource
from context.source.history_source import HistoryContextSource
from context.compressor.base import BaseCompressor
from context.compressor.history import BaseHistoryCompactor
from context.formatter.base import BaseFormatter
from context.formatter.plain import PlainFormatter
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
import logging

logger = logging.getLogger(__name__)


class ContextManager:
    """上下文管理器 — Agent 使用的统一接口

    Example:
        # 简单模式
        manager = ContextManager(max_tokens=4000)

        # 标准模式
        manager = ContextManager(max_tokens=8000)
        manager.add_source(RAGContextSource(pipeline), weight=0.8)
        manager.add_source(MemoryContextSource(memory_manage), weight=0.6)
        manager.set_formatter(XMLFormatter())

        # Agent 调用
        context_str = manager.build_context("什么是RAG？", history=agent.history)
    """

    def __init__(
        self,
        builder: Optional[ContextBuilder] = None,
        max_tokens: int = 8000,
        formatter: Optional[BaseFormatter] = None,
        budget: Optional[TokenBudget] = None,
        auto_history: bool = True,
        history_max_turns: int = 50,
    ):
        """
        Args:
            builder: 预配置的 ContextBuilder（可选）
            max_tokens: 总 token 上限
            formatter: 格式化器
            budget: token 预算（覆盖 max_tokens）
            auto_history: 是否自动注入对话历史
            history_max_turns: 历史最大轮次
        """
        if budget is None:
            budget = TokenBudget(max_tokens=max_tokens)

        if builder is not None:
            self._builder = builder
        else:
            self._builder = ContextBuilder(budget=budget)

        if formatter:
            self._builder.set_formatter(formatter)

        self._auto_history = auto_history
        self._history_source = HistoryContextSource(max_turns=history_max_turns)
        self._history_added = False
        self._last_usage: Dict[str, Any] = {}

    # ---- 配置 API（代理到 builder） ----

    def add_source(
        self,
        source: BaseContextSource,
        weight: float = 1.0,
        compressor: Optional[BaseCompressor] = None,
    ) -> "ContextManager":
        """添加上下文来源"""
        self._builder.add_source(source, weight=weight, compressor=compressor)
        return self

    def set_compressor(self, compressor: BaseCompressor) -> "ContextManager":
        """设置全局压缩器"""
        self._builder.set_compressor(compressor)
        return self

    def set_formatter(self, formatter: BaseFormatter) -> "ContextManager":
        """设置格式化器"""
        self._builder.set_formatter(formatter)
        return self

    def set_history_compactor(self, compactor: BaseHistoryCompactor) -> "ContextManager":
        """设置 history 专用压缩器。"""
        self._builder.set_history_compactor(compactor)
        return self

    # ---- 核心 API ----

    def build_context(
        self,
        query: str,
        history: Optional[List] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """构建格式化的上下文字符串

        Args:
            query: 当前用户查询
            history: 对话历史（Message 列表）

        Returns:
            格式化后的上下文字符串，可拼接到 system prompt
        """
        self._inject_history(history, include_history=include_history)
        return self._builder.build_text(query, **kwargs)

    def build_window(
        self,
        query: str,
        history: Optional[List] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> ContextWindow:
        """构建 ContextWindow 对象（供需要更细粒度控制的场景）

        Args:
            query: 当前用户查询
            history: 对话历史

        Returns:
            ContextWindow 实例
        """
        self._inject_history(history, include_history=include_history)
        return self._builder.build(query, **kwargs)

    def build_messages(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List] = None,
        include_history: Optional[bool] = None,
        include_query: bool = True,
        include_qeury: Optional[bool] = None,
        max_turns: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """构建多轮 messages（system + history + current user）。

        Args:
            query: 当前用户查询
            system_prompt: 系统提示
            history: 对话历史
            include_history: 是否包含 history（None 时遵循 auto_history）
            include_query: 是否追加当前 query 作为 user 消息
            include_qeury: 兼容旧拼写参数（优先级低于 include_query）
            max_turns: history 最大条数

        Returns:
            消息列表（dict 格式）
        """
        # 兼容历史拼写参数 include_qeury
        if include_qeury is not None:
            include_query = include_qeury
        use_history = self._auto_history if include_history is None else include_history
        effective_max_turns = max_turns

        return self._builder.build_messages(
            query=query,
            history=history,
            system_prompt=system_prompt,
            include_history=use_history,
            include_query=include_query,
            max_turns=effective_max_turns,
            **kwargs,
        )

    def compact_history(
        self,
        history: Optional[List[Any]],
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """显式压缩 history，返回压缩后的 history 消息列表。"""
        effective_max_turns = max_turns if max_turns is not None else self._history_source.max_turns
        target_budget = max_tokens if max_tokens is not None else self.budget.max_tokens
        return self._builder.compact_history(
            history=history,
            max_tokens=target_budget,
            max_turns=effective_max_turns,
        )

    def analyze_messages_usage(
        self,
        messages: Optional[List[Any]],
        *,
        max_tokens: Optional[int] = None,
        label: str = "visible_messages",
    ) -> Dict[str, Any]:
        """分析一组消息的 token 使用情况。"""
        normalized = self._normalize_messages(messages)
        budget_max_tokens = max_tokens if max_tokens is not None else self.budget.max_tokens
        used_tokens = self._builder.counter.count_messages(normalized)
        remaining_tokens = max(0, budget_max_tokens - used_tokens)

        return {
            "label": label,
            "message_count": len(normalized),
            "used_tokens": used_tokens,
            "remaining_tokens": remaining_tokens,
            "max_tokens": budget_max_tokens,
            "history_compacted": self.last_history_was_compacted,
            "tracked_at": datetime.now().isoformat(),
        }

    def update_last_usage(
        self,
        messages: Optional[List[Any]],
        *,
        max_tokens: Optional[int] = None,
        label: str = "visible_messages",
    ) -> Dict[str, Any]:
        """刷新并保存最近一次消息构造的 token 使用快照。"""
        self._last_usage = self.analyze_messages_usage(
            messages,
            max_tokens=max_tokens,
            label=label,
        )
        return dict(self._last_usage)

    def set_last_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        """显式恢复最近一次上下文使用快照。"""
        self._last_usage = dict(usage or {})

    # ---- 内部 ----

    def _inject_history(
        self,
        history: Optional[List],
        include_history: Optional[bool] = None,
    ) -> None:
        """将对话历史注入 builder（仅首次或有新历史时）"""
        use_history = self._auto_history if include_history is None else include_history

        if not use_history:
            # 显式禁用时，避免复用到旧 history
            self._history_source.set_history([])
            return

        # 每次都更新，避免复用到旧 history
        self._history_source.set_history(history or [])

        if not self._history_added:
            from context.compressor.sliding_window import SlidingWindowCompressor

            self._builder.add_source(
                self._history_source,
                weight=0.7,
                compressor=SlidingWindowCompressor(
                    max_items=self._history_source.max_turns
                ),
            )
            self._history_added = True

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
    def last_compacted_history(self) -> List[Dict[str, Any]]:
        return list(getattr(self._builder, "_last_compacted_history", []) or [])

    @property
    def last_history_was_compacted(self) -> bool:
        return bool(getattr(self._builder, "_last_history_was_compacted", False))

    @property
    def last_usage(self) -> Dict[str, Any]:
        return dict(self._last_usage)

    @staticmethod
    def _normalize_messages(messages: Optional[List[Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages or []:
            if hasattr(message, "to_dict"):
                payload = message.to_dict()
            elif isinstance(message, dict):
                payload = json.loads(json.dumps(message, ensure_ascii=False, default=str))
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
