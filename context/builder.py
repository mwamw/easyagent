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
from context.formatter.base import BaseFormatter
from context.formatter.plain import PlainFormatter
from context.token.counter import TokenCounter
from context.token.budget import TokenBudget
import logging

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

    # ---- 构建 ----

    def build(self, query: str, **kwargs) -> ContextWindow:
        """核心构建方法：获取 → 压缩 → 组装

        Args:
            query: 当前查询

        Returns:
            填充好的 ContextWindow
        """
        window = ContextWindow(
            max_tokens=self._budget.max_tokens,
            token_counter=self._counter,
        )

        all_items: List[ContextItem] = []

        # 1. 从各来源收集
        for source, weight in self._sources:
            source_name = source.source_name
            source_budget = self._budget.get_budget(source_name)

            try:
                items = source.fetch(query, max_tokens=source_budget, **kwargs)
            except Exception as e:
                logger.warning("来源 %s 获取失败: %s", source_name, e)
                continue

            # 计算精确 token 数 & 应用权重
            for item in items:
                if item.token_count == 0:
                    item.token_count = self._counter.count(item.content)
                item.priority = min(item.priority * weight, 1.0)

            # 2. 来源级压缩
            src_compressor = self._compressors.get(source_name)
            if src_compressor:
                items = src_compressor.compress(items, max_tokens=source_budget)

            all_items.extend(items)

        # 3. 全局压缩
        if self._global_compressor:
            all_items = self._global_compressor.compress(all_items)

        # 4. 装入 window
        # 按优先级降序排列，优先添加高优先级项
        all_items.sort(key=lambda it: it.priority, reverse=True)
        for item in all_items:
            if not window.add(item):
                continue  # 预算用完

        return window

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
        include_history: bool = True,
        include_query: bool = True,
        max_turns: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """构建多轮 messages。

        规则：
        1. history 以多轮对话消息格式保留。
        2. 除 history 外的其他来源合并为一条 system 消息。
        3. 当前 query 作为最后一条 user 消息。
        """
        window = self.build(query, **kwargs)
        history =[]
        # 非 history 来源统一拼成 system 上下文
        non_history_groups: Dict[str, List[ContextItem]] = {}
        for item in window.items:
            if item.source == "history":
                temp={"role": item.metadata.get('role', 'user'), "content": item.content}
                history.append(temp)
                continue
            non_history_groups.setdefault(item.source, []).append(item)

        context_text = self._formatter.format_all(non_history_groups)
        system_parts: List[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if context_text:
            system_parts.append(context_text)

        messages: List[Dict[str, str]] = []

        if system_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        if include_history:
            messages.extend(self._normalize_history_messages(history, max_turns=max_turns))

        if include_query and query:
            messages.append({"role": "user", "content": query})

        return messages

    def _normalize_history_messages(
        self,
        history: Optional[List[Any]],
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """将输入历史标准化为消息字典列表。"""
        if not history:
            return []

        selected = history[-max_turns:] if (max_turns and max_turns > 0) else history
        selected = list(reversed(selected))  # 最新消息优先
        normalized: List[Dict[str, str]] = []

        for msg in selected:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = "user"
                content = str(msg)

            normalized.append({"role": str(role), "content": str(content)})

        return normalized
    
    @property
    def formatter(self) -> BaseFormatter:
        return self._formatter

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    @property
    def source_names(self) -> List[str]:
        """返回当前已注册来源名列表"""
        return [source.source_name for source, _ in self._sources]
