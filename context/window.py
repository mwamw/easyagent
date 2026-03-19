"""
上下文窗口核心数据结构

ContextItem: 单条上下文
ContextWindow: 上下文容器，管理 token 预算
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .token.counter import TokenCounter


@dataclass
class ContextItem:
    """单条上下文项"""

    content: str
    source: str  # "system" | "history" | "rag" | "memory" | "tool"
    priority: float = 0.5  # 0.0~1.0, 越高越优先保留
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_count == 0 and self.content:
            # 延迟计数；调用方也可以提前设置
            self.token_count = max(1, int(len(self.content) / 3.5))


class ContextWindow:
    """上下文窗口，管理多个 ContextItem 并守住 token 预算"""

    def __init__(
        self,
        max_tokens: int = 8000,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.max_tokens = max_tokens
        self.counter = token_counter or TokenCounter()
        self._items: List[ContextItem] = []

    # ---- 增删查 ----

    def add(self, item: ContextItem) -> bool:
        """添加项，若超预算返回 False"""
        if item.token_count == 0:
            item.token_count = self.counter.count(item.content)
        if self.total_tokens + item.token_count > self.max_tokens:
            return False
        self._items.append(item)
        return True

    def add_force(self, item: ContextItem) -> None:
        """强制添加（不检查预算）"""
        if item.token_count == 0:
            item.token_count = self.counter.count(item.content)
        self._items.append(item)

    def remove(self, item: ContextItem) -> None:
        self._items.remove(item)

    def clear(self) -> None:
        self._items.clear()

    # ---- 属性 ----

    @property
    def items(self) -> List[ContextItem]:
        return list(self._items)

    @property
    def total_tokens(self) -> int:
        return sum(it.token_count for it in self._items)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    def fits_budget(self, extra_tokens: int = 0) -> bool:
        return self.total_tokens + extra_tokens <= self.max_tokens

    # ---- 按来源 / 优先级操作 ----

    def items_by_source(self, source: str) -> List[ContextItem]:
        return [it for it in self._items if it.source == source]

    def tokens_by_source(self) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        for it in self._items:
            usage[it.source] = usage.get(it.source, 0) + it.token_count
        return usage

    def sort_by_priority(self, descending: bool = True) -> None:
        """按优先级排序"""
        self._items.sort(key=lambda it: it.priority, reverse=descending)

    def trim_to_budget(self) -> List[ContextItem]:
        """按优先级从低到高移除项，直到符合预算。返回被移除的项。"""
        self._items.sort(key=lambda it: it.priority)
        removed = []
        while self.total_tokens > self.max_tokens and self._items:
            removed.append(self._items.pop(0))
        # 恢复原顺序
        self._items.sort(key=lambda it: it.priority, reverse=True)
        return removed

    # ---- 转换 ----

    def to_text(self, separator: str = "\n\n") -> str:
        """拼接为纯文本"""
        return separator.join(it.content for it in self._items if it.content)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"ContextWindow(items={len(self._items)}, "
            f"tokens={self.total_tokens}/{self.max_tokens})"
        )
