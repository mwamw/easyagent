"""
上下文格式化器抽象基类
"""
from abc import ABC, abstractmethod
from typing import List
from context.window import ContextItem


class BaseFormatter(ABC):
    """上下文格式化器基类"""

    @abstractmethod
    def format(self, items: List[ContextItem], source: str = "") -> str:
        """将上下文项列表格式化为字符串

        Args:
            items: 上下文项列表
            source: 来源标识（可选，用于添加标题）

        Returns:
            格式化后的字符串
        """
        ...

    def format_all(self, items_by_source: dict) -> str:
        """按来源分组格式化

        Args:
            items_by_source: {source_name: [ContextItem, ...]}

        Returns:
            完整格式化字符串
        """
        parts = []
        for source, items in items_by_source.items():
            if items:
                parts.append(self.format(items, source))
        return "\n\n".join(parts)
