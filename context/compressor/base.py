"""
上下文压缩器抽象基类
"""
from abc import ABC, abstractmethod
from typing import List
from context.window import ContextItem


class BaseCompressor(ABC):
    """上下文压缩器基类"""

    @abstractmethod
    def compress(
        self,
        items: List[ContextItem],
        max_tokens: int = 0,
    ) -> List[ContextItem]:
        """压缩上下文项列表

        Args:
            items: 待压缩的上下文项
            max_tokens: token 预算上限（0 表示不限）

        Returns:
            压缩后的上下文项列表
        """
        ...
