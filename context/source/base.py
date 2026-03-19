"""
上下文来源抽象基类
"""
from abc import ABC, abstractmethod
from typing import List
from context.window import ContextItem


class BaseContextSource(ABC):
    """上下文来源基类，所有来源适配器需继承此类"""

    @abstractmethod
    def fetch(self, query: str, max_tokens: int = 0, **kwargs) -> List[ContextItem]:
        """从来源获取上下文项

        Args:
            query: 当前查询
            max_tokens: token 预算上限（0 表示不限）

        Returns:
            上下文项列表
        """
        ...

    @property
    def source_name(self) -> str:
        """来源标识符"""
        return self.__class__.__name__
