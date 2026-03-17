"""检索器基类"""
from abc import ABC, abstractmethod
from typing import List

from ..document import Document_Chunk


class BaseRetriever(ABC):
    """
    检索器抽象基类

    所有检索策略都应继承此类并实现 retrieve 方法。

    Example:
        >>> class MyRetriever(BaseRetriever):
        ...     def retrieve(self, query, k=4):
        ...         return []
        >>> retriever = MyRetriever()
        >>> docs = retriever("查询内容")
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[Document_Chunk]:
        """
        检索与查询相关的文档块

        Args:
            query: 查询文本
            k: 返回的最大文档块数

        Returns:
            相关文档块列表
        """
        pass

    def __call__(self, query: str, k: int = 4) -> List[Document_Chunk]:
        """允许直接调用检索器"""
        return self.retrieve(query, k)
