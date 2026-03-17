"""嵌入模型基类"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """
    嵌入模型抽象基类

    所有嵌入模型都应继承此类并实现以下方法。

    Example:
        >>> class MyEmbedding(BaseEmbedding):
        ...     def embed_documents(self, texts):
        ...         return [[0.1] * 128 for _ in texts]
        ...     def embed_query(self, text):
        ...         return [0.1] * 128
        ...     @property
        ...     def dimension(self):
        ...         return 128
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档文本

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入向量维度"""
        pass
