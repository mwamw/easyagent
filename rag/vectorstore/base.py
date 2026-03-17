"""向量存储基类"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from ..document import Document_Chunk


class BaseVectorStore(ABC):
    """
    向量存储抽象基类

    所有向量存储后端都应继承此类。

    Example:
        >>> class MyStore(BaseVectorStore):
        ...     def add_documents(self, chunks, embeddings): ...
        ...     def similarity_search(self, query_embedding, k=4): ...
    """

    @abstractmethod
    def add_documents(
        self,
        chunks: List[Document_Chunk],
        embeddings: List[List[float]],
    ):
        """添加文档块及其嵌入向量"""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document_Chunk]:
        """相似度搜索，返回最相关的文档块"""
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document_Chunk, float]]:
        """相似度搜索，返回文档块及相似度分数（分数越高越相关）"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """删除指定 ID 的文档块"""
        pass

    @abstractmethod
    def clear(self):
        """清空所有数据"""
        pass

    def count(self) -> int:
        """返回存储的文档块数量"""
        return 0
