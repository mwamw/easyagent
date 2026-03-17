"""向量检索器"""
from typing import List, Optional, Dict, Any
import logging

from .base import BaseRetriever
from ..document import Document_Chunk

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """
    向量相似度检索器

    基于嵌入向量的余弦相似度进行检索。

    Args:
        vectorstore: 向量存储实例
        embedding: 嵌入模型实例
        k: 默认返回的文档块数
        score_threshold: 相似度阈值（低于此值的结果将被过滤）

    Example:
        >>> retriever = VectorRetriever(
        ...     vectorstore=store,
        ...     embedding=embedding,
        ...     k=5,
        ...     score_threshold=0.7,
        ... )
        >>> chunks = retriever.retrieve("查询内容")
    """

    def __init__(
        self,
        vectorstore,
        embedding,
        k: int = 4,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ):
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.k = k
        self.score_threshold = score_threshold
        self.filter = filter

    def retrieve(self, query: str, k: int = None) -> List[Document_Chunk]:
        k = k or self.k
        query_embedding = self.embedding.embed_query(query)

        if self.score_threshold is not None:
            results = self.vectorstore.similarity_search_with_score(
                query_embedding, k=k, filter=self.filter,
            )
            return [chunk for chunk, score in results if score >= self.score_threshold]

        return self.vectorstore.similarity_search(
            query_embedding, k=k, filter=self.filter,
        )
