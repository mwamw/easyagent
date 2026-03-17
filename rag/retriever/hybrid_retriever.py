"""混合检索器（高级）"""
from typing import List, Optional, Dict
import logging

from .base import BaseRetriever
from ..document import Document_Chunk

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    混合检索器

    结合向量检索和 BM25 关键词检索，使用 Reciprocal Rank Fusion (RRF) 合并结果。
    融合了语义理解和关键词匹配的优势，通常比单一检索方式效果更好。

    Args:
        vector_retriever: 向量检索器实例
        bm25_retriever: BM25 检索器实例
        vector_weight: 向量检索权重
        bm25_weight: BM25 检索权重
        k: 默认返回的文档块数
        rrf_k: RRF 常数（默认 60）

    Example:
        >>> hybrid = HybridRetriever(
        ...     vector_retriever=vec_retriever,
        ...     bm25_retriever=bm25_retriever,
        ...     vector_weight=0.6,
        ...     bm25_weight=0.4,
        ... )
        >>> results = hybrid.retrieve("混合查询")
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        k: int = 4,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = None) -> List[Document_Chunk]:
        k = k or self.k
        fetch_k = k * 3  # 多取一些用于融合

        vector_results = self.vector_retriever.retrieve(query, k=fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, k=fetch_k)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        all_chunks: Dict[str, Document_Chunk] = {}

        for rank, chunk in enumerate(vector_results):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.vector_weight / (self.rrf_k + rank + 1)
            all_chunks[cid] = chunk

        for rank, chunk in enumerate(bm25_results):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.bm25_weight / (self.rrf_k + rank + 1)
            all_chunks[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [all_chunks[cid] for cid in sorted_ids]
