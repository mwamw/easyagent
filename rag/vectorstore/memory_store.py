"""内存向量存储"""
from typing import List, Dict, Any, Optional, Tuple
import logging

from .base import BaseVectorStore
from ..document import Document_Chunk

logger = logging.getLogger(__name__)


class MemoryVectorStore(BaseVectorStore):
    """
    内存向量存储

    使用 numpy 进行余弦相似度计算，数据存储在内存中。
    适用于原型开发、测试和小数据集场景。

    Example:
        >>> store = MemoryVectorStore()
        >>> store.add_documents(chunks, embeddings)
        >>> results = store.similarity_search(query_embedding, k=5)
    """

    def __init__(self):
        self._chunks: Dict[str, Document_Chunk] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def add_documents(
        self,
        chunks: List[Document_Chunk],
        embeddings: List[List[float]],
    ):
        for chunk, emb in zip(chunks, embeddings):
            self._chunks[chunk.chunk_id] = chunk
            self._embeddings[chunk.chunk_id] = emb
        logger.debug(f"添加 {len(chunks)} 个文档块，当前共 {len(self._chunks)} 个")

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document_Chunk]:
        results = self.similarity_search_with_score(query_embedding, k, filter)
        return [chunk for chunk, _ in results]

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document_Chunk, float]]:
        import numpy as np

        if not self._chunks:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scores: List[Tuple[Document_Chunk, float]] = []

        for chunk_id, emb in self._embeddings.items():
            chunk = self._chunks[chunk_id]

            # 元数据过滤
            if filter:
                if not all(chunk.metadata.get(key) == val for key, val in filter.items()):
                    continue

            emb_arr = np.array(emb)
            emb_norm = np.linalg.norm(emb_arr)
            if emb_norm == 0:
                continue

            similarity = float(np.dot(query, emb_arr) / (query_norm * emb_norm))
            scores.append((chunk, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def delete(self, ids: List[str]):
        for id_ in ids:
            self._chunks.pop(id_, None)
            self._embeddings.pop(id_, None)

    def clear(self):
        self._chunks.clear()
        self._embeddings.clear()

    def count(self) -> int:
        return len(self._chunks)
