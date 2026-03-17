"""多查询检索器（高级）"""
from typing import List, Any, Optional
import logging

from .base import BaseRetriever
from ..document import Document_Chunk
logger = logging.getLogger(__name__)


class MultiQueryRetriever(BaseRetriever):
    """
    多查询检索器

    使用 LLM 从不同角度生成多个查询变体，分别检索后合并去重。
    能够覆盖更多相关文档，提高召回率。

    Args:
        base_retriever: 基础检索器
        llm: LLM 实例（需实现 invoke 方法）
        num_queries: 生成的额外查询数量

    Example:
        >>> multi = MultiQueryRetriever(
        ...     base_retriever=vector_retriever,
        ...     llm=llm,
        ...     num_queries=3,
        ... )
        >>> results = multi.retrieve("什么是 RAG?")
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Any,
        num_queries: int = 3,
    ):
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries

    def _generate_queries(self, query: str) -> List[str]:
        """使用 LLM 生成多角度查询"""
        prompt = (
            f"请根据以下用户查询，从不同角度生成 {self.num_queries} 个相关查询，"
            f"用于更全面地检索信息。每个查询占一行，不需要编号。\n\n"
            f"用户查询：{query}"
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            return [query] + queries[:self.num_queries]
        except Exception as e:
            logger.warning(f"生成多查询失败: {e}")
            return [query]

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document_Chunk]:
        k = k or 4
        queries = self._generate_queries(query)

        all_chunks: dict = {}
        for q in queries:
            results = self.base_retriever.retrieve(q, k=k)
            for chunk in results:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk

        return list(all_chunks.values())[:k * 2]
