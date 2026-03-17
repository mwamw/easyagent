"""重排序检索器（高级）"""
import re
from typing import List, Any, Optional
import logging

from .base import BaseRetriever
from ..document import Document_Chunk

logger = logging.getLogger(__name__)


class ReRankRetriever(BaseRetriever):
    """
    重排序检索器

    先使用基础检索器获取候选文档，再使用 LLM 对候选文档进行相关性评分并重排序。
    能够显著提高检索精度。

    Args:
        base_retriever: 基础检索器
        llm: LLM 实例（需实现 invoke 方法）
        top_k: 最终返回的文档数
        initial_k: 初始检索的候选文档数

    Example:
        >>> reranker = ReRankRetriever(
        ...     base_retriever=vector_retriever,
        ...     llm=llm,
        ...     top_k=5,
        ...     initial_k=20,
        ... )
        >>> results = reranker.retrieve("精确查询")
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Any,
        top_k: int = 4,
        initial_k: int = 20,
    ):
        self.base_retriever = base_retriever
        self.llm = llm
        self.top_k = top_k
        self.initial_k = initial_k

    def retrieve(self, query: str, k: int = None) -> List[Document_Chunk]:
        k = k or self.top_k
        candidates = self.base_retriever.retrieve(query, k=self.initial_k)

        if not candidates or len(candidates) <= k:
            return candidates

        scored = []
        for chunk in candidates:
            score = self._score_with_llm(query, chunk.content)
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:k]]

    def _score_with_llm(self, query: str, document: str) -> int:
        """使用 LLM 评估文档与查询的相关性"""
        prompt = (
            "请评估以下文档与查询的相关性，返回 0-10 的整数分数（仅返回数字）。\n\n"
            f"查询：{query}\n"
            f"文档：{document[:500]}\n\n"
            "相关性分数："
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            match = re.search(r'\d+', response.strip())
            return min(int(match.group()), 10) if match else 0
        except Exception as e:
            logger.warning(f"LLM 评分失败: {e}")
            return 0
