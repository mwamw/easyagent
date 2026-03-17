"""HyDE 查询转换器（高级）"""
import logging
from typing import Any

from .base import BaseQueryTransformer

logger = logging.getLogger(__name__)


class HyDETransformer(BaseQueryTransformer):
    """
    HyDE (Hypothetical Document Embeddings) 查询转换器

    使用 LLM 生成一段可能包含答案的假设性文档，然后用该文档
    （而非原始查询）进行嵌入检索。这种方法能显著提高检索效果，
    因为假设性文档与真实文档在嵌入空间中更接近。

    参考论文: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"

    Args:
        llm: LLM 实例（需实现 invoke 方法）

    Example:
        >>> transformer = HyDETransformer(llm)
        >>> hypothetical_doc = transformer.transform("什么是RAG?")
    """

    def __init__(self, llm: Any):
        self.llm = llm

    def transform(self, query: str) -> str:
        prompt = (
            "请针对以下问题，写一段可能包含答案的文档段落。\n"
            "不需要保证准确性，但应包含相关的关键概念和术语。\n\n"
            f"问题：{query}\n\n"
            "假设性文档："
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.strip()
        except Exception as e:
            logger.warning(f"HyDE 转换失败: {e}")
            return query
