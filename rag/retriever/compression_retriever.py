"""上下文压缩检索器（高级）"""
from typing import List, Any,Optional
import logging

from .base import BaseRetriever
from ..document import Document_Chunk
from core.llm import EasyLLM
logger = logging.getLogger(__name__)


class CompressionRetriever(BaseRetriever):
    """
    上下文压缩检索器

    先检索文档，然后使用 LLM 提取与查询最相关的内容，移除不相关部分。
    能够减少噪声，提高生成质量。

    Args:
        base_retriever: 基础检索器
        llm: LLM 实例（需实现 invoke 方法）
        k: 默认返回的文档块数

    Example:
        >>> compressor = CompressionRetriever(
        ...     base_retriever=vector_retriever,
        ...     llm=llm,
        ...     k=5,
        ... )
        >>> results = compressor.retrieve("精确查询")
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: EasyLLM,
        k: int = 4,
    ):
        self.base_retriever = base_retriever
        self.llm = llm
        self.k = k

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document_Chunk]:
        k = k or self.k
        docs = self.base_retriever.retrieve(query, k=k)

        compressed = []
        for doc in docs:
            result = self._compress(query, doc)
            if result:
                compressed.append(result)

        return compressed

    def _compress(self, query: str, chunk: Document_Chunk):
        """使用 LLM 压缩文档内容"""
        prompt = (
            "请从以下文档中提取与查询最相关的内容。"
            "如果文档完全不相关，请只返回'无关'。\n\n"
            f"查询：{query}\n"
            f"文档：{chunk.content}\n\n"
            "相关内容："
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            content = response.strip()

            if "无关" == content or not content:
                return None

            return Document_Chunk(
                document_id=chunk.document_id,
                document_path=chunk.document_path,
                chunk_id=chunk.chunk_id,
                content=content,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            )
        except Exception as e:
            logger.warning(f"压缩文档失败: {e}")
            return chunk
