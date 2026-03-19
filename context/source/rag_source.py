"""
RAG 上下文来源

对接 rag/pipeline.RAGPipeline，将检索结果转为 ContextItem。
"""
from typing import List, Optional, Any
from context.window import ContextItem
from context.source.base import BaseContextSource


class RAGContextSource(BaseContextSource):
    """从 RAG Pipeline 获取检索上下文"""

    def __init__(
        self,
        pipeline: Any = None,
        retriever: Any = None,
        embedding: Any = None,
        k: int = 5,
        base_priority: float = 0.7,
    ):
        """
        Args:
            pipeline: RAGPipeline 实例（优先使用）
            retriever: 独立 Retriever 实例（无 pipeline 时使用）
            embedding: Embedding 实例（配合独立 retriever 使用）
            k: 返回的文档数
            base_priority: 基础优先级
        """
        self.pipeline = pipeline
        self.retriever = retriever
        self.embedding = embedding
        self.k = k
        self.base_priority = base_priority

    def fetch(self, query: str, max_tokens: int = 0, **kwargs) -> List[ContextItem]:
        k = kwargs.get("k", self.k)
        chunks = self._retrieve(query, k)

        if not chunks:
            return []

        items = []
        for i, chunk in enumerate(chunks):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            metadata = {}

            if hasattr(chunk, "metadata") and chunk.metadata:
                metadata.update(chunk.metadata)
            if hasattr(chunk, "document_path"):
                metadata["document_path"] = chunk.document_path
            if hasattr(chunk, "chunk_id"):
                metadata["chunk_id"] = chunk.chunk_id

            # 排名越靠前优先级越高
            priority = self.base_priority + 0.2 * (1.0 - i / max(len(chunks) - 1, 1))

            items.append(ContextItem(
                content=content,
                source="rag",
                priority=min(priority, 1.0),
                metadata=metadata,
            ))

        return items

    def _retrieve(self, query: str, k: int) -> list:
        """执行检索"""
        if self.pipeline is not None:
            # 使用 pipeline 的 retriever
            retriever = getattr(self.pipeline, "retriever", None)
            if retriever:
                return retriever.retrieve(query, k=k)
            return []

        if self.retriever is not None:
            return self.retriever.retrieve(query, k=k)

        return []

    @property
    def source_name(self) -> str:
        return "rag"
