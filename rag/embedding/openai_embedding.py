"""OpenAI 嵌入模型"""
from typing import List, Optional
import logging

from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI 嵌入模型

    支持 OpenAI 及兼容 API（如 DeepSeek、智谱等）。

    Args:
        model: 模型名称
        api_key: API 密钥（默认从环境变量 OPENAI_API_KEY 读取）
        base_url: API 地址（默认从环境变量 OPENAI_BASE_URL 读取）
        batch_size: 批量嵌入的大小

    Example:
        >>> embedding = OpenAIEmbedding(model="text-embedding-3-small")
        >>> vectors = embedding.embed_documents(["Hello", "World"])
        >>> query_vec = embedding.embed_query("Hi")
    """

    DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 128,
    ):
        from openai import OpenAI
        import os

        self.model = model
        self.batch_size = batch_size
        self._dimension: Optional[int] = self.DIMENSION_MAP.get(model)

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # 过滤空文本
            batch = [t if t.strip() else " " for t in batch]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # 自动检测维度
            if self._dimension is None and batch_embeddings:
                self._dimension = len(batch_embeddings[0])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.embed_documents([text])
        return result[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # 触发一次嵌入来检测维度
            self.embed_query("test")
        return self._dimension or 1536
