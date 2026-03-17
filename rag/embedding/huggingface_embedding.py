"""HuggingFace 嵌入模型"""
from typing import List, Optional
import logging

from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class HuggingFaceEmbedding(BaseEmbedding):
    """
    基于 sentence-transformers 的本地嵌入模型

    Args:
        model_name: 模型名称或路径
        device: 运行设备 ("cpu", "cuda", "mps" 等)
        normalize: 是否归一化嵌入向量

    Example:
        >>> embedding = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
        >>> vectors = embedding.embed_documents(["你好", "世界"])
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
