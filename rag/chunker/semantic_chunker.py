"""语义文本分块器（高级）"""
import re
import logging
from typing import List, Optional

from .base import BaseChunker
from ..document import Document, Document_Chunk
from ..embedding import BaseEmbedding
logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    语义文本分块器

    使用嵌入模型计算句子间的语义相似度，在语义断裂处进行分割。
    相比固定大小分块，能更好地保持语义完整性。

    Args:
        embedding: 嵌入模型实例（需实现 embed_documents 方法）
        breakpoint_threshold_type: 断点阈值类型
            - "percentile": 百分位数（默认）
            - "standard_deviation": 标准差
            - "interquartile": 四分位距
        breakpoint_threshold_amount: 断点阈值参数
        min_chunk_size: 最小块大小（字符数），过小的块会与相邻块合并

    Example:
        >>> from rag.embedding import OpenAIEmbedding
        >>> embedding = OpenAIEmbedding()
        >>> chunker = SemanticChunker(embedding, breakpoint_threshold_type="percentile")
        >>> chunks = chunker.split(document)
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        min_chunk_size: int = 100,
    ):
        self.embedding = embedding
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size

    def split(self, document: Document) -> List[Document_Chunk]:
        text = document.content
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [self._create_chunk(document, text, 0)]

        # 计算句子嵌入
        embeddings = self.embedding.embed_documents(sentences)

        # 计算相邻句子间的语义距离
        distances = self._calculate_distances(embeddings)

        # 找到语义断点
        breakpoints = self._find_breakpoints(distances)

        # 按断点分组句子
        return self._group_sentences(document, sentences, breakpoints)

    def _split_sentences(self, text: str) -> List[str]:
        """分割文本为句子"""
        sentences = re.split(r'(?<=[。.!?！？\n])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_distances(self, embeddings: List[List[float]]) -> List[float]:
        """计算相邻句子嵌入的余弦距离"""
        import numpy as np

        distances = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                distances.append(1.0)
            else:
                similarity = np.dot(a, b) / (norm_a * norm_b)
                distances.append(1.0 - float(similarity))
        return distances

    def _find_breakpoints(self, distances: List[float]) -> List[int]:
        """根据阈值策略找到语义断点"""
        import numpy as np

        if not distances:
            return []

        arr = np.array(distances)

        if self.breakpoint_threshold_type == "percentile":
            threshold = float(np.percentile(arr, self.breakpoint_threshold_amount))
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = float(np.mean(arr) + self.breakpoint_threshold_amount * np.std(arr))
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            threshold = float(q3 + self.breakpoint_threshold_amount * iqr)
        else:
            threshold = float(np.percentile(arr, 95))

        # 断点索引：距离超过阈值的位置（表示下一个句子是新块的开始）
        return [i + 1 for i, d in enumerate(distances) if d > threshold]

    def _group_sentences(
        self,
        document: Document,
        sentences: List[str],
        breakpoints: List[int],
    ) -> List[Document_Chunk]:
        """按断点分组句子为文档块"""
        chunks = []
        start = 0
        breakpoints = sorted(set(breakpoints + [len(sentences)]))

        for bp in breakpoints:
            chunk_text = " ".join(sentences[start:bp])
            if not chunk_text.strip():
                start = bp
                continue

            if len(chunk_text) < self.min_chunk_size and chunks:
                # 过小的块与前一个块合并
                prev = chunks[-1]
                chunks[-1] = self._create_chunk(
                    document,
                    prev.content + " " + chunk_text,
                    prev.chunk_index,
                )
            else:
                chunks.append(self._create_chunk(document, chunk_text, len(chunks)))
            start = bp

        return chunks
