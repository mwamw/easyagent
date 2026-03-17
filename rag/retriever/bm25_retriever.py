"""BM25 关键词检索器（高级）"""
from typing import List, Optional
import logging

from .base import BaseRetriever
from ..document import Document_Chunk

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """
    BM25 关键词检索器

    基于 BM25 算法进行关键词匹配检索，支持中英文。
    需要安装 rank-bm25: pip install rank-bm25
    中文分词需要安装 jieba: pip install jieba

    Args:
        chunks: 初始文档块列表
        k: 默认返回的文档块数
        language: 语言 ("zh", "en", "auto")

    Example:
        >>> retriever = BM25Retriever(chunks=all_chunks, k=5, language="zh")
        >>> results = retriever.retrieve("关键词查询")
    """

    def __init__(
        self,
        chunks: Optional[List[Document_Chunk]] = None,
        k: int = 4,
        language: str = "auto",
    ):
        self.k = k
        self.language = language
        self._chunks: List[Document_Chunk] = []
        self._bm25 = None

        if chunks:
            self.add_documents(chunks)

    def add_documents(self, chunks: List[Document_Chunk]):
        """添加文档块到检索索引"""
        self._chunks.extend(chunks)
        tokenized_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.debug(f"BM25 索引已更新，共 {len(self._chunks)} 个文档块")

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if self.language == "zh" or (
            self.language == "auto" and self._contains_chinese(text)
        ):
            try:
                import jieba
                return list(jieba.cut(text))
            except ImportError:
                logger.warning("jieba 未安装，使用简单分词。pip install jieba")
        return text.lower().split()

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def retrieve(self, query: str, k: int = None) -> List[Document_Chunk]:
        k = k or self.k
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        return [self._chunks[i] for i in top_indices if scores[i] > 0]
