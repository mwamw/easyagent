"""固定大小文本分块器"""
from typing import List
import logging

from .base import BaseChunker
from ..document import Document, Document_Chunk

logger = logging.getLogger(__name__)


class FixedChunker(BaseChunker):
    """
    固定大小文本分块器

    按固定字符数将文本分割成块，支持重叠。

    Example:
        >>> chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.split(document)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Document_Chunk]:
        text = document.content
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(self._create_chunk(document, chunk_text, chunk_index))
                chunk_index += 1

            start += self.chunk_size - self.chunk_overlap

        return chunks
