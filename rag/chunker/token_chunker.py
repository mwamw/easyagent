"""基于 Token 的文本分块器"""
from typing import List, Optional
import logging

from .base import BaseChunker
from ..document import Document, Document_Chunk

logger = logging.getLogger(__name__)


class TokenChunker(BaseChunker):
    """
    基于 Token 的文本分块器

    使用 tiktoken 按 token 数量分割文本，适用于需要精确控制 token 数量的场景。

    Args:
        chunk_size: 每个块的最大 token 数
        chunk_overlap: 相邻块的重叠 token 数
        encoding_name: tiktoken 编码名称

    Example:
        >>> chunker = TokenChunker(chunk_size=256, chunk_overlap=32)
        >>> chunks = chunker.split(document)
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding_name)

    def split(self, document: Document) -> List[Document_Chunk]:
        text = document.content
        if not text:
            return []

        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(self._create_chunk(
                    document, chunk_text, chunk_index,
                    extra_metadata={"token_count": len(chunk_tokens)},
                ))
                chunk_index += 1

            step = self.chunk_size - self.chunk_overlap
            start += step

        return chunks
