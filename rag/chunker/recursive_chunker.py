"""递归字符文本分块器"""
from typing import List, Optional
import logging

from .base import BaseChunker
from ..document import Document, Document_Chunk

logger = logging.getLogger(__name__)


class RecursiveCharacterChunker(BaseChunker):
    """
    递归字符文本分块器

    按照分隔符优先级递归分割文本，尽量在自然边界（段落、句子等）处分割。

    Example:
        >>> chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.split(document)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " ", ""
        ]

    def split(self, document: Document) -> List[Document_Chunk]:
        text = document.content
        if not text:
            return []

        texts = self._split_text(text, self.separators)
        return [
            self._create_chunk(document, t, i)
            for i, t in enumerate(texts)
            if t.strip()
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本"""
        final_chunks: List[str] = []

        # 找到文本中存在的最佳分隔符
        separator = separators[-1]
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # 按分隔符分割
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 处理每个片段
        good_splits: List[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                # 先合并已有的小片段
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                # 递归处理过大的片段
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        # 合并剩余的小片段
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并小片段为较大的块，处理重叠"""
        merged: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            sep_len = len(separator) if current_parts else 0

            if current_len + split_len + sep_len > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts)
                if chunk_text.strip():
                    merged.append(chunk_text)
                # 处理重叠：保留尾部部分
                while current_len > self.chunk_overlap and len(current_parts) > 1:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + len(separator)

            current_parts.append(split)
            current_len = sum(len(s) for s in current_parts) + len(separator) * max(0, len(current_parts) - 1)

        if current_parts:
            chunk_text = separator.join(current_parts)
            if chunk_text.strip():
                merged.append(chunk_text)

        return merged
