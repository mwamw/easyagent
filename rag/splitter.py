"""
文本分割器模块
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Callable
import re
import logging

from .document import Document

logger = logging.getLogger(__name__)


class TextSplitter(ABC):
    """
    文本分割器抽象基类
    
    将长文档分割成较小的块，便于向量化和检索。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        """
        初始化分割器
        
        Args:
            chunk_size: 每个块的最大大小
            chunk_overlap: 块之间的重叠大小
            length_function: 计算文本长度的函数
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        pass
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档列表
        """
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["chunk_total"] = len(chunks)
                result.append(Document(page_content=chunk, metadata=metadata))
        
        logger.info(f"将 {len(documents)} 个文档分割为 {len(result)} 个块")
        return result
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并分割后的文本块，确保不超过 chunk_size"""
        result = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    result.append(separator.join(current_chunk))
                    # 保留重叠部分
                    while current_length > self.chunk_overlap and current_chunk:
                        removed = current_chunk.pop(0)
                        current_length -= self.length_function(removed) + len(separator)
                
            current_chunk.append(split)
            current_length += split_length + len(separator)
        
        if current_chunk:
            result.append(separator.join(current_chunk))
        
        return result


class CharacterTextSplitter(TextSplitter):
    """
    字符分割器
    
    按固定字符分割文本。
    
    Example:
        >>> splitter = CharacterTextSplitter(separator="\\n\\n", chunk_size=500)
        >>> chunks = splitter.split_text(long_text)
    """
    
    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        keep_separator: bool = False,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separator = separator
        self.keep_separator = keep_separator
    
    def split_text(self, text: str) -> List[str]:
        """按分隔符分割文本"""
        if self.separator:
            if self.keep_separator:
                splits = re.split(f"({re.escape(self.separator)})", text)
                splits = [s for s in splits if s]  # 移除空字符串
            else:
                splits = text.split(self.separator)
        else:
            splits = list(text)
        
        return self._merge_splits(splits, self.separator if self.keep_separator else " ")


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    递归字符分割器
    
    使用多个分隔符递归分割文本，优先使用较大的分隔符。
    这是最常用的分割器，能够更好地保持文本的语义完整性。
    
    Example:
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        >>> chunks = splitter.split_text(long_text)
    """
    
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        keep_separator: bool = True,
    ):
        """
        初始化递归分割器
        
        Args:
            separators: 分隔符列表（按优先级排序）
            chunk_size: 每个块的最大大小
            chunk_overlap: 块之间的重叠大小
            keep_separator: 是否保留分隔符
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        self.keep_separator = keep_separator
    
    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割实现"""
        final_chunks = []
        
        # 找到合适的分隔符
        separator = separators[-1]  # 默认使用最后一个
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
        
        # 使用选定的分隔符分割
        if separator:
            if self.keep_separator:
                splits = re.split(f"({re.escape(separator)})", text)
                # 将分隔符合并到前一个块
                merged_splits = []
                for i, split in enumerate(splits):
                    if split == separator and merged_splits:
                        merged_splits[-1] += split
                    else:
                        merged_splits.append(split)
                splits = [s for s in merged_splits if s]
            else:
                splits = text.split(separator)
        else:
            splits = list(text)
        
        # 处理每个分割
        good_splits = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # 块太大，需要递归分割
                if good_splits:
                    merged = self._merge_splits(good_splits, separator if self.keep_separator else " ")
                    final_chunks.extend(merged)
                    good_splits = []
                
                # 使用下一个分隔符继续分割
                if separator in separators:
                    idx = separators.index(separator)
                    if idx + 1 < len(separators):
                        other_chunks = self._split_text(split, separators[idx + 1:])
                        final_chunks.extend(other_chunks)
                    else:
                        final_chunks.append(split)
                else:
                    final_chunks.append(split)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator if self.keep_separator else " ")
            final_chunks.extend(merged)
        
        return final_chunks


class TokenTextSplitter(TextSplitter):
    """
    Token 分割器
    
    按 token 数量分割文本（需要 tiktoken 库）。
    """
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("TokenTextSplitter 需要 tiktoken 库。请运行: pip install tiktoken")
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(self.tokenizer.encode(x))
        )
    
    def split_text(self, text: str) -> List[str]:
        """按 token 分割文本"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - self.chunk_overlap
        
        return chunks
