"""文本分块器基类"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import uuid
import logging

from ..document import Document, Document_Chunk

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    文本分块器抽象基类

    所有分块策略都应继承此类并实现 split 方法。

    Example:
        >>> class MyChunker(BaseChunker):
        ...     def split(self, document):
        ...         return [self._create_chunk(document, document.content, 0)]
    """

    @abstractmethod
    def split(self, document: Document) -> List[Document_Chunk]:
        """
        将文档分割成多个块

        Args:
            document: 要分割的文档

        Returns:
            文档块列表
        """
        pass

    def split_batch(self, documents: List[Document]) -> List[Document_Chunk]:
        """批量分割文档"""
        chunks = []
        for doc in documents:
            try:
                chunks.extend(self.split(doc))
            except Exception as e:
                logger.warning(f"分割文档失败 {doc.document_path}: {e}")
        return chunks

    def _create_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Document_Chunk:
        """创建文档块的辅助方法"""
        metadata = {**document.metadata}
        if extra_metadata:
            metadata.update(extra_metadata)

        return Document_Chunk(
            document_id=document.document_id or "",
            document_path=document.document_path or "",
            chunk_id=str(uuid.uuid4()),
            content=content,
            chunk_index=chunk_index,
            metadata=metadata,
        )
