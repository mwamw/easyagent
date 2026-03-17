"""ChromaDB 向量存储"""
from typing import List, Dict, Any, Optional, Tuple
import logging

from .base import BaseVectorStore
from ..document import Document_Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    基于 ChromaDB 的向量存储

    支持持久化和内存两种模式。

    Args:
        collection_name: 集合名称
        persist_directory: 持久化目录（为 None 时使用内存模式）

    Example:
        >>> # 内存模式
        >>> store = ChromaVectorStore(collection_name="my_docs")
        >>> # 持久化模式
        >>> store = ChromaVectorStore("my_docs", persist_directory="./chroma_db")
    """

    def __init__(
        self,
        collection_name: str = "easyagent_rag",
        persist_directory: Optional[str] = None,
    ):
        import chromadb

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: List[Document_Chunk],
        embeddings: List[List[float]],
    ):
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]

        metadatas = []
        for chunk in chunks:
            meta: Dict[str, Any] = {
                "document_id": chunk.document_id,
                "document_path": chunk.document_path,
                "chunk_index": chunk.chunk_index,
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        # ChromaDB 限制单次添加数量，分批处理
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self._collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        logger.debug(f"添加 {len(chunks)} 个文档块到 ChromaDB")

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document_Chunk]:
        results = self.similarity_search_with_score(query_embedding, k, filter)
        return [chunk for chunk, _ in results]

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document_Chunk, float]]:
        total = self._collection.count()
        if total == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, total),
        }
        if filter:
            kwargs["where"] = filter

        results = self._collection.query(**kwargs)

        chunks_with_scores: List[Tuple[Document_Chunk, float]] = []

        if results and results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                chunk = Document_Chunk(
                    document_id=meta.get("document_id", ""),
                    document_path=meta.get("document_path", ""),
                    chunk_id=id_,
                    content=results["documents"][0][i],
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata=meta,
                )
                # ChromaDB 返回 cosine distance，转换为 similarity
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                similarity = 1.0 - distance
                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def delete(self, ids: List[str]):
        if ids:
            self._collection.delete(ids=ids)

    def clear(self):
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata=metadata
        )

    def count(self) -> int:
        return self._collection.count()
