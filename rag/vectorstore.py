"""
向量存储模块
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Tuple
import logging
import hashlib

from .document import Document

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    向量存储抽象基类
    
    用于存储和检索文档向量。
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档
        
        Args:
            documents: 文档列表
            
        Returns:
            文档 ID 列表
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 分数) 元组列表
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB 向量存储
    
    使用 ChromaDB 作为向量存储后端。
    
    Example:
        >>> store = ChromaVectorStore(collection_name="my_docs")
        >>> store.add_documents(documents)
        >>> results = store.similarity_search("查询内容", k=5)
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ):
        """
        初始化 ChromaDB 向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录（可选）
            embedding_function: 嵌入函数（可选，默认使用 ChromaDB 内置）
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaVectorStore 需要 chromadb 库。请运行: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 初始化客户端
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        logger.info(f"ChromaVectorStore 初始化完成: collection={collection_name}")
    
    def _generate_id(self, content: str, index: int) -> str:
        """生成唯一 ID"""
        return hashlib.md5(f"{content}{index}".encode()).hexdigest()
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量存储"""
        if not documents:
            return []
        
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = self._generate_id(doc.page_content, i)
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        self._collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"添加了 {len(documents)} 个文档到向量存储")
        return ids
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """添加文本到向量存储"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        return self.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """相似度搜索"""
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()) if self._collection.count() > 0 else 1,
            where=filter
        )
        
        documents = []
        if results and results['documents'] and results['documents'][0]:
            for i, text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索"""
        if self._collection.count() == 0:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            where=filter,
            include=["documents", "metadatas", "distances"]
        )
        
        doc_score_pairs = []
        if results and results['documents'] and results['documents'][0]:
            for i, text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results.get('distances') else 0.0
                doc = Document(page_content=text, metadata=metadata)
                doc_score_pairs.append((doc, distance))
        
        return doc_score_pairs
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        self._collection.delete(ids=ids)
        logger.info(f"删除了 {len(ids)} 个文档")
    
    def clear(self) -> None:
        """清空集合"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)
        logger.info("向量存储已清空")
    
    @property
    def count(self) -> int:
        """获取文档数量"""
        return self._collection.count()
    
    def __repr__(self) -> str:
        return f"ChromaVectorStore(collection={self.collection_name}, count={self.count})"
