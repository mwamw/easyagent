"""
检索器模块
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

from .document import Document
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    检索器抽象基类
    
    用于从文档集合中检索相关文档。
    """
    
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        pass
    
    def __call__(self, query: str) -> List[Document]:
        """允许直接调用检索器"""
        return self.get_relevant_documents(query)


class VectorStoreRetriever(BaseRetriever):
    """
    向量存储检索器
    
    基于向量相似度从 VectorStore 中检索文档。
    
    Example:
        >>> store = ChromaVectorStore(collection_name="docs")
        >>> retriever = VectorStoreRetriever(vectorstore=store, k=5)
        >>> docs = retriever.get_relevant_documents("查询内容")
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        k: int = 4,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
    ):
        """
        初始化向量存储检索器
        
        Args:
            vectorstore: 向量存储实例
            k: 返回的最大文档数
            search_type: 搜索类型 ("similarity" 或 "similarity_score_threshold")
            score_threshold: 分数阈值（仅在 search_type="similarity_score_threshold" 时使用）
            filter: 元数据过滤条件
        """
        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.filter = filter
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        if self.search_type == "similarity_score_threshold" and self.score_threshold is not None:
            doc_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.k, filter=self.filter
            )
            return [doc for doc, score in doc_scores if score <= self.score_threshold]
        else:
            return self.vectorstore.similarity_search(
                query, k=self.k, filter=self.filter
            )


class ContextualCompressionRetriever(BaseRetriever):
    """
    上下文压缩检索器
    
    先检索文档，然后使用 LLM 压缩/过滤不相关的内容。
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Any,
        compression_prompt: Optional[str] = None
    ):
        """
        初始化上下文压缩检索器
        
        Args:
            base_retriever: 基础检索器
            llm: LLM 实例
            compression_prompt: 压缩提示词
        """
        self.base_retriever = base_retriever
        self.llm = llm
        self.compression_prompt = compression_prompt or """
请根据用户查询，从以下文档中提取最相关的内容：

用户查询：{query}

文档内容：
{documents}

请只返回与查询直接相关的内容，删除不相关的部分。保持原始文档的准确性。
"""
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取并压缩相关文档"""
        # 获取原始文档
        docs = self.base_retriever.get_relevant_documents(query)
        
        if not docs:
            return []
        
        # 压缩每个文档
        compressed_docs = []
        for doc in docs:
            prompt = self.compression_prompt.format(
                query=query,
                documents=doc.page_content
            )
            
            try:
                compressed_content = self.llm.invoke([{"role": "user", "content": prompt}])
                if compressed_content and compressed_content.strip():
                    compressed_docs.append(Document(
                        page_content=compressed_content,
                        metadata=doc.metadata
                    ))
            except Exception as e:
                logger.warning(f"压缩文档失败: {e}")
                compressed_docs.append(doc)  # 保留原文档
        
        return compressed_docs


class MultiQueryRetriever(BaseRetriever):
    """
    多查询检索器
    
    使用 LLM 生成多个相关查询，然后合并检索结果。
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Any,
        num_queries: int = 3,
    ):
        """
        初始化多查询检索器
        
        Args:
            base_retriever: 基础检索器
            llm: LLM 实例
            num_queries: 生成的查询数量
        """
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries
    
    def _generate_queries(self, query: str) -> List[str]:
        """使用 LLM 生成多个查询"""
        prompt = f"""请根据以下用户查询，生成 {self.num_queries} 个不同角度的相关查询，用于更全面地检索相关信息。

用户查询：{query}

请用 JSON 数组格式返回查询列表，例如：["查询1", "查询2", "查询3"]
"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            import json
            queries = json.loads(response)
            return [query] + queries  # 保留原始查询
        except Exception as e:
            logger.warning(f"生成多查询失败: {e}")
            return [query]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """使用多个查询检索并合并结果"""
        queries = self._generate_queries(query)
        
        all_docs = []
        seen_contents = set()
        
        for q in queries:
            docs = self.base_retriever.get_relevant_documents(q)
            for doc in docs:
                # 去重
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        return all_docs
