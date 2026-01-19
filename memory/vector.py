"""
向量记忆模块

支持使用向量数据库进行语义检索的记忆系统。
默认使用 ChromaDB 作为向量存储后端。
"""
from typing import List, Optional, Any, Dict
from .base import BaseMemory
from core.Message import Message, UserMessage, AssistantMessage
import logging
import hashlib

logger = logging.getLogger(__name__)


class VectorMemory(BaseMemory):
    """
    向量记忆
    
    使用向量数据库存储对话历史，支持语义检索相关记忆。
    适用于需要长期记忆和上下文检索的场景。
    
    Attributes:
        collection_name: 向量集合名称
        embedding_model: 嵌入模型
        top_k: 检索时返回的结果数量
    
    Example:
        >>> memory = VectorMemory(collection_name="chat_history")
        >>> memory.add_user_message("我喜欢打篮球")
        >>> memory.add_assistant_message("篮球是一项很好的运动！")
        >>> # 后续对话中检索相关记忆
        >>> context = memory.search("运动爱好")
    """
    
    def __init__(
        self,
        collection_name: str = "conversation_memory",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        top_k: int = 5,
        memory_key: str = "relevant_history"
    ):
        """
        初始化向量记忆
        
        Args:
            collection_name: ChromaDB 集合名称
            persist_directory: 持久化目录（可选，不设置则使用内存存储）
            embedding_function: 自定义嵌入函数（可选）
            top_k: 检索返回的最大结果数
            memory_key: 记忆变量的键名
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.top_k = top_k
        self.memory_key = memory_key
        self._messages: List[Message] = []
        
        # 延迟导入 chromadb
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "VectorMemory 需要 chromadb 库。请运行: pip install chromadb"
            )
        
        # 初始化 ChromaDB 客户端
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        logger.info(f"VectorMemory 初始化完成: collection={collection_name}")
    
    def _generate_id(self, content: str) -> str:
        """生成唯一 ID"""
        return hashlib.md5(f"{content}{len(self._messages)}".encode()).hexdigest()
    
    def add_message(self, message: Message) -> None:
        """添加消息到向量记忆"""
        self._messages.append(message)
        
        # 存储到向量数据库
        doc_id = self._generate_id(message.content)
        self._collection.add(
            documents=[message.content],
            metadatas=[{"role": message.role, "index": len(self._messages) - 1}],
            ids=[doc_id]
        )
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message(UserMessage(content))
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.add_message(AssistantMessage(content))
    
    def get_messages(self) -> List[Message]:
        """获取所有消息"""
        return self._messages.copy()
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        语义搜索相关记忆
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量（默认使用初始化时设置的值）
            
        Returns:
            相关记忆列表，每项包含 content, role, distance
        """
        k = top_k or self.top_k
        
        if self._collection.count() == 0:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count())
        )
        
        memories = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results.get('distances') else None
                memories.append({
                    "content": doc,
                    "role": metadata.get("role", "unknown"),
                    "distance": distance
                })
        
        return memories
    
    def get_context(self, max_tokens: Optional[int] = None, query: Optional[str] = None) -> str:
        """
        获取记忆上下文
        
        如果提供 query，则返回语义相关的记忆；
        否则返回最近的消息。
        
        Args:
            max_tokens: 最大 token 数限制
            query: 搜索查询（可选）
            
        Returns:
            格式化的记忆上下文
        """
        if query:
            memories = self.search(query)
            lines = [f"{m['role']}: {m['content']}" for m in memories]
        else:
            # 返回最近的消息
            recent = self._messages[-self.top_k:] if len(self._messages) > self.top_k else self._messages
            lines = [f"{m.role}: {m.content}" for m in recent]
        
        context = "\n".join(lines)
        
        if max_tokens and len(context) > max_tokens * 2:
            context = context[-(max_tokens * 2):]
        
        return context
    
    def get_memory_variables(self) -> dict:
        """获取记忆变量"""
        return {self.memory_key: self.get_context()}
    
    def clear(self) -> None:
        """清空记忆"""
        self._messages.clear()
        # 删除并重建集合
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)
        logger.info("VectorMemory 已清空")
    
    def __repr__(self) -> str:
        return f"VectorMemory(collection={self.collection_name}, count={self._collection.count()})"
