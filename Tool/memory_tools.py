"""
Memory 和 RAG 工具封装

将 Memory 和 RAG 检索封装成工具，让 LLM 可以通过 Tool Calling 自主调用。
"""
from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, Field
from Tool.BaseTool import Tool

if TYPE_CHECKING:
    from memory.base import BaseMemory
    from rag.retriever import BaseRetriever


# ============ Memory 工具 ============

class MemorySearchParams(BaseModel):
    """记忆搜索参数"""
    query: str = Field(description="搜索查询，用于在记忆中查找相关内容")
    top_k: int = Field(default=5, description="返回结果数量")


class MemorySaveParams(BaseModel):
    """记忆保存参数"""
    content: str = Field(description="要保存到记忆中的内容")
    metadata: Optional[dict] = Field(default=None, description="附加元数据")


class MemorySearchTool(Tool):
    """
    记忆搜索工具
    
    让 LLM 可以搜索对话历史和长期记忆。
    
    Example:
        >>> memory = ConversationBufferMemory()
        >>> tool = MemorySearchTool(memory)
        >>> registry.registerTool(tool)
    """
    
    def __init__(self, memory: "BaseMemory"):
        super().__init__(
            name="search_memory",
            description="在对话历史和记忆中搜索相关信息。当需要回忆之前讨论过的内容时使用。",
            parameters=MemorySearchParams
        )
        self.memory = memory
    
    def run(self, parameters: dict) -> str:
        query = parameters.get("query", "")
        top_k = parameters.get("top_k", 5)
        
        # 检查是否是向量记忆
        if hasattr(self.memory, 'search'):
            # VectorMemory
            results = self.memory.search(query, top_k=top_k)
            if results:
                formatted = []
                for i, r in enumerate(results):
                    formatted.append(f"[{i+1}] {r.get('role', 'unknown')}: {r.get('content', '')}")
                return "\n".join(formatted)
            return "未找到相关记忆"
        else:
            # BufferMemory
            context = self.memory.get_context()
            if context:
                return f"对话历史:\n{context}"
            return "记忆为空"


class MemorySaveTool(Tool):
    """
    记忆保存工具
    
    让 LLM 可以主动保存重要信息到长期记忆。
    """
    
    def __init__(self, memory: "BaseMemory"):
        super().__init__(
            name="save_to_memory",
            description="将重要信息保存到长期记忆中。当用户分享重要个人信息或偏好时使用。",
            parameters=MemorySaveParams
        )
        self.memory = memory
    
    def run(self, parameters: dict) -> str:
        content = parameters.get("content", "")
        
        if not content:
            return "错误：内容不能为空"
        
        try:
            self.memory.add_assistant_message(f"[记忆保存] {content}")
            return f"已保存到记忆: {content[:50]}..."
        except Exception as e:
            return f"保存失败: {e}"


# ============ RAG 工具 ============

class RAGSearchParams(BaseModel):
    """RAG 检索参数"""
    query: str = Field(description="检索查询，用于在知识库中搜索相关文档")
    top_k: int = Field(default=5, description="返回文档数量")


class RAGSearchTool(Tool):
    """
    RAG 检索工具
    
    让 LLM 可以从知识库中检索相关文档。
    
    Example:
        >>> retriever = VectorStoreRetriever(vectorstore)
        >>> tool = RAGSearchTool(retriever, name="search_knowledge")
        >>> registry.registerTool(tool)
    """
    
    def __init__(
        self, 
        retriever: "BaseRetriever",
        name: str = "search_knowledge",
        description: str = "在知识库中搜索相关信息。当需要查找专业知识或文档内容时使用。"
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=RAGSearchParams
        )
        self.retriever = retriever
    
    def run(self, parameters: dict) -> str:
        query = parameters.get("query", "")
        top_k = parameters.get("top_k", 5)
        
        if not query:
            return "错误：查询不能为空"
        
        try:
            # 调整检索器的 k 值（如果支持）
            if hasattr(self.retriever, 'k'):
                original_k = self.retriever.k
                self.retriever.k = top_k
            
            docs = self.retriever.get_relevant_documents(query)
            
            # 恢复原始 k 值
            if hasattr(self.retriever, 'k'):
                self.retriever.k = original_k
            
            if not docs:
                return f"未找到与 '{query}' 相关的文档"
            
            # 格式化结果
            results = []
            for i, doc in enumerate(docs[:top_k]):
                source = doc.metadata.get("source", "未知来源")
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                results.append(f"[文档 {i+1}] 来源: {source}\n{content}")
            
            return "\n\n".join(results)
        
        except Exception as e:
            return f"检索失败: {e}"


# ============ 工具注册辅助函数 ============

def register_memory_tools(registry, memory: "BaseMemory", include_save: bool = True):
    """
    向工具注册表注册记忆工具
    
    Args:
        registry: ToolRegistry 实例
        memory: Memory 实例
        include_save: 是否包含保存工具
    """
    registry.registerTool(MemorySearchTool(memory))
    if include_save:
        registry.registerTool(MemorySaveTool(memory))


def register_rag_tool(
    registry, 
    retriever: "BaseRetriever",
    name: str = "search_knowledge",
    description: Optional[str] = None
):
    """
    向工具注册表注册 RAG 检索工具
    
    Args:
        registry: ToolRegistry 实例
        retriever: Retriever 实例
        name: 工具名称
        description: 工具描述
    """
    tool = RAGSearchTool(
        retriever=retriever,
        name=name,
        description=description or f"在知识库中搜索相关信息（{name}）"
    )
    registry.registerTool(tool)
