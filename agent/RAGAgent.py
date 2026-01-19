"""
RAG Agent

检索增强生成 Agent，结合向量检索和 LLM 生成。
"""
from typing import Optional, List, Any
from typing_extensions import override
import logging

from .BasicAgent import BasicAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from rag.retriever import BaseRetriever
from rag.document import Document
from core.Exception import *

logger = logging.getLogger(__name__)


class RAGAgent(BasicAgent):
    """
    RAG (检索增强生成) Agent
    
    结合向量检索和 LLM，先从知识库检索相关文档，
    然后基于检索结果生成回答。
    
    特点：
    - 支持多种检索器
    - 可配置检索数量和相似度阈值
    - 支持文档引用和来源标注
    
    Example:
        >>> from rag import ChromaVectorStore, VectorStoreRetriever
        >>> 
        >>> store = ChromaVectorStore(collection_name="knowledge")
        >>> retriever = VectorStoreRetriever(vectorstore=store, k=5)
        >>> 
        >>> agent = RAGAgent(
        ...     name="rag_agent",
        ...     llm=llm,
        ...     retriever=retriever
        ... )
        >>> answer = agent.invoke("什么是机器学习？")
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        retriever: BaseRetriever,
        system_prompt: Optional[str] = None,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        top_k: int = 5,
        include_sources: bool = True,
        context_max_length: int = 3000,
    ):
        """
        初始化 RAG Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            retriever: 检索器实例
            system_prompt: 系统提示词
            enable_tool: 是否启用工具（可与 RAG 结合使用）
            tool_registry: 工具注册表
            description: Agent 描述
            config: 配置
            top_k: 检索文档数量
            include_sources: 是否在回答中包含来源引用
            context_max_length: 上下文最大长度（字符数）
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            enable_tool=enable_tool,
            tool_registry=tool_registry,
            description=description,
            config=config
        )
        
        self.retriever = retriever
        self.top_k = top_k
        self.include_sources = include_sources
        self.context_max_length = context_max_length
        self._last_retrieved_docs: List[Document] = []
    
    @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """执行 RAG 问答"""
        self._validate_invoke_params(query, max_iter, temperature)
        
        logger.info(f"RAG 查询: {query[:50]}...")
        
        # 1. 检索相关文档
        docs = self._retrieve_documents(query)
        self._last_retrieved_docs = docs
        
        if not docs:
            logger.warning("未检索到相关文档")
            # 没有检索到文档时，直接使用 LLM 回答
            return self._answer_without_context(query, temperature, **kwargs)
        
        logger.info(f"检索到 {len(docs)} 个相关文档")
        
        # 2. 构建上下文
        context = self._build_context(docs)
        
        # 3. 生成回答
        if self.enable_tool and self.tool_registry:
            answer = self._answer_with_tools(query, context, max_iter, temperature)
        else:
            answer = self._answer_with_context(query, context, temperature, **kwargs)
        
        # 4. 添加来源引用
        if self.include_sources:
            answer = self._add_sources(answer, docs)
        
        # 保存历史
        self.history.append(UserMessage(query))
        self.history.append(AssistantMessage(answer))
        
        return answer
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        try:
            docs = self.retriever.get_relevant_documents(query)
            return docs[:self.top_k] if len(docs) > self.top_k else docs
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return []
    
    def _build_context(self, docs: List[Document]) -> str:
        """构建上下文"""
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(docs):
            doc_text = f"[文档 {i+1}]\n{doc.page_content}"
            
            # 检查长度限制
            if total_length + len(doc_text) > self.context_max_length:
                # 截断当前文档
                remaining = self.context_max_length - total_length
                if remaining > 100:
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _answer_with_context(
        self, 
        query: str, 
        context: str, 
        temperature: float,
        **kwargs
    ) -> str:
        """基于上下文生成回答"""
        system_content = self._build_rag_prompt()
        
        user_content = f"""## 参考资料
{context}

## 用户问题
{query}

请根据参考资料回答用户问题。如果参考资料中没有相关信息，请明确说明。"""
        
        messages = [
            SystemMessage(system_content),
            UserMessage(user_content)
        ]
        
        try:
            response = self.llm.invoke(messages, temperature=temperature, **kwargs)
            return str(response) if not isinstance(response, str) else response
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            raise LLMInvokeError(f"RAG 回答生成失败: {e}") from e
    
    def _answer_without_context(
        self, 
        query: str, 
        temperature: float,
        **kwargs
    ) -> str:
        """在没有检索结果时回答"""
        messages = [
            SystemMessage(self.system_prompt or "你是一个有用的助手。"),
            UserMessage(f"抱歉，我无法从知识库中找到与 '{query}' 相关的信息。请问你还有其他问题吗？（注：我会尽量基于我的知识回答，但可能不完全准确）")
        ]
        
        return self.llm.invoke(messages, temperature=temperature, **kwargs)
    
    def _answer_with_tools(
        self, 
        query: str, 
        context: str, 
        max_iter: int, 
        temperature: float
    ) -> str:
        """结合 RAG 和工具调用"""
        messages: list = []
        
        # 系统提示包含工具信息
        system_content = self._build_rag_prompt()
        if self.tool_registry:
            tools_desc = self.tool_registry.get_tools_description()
            system_content += f"\n\n## 可用工具\n{tools_desc}"
        
        messages.append(SystemMessage(system_content))
        
        # 用户消息包含上下文
        user_content = f"""## 参考资料
{context}

## 用户问题
{query}

请根据参考资料回答。如果需要更多信息，可以使用工具查询。"""
        
        messages.append(UserMessage(user_content))
        
        # 使用父类的工具调用逻辑
        return super().invoke_with_tool(query, messages, max_iter, temperature)
    
    def _build_rag_prompt(self) -> str:
        """构建 RAG 系统提示词"""
        base_prompt = self.system_prompt or "你是一个知识问答助手。"
        
        return f"""{base_prompt}

## 回答要求
1. 基于参考资料中的信息回答问题
2. 如果参考资料中没有相关信息，请明确说明
3. 保持回答的准确性，不要编造不存在的信息
4. 可以适当整合多个文档的信息
5. 如果需要，可以引用具体的文档编号"""
    
    def _add_sources(self, answer: str, docs: List[Document]) -> str:
        """添加来源引用"""
        sources = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", doc.metadata.get("filename", f"文档{i+1}"))
            if source and source not in sources:
                sources.append(source)
        
        if sources:
            sources_text = "\n".join([f"- {s}" for s in sources[:5]])
            return f"{answer}\n\n---\n**参考来源：**\n{sources_text}"
        
        return answer
    
    def get_last_retrieved_docs(self) -> List[Document]:
        """获取最后一次检索的文档"""
        return self._last_retrieved_docs.copy()
    
    def set_retriever(self, retriever: BaseRetriever) -> None:
        """设置检索器"""
        self.retriever = retriever
