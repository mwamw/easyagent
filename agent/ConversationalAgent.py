"""
对话记忆 Agent

集成 V2 记忆系统，支持多轮对话的上下文保持。
"""
from typing import Optional, TYPE_CHECKING
from typing_extensions import override
import logging

from .BasicAgent import BasicAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from core.Exception import *

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage

logger = logging.getLogger(__name__)


class ConversationalAgent(BasicAgent):
    """
    对话记忆 Agent
    
    集成 V2 记忆系统 (MemoryManage)，能够记住对话历史并在回答中使用上下文信息。
    适合多轮对话场景。
    
    特点：
    - 自动将每轮对话保存到 Working Memory
    - 通过 get_enhanced_prompt() 自动注入记忆上下文
    - 支持工具调用（包括记忆工具）
    - 后台自动提炼长期记忆
    
    Example:
        >>> from memory import MemoryManage, MemoryConfig, WorkingMemory
        >>> from memory.V2.Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
        >>> 
        >>> config = MemoryConfig()
        >>> embedding = HuggingfaceEmbeddingModel()
        >>> working = WorkingMemory(config, embedding)
        >>> mm = MemoryManage(config, user_id="user1",
        ...     enable_working=True, working_memory=working)
        >>> 
        >>> agent = ConversationalAgent(
        ...     name="chatbot",
        ...     llm=llm,
        ...     memory_manage=mm
        ... )
        >>> agent.invoke("你好，我叫张三")
        >>> agent.invoke("你还记得我的名字吗？")
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        memory_manage: Optional["MemoryManage"] = None,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        auto_save_to_working: bool = True,
    ):
        """
        初始化对话 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            memory_manage: V2 记忆管理器
            enable_tool: 是否启用工具
            tool_registry: 工具注册表
            description: Agent 描述
            config: 配置
            auto_save_to_working: 是否自动将每轮对话保存到 Working Memory
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or "你是一个友好的对话助手，善于记住对话内容并提供连贯的回复。",
            enable_tool=enable_tool,
            tool_registry=tool_registry,
            description=description,
            config=config
        )
        
        self.auto_save_to_working = auto_save_to_working
        
        # 绑定 V2 记忆系统
        if memory_manage is not None:
            self.with_memory(memory_manage)
    
    @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """执行对话
        
        依赖父类 BasicAgent.invoke() 处理 LLM 调用和 tool 循环，
        并在调用前后自动管理记忆。
        """
        self._validate_invoke_params(query, max_iter, temperature)
        
        # 调用父类 invoke（会使用 get_enhanced_prompt() 注入 Working Memory）
        response = super().invoke(query, max_iter, temperature, **kwargs)
        
        # 自动将本轮对话保存到 Working Memory 便签本
        if self.auto_save_to_working and self.memory_manage:
            try:
                turn_content = f"用户: {query}\n助手: {response}"
                self.memory_manage.add_memory(
                    content=turn_content,
                    memory_type="working",
                    importance=0.5,
                    metadata={"source": "conversation_auto_save"}
                )
            except Exception as e:
                logger.warning(f"自动保存对话到 Working Memory 失败: {e}")
        
        return response
    
    @override
    def clear_history(self) -> None:
        """清空对话历史和 Working Memory"""
        super().clear_history()
        if self.memory_manage and "working" in self.memory_manage.memory_types:
            try:
                self.memory_manage.memory_types["working"].clear_memory()
                logger.info("Working Memory 已清空")
            except Exception as e:
                logger.warning(f"清空 Working Memory 失败: {e}")
        logger.info("对话历史和记忆已清空")
    
    def __repr__(self) -> str:
        has_memory = self.memory_manage is not None
        return f"ConversationalAgent(name={self.name}, has_memory={has_memory})"
