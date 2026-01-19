"""
Agent 基类模块
"""
from .Message import Message
from typing import Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from .Config import Config
from .llm import EasyLLM

if TYPE_CHECKING:
    from memory.base import BaseMemory


class BaseAgent(ABC):
    """
    Agent 抽象基类
    
    所有 Agent 实现都应该继承此类。
    提供可选的记忆系统支持。
    
    Attributes:
        name: Agent 名称
        llm: LLM 实例
        system_prompt: 系统提示词
        description: Agent 描述
        config: 配置
        history: 对话历史（简单列表）
        memory: 记忆系统（可选）
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        memory: Optional["BaseMemory"] = None,
        enable_memory: bool = False,
    ):
        """
        初始化 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            description: Agent 描述
            config: 配置
            memory: 记忆系统实例（可选）
            enable_memory: 是否启用记忆（如果 memory 为 None 则使用默认记忆）
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.description = description
        self.config = config or Config.from_env()
        self.history = []
        
        # 记忆系统
        self.enable_memory = enable_memory or (memory is not None)
        self._memory = memory
        
        if self.enable_memory and self._memory is None:
            # 延迟导入，避免循环依赖
            from memory.buffer import ConversationBufferMemory
            self._memory = ConversationBufferMemory(max_messages=20)
    
    @property
    def memory(self) -> Optional["BaseMemory"]:
        """获取记忆系统"""
        return self._memory
    
    @memory.setter
    def memory(self, value: Optional["BaseMemory"]) -> None:
        """设置记忆系统"""
        self._memory = value
        self.enable_memory = value is not None
    
    @abstractmethod
    def invoke(self, query: str, max_iter: int, temperature: float, **kwargs) -> str:
        """执行 Agent"""
        pass
    
    def add_message(self, message: Message) -> None:
        """添加消息到历史"""
        self.history.append(message)
        if len(self.history) > self.config.max_history_length:
            self.history.pop(0)
        
        # 同时添加到记忆系统
        if self.enable_memory and self._memory is not None:
            self._memory.add_message(message)
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        from .Message import UserMessage
        self.add_message(UserMessage(content))
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        from .Message import AssistantMessage
        self.add_message(AssistantMessage(content))
    
    def get_memory_context(self, query: Optional[str] = None) -> str:
        """
        获取记忆上下文
        
        Args:
            query: 查询（用于向量记忆的语义检索）
            
        Returns:
            格式化的记忆上下文
        """
        if not self.enable_memory or self._memory is None:
            return ""
        
        # 检查是否支持 query 参数（向量记忆）
        try:
            return self._memory.get_context(query=query)
        except TypeError:
            return self._memory.get_context()
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()
        if self.enable_memory and self._memory is not None:
            self._memory.clear()
    
    def get_history(self):
        """获取对话历史"""
        return self.history
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, description={self.description})"