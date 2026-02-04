"""
记忆系统基类模块,定义了记忆系统的基本接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from core.Message import Message


class BaseMemory(ABC):
    """
    记忆系统抽象基类
    
    所有记忆实现必须继承此类并实现以下方法：
    - add_message: 添加消息到记忆
    - get_messages: 获取记忆中的消息
    - get_context: 获取记忆上下文（用于注入到提示词）
    - clear: 清空记忆
    """
    
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """
        添加消息到记忆
        
        Args:
            message: 要添加的消息对象
        """
        pass
    
    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """
        添加用户消息
        
        Args:
            content: 消息内容
        """
        pass
    
    @abstractmethod
    def add_assistant_message(self, content: str) -> None:
        """
        添加助手消息
        
        Args:
            content: 消息内容
        """
        pass
    
    @abstractmethod
    def get_messages(self) -> List[Message]:
        """
        获取记忆中的所有消息
        
        Returns:
            消息列表
        """
        pass
    
    @abstractmethod
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        获取记忆上下文（用于注入到提示词）
        
        Args:
            max_tokens: 最大 token 数限制（可选）
            
        Returns:
            格式化的上下文字符串
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass
    
    @abstractmethod
    def get_memory_variables(self) -> dict:
        """
        获取记忆变量（用于模板填充）
        
        Returns:
            包含记忆内容的字典
        """
        pass
    
    def __len__(self) -> int:
        """返回记忆中的消息数量"""
        return len(self.get_messages())
