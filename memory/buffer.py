"""
对话缓冲记忆模块
"""
from typing import List, Optional
from .base import BaseMemory
from core.Message import Message, UserMessage, AssistantMessage


class ConversationBufferMemory(BaseMemory):
    """
    对话缓冲记忆
    
    保存最近 N 条消息的简单记忆实现。
    当消息数量超过限制时，会自动删除最早的消息。
    
    Attributes:
        max_messages: 最大消息数量
        messages: 消息列表
        human_prefix: 用户消息前缀
        ai_prefix: AI 消息前缀
    
    Example:
        >>> memory = ConversationBufferMemory(max_messages=10)
        >>> memory.add_user_message("你好")
        >>> memory.add_assistant_message("你好！有什么可以帮助你的？")
        >>> print(memory.get_context())
        用户: 你好
        助手: 你好！有什么可以帮助你的？
    """
    
    def __init__(
        self, 
        max_messages: int = 20,
        human_prefix: str = "用户",
        ai_prefix: str = "助手",
        memory_key: str = "history"
    ):
        """
        初始化对话缓冲记忆
        
        Args:
            max_messages: 最大保存的消息数量
            human_prefix: 用户消息前缀
            ai_prefix: AI 消息前缀
            memory_key: 记忆变量的键名
        """
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
    
    def add_message(self, message: Message) -> None:
        """添加消息到记忆"""
        self.messages.append(message)
        self._prune()
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message(UserMessage(content))
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.add_message(AssistantMessage(content))
    
    def add_messages(self, messages: List[Message]) -> None:
        """批量添加消息"""
        for msg in messages:
            self.messages.append(msg)
        self._prune()
    
    def _prune(self) -> None:
        """修剪超出限制的消息"""
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_messages(self) -> List[Message]:
        """获取所有消息"""
        return self.messages.copy()
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        获取格式化的对话上下文
        
        Args:
            max_tokens: 最大 token 数限制（简单实现：按字符数估算）
            
        Returns:
            格式化的对话历史字符串
        """
        lines = []
        for msg in self.messages:
            if msg.role == "user":
                prefix = self.human_prefix
            elif msg.role == "assistant":
                prefix = self.ai_prefix
            else:
                prefix = msg.role
            lines.append(f"{prefix}: {msg.content}")
        
        context = "\n".join(lines)
        
        # 简单的 token 限制（按字符估算，1 token ≈ 2 字符）
        if max_tokens and len(context) > max_tokens * 2:
            context = context[-(max_tokens * 2):]
            # 找到第一个完整的行
            first_newline = context.find("\n")
            if first_newline > 0:
                context = context[first_newline + 1:]
        
        return context
    
    def get_memory_variables(self) -> dict:
        """获取记忆变量"""
        return {self.memory_key: self.get_context()}
    
    def clear(self) -> None:
        """清空记忆"""
        self.messages.clear()
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """获取最后 N 条消息"""
        return self.messages[-n:] if n <= len(self.messages) else self.messages.copy()
    
    def __repr__(self) -> str:
        return f"ConversationBufferMemory(messages={len(self.messages)}, max={self.max_messages})"
