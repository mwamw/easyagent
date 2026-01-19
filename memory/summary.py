"""
对话摘要记忆模块

使用 LLM 对长对话进行压缩摘要，节省 Token 消耗。
"""
from typing import List, Optional, TYPE_CHECKING
from .base import BaseMemory
from core.Message import Message, UserMessage, AssistantMessage

if TYPE_CHECKING:
    from core.llm import EasyLLM


class ConversationSummaryMemory(BaseMemory):
    """
    对话摘要记忆
    
    使用 LLM 对历史对话进行摘要，适合长对话场景。
    当缓冲区消息超过阈值时，会自动触发摘要。
    
    Attributes:
        llm: 用于生成摘要的 LLM 实例
        summary: 当前摘要内容
        buffer: 待摘要的新消息
        summary_threshold: 触发摘要的消息数阈值
    
    Example:
        >>> from core.llm import EasyLLM
        >>> llm = EasyLLM()
        >>> memory = ConversationSummaryMemory(llm=llm, summary_threshold=5)
        >>> memory.add_user_message("你好")
        >>> memory.add_assistant_message("你好！有什么可以帮助你的？")
        >>> print(memory.get_context())
    """
    
    def __init__(
        self,
        llm: "EasyLLM",
        summary_threshold: int = 6,
        human_prefix: str = "用户",
        ai_prefix: str = "助手",
        memory_key: str = "history"
    ):
        """
        初始化对话摘要记忆
        
        Args:
            llm: LLM 实例，用于生成摘要
            summary_threshold: 触发摘要的消息数量阈值
            human_prefix: 用户消息前缀
            ai_prefix: AI 消息前缀
            memory_key: 记忆变量的键名
        """
        self.llm = llm
        self.summary_threshold = summary_threshold
        self.summary: str = ""
        self.buffer: List[Message] = []
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
    
    def add_message(self, message: Message) -> None:
        """添加消息到缓冲区"""
        self.buffer.append(message)
        
        # 检查是否需要触发摘要
        if len(self.buffer) >= self.summary_threshold:
            self._summarize()
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message(UserMessage(content))
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.add_message(AssistantMessage(content))
    
    def add_messages(self, messages: List[Message]) -> None:
        """批量添加消息"""
        for msg in messages:
            self.buffer.append(msg)
        
        if len(self.buffer) >= self.summary_threshold:
            self._summarize()
    
    def _summarize(self) -> None:
        """使用 LLM 生成摘要"""
        if not self.buffer:
            return
        
        # 格式化缓冲区消息
        buffer_text = self._format_messages(self.buffer)
        
        # 构建摘要提示词
        if self.summary:
            prompt = f"""请将以下对话历史和新对话合并为一个简洁的摘要。保留关键信息和重要细节。

当前摘要：
{self.summary}

新对话：
{buffer_text}

请输出更新后的摘要（不超过 300 字）："""
        else:
            prompt = f"""请将以下对话摘要为简洁的要点，保留关键信息和重要细节。

对话内容：
{buffer_text}

请输出摘要（不超过 200 字）："""
        
        try:
            # 调用 LLM 生成摘要
            from core.Message import SystemMessage
            messages = [
                SystemMessage("你是一个对话摘要助手，擅长提取关键信息并生成简洁的摘要。"),
                UserMessage(prompt)
            ]
            self.summary = self.llm.invoke(messages, temperature=0.3)
            
            # 清空缓冲区
            self.buffer.clear()
            
        except Exception as e:
            # 摘要失败时保留缓冲区的一半消息
            import logging
            logging.warning(f"摘要生成失败: {e}，保留部分缓冲区")
            half = len(self.buffer) // 2
            self.buffer = self.buffer[half:]
    
    def _format_messages(self, messages: List[Message]) -> str:
        """格式化消息列表"""
        lines = []
        for msg in messages:
            if msg.role == "user":
                prefix = self.human_prefix
            elif msg.role == "assistant":
                prefix = self.ai_prefix
            else:
                prefix = msg.role
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
    
    def get_messages(self) -> List[Message]:
        """获取缓冲区中的消息"""
        return self.buffer.copy()
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        获取记忆上下文
        
        返回摘要 + 最近的缓冲区消息。
        
        Args:
            max_tokens: 最大 token 数限制
            
        Returns:
            格式化的记忆上下文
        """
        parts = []
        
        if self.summary:
            parts.append(f"[对话摘要]\n{self.summary}")
        
        if self.buffer:
            recent = self._format_messages(self.buffer)
            parts.append(f"[最近对话]\n{recent}")
        
        context = "\n\n".join(parts)
        
        # 简单的 token 限制
        if max_tokens and len(context) > max_tokens * 2:
            context = context[-(max_tokens * 2):]
        
        return context
    
    def get_summary(self) -> str:
        """获取当前摘要"""
        return self.summary
    
    def get_memory_variables(self) -> dict:
        """获取记忆变量"""
        return {self.memory_key: self.get_context()}
    
    def clear(self) -> None:
        """清空记忆"""
        self.summary = ""
        self.buffer.clear()
    
    def force_summarize(self) -> None:
        """强制执行摘要（不管缓冲区大小）"""
        if self.buffer:
            self._summarize()
    
    def __repr__(self) -> str:
        return f"ConversationSummaryMemory(buffer={len(self.buffer)}, has_summary={bool(self.summary)})"
