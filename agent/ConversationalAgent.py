"""
对话记忆 Agent

集成记忆系统，支持多轮对话的上下文保持。
"""
from typing import Optional, List, Generator
from typing_extensions import override
import logging

from .BasicAgent import BasicAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from memory.base import BaseMemory
from memory.buffer import ConversationBufferMemory
from core.Exception import *

logger = logging.getLogger(__name__)


class ConversationalAgent(BasicAgent):
    """
    对话记忆 Agent
    
    集成记忆系统，能够记住对话历史并在回答中使用上下文信息。
    适合多轮对话场景。
    
    特点：
    - 支持可配置的记忆后端
    - 自动管理对话上下文
    - 支持记忆摘要和检索
    
    Example:
        >>> from memory import ConversationBufferMemory
        >>> 
        >>> memory = ConversationBufferMemory(max_messages=20)
        >>> agent = ConversationalAgent(
        ...     name="chatbot",
        ...     llm=llm,
        ...     memory=memory
        ... )
        >>> agent.invoke("你好，我叫张三")
        >>> agent.invoke("你还记得我的名字吗？")
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        memory: Optional[BaseMemory] = None,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        memory_key: str = "history",
        human_prefix: str = "用户",
        ai_prefix: str = "助手",
    ):
        """
        初始化对话 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            memory: 记忆系统（默认使用 ConversationBufferMemory）
            enable_tool: 是否启用工具
            tool_registry: 工具注册表
            description: Agent 描述
            config: 配置
            memory_key: 记忆变量名
            human_prefix: 用户消息前缀
            ai_prefix: AI 消息前缀
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
        
        # 初始化记忆系统
        self.memory = memory or ConversationBufferMemory(
            max_messages=20,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            memory_key=memory_key
        )
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
    
    @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """执行对话"""
        self._validate_invoke_params(query, max_iter, temperature)
        
        if self.enable_tool and self.tool_registry:
            response = self._invoke_with_memory_and_tools(query, max_iter, temperature)
        else:
            response = self._invoke_with_memory(query, temperature, **kwargs)
        
        # 保存到记忆
        self.memory.add_user_message(query)
        self.memory.add_assistant_message(response)
        
        return response
    
    def _invoke_with_memory(self, query: str, temperature: float, **kwargs) -> str:
        """带记忆的普通调用"""
        # 获取记忆上下文
        memory_context = self.memory.get_context()
        
        # 构建系统提示词
        system_content = self._build_system_prompt_with_memory(memory_context)
        
        messages = [
            SystemMessage(system_content),
            UserMessage(query)
        ]
        
        try:
            response = self.llm.invoke(messages, temperature=temperature, **kwargs)
            
            if response is None:
                raise LLMInvokeError("LLM 返回了空响应!")
            
            return str(response) if not isinstance(response, str) else response
        except Exception as e:
            logger.error(f"对话调用失败: {e}")
            raise LLMInvokeError(f"对话调用失败: {e}") from e
    
    def _invoke_with_memory_and_tools(
        self, 
        query: str, 
        max_iter: int, 
        temperature: float
    ) -> str:
        """带记忆和工具的调用"""
        # 获取记忆上下文并注入到消息中
        memory_context = self.memory.get_context()
        
        messages: list = []
        
        # 添加带记忆的系统提示
        system_content = self._build_system_prompt_with_memory(memory_context)
        messages.append(SystemMessage(system_content))
        
        # 添加用户查询
        messages.append(UserMessage(query))
        
        # 使用父类的工具调用逻辑（跳过添加系统消息的部分）
        return self._execute_tool_loop(messages, max_iter, temperature)
    
    def _execute_tool_loop(
        self, 
        messages: List[Message], 
        max_iter: int, 
        temperature: float
    ) -> str:
        """执行工具调用循环"""
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        final_response: Optional[str] = None
        
        while max_iter > 0:
            try:
                response = self.llm.invoke_with_tools(
                    messages,
                    self.tool_registry.get_openai_tools(),
                    temperature=temperature
                )
                
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
            except Exception as e:
                logger.error(f"工具调用失败: {e}")
                final_response = f"对话调用失败: {str(e)}"
                break
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = "unknown_tool"
                    tool_id = getattr(tool_call, 'id', 'unknown')
                    
                    try:
                        tool_name = self._safe_get_tool_name(tool_call)
                        tool_args = self._safe_parse_tool_args(tool_call)
                        tool_result = self._safe_execute_tool(tool_name, tool_args)
                        messages.append(self.tool_message_class(tool_result, tool_id, name=tool_name))
                    except Exception as e:
                        error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                        messages.append(self.tool_message_class(error_msg, tool_id, name=tool_name))
            else:
                content = getattr(response, 'content', None)
                if content is not None:
                    final_response = content
                else:
                    final_response = ""
                break
            
            max_iter -= 1
        
        if final_response is None:
            final_response = "超过最大迭代次数，对话调用失败!"
        
        return final_response
    
    def _build_system_prompt_with_memory(self, memory_context: str) -> str:
        """构建包含记忆的系统提示词"""
        base_prompt = self.system_prompt or "你是一个友好的对话助手。"
        
        if memory_context:
            return f"""{base_prompt}

## 对话历史
以下是之前的对话记录，请在回答时考虑这些上下文：

{memory_context}

---
请基于以上对话历史，自然地继续对话。"""
        
        return base_prompt
    
    def get_memory(self) -> BaseMemory:
        """获取记忆系统"""
        return self.memory
    
    def set_memory(self, memory: BaseMemory) -> None:
        """设置记忆系统"""
        self.memory = memory
    
    @override
    def clear_history(self) -> None:
        """清空对话历史和记忆"""
        super().clear_history()
        self.memory.clear()
        logger.info("对话历史和记忆已清空")
    
    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        return self.memory.get_context()
