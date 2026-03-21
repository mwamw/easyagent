"""
Provider 基类

定义 LLM Provider 的统一接口。
提供基于 OpenAI SDK 的通用实现，子类只需覆写差异化方法。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Generator
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    LLM Provider 抽象基类
    
    所有 Provider（OpenAI、Claude、Gemini 等）均通过 OpenAI 兼容层调用。
    通用的 invoke / stream / invoke_with_tools 已在此基类中实现，
    子类只需覆写 format_tool_result 等差异化方法即可。
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs
    ):
        """
        初始化 Provider
        
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 地址
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 超时时间
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self.client = self._create_client()
    
    def _create_client(self) -> OpenAI:
        """创建 OpenAI 兼容客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    # ==================== 通用调用实现 ====================

    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str | None:
        """同步调用 LLM"""
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info(f"✅ {self.provider_name} Provider 响应成功")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 调用失败: {e}")
            raise
    
    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用 LLM"""
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            logger.info(f"✅ {self.provider_name} Provider 流式响应开始")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 流式调用失败: {e}")
            raise
    
    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """带工具调用的 LLM 调用"""
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info(f"✅ {self.provider_name} Provider 工具调用响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 工具调用失败: {e}")
            raise
    
    # ==================== 需子类覆写的方法 ====================

    @abstractmethod
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具执行结果为该 Provider 需要的消息格式
        
        Args:
            content: 工具执行结果
            tool_id: 工具调用 ID
            tool_name: 工具名称
            
        Returns:
            格式化后的消息字典
        """
        pass
    
    # ==================== 通用辅助方法 ====================

    def get_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容（如果模型支持）"""
        return getattr(response, 'reasoning_content', None)
    
    def get_response_content(self, response: Any) -> Optional[str]:
        """提取响应内容"""
        return getattr(response, 'content', None)
    
    def has_tool_calls(self, response: Any) -> bool:
        """检查响应是否包含工具调用"""
        return hasattr(response, 'tool_calls') and response.tool_calls
    
    def get_tool_calls(self, response: Any) -> list:
        """获取工具调用列表"""
        if self.has_tool_calls(response):
            return response.tool_calls
        return []
    
    @property
    def provider_name(self) -> str:
        """Provider 名称"""
        return self.__class__.__name__.replace('Provider', '').lower()
