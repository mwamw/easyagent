"""
Provider 基类

定义 LLM Provider 的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Generator
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    LLM Provider 抽象基类
    
    每个具体的 Provider（OpenAI、Claude、Gemini 等）都需要实现这些方法。
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
    
    @abstractmethod
    def _create_client(self) -> Any:
        """创建 API 客户端"""
        pass
    
    @abstractmethod
    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        同步调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            LLM 响应内容
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Yields:
            响应内容片段
        """
        pass
    
    @abstractmethod
    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        带工具调用的 LLM 调用
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            temperature: 温度参数
            
        Returns:
            LLM 响应对象（包含 content 和 tool_calls）
        """
        pass
    
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
    
    def get_thinking_content(self, response: Any) -> Optional[str]:
        """
        提取思考内容（如果模型支持）
        
        Args:
            response: LLM 响应对象
            
        Returns:
            思考内容，如果没有则返回 None
        """
        # 默认尝试从 reasoning_content 获取
        return getattr(response, 'reasoning_content', None)
    
    def get_response_content(self, response: Any) -> Optional[str]:
        """
        提取响应内容
        
        Args:
            response: LLM 响应对象
            
        Returns:
            响应内容
        """
        return getattr(response, 'content', None)
    
    def has_tool_calls(self, response: Any) -> bool:
        """
        检查响应是否包含工具调用
        
        Args:
            response: LLM 响应对象
            
        Returns:
            是否有工具调用
        """
        return hasattr(response, 'tool_calls') and response.tool_calls
    
    def get_tool_calls(self, response: Any) -> list:
        """
        获取工具调用列表
        
        Args:
            response: LLM 响应对象
            
        Returns:
            工具调用列表
        """
        if self.has_tool_calls(response):
            return response.tool_calls
        return []
    
    @property
    def provider_name(self) -> str:
        """Provider 名称"""
        return self.__class__.__name__.replace('Provider', '').lower()
