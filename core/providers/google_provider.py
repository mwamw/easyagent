"""
Google Provider

支持 Google Gemini API。
"""
from typing import Optional, Any, Generator
from openai import OpenAI
import logging

from .base import BaseProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """
    Google Gemini Provider
    
    适用于：
    - Gemini Pro
    - Gemini Flash
    - 其他 Google AI 模型
    
    注意：使用 OpenAI 兼容层调用 Gemini
    """
    
    def _create_client(self) -> OpenAI:
        """创建客户端（使用 OpenAI 兼容接口）"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """同步调用"""
        temperature = temperature or self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info("✅ Google Provider 响应成功")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Google Provider 调用失败: {e}")
            raise
    
    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用"""
        temperature = temperature or self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            logger.info("✅ Google Provider 流式响应开始")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"❌ Google Provider 流式调用失败: {e}")
            raise
    
    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """带工具调用"""
        temperature = temperature or self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info("✅ Google Provider 工具调用响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ Google Provider 工具调用失败: {e}")
            raise
    
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具结果（Google/Gemini 格式）
        
        Gemini 需要 function role 和 name 字段：
        {"role": "function", "content": "...", "tool_call_id": "...", "name": "..."}
        """
        return {
            "role": "function",
            "content": content,
            "tool_call_id": tool_id,
            "name": tool_name
        }
