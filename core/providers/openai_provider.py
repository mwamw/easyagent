"""
OpenAI Provider

支持 OpenAI API 及其兼容接口（DeepSeek、Qwen、Kimi 等）。
"""
from typing import Optional, Any, Generator
from openai import OpenAI
import logging

from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI API Provider
    
    适用于：
    - OpenAI (GPT-4, GPT-3.5)
    - DeepSeek
    - Qwen (通义千问)
    - Kimi (Moonshot)
    - 智谱 AI
    - 其他 OpenAI API 兼容服务
    """
    
    def _create_client(self) -> OpenAI:
        """创建 OpenAI 客户端"""
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
            logger.info("✅ OpenAI Provider 响应成功")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ OpenAI Provider 调用失败: {e}")
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
            logger.info("✅ OpenAI Provider 流式响应开始")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"❌ OpenAI Provider 流式调用失败: {e}")
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
            logger.info("✅ OpenAI Provider 工具调用响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ OpenAI Provider 工具调用失败: {e}")
            raise
    
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具结果（OpenAI 格式）
        
        OpenAI 格式：
        {"role": "tool", "content": "...", "tool_call_id": "..."}
        """
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_id
        }
