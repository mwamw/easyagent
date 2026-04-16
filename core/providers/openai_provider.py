"""
OpenAI Provider

支持 OpenAI API 及其兼容接口（DeepSeek、Qwen、Kimi 等）。
"""
import logging
from .openai_compatible_provider import OpenAICompatibleProviderBase

logger = logging.getLogger(__name__)


class OpenAIProvider(OpenAICompatibleProviderBase):
    """
    OpenAI API Provider
    
    适用于：
    - OpenAI (GPT-4, GPT-3.5)
    - DeepSeek
    - Qwen (通义千问)
    - Kimi (Moonshot)
    - 智谱 AI
    - 其他 OpenAI API 兼容服务
    
    invoke / stream / invoke_with_tools 继承自 BaseProvider。
    """
    
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
