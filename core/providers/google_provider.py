"""
Google Provider

支持 Google Gemini API（通过 OpenAI 兼容层）。
"""
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
    
    注意：使用 OpenAI 兼容层调用 Gemini。
    invoke / stream / invoke_with_tools 继承自 BaseProvider。
    """
    
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
