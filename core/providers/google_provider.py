"""
Google Provider

支持 Google Gemini API（通过 OpenAI 兼容层）。
"""

from .openai_compatible_provider import OpenAICompatibleProviderBase


class GoogleProvider(OpenAICompatibleProviderBase):
    """
    Google Gemini Provider

    适用于通过 OpenAI 兼容网关访问 Gemini 的场景。
    """

    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str,
    ) -> dict:
        return {
            "role": "function",
            "content": content,
            "tool_call_id": tool_id,
            "name": tool_name,
        }
