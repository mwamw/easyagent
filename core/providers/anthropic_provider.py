"""
Anthropic Provider

支持 Claude API（通过 OpenAI 兼容层）。
"""
from typing import Optional, Any
import json
import logging
from .base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude Provider
    
    适用于：
    - Claude 3.5 Sonnet
    - Claude 3 Opus
    - Claude 3 Haiku
    - Claude Thinking 系列
    
    注意：通过 OpenAI 兼容层调用，需要特殊处理工具结果格式。
    invoke / stream / invoke_with_tools 继承自 BaseProvider。
    """
    
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具结果（Claude 格式）
        
        Claude 原生 API 需要嵌套格式：
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "...", "content": "..."}
            ]
        }
        
        注意：如果使用的是完全兼容 OpenAI 的代理，可能需要使用 OpenAI 格式。
        这里提供 Claude 原生格式，如果不工作请切换到 OpenAI 格式。
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content
                }
            ]
        }
    
    def get_thinking_content(self, response: Any) -> Optional[str]:
        """
        提取 Claude 思考内容
        
        Claude Thinking 模型会在 reasoning_content 中返回思考过程
        """
        return getattr(response, 'reasoning_content', None)
    
    def is_thinking_model(self) -> bool:
        """检查是否是 Thinking 模型"""
        return 'thinking' in self.model.lower()
    
    def format_assistant_response(self, response: Any) -> dict:
        """
        将 OpenAI 格式的 assistant 响应转换为 Claude 格式
        
        OpenAI 格式:
        ChatCompletionMessage(tool_calls=[ChatCompletionMessageFunctionToolCall(...)])
        
        Claude 格式:
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "xxx", "name": "search", "input": {...}}
            ]
        }
        """
        # 如果有 tool_calls，转换为 Claude 的 tool_use 格式
        if hasattr(response, 'tool_calls') and response.tool_calls:
            content = []
            for tool_call in response.tool_calls:
                try:
                    input_data = json.loads(tool_call.function.arguments)
                except Exception:
                    input_data = {}
                
                content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": input_data
                })
            
            return {
                "role": "assistant",
                "content": content
            }
        
        # 如果没有 tool_calls，只是普通文本响应
        return {
            "role": "assistant",
            "content": getattr(response, 'content', '') or ''
        }
