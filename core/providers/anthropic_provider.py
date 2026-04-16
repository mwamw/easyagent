"""
Anthropic Provider

支持 Claude API（通过 OpenAI 兼容层）。
"""

from typing import Optional, Any
import json

from .openai_compatible_provider import OpenAICompatibleProviderBase


class AnthropicProvider(OpenAICompatibleProviderBase):
    """
    Anthropic Claude Provider

    适用于通过 OpenAI 兼容网关访问 Claude 的场景。
    """

    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str,
    ) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            ],
        }

    def get_thinking_content(self, response: Any) -> Optional[str]:
        return getattr(response, "reasoning_content", None)

    def is_thinking_model(self) -> bool:
        return "thinking" in self.model.lower()

    def format_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict:
        if hasattr(response, "tool_calls") and response.tool_calls:
            content = []
            response_text = getattr(response, "content", "") or ""
            if response_text:
                content.append(
                    {
                        "type": "text",
                        "text": response_text,
                    }
                )
            for tool_call in response.tool_calls:
                try:
                    input_data = json.loads(tool_call.function.arguments)
                except Exception:
                    input_data = {}

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": input_data,
                    }
                )

            message = {
                "role": "assistant",
                "content": content,
            }
            if include_reasoning:
                thinking = self.get_thinking_content(response)
                if thinking:
                    message["reasoning_content"] = thinking
            return message

        if isinstance(getattr(response, "content", None), list):
            message = {
                "role": "assistant",
                "content": [self._normalize_content_block(block) for block in response.content],
            }
        else:
            message = {
                "role": "assistant",
                "content": getattr(response, "content", "") or "",
            }
        if include_reasoning:
            thinking = self.get_thinking_content(response)
            if thinking:
                message["reasoning_content"] = thinking
        return message

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict:
        if tool_calls:
            blocks = []
            if content:
                blocks.append(
                    {
                        "type": "text",
                        "text": content,
                    }
                )
            for tool_call in tool_calls:
                try:
                    input_data = json.loads(tool_call["arguments"])
                except Exception:
                    input_data = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": input_data,
                    }
                )
            message = {
                "role": "assistant",
                "content": blocks,
            }
        else:
            message = {
                "role": "assistant",
                "content": content or "",
            }
        if thinking:
            message["reasoning_content"] = thinking
        return message

    @staticmethod
    def _normalize_content_block(block: Any) -> dict[str, Any]:
        if isinstance(block, dict):
            return dict(block)
        block_type = getattr(block, "type", None)
        if block_type == "text":
            return {
                "type": "text",
                "text": getattr(block, "text", ""),
            }
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", None) or {},
            }
        if block_type == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None),
                "content": getattr(block, "content", ""),
            }
        return {"type": block_type or "unknown"}
