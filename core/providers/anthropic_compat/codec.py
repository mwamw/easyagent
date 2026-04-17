from __future__ import annotations

import json
from typing import Any, Optional

from ...history import (
    CanonicalBlock,
    CanonicalMessage,
    _generic_canonical_messages_from_history_entry,
    _json_safe,
    _reasoning_text,
    _stringify,
    coerce_canonical_message,
)
from ..openai_compat.codec import OpenAIChatCodec


def _canonical_block_from_anthropic_content(block: Any) -> CanonicalBlock:
    serialized = AnthropicCompatCodec._serialize_block(block)
    block_type = serialized.get("type")
    if block_type == "text":
        return CanonicalBlock(
            type="text",
            text=serialized.get("text", ""),
            payload=_json_safe(serialized),
            metadata={"provider_block_type": "text"},
        )
    if block_type in {"thinking", "redacted_thinking"}:
        return CanonicalBlock(
            type="reasoning",
            text=_reasoning_text(serialized),
            summary=_json_safe(serialized.get("summary")),
            signature=serialized.get("signature"),
            payload=_json_safe(serialized),
            metadata={"provider_block_type": block_type},
        )
    if block_type == "tool_use":
        return CanonicalBlock(
            type="function_call",
            call_id=serialized.get("id"),
            name=serialized.get("name"),
            arguments=_json_safe(serialized.get("input", {})),
            payload=_json_safe(serialized),
            metadata={"provider_block_type": "tool_use"},
        )
    if block_type == "tool_result":
        return CanonicalBlock(
            type="function_response",
            call_id=serialized.get("tool_use_id"),
            name=serialized.get("name"),
            output=_json_safe(serialized.get("content")),
            payload=_json_safe(serialized),
            metadata={"provider_block_type": "tool_result"},
        )
    return CanonicalBlock(type="provider_item", payload=_json_safe(serialized))


def _canonical_from_anthropic(message: Any, provider_name: str) -> list[CanonicalMessage]:
    canonical = coerce_canonical_message(message)
    if canonical is not None:
        return [canonical]
    if isinstance(message, list):
        entries: list[CanonicalMessage] = []
        for item in message:
            entries.extend(_canonical_from_anthropic(item, provider_name))
        return entries
    if not isinstance(message, dict):
        return _generic_canonical_messages_from_history_entry(message, provider_name=provider_name)

    raw_role = str(message.get("role") or "assistant")
    role_map = {"assistant": "assistant", "user": "user", "system": "system"}
    role = role_map.get(raw_role, "assistant")
    blocks: list[CanonicalBlock] = []
    content = message.get("content")
    if isinstance(content, list):
        blocks.extend(_canonical_block_from_anthropic_content(block) for block in content)
    elif content is not None and content != "":
        blocks.append(CanonicalBlock(type="text", text=_stringify(content)))

    reasoning_content = message.get("reasoning_content") or message.get("thinking")
    if reasoning_content and not any(block.type == "reasoning" for block in blocks):
        blocks.insert(
            0,
            CanonicalBlock(
                type="reasoning",
                text=_reasoning_text(reasoning_content),
                payload=_json_safe(reasoning_content),
                metadata={"provider_block_type": "reasoning_content"},
            ),
        )

    if raw_role == "user" and blocks and all(block.type == "function_response" for block in blocks):
        role = "tool"
    if role == "system" and isinstance(content, str):
        blocks = [CanonicalBlock(type="text", text=content)]

    if not blocks:
        blocks = [CanonicalBlock(type="provider_item", payload=_json_safe(message))]

    return [
        CanonicalMessage(
            role=role,  # type: ignore[arg-type]
            content=blocks,
            provider=provider_name,
            provider_message_type=raw_role,
            metadata={},
        )
    ]


class AnthropicCompatCodec(OpenAIChatCodec):
    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _canonical_from_anthropic(message, self.provider_name)

    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
        }

    def build_assistant_message(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict[str, Any]:
        if tool_calls:
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
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
            message: dict[str, Any] = {"role": "assistant", "content": blocks}
        else:
            message = {"role": "assistant", "content": content or ""}
        if thinking:
            message["reasoning_content"] = thinking
        return message

    @staticmethod
    def _normalize_content_block(block: Any) -> dict[str, Any]:
        if isinstance(block, dict):
            return dict(block)
        block_type = getattr(block, "type", None)
        if block_type == "text":
            return {"type": "text", "text": getattr(block, "text", "")}
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

    @staticmethod
    def _serialize_block(block: Any) -> dict[str, Any]:
        if isinstance(block, dict):
            return dict(block)
        block_type = getattr(block, "type", None)
        if block_type == "text":
            return {"type": "text", "text": getattr(block, "text", "")}
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", None) or {},
            }
        if block_type == "tool_result":
            payload = {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None),
                "content": getattr(block, "content", ""),
            }
            if getattr(block, "name", None):
                payload["name"] = getattr(block, "name")
            return payload
        if block_type == "thinking":
            payload = {
                "type": "thinking",
                "thinking": getattr(block, "thinking", None) or getattr(block, "text", "") or "",
            }
            if getattr(block, "signature", None):
                payload["signature"] = getattr(block, "signature")
            return payload
        if block_type == "redacted_thinking":
            payload = {"type": "redacted_thinking"}
            if getattr(block, "data", None) is not None:
                payload["data"] = getattr(block, "data")
            return payload
        return {"type": block_type or "unknown"}

    def build_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict[str, Any]:
        if hasattr(response, "tool_calls") and response.tool_calls:
            content = []
            response_text = getattr(response, "content", "") or ""
            if response_text:
                content.append({"type": "text", "text": response_text})
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
            message: dict[str, Any] = {"role": "assistant", "content": content}
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
            message = {"role": "assistant", "content": getattr(response, "content", "") or ""}
        if include_reasoning:
            thinking = self.get_thinking_content(response)
            if thinking:
                message["reasoning_content"] = thinking
        return message
