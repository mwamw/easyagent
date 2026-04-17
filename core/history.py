from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .Message import Message

CanonicalBlockType = Literal[
    "text",
    "reasoning",
    "function_call",
    "function_response",
    "provider_item",
]
CanonicalRole = Literal["system", "user", "assistant", "tool"]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _reasoning_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _reasoning_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("thinking", "text", "content"):
            text = value.get(key)
            if isinstance(text, str) and text:
                return text
        summary = value.get("summary")
        if summary is not None:
            return _reasoning_text(summary)
        return ""
    if hasattr(value, "text"):
        text = getattr(value, "text", None)
        if isinstance(text, str) and text:
            return text
    if hasattr(value, "thinking"):
        thinking = getattr(value, "thinking", None)
        if isinstance(thinking, str) and thinking:
            return thinking
    if hasattr(value, "summary"):
        return _reasoning_text(getattr(value, "summary", None))
    return _stringify(value)


class CanonicalBlock(BaseModel):
    type: CanonicalBlockType
    text: Optional[str] = None
    summary: Any = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Any = None
    output: Any = None
    signature: Optional[str | bytes] = None
    payload: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.model_dump(exclude_none=True))


class CanonicalMessage(BaseModel):
    record_type: Literal["canonical_message"] = "canonical_message"
    role: CanonicalRole
    content: list[CanonicalBlock] = Field(default_factory=list)
    provider: Optional[str] = None
    provider_message_type: Optional[str] = None
    time: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.model_dump(exclude_none=True))

    def text_content(self) -> str:
        fragments: list[str] = []
        for block in self.content:
            if block.type == "text" and block.text:
                fragments.append(block.text)
            elif block.type == "reasoning":
                text = block.text or _reasoning_text(block.summary)
                if text:
                    fragments.append(text)
            elif block.type == "function_call":
                name = block.name or "tool"
                fragments.append(f"[function_call:{name}]")
            elif block.type == "function_response":
                name = block.name or block.call_id or "tool"
                fragments.append(f"[function_response:{name}]")
        return "\n".join(fragment for fragment in fragments if fragment)


class ReplayHistoryState(BaseModel):
    provider_name: Optional[str] = None
    messages: list[Any] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.model_dump(exclude_none=True))


def is_canonical_message(value: Any) -> bool:
    if isinstance(value, CanonicalMessage):
        return True
    return isinstance(value, dict) and value.get("record_type", value.get("schema")) == "canonical_message"


def coerce_canonical_message(value: Any) -> Optional[CanonicalMessage]:
    if value is None:
        return None
    if isinstance(value, CanonicalMessage):
        return value
    if isinstance(value, dict) and value.get("record_type", value.get("schema")) == "canonical_message":
        if "record_type" not in value and "schema" in value:
            value = dict(value)
            value["record_type"] = value.pop("schema")
        return CanonicalMessage.model_validate(value)
    return None


def canonical_text_content(value: Any) -> str:
    message = coerce_canonical_message(value)
    if message is None:
        return ""
    return message.text_content()


def _generic_canonical_block_from_content_item(item: Any) -> CanonicalBlock:
    if isinstance(item, CanonicalBlock):
        return item

    if not isinstance(item, dict):
        return CanonicalBlock(type="text", text=_stringify(item))

    item_type = str(item.get("type") or "")
    if item_type in {"text", "output_text", "summary_text"} or (not item_type and isinstance(item.get("text"), str)):
        return CanonicalBlock(
            type="text",
            text=str(item.get("text", "")),
            payload=_json_safe(item),
            metadata={"provider_block_type": item_type},
        )
    return CanonicalBlock(
        type="provider_item",
        payload=_json_safe(item),
        metadata={"provider_block_type": item_type or "unknown"},
    )


def _generic_canonical_messages_from_history_entry(
    message: Any,
    *,
    provider_name: Optional[str] = None,
) -> list[CanonicalMessage]:
    if message is None:
        return []

    canonical = coerce_canonical_message(message)
    if canonical is not None:
        return [canonical]

    if isinstance(message, list):
        entries: list[CanonicalMessage] = []
        for item in message:
            entries.extend(_generic_canonical_messages_from_history_entry(item, provider_name=provider_name))
        return entries

    if isinstance(message, Message):
        role = "tool" if message.role in {"tool", "function"} else str(message.role)
        blocks: list[CanonicalBlock]
        if role == "tool":
            blocks = [
                CanonicalBlock(
                    type="function_response",
                    call_id=getattr(message, "tool_call_id", None),
                    name=getattr(message, "name", None),
                    output=message.content,
                    payload=_json_safe(message.to_dict()),
                    metadata={"provider_block_type": "message_tool_result"},
                )
            ]
        else:
            blocks = [CanonicalBlock(type="text", text=message.content)]
        return [
            CanonicalMessage(
                role=role,  # type: ignore[arg-type]
                content=blocks,
                provider=provider_name,
                provider_message_type=message.role,
                time=message.time,
                metadata=_json_safe(message.metadata or {}),
            )
        ]

    if not isinstance(message, dict):
        return [
            CanonicalMessage(
                role="assistant",
                content=[CanonicalBlock(type="text", text=_stringify(message))],
                provider=provider_name,
                provider_message_type="unknown",
            )
        ]

    if message.get("record_type", message.get("schema")) == "canonical_message":
        if "record_type" not in message and "schema" in message:
            message = dict(message)
            message["record_type"] = message.pop("schema")
        return [CanonicalMessage.model_validate(message)]

    raw_role = str(message.get("role") or "assistant")
    role_map = {"tool": "tool", "function": "tool", "model": "assistant"}
    role = role_map.get(raw_role, raw_role)
    blocks: list[CanonicalBlock] = []

    if role in {"assistant", "user", "system"}:
        content = message.get("content")
        if isinstance(content, list):
            blocks.extend(_generic_canonical_block_from_content_item(item) for item in content)
        elif content is not None and content != "":
            blocks.append(CanonicalBlock(type="text", text=_stringify(content)))

    if role == "tool":
        output = message.get("content")
        blocks = [
            CanonicalBlock(
                type="function_response",
                call_id=message.get("tool_call_id") or message.get("call_id") or message.get("id"),
                name=message.get("name") or message.get("tool_name"),
                output=_json_safe(output),
                payload=_json_safe(message),
                metadata={"provider_block_type": raw_role},
            )
        ]

    if raw_role == "user" and blocks and all(block.type == "function_response" for block in blocks):
        role = "tool"

    if not blocks:
        blocks = [CanonicalBlock(type="provider_item", payload=_json_safe(message))]

    return [
        CanonicalMessage(
            role=role,  # type: ignore[arg-type]
            content=blocks,
            provider=provider_name,
            provider_message_type=str(message.get("type") or raw_role),
            metadata={},
        )
    ]
