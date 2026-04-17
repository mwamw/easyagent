from __future__ import annotations

from typing import Any, AsyncGenerator, Generator, Optional

from ...history import (
    CanonicalBlock,
    CanonicalMessage,
    _generic_canonical_block_from_content_item,
    _generic_canonical_messages_from_history_entry,
    _json_safe,
    _reasoning_text,
    _stringify,
    coerce_canonical_message,
)
from ..shared import BaseProviderCodec


def _canonical_from_openai_chat_like(message: Any, provider_name: str) -> list[CanonicalMessage]:
    canonical = coerce_canonical_message(message)
    if canonical is not None:
        return [canonical]
    if not isinstance(message, dict):
        return _generic_canonical_messages_from_history_entry(message, provider_name=provider_name)

    raw_role = str(message.get("role") or "assistant")
    role_map = {"tool": "tool", "function": "tool", "model": "assistant"}
    role = role_map.get(raw_role, raw_role)
    blocks: list[CanonicalBlock] = []

    content = message.get("content")
    if role in {"assistant", "user", "system"}:
        if isinstance(content, list):
            blocks.extend(_generic_canonical_block_from_content_item(item) for item in content)
        elif content is not None and content != "":
            blocks.append(CanonicalBlock(type="text", text=_stringify(content)))

    reasoning_content = message.get("reasoning_content") or message.get("thinking")
    if reasoning_content:
        blocks.insert(
            0,
            CanonicalBlock(
                type="reasoning",
                text=_reasoning_text(reasoning_content),
                payload=_json_safe(reasoning_content),
                metadata={"provider_block_type": "reasoning_content"},
            ),
        )

    if role == "assistant":
        for tool_call in message.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
            arguments = tool_call.get("arguments")
            if arguments is None:
                arguments = function.get("arguments")
            blocks.append(
                CanonicalBlock(
                    type="function_call",
                    call_id=tool_call.get("id") or tool_call.get("call_id"),
                    name=tool_call.get("name") or function.get("name"),
                    arguments=_json_safe(arguments),
                    payload=_json_safe(tool_call),
                    metadata={"provider_block_type": "tool_call"},
                )
            )

    if role == "tool":
        blocks = [
            CanonicalBlock(
                type="function_response",
                call_id=message.get("tool_call_id") or message.get("call_id") or message.get("id"),
                name=message.get("name") or message.get("tool_name"),
                output=_json_safe(message.get("content")),
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


class OpenAIChatCodec(BaseProviderCodec):
    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _canonical_from_openai_chat_like(message, self.provider_name)

    def is_request_ready_message(self, message: Any) -> bool:
        return isinstance(message, dict) and "role" in message and (
            "content" in message or "tool_calls" in message
        )

    def build_assistant_message(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if thinking:
            message["reasoning_content"] = thinking
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
                for tool_call in tool_calls
            ]
        return message

    def build_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict[str, Any]:
        content = getattr(response, "content", None) or None
        thinking = self.get_thinking_content(response) if include_reasoning else None
        tool_calls_data = getattr(response, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls_data):
            function = getattr(tool_call, "function", None)
            tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None) or f"tool_call_{index}",
                    "name": getattr(function, "name", None) or "",
                    "arguments": getattr(function, "arguments", None) or "",
                }
            )
        return self.build_assistant_message(content=content, tool_calls=tool_calls or None, thinking=thinking)

    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, Any]:
        return {"role": "tool", "content": content, "tool_call_id": tool_id, "name": tool_name}

    def get_response_content(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        if isinstance(response, str):
            return response
        return getattr(response, "content", None)

    def get_thinking_content(self, response: Any) -> Optional[str]:
        if response is None or isinstance(response, str):
            return None
        return getattr(response, "reasoning_content", None) or getattr(response, "reasoning", None)

    def has_tool_calls(self, response: Any) -> bool:
        return bool(getattr(response, "tool_calls", None))

    def get_tool_calls(self, response: Any) -> list[Any]:
        return list(getattr(response, "tool_calls", []) or [])

    def _init_chat_tool_stream_state(self) -> dict[str, Any]:
        return {"text_parts": [], "thinking_parts": [], "tool_calls": {}, "terminal_emitted": False}

    def _extract_chat_stream_events(self, chunk: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return events

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return events

        reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning_delta:
            state["thinking_parts"].append(reasoning_delta)
            events.append({"type": "thinking_delta", "delta": reasoning_delta})

        content_delta = getattr(delta, "content", None) or ""
        if content_delta:
            state["text_parts"].append(content_delta)
            events.append({"type": "text_delta", "delta": content_delta})

        for tool_call in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_call, "index", None)
            if index is None:
                index = len(state["tool_calls"])
            current = state["tool_calls"].setdefault(index, {"id": None, "name": "", "arguments": ""})
            if getattr(tool_call, "id", None):
                current["id"] = tool_call.id
            function = getattr(tool_call, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    current["name"] = function.name
                if getattr(function, "arguments", None):
                    current["arguments"] += function.arguments

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason in {"stop", "length", "content_filter"}:
            tool_calls = self._normalize_chat_tool_calls(state["tool_calls"])
            if tool_calls:
                events.append(
                    {
                        "type": "tool_calls",
                        "tool_calls": tool_calls,
                        "content": "".join(state["text_parts"]),
                        "thinking": "".join(state["thinking_parts"]),
                    }
                )
            else:
                events.append(
                    {
                        "type": "final_response",
                        "content": "".join(state["text_parts"]),
                        "thinking": "".join(state["thinking_parts"]),
                    }
                )
            state["terminal_emitted"] = True
        return events

    def _normalize_chat_tool_calls(self, tool_calls_by_index: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            tool_call = dict(tool_calls_by_index[index])
            if not tool_call.get("id"):
                tool_call["id"] = f"call_{index}"
            if not tool_call.get("arguments"):
                tool_call["arguments"] = "{}"
            normalized.append(tool_call)
        return normalized

    def _finalize_chat_tool_stream_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["terminal_emitted"]:
            return {"type": "stream_end"}
        tool_calls = self._normalize_chat_tool_calls(state["tool_calls"])
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
            }
        return {
            "type": "final_response",
            "content": "".join(state["text_parts"]),
            "thinking": "".join(state["thinking_parts"]),
        }

    def stream_events(self, raw_stream: Any, *, tools: bool = False) -> Generator[dict[str, Any], None, None]:
        state = self._init_chat_tool_stream_state()
        for chunk in raw_stream:
            for event in self._extract_chat_stream_events(chunk, state):
                yield event
        final_event = self._finalize_chat_tool_stream_state(state)
        if final_event.get("type") != "stream_end":
            yield final_event

    async def astream_events(self, raw_stream: Any, *, tools: bool = False) -> AsyncGenerator[dict[str, Any], None]:
        state = self._init_chat_tool_stream_state()
        async for chunk in raw_stream:
            for event in self._extract_chat_stream_events(chunk, state):
                yield event
        final_event = self._finalize_chat_tool_stream_state(state)
        if final_event.get("type") != "stream_end":
            yield final_event
