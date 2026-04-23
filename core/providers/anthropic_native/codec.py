from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Generator, Optional

from ...history import (
    CanonicalBlock,
    CanonicalMessage,
    _generic_canonical_messages_from_history_entry,
    _json_safe,
    _reasoning_text,
    _stringify,
    coerce_canonical_message,
)
from ..shared import BaseProviderCodec

logger = logging.getLogger(__name__)


def _canonical_block_from_anthropic_content(block: Any) -> CanonicalBlock:
    serialized = AnthropicNativeCodec._serialize_block(block)
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


class AnthropicNativeCodec(BaseProviderCodec):
    def build_request_token_payload(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        pending_messages: Optional[list[Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Any:
        messages = [_json_safe(item) for item in replay_history]
        if pending_messages:
            messages.extend(_json_safe(item) for item in pending_messages)
        payload: dict[str, Any] = {"messages": messages}
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = _json_safe(tools)
        if reasoning:
            payload["thinking"] = _json_safe(reasoning)
        return payload

    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _canonical_from_anthropic(message, self.provider_name)

    def query_to_replay(self, query: str) -> list[Any]:
        return [{"role": "user", "content": query}]

    def append_replay_entry(self, prepared: list[Any], item: Any) -> None:
        if self._is_tool_result_turn(item) and prepared and self._is_tool_result_turn(prepared[-1]):
            prepared[-1]["content"].extend(item["content"])
            return
        prepared.append(item)

    def is_request_ready_message(self, message: Any) -> bool:
        return isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"} and "content" in message

    @staticmethod
    def _is_tool_result_turn(message: Any) -> bool:
        if not isinstance(message, dict) or message.get("role") != "user":
            return False
        content = message.get("content")
        if not isinstance(content, list) or not content:
            return False
        return all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)

    @staticmethod
    def _as_dict(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return value
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                return value
        return value

    @classmethod
    def _serialize_block(cls, block: Any) -> dict[str, Any]:
        block = cls._as_dict(block)
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                return {"type": "text", "text": block.get("text", "")}
            if block_type == "tool_use":
                return {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}) or {},
                }
            if block_type == "tool_result":
                payload = {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "content": block.get("content", ""),
                }
                if block.get("name"):
                    payload["name"] = block.get("name")
                return payload
            if block_type == "thinking":
                payload = {"type": "thinking", "thinking": block.get("thinking") or block.get("text", "")}
                if block.get("signature"):
                    payload["signature"] = block["signature"]
                return payload
            if block_type == "redacted_thinking":
                payload = {"type": "redacted_thinking"}
                if block.get("data") is not None:
                    payload["data"] = block["data"]
                return payload
            return block
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

    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, Any]:
        payload = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
        }
        if tool_name:
            payload["content"][0]["name"] = tool_name
        return payload

    def tool_result_to_canonical(self, content: str, tool_id: str, tool_name: str) -> list[CanonicalMessage]:
        payload = {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content,
        }
        if tool_name:
            payload["name"] = tool_name
        return [
            CanonicalMessage(
                role="tool",
                content=[
                    CanonicalBlock(
                        type="function_response",
                        call_id=tool_id,
                        name=tool_name or None,
                        output=content,
                        payload=payload,
                        metadata={"provider_block_type": "tool_result"},
                    )
                ],
                provider=self.provider_name,
                provider_message_type="user",
            )
        ]

    def build_assistant_message(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": thinking})
        if content:
            blocks.append({"type": "text", "text": content})
        for tool_call in tool_calls or []:
            try:
                arguments = json.loads(tool_call["arguments"])
            except Exception:
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": arguments,
                }
            )
        return {"role": "assistant", "content": blocks}

    def get_thinking_content(self, response: Any) -> Optional[str]:
        thoughts: list[str] = []
        for block in getattr(response, "content", None) or []:
            serialized = self._serialize_block(block)
            if serialized.get("type") == "thinking" and serialized.get("thinking"):
                thoughts.append(serialized["thinking"])
        return "".join(thoughts) or None

    def get_response_content(self, response: Any) -> Optional[str]:
        content_blocks = getattr(response, "content", None) or []
        texts: list[str] = []
        for block in content_blocks:
            serialized = self._serialize_block(block)
            if serialized.get("type") == "text" and serialized.get("text"):
                texts.append(serialized["text"])
        return "".join(texts) or None

    def has_tool_calls(self, response: Any) -> bool:
        return bool(self.get_tool_calls(response))

    def get_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for index, block in enumerate(getattr(response, "content", None) or []):
            serialized = self._serialize_block(block)
            if serialized.get("type") != "tool_use":
                continue
            tool_calls.append(
                {
                    "id": serialized.get("id") or f"tool_call_{index}",
                    "name": serialized.get("name", ""),
                    "arguments": serialized.get("input", {}) or {},
                }
            )
        return tool_calls

    def build_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict[str, Any]:
        content = []
        for block in getattr(response, "content", None) or []:
            serialized = self._serialize_block(block)
            if serialized.get("type") in {"thinking", "redacted_thinking"} and not include_reasoning:
                continue
            content.append(serialized)
        return {"role": "assistant", "content": content}

    def response_to_replay(self, response: Any, *, include_reasoning: bool = False) -> list[Any]:
        if response is None:
            return []
        return [self.build_assistant_response(response, include_reasoning=include_reasoning)]

    def response_to_canonical(self, response: Any, *, include_reasoning: bool = False) -> list[CanonicalMessage]:
        if response is None:
            return []
        blocks: list[CanonicalBlock] = []
        for block in getattr(response, "content", None) or []:
            serialized = self._serialize_block(block)
            block_type = serialized.get("type")
            if block_type in {"thinking", "redacted_thinking"} and not include_reasoning:
                continue
            if block_type == "text":
                blocks.append(
                    CanonicalBlock(
                        type="text",
                        text=serialized.get("text", ""),
                        payload=_json_safe(serialized),
                        metadata={"provider_block_type": "text"},
                    )
                )
            elif block_type in {"thinking", "redacted_thinking"}:
                blocks.append(
                    CanonicalBlock(
                        type="reasoning",
                        text=_reasoning_text(serialized),
                        summary=_json_safe(serialized.get("summary")),
                        signature=serialized.get("signature"),
                        payload=_json_safe(serialized),
                        metadata={"provider_block_type": block_type},
                    )
                )
            elif block_type == "tool_use":
                blocks.append(
                    CanonicalBlock(
                        type="function_call",
                        call_id=serialized.get("id"),
                        name=serialized.get("name"),
                        arguments=_json_safe(serialized.get("input", {})),
                        payload=_json_safe(serialized),
                        metadata={"provider_block_type": "tool_use"},
                    )
                )
            elif block_type == "tool_result":
                blocks.append(
                    CanonicalBlock(
                        type="function_response",
                        call_id=serialized.get("tool_use_id"),
                        name=serialized.get("name"),
                        output=_json_safe(serialized.get("content")),
                        payload=_json_safe(serialized),
                        metadata={"provider_block_type": "tool_result"},
                    )
                )
        if not blocks:
            blocks = [CanonicalBlock(type="provider_item", payload=_json_safe(response))]
        return [
            CanonicalMessage(
                role="assistant",
                content=blocks,
                provider=self.provider_name,
                provider_message_type="assistant",
            )
        ]

    def canonical_message_to_replay(self, message: Any) -> list[Any]:
        canonical = coerce_canonical_message(message)
        if canonical is None:
            entries: list[Any] = []
            for entry in self.history_entry_to_canonical(message):
                entries.extend(self.canonical_message_to_replay(entry))
            return entries

        blocks: list[dict[str, Any]] = []
        for block in canonical.content:
            payload = self._provider_payload_for_current_provider(canonical, block)
            provider_block_type = block.metadata.get("provider_block_type") if isinstance(block.metadata, dict) else None
            if block.type == "text":
                if block.text:
                    blocks.append({"type": "text", "text": block.text})
                continue
            if block.type == "reasoning":
                text = block.text or _reasoning_text(block.summary or block.payload)
                if isinstance(payload, dict) and payload.get("type") in {"thinking", "redacted_thinking"}:
                    blocks.append(dict(payload))
                    continue
                if provider_block_type == "redacted_thinking" and isinstance(payload, dict):
                    blocks.append(dict(payload))
                    continue
                thinking_block: dict[str, Any] = {"type": "thinking", "thinking": text}
                signature = self._signature_for_current_provider(canonical, block)
                if signature:
                    thinking_block["signature"] = signature
                blocks.append(thinking_block)
                continue
            if block.type == "function_call":
                if isinstance(payload, dict) and payload.get("type") == "tool_use":
                    blocks.append(dict(payload))
                    continue
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.call_id,
                        "name": block.name or "",
                        "input": self._dict_arguments(block.arguments),
                    }
                )
                continue
            if block.type == "function_response":
                content = block.output
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": block.call_id,
                    "content": content,
                }
                if block.name:
                    tool_result["name"] = block.name
                blocks.append(tool_result)
                continue
            if block.type == "provider_item" and isinstance(payload, dict):
                blocks.append(dict(payload))

        if canonical.role == "tool":
            return [{"role": "user", "content": blocks or [{"type": "tool_result", "tool_use_id": None, "content": ""}]}]

        if not blocks:
            return []

        if canonical.role == "system":
            text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict) and block.get("type") == "text")
            if not text:
                text = canonical.text_content()
            return [{"role": "system", "content": text}]

        content: Any
        if blocks and all(isinstance(block, dict) and block.get("type") == "text" for block in blocks):
            content = "".join(block.get("text", "") for block in blocks)
        else:
            content = blocks
        return [{"role": "assistant" if canonical.role == "assistant" else "user", "content": content}]

    @staticmethod
    def _dict_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                loaded = json.loads(arguments)
            except Exception:
                return {}
            if isinstance(loaded, dict):
                return loaded
            return {}
        return {}

    def _iter_stream_events(self, stream_source: Any) -> Generator[Any, None, None]:
        if hasattr(stream_source, "__iter__"):
            for event in stream_source:
                yield event
            return
        if hasattr(stream_source, "events"):
            for event in stream_source.events():
                yield event

    async def _aiter_stream_events(self, stream_source: Any) -> AsyncGenerator[Any, None]:
        if hasattr(stream_source, "__aiter__"):
            async for event in stream_source:
                yield event
            return
        if hasattr(stream_source, "__aenter__"):
            async with stream_source as active_stream:
                if hasattr(active_stream, "__aiter__"):
                    async for event in active_stream:
                        yield event
                elif hasattr(active_stream, "events"):
                    async for event in active_stream.events():
                        yield event
            return
        if hasattr(stream_source, "__aiter__"):
            async for event in stream_source:
                yield event

    def _init_anthropic_stream_state(self) -> dict[str, Any]:
        return {
            "text_parts": [],
            "thinking_parts": [],
            "tool_calls": {},
            "assistant_blocks": {},
            "terminal_emitted": False,
            "usage": None,
        }

    def _extract_stream_usage(self, value: Any) -> Any:
        payload = self._as_dict(value)
        if isinstance(payload, dict) and payload.get("usage") is not None:
            return payload.get("usage")
        if hasattr(value, "usage"):
            return getattr(value, "usage", None)
        return None

    def _extract_stream_delta_text(self, delta: Any) -> Optional[str]:
        delta = self._as_dict(delta)
        if isinstance(delta, dict):
            return delta.get("text")
        return getattr(delta, "text", None)

    def _extract_stream_delta_thinking(self, delta: Any) -> Optional[str]:
        delta = self._as_dict(delta)
        if isinstance(delta, dict):
            return delta.get("thinking") or delta.get("text")
        return getattr(delta, "thinking", None) or getattr(delta, "text", None)

    def _extract_stream_delta_signature(self, delta: Any) -> Optional[str]:
        delta = self._as_dict(delta)
        if isinstance(delta, dict):
            return delta.get("signature")
        return getattr(delta, "signature", None)

    def _extract_stream_delta_json(self, delta: Any) -> Optional[str]:
        delta = self._as_dict(delta)
        if isinstance(delta, dict):
            return delta.get("partial_json")
        return getattr(delta, "partial_json", None)

    @staticmethod
    def _set_stream_assistant_block(state: dict[str, Any], index: int, block: dict[str, Any]) -> dict[str, Any]:
        stored = dict(block)
        state["assistant_blocks"][index] = stored
        return stored

    def _finalize_stream_tool_calls(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index in sorted(state["tool_calls"]):
            item = dict(state["tool_calls"][index])
            input_payload = item.pop("input_json", "") or ""
            parsed_arguments = self._safe_parse_input_json(input_payload)
            if parsed_arguments:
                item["arguments"] = parsed_arguments
            else:
                fallback_input = item.pop("input", {}) or {}
                item["arguments"] = fallback_input if isinstance(fallback_input, dict) else {}
            if not item.get("id"):
                item["id"] = f"tool_call_{index}"
            calls.append(item)
        return calls

    @staticmethod
    def _safe_parse_input_json(value: str) -> dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            logger.warning("Failed to parse Anthropic tool_use input_json delta: %s", value)
        return {}

    def _build_stream_assistant_message(self, state: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        finalized_tool_calls = {
            item["id"]: item for item in self._finalize_stream_tool_calls(state) if item.get("id")
        }
        has_thinking_block = False
        for index in sorted(state["assistant_blocks"]):
            block = dict(state["assistant_blocks"][index])
            if block.get("type") in {"thinking", "redacted_thinking"}:
                has_thinking_block = True
            if block.get("type") == "tool_use" and block.get("id") in finalized_tool_calls:
                block["input"] = finalized_tool_calls[block["id"]].get("arguments", {}) or {}
            content.append(block)
        if state["thinking_parts"] and not has_thinking_block:
            content.insert(0, {"type": "thinking", "thinking": "".join(state["thinking_parts"])})
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if state["thinking_parts"]:
            message["reasoning_content"] = "".join(state["thinking_parts"])
        return message

    def _extract_anthropic_stream_events(self, event: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
        payload = self._as_dict(event)
        event_type = payload.get("type") if isinstance(payload, dict) else getattr(event, "type", None)
        events: list[dict[str, Any]] = []
        if not event_type:
            return events
        if event_type == "message_start":
            message = payload.get("message") if isinstance(payload, dict) else getattr(event, "message", None)
            usage = self._extract_stream_usage(message)
            if usage is not None:
                state["usage"] = usage
            return events
        if event_type == "content_block_start":
            index = payload.get("index", 0) if isinstance(payload, dict) else getattr(event, "index", 0)
            block = payload.get("content_block") if isinstance(payload, dict) else getattr(event, "content_block", None)
            serialized = self._serialize_block(block)
            if serialized.get("type") == "tool_use":
                self._set_stream_assistant_block(state, index, serialized)
                state["tool_calls"][index] = {
                    "id": serialized.get("id"),
                    "name": serialized.get("name", ""),
                    "input_json": "",
                    "input": serialized.get("input", {}) or {},
                }
            elif serialized.get("type") == "thinking":
                stored = self._set_stream_assistant_block(state, index, serialized)
                thinking_text = serialized.get("thinking", "")
                if thinking_text:
                    state["thinking_parts"].append(thinking_text)
                    events.append({"type": "thinking_delta", "delta": thinking_text})
                if serialized.get("signature"):
                    stored["signature"] = serialized["signature"]
            elif serialized.get("type") == "text":
                self._set_stream_assistant_block(state, index, serialized)
                text = serialized.get("text", "")
                if text:
                    state["text_parts"].append(text)
                    events.append({"type": "text_delta", "delta": text})
            elif serialized.get("type") == "redacted_thinking":
                self._set_stream_assistant_block(state, index, serialized)
            return events
        if event_type == "content_block_delta":
            index = payload.get("index", 0) if isinstance(payload, dict) else getattr(event, "index", 0)
            delta = payload.get("delta") if isinstance(payload, dict) else getattr(event, "delta", None)
            delta_type = delta.get("type") if isinstance(delta, dict) else getattr(delta, "type", None)
            if delta_type == "text_delta":
                text = self._extract_stream_delta_text(delta) or ""
                if text:
                    state["text_parts"].append(text)
                    current_block = state["assistant_blocks"].setdefault(index, {"type": "text", "text": ""})
                    current_block["text"] = current_block.get("text", "") + text
                    events.append({"type": "text_delta", "delta": text})
            elif delta_type == "thinking_delta":
                thinking = self._extract_stream_delta_thinking(delta) or ""
                if thinking:
                    state["thinking_parts"].append(thinking)
                    current_block = state["assistant_blocks"].setdefault(index, {"type": "thinking", "thinking": ""})
                    current_block["thinking"] = current_block.get("thinking", "") + thinking
                    events.append({"type": "thinking_delta", "delta": thinking})
            elif delta_type == "signature_delta":
                signature = self._extract_stream_delta_signature(delta) or ""
                if signature:
                    current_block = state["assistant_blocks"].setdefault(index, {"type": "thinking", "thinking": ""})
                    current_block["signature"] = signature
            elif delta_type == "input_json_delta":
                json_fragment = self._extract_stream_delta_json(delta) or ""
                current = state["tool_calls"].setdefault(
                    index, {"id": None, "name": "", "input_json": "", "input": {}}
                )
                current["input_json"] += json_fragment
            return events
        if event_type == "message_delta":
            usage = self._extract_stream_usage(payload) or self._extract_stream_usage(
                payload.get("delta") if isinstance(payload, dict) else getattr(event, "delta", None)
            )
            if usage is not None:
                state["usage"] = usage
            delta = payload.get("delta") if isinstance(payload, dict) else getattr(event, "delta", None)
            stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else getattr(delta, "stop_reason", None)
            if stop_reason == "tool_use":
                events.append(
                    {
                        "type": "tool_calls",
                        "tool_calls": self._finalize_stream_tool_calls(state),
                        "content": "".join(state["text_parts"]),
                        "thinking": "".join(state["thinking_parts"]),
                        "assistant_items": [self._build_stream_assistant_message(state)],
                        "usage": state.get("usage"),
                    }
                )
                state["terminal_emitted"] = True
            return events
        if event_type in {"message_stop", "message_complete"}:
            events.append(
                {
                    "type": "final_response",
                    "content": "".join(state["text_parts"]),
                    "thinking": "".join(state["thinking_parts"]),
                    "assistant_items": [self._build_stream_assistant_message(state)],
                    "usage": state.get("usage"),
                }
            )
            state["terminal_emitted"] = True
            return events
        return events

    def _finalize_anthropic_stream_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("terminal_emitted"):
            return {"type": "stream_end"}
        tool_calls = self._finalize_stream_tool_calls(state)
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
                "assistant_items": [self._build_stream_assistant_message(state)],
                "usage": state.get("usage"),
            }
        return {
            "type": "final_response",
            "content": "".join(state["text_parts"]),
            "thinking": "".join(state["thinking_parts"]),
            "assistant_items": [self._build_stream_assistant_message(state)],
            "usage": state.get("usage"),
        }

    def stream_events(self, raw_stream: Any, *, tools: bool = False) -> Generator[dict[str, Any], None, None]:
        state = self._init_anthropic_stream_state()
        for raw_event in self._iter_stream_events(raw_stream):
            for event in self._extract_anthropic_stream_events(raw_event, state):
                yield event
        final_event = self._finalize_anthropic_stream_state(state)
        if final_event.get("type") != "stream_end":
            yield final_event

    async def astream_events(self, raw_stream: Any, *, tools: bool = False) -> AsyncGenerator[dict[str, Any], None]:
        state = self._init_anthropic_stream_state()
        async for raw_event in self._aiter_stream_events(raw_stream):
            for event in self._extract_anthropic_stream_events(raw_event, state):
                yield event
        final_event = self._finalize_anthropic_stream_state(state)
        if final_event.get("type") != "stream_end":
            yield final_event
