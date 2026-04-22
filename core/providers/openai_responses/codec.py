from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator, Optional

from ...history import (
    CanonicalBlock,
    CanonicalMessage,
    _generic_canonical_block_from_content_item,
    _generic_canonical_messages_from_history_entry,
    _json_safe,
    _reasoning_text,
    coerce_canonical_message,
)
from ..openai_compat.codec import _canonical_from_openai_chat_like
from ..shared import BaseProviderCodec


def _canonical_from_responses(message: Any, provider_name: str) -> list[CanonicalMessage]:
    canonical = coerce_canonical_message(message)
    if canonical is not None:
        return [canonical]
    if isinstance(message, list):
        entries: list[CanonicalMessage] = []
        for item in message:
            entries.extend(_canonical_from_responses(item, provider_name))
        return entries
    if not isinstance(message, dict):
        return _generic_canonical_messages_from_history_entry(message, provider_name=provider_name)

    item_type = message.get("type")
    if item_type == "reasoning":
        return [
            CanonicalMessage(
                role="assistant",
                content=[
                    CanonicalBlock(
                        type="reasoning",
                        text=_reasoning_text(message),
                        summary=_json_safe(message.get("summary")),
                        signature=message.get("signature"),
                        payload=_json_safe(message),
                        metadata={"provider_block_type": "reasoning"},
                    )
                ],
                provider=provider_name,
                provider_message_type="reasoning",
            )
        ]
    if item_type == "function_call":
        return [
            CanonicalMessage(
                role="assistant",
                content=[
                    CanonicalBlock(
                        type="function_call",
                        call_id=message.get("call_id") or message.get("id"),
                        name=message.get("name"),
                        arguments=_json_safe(message.get("arguments")),
                        payload=_json_safe(message),
                        metadata={"provider_block_type": "function_call"},
                    )
                ],
                provider=provider_name,
                provider_message_type="function_call",
            )
        ]
    if item_type == "function_call_output":
        return [
            CanonicalMessage(
                role="tool",
                content=[
                    CanonicalBlock(
                        type="function_response",
                        call_id=message.get("call_id"),
                        output=_json_safe(message.get("output")),
                        payload=_json_safe(message),
                        metadata={"provider_block_type": "function_call_output"},
                    )
                ],
                provider=provider_name,
                provider_message_type="function_call_output",
            )
        ]
    if item_type == "message":
        role = str(message.get("role") or "assistant")
        content_blocks = message.get("content", [])
        blocks = [_generic_canonical_block_from_content_item(item) for item in content_blocks]
        return [
            CanonicalMessage(
                role=role,  # type: ignore[arg-type]
                content=blocks or [CanonicalBlock(type="text", text="")],
                provider=provider_name,
                provider_message_type="message",
            )
        ]
    return _canonical_from_openai_chat_like(message, provider_name)


class OpenAIResponsesCodec(BaseProviderCodec):
    def build_request_token_payload(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        pending_messages: Optional[list[Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "input": [_json_safe(item) for item in replay_history],
        }
        if pending_messages:
            payload["input"].extend(_json_safe(item) for item in pending_messages)
        if system_prompt:
            payload["instructions"] = system_prompt
        if tools:
            payload["tools"] = _json_safe(tools)
        if reasoning:
            payload["reasoning"] = _json_safe(reasoning)
        return payload

    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _canonical_from_responses(message, self.provider_name)

    def query_to_replay(self, query: str) -> list[Any]:
        return [{"role": "user", "content": query}]

    def response_to_replay(self, response: Any, *, include_reasoning: bool = False) -> list[Any]:
        output = getattr(response, "output", None)
        if not output:
            content = getattr(response, "output_text", None)
            if content:
                return [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                ]
            return []
        items: list[Any] = []
        for item in output:
            serialized = self._serialize_assistant_history_item(item, include_reasoning=include_reasoning)
            if serialized is not None:
                items.append(serialized)
        return items

    def response_to_canonical(self, response: Any, *, include_reasoning: bool = False) -> list[CanonicalMessage]:
        entries: list[CanonicalMessage] = []
        for item in self.response_to_replay(response, include_reasoning=include_reasoning):
            entries.extend(self.history_entry_to_canonical(item))
        return entries

    def tool_result_to_canonical(self, content: str, tool_id: str, tool_name: str) -> list[CanonicalMessage]:
        return [
            CanonicalMessage(
                role="tool",
                content=[
                    CanonicalBlock(
                        type="function_response",
                        call_id=tool_id,
                        name=tool_name or None,
                        output=content,
                        payload={"type": "function_call_output", "call_id": tool_id, "output": content},
                        metadata={"provider_block_type": "function_call_output"},
                    )
                ],
                provider=self.provider_name,
                provider_message_type="function_call_output",
            )
        ]

    def canonical_message_to_replay(self, message: Any) -> list[Any]:
        canonical = coerce_canonical_message(message)
        if canonical is None:
            entries: list[Any] = []
            for entry in self.history_entry_to_canonical(message):
                entries.extend(self.canonical_message_to_replay(entry))
            return entries

        items: list[Any] = []
        text_fragments: list[str] = []

        for block in canonical.content:
            payload = block.payload if isinstance(block.payload, dict) else None
            if block.type == "text" and block.text:
                text_fragments.append(block.text)
                continue
            if block.type == "reasoning":
                if isinstance(payload, dict) and payload.get("type") == "reasoning":
                    items.append(dict(payload))
                else:
                    item: dict[str, Any] = {"type": "reasoning"}
                    if block.text:
                        item["summary"] = [{"type": "summary_text", "text": block.text}]
                    if block.signature:
                        item["signature"] = block.signature
                    items.append(item)
                continue
            if block.type == "function_call":
                if isinstance(payload, dict) and payload.get("type") == "function_call":
                    items.append(dict(payload))
                else:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block.call_id,
                            "name": block.name or "",
                            "arguments": block.arguments if isinstance(block.arguments, str) else json.dumps(block.arguments, ensure_ascii=False, default=str),
                        }
                    )
                continue
            if block.type == "function_response":
                if isinstance(payload, dict) and payload.get("type") == "function_call_output":
                    items.append(dict(payload))
                else:
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.call_id,
                            "output": _json_safe(block.output),
                        }
                    )
                continue
            if block.type == "provider_item" and isinstance(payload, dict):
                items.append(dict(payload))

        if text_fragments:
            text = "".join(text_fragments)
            if canonical.role == "assistant":
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            else:
                items.append({"role": canonical.role, "content": text})

        if not items:
            items.append({"role": canonical.role, "content": ""})
        return items

    def is_request_ready_message(self, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        item_type = message.get("type")
        if item_type in {"message", "function_call", "function_call_output", "reasoning"}:
            return True
        return "role" in message and "content" in message

    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, Any]:
        return {"type": "function_call_output", "call_id": tool_id, "output": content}

    def build_assistant_message(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if thinking:
            result.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": thinking}]})
        if tool_calls:
            for tool_call in tool_calls:
                result.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                )
        if content:
            result.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            )
        return result

    def has_tool_calls(self, response: Any) -> bool:
        output = getattr(response, "output", None)
        if output is None:
            return False
        return any(getattr(item, "type", None) == "function_call" for item in output)

    def get_tool_calls(self, response: Any) -> list[Any]:
        output = getattr(response, "output", None)
        if not output:
            return []
        return [item for item in output if getattr(item, "type", None) == "function_call"]

    def get_thinking_content(self, response: Any) -> Optional[str]:
        output = getattr(response, "output", None)
        if not output:
            return None
        reasoning_text = []
        for item in output:
            if getattr(item, "type", None) == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    if isinstance(summary, list):
                        parts = []
                        for s in summary:
                            text = getattr(s, "text", None) or (s if isinstance(s, str) else "")
                            if text:
                                parts.append(text)
                        if parts:
                            reasoning_text.append("\n".join(parts))
                    else:
                        reasoning_text.append(str(summary))
        if reasoning_text:
            return "\n".join(reasoning_text)
        return None

    @staticmethod
    def _extract_output_message_text(item: Any) -> str:
        if getattr(item, "type", None) != "message":
            return ""
        content = getattr(item, "content", None)
        parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "output_text":
                    text = getattr(block, "text", None) or ""
                    if text:
                        parts.append(text)
        elif isinstance(content, str):
            parts.append(content)
        return "".join(parts)

    @staticmethod
    def _select_message_item(items: list[Any]) -> Optional[Any]:
        if not items:
            return None
        assistant_messages = []
        final_messages = []
        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type != "message":
                continue
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            phase = item.get("phase") if isinstance(item, dict) else getattr(item, "phase", None)
            if role == "assistant":
                assistant_messages.append(item)
                if phase == "final_answer":
                    final_messages.append(item)
        if final_messages:
            return final_messages[-1]
        if assistant_messages:
            return assistant_messages[-1]
        return None

    def get_response_content(self, response: Any) -> Optional[str]:
        output = getattr(response, "output", None)
        if not output:
            return getattr(response, "output_text", None)
        selected = self._select_message_item(output)
        if selected is not None:
            content = self._extract_output_message_text(selected)
            if content:
                return content
        return getattr(response, "output_text", None)

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [cls._to_serializable(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_serializable(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._to_serializable(item) for key, item in value.items()}
        if hasattr(value, "to_dict"):
            payload = value.to_dict()
            if isinstance(payload, dict):
                return cls._to_serializable(payload)
        payload: dict[str, Any] = {}
        for attr in (
            "type",
            "text",
            "id",
            "call_id",
            "name",
            "arguments",
            "content",
            "summary",
            "role",
            "status",
            "phase",
        ):
            attr_value = getattr(value, attr, None)
            if attr_value is not None:
                payload[attr] = cls._to_serializable(attr_value)
        if payload:
            return payload
        return str(value)

    def _serialize_output_item(self, item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            payload = item.to_dict()
            if isinstance(payload, dict):
                return self._to_serializable(payload)
        if isinstance(item, dict):
            return self._to_serializable(dict(item))
        payload = {"type": getattr(item, "type", "unknown")}
        for attr in ("id", "call_id", "name", "arguments", "role", "status", "phase"):
            value = getattr(item, attr, None)
            if value is not None:
                payload[attr] = self._to_serializable(value)
        if hasattr(item, "content"):
            payload["content"] = self._to_serializable(getattr(item, "content"))
        if hasattr(item, "summary"):
            payload["summary"] = self._to_serializable(getattr(item, "summary"))
        if hasattr(item, "text"):
            payload["text"] = self._to_serializable(getattr(item, "text"))
        if hasattr(item, "encrypted_content"):
            payload["encrypted_content"] = self._to_serializable(getattr(item, "encrypted_content"))
        return payload

    def _serialize_assistant_history_item(self, item: Any, include_reasoning: bool = False) -> Optional[dict[str, Any]]:
        payload = self._serialize_output_item(item)
        item_type = payload.get("type")
        if item_type == "reasoning":
            if not include_reasoning:
                return None
            reasoning_item: dict[str, Any] = {"type": "reasoning"}
            for key in ("id", "summary", "content", "encrypted_content", "status"):
                if payload.get(key) is not None:
                    reasoning_item[key] = self._to_serializable(payload.get(key))
            return reasoning_item
        if item_type == "message":
            message: dict[str, Any] = {
                "type": "message",
                "role": payload.get("role", "assistant"),
                "content": self._to_serializable(payload.get("content", [])),
            }
            phase = payload.get("phase")
            if phase:
                message["phase"] = phase
            return message
        if item_type == "function_call":
            return {
                "type": "function_call",
                "call_id": payload.get("call_id") or payload.get("id"),
                "name": payload.get("name", ""),
                "arguments": payload.get("arguments", ""),
            }
        if item_type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": payload.get("call_id"),
                "output": payload.get("output", ""),
            }
        return payload

    def build_assistant_response(self, response: Any, include_reasoning: bool = False) -> list[Any]:
        result = []
        for item in getattr(response, "output", []):
            serialized = self._serialize_assistant_history_item(item, include_reasoning=include_reasoning)
            if serialized is not None:
                result.append(serialized)
        return result

    def _init_responses_tool_stream_state(self) -> dict[str, Any]:
        return {
            "text_parts": [],
            "thinking_parts": [],
            "tool_calls": {},
            "output_items": [],
            "output_item_keys": {},
            "terminal_emitted": False,
            "usage": None,
        }

    def _set_stream_output_item(self, state: dict[str, Any], item: Any, serialized_item: dict[str, Any]) -> None:
        key = self._get_stream_output_item_key(item)
        if key in state["output_item_keys"]:
            state["output_items"][state["output_item_keys"][key]] = serialized_item
            return
        state["output_item_keys"][key] = len(state["output_items"])
        state["output_items"].append(serialized_item)

    @staticmethod
    def _get_stream_output_item_key(item: Any) -> str:
        return (
            str(getattr(item, "id", None) or "")
            or str(getattr(item, "call_id", None) or "")
            or f"{getattr(item, 'type', 'unknown')}:{getattr(item, 'name', None) or ''}:{getattr(item, 'role', None) or ''}"
        )

    def _get_stream_output_item(self, state: dict[str, Any], key: Any) -> Optional[dict[str, Any]]:
        index = state["output_item_keys"].get(str(key))
        if index is None:
            return None
        item = state["output_items"][index]
        if isinstance(item, dict):
            return item
        return None

    def _merge_responses_function_call_delta(self, event: Any, state: dict[str, Any]) -> None:
        key = getattr(event, "item_id", None) or getattr(event, "call_id", None) or str(
            getattr(event, "output_index", 0)
        )
        current = state["tool_calls"].setdefault(
            key,
            {"id": getattr(event, "call_id", None), "name": "", "arguments": "", "type": "function"},
        )
        if getattr(event, "call_id", None):
            current["id"] = event.call_id
        delta = getattr(event, "delta", None) or ""
        if delta:
            current["arguments"] += delta
        output_item = self._get_stream_output_item(state, key)
        if output_item is not None:
            output_item["type"] = "function_call"
            if current.get("id"):
                output_item["call_id"] = current["id"]
            if current.get("name"):
                output_item["name"] = current["name"]
            output_item["arguments"] = current["arguments"]

    def _merge_responses_output_item(self, item: Any, state: dict[str, Any]) -> None:
        item_type = getattr(item, "type", None)
        serialized_item = self._serialize_output_item(item)
        self._set_stream_output_item(state, item, serialized_item)
        if item_type == "function_call":
            key = (
                getattr(item, "id", None)
                or getattr(item, "call_id", None)
                or getattr(item, "name", None)
                or str(len(state["tool_calls"]))
            )
            current = state["tool_calls"].setdefault(
                key,
                {"id": getattr(item, "call_id", None), "name": "", "arguments": "", "type": "function"},
            )
            if getattr(item, "call_id", None):
                current["id"] = item.call_id
            if getattr(item, "name", None):
                current["name"] = item.name
            arguments = getattr(item, "arguments", None)
            if arguments:
                current["arguments"] = arguments
        elif item_type == "message":
            message_text = self._extract_output_message_text(item)
            if message_text:
                current_text = "".join(state["text_parts"])
                if not current_text.endswith(message_text):
                    state["text_parts"].append(message_text)

    def _normalize_responses_tool_calls(self, tool_calls_by_key: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, key in enumerate(tool_calls_by_key):
            tool_call = dict(tool_calls_by_key[key])
            if not tool_call.get("id"):
                tool_call["id"] = f"call_{index}"
            if not tool_call.get("arguments"):
                tool_call["arguments"] = json.dumps({})
            normalized.append(tool_call)
        return normalized

    def _build_stream_assistant_items(self, state: dict[str, Any], tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assistant_items = []
        for item in state.get("output_items", []):
            if not isinstance(item, dict):
                continue
            serialized = self._serialize_assistant_history_item(item, include_reasoning=True)
            if serialized is not None:
                assistant_items.append(serialized)
        if assistant_items:
            return assistant_items
        return self.build_assistant_message(
            content="".join(state.get("text_parts", [])),
            tool_calls=tool_calls or None,
            thinking="".join(state.get("thinking_parts", [])) or None,
        )

    def _extract_responses_stream_events(self, event: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        event_type = getattr(event, "type", None)
        if not event_type:
            return events
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = getattr(event, "delta", None) or ""
            if delta:
                state["text_parts"].append(delta)
                events.append({"type": "text_delta", "delta": delta})
            return events
        if event_type in {"response.reasoning.delta", "response.reasoning_summary_text.delta"}:
            delta = getattr(event, "delta", None) or ""
            if delta:
                state["thinking_parts"].append(delta)
                events.append({"type": "thinking_delta", "delta": delta})
            return events
        if event_type == "response.function_call_arguments.delta":
            self._merge_responses_function_call_delta(event, state)
            return events
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            item = getattr(event, "item", None)
            if item is not None:
                self._merge_responses_output_item(item, state)
                if getattr(item, "type", None) == "function_call" and event_type.endswith(".done"):
                    state["terminal_emitted"] = True
            return events
        if event_type == "response.completed":
            response = getattr(event, "response", None)
            usage = getattr(response, "usage", None)
            if usage is not None:
                state["usage"] = usage
            tool_calls = self._normalize_responses_tool_calls(state["tool_calls"])
            if tool_calls:
                events.append(
                    {
                        "type": "tool_calls",
                        "tool_calls": tool_calls,
                        "content": "".join(state["text_parts"]),
                        "thinking": "".join(state["thinking_parts"]),
                        "assistant_items": self._build_stream_assistant_items(state, tool_calls),
                        "usage": state.get("usage"),
                    }
                )
            else:
                selected_message = self._select_message_item(state.get("output_items", []))
                content = (
                    self._extract_output_message_text(selected_message) if selected_message is not None else ""
                ) or "".join(state["text_parts"]) or self.get_response_content(response) or ""
                events.append(
                    {
                        "type": "final_response",
                        "content": content,
                        "thinking": "".join(state["thinking_parts"]),
                        "assistant_items": self._build_stream_assistant_items(state, tool_calls),
                        "usage": state.get("usage"),
                    }
                )
            state["terminal_emitted"] = True
            return events
        return events

    def _finalize_responses_tool_stream_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("terminal_emitted"):
            return {"type": "stream_end"}
        tool_calls = self._normalize_responses_tool_calls(state["tool_calls"])
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
                "assistant_items": self._build_stream_assistant_items(state, tool_calls),
                "usage": state.get("usage"),
            }
        return {
            "type": "final_response",
            "content": "".join(state["text_parts"]),
            "thinking": "".join(state["thinking_parts"]),
            "assistant_items": self._build_stream_assistant_items(state, tool_calls),
            "usage": state.get("usage"),
        }

    def stream_events(self, raw_stream: Any, *, tools: bool = False) -> Generator[dict[str, Any], None, None]:
        state = self._init_responses_tool_stream_state()
        for event in raw_stream:
            for item in self._extract_responses_stream_events(event, state):
                if tools or item.get("type") != "tool_calls":
                    yield item
        final_event = self._finalize_responses_tool_stream_state(state)
        if final_event.get("type") != "stream_end" and (tools or final_event.get("type") != "tool_calls"):
            yield final_event

    async def astream_events(self, raw_stream: Any, *, tools: bool = False) -> AsyncGenerator[dict[str, Any], None]:
        state = self._init_responses_tool_stream_state()
        async for event in raw_stream:
            for item in self._extract_responses_stream_events(event, state):
                if tools or item.get("type") != "tool_calls":
                    yield item
        final_event = self._finalize_responses_tool_stream_state(state)
        if final_event.get("type") != "stream_end" and (tools or final_event.get("type") != "tool_calls"):
            yield final_event
