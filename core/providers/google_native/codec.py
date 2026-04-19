from __future__ import annotations

import json
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


def _canonical_block_from_google_part(part: Any) -> CanonicalBlock:
    block = GoogleNativeCodec._serialize_gemini_part(part)
    if block["type"] == "thinking":
        return CanonicalBlock(
            type="reasoning",
            text=block.get("text", ""),
            signature=block.get("thought_signature"),
            payload=_json_safe(block),
            metadata={"provider_block_type": "thinking"},
        )
    if block["type"] == "text":
        return CanonicalBlock(
            type="text",
            text=block.get("text", ""),
            payload=_json_safe(block),
            metadata={"provider_block_type": "text"},
        )
    if block["type"] == "function_call":
        return CanonicalBlock(
            type="function_call",
            call_id=block.get("id"),
            name=block.get("name"),
            arguments=_json_safe(block.get("args", {})),
            signature=block.get("thought_signature"),
            payload=_json_safe(block),
            metadata={"provider_block_type": "function_call"},
        )
    if block["type"] == "function_response":
        return CanonicalBlock(
            type="function_response",
            call_id=block.get("id"),
            name=block.get("name"),
            output=_json_safe(block.get("response", {})),
            signature=block.get("thought_signature"),
            payload=_json_safe(block),
            metadata={"provider_block_type": "function_response"},
        )
    return CanonicalBlock(type="provider_item", payload=_json_safe(block))


def _canonical_from_google_native(message: Any, provider_name: str) -> list[CanonicalMessage]:
    canonical = coerce_canonical_message(message)
    if canonical is not None:
        return [canonical]
    if isinstance(message, list):
        entries: list[CanonicalMessage] = []
        for item in message:
            entries.extend(_canonical_from_google_native(item, provider_name))
        return entries
    if not isinstance(message, dict):
        return _generic_canonical_messages_from_history_entry(message, provider_name=provider_name)

    raw_role = str(message.get("role") or "user")
    role_map = {"model": "assistant", "assistant": "assistant", "system": "system", "user": "user"}
    role = role_map.get(raw_role, "assistant")
    parts = message.get("parts")
    blocks: list[CanonicalBlock] = []
    if isinstance(parts, list):
        blocks.extend(_canonical_block_from_google_part(part) for part in parts)
    else:
        content = message.get("content")
        if isinstance(content, list):
            blocks.extend(_canonical_block_from_google_part(part) for part in content)
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


class GoogleNativeCodec(BaseProviderCodec):
    def build_request_token_payload(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        pending_messages: Optional[list[Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Any:
        contents = [_json_safe(item) for item in replay_history]
        if pending_messages:
            contents.extend(_json_safe(item) for item in pending_messages)
        payload: dict[str, Any] = {"contents": contents}
        config: dict[str, Any] = {}
        if system_prompt:
            config["system_instruction"] = system_prompt
        if tools:
            config["tools"] = _json_safe(tools)
        if reasoning:
            config["reasoning"] = _json_safe(reasoning)
        if config:
            payload["config"] = config
        return payload

    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _canonical_from_google_native(message, self.provider_name)

    def query_to_replay(self, query: str) -> list[Any]:
        return [{"role": "user", "parts": [{"text": query}]}]

    def append_replay_entry(self, prepared: list[Any], item: Any) -> None:
        if self._is_function_response_turn(item) and prepared and self._is_function_response_turn(prepared[-1]):
            prepared[-1]["parts"].extend(item["parts"])
            return
        prepared.append(item)

    def is_request_ready_message(self, message: Any) -> bool:
        return isinstance(message, dict) and message.get("role") in {"system", "user", "model"} and isinstance(
            message.get("parts"), list
        )

    @staticmethod
    def _is_function_response_turn(message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if message.get("role") != "user":
            return False
        parts = message.get("parts")
        if not isinstance(parts, list) or not parts:
            return False
        return all(isinstance(part, dict) and "function_response" in part for part in parts)

    @staticmethod
    def _serialize_gemini_part(part: Any) -> dict[str, Any]:
        if isinstance(part, dict):
            block_type = part.get("type")
            if block_type == "thinking":
                payload = {"type": "thinking", "text": part.get("text") or part.get("thinking", "")}
                signature = part.get("thought_signature") or part.get("thoughtSignature")
                if signature:
                    payload["thought_signature"] = signature
                return payload
            if block_type == "text":
                return {"type": "text", "text": part.get("text", "")}
            if block_type == "function_call":
                return {
                    "type": "function_call",
                    "id": part.get("id"),
                    "name": part.get("name", ""),
                    "args": part.get("args", {}) or {},
                    **(
                        {"thought_signature": part["thought_signature"]}
                        if part.get("thought_signature") is not None
                        else {}
                    ),
                }
            if block_type == "function_response":
                return {
                    "type": "function_response",
                    "id": part.get("id"),
                    "name": part.get("name", ""),
                    "response": part.get("response", {}) or {},
                    **(
                        {"thought_signature": part["thought_signature"]}
                        if part.get("thought_signature") is not None
                        else {}
                    ),
                }
            if part.get("text") is not None or part.get("thought") is not None:
                text = part.get("text", "")
                thought_flag = part.get("thought")
                thought_signature = part.get("thought_signature") or part.get("thoughtSignature")
                block = {"type": "thinking" if thought_flag else "text", "text": text or ""}
                if thought_signature is not None:
                    block["thought_signature"] = thought_signature
                return block
            if isinstance(part.get("function_call"), dict):
                function_call = part["function_call"]
                block = {
                    "type": "function_call",
                    "id": function_call.get("id"),
                    "name": function_call.get("name", ""),
                    "args": function_call.get("args", {}) or {},
                }
                if part.get("thought_signature") is not None:
                    block["thought_signature"] = part["thought_signature"]
                return block
            if isinstance(part.get("function_response"), dict):
                function_response = part["function_response"]
                block = {
                    "type": "function_response",
                    "id": function_response.get("id"),
                    "name": function_response.get("name", ""),
                    "response": function_response.get("response", {}) or {},
                }
                if part.get("thought_signature") is not None:
                    block["thought_signature"] = part["thought_signature"]
                return block
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            block = {
                "type": "function_call",
                "id": getattr(function_call, "id", None),
                "name": getattr(function_call, "name", ""),
                "args": getattr(function_call, "args", None) or {},
            }
            thought_signature = getattr(part, "thought_signature", None)
            if thought_signature is not None:
                block["thought_signature"] = thought_signature
            return block
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            block = {
                "type": "function_response",
                "id": getattr(function_response, "id", None),
                "name": getattr(function_response, "name", ""),
                "response": getattr(function_response, "response", None) or {},
            }
            thought_signature = getattr(part, "thought_signature", None)
            if thought_signature is not None:
                block["thought_signature"] = thought_signature
            return block
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        thought_flag = getattr(part, "thought", None)
        if thought_flag is None and isinstance(part, dict):
            thought_flag = part.get("thought")
        thought_signature = getattr(part, "thought_signature", None)
        if thought_signature is None and isinstance(part, dict):
            thought_signature = part.get("thought_signature") or part.get("thoughtSignature")
        block = {"type": "thinking" if thought_flag else "text", "text": text or ""}
        if thought_signature is not None:
            block["thought_signature"] = thought_signature
        return block

    @classmethod
    def _candidate_content(cls, response: Any) -> Any:
        if response is None:
            return None
        candidates = getattr(response, "candidates", None)
        if candidates:
            candidate = candidates[0]
            return getattr(candidate, "content", None) or (
                candidate.get("content") if isinstance(candidate, dict) else None
            )
        if isinstance(response, dict):
            return response.get("content")
        return getattr(response, "content", None)

    @classmethod
    def _content_parts(cls, content: Any) -> list[Any]:
        if content is None:
            return []
        if isinstance(content, dict):
            return list(content.get("parts", []) or [])
        parts = getattr(content, "parts", None)
        return list(parts or [])

    @staticmethod
    def _part_key(block: dict[str, Any]) -> tuple[Any, ...]:
        block_type = block.get("type")
        if block_type == "thinking":
            return ("thinking", block.get("text"), block.get("thought_signature"))
        if block_type == "text":
            return ("text", block.get("text"))
        if block_type == "function_call":
            return (
                "function_call",
                block.get("id"),
                block.get("name"),
                json.dumps(block.get("args", {}), ensure_ascii=False, sort_keys=True),
            )
        if block_type == "function_response":
            return (
                "function_response",
                block.get("id"),
                block.get("name"),
                json.dumps(block.get("response", {}), ensure_ascii=False, sort_keys=True),
            )
        return (block_type, json.dumps(block, ensure_ascii=False, sort_keys=True))

    @classmethod
    def _append_stream_block(cls, blocks: list[dict[str, Any]], seen: set[tuple[Any, ...]], block: dict[str, Any]) -> None:
        key = cls._part_key(block)
        if key in seen:
            return
        seen.add(key)
        blocks.append(block)

    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, Any]:
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": tool_id,
                        "name": tool_name,
                        "response": {"result": content},
                    }
                }
            ],
        }

    def tool_result_to_canonical(self, content: str, tool_id: str, tool_name: str) -> list[CanonicalMessage]:
        return [
            CanonicalMessage(
                role="tool",
                content=[
                    CanonicalBlock(
                        type="function_response",
                        call_id=tool_id,
                        name=tool_name,
                        output={"result": content},
                        payload={
                            "type": "function_response",
                            "id": tool_id,
                            "name": tool_name,
                            "response": {"result": content},
                        },
                        metadata={"provider_block_type": "function_response"},
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
        parts: list[dict[str, Any]] = []
        if thinking:
            parts.append({"text": thinking, "thought": True})
        if content:
            parts.append({"text": content})
        for tool_call in tool_calls or []:
            try:
                arguments = json.loads(tool_call["arguments"])
            except Exception:
                arguments = {}
            parts.append(
                {
                    "function_call": {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "args": arguments,
                    }
                }
            )
        return {"role": "model", "parts": parts}

    def build_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict[str, Any]:
        content = self._candidate_content(response)
        parts: list[dict[str, Any]] = []
        for part in self._content_parts(content):
            block = self._serialize_gemini_part(part)
            if block["type"] == "thinking" and not include_reasoning:
                continue
            if block["type"] == "thinking":
                payload = {"text": block.get("text", ""), "thought": True}
                if block.get("thought_signature"):
                    payload["thought_signature"] = block["thought_signature"]
                parts.append(payload)
            elif block["type"] == "text":
                parts.append({"text": block.get("text", "")})
            elif block["type"] == "function_call":
                payload = {
                    "function_call": {
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "args": block.get("args", {}) or {},
                    }
                }
                if block.get("thought_signature"):
                    payload["thought_signature"] = block["thought_signature"]
                parts.append(payload)
            elif block["type"] == "function_response":
                payload = {
                    "function_response": {
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "response": block.get("response", {}) or {},
                    }
                }
                if block.get("thought_signature"):
                    payload["thought_signature"] = block["thought_signature"]
                parts.append(payload)
        return {"role": "model", "parts": parts}

    def response_to_replay(self, response: Any, *, include_reasoning: bool = False) -> list[Any]:
        if response is None:
            return []
        return [self.build_assistant_response(response, include_reasoning=include_reasoning)]

    def response_to_canonical(self, response: Any, *, include_reasoning: bool = False) -> list[CanonicalMessage]:
        if response is None:
            return []
        content = self._candidate_content(response)
        blocks: list[CanonicalBlock] = []
        for part in self._content_parts(content):
            block = self._serialize_gemini_part(part)
            if block["type"] == "thinking" and not include_reasoning:
                continue
            if block["type"] == "thinking":
                blocks.append(
                    CanonicalBlock(
                        type="reasoning",
                        text=block.get("text", ""),
                        signature=block.get("thought_signature"),
                        payload=_json_safe(block),
                        metadata={"provider_block_type": "thinking"},
                    )
                )
            elif block["type"] == "text":
                blocks.append(
                    CanonicalBlock(
                        type="text",
                        text=block.get("text", ""),
                        payload=_json_safe(block),
                        metadata={"provider_block_type": "text"},
                    )
                )
            elif block["type"] == "function_call":
                blocks.append(
                    CanonicalBlock(
                        type="function_call",
                        call_id=block.get("id"),
                        name=block.get("name"),
                        arguments=_json_safe(block.get("args", {})),
                        signature=block.get("thought_signature"),
                        payload=_json_safe(block),
                        metadata={"provider_block_type": "function_call"},
                    )
                )
            elif block["type"] == "function_response":
                blocks.append(
                    CanonicalBlock(
                        type="function_response",
                        call_id=block.get("id"),
                        name=block.get("name"),
                        output=_json_safe(block.get("response", {})),
                        signature=block.get("thought_signature"),
                        payload=_json_safe(block),
                        metadata={"provider_block_type": "function_response"},
                    )
                )
        if not blocks:
            blocks = [CanonicalBlock(type="provider_item", payload=_json_safe(response))]
        return [
            CanonicalMessage(
                role="assistant",
                content=blocks,
                provider=self.provider_name,
                provider_message_type="model",
            )
        ]

    def canonical_message_to_replay(self, message: Any) -> list[Any]:
        canonical = coerce_canonical_message(message)
        if canonical is None:
            entries: list[Any] = []
            for entry in self.history_entry_to_canonical(message):
                entries.extend(self.canonical_message_to_replay(entry))
            return entries

        parts: list[dict[str, Any]] = []
        for block in canonical.content:
            payload = block.payload if isinstance(block.payload, dict) else None
            if block.type == "text":
                if block.text:
                    parts.append({"text": block.text})
                continue
            if block.type == "reasoning":
                text = block.text or _reasoning_text(block.summary or block.payload)
                part = {"text": text, "thought": True}
                signature = block.signature
                if signature is None and isinstance(payload, dict):
                    signature = payload.get("thought_signature") or payload.get("thoughtSignature")
                if signature is not None:
                    part["thought_signature"] = signature
                parts.append(part)
                continue
            if block.type == "function_call":
                if isinstance(payload, dict):
                    provider_part = self._google_part_from_provider_payload(payload)
                    if provider_part is not None:
                        parts.append(provider_part)
                        continue
                parts.append(
                    {
                        "function_call": {
                            "id": block.call_id,
                            "name": block.name or "",
                            "args": self._dict_arguments(block.arguments),
                        },
                        **({"thought_signature": block.signature} if block.signature is not None else {}),
                    }
                )
                continue
            if block.type == "function_response":
                if isinstance(payload, dict):
                    provider_part = self._google_part_from_provider_payload(payload)
                    if provider_part is not None:
                        parts.append(provider_part)
                        continue
                parts.append(
                    {
                        "function_response": {
                            "id": block.call_id,
                            "name": block.name or "",
                            "response": self._function_response_payload(block.output),
                        },
                        **({"thought_signature": block.signature} if block.signature is not None else {}),
                    }
                )
                continue
            if block.type == "provider_item" and isinstance(payload, dict):
                provider_part = self._google_part_from_provider_payload(payload)
                if provider_part is not None:
                    parts.append(provider_part)

        if canonical.role == "system":
            return [{"role": "system", "parts": parts}]
        if canonical.role == "assistant":
            return [{"role": "model", "parts": parts}]
        return [{"role": "user", "parts": parts}]

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

    @staticmethod
    def _function_response_payload(output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            return dict(output)
        return {"result": output}

    @staticmethod
    def _google_part_from_provider_payload(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if "text" in payload or "function_call" in payload or "function_response" in payload or "inline_data" in payload:
            return dict(payload)
        block_type = payload.get("type")
        if block_type == "text":
            return {"text": payload.get("text", "")}
        if block_type == "thinking":
            part = {"text": payload.get("text") or payload.get("thinking", ""), "thought": True}
            signature = payload.get("thought_signature") or payload.get("thoughtSignature")
            if signature is not None:
                part["thought_signature"] = signature
            return part
        if block_type == "function_call":
            part = {
                "function_call": {
                    "id": payload.get("id"),
                    "name": payload.get("name", ""),
                    "args": payload.get("args", {}) or {},
                }
            }
            signature = payload.get("thought_signature") or payload.get("thoughtSignature")
            if signature is not None:
                part["thought_signature"] = signature
            return part
        if block_type == "function_response":
            part = {
                "function_response": {
                    "id": payload.get("id"),
                    "name": payload.get("name", ""),
                    "response": payload.get("response", {}) or {},
                }
            }
            signature = payload.get("thought_signature") or payload.get("thoughtSignature")
            if signature is not None:
                part["thought_signature"] = signature
            return part
        return None

    def get_thinking_content(self, response: Any) -> Optional[str]:
        content = self._candidate_content(response)
        thoughts: list[str] = []
        for part in self._content_parts(content):
            block = self._serialize_gemini_part(part)
            if block["type"] == "thinking" and block.get("text"):
                thoughts.append(block["text"])
        return "".join(thoughts) or None

    def get_response_content(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        direct_text = getattr(response, "text", None)
        if direct_text:
            return direct_text
        content = self._candidate_content(response)
        texts: list[str] = []
        for part in self._content_parts(content):
            block = self._serialize_gemini_part(part)
            if block["type"] == "text" and block.get("text"):
                texts.append(block["text"])
        return "".join(texts) or None

    def has_tool_calls(self, response: Any) -> bool:
        return bool(self.get_tool_calls(response))

    def get_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for index, part in enumerate(self._content_parts(self._candidate_content(response))):
            block = self._serialize_gemini_part(part)
            if block["type"] != "function_call":
                continue
            tool_calls.append(
                {
                    "id": block.get("id") or f"tool_call_{index}",
                    "name": block.get("name", ""),
                    "arguments": block.get("args", {}) or {},
                }
            )
        return tool_calls

    def _build_stream_assistant_message(
        self,
        *,
        text: str,
        thinking: str,
        raw_blocks: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        if raw_blocks:
            for block in raw_blocks:
                if block["type"] == "thinking":
                    payload = {"text": block.get("text", ""), "thought": True}
                    if block.get("thought_signature"):
                        payload["thought_signature"] = block["thought_signature"]
                    parts.append(payload)
                elif block["type"] == "text":
                    parts.append({"text": block.get("text", "")})
                elif block["type"] == "function_call":
                    payload = {
                        "function_call": {
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "args": block.get("args", {}) or {},
                        }
                    }
                    if block.get("thought_signature"):
                        payload["thought_signature"] = block["thought_signature"]
                    parts.append(payload)
                elif block["type"] == "function_response":
                    parts.append(
                        {
                            "function_response": {
                                "id": block.get("id"),
                                "name": block.get("name", ""),
                                "response": block.get("response", {}) or {},
                            }
                        }
                    )
        if not parts:
            if thinking:
                parts.append({"text": thinking, "thought": True})
            if text:
                parts.append({"text": text})
        return {"role": "model", "parts": parts}

    def stream_events(self, raw_stream: Any, *, tools: bool = False) -> Generator[dict[str, Any], None, None]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        seen_block_keys: set[tuple[Any, ...]] = set()
        for chunk in raw_stream:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(chunk_text)
                yield {"type": "text_delta", "delta": chunk_text}
            for part in self._content_parts(self._candidate_content(chunk)):
                block = self._serialize_gemini_part(part)
                if block["type"] == "thinking":
                    thinking_text = block.get("text", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)
                        yield {"type": "thinking_delta", "delta": thinking_text}
                    self._append_stream_block(assistant_blocks, seen_block_keys, block)
                elif block["type"] == "function_call":
                    self._append_stream_block(assistant_blocks, seen_block_keys, block)
                    function_calls.append(
                        {
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "arguments": block.get("args", {}) or {},
                        }
                    )
        if function_calls:
            yield {
                "type": "tool_calls",
                "tool_calls": function_calls,
                "content": "".join(text_parts),
                "thinking": "".join(thinking_parts),
                "assistant_items": [
                    self._build_stream_assistant_message(
                        text="".join(text_parts),
                        thinking="".join(thinking_parts),
                        raw_blocks=assistant_blocks,
                    )
                ],
            }
            return
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "assistant_items": [
                self._build_stream_assistant_message(
                    text="".join(text_parts),
                    thinking="".join(thinking_parts),
                    raw_blocks=assistant_blocks,
                )
            ],
        }

    async def astream_events(self, raw_stream: Any, *, tools: bool = False) -> AsyncGenerator[dict[str, Any], None]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        seen_block_keys: set[tuple[Any, ...]] = set()
        async for chunk in raw_stream:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(chunk_text)
                yield {"type": "text_delta", "delta": chunk_text}
            for part in self._content_parts(self._candidate_content(chunk)):
                block = self._serialize_gemini_part(part)
                if block["type"] == "thinking":
                    thinking_text = block.get("text", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)
                        yield {"type": "thinking_delta", "delta": thinking_text}
                    self._append_stream_block(assistant_blocks, seen_block_keys, block)
                elif block["type"] == "function_call":
                    self._append_stream_block(assistant_blocks, seen_block_keys, block)
                    function_calls.append(
                        {
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "arguments": block.get("args", {}) or {},
                        }
                    )
        if function_calls:
            yield {
                "type": "tool_calls",
                "tool_calls": function_calls,
                "content": "".join(text_parts),
                "thinking": "".join(thinking_parts),
                "assistant_items": [
                    self._build_stream_assistant_message(
                        text="".join(text_parts),
                        thinking="".join(thinking_parts),
                        raw_blocks=assistant_blocks,
                    )
                ],
            }
            return
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "assistant_items": [
                self._build_stream_assistant_message(
                    text="".join(text_parts),
                    thinking="".join(thinking_parts),
                    raw_blocks=assistant_blocks,
                )
            ],
        }
