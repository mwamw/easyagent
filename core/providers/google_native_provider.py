"""
Google Provider

使用 Google 原生 `google-genai` Python SDK 调用 Gemini API。
"""

from __future__ import annotations

from typing import Optional, Any, Generator, AsyncGenerator
import json
import logging

from .base import BaseProvider

logger = logging.getLogger(__name__)


class GoogleNativeProvider(BaseProvider):
    """
    Google Gemini Provider

    使用 `google.genai.Client` / `client.aio` 原生 SDK，而非 OpenAI 兼容层。
    """

    def _create_client(self) -> Any:
        injected = self.kwargs.get("client")
        if injected is not None:
            return injected
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GoogleProvider requires the `google-genai` package. "
                "Install it with `pip install google-genai`."
            ) from exc
        client_args: dict[str, Any] = {"api_key": self.api_key}
        if getattr(self, "base_url", None):
            client_args["http_options"] = {"base_url": self.base_url}
            
        return genai.Client(**client_args)

    def _get_async_client(self) -> Any:
        injected = self.kwargs.get("async_client")
        if injected is not None:
            return injected
        if self._async_client is None:
            client = self.client or self._create_client()
            aio_client = getattr(client, "aio", None)
            if aio_client is None:
                raise RuntimeError("google-genai client does not expose `.aio` async access.")
            self._async_client = aio_client
        return self._async_client

    def _build_thinking_config(self, reasoning: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not reasoning:
            return None
        if "thinking_config" in reasoning and isinstance(reasoning["thinking_config"], dict):
            return dict(reasoning["thinking_config"])
        if "thinking_budget" in reasoning:
            return {"thinking_budget": reasoning["thinking_budget"]}
        if "thinking_level" in reasoning:
            return {"thinking_level": reasoning["thinking_level"]}
        effort = reasoning.get("effort")
        if not effort:
            return None
        if self.model and "gemini-3" in self.model.lower():
            level_map = {
                "low": "low",
                "medium": "medium",
                "high": "high",
            }
            return {"thinking_level": level_map.get(str(effort), "medium")}
        budget_map = {
            "low": 1024,
            "medium": 8192,
            "high": 24576,
        }
        return {"thinking_budget": budget_map.get(str(effort), 8192)}

    @staticmethod
    def _safe_json_loads(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    @staticmethod
    def _stream_block_key(block: dict[str, Any]) -> tuple[Any, ...]:
        block_type = block.get("type")
        if block_type == "thinking":
            return (
                "thinking",
                block.get("thought_signature"),
                block.get("text", ""),
            )
        if block_type == "function_call":
            return (
                "function_call",
                block.get("id"),
                block.get("name", ""),
                json.dumps(block.get("args", {}) or {}, ensure_ascii=False, sort_keys=True),
            )
        if block_type == "text":
            return ("text", block.get("text", ""))
        return (block_type, json.dumps(block, ensure_ascii=False, sort_keys=True))

    @classmethod
    def _append_stream_block(
        cls,
        blocks: list[dict[str, Any]],
        seen: set[tuple[Any, ...]],
        block: dict[str, Any],
    ) -> None:
        key = cls._stream_block_key(block)
        if key in seen:
            return
        seen.add(key)
        blocks.append(dict(block))

    @classmethod
    def _build_stream_assistant_message(
        cls,
        *,
        text: str,
        thinking: str,
        raw_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        blocks = [dict(block) for block in raw_blocks]
        has_text = any(block.get("type") == "text" for block in blocks)
        if text and not has_text:
            insert_at = 0
            while insert_at < len(blocks) and blocks[insert_at].get("type") == "thinking":
                insert_at += 1
            blocks.insert(insert_at, {"type": "text", "text": text})
        message: dict[str, Any] = {
            "role": "assistant",
            "content": blocks or ([{"type": "text", "text": text}] if text == "" else []),
        }
        if thinking:
            message["reasoning_content"] = thinking
        return message

    @classmethod
    def _serialize_gemini_part(cls, part: Any) -> dict[str, Any]:
        part_thought_signature = getattr(part, "thought_signature", None)
        if part_thought_signature is None and isinstance(part, dict):
            part_thought_signature = part.get("thought_signature") or part.get("thoughtSignature")

        if isinstance(part, dict):
            payload = dict(part)
            if payload.get("function_call") is not None:
                function_call = payload.get("function_call") or {}
                block = {
                    "type": "function_call",
                    "id": function_call.get("id"),
                    "name": function_call.get("name", ""),
                    "args": function_call.get("args", {}) or {},
                }
                if part_thought_signature:
                    block["thought_signature"] = part_thought_signature
                return block
            if payload.get("function_response") is not None:
                function_response = payload.get("function_response") or {}
                block = {
                    "type": "function_response",
                    "id": function_response.get("id"),
                    "name": function_response.get("name", ""),
                    "response": function_response.get("response", {}) or {},
                }
                if part_thought_signature:
                    block["thought_signature"] = part_thought_signature
                return block
            if payload.get("text") is not None:
                block_type = "thinking" if payload.get("thought") else "text"
                block = {
                    "type": block_type,
                    "text": payload.get("text", ""),
                }
                thought_signature = payload.get("thought_signature") or payload.get("thoughtSignature")
                if thought_signature:
                    block["thought_signature"] = thought_signature
                return block
            return payload

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            args = getattr(function_call, "args", None)
            if args is None and isinstance(function_call, dict):
                args = function_call.get("args")
            block = {
                "type": "function_call",
                "id": getattr(function_call, "id", None) if not isinstance(function_call, dict) else function_call.get("id"),
                "name": getattr(function_call, "name", None) if not isinstance(function_call, dict) else function_call.get("name", ""),
                "args": args or {},
            }
            if part_thought_signature:
                block["thought_signature"] = part_thought_signature
            return block

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            response = getattr(function_response, "response", None)
            if response is None and isinstance(function_response, dict):
                response = function_response.get("response")
            block = {
                "type": "function_response",
                "id": getattr(function_response, "id", None) if not isinstance(function_response, dict) else function_response.get("id"),
                "name": getattr(function_response, "name", None) if not isinstance(function_response, dict) else function_response.get("name", ""),
                "response": response or {},
            }
            if part_thought_signature:
                block["thought_signature"] = part_thought_signature
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
        block = {
            "type": "thinking" if thought_flag else "text",
            "text": text or "",
        }
        if thought_signature:
            block["thought_signature"] = thought_signature
        return block

    @classmethod
    def _candidate_content(cls, response: Any) -> Any:
        if response is None:
            return None
        candidates = getattr(response, "candidates", None)
        if candidates:
            candidate = candidates[0]
            return getattr(candidate, "content", None) or (candidate.get("content") if isinstance(candidate, dict) else None)
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

    def _message_to_google_parts(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        content = message.get("content")
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return [{"text": str(content or "")}]

        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append({"text": str(block)})
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append({"text": block.get("text", "")})
                continue
            if block_type == "thinking":
                part = {
                    "text": block.get("text", "") or block.get("thinking", ""),
                    "thought": True,
                }
                if block.get("thought_signature"):
                    part["thought_signature"] = block["thought_signature"]
                parts.append(part)
                continue
            if block_type == "function_call":
                part = {
                    "function_call": {
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "args": block.get("args", {}) or {},
                    }
                }
                if block.get("thought_signature"):
                    part["thought_signature"] = block["thought_signature"]
                parts.append(part)
                continue
            if block_type == "function_response":
                part = {
                    "function_response": {
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "response": block.get("response", {}) or {},
                    }
                }
                if block.get("thought_signature"):
                    part["thought_signature"] = block["thought_signature"]
                parts.append(part)
                continue
            if block_type == "tool_use":
                parts.append(
                    {
                        "function_call": {
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "args": block.get("input", {}) or {},
                        }
                    }
                )
                continue
            if block_type == "tool_result":
                parts.append(
                    {
                        "function_response": {
                            "id": block.get("tool_use_id"),
                            "name": block.get("name", ""),
                            "response": {"result": block.get("content", "")},
                        }
                    }
                )
                continue
        return parts

    def _build_generate_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_chunks: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            if role == "system":
                system_content = message.get("content")
                if isinstance(system_content, str):
                    system_chunks.append(system_content)
                elif isinstance(system_content, list):
                    for block in system_content:
                        if isinstance(block, dict):
                            system_chunks.append(
                                block.get("text")
                                or block.get("thinking")
                                or json.dumps(block, ensure_ascii=False)
                            )
                        else:
                            system_chunks.append(str(block))
                continue
            parts = self._message_to_google_parts(message)
            if not parts:
                continue
            google_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": google_role,
                    "parts": parts,
                }
            )

        config: dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        if system_chunks:
            config["system_instruction"] = "\n\n".join(chunk for chunk in system_chunks if chunk)
        if tools:
            config["tools"] = self._convert_tools(tools)
            config["automatic_function_calling"] = {"disable": True}
        thinking_config = self._build_thinking_config(reasoning)
        if thinking_config:
            config["thinking_config"] = thinking_config
        config.update(kwargs)
        return contents, config

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function = tool["function"]
                declarations.append(
                    {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}) or {},
                    }
                )
        if not declarations:
            return []
        return [{"function_declarations": declarations}]

    def invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        contents, config = self._build_generate_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config or None,
        )

    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str | None:
        response = self.invoke_raw(messages, temperature=temperature, reasoning=reasoning, **kwargs)
        return self.get_response_content(response) or ""

    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        contents, config = self._build_generate_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config or None,
        )

    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        for event in self.stream_events(messages, temperature=temperature, reasoning=reasoning, **kwargs):
            if event.get("type") == "text_delta":
                yield event.get("delta", "") or ""

    def stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        contents, config = self._build_generate_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config or None,
        )
        for chunk in response:
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
                elif block["type"] == "function_call":
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
            }
            return
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "".join(thinking_parts),
        }

    def stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        contents, config = self._build_generate_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        seen_block_keys: set[tuple[Any, ...]] = set()
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config or None,
        )
        for chunk in response:
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
                "assistant_items": self._build_stream_assistant_message(
                    text="".join(text_parts),
                    thinking="".join(thinking_parts),
                    raw_blocks=assistant_blocks,
                ),
            }
            return
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "assistant_items": self._build_stream_assistant_message(
                text="".join(text_parts),
                thinking="".join(thinking_parts),
                raw_blocks=assistant_blocks,
            ),
        }

    async def async_invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        contents, config = self._build_generate_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        return await async_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config or None,
        )

    async def async_invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str | None:
        response = await self.async_invoke_raw(messages, temperature=temperature, reasoning=reasoning, **kwargs)
        return self.get_response_content(response) or ""

    async def async_invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        contents, config = self._build_generate_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        return await async_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config or None,
        )

    async def async_stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        async for event in self.async_stream_events(messages, temperature=temperature, reasoning=reasoning, **kwargs):
            if event.get("type") == "text_delta":
                yield event.get("delta", "") or ""

    async def async_stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        contents, config = self._build_generate_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        seen_block_keys: set[tuple[Any, ...]] = set()
        stream = await async_client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config or None,
        )
        async for chunk in stream:
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
            }
            return
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "".join(thinking_parts),
        }

    async def async_stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        contents, config = self._build_generate_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        seen_block_keys: set[tuple[Any, ...]] = set()
        stream = await async_client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config or None,
        )
        async for chunk in stream:
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
                "assistant_items": self._build_stream_assistant_message(
                    text="".join(text_parts),
                    thinking="".join(thinking_parts),
                    raw_blocks=assistant_blocks,
                ),
            }
            return
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "assistant_items": self._build_stream_assistant_message(
                text="".join(text_parts),
                thinking="".join(thinking_parts),
                raw_blocks=assistant_blocks,
            ),
        }

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
                    "type": "function_response",
                    "id": tool_id,
                    "name": tool_name,
                    "response": {"result": content},
                }
            ],
        }

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict:
        blocks: list[dict[str, Any]] = []
        if content:
            blocks.append({"type": "text", "text": content})
        if tool_calls:
            for tool_call in tool_calls:
                args = self._safe_json_loads(tool_call.get("arguments", {}))
                if not isinstance(args, dict):
                    args = {}
                blocks.append(
                    {
                        "type": "function_call",
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name", ""),
                        "args": args,
                    }
                )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": blocks or ([{"type": "text", "text": ""}] if content == "" else []),
        }
        if thinking:
            message["reasoning_content"] = thinking
        return message

    def format_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict:
        content = self._candidate_content(response)
        role = "assistant"
        if isinstance(content, dict):
            role = "assistant" if content.get("role") == "model" else content.get("role", "assistant")
        else:
            role = "assistant" if getattr(content, "role", "model") == "model" else getattr(content, "role", "assistant")
        blocks = []
        for part in self._content_parts(content):
            block = self._serialize_gemini_part(part)
            if block["type"] == "thinking" and not include_reasoning:
                continue
            blocks.append(block)
        message: dict[str, Any] = {
            "role": "assistant" if role == "model" else role,
            "content": blocks,
        }
        if include_reasoning:
            thinking_text = self.get_thinking_content(response)
            if thinking_text:
                message["reasoning_content"] = thinking_text
        return message

    def prepare_message_for_request(self, message: Any) -> Any:
        if not isinstance(message, dict):
            return message
        payload = dict(message)
        if isinstance(payload.get("tool_calls"), list):
            return self.format_assistant_message(
                content=payload.get("content"),
                tool_calls=[
                    {
                        "id": item.get("id"),
                        "name": item.get("name")
                        or (item.get("function") or {}).get("name", ""),
                        "arguments": item.get("arguments")
                        or (item.get("function") or {}).get("arguments", "{}"),
                    }
                    for item in payload.get("tool_calls", [])
                    if isinstance(item, dict)
                ],
                thinking=payload.get("thinking") or payload.get("reasoning_content"),
            )
        if payload.get("role") in {"tool", "function"}:
            return self.format_tool_result(
                str(payload.get("content", "")),
                str(payload.get("tool_call_id") or payload.get("id") or ""),
                str(payload.get("name") or payload.get("tool_name") or ""),
            )
        return payload

    def prepare_messages_for_request(self, messages: list[Any]) -> list[Any]:
        prepared: list[Any] = []
        for message in messages:
            payload = self.prepare_message_for_request(message)
            if payload is None:
                continue
            if (
                self._is_function_response_message(payload)
                and prepared
                and self._is_function_response_message(prepared[-1])
            ):
                prepared[-1]["content"].extend(payload["content"])
                continue
            prepared.append(payload)
        return prepared

    @staticmethod
    def _is_function_response_message(message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if message.get("role") != "user":
            return False
        content = message.get("content")
        if not isinstance(content, list) or not content:
            return False
        return all(
            isinstance(part, dict) and part.get("type") == "function_response"
            for part in content
        )

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
        content = self._candidate_content(response)
        tool_calls: list[dict[str, Any]] = []
        for index, part in enumerate(self._content_parts(content)):
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
