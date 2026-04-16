"""
Anthropic Provider

使用 Anthropic 原生 Python SDK 调用 Claude API。
"""

from __future__ import annotations

from typing import Optional, Any, Generator, AsyncGenerator
import contextlib
import json
import logging

from .base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicNativeProvider(BaseProvider):
    """
    Anthropic Claude Provider

    使用 `anthropic.Anthropic` / `AsyncAnthropic` 原生 SDK，而非 OpenAI 兼容层。
    """

    def _create_client(self) -> Any:
        injected = self.kwargs.get("client")
        if injected is not None:
            return injected
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicProvider requires the `anthropic` package. "
                "Install it with `pip install anthropic`."
            ) from exc
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if "max_retries" in self.kwargs:
            client_kwargs["max_retries"] = self.kwargs["max_retries"]
        return Anthropic(**client_kwargs)

    def _get_async_client(self) -> Any:
        injected = self.kwargs.get("async_client")
        if injected is not None:
            return injected
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicProvider requires the `anthropic` package. "
                    "Install it with `pip install anthropic`."
                ) from exc
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if "max_retries" in self.kwargs:
                client_kwargs["max_retries"] = self.kwargs["max_retries"]
            self._async_client = AsyncAnthropic(**client_kwargs)
        return self._async_client

    @staticmethod
    def _thinking_config(reasoning: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not reasoning:
            return None
        if "thinking" in reasoning and isinstance(reasoning["thinking"], dict):
            return dict(reasoning["thinking"])
        if "budget_tokens" in reasoning:
            return {
                "type": "enabled",
                "budget_tokens": reasoning["budget_tokens"],
            }
        effort = reasoning.get("effort")
        if not effort:
            return None
        budget_map = {
            "low": 1024,
            "medium": 4096,
            "high": 16384,
        }
        return {
            "type": "enabled",
            "budget_tokens": budget_map.get(str(effort), 4096),
        }

    @staticmethod
    # def _as_dict(value: Any) -> Any:
    #     if isinstance(value, dict):
    #         return dict(value)
    #     if hasattr(value, "to_dict"):
    #         payload = value.to_dict()
    #         if isinstance(payload, dict):
    #             return payload
    #     return value
    @staticmethod
    def _as_dict(value: Any) -> Any:
        if isinstance(value, dict):
            return dict(value)

        # Anthropic SDK 的 Pydantic 模型（如 ParsedTextBlock / Message / stream
        # 在调用 to_dict() 时可能触发 PydanticSerializationUnexpectedValue warning。
        # 本 provider 后续逻辑已经支持 getattr 读取，因此这里保留原对象。
        module = getattr(type(value), "__module__", "")
        if module.startswith("anthropic."):
            return value

        if hasattr(value, "to_dict"):
            payload = value.to_dict()
            if isinstance(payload, dict):
                return payload
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
                return {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "content": block.get("content", ""),
                    **({"name": block.get("name")} if block.get("name") else {}),
                }
            if block_type == "thinking":
                payload = {
                    "type": "thinking",
                    "thinking": block.get("thinking") or block.get("text", ""),
                }
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

    @classmethod
    def _message_content_blocks(cls, message: dict[str, Any]) -> list[dict[str, Any]] | str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")

        blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                blocks.append({"type": "text", "text": str(block)})
                continue
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
                continue
            if block_type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}) or {},
                    }
                )
                continue
            if block_type == "tool_result":
                payload = {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "content": block.get("content", ""),
                }
                if block.get("name"):
                    payload["name"] = block["name"]
                blocks.append(payload)
                continue
            if block_type in {"thinking", "redacted_thinking"}:
                blocks.append(dict(block))
                continue
            if block_type == "function_call":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "input": block.get("args", {}) or {},
                    }
                )
                continue
            if block_type == "function_response":
                payload = {
                    "type": "tool_result",
                    "tool_use_id": block.get("id"),
                    "content": json.dumps(block.get("response", {}) or {}, ensure_ascii=False),
                }
                if block.get("name"):
                    payload["name"] = block["name"]
                blocks.append(payload)
                continue
        return blocks

    def _build_messages_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        request_messages: list[dict[str, Any]] = []
        system_chunks: list[str] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            # if message.get("reasoning_content"):
            #     message.pop("reasoning_content")
            role = str(message.get("role", "user"))
            if role == "system":
                content = message.get("content")
                if isinstance(content, str):
                    system_chunks.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            system_chunks.append(
                                block.get("text")
                                or block.get("thinking")
                                or json.dumps(block, ensure_ascii=False)
                            )
                        else:
                            system_chunks.append(str(block))
                continue
            anthropic_role = "assistant" if role == "assistant" else "user"
            request_messages.append(
                {
                    "role": anthropic_role,
                    "content": self._message_content_blocks(message),
                }
            )

        params: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": self.max_tokens or 4096,
        }
        if system_chunks:
            params["system"] = "\n\n".join(chunk for chunk in system_chunks if chunk)
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = self._convert_tools(tools)
        thinking = self._thinking_config(reasoning)
        if thinking:
            params["thinking"] = thinking
        if "betas" in kwargs:
            params["betas"] = kwargs.pop("betas")
        params.update(kwargs)
        return params

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function = tool["function"]
                converted.append(
                    {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "input_schema": function.get("parameters", {}) or {},
                    }
                )
        return converted

    def invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        params = self._build_messages_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        return self.client.messages.create(**params)

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
        params = self._build_messages_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        return self.client.messages.create(**params)

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

    def _iter_stream_events(self, stream_source: Any) -> Generator[Any, None, None]:
        if hasattr(stream_source, "__enter__") and hasattr(stream_source, "__exit__"):
            with stream_source as active_stream:
                if hasattr(active_stream, "__iter__"):
                    yield from active_stream
                elif hasattr(active_stream, "events"):
                    yield from active_stream.events()
            return
        if hasattr(stream_source, "__iter__"):
            yield from stream_source

    async def _aiter_stream_events(self, stream_source: Any) -> AsyncGenerator[Any, None]:
        if hasattr(stream_source, "__aenter__") and hasattr(stream_source, "__aexit__"):
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
        }

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
    def _set_stream_assistant_block(
        state: dict[str, Any],
        index: int,
        block: dict[str, Any],
    ) -> dict[str, Any]:
        stored = dict(block)
        state["assistant_blocks"][index] = stored
        return stored

    def _build_stream_assistant_message(self, state: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        finalized_tool_calls = {
            item["id"]: item
            for item in self._finalize_stream_tool_calls(state)
            if item.get("id")
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
            content.insert(
                0,
                {
                    "type": "thinking",
                    "thinking": "".join(state["thinking_parts"]),
                },
            )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if state["thinking_parts"]:
            message["reasoning_content"] = "".join(state["thinking_parts"])
        return message

    def _extract_anthropic_stream_events(
        self,
        event: Any,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        payload = self._as_dict(event)
        event_type = payload.get("type") if isinstance(payload, dict) else getattr(event, "type", None)
        events: list[dict[str, Any]] = []
        if not event_type:
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
                current = state["tool_calls"].setdefault(index, {"id": None, "name": "", "input_json": "", "input": {}})
                current["input_json"] += json_fragment
            return events

        if event_type == "message_delta":
            delta = payload.get("delta") if isinstance(payload, dict) else getattr(event, "delta", None)
            stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else getattr(delta, "stop_reason", None)
            if stop_reason == "tool_use":
                events.append(
                    {
                        "type": "tool_calls",
                        "tool_calls": self._finalize_stream_tool_calls(state),
                        "content": "".join(state["text_parts"]),
                        "thinking": "".join(state["thinking_parts"]),
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
                }
            )
            state["terminal_emitted"] = True
            return events

        return events

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
            }
        return {
            "type": "final_response",
            "content": "".join(state["text_parts"]),
            "thinking": "".join(state["thinking_parts"]),
        }

    def stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        params = self._build_messages_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        stream_method = getattr(self.client.messages, "stream", None)
        if stream_method is not None:
            stream_source = stream_method(**params)
        else:
            params["stream"] = True
            stream_source = self.client.messages.create(**params)
        state = self._init_anthropic_stream_state()
        for raw_event in self._iter_stream_events(stream_source):
            for event in self._extract_anthropic_stream_events(raw_event, state):
                yield event
        final_event = self._finalize_anthropic_stream_state(state)
        if final_event.get("type") != "stream_end":
            yield final_event

    def stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        params = self._build_messages_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        stream_method = getattr(self.client.messages, "stream", None)
        if stream_method is not None:
            stream_source = stream_method(**params)
        else:
            params["stream"] = True
            stream_source = self.client.messages.create(**params)
        state = self._init_anthropic_stream_state()
        for raw_event in self._iter_stream_events(stream_source):
            for event in self._extract_anthropic_stream_events(raw_event, state):
                if event.get("type") == "tool_calls":
                    event["assistant_items"] = self._build_stream_assistant_message(state)
                elif event.get("type") == "final_response":
                    event["assistant_items"] = self._build_stream_assistant_message(state)
                yield event
        final_event = self._finalize_anthropic_stream_state(state)
        if final_event.get("type") == "tool_calls":
            final_event["assistant_items"] = self._build_stream_assistant_message(state)
        elif final_event.get("type") == "final_response":
            final_event["assistant_items"] = self._build_stream_assistant_message(state)
        if final_event.get("type") != "stream_end":
            yield final_event

    async def async_invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        params = self._build_messages_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        return await async_client.messages.create(**params)

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
        params = self._build_messages_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        return await async_client.messages.create(**params)

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
        params = self._build_messages_request(
            messages,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        stream_method = getattr(async_client.messages, "stream", None)
        if stream_method is not None:
            stream_source = stream_method(**params)
        else:
            params["stream"] = True
            stream_source = await async_client.messages.create(**params)
        state = self._init_anthropic_stream_state()
        async for raw_event in self._aiter_stream_events(stream_source):
            for event in self._extract_anthropic_stream_events(raw_event, state):
                yield event
        final_event = self._finalize_anthropic_stream_state(state)
        if final_event.get("type") != "stream_end":
            yield final_event

    async def async_stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        params = self._build_messages_request(
            messages,
            tools=tools,
            temperature=temperature if temperature is not None else self.temperature,
            reasoning=reasoning,
            **kwargs,
        )
        async_client = self._get_async_client()
        stream_method = getattr(async_client.messages, "stream", None)
        if stream_method is not None:
            stream_source = stream_method(**params)
        else:
            params["stream"] = True
            stream_source = await async_client.messages.create(**params)
        state = self._init_anthropic_stream_state()
        async for raw_event in self._aiter_stream_events(stream_source):
            for event in self._extract_anthropic_stream_events(raw_event, state):
                if event.get("type") == "tool_calls":
                    event["assistant_items"] = self._build_stream_assistant_message(state)
                elif event.get("type") == "final_response":
                    event["assistant_items"] = self._build_stream_assistant_message(state)
                yield event
        final_event = self._finalize_anthropic_stream_state(state)
        if final_event.get("type") == "tool_calls":
            final_event["assistant_items"] = self._build_stream_assistant_message(state)
        elif final_event.get("type") == "final_response":
            final_event["assistant_items"] = self._build_stream_assistant_message(state)
        if final_event.get("type") != "stream_end":
            yield final_event

    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str,
    ) -> dict:
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            ],
        }
        if tool_name:
            message["content"][0]["name"] = tool_name
        return message

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict:
        blocks: list[dict[str, Any]] = []
        if content:
            blocks.append(
                {
                    "type": "text",
                    "text": content,
                }
            )
        if tool_calls:
            for tool_call in tool_calls:
                arguments = tool_call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name", ""),
                        "input": arguments or {},
                    }
                )
        message = {
            "role": "assistant",
            "content": blocks if blocks else (content or ""),
        }
        if thinking:
            message["reasoning_content"] = thinking
        return message

    def get_thinking_content(self, response: Any) -> Optional[str]:
        content_blocks = getattr(response, "content", None) or []
        thoughts: list[str] = []
        for block in content_blocks:
            serialized = self._serialize_block(block)
            if serialized.get("type") == "thinking" and serialized.get("thinking"):
                thoughts.append(serialized["thinking"])
        return "".join(thoughts) or None

    def is_thinking_model(self) -> bool:
        return "thinking" in self.model.lower()

    def format_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict:
        blocks = []
        for block in getattr(response, "content", None) or []:
            serialized = self._serialize_block(block)
            if serialized.get("type") in {"thinking", "redacted_thinking"} and not include_reasoning:
                continue
            blocks.append(serialized)
        message = {
            "role": "assistant",
            "content": blocks,
        }
        if include_reasoning:
            thinking = self.get_thinking_content(response)
            if thinking:
                message["reasoning_content"] = thinking
        return message

    def format_assistant_message_openai_compat(
        self,
        message: dict[str, Any],
    ) -> dict:
        tool_calls = []
        for item in message.get("tool_calls", []) or []:
            if not isinstance(item, dict):
                continue
            function = item.get("function", {}) if isinstance(item.get("function"), dict) else {}
            tool_calls.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name") or function.get("name", ""),
                    "arguments": item.get("arguments") or function.get("arguments", "{}"),
                }
            )
        return self.format_assistant_message(
            content=message.get("content"),
            tool_calls=tool_calls or None,
            thinking=message.get("thinking") or message.get("reasoning_content"),
        )

    def prepare_message_for_request(self, message: Any) -> Any:
        if not isinstance(message, dict):
            return message
        payload = dict(message)
        if payload.get("tool_calls"):
            return self.format_assistant_message_openai_compat(payload)
        if payload.get("role") in {"tool", "function"}:
            return self.format_tool_result(
                str(payload.get("content", "")),
                str(payload.get("tool_call_id") or payload.get("id") or ""),
                str(payload.get("name") or payload.get("tool_name") or ""),
            )
        return payload

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
