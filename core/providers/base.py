"""
Provider 基类

定义与底层 SDK 无关的统一 Provider 抽象与共享辅助方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any, Generator, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    LLM Provider 抽象基类。

    这个基类不再假设底层一定是 OpenAI 兼容 SDK。
    原生 Google / Anthropic Provider 与 OpenAI 兼容 Provider
    都应在这里收敛到同一套接口。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self.client = self._create_client()
        self._async_client: Any = None

    @abstractmethod
    def _create_client(self) -> Any:
        pass

    def _get_async_client(self) -> Any:
        raise NotImplementedError(f"{self.provider_name} provider does not implement async client access")

    def close(self) -> None:
        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        async_client = getattr(self, "_async_client", None)
        aclose = getattr(async_client, "aclose", None)
        if callable(aclose):
            await aclose() 
            return
        close = getattr(async_client, "close", None)
        if callable(close):
            close()

    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str | None:
        response = self.invoke_raw(
            messages,
            temperature=temperature,
            reasoning=reasoning,
            **kwargs,
        )
        return self.get_response_content(response) or ""

    @abstractmethod
    def invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        pass

    def stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        text_parts: list[str] = []
        for chunk in self.stream(
            messages,
            temperature=temperature,
            reasoning=reasoning,
            **kwargs,
        ):
            text_parts.append(chunk)
            yield {
                "type": "text_delta",
                "delta": chunk,
            }
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "",
        }

    @abstractmethod
    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        pass

    async def async_invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str | None:
        response = await self.async_invoke_raw(
            messages,
            temperature=temperature,
            reasoning=reasoning,
            **kwargs,
        )
        return self.get_response_content(response) or ""

    @abstractmethod
    async def async_invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    async def async_stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        pass

    async def async_stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        text_parts: list[str] = []
        async for chunk in self.async_stream(
            messages,
            temperature=temperature,
            reasoning=reasoning,
            **kwargs,
        ):
            text_parts.append(chunk)
            yield {
                "type": "text_delta",
                "delta": chunk,
            }
        yield {
            "type": "final_response",
            "content": "".join(text_parts),
            "thinking": "",
        }

    @abstractmethod
    async def async_invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    async def async_stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        pass

    @abstractmethod
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str,
    ) -> dict:
        pass

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }
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

    def format_assistant_response(self, response: Any, include_reasoning: bool = False) -> dict[str, Any]:
        content = getattr(response, "content", None) or None
        thinking = self.get_thinking_content(response) if include_reasoning else None
        tool_calls_data = getattr(response, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []

        for index, tool_call in enumerate(tool_calls_data):
            function = getattr(tool_call, "function", None)
            tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None) or f"tool_call_{index}",
                    "type": getattr(tool_call, "type", None) or "function",
                    "name": getattr(function, "name", None) or "",
                    "arguments": getattr(function, "arguments", None) or "",
                }
            )

        return self.format_assistant_message(
            content=content,
            tool_calls=tool_calls or None,
            thinking=thinking,
        )

    def prepare_message_for_request(self, message: Any) -> Any:
        if not isinstance(message, dict):
            return message
        item_type = message.get("type")
        if item_type == "reasoning":
            return None
        payload = dict(message)
        if "thinking" in payload:
            payload["reasoning_content"] = payload.pop("thinking")
        return payload

    def prepare_messages_for_request(self, messages: list[Any]) -> list[Any]:
        prepared: list[Any] = []
        for message in messages:
            payload = self.prepare_message_for_request(message)
            if payload is None:
                continue
            prepared.append(payload)
        return prepared

    def _init_chat_tool_stream_state(self) -> dict[str, Any]:
        return {
            "text_parts": [],
            "thinking_parts": [],
            "tool_calls": {},
            "terminal_emitted": False,
        }

    def _extract_chat_stream_events(
        self,
        chunk: Any,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return events

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return events

        reasoning_delta = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
        )
        if reasoning_delta:
            state["thinking_parts"].append(reasoning_delta)
            events.append({"type": "thinking_delta", "delta": reasoning_delta})

        content = getattr(delta, "content", None) or ""
        if content:
            state["text_parts"].append(content)
            events.append({"type": "text_delta", "delta": content})

        for tool_call in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_call, "index", 0) or 0
            current = state["tool_calls"].setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "name": "",
                    "arguments": "",
                },
            )
            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id:
                current["id"] = tool_call_id

            function = getattr(tool_call, "function", None)
            if function is not None:
                function_name = getattr(function, "name", None)
                if function_name:
                    current["name"] = function_name
                function_arguments = getattr(function, "arguments", None)
                if function_arguments:
                    current["arguments"] += function_arguments

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "tool_calls":
            events.append(
                {
                    "type": "tool_calls",
                    "tool_calls": self._normalize_stream_tool_calls(state["tool_calls"]),
                    "content": "".join(state["text_parts"]),
                    "thinking": "".join(state["thinking_parts"]),
                }
            )
            state["terminal_emitted"] = True
        elif finish_reason in {"stop", "length", "content_filter"}:
            events.append(
                {
                    "type": "final_response",
                    "content": "".join(state["text_parts"]),
                    "thinking": "".join(state["thinking_parts"]),
                    "finish_reason": finish_reason,
                }
            )
            state["terminal_emitted"] = True

        return events

    def _finalize_chat_tool_stream_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("terminal_emitted"):
            return {"type": "stream_end"}
        tool_calls = self._normalize_stream_tool_calls(state["tool_calls"])
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

    def _normalize_stream_tool_calls(self, tool_calls_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            tool_call = dict(tool_calls_by_index[index])
            if not tool_call.get("id"):
                tool_call["id"] = f"tool_call_{index}"
            normalized.append(tool_call)
        return normalized

    def get_thinking_content(self, response: Any) -> Optional[str]:
        thinking = getattr(response, "reasoning_content", None)
        return thinking or None

    def get_response_content(self, response: Any) -> Optional[str]:
        return getattr(response, "content", None)

    def has_tool_calls(self, response: Any) -> bool:
        return bool(hasattr(response, "tool_calls") and response.tool_calls)

    def get_tool_calls(self, response: Any) -> list:
        if self.has_tool_calls(response):
            return response.tool_calls
        return []

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.replace("Provider", "").lower()
