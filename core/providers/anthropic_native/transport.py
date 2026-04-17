"""
Anthropic Claude native transport provider.

只负责把 request-ready Claude history message 组装成 messages.create / messages.stream 请求。
"""

from __future__ import annotations

from typing import Any, Optional

from ..base import BaseProvider


class AnthropicNativeProvider(BaseProvider):
    def _create_client(self) -> Any:
        injected = self.kwargs.get("client")
        if injected is not None:
            return injected
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicNativeProvider requires the `anthropic` package. "
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
                    "AnthropicNativeProvider requires the `anthropic` package. "
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
            return {"type": "enabled", "budget_tokens": reasoning["budget_tokens"]}
        effort = reasoning.get("effort")
        if not effort:
            return None
        budget_map = {"low": 1024, "medium": 4096, "high": 16384}
        return {"type": "enabled", "budget_tokens": budget_map.get(str(effort), 4096)}

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
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

    def build_request(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": list(replay_history),
            "max_tokens": self.max_tokens or 4096,
        }
        if system_prompt:
            params["system"] = system_prompt
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = self._convert_tools(tools)
        thinking = self._thinking_config(reasoning)
        if thinking:
            params["thinking"] = thinking
        if stream:
            params["stream"] = True
        params.update(kwargs)
        return params

    def invoke_raw(self, request: Any) -> Any:
        params = dict(request)
        params.pop("stream", None)
        return self.client.messages.create(**params)

    def stream_raw(self, request: Any) -> Any:
        params = dict(request)
        params.pop("stream", None)
        stream_method = getattr(self.client.messages, "stream", None)
        if stream_method is not None:
            return stream_method(**params)
        params["stream"] = True
        return self.client.messages.create(**params)

    async def async_invoke_raw(self, request: Any) -> Any:
        params = dict(request)
        params.pop("stream", None)
        async_client = self._get_async_client()
        return await async_client.messages.create(**params)

    async def async_stream_raw(self, request: Any) -> Any:
        params = dict(request)
        params.pop("stream", None)
        async_client = self._get_async_client()
        stream_method = getattr(async_client.messages, "stream", None)
        if stream_method is not None:
            return stream_method(**params)
        params["stream"] = True
        return await async_client.messages.create(**params)
