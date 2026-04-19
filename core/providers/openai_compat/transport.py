"""
OpenAI Chat transport base.

只负责 request 组装和 chat.completions 原始调用。
"""

from __future__ import annotations

from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI

from ..base import BaseProvider

class OpenAICompatibleProviderBase(BaseProvider):
    def _create_client(self) -> OpenAI:
        injected = self.kwargs.get("client")
        if injected is not None:
            return injected
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _get_async_client(self) -> AsyncOpenAI:
        injected = self.kwargs.get("async_client")
        if injected is not None:
            return injected
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._async_client

    def build_request(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(replay_history)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            params["tools"] = tools
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        return params

    def invoke_raw(self, request: Any) -> Any:
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message

    def stream_raw(self, request: Any) -> Any:
        return self.client.chat.completions.create(**request)

    async def async_invoke_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        response = await async_client.chat.completions.create(**request)
        return response.choices[0].message

    async def async_stream_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        return await async_client.chat.completions.create(**request)


class OpenAIProvider(OpenAICompatibleProviderBase):
    pass
