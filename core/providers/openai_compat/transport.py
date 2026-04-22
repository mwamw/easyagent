"""
OpenAI Chat transport base.

只负责 request 组装和 chat.completions 原始调用。
"""

from __future__ import annotations

from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI

from ..base import BaseProvider, _usage_field, _usage_float, _usage_int

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
        extra_kwargs = dict(kwargs)
        if stream and self.provider_name == "openai":
            stream_options = extra_kwargs.pop("stream_options", None)
            if isinstance(stream_options, dict):
                stream_options = dict(stream_options)
                stream_options.setdefault("include_usage", True)
            else:
                stream_options = {"include_usage": True}
            params["stream_options"] = stream_options
        params.update(extra_kwargs)
        return params

    def invoke_raw(self, request: Any) -> Any:
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        return self.wrap_response_with_usage(
            message,
            usage=getattr(response, "usage", None),
        )

    def stream_raw(self, request: Any) -> Any:
        return self.client.chat.completions.create(**request)

    async def async_invoke_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        response = await async_client.chat.completions.create(**request)
        message = response.choices[0].message
        return self.wrap_response_with_usage(
            message,
            usage=getattr(response, "usage", None),
        )

    async def async_stream_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        return await async_client.chat.completions.create(**request)

    def get_usage_from_response(self, response: Any) -> dict[str, Any]:
        usage = _usage_field(response, "usage")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}

        prompt_details = _usage_field(usage, "prompt_tokens_details") or {}
        completion_details = _usage_field(usage, "completion_tokens_details") or {}
        payload = {
            "inputTokens": _usage_int(usage, "prompt_tokens"),
            "outputTokens": _usage_int(usage, "completion_tokens"),
            "totalTokens": _usage_int(usage, "total_tokens"),
            "cachedInputTokens": _usage_int(prompt_details, "cached_tokens"),
            "reasoningTokens": _usage_int(completion_details, "reasoning_tokens"),
            "costUsd": _usage_float(usage, "cost_usd", "total_cost", "total_cost_usd"),
            "usageSource": "provider",
        }
        if payload["totalTokens"] is None and payload["inputTokens"] is not None and payload["outputTokens"] is not None:
            payload["totalTokens"] = payload["inputTokens"] + payload["outputTokens"]
        return {key: value for key, value in payload.items() if value is not None}


class OpenAIProvider(OpenAICompatibleProviderBase):
    pass
