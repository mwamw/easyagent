"""
OpenAI Responses transport provider.

只负责将 request-ready input items 组装到 responses.create，并返回原始 response / stream。
"""

from __future__ import annotations

from typing import Any, Optional

from ..openai_compat.transport import OpenAICompatibleProviderBase
from ..base import _usage_field, _usage_float, _usage_int


class OpenAIResponsesProvider(OpenAICompatibleProviderBase):
    def _base_params(self, temperature: Optional[float] = None, **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        for key in ("reasoning", "text", "truncation"):
            if key in kwargs:
                params[key] = kwargs.pop(key)
        chat_only = {"stream", "messages", "temperature", "input", "tools", "model"}
        for key, value in kwargs.items():
            if key not in chat_only:
                params[key] = value
        return params

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
        if reasoning:
            kwargs["reasoning"] = reasoning
        params = {
            "model": self.model,
            "input": list(replay_history),
            "stream": stream,
            **self._base_params(temperature, **kwargs),
        }
        if system_prompt:
            instructions = params.get("instructions")
            params["instructions"] = (
                f"{instructions}\n\n{system_prompt}" if isinstance(instructions, str) and instructions else system_prompt
            )
        if tools:
            params["tools"] = tools
        return params

    def invoke_raw(self, request: Any) -> Any:
        return self.client.responses.create(**request)

    def stream_raw(self, request: Any) -> Any:
        return self.client.responses.create(**request)

    async def async_invoke_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        return await async_client.responses.create(**request)

    async def async_stream_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        return await async_client.responses.create(**request)

    def get_usage_from_response(self, response: Any) -> dict[str, Any]:
        usage = _usage_field(response, "usage")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}

        input_details = _usage_field(usage, "input_tokens_details", "input_token_details") or {}
        output_details = _usage_field(usage, "output_tokens_details", "output_token_details") or {}
        payload = {
            "inputTokens": _usage_int(usage, "input_tokens"),
            "outputTokens": _usage_int(usage, "output_tokens"),
            "totalTokens": _usage_int(usage, "total_tokens"),
            "cachedInputTokens": _usage_int(input_details, "cached_tokens"),
            "reasoningTokens": _usage_int(output_details, "reasoning_tokens"),
            "costUsd": _usage_float(usage, "cost_usd", "total_cost", "total_cost_usd"),
            "usageSource": "provider",
        }
        if payload["totalTokens"] is None and payload["inputTokens"] is not None and payload["outputTokens"] is not None:
            payload["totalTokens"] = payload["inputTokens"] + payload["outputTokens"]
        return {key: value for key, value in payload.items() if value is not None}
