"""
Anthropic Claude native transport provider.

只负责把 request-ready Claude history message 组装成 messages.create / messages.stream 请求。
"""

from __future__ import annotations

from typing import Any, Optional

from ..base import BaseProvider, _usage_field, _usage_float, _usage_int


class AnthropicNativeProvider(BaseProvider):
    @staticmethod
    def _normalize_reasoning(reasoning: Optional[dict[str, Any] | str]) -> Optional[dict[str, Any]]:
        if reasoning is None or reasoning is False:
            return None
        if isinstance(reasoning, str):
            reasoning = {"effort": reasoning}
        if not isinstance(reasoning, dict):
            return None
        normalized = dict(reasoning)
        effort = normalized.get("effort")
        if effort is not None:
            effort_key = str(effort).strip().lower().replace("-", "_")
            normalized["effort"] = {
                "extra_high": "xhigh",
                "extra": "xhigh",
                "max": "xhigh",
                "maximum": "xhigh",
            }.get(effort_key, effort_key)
        return normalized

    def _reasoning_override(self, reasoning: dict[str, Any]) -> Optional[dict[str, Any]]:
        overrides = reasoning.get("provider_overrides") or reasoning.get("overrides")
        if not isinstance(overrides, dict):
            return None
        aliases = (self.provider_name, "anthropic_native", "claude_native", "anthropic", "claude")
        for alias in aliases:
            override = overrides.get(alias)
            if isinstance(override, dict):
                return dict(override)
        return None

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

    def _thinking_config(self, reasoning: Optional[dict[str, Any] | str]) -> Optional[dict[str, Any]]:
        reasoning = self._normalize_reasoning(reasoning)
        if not reasoning or reasoning.get("enabled") is False:
            return None
        override = self._reasoning_override(reasoning)
        if override is not None:
            if isinstance(override.get("thinking"), dict):
                return dict(override["thinking"])
            return override
        if "thinking" in reasoning and isinstance(reasoning["thinking"], dict):
            return dict(reasoning["thinking"])
        if "budget_tokens" in reasoning:
            return {"type": "enabled", "budget_tokens": reasoning["budget_tokens"]}
        effort = reasoning.get("effort")
        if not effort:
            return None
        budget_map = {"low": 1024, "medium": 4096, "high": 16384, "xhigh": 32768}
        return {"type": "enabled", "budget_tokens": budget_map.get(str(effort), 4096)}

    def build_request(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any] | str] = None,
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
            params["tools"] = tools
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

    def get_usage_from_response(self, response: Any) -> dict[str, Any]:
        usage = _usage_field(response, "usage")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}

        payload = {
            "inputTokens": _usage_int(usage, "input_tokens"),
            "outputTokens": _usage_int(usage, "output_tokens"),
            "totalTokens": _usage_int(usage, "total_tokens"),
            "cacheReadTokens": _usage_int(usage, "cache_read_input_tokens"),
            "cacheCreationTokens": _usage_int(usage, "cache_creation_input_tokens"),
            "cachedInputTokens": _usage_int(usage, "cache_read_input_tokens"),
            "costUsd": _usage_float(usage, "cost_usd", "total_cost", "total_cost_usd"),
            "usageSource": "provider",
        }
        if payload["totalTokens"] is None and payload["inputTokens"] is not None and payload["outputTokens"] is not None:
            payload["totalTokens"] = payload["inputTokens"] + payload["outputTokens"]
        return self.normalize_usage_metrics(payload)
