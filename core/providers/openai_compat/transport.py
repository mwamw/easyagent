"""
OpenAI Chat transport base.

只负责 request 组装和 chat.completions 原始调用。
"""

from __future__ import annotations

from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI

from ..base import BaseProvider, _usage_field, _usage_float, _usage_int

class OpenAICompatibleProviderBase(BaseProvider):
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
        aliases = (self.provider_name, "openai_compat", "openai", "chat_completions")
        for alias in aliases:
            override = overrides.get(alias)
            if isinstance(override, dict):
                return dict(override)
        return None

    def _compile_reasoning_params(self, reasoning: Optional[dict[str, Any] | str]) -> dict[str, Any]:
        normalized = self._normalize_reasoning(reasoning)
        if not normalized or normalized.get("enabled") is False:
            return {}
        override = self._reasoning_override(normalized)
        if override is not None:
            return override
        if "reasoning_effort" in normalized:
            return {"reasoning_effort": normalized["reasoning_effort"]}
        effort = normalized.get("effort")
        if not effort:
            return {}
        effort_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        }
        return {"reasoning_effort": effort_map.get(str(effort), "medium")}

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
        reasoning: Optional[dict[str, Any] | str] = None,
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
        params.update(self._compile_reasoning_params(reasoning))
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
        return self.normalize_usage_metrics(payload)


class OpenAIProvider(OpenAICompatibleProviderBase):
    pass
