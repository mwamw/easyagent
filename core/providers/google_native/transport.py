"""
Google Gemini native transport provider.

只负责把 request-ready Gemini history turn 组装成 generate_content 请求。
"""

from __future__ import annotations

from typing import Any, Optional

from ..base import BaseProvider, _usage_field, _usage_float, _usage_int


class GoogleNativeProvider(BaseProvider):
    def _create_client(self) -> Any:
        injected = self.kwargs.get("client")
        if injected is not None:
            return injected
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GoogleNativeProvider requires the `google-genai` package. "
                "Install it with `pip install google-genai`."
            ) from exc
        client_args: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_args["http_options"] = {"base_url": self.base_url}
        return genai.Client(**client_args)

    def _get_async_client(self) -> Any:
        injected = self.kwargs.get("async_client")
        if injected is not None:
            return injected
        if self._async_client is None:
            aio_client = getattr(self.client, "aio", None)
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
            level_map = {"low": "low", "medium": "medium", "high": "high"}
            return {"thinking_level": level_map.get(str(effort), "medium")}
        budget_map = {"low": 1024, "medium": 8192, "high": 24576}
        return {"thinking_budget": budget_map.get(str(effort), 8192)}

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
        contents = list(replay_history)
        config: dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        if system_prompt:
            config["system_instruction"] = system_prompt
        if tools:
            config["tools"] = tools
            config["automatic_function_calling"] = {"disable": True}
        thinking_config = self._build_thinking_config(reasoning)
        if thinking_config:
            config["thinking_config"] = thinking_config
        config.update(kwargs)
        return {
            "model": self.model,
            "contents": contents,
            "config": config or None,
            "_stream": stream,
        }

    @staticmethod
    def _request_kwargs(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": request["model"],
            "contents": request["contents"],
            "config": request.get("config"),
        }

    def invoke_raw(self, request: Any) -> Any:
        return self.client.models.generate_content(**self._request_kwargs(request))

    def stream_raw(self, request: Any) -> Any:
        return self.client.models.generate_content_stream(**self._request_kwargs(request))

    async def async_invoke_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        return await async_client.models.generate_content(**self._request_kwargs(request))

    async def async_stream_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        stream = async_client.models.generate_content_stream(**self._request_kwargs(request))
        if hasattr(stream, "__await__"):
            stream = await stream
        return stream

    def get_usage_from_response(self, response: Any) -> dict[str, Any]:
        usage = _usage_field(response, "usage_metadata", "usageMetadata", "usage")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage_metadata") or response.get("usageMetadata") or response.get("usage")
        if usage is None:
            return {}

        payload = {
            "inputTokens": _usage_int(usage, "prompt_token_count", "promptTokenCount", "input_token_count", "inputTokenCount"),
            "outputTokens": _usage_int(usage, "candidates_token_count", "candidatesTokenCount", "output_token_count", "outputTokenCount"),
            "totalTokens": _usage_int(usage, "total_token_count", "totalTokenCount"),
            "cachedInputTokens": _usage_int(usage, "cached_content_token_count", "cachedContentTokenCount"),
            "reasoningTokens": _usage_int(usage, "thoughts_token_count", "thoughtsTokenCount"),
            "toolUsePromptTokens": _usage_int(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount"),
            "costUsd": _usage_float(usage, "cost_usd", "total_cost", "total_cost_usd"),
            "usageSource": "provider",
        }
        if payload["totalTokens"] is None and payload["inputTokens"] is not None and payload["outputTokens"] is not None:
            payload["totalTokens"] = payload["inputTokens"] + payload["outputTokens"]
        return {key: value for key, value in payload.items() if value is not None}
