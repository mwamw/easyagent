"""
Google Gemini native transport provider.

只负责把 request-ready Gemini history turn 组装成 generate_content 请求。
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from ..base import BaseProvider, _usage_field, _usage_float, _usage_int

logger = logging.getLogger(__name__)


class GoogleNativeProvider(BaseProvider):
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
        aliases = (self.provider_name, "google_native", "gemini_native", "google", "gemini")
        for alias in aliases:
            override = overrides.get(alias)
            if isinstance(override, dict):
                return dict(override)
        return None

    @staticmethod
    def _message_has_function_response(message: Any) -> bool:
        if not isinstance(message, dict) or message.get("role") != "user":
            return False
        parts = message.get("parts")
        if not isinstance(parts, list):
            return False
        return any(isinstance(part, dict) and isinstance(part.get("function_response"), dict) for part in parts)

    @staticmethod
    def _message_has_signed_function_call(message: Any) -> bool:
        if not isinstance(message, dict) or message.get("role") != "model":
            return False
        parts = message.get("parts")
        if not isinstance(parts, list):
            return False
        return any(
            isinstance(part, dict)
            and isinstance(part.get("function_call"), dict)
            and part.get("thought_signature")
            for part in parts
        )

    @staticmethod
    def _is_invalid_thought_signature_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "thought signature is not valid" in text or "corrupted thought signature" in text

    def _build_invalid_signature_fallback_request(self, request: Any) -> Optional[dict[str, Any]]:
        if not isinstance(request, dict):
            return None
        contents = request.get("contents")
        if not isinstance(contents, list) or len(contents) < 2:
            return None

        fallback_index: Optional[int] = None
        for index in range(len(contents) - 1):
            current = contents[index]
            nxt = contents[index + 1]
            if self._message_has_signed_function_call(current) and self._message_has_function_response(nxt):
                fallback_index = index

        if fallback_index is None:
            fallback_request = copy.deepcopy(request)
            changed = False
            for message in fallback_request.get("contents", []):
                if not isinstance(message, dict):
                    continue
                parts = message.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if isinstance(part, dict) and "thought_signature" in part:
                        part.pop("thought_signature", None)
                        changed = True
            return fallback_request if changed else None

        fallback_request = copy.deepcopy(request)
        fallback_request["contents"] = [
            message
            for index, message in enumerate(copy.deepcopy(contents))
            if index != fallback_index
        ]
        return fallback_request

    def _log_invalid_signature_retry(self, request: Any, fallback_request: Any) -> None:
        original_len = len(request.get("contents", [])) if isinstance(request, dict) else None
        fallback_len = len(fallback_request.get("contents", [])) if isinstance(fallback_request, dict) else None
        logger.warning(
            "google_native returned invalid thought signature; retrying with sanitized thought-signature history "
            "(contents %s -> %s)",
            original_len,
            fallback_len,
        )

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

    def _build_thinking_config(self, reasoning: Optional[dict[str, Any] | str]) -> Optional[dict[str, Any]]:
        reasoning = self._normalize_reasoning(reasoning)
        if not reasoning or reasoning.get("enabled") is False:
            return None
        override = self._reasoning_override(reasoning)
        if override is not None:
            if isinstance(override.get("thinking_config"), dict):
                return dict(override["thinking_config"])
            return override
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
            level_map = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high"}
            return {"thinking_level": level_map.get(str(effort), "medium")}
        budget_map = {"low": 1024, "medium": 8192, "high": 24576, "xhigh": 32768}
        return {"thinking_budget": budget_map.get(str(effort), 8192)}

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
        try:
            return self.client.models.generate_content(**self._request_kwargs(request))
        except Exception as exc:
            if not self._is_invalid_thought_signature_error(exc):
                raise
            fallback_request = self._build_invalid_signature_fallback_request(request)
            if fallback_request is None:
                raise
            self._log_invalid_signature_retry(request, fallback_request)
            return self.client.models.generate_content(**self._request_kwargs(fallback_request))

    def stream_raw(self, request: Any) -> Any:
        try:
            return self.client.models.generate_content_stream(**self._request_kwargs(request))
        except Exception as exc:
            if not self._is_invalid_thought_signature_error(exc):
                raise
            fallback_request = self._build_invalid_signature_fallback_request(request)
            if fallback_request is None:
                raise
            self._log_invalid_signature_retry(request, fallback_request)
            return self.client.models.generate_content_stream(**self._request_kwargs(fallback_request))

    async def async_invoke_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        try:
            return await async_client.models.generate_content(**self._request_kwargs(request))
        except Exception as exc:
            if not self._is_invalid_thought_signature_error(exc):
                raise
            fallback_request = self._build_invalid_signature_fallback_request(request)
            if fallback_request is None:
                raise
            self._log_invalid_signature_retry(request, fallback_request)
            return await async_client.models.generate_content(**self._request_kwargs(fallback_request))

    async def async_stream_raw(self, request: Any) -> Any:
        async_client = self._get_async_client()
        try:
            stream = async_client.models.generate_content_stream(**self._request_kwargs(request))
            if hasattr(stream, "__await__"):
                stream = await stream
            return stream
        except Exception as exc:
            if not self._is_invalid_thought_signature_error(exc):
                raise
            fallback_request = self._build_invalid_signature_fallback_request(request)
            if fallback_request is None:
                raise
            self._log_invalid_signature_retry(request, fallback_request)
            stream = async_client.models.generate_content_stream(**self._request_kwargs(fallback_request))
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
