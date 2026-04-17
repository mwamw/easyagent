"""
OpenAI Responses transport provider.

只负责将 request-ready input items 组装到 responses.create，并返回原始 response / stream。
"""

from __future__ import annotations

from typing import Any, Optional

from ..openai_compat.transport import OpenAICompatibleProviderBase


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

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                function = tool["function"]
                flat.append(
                    {
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}) or {},
                    }
                )
            else:
                flat.append(tool)
        return flat

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
            params["tools"] = self._convert_tools(tools)
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
