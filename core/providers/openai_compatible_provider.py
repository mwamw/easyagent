"""
OpenAI 兼容 Provider 基类

承载基于 `openai` SDK 且遵循 Chat Completions 风格协议的通用实现。
"""

from __future__ import annotations

from typing import Optional, Any, Generator, AsyncGenerator
import logging

from openai import OpenAI, AsyncOpenAI

from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProviderBase(BaseProvider):
    """OpenAI Chat Completions 风格 Provider 的共享实现。"""

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

    def invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        temperature = temperature if temperature is not None else self.temperature
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = self.client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 原始响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 原始调用失败: {e}")
            raise

    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        temperature = temperature if temperature is not None else self.temperature
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = self.client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 流式响应开始")
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 流式调用失败: {e}")
            raise

    def stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        temperature = temperature if temperature is not None else self.temperature
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        terminal_emitted = False
        try:
            response = self.client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 事件流响应开始")
            for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_delta:
                    thinking_parts.append(reasoning_delta)
                    yield {"type": "thinking_delta", "delta": reasoning_delta}
                content_delta = getattr(delta, "content", None) or ""
                if content_delta:
                    text_parts.append(content_delta)
                    yield {"type": "text_delta", "delta": content_delta}
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason in {"stop", "length", "content_filter"}:
                    terminal_emitted = True
                    yield {
                        "type": "final_response",
                        "content": "".join(text_parts),
                        "thinking": "".join(thinking_parts),
                        "finish_reason": finish_reason,
                    }
            if not terminal_emitted:
                yield {
                    "type": "final_response",
                    "content": "".join(text_parts),
                    "thinking": "".join(thinking_parts),
                }
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 事件流调用失败: {e}")
            raise

    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        temperature = temperature if temperature is not None else self.temperature
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = self.client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 工具调用响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 工具调用失败: {e}")
            raise

    def stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        temperature = temperature if temperature is not None else self.temperature
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = self.client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 流式工具调用开始")
            state = self._init_chat_tool_stream_state()
            for chunk in response:
                for event in self._extract_chat_stream_events(chunk, state):
                    yield event
            yield self._finalize_chat_tool_stream_state(state)
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 流式工具调用失败: {e}")
            raise

    async def async_invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = await async_client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 异步原始响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步原始调用失败: {e}")
            raise

    async def async_stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = await async_client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 异步流式响应开始")
            async for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步流式调用失败: {e}")
            raise

    async def async_stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        terminal_emitted = False
        try:
            response = await async_client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 异步事件流响应开始")
            async for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_delta:
                    thinking_parts.append(reasoning_delta)
                    yield {"type": "thinking_delta", "delta": reasoning_delta}
                content_delta = getattr(delta, "content", None) or ""
                if content_delta:
                    text_parts.append(content_delta)
                    yield {"type": "text_delta", "delta": content_delta}
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason in {"stop", "length", "content_filter"}:
                    terminal_emitted = True
                    yield {
                        "type": "final_response",
                        "content": "".join(text_parts),
                        "thinking": "".join(thinking_parts),
                        "finish_reason": finish_reason,
                    }
            if not terminal_emitted:
                yield {
                    "type": "final_response",
                    "content": "".join(text_parts),
                    "thinking": "".join(thinking_parts),
                }
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步事件流调用失败: {e}")
            raise

    async def async_invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = await async_client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 异步工具调用响应成功")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步工具调用失败: {e}")
            raise

    async def async_stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if reasoning and reasoning.get("effort"):
            params["reasoning_effort"] = reasoning["effort"]
        params.update(kwargs)
        try:
            response = await async_client.chat.completions.create(**params)
            logger.info(f"✅ {self.provider_name} Provider 异步流式工具调用开始")
            state = self._init_chat_tool_stream_state()
            async for chunk in response:
                for event in self._extract_chat_stream_events(chunk, state):
                    yield event
            yield self._finalize_chat_tool_stream_state(state)
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步流式工具调用失败: {e}")
            raise
