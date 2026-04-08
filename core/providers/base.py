"""
Provider 基类

定义 LLM Provider 的统一接口。
提供基于 OpenAI SDK 的通用实现，子类只需覆写差异化方法。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    LLM Provider 抽象基类
    
    所有 Provider（OpenAI、Claude、Gemini 等）均通过 OpenAI 兼容层调用。
    通用的 invoke / stream / invoke_with_tools 已在此基类中实现，
    子类只需覆写 format_tool_result 等差异化方法即可。
    
    异步支持：
    - async_invoke / async_stream / async_invoke_with_tools 使用 AsyncOpenAI 客户端
    - 异步客户端惰性创建，首次调用异步方法时才初始化
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs
    ):
        """
        初始化 Provider
        
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 地址
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 超时时间
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self.client = self._create_client()
        self._async_client: Optional[AsyncOpenAI] = None
    
    def _create_client(self) -> OpenAI:
        """创建 OpenAI 兼容客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def _get_async_client(self) -> AsyncOpenAI:
        """获取或创建 AsyncOpenAI 客户端（惰性初始化）"""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
        return self._async_client
    
    # ==================== 同步调用实现 ====================

    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str | None:
        """同步调用 LLM"""
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            msg = response.choices[0].message
            content = msg.content
            if not content and self.get_thinking_content(response):
                content = self.get_thinking_content(response)
            logger.info(f"✅ {self.provider_name} Provider 响应成功")
            return content or ""
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 调用失败: {e}")
            raise
    
    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用 LLM"""
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
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
    
    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """带工具调用的 LLM 调用"""
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
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
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """同步流式工具调用，返回统一事件流。"""
        temperature = temperature if temperature is not None else self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            logger.info(f"✅ {self.provider_name} Provider 流式工具调用开始")
            state = self._init_chat_tool_stream_state()
            for chunk in response:
                for event in self._extract_chat_stream_events(chunk, state):
                    yield event
            yield self._finalize_chat_tool_stream_state(state)
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 流式工具调用失败: {e}")
            raise

    # ==================== 异步调用实现 ====================

    async def async_invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str | None:
        """异步调用 LLM"""
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        
        try:
            logger.info(f"messages:{messages}")
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info(f"response:{response}")
            msg = response.choices[0].message
            content = msg.content
            
            if not content and self.get_thinking_content(response):
                content = self.get_thinking_content(response)
            logger.info(f"✅ {self.provider_name} Provider 异步响应成功")
            return content or ""
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步调用失败: {e}")
            raise

    async def async_stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """异步流式调用 LLM"""
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        
        try:
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            logger.info(f"✅ {self.provider_name} Provider 异步流式响应开始")
            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步流式调用失败: {e}")
            raise

    async def async_invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """异步带工具调用的 LLM 调用"""
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()
        
        try:
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
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
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式工具调用，返回统一事件流。"""
        temperature = temperature if temperature is not None else self.temperature
        async_client = self._get_async_client()

        try:
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            logger.info(f"✅ {self.provider_name} Provider 异步流式工具调用开始")
            state = self._init_chat_tool_stream_state()
            async for chunk in response:
                for event in self._extract_chat_stream_events(chunk, state):
                    yield event
            yield self._finalize_chat_tool_stream_state(state)
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步流式工具调用失败: {e}")
            raise
    
    # ==================== 需子类覆写的方法 ====================

    @abstractmethod
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具执行结果为该 Provider 需要的消息格式
        
        Args:
            content: 工具执行结果
            tool_id: 工具调用 ID
            tool_name: 工具名称
            
        Returns:
            格式化后的消息字典
        """
        pass

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None
    ) -> dict[str, Any]:
        """根据统一 tool call 结构构造 assistant 消息。"""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
                for tool_call in tool_calls
            ]
        return message

    def format_assistant_response(self, response: Any) -> dict[str, Any]:
        """将 provider 响应对象转换为可复用、可序列化的 assistant 消息。"""
        content = getattr(response, "content", None) or ""
        tool_calls_data = getattr(response, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []

        for index, tool_call in enumerate(tool_calls_data):
            function = getattr(tool_call, "function", None)
            tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None) or f"tool_call_{index}",
                    "type": getattr(tool_call, "type", None) or "function",
                    "name": getattr(function, "name", None) or "",
                    "arguments": getattr(function, "arguments", None) or "",
                }
            )

        return self.format_assistant_message(
            content=content,
            tool_calls=tool_calls or None,
        )

    # ==================== 通用辅助方法 ====================

    def _init_chat_tool_stream_state(self) -> dict[str, Any]:
        return {
            "text_parts": [],
            "thinking_parts": [],
            "tool_calls": {},
            "terminal_emitted": False,
        }

    def _extract_chat_stream_events(
        self,
        chunk: Any,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return events

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return events

        reasoning_delta = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
        )
        if reasoning_delta:
            state["thinking_parts"].append(reasoning_delta)
            events.append({
                "type": "thinking_delta",
                "delta": reasoning_delta,
            })

        content = getattr(delta, "content", None) or ""
        if content:
            state["text_parts"].append(content)
            events.append({
                "type": "text_delta",
                "delta": content,
            })

        for tool_call in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_call, "index", 0) or 0
            current = state["tool_calls"].setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "name": "",
                    "arguments": "",
                },
            )

            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id:
                current["id"] = tool_call_id

            function = getattr(tool_call, "function", None)
            if function is not None:
                function_name = getattr(function, "name", None)
                if function_name:
                    current["name"] = function_name
                function_arguments = getattr(function, "arguments", None)
                if function_arguments:
                    current["arguments"] += function_arguments

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "tool_calls":
            events.append({
                "type": "tool_calls",
                "tool_calls": self._normalize_stream_tool_calls(state["tool_calls"]),
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
            })
            state["terminal_emitted"] = True
        elif finish_reason in {"stop", "length", "content_filter"}:
            events.append({
                "type": "final_response",
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
                "finish_reason": finish_reason,
            })
            state["terminal_emitted"] = True

        return events

    def _finalize_chat_tool_stream_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("terminal_emitted"):
            return {
                "type": "stream_end",
            }
        tool_calls = self._normalize_stream_tool_calls(state["tool_calls"])
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
            }
        return {
            "type": "final_response",
            "content": "".join(state["text_parts"]),
            "thinking": "".join(state["thinking_parts"]),
        }

    def _normalize_stream_tool_calls(self, tool_calls_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            tool_call = dict(tool_calls_by_index[index])
            if not tool_call.get("id"):
                tool_call["id"] = f"tool_call_{index}"
            normalized.append(tool_call)
        return normalized

    def get_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容（如果模型支持）"""
        thinking= getattr(response, 'reasoning_content', None)
        content= getattr(response, 'content', None)
        if thinking:
            return thinking
        elif content:
            return content
        else:
            return None
    def get_response_content(self, response: Any) -> Optional[str]:
        """提取响应内容"""
        return getattr(response, 'content', None)
    
    def has_tool_calls(self, response: Any) -> bool:
        """检查响应是否包含工具调用"""
        return hasattr(response, 'tool_calls') and response.tool_calls
    
    def get_tool_calls(self, response: Any) -> list:
        """获取工具调用列表"""
        if self.has_tool_calls(response):
            return response.tool_calls
        return []
    
    @property
    def provider_name(self) -> str:
        """Provider 名称"""
        return self.__class__.__name__.replace('Provider', '').lower()
