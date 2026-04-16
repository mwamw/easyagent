"""
OpenAI Responses API Provider

使用 OpenAI 新版 Responses API (client.responses.create) 而非旧版
Chat Completions API (client.chat.completions.create)。

新 API 主要差异：
- 入参：input= 替代 messages=
- 工具定义：扁平结构（无嵌套 "function" 键）
- 响应：response.output 列表（按 type 区分 reasoning / function_call / message）
- 工具结果：{"type": "function_call_output", "call_id": ..., "output": ...}
- Reasoning：支持原生 reasoning={"effort": ..., "summary": ...} 参数
"""
from typing import Optional, Any, Generator
import logging
import json
from .openai_compatible_provider import OpenAICompatibleProviderBase

logger = logging.getLogger(__name__)


class OpenAIResponsesProvider(OpenAICompatibleProviderBase):
    """
    OpenAI Responses API Provider

    适用场景：
    - 需要原生 Reasoning（reasoning={"effort": "medium"} 参数）的模型
    - 希望直接使用 responses.create 接口风格的场景
    - gpt-4o、o1、o3、gpt-5 等支持 Responses API 的模型

    与 OpenAIProvider 的区别：
    - 使用 client.responses.create 而非 client.chat.completions.create
    - 工具定义使用扁平格式
    - 响应对象为 response.output 列表
    """

    # ==================== 核心调用 ====================

    def invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str | None:
        """同步调用（无工具）"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 响应成功")
            return response.output_text
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 调用失败: {e}")
            raise

    def invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """同步调用并返回完整 Responses API response。"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        try:
            converted_input = self._convert_input(messages)
            response = self.client.responses.create(
                model=self.model,
                input=converted_input,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 原始响应成功")
            return response
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 原始调用失败: {e}")
            raise

    def stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式调用（无工具）"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                stream=True,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 流式响应开始")
            for event in response:
                # 兼容不同 SDK 版本的事件格式
                # response.stream_event 或直接迭代 delta
                delta = getattr(event, "delta", None)
                if delta is not None:
                    text = getattr(delta, "text", None) or (delta if isinstance(delta, str) else "")
                    if text:
                        yield text
                else:
                    # 某些版本直接 yield 文本字符串
                    if isinstance(event, str) and event:
                        yield event
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 流式调用失败: {e}")
            raise

    def stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """同步流式调用，返回包含 thinking / text / final 的统一事件。"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        try:
            converted_input = self._convert_input(messages)
            response = self.client.responses.create(
                model=self.model,
                input=converted_input,
                stream=True,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 事件流响应开始")
            state = self._init_responses_tool_stream_state()
            for event in response:
                for item in self._extract_responses_stream_events(event, state):
                    if item.get("type") != "tool_calls":
                        yield item
            final_event = self._finalize_responses_tool_stream_state(state)
            if final_event.get("type") != "stream_end" and final_event.get("type") != "tool_calls":
                yield final_event
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 事件流调用失败: {e}")
            raise

    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        带工具调用的 LLM 调用（Responses API 格式）

        工具定义自动从 OpenAI 嵌套格式转换为 Responses API 扁平格式：
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
          → {"type": "function", "name": ..., "description": ..., "parameters": ...}

        返回原始 response 对象（由上层通过辅助方法解析）。
        """
        if reasoning:
            kwargs["reasoning"] = reasoning
        try:
            flat_tools = self._convert_tools(tools)
            params = self._base_params(temperature, **kwargs)

            # 将 Message 对象转换为 dict（Responses API 只接受 dict 或 SDK output 对象）
            converted_input = self._convert_input(messages)

            logger.debug(f"📤 发送给 Responses API 的 input 条目数: {len(converted_input)}")
            response = self.client.responses.create(
                model=self.model,
                input=converted_input,
                tools=flat_tools,
                **params
            )
            # 调试：打印 output 中各条目的 type，方便诊断工具调用检测问题
            _output = getattr(response, "output", None)
            if _output is not None:
                _types = [getattr(item, "type", repr(item)) for item in _output]
                logger.info(f"📦 Responses API output 类型列表: {_types}")
            logger.info(f"✅ {self.provider_name} Provider 工具调用响应成功")
            return response
        except Exception as e:
            # logger.error(f"❌ {self.provider_name} Provider 工具调用失败: {e}")
            raise

    def stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,

        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """同步流式工具调用（Responses API 格式）。"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        try:
            flat_tools = self._convert_tools(tools)
            params = self._base_params(temperature, **kwargs)
            converted_input = self._convert_input(messages)
            response = self.client.responses.create(
                model=self.model,
                input=converted_input,
                tools=flat_tools,
                stream=True,
                **params
            )
            logger.info(f"✅ {self.provider_name} Provider 流式工具调用开始")
            state = self._init_responses_tool_stream_state()
            for event in response:
                for item in self._extract_responses_stream_events(event, state):
                    yield item
            yield self._finalize_responses_tool_stream_state(state)
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 流式工具调用失败: {e}")
            raise


    # ==================== 异步调用实现 ====================

    async def async_invoke(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,

        **kwargs
    ) -> str | None:
        """异步调用（无工具）"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        async_client = self._get_async_client()
        try:
            response = await async_client.responses.create(
                model=self.model,
                input=messages,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 异步响应成功")
            return response.output_text
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步调用失败: {e}")
            raise

    async def async_invoke_raw(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,

        **kwargs
    ) -> Any:
        """异步调用并返回完整 Responses API response。"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        async_client = self._get_async_client()
        try:
            converted_input = self._convert_input(messages)
            response = await async_client.responses.create(
                model=self.model,
                input=converted_input,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 异步原始响应成功")
            return response
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步原始调用失败: {e}")
            raise

    async def async_stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        """异步流式调用（无工具）"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        async_client = self._get_async_client()
        try:
            response = await async_client.responses.create(
                model=self.model,
                input=messages,
                stream=True,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 异步流式响应开始")
            async for event in response:
                delta = getattr(event, "delta", None)
                if delta is not None:
                    text = getattr(delta, "text", None) or (delta if isinstance(delta, str) else "")
                    if text:
                        yield text
                else:
                    if isinstance(event, str) and event:
                        yield event
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步流式调用失败: {e}")
            raise

    async def async_stream_events(
        self,
        messages: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        """异步流式调用，返回包含 thinking / text / final 的统一事件。"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        async_client = self._get_async_client()
        try:
            converted_input = self._convert_input(messages)
            response = await async_client.responses.create(
                model=self.model,
                input=converted_input,
                stream=True,
                **self._base_params(temperature, **kwargs)
            )
            logger.info(f"✅ {self.provider_name} Provider 异步事件流响应开始")
            state = self._init_responses_tool_stream_state()
            async for event in response:
                for item in self._extract_responses_stream_events(event, state):
                    if item.get("type") != "tool_calls":
                        yield item
            final_event = self._finalize_responses_tool_stream_state(state)
            if final_event.get("type") != "stream_end" and final_event.get("type") != "tool_calls":
                yield final_event
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步事件流调用失败: {e}")
            raise

    async def async_invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,

        **kwargs
    ) -> Any:
        """异步带工具调用（Responses API 格式）"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        async_client = self._get_async_client()
        try:
            flat_tools = self._convert_tools(tools)
            params = self._base_params(temperature, **kwargs)
            converted_input = self._convert_input(messages)

            logger.debug(f"📤 异步发送给 Responses API 的 input 条目数: {len(converted_input)}")
            response = await async_client.responses.create(
                model=self.model,
                input=converted_input,
                tools=flat_tools,
                **params
            )
            _output = getattr(response, "output", None)
            if _output is not None:
                _types = [getattr(item, "type", repr(item)) for item in _output]
                logger.info(f"📦 Responses API 异步 output 类型列表: {_types}")
            logger.info(f"✅ {self.provider_name} Provider 异步工具调用响应成功")
            return response
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步工具调用失败: {e}")
            raise

    async def async_stream_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,

        **kwargs
    ):
        """异步流式工具调用（Responses API 格式）。"""
        if reasoning:
            kwargs["reasoning"] = reasoning
        async_client = self._get_async_client()
        try:
            flat_tools = self._convert_tools(tools)
            params = self._base_params(temperature, **kwargs)
            converted_input = self._convert_input(messages)
            response = await async_client.responses.create(
                model=self.model,
                input=converted_input,
                tools=flat_tools,
                stream=True,
                **params
            )
            logger.info(f"✅ {self.provider_name} Provider 异步流式工具调用开始")
            state = self._init_responses_tool_stream_state()
            async for event in response:
                for item in self._extract_responses_stream_events(event, state):
                    yield item
            yield self._finalize_responses_tool_stream_state(state)
        except Exception as e:
            logger.error(f"❌ {self.provider_name} Provider 异步流式工具调用失败: {e}")
            raise


    # ==================== 响应解析辅助方法 ====================

    def has_tool_calls(self, response: Any) -> bool:
        """检查响应中是否包含工具调用（function_call 类型条目）"""
        output = getattr(response, "output", None)
        if output is None:
            return False
        return any(getattr(item, "type", None) == "function_call" for item in output)

    def get_tool_calls(self, response: Any) -> list:
        """返回所有 function_call 类型的 output 条目"""
        output = getattr(response, "output", None)
        if not output:
            return []
        return [item for item in output if getattr(item, "type", None) == "function_call"]

    def get_thinking_content(self, response: Any) -> Optional[str]:
        """
        提取 reasoning 输出内容

        Responses API 的思考内容在：
            response.output[i].type == "reasoning"
            response.output[i].summary  (list of summary blocks)
            
        如果原生 reasoning 不存在，但包含工具调用且有伴随的 message，
        也将这个 message 视作“执行工具前的思考文本”。
        """
        output = getattr(response, "output", None)
        if not output:
            return None
            
        reasoning_text = []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    # summary 是一个列表，每个元素有 text 字段
                    if isinstance(summary, list):
                        parts = []
                        for s in summary:
                            text = getattr(s, "text", None) or (s if isinstance(s, str) else "")
                            if text:
                                parts.append(text)
                        if parts:
                            reasoning_text.append("\n".join(parts))
                    else:
                        reasoning_text.append(str(summary))

        # 优先返回原生的 reasoning
        if reasoning_text:
            return "\n".join(reasoning_text)

        return None

    def get_response_content(self, response: Any) -> Optional[str]:
        """提取最终文本内容（message 类型条目）"""
        output = getattr(response, "output", None)
        if not output:
            return None
        selected = self._select_message_item(output)
        if selected is not None:
            content = self._extract_output_message_text(selected)
            if content:
                return content
        # 兜底：output_text 属性
        return getattr(response, "output_text", None)

    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具执行结果（Responses API 格式）

        Responses API 工具结果格式：
        {
            "type": "function_call_output",
            "call_id": "<tool_call_id>",
            "output": "<result_string>"
        }
        """
        return {
            "type": "function_call_output",
            "call_id": tool_id,
            "output": content,
        }

    def format_assistant_response(self, response: Any, include_reasoning: bool = False) -> list:
        """
        将 response.output 列表直接作为 assistant 消息追加到 input

        Responses API 支持将上一轮的 output 直接作为下一轮 input 的元素。
        """
        result = []
        for item in getattr(response, "output", []):
            serialized = self._serialize_assistant_history_item(item, include_reasoning=include_reasoning)
            if serialized is not None:
                result.append(serialized)
        return result

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if thinking:
            result.append(
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": thinking,
                        }
                    ],
                }
            )
        if tool_calls:
            for tool_call in tool_calls:
                result.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                )
        if content:
            result.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                        }
                    ],
                }
            )
        return result

    # ==================== 私有辅助 ====================

    def _init_responses_tool_stream_state(self) -> dict[str, Any]:
        return {
            "text_parts": [],
            "thinking_parts": [],
            "tool_calls": {},
            "output_items": [],
            "output_item_keys": {},
            "terminal_emitted": False,
        }

    def _extract_responses_stream_events(
        self,
        event: Any,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        event_type = getattr(event, "type", None)
        if not event_type:
            return events

        if event_type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            delta = getattr(event, "delta", None) or ""
            if delta:
                state["text_parts"].append(delta)
                events.append({"type": "text_delta", "delta": delta})
            return events

        if event_type in {
            "response.reasoning.delta",
            "response.reasoning_summary_text.delta",
        }:
            delta = getattr(event, "delta", None) or ""
            if delta:
                state["thinking_parts"].append(delta)
                events.append({"type": "thinking_delta", "delta": delta})
            return events

        if event_type == "response.function_call_arguments.delta":
            self._merge_responses_function_call_delta(event, state)
            return events

        if event_type in {
            "response.output_item.added",
            "response.output_item.done",
        }:
            item = getattr(event, "item", None)
            if item is not None:
                self._merge_responses_output_item(item, state)
                if getattr(item, "type", None) == "function_call" and event_type.endswith(".done"):
                    state["terminal_emitted"] = True
            return events

        if event_type == "response.completed":
            tool_calls = self._normalize_responses_tool_calls(state["tool_calls"])
            if tool_calls:
                events.append(
                    {
                        "type": "tool_calls",
                        "tool_calls": tool_calls,
                        "content": "".join(state["text_parts"]),
                        "thinking": "".join(state["thinking_parts"]),
                        "assistant_items": self._build_stream_assistant_items(state, tool_calls),
                    }
                )
            else:
                response = getattr(event, "response", None)
                selected_message = self._select_message_item(state.get("output_items", []))
                content = (
                    self._extract_output_message_text(selected_message)
                    if selected_message is not None
                    else ""
                ) or "".join(state["text_parts"]) or self.get_response_content(response) or ""
                events.append(
                    {
                        "type": "final_response",
                        "content": content,
                        "thinking": "".join(state["thinking_parts"]),
                        "assistant_items": self._build_stream_assistant_items(state, tool_calls),
                    }
                )
            state["terminal_emitted"] = True
            return events

        return events

    def _merge_responses_function_call_delta(self, event: Any, state: dict[str, Any]) -> None:
        key = (
            getattr(event, "item_id", None)
            or getattr(event, "call_id", None)
            or str(getattr(event, "output_index", 0))
        )
        current = state["tool_calls"].setdefault(
            key,
            {
                "id": getattr(event, "call_id", None),
                "name": "",
                "arguments": "",
                "type": "function",
            },
        )
        if getattr(event, "call_id", None):
            current["id"] = event.call_id
        delta = getattr(event, "delta", None) or ""
        if delta:
            current["arguments"] += delta
        output_item = self._get_stream_output_item(state, key)
        if output_item is not None:
            output_item["type"] = "function_call"
            if current.get("id"):
                output_item["call_id"] = current["id"]
            if current.get("name"):
                output_item["name"] = current["name"]
            output_item["arguments"] = current["arguments"]

    def _merge_responses_output_item(self, item: Any, state: dict[str, Any]) -> None:
        item_type = getattr(item, "type", None)
        serialized_item = self._serialize_output_item(item)
        self._set_stream_output_item(state, item, serialized_item)
        if item_type == "function_call":
            key = (
                getattr(item, "id", None)
                or getattr(item, "call_id", None)
                or getattr(item, "name", None)
                or str(len(state["tool_calls"]))
            )
            current = state["tool_calls"].setdefault(
                key,
                {
                    "id": getattr(item, "call_id", None),
                    "name": "",
                    "arguments": "",
                    "type": "function",
                },
            )
            if getattr(item, "call_id", None):
                current["id"] = item.call_id
            if getattr(item, "name", None):
                current["name"] = item.name
            arguments = getattr(item, "arguments", None)
            if arguments:
                current["arguments"] = arguments
        elif item_type == "message":
            message_text = self._extract_output_message_text(item)
            if message_text:
                current_text = "".join(state["text_parts"])
                if not current_text.endswith(message_text):
                    state["text_parts"].append(message_text)

    @staticmethod
    def _extract_output_message_text(item: Any) -> str:
        if getattr(item, "type", None) != "message":
            return ""
        content = getattr(item, "content", None)
        parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "output_text":
                    text = getattr(block, "text", None) or ""
                    if text:
                        parts.append(text)
        elif isinstance(content, str):
            parts.append(content)
        return "".join(parts)

    def _normalize_responses_tool_calls(self, tool_calls_by_key: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, key in enumerate(tool_calls_by_key):
            tool_call = dict(tool_calls_by_key[key])
            if not tool_call.get("id"):
                tool_call["id"] = f"call_{index}"
            if not tool_call.get("arguments"):
                tool_call["arguments"] = json.dumps({})
            normalized.append(tool_call)
        return normalized

    def _finalize_responses_tool_stream_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("terminal_emitted"):
            return {"type": "stream_end"}
        tool_calls = self._normalize_responses_tool_calls(state["tool_calls"])
        if tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": "".join(state["text_parts"]),
                "thinking": "".join(state["thinking_parts"]),
                "assistant_items": self._build_stream_assistant_items(state, tool_calls),
            }
        return {
            "type": "final_response",
            "content": "".join(state["text_parts"]),
            "thinking": "".join(state["thinking_parts"]),
            "assistant_items": self._build_stream_assistant_items(state, tool_calls),
        }

    def _build_stream_assistant_items(
        self,
        state: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        assistant_items = []
        for item in state.get("output_items", []):
            if not isinstance(item, dict):
                continue
            serialized = self._serialize_assistant_history_item(item, include_reasoning=True)
            if serialized is not None:
                assistant_items.append(serialized)
        if assistant_items:
            return assistant_items
        return self.format_assistant_message(
            content="".join(state.get("text_parts", [])),
            tool_calls=tool_calls or None,
            thinking="".join(state.get("thinking_parts", [])) or None,
        )

    def _set_stream_output_item(
        self,
        state: dict[str, Any],
        item: Any,
        serialized_item: dict[str, Any],
    ) -> None:
        key = self._get_stream_output_item_key(item)
        if key in state["output_item_keys"]:
            state["output_items"][state["output_item_keys"][key]] = serialized_item
            return
        state["output_item_keys"][key] = len(state["output_items"])
        state["output_items"].append(serialized_item)

    @staticmethod
    def _get_stream_output_item_key(item: Any) -> str:
        return (
            str(getattr(item, "id", None) or "")
            or str(getattr(item, "call_id", None) or "")
            or f"{getattr(item, 'type', 'unknown')}:{getattr(item, 'name', None) or ''}:{getattr(item, 'role', None) or ''}"
        )

    def _get_stream_output_item(
        self,
        state: dict[str, Any],
        key: Any,
    ) -> Optional[dict[str, Any]]:
        index = state["output_item_keys"].get(str(key))
        if index is None:
            return None
        item = state["output_items"][index]
        if isinstance(item, dict):
            return item
        return None

    def _serialize_output_item(self, item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            payload = item.to_dict()
            if isinstance(payload, dict):
                return self._to_serializable(payload)
        if isinstance(item, dict):
            return self._to_serializable(dict(item))

        payload = {"type": getattr(item, "type", "unknown")}
        for attr in ("id", "call_id", "name", "arguments", "role", "status", "phase"):
            value = getattr(item, attr, None)
            if value is not None:
                payload[attr] = self._to_serializable(value)
        if hasattr(item, "content"):
            payload["content"] = self._to_serializable(getattr(item, "content"))
        if hasattr(item, "summary"):
            payload["summary"] = self._to_serializable(getattr(item, "summary"))
        if hasattr(item, "text"):
            payload["text"] = self._to_serializable(getattr(item, "text"))
        if hasattr(item, "encrypted_content"):
            payload["encrypted_content"] = self._to_serializable(getattr(item, "encrypted_content"))
        return payload

    def _serialize_assistant_history_item(self, item: Any, include_reasoning: bool = False) -> Optional[dict[str, Any]]:
        payload = self._serialize_output_item(item)
        item_type = payload.get("type")
        if item_type == "reasoning":
            if not include_reasoning:
                return None
            return {
                "type": "reasoning",
                "summary": self._to_serializable(payload.get("summary", [])),
            }
        if item_type == "message":
            message: dict[str, Any] = {
                "type": "message",
                "role": payload.get("role", "assistant"),
                "content": self._to_serializable(payload.get("content", [])),
            }
            phase = payload.get("phase")
            if phase:
                message["phase"] = phase
            return message
        if item_type == "function_call":
            return {
                "type": "function_call",
                "call_id": payload.get("call_id") or payload.get("id"),
                "name": payload.get("name", ""),
                "arguments": payload.get("arguments", ""),
            }
        if item_type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": payload.get("call_id"),
                "output": payload.get("output", ""),
            }
        return payload

    @staticmethod
    def _select_message_item(items: list[Any]) -> Optional[Any]:
        if not items:
            return None
        assistant_messages = []
        final_messages = []
        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type != "message":
                continue
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            phase = item.get("phase") if isinstance(item, dict) else getattr(item, "phase", None)
            if role == "assistant":
                assistant_messages.append(item)
                if phase == "final_answer":
                    final_messages.append(item)
        if final_messages:
            return final_messages[-1]
        if assistant_messages:
            return assistant_messages[-1]
        return None

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [cls._to_serializable(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_serializable(item) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._to_serializable(item)
                for key, item in value.items()
            }
        if hasattr(value, "to_dict"):
            payload = value.to_dict()
            if isinstance(payload, dict):
                return cls._to_serializable(payload)
        payload: dict[str, Any] = {}
        for attr in ("type", "text", "id", "call_id", "name", "arguments", "content", "summary", "role", "status", "phase"):
            attr_value = getattr(value, attr, None)
            if attr_value is not None:
                payload[attr] = cls._to_serializable(attr_value)
        if payload:
            return payload
        return str(value)

    def _base_params(self, temperature: Optional[float] = None, **kwargs) -> dict:
        """整理传给 responses.create 的通用参数"""
        params: dict = {}
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        # Responses API 不支持 temperature 参数（reasoning 模型均不支持）
        # 如果调用方传入了 reasoning 参数，直接透传
        for key in ("reasoning", "text", "truncation"):
            if key in kwargs:
                params[key] = kwargs.pop(key)
        # 其余 kwargs 透传（过滤掉 chat API 专属字段）
        _chat_only = {"stream", "messages", "temperature"}
        for k, v in kwargs.items():
            if k not in _chat_only:
                params[k] = v
        return params

    @staticmethod
    def _convert_tools(tools: list) -> list:
        """
        将 OpenAI Chat API 格式的工具定义转换为 Responses API 扁平格式

        输入（Chat API 格式）：
            [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

        输出（Responses API 格式）：
            [{"type": "function", "name": ..., "description": ..., "parameters": ...}]
        """
        flat = []
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get("type") == "function" and "function" in tool:
                    # Chat API 嵌套格式 → 扁平化
                    fn = tool["function"]
                    flat.append({
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    })
                else:
                    # 已经是扁平格式或其他格式，直接传入
                    flat.append(tool)
            else:
                flat.append(tool)
        return flat

    @staticmethod
    def _convert_input(messages: list) -> list:
        """
        将 messages 列表中的 Message 对象转换为字典

        Responses API 的 input 接受：
          - 普通山年字典（{"role": ..., "content": ...}）
          - SDK 返回的 output 对象（ResponseFunctionToolCall 等）
          - 工具结果字典（{"type": "function_call_output", ...}）
        不接受我们自定义的 Message 类对象。
        """
        result = []
        for item in messages:
            # 如果是我们的 Message 对象，转换为字典
            if hasattr(item, 'to_dict'):
                sanitized = OpenAIResponsesProvider._sanitize_input_item(item.to_dict())
            elif hasattr(item, 'role') and hasattr(item, 'content') and not isinstance(item, dict):
                # 备用：手动构建字典
                sanitized = OpenAIResponsesProvider._sanitize_input_item(
                    {"role": item.role, "content": item.content or ""}
                )
            else:
                # 已经是字典或 SDK 对象，直接保留
                sanitized = OpenAIResponsesProvider._sanitize_input_item(item)
            if sanitized is not None:
                result.append(sanitized)
        return result

    @staticmethod
    def _sanitize_input_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        item_type = item.get("type")
        if item_type == "message":
            return {
                "type": "message",
                "role": item.get("role", "assistant"),
                "content": item.get("content", []),
            }
        if item_type == "function_call":
            return {
                "type": "function_call",
                "call_id": item.get("call_id") or item.get("id"),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            }
        if item_type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": item.get("call_id"),
                "output": item.get("output", ""),
            }
        if item_type == "reasoning":
            sanitized = {
                "type": "reasoning",
            }
            for key in ("id", "summary", "content", "encrypted_content", "status"):
                if key in item and item.get(key) is not None:
                    sanitized[key] = item.get(key)
            return sanitized
        if "role" in item and "content" in item:
            sanitized = {
                "role": item.get("role"),
                "content": item.get("content", ""),
            }
            if item.get("reasoning_content") is not None:
                sanitized["reasoning_content"] = item.get("reasoning_content")
            return sanitized
        return item

    def prepare_message_for_request(self, message: Any) -> Any:
        """将 history 中 richer message 转换为 Responses API 可接受的输入项。"""
        return self._sanitize_input_item(message)
