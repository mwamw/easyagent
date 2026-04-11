from typing_extensions import override
from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage, MetaUserMessage
from core.Config import Config
from typing import Optional, Any, AsyncGenerator, TYPE_CHECKING
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
import asyncio
import json
import logging
from datetime import datetime
from uuid import uuid4
from core.Exception import *
from prompt import (
    PromptBlock,
    SystemPromptTemplate,
    build_output_efficiency_section,
    build_safety_section,
    build_task_execution_section,
    build_tone_style_section,
    build_tool_policy_section,
    build_visibility_section,
    format_tool_catalog,
    format_tool_inventory,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage
    from context.manager import ContextManager

class BasicAgent(BaseAgent):
    """基础智能体实现，支持可选的工具调用和记忆功能"""

    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        verbose_thinking: bool = False,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        history_via_context_manager: bool = False,
        callback_manager=None,
        skill_manager=None,
    ):
        """
        初始化 BasicAgent
        
        Args:
            name: 智能体名称
            llm: LLM 实例
            system_prompt: 系统提示词
            enable_tool: 是否启用工具调用
            tool_registry: 工具注册表
            description: 智能体描述
            config: 配置对象
            verbose_thinking: 是否显示 LLM 的思考过程
            
        Raises:
            ParameterValidationError: 参数验证失败
            ToolRegistryError: 工具注册表配置错误
        """
        # 参数验证
        if not name or not isinstance(name, str):
            raise ParameterValidationError("智能体名称 'name' 必须是非空字符串!")
        
        if llm is None:
            raise ParameterValidationError("必须提供有效的 LLM 实例!")
        
        if not isinstance(llm, EasyLLM):
            raise ParameterValidationError(f"llm 参数必须是 EasyLLM 类型，收到: {type(llm).__name__}")
        
        super().__init__(
            name=name, 
            llm=llm, 
            system_prompt=system_prompt, 
            description=description, 
            config=config,
            enable_tool=enable_tool,
            tool_registry=tool_registry,
            memory_manage=memory_manage,
            context_manager=context_manager,
            callback_manager=callback_manager,
            skill_manager=skill_manager,
        )
        

        self.verbose_thinking = verbose_thinking
        self.trace_history: list[dict[str, Any]] = []  # 记录完整会话转录
        self._trace_session_id = f"trace_{uuid4().hex}"
        self._trace_event_counter = 0
        self._trace_seq = 0
        self._trace_turn_counter = 0
        self._current_query: str = ""  # 当前查询（供 get_enhanced_prompt 使用）
        self.history_via_context_manager = history_via_context_manager
        self._extension_prompt_blocks: list[PromptBlock] = []

        logger.info(f"BasicAgent '{name}' 初始化完成，工具调用: {'启用' if enable_tool else '禁用'}，provider: {llm.provider_name}")

    def _get_serializable_state(self) -> dict[str, Any]:
        state = super()._get_serializable_state()
        state.update({
            "verbose_thinking": self.verbose_thinking,
            "history_via_context_manager": self.history_via_context_manager,
            "trace_history": self.get_trace_history(),
            "trace_session_id": self._trace_session_id,
            "trace_event_counter": self._trace_event_counter,
            "trace_seq": self._trace_seq,
            "trace_turn_counter": self._trace_turn_counter,
        })
        return state

    def _restore_serializable_state(self, state: Optional[dict[str, Any]]) -> None:
        super()._restore_serializable_state(state)
        state = state or {}
        self.verbose_thinking = state.get("verbose_thinking", False)
        self.history_via_context_manager = state.get("history_via_context_manager", False)
        self.trace_history = list(state.get("trace_history", []))
        self._trace_session_id = state.get("trace_session_id") or f"trace_{uuid4().hex}"
        self._trace_event_counter = int(state.get("trace_event_counter") or 0)
        self._trace_seq = int(state.get("trace_seq") or 0)
        self._trace_turn_counter = int(state.get("trace_turn_counter") or 0)

        legacy_thinking = [str(item) for item in state.get("thinking_history", []) if item is not None]
        if not self.trace_history and legacy_thinking:
            for content in legacy_thinking:
                self.trace_history.append(
                    {
                        "id": self._next_trace_event_id(),
                        "session_id": self._trace_session_id,
                        "turn_id": None,
                        "seq": self._next_trace_seq(),
                        "type": "reasoning",
                        "timestamp": datetime.now().isoformat(),
                        "role": "assistant",
                        "content": content,
                        "metadata": {"mode": "legacy"},
                    }
                )
        self._normalize_trace_history()

    @staticmethod
    def _as_history_entries(message: Any) -> list[Any]:
        """将单条或多条 provider 消息统一展开为 history 条目列表。"""
        if message is None:
            return []
        if isinstance(message, list):
            entries: list[Any] = []
            for item in message:
                entries.extend(BasicAgent._as_history_entries(item))
            return entries
        return [message]

    @staticmethod
    def _trace_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Message):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [BasicAgent._trace_safe(item) for item in value]
        if isinstance(value, tuple):
            return [BasicAgent._trace_safe(item) for item in value]
        if isinstance(value, dict):
            return {key: BasicAgent._trace_safe(item) for key, item in value.items()}
        return BaseAgent._make_json_safe(value)

    def _next_trace_event_id(self) -> str:
        self._trace_event_counter += 1
        return f"evt_{self._trace_event_counter:06d}"

    def _next_trace_seq(self) -> int:
        self._trace_seq += 1
        return self._trace_seq

    def _next_turn_id(self) -> str:
        self._trace_turn_counter += 1
        return f"turn_{self._trace_turn_counter:04d}"

    def _normalize_trace_history(self) -> None:
        normalized: list[dict[str, Any]] = []
        max_seq = self._trace_seq
        max_event_counter = self._trace_event_counter
        max_turn_counter = self._trace_turn_counter

        for event in self.trace_history:
            if not isinstance(event, dict):
                continue
            normalized_event = dict(event)
            event_type = str(normalized_event.get("type") or "unknown")
            if "timestamp" not in normalized_event:
                normalized_event["timestamp"] = normalized_event.pop("time", datetime.now().isoformat())
            if "session_id" not in normalized_event:
                normalized_event["session_id"] = self._trace_session_id
            if "seq" not in normalized_event:
                max_seq += 1
                normalized_event["seq"] = max_seq
            else:
                try:
                    max_seq = max(max_seq, int(normalized_event["seq"]))
                except Exception:
                    max_seq += 1
                    normalized_event["seq"] = max_seq
            if "id" not in normalized_event:
                max_event_counter += 1
                normalized_event["id"] = f"evt_{max_event_counter:06d}"
            else:
                event_id = str(normalized_event["id"])
                if event_id.startswith("evt_"):
                    try:
                        max_event_counter = max(max_event_counter, int(event_id.split("_", 1)[1]))
                    except Exception:
                        pass
            if "turn_id" in normalized_event and normalized_event["turn_id"]:
                turn_id = str(normalized_event["turn_id"])
                if turn_id.startswith("turn_"):
                    try:
                        max_turn_counter = max(max_turn_counter, int(turn_id.split("_", 1)[1]))
                    except Exception:
                        pass
            if "metadata" not in normalized_event or normalized_event["metadata"] is None:
                normalized_event["metadata"] = {}
            if "role" not in normalized_event:
                if event_type in {"reasoning", "thinking", "assistant_message"}:
                    normalized_event["role"] = "assistant"
                elif event_type == "user_message":
                    normalized_event["role"] = "user"
                elif event_type == "tool_result":
                    normalized_event["role"] = "tool"
                else:
                    normalized_event["role"] = "assistant"
            if event_type == "thinking":
                normalized_event["type"] = "reasoning"
            normalized.append(normalized_event)

        normalized.sort(key=lambda item: (int(item.get("seq", 0) or 0), str(item.get("timestamp", ""))))
        self.trace_history = normalized
        self._trace_seq = max_seq
        self._trace_event_counter = max_event_counter
        self._trace_turn_counter = max_turn_counter

    def _record_trace_event(
        self,
        event_type: str,
        *,
        role: str,
        content: str = "",
        turn_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **payload,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "id": self._next_trace_event_id(),
            "session_id": self._trace_session_id,
            "turn_id": turn_id,
            "seq": self._next_trace_seq(),
            "type": event_type,
            "timestamp": timestamp or datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": self._trace_safe(metadata or {}),
        }
        if parent_id is not None:
            event["parent_id"] = parent_id
        for key, value in payload.items():
            if value is None:
                continue
            event[key] = self._trace_safe(value)
        self.trace_history.append(event)
        return event

    def _begin_trace_turn(self, raw_query: str) -> tuple[str, str]:
        turn_id = self._next_turn_id()
        user_event = self._record_trace_event(
            "user_message",
            role="user",
            content=raw_query,
            turn_id=turn_id,
        )
        return turn_id, str(user_event["id"])

    def _get_last_turn_event_id(
        self,
        turn_id: Optional[str],
        *,
        exclude_types: Optional[set[str]] = None,
    ) -> Optional[str]:
        exclude = exclude_types or set()
        for event in reversed(self.trace_history):
            if event.get("turn_id") != turn_id:
                continue
            if event.get("type") in exclude:
                continue
            return str(event.get("id"))
        return None

    def _set_round_reasoning(
        self,
        content: str,
        *,
        turn_id: Optional[str],
        round_number: Optional[int],
        mode: str,
        stream: bool,
    ) -> Optional[str]:
        if not content:
            return None
        for event in reversed(self.trace_history):
            if (
                event.get("type") in {"reasoning", "thinking"}
                and event.get("turn_id") == turn_id
                and event.get("round") == round_number
                and (event.get("metadata") or {}).get("mode") == mode
                and (event.get("metadata") or {}).get("stream") == stream
            ):
                event["content"] = content
                return str(event.get("id"))
        reasoning_event = self._record_trace_event(
            "reasoning",
            role="assistant",
            content=content,
            turn_id=turn_id,
            round=round_number,
            metadata={
                "mode": mode,
                "stream": stream,
                "visibility": "internal",
            },
        )
        return str(reasoning_event["id"])

    def _record_assistant_trace(
        self,
        turn_id: Optional[str],
        content: Optional[str],
        *,
        parent_id: Optional[str] = None,
        stage: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> Optional[str]:
        if not content:
            return None
        event = self._record_trace_event(
            "assistant_message",
            role="assistant",
            content=content,
            turn_id=turn_id,
            parent_id=parent_id,
            round=round_number,
            metadata={
                "stage": stage,
                "mode": mode,
                "stream": stream,
            },
        )
        return str(event["id"])

    def _record_tool_call(
        self,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: Any,
        tool_id: str,
        *,
        parent_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> str:
        event = self._record_trace_event(
            "tool_call",
            role="assistant",
            content="",
            turn_id=turn_id,
            parent_id=parent_id,
            round=round_number,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_id,
            metadata={
                "mode": mode,
                "stream": stream,
            },
        )
        return str(event["id"])

    def _record_tool_result(
        self,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: Any,
        tool_id: str,
        content: Any,
        *,
        parent_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        success: Optional[bool] = None,
    ) -> str:
        event = self._record_trace_event(
            "tool_result",
            role="tool",
            content=str(content),
            turn_id=turn_id,
            parent_id=parent_id,
            round=round_number,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_id,
            metadata={
                "mode": mode,
                "stream": stream,
                "success": success,
            },
        )
        return str(event["id"])

    def _record_turn_end(
        self,
        turn_id: Optional[str],
        *,
        final_event_id: Optional[str] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        status: str = "completed",
    ) -> str:
        event = self._record_trace_event(
            "turn_end",
            role="assistant",
            content="",
            turn_id=turn_id,
            metadata={
                "mode": mode,
                "stream": stream,
                "status": status,
                "final_event_id": final_event_id,
            },
        )
        return str(event["id"])

    @staticmethod
    def _new_stream_display_state() -> dict[str, Any]:
        return {
            "current_round": 0,
            "current_section": None,
            "thinking_text": "",
            "content_text": "",
        }

    @staticmethod
    def _start_stream_round(state: dict[str, Any], round_number: int) -> None:
        if state["current_round"] > 0:
            print()
        print(f"round {round_number}")
        state["current_round"] = round_number
        state["current_section"] = "round"
        state["thinking_text"] = ""
        state["content_text"] = ""
        state["tool_calls"] = ""
    @staticmethod
    def _print_stream_header(state: dict[str, Any], header: str) -> None:
        if state["current_section"] is None:
            print(f"{header}:")
        elif state["current_section"] != header:
            print()
            print(f"{header}:")
        state["current_section"] = header

    @classmethod
    def _append_stream_text(
        cls,
        state: dict[str, Any],
        header: str,
        state_key: str,
        text: str,
    ) -> None:
        if not text:
            return
        cls._print_stream_header(state, header)
        print(text, end="", flush=True)
        state[state_key] += text

    @classmethod
    def _append_stream_snapshot(
        cls,
        state: dict[str, Any],
        header: str,
        state_key: str,
        full_text: str,
    ) -> None:
        if not full_text:
            return
        delta = cls._snapshot_suffix(state[state_key], full_text)
        if not delta:
            return
        cls._append_stream_text(state, header, state_key, delta)

    @staticmethod
    def _snapshot_suffix(displayed: str, full_text: str) -> str:
        # 截取 full_text 中从 displayed 之后开始的后缀
        if not full_text:
            return ""
        if full_text.startswith(displayed):
            return full_text[len(displayed):]
        if displayed:
            return ""
        return full_text

    @classmethod
    def _display_stream_event(
        cls,
        state: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        event_type = event.get("type")
        if event_type == "round_start":
            cls._start_stream_round(state, int(event.get("round", 1) or 1))
            return
        if event_type == "thinking_delta":
            cls._append_stream_text(
                state,
                "thinking content",
                "thinking_text",
                event.get("delta", "") or "",
            )
            return
        if event_type == "text_delta":
            cls._append_stream_text(
                state,
                "content",
                "content_text",
                event.get("delta", "") or "",
            )
            return
        if event_type in {"tool_calls", "final_response", "final"}:
            cls._append_stream_snapshot(
                state,
                "thinking content",
                "thinking_text",
                event.get("thinking", "") or "",
            )
            cls._append_stream_snapshot(
                state,
                "content",
                "content_text",
                event.get("content", "") or "",
            )
        if event_type == "tool_call":
            cls._append_stream_text(
                state,
                "tool_calls",
                "tool_calls",
                f"{event.get('tool_name','')} : {event.get('tool_args','')}\n"
            )
    @staticmethod
    def _print_stream_final(state: dict[str, Any], final_text: str) -> None:
        if state["current_section"] is None:
            print("final res:")
        else:
            print()
            print("final res:")
        print(final_text)
        state["current_section"] = "final res"

    @classmethod
    def _build_constructor_kwargs_from_snapshot(
        cls,
        snapshot: dict[str, Any],
        llm: EasyLLM,
        tool_registry: Optional["ToolRegistry"] = None,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        callback_manager=None,
        skill_manager=None,
    ) -> dict[str, Any]:
        kwargs = super()._build_constructor_kwargs_from_snapshot(
            snapshot,
            llm=llm,
            tool_registry=tool_registry,
            memory_manage=memory_manage,
            context_manager=context_manager,
            callback_manager=callback_manager,
            skill_manager=skill_manager,
        )
        state = snapshot.get("state") or {}
        kwargs.update(
            {
                "history_via_context_manager": state.get("history_via_context_manager", False),
            }
        )
        return kwargs


    # @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) :
        """
        调用智能体
        
        Args:
            query: 用户输入
            max_iter: 最大迭代次数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            智能体返回结果
            
        Raises:
            ParameterValidationError: 参数验证失败
            LLMInvokeError: LLM 调用失败
        """
        if self.verbose_thinking:
            kwargs["reasoning"] = {"effort": "medium","summary": "auto"}
        # 参数验证
        self._validate_invoke_params(query, max_iter, temperature)
        original_query = query
        self._current_query = query  # 供 get_enhanced_prompt 使用
        self._clear_ephemeral_skill_state()
        
        # Skill 前置拦截
        query = self.skill_manager.on_before_invoke(query)
        
        self.callback_manager.on_agent_start(self.name, query)
        
        messages: list[Message | dict[str, str]] = []
 
        if self.enable_tool :
            logger.info("使用工具模式调用智能体")
            try:
                result = self.invoke_with_tool(
                    query,
                    messages,
                    max_iter,
                    temperature,
                    trace_query=original_query,
                    **kwargs,
                )
                self.callback_manager.on_agent_end(self.name, result, success=True)
                return result
            except Exception as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            finally:
                self._clear_ephemeral_skill_state()
        else:
            logger.info("使用普通模式调用智能体")
            try:
                messages = self._build_start_messages(query)
                turn_id, last_trace_event_id = self._begin_trace_turn(original_query)
                
                self._capture_context_usage(messages, label="invoke_plain")
                self.callback_manager.on_llm_start(messages)
                response = self.llm.invoke(messages, temperature=temperature, **kwargs)
                self.callback_manager.on_llm_end(response or "")
                
                # 验证响应
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                if not isinstance(response, str):
                    logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                    response = str(response)
                
                # Skill 后置拦截
                response = self.skill_manager.on_after_invoke(query, response)
                self.add_message(UserMessage(query))
                self.add_message(AssistantMessage(response))
                final_event_id = self._record_assistant_trace(
                    turn_id,
                    response,
                    parent_id=last_trace_event_id,
                    stage="final",
                    round_number=1,
                    mode="plain",
                    stream=False,
                ) or last_trace_event_id
                self._record_turn_end(
                    turn_id,
                    final_event_id=final_event_id,
                    mode="plain",
                    stream=False,
                )
                self.callback_manager.on_agent_end(self.name, response, success=True)
                return response
                
            except LLMInvokeError as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise LLMInvokeError(f"LLM 调用失败: {e}") from e
            finally:
                self._clear_ephemeral_skill_state()

    def stream_invoke(self,query: str,temperature: float = 0.7, **kwargs):
        original_query = query
        self._current_query = query
        self._clear_ephemeral_skill_state()
        if self.enable_tool:
            logger.info("使用工具模式流式调用智能体")
            display_state = self._new_stream_display_state()
            final_result = ""
            try:
                for event in self.stream_invoke_with_tool(query, temperature=temperature, trace_query=original_query, **kwargs):
                    self._display_stream_event(display_state, event)
                    if event["type"] == "final":
                        final_result = event["content"]
                        self._print_stream_final(display_state, final_result)
                return final_result
            finally:
                self._clear_ephemeral_skill_state()
        else:
            self._validate_invoke_params(query, 1, temperature)
            original_query = query
            query = self.skill_manager.on_before_invoke(query)
            self.callback_manager.on_agent_start(self.name, query)
            messages = self._build_start_messages(query)
            final_results=[]
            display_state = self._new_stream_display_state()
            try:
                turn_id, last_trace_event_id = self._begin_trace_turn(original_query)
                self._capture_context_usage(messages, label="stream_invoke_plain")
                self.callback_manager.on_llm_start(messages)
                for chunk in self.llm.stream(messages, temperature=temperature, **kwargs):
                    self._append_stream_text(display_state, "content", "content_text", chunk)
                    final_results.append(chunk)
                result = "".join(final_results)
                self.callback_manager.on_llm_end(result)
                result = self.skill_manager.on_after_invoke(query, result)
                self.add_message(UserMessage(query))
                self.add_message(AssistantMessage(result))
                final_event_id = self._record_assistant_trace(
                    turn_id,
                    result,
                    parent_id=last_trace_event_id,
                    stage="final",
                    round_number=1,
                    mode="plain",
                    stream=True,
                ) or last_trace_event_id
                self._record_turn_end(
                    turn_id,
                    final_event_id=final_event_id,
                    mode="plain",
                    stream=True,
                )
                self.callback_manager.on_agent_end(self.name, result, success=True)
                self._print_stream_final(display_state, result)
                return result
            except Exception as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            finally:
                self._clear_ephemeral_skill_state()

    def stream_invoke_with_tool(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs
        ):
        """同步流式工具调用，逐步产出统一事件。"""
        async_gen = self.astream_invoke_with_tool(
            query,
            max_iter=max_iter,
            temperature=temperature,
            **kwargs,
        )
        loop = asyncio.new_event_loop()
        previous_loop = None
        try:
            try:
                previous_loop = asyncio.get_event_loop_policy().get_event_loop()
            except RuntimeError:
                previous_loop = None
            asyncio.set_event_loop(loop)
            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
                yield event
        finally:
            try:
                loop.run_until_complete(async_gen.aclose())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            asyncio.set_event_loop(previous_loop)
            loop.close()

    def invoke_with_tool(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        使用工具调用模式调用智能体（同步版本）
        
        Args:
            query: 用户输入
            messages: 消息列表
            max_iter: 最大迭代次数
            temperature: 温度参数
            
        Returns:
            智能体返回结果
        """
        self.enable_tool = True
        self._clear_ephemeral_skill_state()
        
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        raw_query = trace_query if trace_query is not None else query
        messages = self._build_start_messages(query)
        
        final_response: Optional[str] = None
        turn_history: list[Any] = [UserMessage(query)]
        turn_id, turn_root_event_id = self._begin_trace_turn(raw_query)
        iteration_count = 0
        
        try:
            while max_iter > 0:
                iteration_count += 1
                logger.debug(f"工具调用迭代 {iteration_count}")
                
                try:
                    self._capture_context_usage(messages, label="invoke_tool")
                    self.callback_manager.on_llm_start(messages)
                    response = self.llm.invoke_with_tools(
                        messages,
                        self.tool_registry.get_openai_tools(),
                        temperature=temperature,
                        **kwargs
                    )
                    self.callback_manager.on_llm_end(getattr(response, 'content', '') or '')
                    
                    if response is None:
                        raise LLMInvokeError("LLM 返回了空响应!")

                    _output = getattr(response, "output", None)
                    if _output is not None:
                        _types = [getattr(item, "type", repr(item)) for item in _output]
                        logger.info(f"📦 response.output 类型列表: {_types}")
                except LLMInvokeError:
                    raise
                except Exception as e:
                    logger.error(f"智能体调用失败: {e}")
                    final_response = f"智能体调用失败: {str(e)}"
                    break

                thinking_content = self.llm.get_thinking_content(response)
                reasoning_event_id: Optional[str] = None
                if thinking_content:
                    reasoning_event_id = self._set_round_reasoning(
                        thinking_content,
                        turn_id=turn_id,
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                    )
                    if self.verbose_thinking:
                        logger.info(f"💭 模型思考: {thinking_content}")
                    # messages.append(AssistantMessage(thinking_content))

                if self.llm.has_tool_calls(response):
                    formatted_response = self.llm.format_assistant_response(response)
                    if isinstance(formatted_response, list):
                        messages.extend(formatted_response)
                    else:
                        messages.append(formatted_response)
                    turn_history.extend(self._as_history_entries(formatted_response))
                    assistant_parent_id = reasoning_event_id or turn_root_event_id
                    assistant_event_id = self._record_assistant_trace(
                        turn_id,
                        self.llm.get_response_content(response),
                        parent_id=assistant_parent_id,
                        stage="pre_tool",
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                    )

                    for tool_call in self.llm.get_tool_calls(response):
                        tool_name = "unknown_tool"
                        tool_args: dict[str, Any] = {}
                        tool_id = (
                            getattr(tool_call, 'call_id', None)
                            or getattr(tool_call, 'id', 'unknown')
                        )

                        try:
                            tool_name = self._safe_get_tool_name(tool_call)
                        except Exception as e:
                            logger.warning(f"获取工具名称失败: {e}，使用默认名称")
                            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                                tool_name = tool_call.function.name or "unknown_tool"
                            elif hasattr(tool_call, 'name'):
                                tool_name = tool_call.name or "unknown_tool"

                        try:
                            tool_args = self._safe_parse_tool_args(tool_call)
                            logger.info(f"{self.name}执行工具: {tool_name}，参数: {tool_args}")
                            tool_call_event_id = self._record_tool_call(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                            )
                            tool_result = self._safe_execute_tool(tool_name, tool_args)
                            self._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                tool_result,
                                parent_id=tool_call_event_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=True,
                            )
                            tool_msg = self.llm.format_tool_result(tool_result, tool_id, tool_name)
                            messages.append(tool_msg)
                            turn_history.append(tool_msg)
                            self._maybe_inject_runtime_skill_context(tool_name=tool_name, messages=messages)

                        except ToolExecutionError as e:
                            logger.error(f"工具 '{tool_name}' 执行失败: {e}")
                            error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                            self._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                error_msg,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=False,
                            )
                            tool_msg = self.llm.format_tool_result(error_msg, tool_id, tool_name)
                            messages.append(tool_msg)
                            turn_history.append(tool_msg)
                        except Exception as e:
                            logger.error(f"处理工具 '{tool_name}' 调用时发生未知错误: {e}")
                            error_msg = f"工具 '{tool_name}' 处理失败: {str(e)}"
                            self._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                error_msg,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=False,
                            )
                            tool_msg = self.llm.format_tool_result(error_msg, tool_id, tool_name)
                            messages.append(tool_msg)
                            turn_history.append(tool_msg)
                else:
                    content = self.llm.get_response_content(response) or getattr(response, 'content', None)
                    if content is not None:
                        final_response = content
                    else:
                        logger.warning("LLM 响应中没有内容")
                        final_response = ""
                    break
                
                max_iter -= 1
            
            if final_response is None:
                logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
                final_response = "超过最大迭代次数，智能体调用失败!"
            
            final_response = self.skill_manager.on_after_invoke(query, final_response)
            turn_history.append(AssistantMessage(final_response))
            self.add_messages(turn_history)
            final_event_id = self._record_assistant_trace(
                turn_id,
                final_response,
                parent_id=self._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                stage="final",
                round_number=iteration_count or 1,
                mode="tool",
                stream=False,
            )
            self._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="tool",
                stream=False,
            )
            return final_response
        finally:
            self._clear_ephemeral_skill_state()

    async def ainvoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """
        原生异步调用智能体
        
        Args:
            query: 用户输入
            max_iter: 最大迭代次数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            智能体返回结果
        """
        if self.verbose_thinking:
            kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}
        self._validate_invoke_params(query, max_iter, temperature)
        original_query = query
        self._current_query = query
        self._clear_ephemeral_skill_state()
        
        # Skill 前置拦截
        query = self.skill_manager.on_before_invoke(query)
        
        self.callback_manager.on_agent_start(self.name, query)
        
        if self.enable_tool:
            logger.info("使用异步工具模式调用智能体")
            try:
                result = await self.ainvoke_with_tool(
                    query,
                    [],
                    max_iter,
                    temperature,
                    trace_query=original_query,
                    **kwargs,
                )
                self.callback_manager.on_agent_end(self.name, result, success=True)
                return result
            except Exception as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            finally:
                self._clear_ephemeral_skill_state()
        else:
            logger.info("使用异步普通模式调用智能体")
            try:
                messages = self._build_start_messages(query)
                turn_id, last_trace_event_id = self._begin_trace_turn(original_query)
                
                self._capture_context_usage(messages, label="ainvoke_plain")
                self.callback_manager.on_llm_start(messages)
                response = await self.llm.ainvoke(messages, temperature=temperature, **kwargs)
                self.callback_manager.on_llm_end(response or "")
                
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                if not isinstance(response, str):
                    logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                    response = str(response)
                
                response = self.skill_manager.on_after_invoke(query, response)
                self.add_message(UserMessage(query))
                self.add_message(AssistantMessage(response))
                final_event_id = self._record_assistant_trace(
                    turn_id,
                    response,
                    parent_id=last_trace_event_id,
                    stage="final",
                    round_number=1,
                    mode="plain",
                    stream=False,
                ) or last_trace_event_id
                self._record_turn_end(
                    turn_id,
                    final_event_id=final_event_id,
                    mode="plain",
                    stream=False,
                )
                self.callback_manager.on_agent_end(self.name, response, success=True)
                return response
                
            except LLMInvokeError as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            except Exception as e:
                logger.error(f"LLM 异步调用失败: {e}")
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise LLMInvokeError(f"LLM 异步调用失败: {e}") from e
            finally:
                self._clear_ephemeral_skill_state()

    async def ainvoke_with_tool(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        原生异步工具调用模式
        
        与 invoke_with_tool 对称，但 LLM 调用和工具执行均为异步。
        """
        self.enable_tool = True
        self._clear_ephemeral_skill_state()
        
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        raw_query = trace_query if trace_query is not None else query
        messages = self._build_start_messages(query)
        
        final_response: Optional[str] = None
        turn_history: list[Any] = [UserMessage(query)]
        turn_id, turn_root_event_id = self._begin_trace_turn(raw_query)
        iteration_count = 0
        
        try:
            while max_iter > 0:
                iteration_count += 1
                logger.debug(f"异步工具调用迭代 {iteration_count}")
                
                try:
                    self._capture_context_usage(messages, label="ainvoke_tool")
                    self.callback_manager.on_llm_start(messages)
                    response = await self.llm.ainvoke_with_tools(
                        messages,
                        self.tool_registry.get_openai_tools(),
                        temperature=temperature,
                        **kwargs
                    )
                    self.callback_manager.on_llm_end(getattr(response, 'content', '') or '')
                    
                    if response is None:
                        raise LLMInvokeError("LLM 返回了空响应!")

                except LLMInvokeError:
                    raise
                except Exception as e:
                    logger.error(f"异步智能体调用失败: {e}")
                    final_response = f"智能体调用失败: {str(e)}"
                    break

                thinking_content = self.llm.get_thinking_content(response)
                reasoning_event_id: Optional[str] = None
                if thinking_content:
                    reasoning_event_id = self._set_round_reasoning(
                        thinking_content,
                        turn_id=turn_id,
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                    )
                    if self.verbose_thinking:
                        logger.info(f"💭 模型思考: {thinking_content}")

                if self.llm.has_tool_calls(response):
                    formatted_response = self.llm.format_assistant_response(response)
                    tool_calls = self.llm.get_tool_calls(response)
                    if isinstance(formatted_response, list):
                        messages.extend(formatted_response)
                    else:
                        messages.append(formatted_response)
                    turn_history.extend(self._as_history_entries(formatted_response))
                    assistant_parent_id = reasoning_event_id or turn_root_event_id
                    assistant_event_id = self._record_assistant_trace(
                        turn_id,
                        self.llm.get_response_content(response),
                        parent_id=assistant_parent_id,
                        stage="pre_tool",
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                    )

                    async def _process_single_tool(tool_call) -> Message | dict:
                        tool_name = "unknown_tool"
                        tool_args: dict[str, Any] = {}
                        tool_id = (
                            getattr(tool_call, 'call_id', None)
                            or getattr(tool_call, 'id', 'unknown')
                        )

                        try:
                            tool_name = self._safe_get_tool_name(tool_call)
                        except Exception as e:
                            logger.warning(f"获取工具名称失败: {e}，使用默认名称")
                            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                                tool_name = tool_call.function.name or "unknown_tool"
                            elif hasattr(tool_call, 'name'):
                                tool_name = tool_call.name or "unknown_tool"

                        try:
                            tool_args = self._safe_parse_tool_args(tool_call)
                            logger.info(f"{self.name} 并发异步执行工具: {tool_name}，参数: {tool_args}")
                            tool_call_event_id = self._record_tool_call(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                            )
                            tool_result = await self._async_safe_execute_tool(tool_name, tool_args)
                            self._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                tool_result,
                                parent_id=tool_call_event_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=True,
                            )
                            return self.llm.format_tool_result(tool_result, tool_id, tool_name)

                        except ToolExecutionError as e:
                            logger.error(f"工具 '{tool_name}' 执行失败: {e}")
                            error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                            self._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                error_msg,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=False,
                            )
                            return self.llm.format_tool_result(error_msg, tool_id, tool_name)
                        except Exception as e:
                            logger.error(f"处理工具 '{tool_name}' 调用时发生未知错误: {e}")
                            error_msg = f"工具 '{tool_name}' 处理失败: {str(e)}"
                            self._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                error_msg,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=False,
                            )
                            return self.llm.format_tool_result(error_msg, tool_id, tool_name)

                    tasks = [
                        _process_single_tool(tc)
                        for tc in tool_calls
                    ]
                    tool_msgs = await asyncio.gather(*tasks)
                    messages.extend(tool_msgs)
                    turn_history.extend(tool_msgs)
                    if any(self._safe_get_tool_name(tc) == "skill_tool" for tc in tool_calls):
                        self._maybe_inject_runtime_skill_context(tool_name="skill_tool", messages=messages)

                else:
                    content = self.llm.get_response_content(response) or getattr(response, 'content', None)
                    if content is not None:
                        final_response = content
                    else:
                        logger.warning("LLM 响应中没有内容")
                        final_response = ""
                    break
                
                max_iter -= 1
            
            if final_response is None:
                logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
                final_response = "超过最大迭代次数，智能体调用失败!"
            
            final_response = self.skill_manager.on_after_invoke(query, final_response)
            turn_history.append(AssistantMessage(final_response))
            self.add_messages(turn_history)
            final_event_id = self._record_assistant_trace(
                turn_id,
                final_response,
                parent_id=self._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                stage="final",
                round_number=iteration_count or 1,
                mode="tool",
                stream=False,
            )
            self._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="tool",
                stream=False,
            )
            return final_response
        finally:
            self._clear_ephemeral_skill_state()

    async def astream_invoke(self, query: str, temperature: float = 0.7, **kwargs) -> str:
        """
        异步流式调用智能体
        
        Args:
            query: 用户输入
            temperature: 温度参数
            
        Returns:
            完整响应文本
        """
        original_query = query
        self._current_query = query
        self._clear_ephemeral_skill_state()
        if self.enable_tool:
            display_state = self._new_stream_display_state()
            final_result = ""
            try:
                async for event in self.astream_invoke_with_tool(query, temperature=temperature, trace_query=original_query, **kwargs):
                    self._display_stream_event(display_state, event)
                    if event["type"] == "final":
                        final_result = event["content"]
                        self._print_stream_final(display_state, final_result)
                return final_result
            finally:
                self._clear_ephemeral_skill_state()
        
        self._validate_invoke_params(query, 1, temperature)
        original_query = query
        query = self.skill_manager.on_before_invoke(query)
        self.callback_manager.on_agent_start(self.name, query)
        messages = self._build_start_messages(query)
        final_results = []
        display_state = self._new_stream_display_state()
        try:
            turn_id, last_trace_event_id = self._begin_trace_turn(original_query)
            self._capture_context_usage(messages, label="astream_invoke_plain")
            self.callback_manager.on_llm_start(messages)
            async for chunk in self.llm.astream(messages, temperature=temperature, **kwargs):
                self._append_stream_text(display_state, "content", "content_text", chunk)
                final_results.append(chunk)
            
            result = "".join(final_results)
            self.callback_manager.on_llm_end(result)
            result = self.skill_manager.on_after_invoke(query, result)
            self.add_message(UserMessage(query))
            self.add_message(AssistantMessage(result))
            final_event_id = self._record_assistant_trace(
                turn_id,
                result,
                parent_id=last_trace_event_id,
                stage="final",
                round_number=1,
                mode="plain",
                stream=True,
            ) or last_trace_event_id
            self._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="plain",
                stream=True,
            )
            self.callback_manager.on_agent_end(self.name, result, success=True)
            self._print_stream_final(display_state, result)
            return result
        except Exception as e:
            self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
            raise
        finally:
            self._clear_ephemeral_skill_state()

    async def astream_invoke_with_tool(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式工具调用，逐步产出文本、工具与最终结果事件。"""
        if self.verbose_thinking:
            kwargs.setdefault("reasoning", {"effort": "medium", "summary": "auto"})
        self._validate_invoke_params(query, max_iter, temperature)
        raw_query = trace_query if trace_query is not None else query
        self._current_query = query
        self._clear_ephemeral_skill_state()
        query = self.skill_manager.on_before_invoke(query)
        self.callback_manager.on_agent_start(self.name, query)

        if self.tool_registry is None:
            error = ToolRegistryError("工具调用需要提供 ToolRegistry!")
            self.callback_manager.on_agent_end(self.name, "", success=False, error=error)
            raise error

        messages = self._build_start_messages(query)
        final_response = ""
        turn_history: list[Any] = [UserMessage(query)]
        turn_id, turn_root_event_id = self._begin_trace_turn(raw_query)
        round_index = 0

        try:
            while max_iter > 0:
                round_index += 1

                self._capture_context_usage(messages, label="astream_invoke_tool")
                self.callback_manager.on_llm_start(messages)
                llm_stream = self.llm.astream_with_tools(
                    messages,
                    self.tool_registry.get_openai_tools(),
                    temperature=temperature,
                    **kwargs
                )
                should_continue = False
                streamed_thinking = ""
                streamed_content = ""
                try:
                    yield {
                        "type": "round_start",
                        "round": round_index,
                    }
                    async for event in llm_stream:
                        event_type = event.get("type")

                        if event_type == "stream_end":
                            continue

                        if event_type == "text_delta":
                            streamed_content += event.get("delta", "") or ""
                            yield event
                            continue

                        if event_type == "thinking_delta":
                            delta = event.get("delta", "")
                            streamed_thinking += delta
                            self._set_round_reasoning(
                                streamed_thinking,
                                turn_id=turn_id,
                                round_number=round_index,
                                mode="tool",
                                stream=True,
                            )
                            if self.verbose_thinking:
                                yield event
                            continue

                        if event_type == "tool_calls":
                            thinking_suffix = self._snapshot_suffix(
                                streamed_thinking,
                                event.get("thinking", "") or "",
                            )
                            if thinking_suffix:
                                streamed_thinking += thinking_suffix
                                self._set_round_reasoning(
                                    streamed_thinking,
                                    turn_id=turn_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                                if self.verbose_thinking:
                                    yield {
                                        "type": "thinking_delta",
                                        "delta": thinking_suffix,
                                    }

                            content_suffix = self._snapshot_suffix(
                                streamed_content,
                                event.get("content", "") or "",
                            )
                            if content_suffix:
                                streamed_content += content_suffix
                                yield {
                                    "type": "text_delta",
                                    "delta": content_suffix,
                                }

                            assistant_items = event.get("assistant_items")
                            if assistant_items:
                                assistant_message = assistant_items
                            else:
                                assistant_message = self.llm.format_assistant_message(
                                    content=event.get("content"),
                                    tool_calls=event.get("tool_calls"),
                                )
                            if isinstance(assistant_message, list):
                                messages.extend(assistant_message)
                            else:
                                messages.append(assistant_message)
                            turn_history_entries = self._as_history_entries(assistant_message)
                            turn_history.extend(turn_history_entries)

                            self.callback_manager.on_llm_end(event.get("content", "") or "")
                            reasoning_event_id = None
                            if streamed_thinking:
                                reasoning_event_id = self._set_round_reasoning(
                                    streamed_thinking,
                                    turn_id=turn_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                            assistant_parent_id = reasoning_event_id or turn_root_event_id
                            assistant_event_id = self._record_assistant_trace(
                                turn_id,
                                event.get("content"),
                                parent_id=assistant_parent_id,
                                stage="pre_tool",
                                round_number=round_index,
                                mode="tool",
                                stream=True,
                            )

                            for tool_call in event.get("tool_calls", []):
                                tool_name = self._safe_get_tool_name(tool_call)
                                tool_args = self._safe_parse_tool_args(tool_call)
                                tool_id = tool_call.get("id", "unknown") if isinstance(tool_call, dict) else (
                                    getattr(tool_call, "call_id", None)
                                    or getattr(tool_call, "id", "unknown")
                                )
                                yield {
                                    "type": "tool_call",
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "tool_id": tool_id,
                                }
                                tool_call_event_id = self._record_tool_call(
                                    turn_id,
                                    tool_name,
                                    tool_args,
                                    tool_id,
                                    parent_id=assistant_event_id or assistant_parent_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                                try:
                                    tool_result = await self._async_safe_execute_tool(tool_name, tool_args)
                                    tool_success = True
                                except Exception as e:
                                    logger.error(f"流式工具 '{tool_name}' 执行失败: {e}")
                                    tool_result = f"工具 '{tool_name}' 执行失败: {e}"
                                    tool_success = False
                                yield {
                                    "type": "tool_result",
                                    "tool_name": tool_name,
                                    "tool_id": tool_id,
                                    "tool_args": tool_args,
                                    "content": tool_result,
                                }
                                self._record_tool_result(
                                    turn_id,
                                    tool_name,
                                    tool_args,
                                    tool_id,
                                    tool_result,
                                    parent_id=tool_call_event_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                    success=tool_success,
                                )
                                tool_message = self.llm.format_tool_result(tool_result, tool_id, tool_name)
                                messages.append(tool_message)
                                turn_history.append(tool_message)
                                self._maybe_inject_runtime_skill_context(tool_name=tool_name, messages=messages)

                            max_iter -= 1
                            should_continue = True
                            break

                        if event_type == "final_response":
                            final_response = event.get("content", "") or ""
                            self.callback_manager.on_llm_end(final_response)
                            if event.get("thinking"):
                                self._set_round_reasoning(
                                    event.get("thinking", "") or "",
                                    turn_id=turn_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                            final_response = self.skill_manager.on_after_invoke(query, final_response)
                            turn_history.append(AssistantMessage(final_response))
                            self.add_messages(turn_history)
                            final_event_id = self._record_assistant_trace(
                                turn_id,
                                final_response,
                                parent_id=self._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                                stage="final",
                                round_number=round_index,
                                mode="tool",
                                stream=True,
                            )
                            self._record_turn_end(
                                turn_id,
                                final_event_id=final_event_id,
                                mode="tool",
                                stream=True,
                            )
                            self.callback_manager.on_agent_end(self.name, final_response, success=True)
                            yield {
                                "type": "final",
                                "content": final_response,
                                "thinking": event.get("thinking", ""),
                            }
                            return
                finally:
                    await llm_stream.aclose()

                if should_continue:
                    continue
                break

            if not final_response:
                final_response = "超过最大迭代次数，智能体调用失败!"
            final_response = self.skill_manager.on_after_invoke(query, final_response)
            turn_history.append(AssistantMessage(final_response))
            self.add_messages(turn_history)
            final_event_id = self._record_assistant_trace(
                turn_id,
                final_response,
                parent_id=self._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                stage="final",
                round_number=round_index or 1,
                mode="tool",
                stream=True,
            )
            self._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="tool",
                stream=True,
            )
            self.callback_manager.on_agent_end(self.name, final_response, success=True)
            yield {
                "type": "final",
                "content": final_response,
                "thinking": "",
            }
        except Exception as e:
            self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
            yield {
                "type": "error",
                "error": str(e),
            }
            raise
        finally:
            self._clear_ephemeral_skill_state()

 
    @override
    def get_enhanced_prompt(self) -> str:
        """
        获取增强后的系统提示词。
        
        Returns:
            增强后的系统提示词
        """
        return self.get_system_prompt_template().render()

    def get_system_prompt_template(self) -> SystemPromptTemplate:
        """返回当前 Agent 的系统提示词模板。"""
        return SystemPromptTemplate(self.get_system_prompt_blocks())

    def get_system_prompt_blocks(self) -> list[PromptBlock]:
        """返回当前 Agent 的系统提示词分块。"""
        if not self.enable_tool or not self.tool_registry:
            blocks = [
                PromptBlock(
                    name="identity",
                    content=self.system_prompt or "你是一个有用的 AI 助手，帮助用户回答问题并完成任务。",
                    order=0,
                )
            ]
            blocks.extend(self._build_core_prompt_blocks(start_order=10, include_tool_policy=False))
            blocks.extend(self._build_shared_prompt_blocks(start_order=100, include_custom_prompt=False))
            return blocks

        blocks = [
            PromptBlock(
                name="identity",
                content="你是一个智能助手，具备使用工具解决问题的能力。",
                order=0,
            ),
        ]
        blocks.extend(self._build_core_prompt_blocks(start_order=10, include_tool_policy=True))
        tool_block = self._build_tool_inventory_block(order=60)
        if tool_block is not None:
            blocks.append(tool_block)
        blocks.extend(self._build_shared_prompt_blocks(start_order=100))
        return blocks

    def _build_core_prompt_blocks(
        self,
        *,
        start_order: int,
        include_tool_policy: bool,
    ) -> list[PromptBlock]:
        """构建接近 Claude Code 风格的核心系统分块。"""
        blocks = [
            PromptBlock(name="visibility", content=build_visibility_section(), order=start_order),
            PromptBlock(name="task_execution", content=build_task_execution_section(), order=start_order + 10),
            PromptBlock(name="safety", content=build_safety_section(), order=start_order + 20),
        ]
        if include_tool_policy:
            blocks.append(
                PromptBlock(
                    name="tool_policy",
                    content=build_tool_policy_section(),
                    order=start_order + 30,
                )
            )
            style_order = start_order + 40
        else:
            style_order = start_order + 30

        blocks.extend(
            [
                PromptBlock(name="tone_style", content=build_tone_style_section(), order=style_order),
                PromptBlock(
                    name="output_efficiency",
                    content=build_output_efficiency_section(),
                    order=style_order + 10,
                ),
            ]
        )
        return blocks

    def _get_tool_catalog_prompt(self) -> str:
        """获取格式化后的工具清单。"""
        if not self.tool_registry:
            return ""
        try:
            tool_descriptions = self.tool_registry.get_tools_description()
        except Exception as e:
            logger.error(f"获取工具描述失败: {e}")
            return "（工具描述获取失败）"
        return format_tool_catalog(tool_descriptions)

    def _get_tool_inventory_prompt(self, *, include_parameters: bool = False) -> str:
        """获取工具简表或详细清单。"""
        if not self.tool_registry:
            return ""
        try:
            tool_descriptions = self.tool_registry.get_tools_description()
        except Exception as e:
            logger.error(f"获取工具描述失败: {e}")
            return "（工具描述获取失败）"
        return format_tool_inventory(tool_descriptions, include_parameters=include_parameters)

    def _should_include_tool_inventory_block(self) -> bool:
        """普通 Function Calling Agent 默认不在 system prompt 注入工具清单。"""
        return False

    def _tool_inventory_mode(self) -> str:
        """控制工具清单模式。支持 none/compact/full。"""
        return "none"

    def _build_tool_inventory_block(self, order: int) -> PromptBlock | None:
        """根据配置构建可选工具清单块。"""
        if not self._should_include_tool_inventory_block():
            return None

        mode = self._tool_inventory_mode()
        if mode == "none":
            return None

        content = self._get_tool_inventory_prompt(include_parameters=(mode == "full"))
        if not content:
            return None

        title = "## 可用工具"
        if mode == "compact":
            title = "## 可用工具概览"

        return PromptBlock(
            name="tool_inventory",
            content=f"{title}\n{content}",
            order=order,
        )

    def _build_shared_prompt_blocks(
        self,
        *,
        start_order: int,
        include_custom_prompt: bool = True,
        include_memory: bool = True,
        include_skills: bool = True,
    ) -> list[PromptBlock]:
        """构建跨 Agent 共用的系统提示词分块。"""
        blocks: list[PromptBlock] = []
        order = start_order

        if include_custom_prompt and self.system_prompt and self.system_prompt.strip():
            blocks.append(
                PromptBlock(
                    name="custom_instructions",
                    content=f"## 额外指令\n{self.system_prompt.strip()}",
                    order=order,
                )
            )
            order += 10

        skill_policy_prompt = self.skill_manager.build_skill_policy_prompt()
        if skill_policy_prompt:
            blocks.append(
                PromptBlock(
                    name="skill_policy",
                    content=skill_policy_prompt,
                    order=order,
                )
            )
            order += 10

        skill_listing_prompt = self.skill_manager.build_skill_listing_prompt()
        if skill_listing_prompt:
            blocks.append(
                PromptBlock(
                    name="skill_listing",
                    content=skill_listing_prompt,
                    order=order,
                )
            )
            order += 10

        memory_prompt = self._build_memory_prompt() if include_memory else ""
        if memory_prompt:
            blocks.append(
                PromptBlock(
                    name="memory",
                    content=memory_prompt,
                    order=order,
                )
            )
            order += 10

        exclude_names = {"memory"} if memory_prompt else None
        if include_skills:
            skills_prompt = self._build_skills_prompt(exclude_names=exclude_names)
            if skills_prompt:
                blocks.append(
                    PromptBlock(
                        name="skills",
                        content=skills_prompt,
                        order=order,
                    )
                )
                order += 10

        blocks.extend(self._get_extension_prompt_blocks(start_order=order))

        return blocks

    def _append_runtime_skill_context_message(self, messages: list[Any]) -> None:
        """向当前推理链追加临时 Skill 正文上下文，但不写入长期 history。"""
        runtime_prompt = self.skill_manager.build_runtime_skill_context_prompt()
        if not runtime_prompt:
            return
        messages.append(
            MetaUserMessage(
                runtime_prompt,
                metadata={
                    "source": "skill_tool",
                    "kind": "runtime_skill_context",
                },
            )
        )

    def _clear_ephemeral_skill_state(self) -> None:
        """清理当前轮的临时 Skill 状态。"""
        self.skill_manager.clear_ephemeral_state()

    def _maybe_inject_runtime_skill_context(
        self,
        *,
        tool_name: str,
        messages: list[Any],
    ) -> None:
        """当 skill_tool 被调用后，将临时 Skill 正文上下文注入当前推理链。"""
        if tool_name != "skill_tool":
            return
        if not self.skill_manager.has_runtime_skill_context():
            return
        self._append_runtime_skill_context_message(messages)

    def _get_extension_prompt_blocks(self, start_order: int) -> list[PromptBlock]:
        """
        预留给 Claude Code 专属 section 或 provider-specific section 的扩展接口。

        默认不启用任何额外块，后续可以由子类覆写，或外部通过组合方式注入。
        """
        blocks = list(self._extension_prompt_blocks)
        normalized: list[PromptBlock] = []
        for index, block in enumerate(blocks):
            normalized.append(
                PromptBlock(
                    name=block.name,
                    content=block.content,
                    order=start_order + index * 10 if block.order == 0 else block.order,
                    enabled=block.enabled,
                    metadata=dict(block.metadata),
                )
            )
        return normalized

    def with_prompt_block(self, block: PromptBlock) -> "BasicAgent":
        """注册一个额外的系统提示词分块。"""
        self._extension_prompt_blocks.append(block)
        return self

    def with_prompt_blocks(self, blocks: list[PromptBlock]) -> "BasicAgent":
        """批量注册额外的系统提示词分块。"""
        self._extension_prompt_blocks.extend(blocks)
        return self

    def _use_context_history(self) -> bool:
        """是否由 ContextManager 管理 history 注入"""
        return bool(self.history_via_context_manager and self.context_manager)

    def _context_include_history(self) -> bool:
        """给 ContextManager 的 include_history 开关"""
        return self._use_context_history()

    def _append_runtime_history(self, messages: list[Message | dict[str, str]]) -> None:
        """按当前模式追加 history 到消息序列。"""
        if self._use_context_history():
            return
        for message in self.history:    
            messages.append(message)

    def _build_start_messages(self, query: str) -> list[Any]:
        """构建发送给 LLM 的起始消息。

        当配置了 ContextManager 时，起始 messages 由 ContextManager 统一管理和构造：
        - history 保持多轮对话结构
        - 非 history 来源聚合到单条 system 消息
        """
        system_prompt = self.get_enhanced_prompt()

        if self.context_manager is not None:
            try:
                messages = self.context_manager.build_messages(
                    query=query,
                    history=self.history,
                    system_prompt=system_prompt,
                    include_history=True,
                    include_query=True,
                )
                if self.context_manager.last_history_was_compacted:
                    self.history = list(self.context_manager.last_compacted_history)
                return messages
            except Exception as e:
                logger.warning(f"ContextManager 构建 messages 失败，回退默认拼接: {e}")

        messages: list[Any] = [SystemMessage(system_prompt)]
        self._append_runtime_history(messages)
        messages.append(UserMessage(query))
        return messages

        

    def get_trace_history(self) -> list[dict[str, Any]]:
        """获取完整会话转录。"""
        return self._make_json_safe(self.trace_history)

    def clear_trace_history(self) -> None:
        """清空完整会话转录。"""
        self.trace_history.clear()
        self._trace_event_counter = 0
        self._trace_seq = 0
        self._trace_turn_counter = 0
