from typing_extensions import override
from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage, MetaUserMessage
from core.Config import Config
from typing import Optional, Any, AsyncGenerator, TYPE_CHECKING
from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from .components.history_message_assembler import (
    BaseHistoryMessageAssembler,
    DefaultHistoryMessageAssembler,
)
from .components.invocation_runner import BaseInvocationRunner, DefaultInvocationRunner
from .components.prompt_composer import BasePromptComposer, DefaultPromptComposer
from .components.runtime_skill_context_bridge import (
    BaseRuntimeSkillContextBridge,
    DefaultRuntimeSkillContextBridge,
)
from .components.stream_renderer import BaseStreamDisplayRenderer, ConsoleStreamDisplayRenderer
from .components.tool_interrupt_controller import BaseToolInterruptController, InMemoryToolInterruptController
from .components.tool_loop_engine import BaseToolLoopEngine, DefaultToolLoopEngine
from .components.trace_recorder import BaseTraceRecorder, InMemoryTraceRecorder
import logging
from core.Exception import *
from prompt import (
    PromptBlock,
    SystemPromptTemplate,
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
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        history_via_context_manager: bool = False,
        callback_manager=None,
        skill_manager=None,
        verbose_thinking: bool = True,
        trace_recorder: Optional[BaseTraceRecorder] = None,
        stream_renderer: Optional[BaseStreamDisplayRenderer] = None,
        prompt_composer: Optional[BasePromptComposer] = None,
        history_message_assembler: Optional[BaseHistoryMessageAssembler] = None,
        runtime_skill_context_bridge: Optional[BaseRuntimeSkillContextBridge] = None,
        tool_interrupt_controller: Optional[BaseToolInterruptController] = None,
        tool_loop_engine: Optional[BaseToolLoopEngine] = None,
        invocation_runner: Optional[BaseInvocationRunner] = None,
        reasoning: Optional[dict[str, Any]] = None,
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
        self.reasoning: Optional[dict[str, Any]] = None
        if reasoning:
            assert isinstance(reasoning, dict)
            if "effort" in reasoning:
                assert reasoning['effort'] in ["low", "medium", "high"]
            if "summary" in reasoning:
                assert reasoning['summary'] in ["auto", "none"]
            self.reasoning = reasoning
        self.verbose_thinking = verbose_thinking
        self.trace_recorder = trace_recorder or InMemoryTraceRecorder()
        self.stream_renderer = stream_renderer or ConsoleStreamDisplayRenderer()
        self.prompt_composer = prompt_composer or DefaultPromptComposer()
        self.history_message_assembler = (
            history_message_assembler or DefaultHistoryMessageAssembler()
        )
        self.runtime_skill_context_bridge = (
            runtime_skill_context_bridge or DefaultRuntimeSkillContextBridge()
        )
        self.tool_interrupt_controller = (
            tool_interrupt_controller or InMemoryToolInterruptController()
        )
        self.tool_loop_engine = tool_loop_engine or DefaultToolLoopEngine()
        self.invocation_runner = invocation_runner or DefaultInvocationRunner()
        self._current_query: str = ""  # 当前查询（供 get_enhanced_prompt 使用）
        self.history_via_context_manager = history_via_context_manager

        logger.info(f"BasicAgent '{name}' 初始化完成，工具调用: {'启用' if enable_tool else '禁用'}，provider: {llm.provider_name}")

    def _get_serializable_state(self) -> dict[str, Any]:
        state = super()._get_serializable_state()
        trace_state = self.trace_recorder.export_state()
        state.update({
            "history_via_context_manager": self.history_via_context_manager,
            "reasoning": self.reasoning,
            "verbose_thinking": self.verbose_thinking,
        })
        state.update(trace_state)
        state.update(self.tool_interrupt_controller.export_state())
        return state

    def _restore_serializable_state(self, state: Optional[dict[str, Any]]) -> None:
        super()._restore_serializable_state(state)
        state = state or {}
        self.history_via_context_manager = state.get("history_via_context_manager", False)
        self.trace_recorder.restore_state(state)
        self.tool_interrupt_controller.restore_state(state)
        self.reasoning = state.get("reasoning")
        self.verbose_thinking = bool(state.get("verbose_thinking", False))

    @property
    def trace_history(self) -> list[dict[str, Any]]:
        return self.trace_recorder.trace_history

    @trace_history.setter
    def trace_history(self, value: list[dict[str, Any]]) -> None:
        self.trace_recorder.trace_history = list(value or [])

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

    def _build_assistant_history_entries(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[Any]:
        formatted = self.llm.format_assistant_message(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
        )
        entries = self._as_history_entries(formatted)
        if entries:
            return entries
        if content:
            return [AssistantMessage(content)]
        return []

    def _build_assistant_history_entries_from_response(
        self,
        response: Any,
        *,
        include_reasoning: bool = True,
        fallback_content: Optional[str] = None,
        fallback_thinking: Optional[str] = None,
    ) -> list[Any]:
        provider_content = self.llm.get_response_content(response)
        provider_thinking = (
            fallback_thinking
            if fallback_thinking is not None
            else self.llm.get_thinking_content(response)
        )
        if fallback_content is None or provider_content == fallback_content:
            formatted = self.llm.format_assistant_response(
                response,
                include_reasoning=include_reasoning,
            )
            entries = self._as_history_entries(formatted)
            if entries:
                return entries
        return self._build_assistant_history_entries(
            content=fallback_content if fallback_content is not None else provider_content,
            thinking=provider_thinking if include_reasoning else None,
        )

    def _begin_trace_turn(self, raw_query: str) -> tuple[str, str]:
        return self.trace_recorder.begin_turn(raw_query)

    def _get_last_turn_event_id(
        self,
        turn_id: Optional[str],
        *,
        exclude_types: Optional[set[str]] = None,
    ) -> Optional[str]:
        return self.trace_recorder.get_last_turn_event_id(turn_id, exclude_types=exclude_types)

    def _set_round_reasoning(
        self,
        content: str,
        *,
        turn_id: Optional[str],
        round_number: Optional[int],
        mode: str,
        stream: bool,
    ) -> Optional[str]:
        return self.trace_recorder.set_round_reasoning(
            content,
            turn_id=turn_id,
            round_number=round_number,
            mode=mode,
            stream=stream,
        )

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
        allow_empty: bool = False,
    ) -> Optional[str]:
        return self.trace_recorder.record_assistant_message(
            turn_id,
            content,
            parent_id=parent_id,
            stage=stage,
            round_number=round_number,
            mode=mode,
            stream=stream,
            allow_empty=allow_empty,
        )

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
        return self.trace_recorder.record_tool_call(
            turn_id,
            tool_name,
            tool_args,
            tool_id,
            parent_id=parent_id,
            round_number=round_number,
            mode=mode,
            stream=stream,
        )

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
        return self.trace_recorder.record_tool_result(
            turn_id,
            tool_name,
            tool_args,
            tool_id,
            content,
            parent_id=parent_id,
            round_number=round_number,
            mode=mode,
            stream=stream,
            success=success,
        )

    def _record_turn_end(
        self,
        turn_id: Optional[str],
        *,
        final_event_id: Optional[str] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        status: str = "completed",
    ) -> str:
        return self.trace_recorder.record_turn_end(
            turn_id,
            final_event_id=final_event_id,
            mode=mode,
            stream=stream,
            status=status,
        )

    def _new_stream_display_state(self) -> Any:
        return self.stream_renderer.create_state()

    def _display_stream_event(self, state: Any, event: dict[str, Any]) -> None:
        self.stream_renderer.render_event(state, event)

    @staticmethod
    def _stream_snapshot_suffix(displayed: str, full_text: str) -> str:
        if not full_text:
            return ""
        if full_text.startswith(displayed):
            return full_text[len(displayed):]
        if displayed:
            return ""
        return full_text

    def get_last_tool_interrupt(self) -> Optional[dict[str, Any]]:
        return self.tool_interrupt_controller.get_last_interrupt()

    def _clear_last_tool_interrupt(self) -> None:
        self.tool_interrupt_controller.clear_last_interrupt()

    def _finalize_tool_interrupt(
        self,
        *,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        round_number: int,
        tool_result: ToolResult,
        parent_id: Optional[str],
        mode: str,
        stream: bool,
        turn_history: Optional[list[Any]] = None,
        tool_message: Optional[Any] = None,
    ) -> ToolInterruption:
        result_text = tool_result.to_display_string()
        result_event_id = self._record_tool_result(
            turn_id,
            tool_name,
            tool_args,
            tool_id,
            result_text,
            parent_id=parent_id,
            round_number=round_number,
            mode=mode,
            stream=stream,
            success=False,
        )
        if turn_history is not None:
            if tool_message is not None:
                turn_history.append(tool_message)
            self.add_messages(turn_history)
        self._record_turn_end(
            turn_id,
            final_event_id=result_event_id,
            mode=mode,
            stream=stream,
            status=tool_result.status,
        )
        return self.tool_interrupt_controller.create_interruption(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_id=tool_id,
            round_number=round_number,
            tool_result=tool_result,
        )

    def _print_stream_final(self, state: Any, final_text: str) -> None:
        self.stream_renderer.render_final(state, final_text)

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
                "verbose_thinking": state.get("verbose_thinking", False),
            }
        )
        return kwargs


    # @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) :
        return self.invocation_runner.invoke(
            self,
            query,
            max_iter=max_iter,
            temperature=temperature,
            **kwargs,
        )

    def stream_invoke(self,query: str,temperature: float = 0.7, **kwargs):
        return self.invocation_runner.stream_invoke(
            self,
            query,
            temperature=temperature,
            **kwargs,
        )

    def stream_invoke_with_tool(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs
        ):
        yield from self.invocation_runner.stream_invoke_with_tool(
            self,
            query,
            max_iter=max_iter,
            temperature=temperature,
            **kwargs,
        )#type: ignore

    def invoke_with_tool(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs
    ) -> str:
        return self.tool_loop_engine.invoke(
            self,
            query,
            messages,
            max_iter=max_iter,
            temperature=temperature,
            trace_query=trace_query,
            **kwargs,
        )

    async def ainvoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        return await self.invocation_runner.ainvoke(
            self,
            query,
            max_iter=max_iter,
            temperature=temperature,
            **kwargs,
        )

    async def ainvoke_with_tool(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs
    ) -> str:
        return await self.tool_loop_engine.ainvoke(
            self,
            query,
            messages,
            max_iter=max_iter,
            temperature=temperature,
            trace_query=trace_query,
            **kwargs,
        )

    async def astream_invoke(self, query: str,max_iter:int=10, temperature: float = 0.7, **kwargs) -> str:
        return await self.invocation_runner.astream_invoke(
            self,
            query,
            max_iter=max_iter,
            temperature=temperature,
            **kwargs,
        )

    async def astream_invoke_with_tool(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self.tool_loop_engine.astream_invoke(
            self,
            query,
            max_iter=max_iter,
            temperature=temperature,
            trace_query=trace_query,
            **kwargs,
        ):
            yield event

 
    @override
    def get_enhanced_prompt(self) -> str:
        return self.prompt_composer.get_enhanced_prompt(self)

    def get_system_prompt_template(self) -> SystemPromptTemplate:
        return self.prompt_composer.get_system_prompt_template(self)

    def get_system_prompt_blocks(self) -> list[PromptBlock]:
        return self.prompt_composer.get_system_prompt_blocks(self)

    def _build_core_prompt_blocks(
        self,
        *,
        start_order: int,
        include_tool_policy: bool,
    ) -> list[PromptBlock]:
        return self.prompt_composer.build_core_prompt_blocks(
            self,
            start_order=start_order,
            include_tool_policy=include_tool_policy,
        )

    def _get_tool_catalog_prompt(self) -> str:
        return self.prompt_composer.get_tool_catalog_prompt(self)

    def _get_tool_inventory_prompt(self, *, include_parameters: bool = False) -> str:
        return self.prompt_composer.get_tool_inventory_prompt(
            self,
            include_parameters=include_parameters,
        )

    def _should_include_tool_inventory_block(self) -> bool:
        """普通 Function Calling Agent 默认不在 system prompt 注入工具清单。"""
        return False

    def _tool_inventory_mode(self) -> str:
        """控制工具清单模式。支持 none/compact/full。"""
        return "none"

    def _build_tool_inventory_block(self, order: int) -> PromptBlock | None:
        return self.prompt_composer.build_tool_inventory_block(self, order)

    def _build_shared_prompt_blocks(
        self,
        *,
        start_order: int,
        include_custom_prompt: bool = True,
        include_memory: bool = True,
        include_skills: bool = True,
    ) -> list[PromptBlock]:
        return self.prompt_composer.build_shared_prompt_blocks(
            self,
            start_order=start_order,
            include_custom_prompt=include_custom_prompt,
            include_memory=include_memory,
            include_skills=include_skills,
        )

    def _append_runtime_skill_context_message(self, messages: list[Any]) -> None:
        self.runtime_skill_context_bridge.append_runtime_skill_context_message(self, messages)

    def _append_tool_ephemeral_context_message(
        self,
        *,
        tool_name: str,
        context: Any,
        messages: list[Any],
    ) -> None:
        self.runtime_skill_context_bridge.append_tool_ephemeral_context_message(
            self,
            tool_name=tool_name,
            context=context,
            messages=messages,
        )

    def _clear_ephemeral_skill_state(self) -> None:
        self.runtime_skill_context_bridge.clear_ephemeral_skill_state(self)

    def _maybe_inject_runtime_skill_context(
        self,
        *,
        tool_name: str,
        messages: list[Any],
    ) -> None:
        self.runtime_skill_context_bridge.maybe_inject_runtime_skill_context(
            self,
            tool_name=tool_name,
            messages=messages,
        )

    def _maybe_inject_tool_ephemeral_context(
        self,
        *,
        tool_name: str,
        tool_result: ToolResult,
        messages: list[Any],
    ) -> None:
        self.runtime_skill_context_bridge.maybe_inject_tool_ephemeral_context(
            self,
            tool_name=tool_name,
            tool_result=tool_result,
            messages=messages,
        )

    def _get_extension_prompt_blocks(self, start_order: int) -> list[PromptBlock]:
        return self.prompt_composer.get_extension_prompt_blocks(start_order)

    def with_prompt_block(self, block: PromptBlock) -> "BasicAgent":
        self.prompt_composer.with_prompt_block(block)
        return self

    def with_prompt_blocks(self, blocks: list[PromptBlock]) -> "BasicAgent":
        self.prompt_composer.with_prompt_blocks(blocks)
        return self

    def _use_context_history(self) -> bool:
        return self.history_message_assembler.use_context_history(self)

    def _context_include_history(self) -> bool:
        return self.history_message_assembler.context_include_history(self)

    def _append_runtime_history(self, messages: list[Message | dict[str, str]]) -> None:
        self.history_message_assembler.append_runtime_history(self, messages)

    def _build_start_messages(self, query: str) -> list[Any]:
        return self.history_message_assembler.build_start_messages(self, query)

        

    def get_trace_history(self) -> list[dict[str, Any]]:
        """获取完整会话转录。"""
        return self._make_json_safe(self.trace_history)

    def clear_trace_history(self) -> None:
        """清空完整会话转录。"""
        self.trace_recorder.clear()
