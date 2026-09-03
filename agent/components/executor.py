"""Default provider-neutral Agent executor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, AsyncIterator, Callable, Iterator
from uuid import uuid4

from core.Config import Config
from core.Exception import AgentStopRequested, HookExecutionError, ToolInterruption
from core.history import CanonicalMessage
from core.hooks import HookManager
from core.llm import EasyLLM
from core.permissions import PermissionContext, PermissionEngine
from core.request_compiler import compile_prompt_blocks
from core.request_input import ReplayRequestInput
from context.manager import ContextManager
from metamessage import BaseMetaMessageManager, MetaMessage, MetaMessageLifecycle
from runtime import AgentStreamEvent, AgentStreamEventType, ExecutionContext, RuntimeEventBus, RuntimeEventType
from Tool.BaseTool import ToolResult
from Tool.ToolRegistry import ToolRegistry

from .conversation_history import ConversationHistory
from .prompt_composer import BaseSystemPromptComposer, PromptBuildContext
from .tool_interrupt_controller import BaseToolInterruptController, InMemoryToolInterruptController


class AgentInvocationPhase(str, Enum):
    CREATED = "created"
    STARTED = "started"
    COMPILING = "compiling"
    LLM = "llm"
    TOOL = "tool"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class AgentInvocationState:
    invocation_id: str
    query: str
    stream: bool
    phase: AgentInvocationPhase = AgentInvocationPhase.CREATED
    round_number: int = 0
    output: str = ""
    error: Exception | None = None


@dataclass(slots=True)
class AgentExecutionServices:
    agent_id: str
    llm: EasyLLM
    config: Config
    history: ConversationHistory
    prompt_composer: BaseSystemPromptComposer
    prompt_context_factory: Callable[[str], PromptBuildContext]
    execution_context: ExecutionContext
    event_bus: RuntimeEventBus
    metamessage_manager: BaseMetaMessageManager
    permission_engine: PermissionEngine
    permission_context: PermissionContext
    hook_manager: HookManager
    tool_registry: ToolRegistry | None = None
    context_manager: ContextManager | None = None
    reasoning: dict[str, Any] | None = None
    interrupt_controller: BaseToolInterruptController = field(default_factory=InMemoryToolInterruptController)
    stop_checker: Callable[[], str | None] = lambda: None
    mailbox_sync: Callable[[], None] = lambda: None


class BaseAgentExecutor(ABC):
    @abstractmethod
    def invoke(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[AgentStreamEvent]:
        raise NotImplementedError

    @abstractmethod
    def astream(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        raise NotImplementedError


class DefaultAgentExecutor(BaseAgentExecutor):
    def __init__(self) -> None:
        self.last_state: AgentInvocationState | None = None

    @staticmethod
    def _validate(query: str, max_iter: int, temperature: float) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    @staticmethod
    def _publish(
        services: AgentExecutionServices,
        event_type: RuntimeEventType,
        state: AgentInvocationState,
        data: dict[str, Any] | None = None,
    ) -> None:
        services.event_bus.publish(
            event_type,
            agent_id=services.agent_id,
            invocation_id=state.invocation_id,
            data=data,
        )

    @staticmethod
    def _stream_event(
        services: AgentExecutionServices,
        state: AgentInvocationState,
        event_type: AgentStreamEventType,
        sequence: int,
        *,
        content: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> AgentStreamEvent:
        event = AgentStreamEvent(
            type=event_type,
            invocation_id=state.invocation_id,
            sequence=sequence,
            content=content,
            data=dict(data or {}),
        )
        services.event_bus.publish(
            RuntimeEventType.STREAM_EVENT,
            agent_id=services.agent_id,
            invocation_id=state.invocation_id,
            data={"stream_event": event},
        )
        return event

    @staticmethod
    def _check_stop(services: AgentExecutionServices) -> None:
        reason = services.stop_checker()
        if reason is not None:
            raise AgentStopRequested(reason or "Agent stop requested")

    @staticmethod
    def _tool_payload(services: AgentExecutionServices) -> Any:
        registry = services.tool_registry
        if registry is None or not registry.get_tool_names():
            return None
        return registry.export_tools(
            services.llm.provider_name or "openai",
            mode=services.config.tool_schema_mode,
        )

    @staticmethod
    def _tool_descriptors(services: AgentExecutionServices) -> list[dict[str, Any]]:
        registry = services.tool_registry
        if registry is None:
            return []
        return registry.list_tool_descriptors(stable=True, include_parameters=True)

    @staticmethod
    def _observable_options(
        services: AgentExecutionServices,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "provider": services.llm.provider_name,
            "model": services.llm.model,
            **dict(options),
        }

    @staticmethod
    def _canonical_request_input(
        services: AgentExecutionServices,
        request_input: ReplayRequestInput,
    ) -> list[CanonicalMessage]:
        return services.llm.replay_to_canonical_history(
            request_input.as_legacy_messages()
        )

    def _build_request(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        *,
        temperature: float,
        kwargs: dict[str, Any],
    ) -> tuple[ReplayRequestInput, Any, dict[str, Any]]:
        state.phase = AgentInvocationPhase.COMPILING
        try:
            services.mailbox_sync()
            services.metamessage_manager.flush()
            prompt_context = services.prompt_context_factory(state.query)
            blocks = services.prompt_composer.compose(prompt_context)
            compiled = compile_prompt_blocks(blocks, cache_policy=services.config.cache_policy)
            tools = self._tool_payload(services)
            if services.context_manager is not None:
                context_budget = services.context_manager.budget.max_tokens
                estimated_tokens = services.llm.count_request_tokens(
                    services.context_manager.counter,
                    services.history.replay,
                    system_prompt=compiled.system_prompt,
                    tools=tools,
                    reasoning=services.reasoning,
                )
                if estimated_tokens > context_budget:
                    compaction_hook = services.hook_manager.before_compaction(
                        {
                            "canonical_history": services.history.canonical,
                            "replay_history": services.history.replay,
                            "max_tokens": context_budget,
                            "estimated_tokens": estimated_tokens,
                            "invocation_id": state.invocation_id,
                        }
                    )
                    if not compaction_hook.blocked:
                        compaction = services.context_manager.compact_persistent_history(
                            compaction_hook.payload.get("canonical_history", services.history.canonical),
                            compaction_hook.payload.get("replay_history", services.history.replay),
                            provider_name=services.llm.provider_name,
                            token_counter=services.context_manager.counter,
                            system_prompt=compiled.system_prompt,
                            tools=tools,
                            reasoning=services.reasoning,
                            max_tokens=int(compaction_hook.payload.get("max_tokens") or context_budget),
                            tokens_before_override=estimated_tokens,
                        )
                        if compaction.was_compacted:
                            services.history.replace(compaction.canonical_history)
                            self._publish(
                                services,
                                RuntimeEventType.HISTORY_COMPACTED,
                                state,
                                {
                                    "tokens_before": compaction.tokens_before,
                                    "tokens_after": compaction.tokens_after,
                                    "metadata": compaction.metadata,
                                    "hook_audit": compaction_hook.audit,
                                },
                            )
            if services.context_manager is not None:
                request_input = services.context_manager.build_request_input(
                    query=state.query,
                    replay_history=services.history.replay,
                    provider_name=services.llm.provider_name,
                    system_prompt=compiled.system_prompt,
                    system_prompt_blocks=compiled.system_prompt_blocks,
                    system_reminder_blocks=compiled.system_reminder_blocks,
                    dynamic_tail_blocks=compiled.dynamic_tail_blocks,
                    on_demand_expansion_blocks=compiled.on_demand_expansion_blocks,
                    cache_policy=compiled.cache_policy,
                    include_query=False,
                    tools=tools,
                    reasoning=services.reasoning,
                )
            else:
                request_input = ReplayRequestInput(
                    provider_name=services.llm.provider_name,
                    replay_history=services.history.replay,
                    system_prompt=compiled.system_prompt,
                    system_prompt_blocks=compiled.system_prompt_blocks,
                    system_reminder_blocks=compiled.system_reminder_blocks,
                    dynamic_tail_blocks=compiled.dynamic_tail_blocks,
                    on_demand_expansion_blocks=compiled.on_demand_expansion_blocks,
                    cache_policy=compiled.cache_policy,
                )
                request_input.apply_runtime_layers()

            options = {
                "temperature": temperature,
                "reasoning": services.reasoning,
                **kwargs,
            }
            hook = services.hook_manager.before_llm_request(
                {
                    "request_input": request_input,
                    "tools": tools,
                    "options": dict(options),
                    "invocation_id": state.invocation_id,
                    "round_number": state.round_number,
                }
            )
            if hook.blocked:
                raise HookExecutionError(
                    hook.message or "LLM request blocked by hook",
                    stage="before_llm_request",
                    error_type=hook.error_type,
                    metadata={"audit": hook.audit},
                )
            request_input = hook.payload.get("request_input", request_input)
            tools = hook.payload.get("tools", tools)
            options = dict(hook.payload.get("options", options))
            return request_input, tools, options
        except Exception:
            raise

    @staticmethod
    def _tool_call_parts(tool_call: Any, fallback_id: str) -> tuple[str, dict[str, Any], str]:
        if isinstance(tool_call, dict):
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            name = tool_call.get("name") or function.get("name") or ""
            arguments = tool_call.get("arguments", function.get("arguments", tool_call.get("input", {})))
            tool_id = tool_call.get("call_id") or tool_call.get("id") or fallback_id
        else:
            function = getattr(tool_call, "function", None)
            name = getattr(tool_call, "name", None) or getattr(function, "name", None) or ""
            arguments = getattr(tool_call, "arguments", None)
            if arguments is None:
                arguments = getattr(function, "arguments", None)
            if arguments is None:
                arguments = getattr(tool_call, "input", {})
            tool_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None) or fallback_id
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON arguments for tool {name or 'unknown'}: {exc}") from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {name or 'unknown'} must be an object")
        if not name:
            raise ValueError("Tool call is missing a tool name")
        return str(name), dict(arguments), str(tool_id)

    @staticmethod
    def _apply_after_tool_hook(
        services: AgentExecutionServices,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> ToolResult:
        tool_spec = (
            services.tool_registry.get_tool_spec(tool_name)
            if services.tool_registry is not None
            else None
        )
        hook = services.hook_manager.after_tool_use(
            {
                "tool_name": tool_name,
                "tool_args": dict(arguments),
                "tool_spec": tool_spec,
                "tool_result": result,
            }
        )
        if hook.blocked:
            return ToolResult.error(
                hook.message or "Tool result blocked by hook",
                error_type=hook.error_type,
                metadata={"hookAudit": hook.audit},
            )
        resolved = hook.payload.get("tool_result", result)
        if not isinstance(resolved, ToolResult):
            raise TypeError("after_tool_use hook must keep tool_result as ToolResult")
        return resolved

    @staticmethod
    def _apply_after_llm_hook(
        services: AgentExecutionServices,
        response: Any,
    ) -> Any:
        hook = services.hook_manager.after_llm_response({"response": response})
        if hook.blocked:
            raise HookExecutionError(
                hook.message or "LLM response blocked by hook",
                stage="after_llm_response",
                error_type=hook.error_type,
                metadata={"audit": hook.audit},
            )
        return hook.payload.get("response", response)

    @staticmethod
    def _prepare_tool(
        services: AgentExecutionServices,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, Any, dict[str, Any], ToolResult | None]:
        registry = services.tool_registry
        if registry is None:
            return tool_name, None, arguments, ToolResult.error("No ToolRegistry is installed", error_type="tool_registry_missing")
        tool_spec = registry.get_tool_spec(tool_name)
        hook = services.hook_manager.before_tool_use(
            {
                "tool_name": tool_name,
                "tool_args": dict(arguments),
                "tool_spec": tool_spec,
            }
        )
        if hook.blocked:
            return tool_name, None, arguments, ToolResult.error(
                hook.message or "Tool call blocked by hook",
                error_type=hook.error_type,
                metadata={"hookAudit": hook.audit},
        )
        tool_name = str(hook.payload.get("tool_name", tool_name))
        arguments = dict(hook.payload.get("tool_args", arguments))
        try:
            tool, validated = registry.validate_tool_call(tool_name, arguments)
            authorization = registry.authorize_tool_call(
                tool,
                validated,
                permission_context=services.permission_context,
                permission_engine=services.permission_engine,
            )
            return tool_name, tool, validated, authorization
        except Exception as exc:
            return tool_name, None, arguments, ToolResult.error(
                str(exc),
                error_type=exc.__class__.__name__,
                metadata={"tool_name": tool_name},
            )

    def _finalize_tool_result(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tool_id: str,
        result: ToolResult,
    ) -> ToolResult:
        result = self._apply_after_tool_hook(
            services,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )
        text = result.to_display_string()
        trace = services.llm.tool_result_to_canonical(text, tool_id, tool_name)
        event_type = RuntimeEventType.TOOL_INVOKE_COMPLETED if result.status == "success" else RuntimeEventType.TOOL_INVOKE_FAILED
        self._publish(
            services,
            event_type,
            state,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_id,
                "output": text,
                "result": result,
                "trace": trace,
                "error_type": result.error_type,
            },
        )
        if result.status == "needs_confirmation" and services.config.interrupt_on_confirmation:
            raise services.interrupt_controller.create_interruption(
                tool_name=tool_name,
                tool_args=arguments,
                tool_id=tool_id,
                round_number=state.round_number,
                tool_result=result,
            )
        services.history.append_tool_result(text, tool_id, tool_name)
        self._emit_tool_context(services, tool_name=tool_name, result=result)
        return result

    @staticmethod
    def _emit_tool_context(
        services: AgentExecutionServices,
        *,
        tool_name: str,
        result: ToolResult,
    ) -> None:
        if result.ephemeral_context is not None:
            value = result.ephemeral_context
            content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str, indent=2)
            if content.strip():
                services.metamessage_manager.emit(
                    MetaMessage(
                        name=f"tool_context:{tool_name}",
                        content=f"Runtime context produced by tool `{tool_name}`:\n{content}",
                        lifecycle=MetaMessageLifecycle.INVOCATION,
                        metadata={"source": "tool_result", "toolName": tool_name},
                    )
                )
        if tool_name != "tool_schema_tool" or services.tool_registry is None:
            return
        names = services.tool_registry.get_deferred_expanded_tool_names()
        if names:
            services.metamessage_manager.emit(
                MetaMessage(
                    name="deferred_tools_expanded",
                    content="The following deferred tool schemas are now available: " + ", ".join(names),
                    lifecycle=MetaMessageLifecycle.INVOCATION,
                    dedup_key=f"deferred:{','.join(names)}",
                    metadata={"source": "deferred_tools", "expandedToolNames": names},
                )
            )

    def _execute_tool(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        tool_call: Any,
        index: int,
    ) -> tuple[str, dict[str, Any], str, ToolResult]:
        tool_name, arguments, tool_id = self._tool_call_parts(tool_call, f"call_{state.round_number}_{index}")
        requested_name = tool_name
        requested_arguments = dict(arguments)
        tool_name, tool, validated, result = self._prepare_tool(services, tool_name=tool_name, arguments=arguments)
        self._publish(
            services,
            RuntimeEventType.TOOL_INVOKE_STARTED,
            state,
            {
                "tool_name": tool_name,
                "arguments": validated,
                "tool_call_id": tool_id,
                "requested_tool_name": requested_name,
                "requested_arguments": requested_arguments,
            },
        )
        if result is None:
            try:
                raw = tool.run(validated)
                result = services.tool_registry.normalize_tool_result(tool_name, raw)  # type: ignore[union-attr]
            except Exception as exc:
                result = ToolResult.error(str(exc), error_type=exc.__class__.__name__)
        result = self._finalize_tool_result(
            services,
            state,
            tool_name=tool_name,
            arguments=validated,
            tool_id=tool_id,
            result=result,
        )
        return tool_name, validated, tool_id, result

    async def _aexecute_tool(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        tool_call: Any,
        index: int,
    ) -> tuple[str, dict[str, Any], str, ToolResult]:
        tool_name, arguments, tool_id = self._tool_call_parts(tool_call, f"call_{state.round_number}_{index}")
        requested_name = tool_name
        requested_arguments = dict(arguments)
        tool_name, tool, validated, result = self._prepare_tool(services, tool_name=tool_name, arguments=arguments)
        self._publish(
            services,
            RuntimeEventType.TOOL_INVOKE_STARTED,
            state,
            {
                "tool_name": tool_name,
                "arguments": validated,
                "tool_call_id": tool_id,
                "requested_tool_name": requested_name,
                "requested_arguments": requested_arguments,
            },
        )
        if result is None:
            try:
                raw = await tool.arun(validated)
                result = services.tool_registry.normalize_tool_result(tool_name, raw)  # type: ignore[union-attr]
            except Exception as exc:
                result = ToolResult.error(str(exc), error_type=exc.__class__.__name__)
        result = self._finalize_tool_result(
            services,
            state,
            tool_name=tool_name,
            arguments=validated,
            tool_id=tool_id,
            result=result,
        )
        return tool_name, validated, tool_id, result

    def _start(self, services: AgentExecutionServices, query: str, *, stream: bool) -> AgentInvocationState:
        state = AgentInvocationState(
            invocation_id=f"invoke_{uuid4().hex}",
            query=query,
            stream=stream,
            phase=AgentInvocationPhase.STARTED,
        )
        self.last_state = state
        services.history.append_query(query)
        self._publish(
            services,
            RuntimeEventType.AGENT_INVOKE_STARTED,
            state,
            {
                "query": query,
                "mode": services.execution_context.execution_mode,
                "stream": stream,
            },
        )
        return state

    def _finish(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        output: str,
        *,
        response: Any = None,
        thinking: str | None = None,
    ) -> str:
        state.phase = AgentInvocationPhase.COMMITTING
        if response is not None:
            services.history.append_response(
                response,
                include_reasoning=bool(services.config.persist_reasoning_history),
            )
        else:
            services.history.append_assistant(
                content=output,
                thinking=thinking if services.config.persist_reasoning_history else None,
            )
        state.output = output
        state.phase = AgentInvocationPhase.COMPLETED
        self._publish(
            services,
            RuntimeEventType.AGENT_INVOKE_COMPLETED,
            state,
            {
                "output": output,
                "output_messages": services.llm.assistant_message_to_canonical(
                    content=output,
                    thinking=thinking or None,
                ),
                "rounds": state.round_number,
            },
        )
        return output

    def _fail(self, services: AgentExecutionServices, state: AgentInvocationState, exc: Exception) -> None:
        state.error = exc
        interrupted = isinstance(exc, (ToolInterruption, AgentStopRequested))
        state.phase = AgentInvocationPhase.INTERRUPTED if interrupted else AgentInvocationPhase.FAILED
        self._publish(
            services,
            RuntimeEventType.AGENT_INVOKE_INTERRUPTED if interrupted else RuntimeEventType.AGENT_INVOKE_FAILED,
            state,
            {
                "error": exc,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "output": state.output,
                "output_messages": [],
            },
        )

    @staticmethod
    def _cleanup(services: AgentExecutionServices) -> None:
        if services.tool_registry is not None:
            services.tool_registry.clear_deferred_tool_expansions()

    def invoke(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        self._validate(query, max_iter, temperature)
        state = self._start(services, query, stream=False)
        try:
            for round_number in range(1, max_iter + 1):
                state.round_number = round_number
                self._check_stop(services)
                request_input, tools, options = self._build_request(
                    services,
                    state,
                    temperature=temperature,
                    kwargs=kwargs,
                )
                llm_id = f"llm_{round_number}_{uuid4().hex}"
                input_messages = self._canonical_request_input(services, request_input)
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_STARTED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "input": input_messages,
                        "tools": self._tool_descriptors(services) if tools else [],
                        "options": self._observable_options(services, options),
                        "request_input": request_input,
                    },
                )
                state.phase = AgentInvocationPhase.LLM
                try:
                    if tools:
                        response = services.llm.invoke_with_tools(request_input, tools, **options)
                    else:
                        response = services.llm.invoke_raw(request_input, **options)
                except Exception as exc:
                    self._publish(
                        services,
                        RuntimeEventType.LLM_INVOKE_FAILED,
                        state,
                        {"llm_invoke_id": llm_id, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                    raise
                output_messages = services.llm.response_to_canonical(response, include_reasoning=True)
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "response": response,
                        "output": output_messages,
                        "usage": services.llm.extract_usage_metrics(response),
                    },
                )
                response = self._apply_after_llm_hook(services, response)
                if tools and services.llm.has_tool_calls(response):
                    services.history.append_response(
                        response,
                        include_reasoning=bool(services.config.persist_reasoning_history),
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(services.llm.get_tool_calls(response)):
                        self._execute_tool(services, state, tool_call, index)
                    continue
                output = services.llm.get_response_content(response) or ""
                return self._finish(services, state, output, response=response)
            raise RuntimeError(f"Agent reached max_iter={max_iter} without a final response")
        except Exception as exc:
            self._fail(services, state, exc)
            raise
        finally:
            self._cleanup(services)

    async def ainvoke(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        self._validate(query, max_iter, temperature)
        state = self._start(services, query, stream=False)
        try:
            for round_number in range(1, max_iter + 1):
                state.round_number = round_number
                self._check_stop(services)
                request_input, tools, options = self._build_request(services, state, temperature=temperature, kwargs=kwargs)
                llm_id = f"llm_{round_number}_{uuid4().hex}"
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_STARTED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "input": self._canonical_request_input(services, request_input),
                        "tools": self._tool_descriptors(services) if tools else [],
                        "options": self._observable_options(services, options),
                        "request_input": request_input,
                    },
                )
                state.phase = AgentInvocationPhase.LLM
                try:
                    if tools:
                        response = await services.llm.ainvoke_with_tools(request_input, tools, **options)
                    else:
                        response = await services.llm.ainvoke_raw(request_input, **options)
                except Exception as exc:
                    self._publish(
                        services,
                        RuntimeEventType.LLM_INVOKE_FAILED,
                        state,
                        {"llm_invoke_id": llm_id, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                    raise
                output_messages = services.llm.response_to_canonical(response, include_reasoning=True)
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "response": response,
                        "output": output_messages,
                        "usage": services.llm.extract_usage_metrics(response),
                    },
                )
                response = self._apply_after_llm_hook(services, response)
                if tools and services.llm.has_tool_calls(response):
                    services.history.append_response(
                        response,
                        include_reasoning=bool(services.config.persist_reasoning_history),
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(services.llm.get_tool_calls(response)):
                        await self._aexecute_tool(services, state, tool_call, index)
                    continue
                output = services.llm.get_response_content(response) or ""
                return self._finish(services, state, output, response=response)
            raise RuntimeError(f"Agent reached max_iter={max_iter} without a final response")
        except Exception as exc:
            self._fail(services, state, exc)
            raise
        finally:
            self._cleanup(services)

    def stream(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[AgentStreamEvent]:
        self._validate(query, max_iter, temperature)
        state = self._start(services, query, stream=True)
        sequence = 0
        try:
            for round_number in range(1, max_iter + 1):
                state.round_number = round_number
                self._check_stop(services)
                request_input, tools, options = self._build_request(services, state, temperature=temperature, kwargs=kwargs)
                llm_id = f"llm_{round_number}_{uuid4().hex}"
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_STARTED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "input": self._canonical_request_input(services, request_input),
                        "tools": self._tool_descriptors(services) if tools else [],
                        "options": self._observable_options(services, options),
                        "request_input": request_input,
                    },
                )
                state.phase = AgentInvocationPhase.LLM
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                terminal: dict[str, Any] | None = None
                try:
                    provider_events = (
                        services.llm.stream_with_tools(request_input, tools, **options)
                        if tools
                        else services.llm.stream_events(request_input, **options)
                    )
                    for provider_event in provider_events:
                        event_type = provider_event.get("type")
                        if event_type == "text_delta":
                            delta = str(provider_event.get("delta") or "")
                            text_parts.append(delta)
                            sequence += 1
                            yield self._stream_event(services, state, AgentStreamEventType.TEXT_DELTA, sequence, content=delta)
                        elif event_type in {"thinking_delta", "reasoning_delta"}:
                            delta = str(provider_event.get("delta") or "")
                            thinking_parts.append(delta)
                            sequence += 1
                            yield self._stream_event(services, state, AgentStreamEventType.REASONING_DELTA, sequence, content=delta)
                        elif event_type in {"tool_calls", "final_response"}:
                            terminal = provider_event
                except Exception as exc:
                    self._publish(
                        services,
                        RuntimeEventType.LLM_INVOKE_FAILED,
                        state,
                        {"llm_invoke_id": llm_id, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                    raise
                terminal = terminal or {"type": "final_response", "content": "".join(text_parts), "thinking": "".join(thinking_parts)}
                content = str(terminal.get("content") or "".join(text_parts))
                thinking = str(terminal.get("thinking") or "".join(thinking_parts))
                tool_calls = list(terminal.get("tool_calls") or [])
                canonical_output = services.llm.assistant_message_to_canonical(
                    content=content,
                    tool_calls=tool_calls or None,
                    thinking=thinking or None,
                )
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "output": canonical_output,
                        "usage": services.llm.extract_usage_metrics({"usage": terminal.get("usage")}),
                    },
                )
                terminal = self._apply_after_llm_hook(services, terminal)
                if not isinstance(terminal, dict):
                    raise TypeError("after_llm_response hook must keep a streamed response as a dict")
                content = str(terminal.get("content") or content)
                thinking = str(terminal.get("thinking") or thinking)
                tool_calls = list(terminal.get("tool_calls") or tool_calls)
                if tool_calls:
                    services.history.append_assistant(
                        content=content,
                        tool_calls=tool_calls,
                        thinking=thinking if services.config.persist_reasoning_history else None,
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(tool_calls):
                        tool_name, arguments, tool_id = self._tool_call_parts(tool_call, f"call_{round_number}_{index}")
                        sequence += 1
                        yield self._stream_event(
                            services,
                            state,
                            AgentStreamEventType.TOOL_CALL,
                            sequence,
                            data={"tool_name": tool_name, "arguments": arguments, "tool_call_id": tool_id},
                        )
                        executed_name, executed_arguments, executed_id, result = self._execute_tool(
                            services, state, tool_call, index
                        )
                        sequence += 1
                        yield self._stream_event(
                            services,
                            state,
                            AgentStreamEventType.TOOL_RESULT,
                            sequence,
                            content=result.to_display_string(),
                            data={
                                "tool_name": executed_name,
                                "arguments": executed_arguments,
                                "tool_call_id": executed_id,
                                "status": result.status,
                                "result": result,
                            },
                        )
                    continue
                output = self._finish(services, state, content, thinking=thinking)
                sequence += 1
                yield self._stream_event(services, state, AgentStreamEventType.FINAL, sequence, content=output)
                return
            raise RuntimeError(f"Agent reached max_iter={max_iter} without a final response")
        except Exception as exc:
            self._fail(services, state, exc)
            sequence += 1
            yield self._stream_event(
                services,
                state,
                AgentStreamEventType.ERROR,
                sequence,
                content=str(exc),
                data={
                    "error_type": exc.__class__.__name__,
                    "interrupted": isinstance(exc, (ToolInterruption, AgentStopRequested)),
                },
            )
            raise
        finally:
            self._cleanup(services)

    async def astream(
        self,
        services: AgentExecutionServices,
        query: str,
        *,
        max_iter: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        self._validate(query, max_iter, temperature)
        state = self._start(services, query, stream=True)
        sequence = 0
        try:
            for round_number in range(1, max_iter + 1):
                state.round_number = round_number
                self._check_stop(services)
                request_input, tools, options = self._build_request(services, state, temperature=temperature, kwargs=kwargs)
                llm_id = f"llm_{round_number}_{uuid4().hex}"
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_STARTED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "input": self._canonical_request_input(services, request_input),
                        "tools": self._tool_descriptors(services) if tools else [],
                        "options": self._observable_options(services, options),
                        "request_input": request_input,
                    },
                )
                state.phase = AgentInvocationPhase.LLM
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                terminal: dict[str, Any] | None = None
                try:
                    if tools:
                        provider_events = services.llm.astream_with_tools(request_input, tools, **options)
                    else:
                        provider_events = services.llm.astream_events(request_input, **options)
                    async for provider_event in provider_events:
                        event_type = provider_event.get("type")
                        if event_type == "text_delta":
                            delta = str(provider_event.get("delta") or "")
                            text_parts.append(delta)
                            sequence += 1
                            yield self._stream_event(services, state, AgentStreamEventType.TEXT_DELTA, sequence, content=delta)
                        elif event_type in {"thinking_delta", "reasoning_delta"}:
                            delta = str(provider_event.get("delta") or "")
                            thinking_parts.append(delta)
                            sequence += 1
                            yield self._stream_event(services, state, AgentStreamEventType.REASONING_DELTA, sequence, content=delta)
                        elif event_type in {"tool_calls", "final_response"}:
                            terminal = provider_event
                except Exception as exc:
                    self._publish(
                        services,
                        RuntimeEventType.LLM_INVOKE_FAILED,
                        state,
                        {"llm_invoke_id": llm_id, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                    raise
                terminal = terminal or {"type": "final_response", "content": "".join(text_parts), "thinking": "".join(thinking_parts)}
                content = str(terminal.get("content") or "".join(text_parts))
                thinking = str(terminal.get("thinking") or "".join(thinking_parts))
                tool_calls = list(terminal.get("tool_calls") or [])
                canonical_output = services.llm.assistant_message_to_canonical(
                    content=content,
                    tool_calls=tool_calls or None,
                    thinking=thinking or None,
                )
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "output": canonical_output,
                        "usage": services.llm.extract_usage_metrics({"usage": terminal.get("usage")}),
                    },
                )
                terminal = self._apply_after_llm_hook(services, terminal)
                if not isinstance(terminal, dict):
                    raise TypeError("after_llm_response hook must keep a streamed response as a dict")
                content = str(terminal.get("content") or content)
                thinking = str(terminal.get("thinking") or thinking)
                tool_calls = list(terminal.get("tool_calls") or tool_calls)
                if tool_calls:
                    services.history.append_assistant(
                        content=content,
                        tool_calls=tool_calls,
                        thinking=thinking if services.config.persist_reasoning_history else None,
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(tool_calls):
                        tool_name, arguments, tool_id = self._tool_call_parts(tool_call, f"call_{round_number}_{index}")
                        sequence += 1
                        yield self._stream_event(
                            services,
                            state,
                            AgentStreamEventType.TOOL_CALL,
                            sequence,
                            data={"tool_name": tool_name, "arguments": arguments, "tool_call_id": tool_id},
                        )
                        executed_name, executed_arguments, executed_id, result = await self._aexecute_tool(
                            services, state, tool_call, index
                        )
                        sequence += 1
                        yield self._stream_event(
                            services,
                            state,
                            AgentStreamEventType.TOOL_RESULT,
                            sequence,
                            content=result.to_display_string(),
                            data={
                                "tool_name": executed_name,
                                "arguments": executed_arguments,
                                "tool_call_id": executed_id,
                                "status": result.status,
                                "result": result,
                            },
                        )
                    continue
                output = self._finish(services, state, content, thinking=thinking)
                sequence += 1
                yield self._stream_event(services, state, AgentStreamEventType.FINAL, sequence, content=output)
                return
            raise RuntimeError(f"Agent reached max_iter={max_iter} without a final response")
        except Exception as exc:
            self._fail(services, state, exc)
            sequence += 1
            yield self._stream_event(
                services,
                state,
                AgentStreamEventType.ERROR,
                sequence,
                content=str(exc),
                data={
                    "error_type": exc.__class__.__name__,
                    "interrupted": isinstance(exc, (ToolInterruption, AgentStopRequested)),
                },
            )
            raise
        finally:
            self._cleanup(services)


__all__ = [
    "AgentExecutionServices",
    "AgentInvocationPhase",
    "AgentInvocationState",
    "BaseAgentExecutor",
    "DefaultAgentExecutor",
]
