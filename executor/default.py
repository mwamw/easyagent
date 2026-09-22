"""Default ReAct execution coordinator."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, AsyncIterator, Iterator, TypeVar
from uuid import uuid4

from core.Exception import AgentStopRequested, HookExecutionError, ToolInterruption
from core.history import CanonicalMessage
from permissions.base import BasePermissionEngine
from permissions.models import PermissionBehavior
from core.request_input import ReplayRequestInput
from context.models import ContextCompactionResult, ContextUsage
from core.scope import RuntimeScope
from runtime.metamessage import MetaMessage, MetaMessageLifecycle
from runtime import AgentStreamEvent, AgentStreamEventType, RuntimeEventType
from Tool.BaseTool import ToolResult
from .base import AgentExecutionServices, AgentInvocationPhase, AgentInvocationState, BaseAgentExecutor


_ContextResult = TypeVar("_ContextResult")


def _expect_context_result(
    method: str,
    value: object,
    expected: type[_ContextResult],
) -> _ContextResult:
    if not isinstance(value, expected):
        raise TypeError(
            f"Context.{method}() must return {expected.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


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
        services.runtime.event_bus.publish(
            event_type,
            agent_id=services.runtime.agent_id,
            invocation_id=state.invocation_id,
            data=data,
        )

    @staticmethod
    def _error_payload(exc: Exception) -> dict[str, Any]:
        """Preserve safe provider diagnostics without leaking request credentials."""

        payload: dict[str, Any] = {
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            try:
                payload["status_code"] = int(status_code)
            except (TypeError, ValueError):
                payload["status_code"] = str(status_code)

        error_code = getattr(exc, "code", None)
        if error_code:
            payload["error_code"] = str(error_code)

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = (
                getattr(exc, "request_id", None)
                or headers.get("x-request-id")
                or headers.get("x-oneapi-request-id")
            )
            edge_trace_id = headers.get("cf-ray")
            if request_id:
                payload["request_id"] = str(request_id)
            if edge_trace_id:
                payload["edge_trace_id"] = str(edge_trace_id)
        return payload

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
        services.runtime.event_bus.publish(
            RuntimeEventType.STREAM_EVENT,
            agent_id=services.runtime.agent_id,
            invocation_id=state.invocation_id,
            data={"stream_event": event},
        )
        return event

    @staticmethod
    def _check_stop(services: AgentExecutionServices) -> None:
        reason = services.runtime.stop_reason_if_requested()
        if reason is not None:
            raise AgentStopRequested(reason or "Agent stop requested")

    @staticmethod
    def _select_terminal_event(
        current: dict[str, Any] | None,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        if current is None or candidate.get("type") == "tool_calls":
            return candidate
        if current.get("type") == "tool_calls":
            return current
        return candidate

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

    def _publish_compaction(self, services, state, result):
        self._publish(services, RuntimeEventType.HISTORY_COMPACTED, state, {
            "tokens_before": result.tokens_before, "tokens_after": result.tokens_after,
            "metadata": result.metadata,
        })

    @staticmethod
    def _usage_output_is_retained(services, canonical_output):
        return bool(services.config.persist_reasoning_history) or not any(
            block.type == "reasoning" for message in canonical_output for block in message.content
        )

    @staticmethod
    def _apply_after_llm_hook(services: AgentExecutionServices, response: Any) -> Any:
        if services.runtime.hooks is None:
            return response
        hook = services.runtime.hooks.after_llm_response({"response": response})
        if hook.blocked:
            raise HookExecutionError(
                hook.message or "LLM response blocked by hook",
                stage="after_llm_response",
                error_type=hook.error_type,
                metadata={"audit": hook.audit},
            )
        return hook.payload.get("response", response)

    def execute_tool_call(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        tool_call: Any,
        index: int = 0,
    ) -> tuple[str, dict[str, Any], str, ToolResult]:
        """Run Hook → Tool → Permission → execution → Context for one model call."""
        registry = services.tool_registry
        if registry is None:
            raise RuntimeError("Model returned a tool call, but no Tool module is installed")

        requested_name, requested_arguments, tool_id = registry.parse_model_call(
            tool_call, f"call_{state.round_number}_{index}"
        )
        tool_name, arguments = requested_name, dict(requested_arguments)

        result = None
        if services.runtime.hooks is not None:
            hook = services.runtime.hooks.before_tool_use({
                "tool_name": tool_name,
                "tool_args": dict(arguments),
                "tool_spec": registry.get_tool_spec(tool_name),
            })
            if hook.blocked:
                result = ToolResult.error(
                    hook.message or "Tool call blocked by hook",
                    error_type=hook.error_type,
                    metadata={"hookAudit": hook.audit},
                )
            else:
                tool_name = str(hook.payload.get("tool_name", tool_name))
                arguments = dict(hook.payload.get("tool_args", arguments))

        tool, validated = None, arguments
        if result is None:
            try:
                tool, validated = registry.validate_tool_call(tool_name, arguments)
                if services.permission_engine is not None:
                    if not isinstance(services.permission_engine, BasePermissionEngine):
                        raise TypeError("Permission module must implement BasePermissionEngine")
                    decision = services.permission_engine.authorize(
                        registry.build_permission_request(
                            tool_name,
                            validated,
                            context=services.authorization_context_factory(),
                        )
                    )
                    if decision.behavior == PermissionBehavior.DENY:
                        result = ToolResult.error(
                            decision.reason or f"Permission denied for tool '{tool_name}'.",
                            error_type="permission_denied",
                            metadata={"permission_behavior": decision.behavior.value,
                                      **dict(decision.metadata)},
                        )
                    elif decision.behavior == PermissionBehavior.ASK:
                        result = ToolResult.needs_confirmation(
                            decision.reason or f"Tool '{tool_name}' requires approval.",
                            metadata={"permission_behavior": decision.behavior.value,
                                      **dict(decision.metadata)},
                        )
            except Exception as exc:
                result = ToolResult.error(
                    str(exc), error_type=exc.__class__.__name__,
                    metadata={"tool_name": tool_name},
                )

        self._publish(services, RuntimeEventType.TOOL_INVOKE_STARTED, state, {
            "tool_name": tool_name,
            "arguments": validated,
            "tool_call_id": tool_id,
            "requested_tool_name": requested_name,
            "requested_arguments": requested_arguments,
        })

        if result is None:
            try:
                result = registry.normalize_tool_result(tool_name, tool.run(validated))
            except Exception as exc:
                result = ToolResult.error(str(exc), error_type=exc.__class__.__name__)

        if services.runtime.hooks is not None:
            hook = services.runtime.hooks.after_tool_use({
                "tool_name": tool_name,
                "tool_args": dict(validated),
                "tool_spec": registry.get_tool_spec(tool_name),
                "tool_permission": registry.get_permission_descriptor(tool_name),
                "tool_result": result,
            })
            if hook.blocked:
                result = ToolResult.error(
                    hook.message or "Tool result blocked by hook",
                    error_type=hook.error_type,
                    metadata={"hookAudit": hook.audit},
                )
            else:
                result = hook.payload.get("tool_result", result)
                if not isinstance(result, ToolResult):
                    raise TypeError("after_tool_use hook must keep tool_result as ToolResult")

        text = result.to_display_string()
        trace = services.llm.tool_result_to_canonical(text, tool_id, tool_name)
        event_type = (
            RuntimeEventType.TOOL_INVOKE_COMPLETED
            if result.status == "success"
            else RuntimeEventType.TOOL_INVOKE_FAILED
        )
        self._publish(services, event_type, state, {
            "tool_name": tool_name,
            "arguments": validated,
            "tool_call_id": tool_id,
            "output": text,
            "result": result,
            "trace": trace,
            "error_type": result.error_type,
        })
        if result.status == "needs_confirmation" and services.config.interrupt_on_confirmation:
            raise services.runtime.interruptions.create_interruption(
                tool_name=tool_name,
                tool_args=validated,
                tool_id=tool_id,
                round_number=state.round_number,
                tool_result=result,
            )
        services.context_manager.append_tool_result(text, tool_id, tool_name)
        services.context_manager.append_tool_context(tool_name, result.ephemeral_context)
        return tool_name, validated, tool_id, result

    async def aexecute_tool_call(
        self,
        services: AgentExecutionServices,
        state: AgentInvocationState,
        tool_call: Any,
        index: int = 0,
    ) -> tuple[str, dict[str, Any], str, ToolResult]:
        """Async tool phase using native async Hook, Permission and Tool interfaces."""
        registry = services.tool_registry
        if registry is None:
            raise RuntimeError("Model returned a tool call, but no Tool module is installed")

        requested_name, requested_arguments, tool_id = registry.parse_model_call(
            tool_call, f"call_{state.round_number}_{index}"
        )
        tool_name, arguments = requested_name, dict(requested_arguments)

        result = None
        if services.runtime.hooks is not None:
            hook = await services.runtime.hooks.arun_stage("before_tool_use", {
                "tool_name": tool_name,
                "tool_args": dict(arguments),
                "tool_spec": registry.get_tool_spec(tool_name),
            })
            if hook.blocked:
                result = ToolResult.error(
                    hook.message or "Tool call blocked by hook",
                    error_type=hook.error_type,
                    metadata={"hookAudit": hook.audit},
                )
            else:
                tool_name = str(hook.payload.get("tool_name", tool_name))
                arguments = dict(hook.payload.get("tool_args", arguments))

        tool, validated = None, arguments
        if result is None:
            try:
                tool, validated = registry.validate_tool_call(tool_name, arguments)
                if services.permission_engine is not None:
                    if not isinstance(services.permission_engine, BasePermissionEngine):
                        raise TypeError("Permission module must implement BasePermissionEngine")
                    decision = await services.permission_engine.aauthorize(
                        registry.build_permission_request(
                            tool_name,
                            validated,
                            context=services.authorization_context_factory(),
                        )
                    )
                    if decision.behavior == PermissionBehavior.DENY:
                        result = ToolResult.error(
                            decision.reason or f"Permission denied for tool '{tool_name}'.",
                            error_type="permission_denied",
                            metadata={"permission_behavior": decision.behavior.value,
                                      **dict(decision.metadata)},
                        )
                    elif decision.behavior == PermissionBehavior.ASK:
                        result = ToolResult.needs_confirmation(
                            decision.reason or f"Tool '{tool_name}' requires approval.",
                            metadata={"permission_behavior": decision.behavior.value,
                                      **dict(decision.metadata)},
                        )
            except Exception as exc:
                result = ToolResult.error(
                    str(exc), error_type=exc.__class__.__name__,
                    metadata={"tool_name": tool_name},
                )

        self._publish(services, RuntimeEventType.TOOL_INVOKE_STARTED, state, {
            "tool_name": tool_name,
            "arguments": validated,
            "tool_call_id": tool_id,
            "requested_tool_name": requested_name,
            "requested_arguments": requested_arguments,
        })

        if result is None:
            try:
                result = registry.normalize_tool_result(
                    tool_name, await tool.arun(validated)
                )
            except Exception as exc:
                result = ToolResult.error(str(exc), error_type=exc.__class__.__name__)

        if services.runtime.hooks is not None:
            hook = await services.runtime.hooks.arun_stage("after_tool_use", {
                "tool_name": tool_name,
                "tool_args": dict(validated),
                "tool_spec": registry.get_tool_spec(tool_name),
                "tool_permission": registry.get_permission_descriptor(tool_name),
                "tool_result": result,
            })
            if hook.blocked:
                result = ToolResult.error(
                    hook.message or "Tool result blocked by hook",
                    error_type=hook.error_type,
                    metadata={"hookAudit": hook.audit},
                )
            else:
                result = hook.payload.get("tool_result", result)
                if not isinstance(result, ToolResult):
                    raise TypeError("after_tool_use hook must keep tool_result as ToolResult")

        text = result.to_display_string()
        trace = services.llm.tool_result_to_canonical(text, tool_id, tool_name)
        event_type = (
            RuntimeEventType.TOOL_INVOKE_COMPLETED
            if result.status == "success"
            else RuntimeEventType.TOOL_INVOKE_FAILED
        )
        self._publish(services, event_type, state, {
            "tool_name": tool_name,
            "arguments": validated,
            "tool_call_id": tool_id,
            "output": text,
            "result": result,
            "trace": trace,
            "error_type": result.error_type,
        })
        if result.status == "needs_confirmation" and services.config.interrupt_on_confirmation:
            raise services.runtime.interruptions.create_interruption(
                tool_name=tool_name,
                tool_args=validated,
                tool_id=tool_id,
                round_number=state.round_number,
                tool_result=result,
            )
        services.context_manager.append_tool_result(text, tool_id, tool_name)
        services.context_manager.append_tool_context(tool_name, result.ephemeral_context)
        return tool_name, validated, tool_id, result

    def _start(self, services: AgentExecutionServices, query: str, *, stream: bool) -> AgentInvocationState:
        state = services.runtime.begin_invocation(query, stream=stream)
        try:
            self.last_state = state
            services.context_manager.append_query(query)
            self._publish(
                services,
                RuntimeEventType.AGENT_INVOKE_STARTED,
                state,
                {
                    "query": query,
                    "mode": services.runtime.execution_mode,
                    "stream": stream,
                },
            )
            return state
        except BaseException:
            self._cleanup(services)
            raise

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
            services.context_manager.append_response(
                response,
                include_reasoning=bool(services.config.persist_reasoning_history),
            )
        else:
            services.context_manager.append_assistant(
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

    def _fail(self, services: AgentExecutionServices, state: AgentInvocationState, exc: BaseException) -> None:
        if state.phase == AgentInvocationPhase.COMPLETED:
            # Closing a stream after its final event is normal consumption.
            return
        state.error = exc
        import asyncio
        interrupted = isinstance(exc, (ToolInterruption, AgentStopRequested, asyncio.CancelledError, GeneratorExit, KeyboardInterrupt))
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
        try:
            if services.tool_registry is not None:
                services.tool_registry.clear_deferred_tool_expansions()
        finally:
            services.runtime.end_invocation()

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
                services.before_context_build()
                scope = RuntimeScope(services.runtime.session_id, services.runtime.workspace_root)
                if services.runtime.hooks is not None:
                    hook = services.runtime.hooks.before_context({
                        "query": state.query,
                        "history": services.context_manager.canonical,
                        "scope": scope,
                        "invocation_id": state.invocation_id,
                        "request_id": f"{state.invocation_id}:{state.round_number}",
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Context preparation blocked",
                            stage="before_context", error_type=hook.error_type,
                        )
                services.context_manager.process_meta_messages()
                prompt_blocks = services.prompt_composer.compose(
                    services.prompt_context_factory(state.query)
                )
                state.phase = AgentInvocationPhase.COMPILING
                tools = self._tool_payload(services)
                options = {
                    "temperature": temperature,
                    "reasoning": services.runtime.reasoning,
                    **kwargs,
                }
                request_input = services.context_manager.build_input(
                    prompt_blocks, tools=tools, options=options, query=state.query,
                    cache_policy=services.config.cache_policy,
                )
                request_input = _expect_context_result(
                    "build_input", request_input, ReplayRequestInput,
                )
                context_usage = services.context_manager.calculate_usage(
                    request_input, tools=tools, options=options,
                )
                context_usage = _expect_context_result(
                    "calculate_usage", context_usage, ContextUsage,
                )
                compaction = None
                compact_required = _expect_context_result(
                    "should_compact",
                    services.context_manager.should_compact(context_usage), bool,
                )
                if compact_required:
                    canonical_history = services.context_manager.canonical
                    replay_history = services.context_manager.replay
                    max_tokens = context_usage.max_tokens
                    if services.runtime.hooks is not None:
                        hook = services.runtime.hooks.before_compaction({
                            "canonical_history": canonical_history,
                            "replay_history": replay_history,
                            "max_tokens": max_tokens,
                            "estimated_tokens": context_usage.estimated_tokens,
                            "token_estimate_source": context_usage.source,
                            "token_estimate": context_usage.metadata,
                            "invocation_id": state.invocation_id,
                        })
                        if hook.blocked:
                            raise HookExecutionError(
                                hook.message or "Request blocked at before_compaction",
                                stage="before_compaction", error_type=hook.error_type,
                                metadata={"audit": hook.audit},
                            )
                        canonical_history = hook.payload.get("canonical_history", canonical_history)
                        replay_history = hook.payload.get("replay_history", replay_history)
                        max_tokens = hook.payload.get("max_tokens", max_tokens)
                    compaction = services.context_manager.compact_context(
                        request_input, context_usage,
                        canonical_history=canonical_history,
                        replay_history=replay_history,
                        tools=tools, options=options, max_tokens=max_tokens,
                    )
                    compaction = _expect_context_result(
                        "compact_context", compaction, ContextCompactionResult,
                    )
                    if compaction.was_compacted:
                        request_input = services.context_manager.build_input(
                            prompt_blocks, tools=tools, options=options,
                            query=state.query,
                            cache_policy=services.config.cache_policy,
                            replay_history=compaction.replay_history,
                        )
                        request_input = _expect_context_result(
                            "build_input", request_input, ReplayRequestInput,
                        )
                if services.runtime.hooks is not None:
                    hook = services.runtime.hooks.before_llm_request({
                        "request_input": request_input,
                        "tools": tools,
                        "options": dict(options),
                        "invocation_id": state.invocation_id,
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Request blocked at before_llm_request",
                            stage="before_llm_request", error_type=hook.error_type,
                            metadata={"audit": hook.audit},
                        )
                    request_input = hook.payload.get("request_input", request_input)
                    tools = hook.payload.get("tools", tools)
                    options = dict(hook.payload.get("options", options))
                services.context_manager.validate_input(
                    request_input, tools=tools, options=options,
                    history_epoch=services.context_manager.history_epoch + int(
                        compaction is not None and compaction.was_compacted
                    ),
                )
                self._check_stop(services)
                if compaction is not None and compaction.was_compacted:
                    services.context_manager.commit_compaction(compaction)
                    self._publish_compaction(services, state, compaction)
                services.context_manager.observe_prepared_input(
                    request_input, tools=tools, options=options, query=state.query,
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
                        {"llm_invoke_id": llm_id, **self._error_payload(exc)},
                    )
                    raise
                output_messages = services.llm.response_to_canonical(response, include_reasoning=True)
                usage = services.llm.extract_usage_metrics(response)
                response_replay = deepcopy(services.llm.response_to_replay(
                    response, include_reasoning=bool(services.config.persist_reasoning_history),
                ))
                services.context_manager.observe_model_result(usage, response_replay,
                    use_output=self._usage_output_is_retained(services, output_messages))
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "response": response,
                        "output": output_messages,
                        "usage": usage,
                    },
                )
                response = self._apply_after_llm_hook(services, response)
                retained_replay = services.llm.response_to_replay(
                    response, include_reasoning=bool(services.config.persist_reasoning_history),
                )
                if retained_replay != response_replay:
                    services.context_manager.observe_model_result(usage, retained_replay, use_output=False)
                if tools and services.llm.has_tool_calls(response):
                    services.context_manager.append_response(
                        response,
                        include_reasoning=bool(services.config.persist_reasoning_history),
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(services.llm.get_tool_calls(response)):
                        self.execute_tool_call(services, state, tool_call, index)
                    continue
                output = services.llm.get_response_content(response) or ""
                return self._finish(services, state, output, response=response)
            raise RuntimeError(f"Agent reached max_iter={max_iter} without a final response")
        except Exception as exc:
            self._fail(services, state, exc)
            raise
        except BaseException as exc:
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
                services.before_context_build()
                scope = RuntimeScope(services.runtime.session_id, services.runtime.workspace_root)
                if services.runtime.hooks is not None:
                    hook = await services.runtime.hooks.arun_stage("before_context", {
                        "query": state.query,
                        "history": services.context_manager.canonical,
                        "scope": scope,
                        "invocation_id": state.invocation_id,
                        "request_id": f"{state.invocation_id}:{state.round_number}",
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Context preparation blocked",
                            stage="before_context", error_type=hook.error_type,
                        )
                await services.context_manager.aprocess_meta_messages()
                prompt_blocks = services.prompt_composer.compose(
                    services.prompt_context_factory(state.query)
                )
                state.phase = AgentInvocationPhase.COMPILING
                tools = self._tool_payload(services)
                options = {
                    "temperature": temperature,
                    "reasoning": services.runtime.reasoning,
                    **kwargs,
                }
                request_input = await services.context_manager.abuild_input(
                    prompt_blocks, tools=tools, options=options, query=state.query,
                    cache_policy=services.config.cache_policy,
                )
                request_input = _expect_context_result(
                    "abuild_input", request_input, ReplayRequestInput,
                )
                context_usage = await services.context_manager.acalculate_usage(
                    request_input, tools=tools, options=options,
                )
                context_usage = _expect_context_result(
                    "acalculate_usage", context_usage, ContextUsage,
                )
                compaction = None
                compact_required = _expect_context_result(
                    "should_compact",
                    services.context_manager.should_compact(context_usage), bool,
                )
                if compact_required:
                    canonical_history = services.context_manager.canonical
                    replay_history = services.context_manager.replay
                    max_tokens = context_usage.max_tokens
                    if services.runtime.hooks is not None:
                        hook = await services.runtime.hooks.arun_stage("before_compaction", {
                            "canonical_history": canonical_history,
                            "replay_history": replay_history,
                            "max_tokens": max_tokens,
                            "estimated_tokens": context_usage.estimated_tokens,
                            "token_estimate_source": context_usage.source,
                            "token_estimate": context_usage.metadata,
                            "invocation_id": state.invocation_id,
                        })
                        if hook.blocked:
                            raise HookExecutionError(
                                hook.message or "Request blocked at before_compaction",
                                stage="before_compaction", error_type=hook.error_type,
                                metadata={"audit": hook.audit},
                            )
                        canonical_history = hook.payload.get("canonical_history", canonical_history)
                        replay_history = hook.payload.get("replay_history", replay_history)
                        max_tokens = hook.payload.get("max_tokens", max_tokens)
                    compaction = await services.context_manager.acompact_context(
                        request_input, context_usage,
                        canonical_history=canonical_history,
                        replay_history=replay_history,
                        tools=tools, options=options, max_tokens=max_tokens,
                    )
                    compaction = _expect_context_result(
                        "acompact_context", compaction, ContextCompactionResult,
                    )
                    if compaction.was_compacted:
                        request_input = await services.context_manager.abuild_input(
                            prompt_blocks, tools=tools, options=options,
                            query=state.query,
                            cache_policy=services.config.cache_policy,
                            replay_history=compaction.replay_history,
                        )
                        request_input = _expect_context_result(
                            "abuild_input", request_input, ReplayRequestInput,
                        )
                if services.runtime.hooks is not None:
                    hook = await services.runtime.hooks.arun_stage("before_llm_request", {
                        "request_input": request_input,
                        "tools": tools,
                        "options": dict(options),
                        "invocation_id": state.invocation_id,
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Request blocked at before_llm_request",
                            stage="before_llm_request", error_type=hook.error_type,
                            metadata={"audit": hook.audit},
                        )
                    request_input = hook.payload.get("request_input", request_input)
                    tools = hook.payload.get("tools", tools)
                    options = dict(hook.payload.get("options", options))
                await services.context_manager.avalidate_input(
                    request_input, tools=tools, options=options,
                    history_epoch=services.context_manager.history_epoch + int(
                        compaction is not None and compaction.was_compacted
                    ),
                )
                self._check_stop(services)
                if compaction is not None and compaction.was_compacted:
                    services.context_manager.commit_compaction(compaction)
                    self._publish_compaction(services, state, compaction)
                await services.context_manager.aobserve_prepared_input(
                    request_input, tools=tools, options=options, query=state.query,
                )
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
                        {"llm_invoke_id": llm_id, **self._error_payload(exc)},
                    )
                    raise
                output_messages = services.llm.response_to_canonical(response, include_reasoning=True)
                usage = services.llm.extract_usage_metrics(response)
                response_replay = deepcopy(services.llm.response_to_replay(
                    response, include_reasoning=bool(services.config.persist_reasoning_history),
                ))
                services.context_manager.observe_model_result(usage, response_replay,
                    use_output=self._usage_output_is_retained(services, output_messages))
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "response": response,
                        "output": output_messages,
                        "usage": usage,
                    },
                )
                response = self._apply_after_llm_hook(services, response)
                retained_replay = services.llm.response_to_replay(
                    response, include_reasoning=bool(services.config.persist_reasoning_history),
                )
                if retained_replay != response_replay:
                    services.context_manager.observe_model_result(usage, retained_replay, use_output=False)
                if tools and services.llm.has_tool_calls(response):
                    services.context_manager.append_response(
                        response,
                        include_reasoning=bool(services.config.persist_reasoning_history),
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(services.llm.get_tool_calls(response)):
                        await self.aexecute_tool_call(services, state, tool_call, index)
                    continue
                output = services.llm.get_response_content(response) or ""
                return self._finish(services, state, output, response=response)
            raise RuntimeError(f"Agent reached max_iter={max_iter} without a final response")
        except Exception as exc:
            self._fail(services, state, exc)
            raise
        except BaseException as exc:
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
                services.before_context_build()
                scope = RuntimeScope(services.runtime.session_id, services.runtime.workspace_root)
                if services.runtime.hooks is not None:
                    hook = services.runtime.hooks.before_context({
                        "query": state.query,
                        "history": services.context_manager.canonical,
                        "scope": scope,
                        "invocation_id": state.invocation_id,
                        "request_id": f"{state.invocation_id}:{state.round_number}",
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Context preparation blocked",
                            stage="before_context", error_type=hook.error_type,
                        )
                services.context_manager.process_meta_messages()
                prompt_blocks = services.prompt_composer.compose(
                    services.prompt_context_factory(state.query)
                )
                state.phase = AgentInvocationPhase.COMPILING
                tools = self._tool_payload(services)
                options = {
                    "temperature": temperature,
                    "reasoning": services.runtime.reasoning,
                    **kwargs,
                }
                request_input = services.context_manager.build_input(
                    prompt_blocks, tools=tools, options=options, query=state.query,
                    cache_policy=services.config.cache_policy,
                )
                request_input = _expect_context_result(
                    "build_input", request_input, ReplayRequestInput,
                )
                context_usage = services.context_manager.calculate_usage(
                    request_input, tools=tools, options=options,
                )
                context_usage = _expect_context_result(
                    "calculate_usage", context_usage, ContextUsage,
                )
                compaction = None
                compact_required = _expect_context_result(
                    "should_compact",
                    services.context_manager.should_compact(context_usage), bool,
                )
                if compact_required:
                    canonical_history = services.context_manager.canonical
                    replay_history = services.context_manager.replay
                    max_tokens = context_usage.max_tokens
                    if services.runtime.hooks is not None:
                        hook = services.runtime.hooks.before_compaction({
                            "canonical_history": canonical_history,
                            "replay_history": replay_history,
                            "max_tokens": max_tokens,
                            "estimated_tokens": context_usage.estimated_tokens,
                            "token_estimate_source": context_usage.source,
                            "token_estimate": context_usage.metadata,
                            "invocation_id": state.invocation_id,
                        })
                        if hook.blocked:
                            raise HookExecutionError(
                                hook.message or "Request blocked at before_compaction",
                                stage="before_compaction", error_type=hook.error_type,
                                metadata={"audit": hook.audit},
                            )
                        canonical_history = hook.payload.get("canonical_history", canonical_history)
                        replay_history = hook.payload.get("replay_history", replay_history)
                        max_tokens = hook.payload.get("max_tokens", max_tokens)
                    compaction = services.context_manager.compact_context(
                        request_input, context_usage,
                        canonical_history=canonical_history,
                        replay_history=replay_history,
                        tools=tools, options=options, max_tokens=max_tokens,
                    )
                    compaction = _expect_context_result(
                        "compact_context", compaction, ContextCompactionResult,
                    )
                    if compaction.was_compacted:
                        request_input = services.context_manager.build_input(
                            prompt_blocks, tools=tools, options=options,
                            query=state.query,
                            cache_policy=services.config.cache_policy,
                            replay_history=compaction.replay_history,
                        )
                        request_input = _expect_context_result(
                            "build_input", request_input, ReplayRequestInput,
                        )
                if services.runtime.hooks is not None:
                    hook = services.runtime.hooks.before_llm_request({
                        "request_input": request_input,
                        "tools": tools,
                        "options": dict(options),
                        "invocation_id": state.invocation_id,
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Request blocked at before_llm_request",
                            stage="before_llm_request", error_type=hook.error_type,
                            metadata={"audit": hook.audit},
                        )
                    request_input = hook.payload.get("request_input", request_input)
                    tools = hook.payload.get("tools", tools)
                    options = dict(hook.payload.get("options", options))
                services.context_manager.validate_input(
                    request_input, tools=tools, options=options,
                    history_epoch=services.context_manager.history_epoch + int(
                        compaction is not None and compaction.was_compacted
                    ),
                )
                self._check_stop(services)
                if compaction is not None and compaction.was_compacted:
                    services.context_manager.commit_compaction(compaction)
                    self._publish_compaction(services, state, compaction)
                services.context_manager.observe_prepared_input(
                    request_input, tools=tools, options=options, query=state.query,
                )
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
                            terminal = self._select_terminal_event(terminal, provider_event)
                except Exception as exc:
                    self._publish(
                        services,
                        RuntimeEventType.LLM_INVOKE_FAILED,
                        state,
                        {"llm_invoke_id": llm_id, **self._error_payload(exc)},
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
                usage = services.llm.extract_usage_metrics({"usage": terminal.get("usage")})
                response_replay = deepcopy(services.llm.assistant_message_to_replay(
                    content=content, tool_calls=tool_calls or None,
                    thinking=thinking if services.config.persist_reasoning_history else None,
                ))
                services.context_manager.observe_model_result(usage, response_replay,
                    use_output=self._usage_output_is_retained(services, canonical_output))
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "output": canonical_output,
                        "usage": usage,
                    },
                )
                terminal = self._apply_after_llm_hook(services, terminal)
                if not isinstance(terminal, dict):
                    raise TypeError("after_llm_response hook must keep a streamed response as a dict")
                content = str(terminal.get("content") or content)
                thinking = str(terminal.get("thinking") or thinking)
                tool_calls = list(terminal.get("tool_calls") or tool_calls)
                retained_replay = services.llm.assistant_message_to_replay(
                    content=content, tool_calls=tool_calls or None,
                    thinking=thinking if services.config.persist_reasoning_history else None,
                )
                if retained_replay != response_replay:
                    services.context_manager.observe_model_result(usage, retained_replay, use_output=False)
                if tool_calls:
                    services.context_manager.append_assistant(
                        content=content,
                        tool_calls=tool_calls,
                        thinking=thinking if services.config.persist_reasoning_history else None,
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(tool_calls):
                        tool_name, arguments, tool_id = services.tool_registry.parse_model_call(tool_call, f"call_{round_number}_{index}")
                        sequence += 1
                        yield self._stream_event(
                            services,
                            state,
                            AgentStreamEventType.TOOL_CALL,
                            sequence,
                            data={"tool_name": tool_name, "arguments": arguments, "tool_call_id": tool_id},
                        )
                        executed_name, executed_arguments, executed_id, result = self.execute_tool_call(
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
                    **self._error_payload(exc),
                    "interrupted": isinstance(exc, (ToolInterruption, AgentStopRequested)),
                },
            )
            raise
        except BaseException as exc:
            self._fail(services, state, exc)
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
                services.before_context_build()
                scope = RuntimeScope(services.runtime.session_id, services.runtime.workspace_root)
                if services.runtime.hooks is not None:
                    hook = await services.runtime.hooks.arun_stage("before_context", {
                        "query": state.query,
                        "history": services.context_manager.canonical,
                        "scope": scope,
                        "invocation_id": state.invocation_id,
                        "request_id": f"{state.invocation_id}:{state.round_number}",
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Context preparation blocked",
                            stage="before_context", error_type=hook.error_type,
                        )
                await services.context_manager.aprocess_meta_messages()
                prompt_blocks = services.prompt_composer.compose(
                    services.prompt_context_factory(state.query)
                )
                state.phase = AgentInvocationPhase.COMPILING
                tools = self._tool_payload(services)
                options = {
                    "temperature": temperature,
                    "reasoning": services.runtime.reasoning,
                    **kwargs,
                }
                request_input = await services.context_manager.abuild_input(
                    prompt_blocks, tools=tools, options=options, query=state.query,
                    cache_policy=services.config.cache_policy,
                )
                request_input = _expect_context_result(
                    "abuild_input", request_input, ReplayRequestInput,
                )
                context_usage = await services.context_manager.acalculate_usage(
                    request_input, tools=tools, options=options,
                )
                context_usage = _expect_context_result(
                    "acalculate_usage", context_usage, ContextUsage,
                )
                compaction = None
                compact_required = _expect_context_result(
                    "should_compact",
                    services.context_manager.should_compact(context_usage), bool,
                )
                if compact_required:
                    canonical_history = services.context_manager.canonical
                    replay_history = services.context_manager.replay
                    max_tokens = context_usage.max_tokens
                    if services.runtime.hooks is not None:
                        hook = await services.runtime.hooks.arun_stage("before_compaction", {
                            "canonical_history": canonical_history,
                            "replay_history": replay_history,
                            "max_tokens": max_tokens,
                            "estimated_tokens": context_usage.estimated_tokens,
                            "token_estimate_source": context_usage.source,
                            "token_estimate": context_usage.metadata,
                            "invocation_id": state.invocation_id,
                        })
                        if hook.blocked:
                            raise HookExecutionError(
                                hook.message or "Request blocked at before_compaction",
                                stage="before_compaction", error_type=hook.error_type,
                                metadata={"audit": hook.audit},
                            )
                        canonical_history = hook.payload.get("canonical_history", canonical_history)
                        replay_history = hook.payload.get("replay_history", replay_history)
                        max_tokens = hook.payload.get("max_tokens", max_tokens)
                    compaction = await services.context_manager.acompact_context(
                        request_input, context_usage,
                        canonical_history=canonical_history,
                        replay_history=replay_history,
                        tools=tools, options=options, max_tokens=max_tokens,
                    )
                    compaction = _expect_context_result(
                        "acompact_context", compaction, ContextCompactionResult,
                    )
                    if compaction.was_compacted:
                        request_input = await services.context_manager.abuild_input(
                            prompt_blocks, tools=tools, options=options,
                            query=state.query,
                            cache_policy=services.config.cache_policy,
                            replay_history=compaction.replay_history,
                        )
                        request_input = _expect_context_result(
                            "abuild_input", request_input, ReplayRequestInput,
                        )
                if services.runtime.hooks is not None:
                    hook = await services.runtime.hooks.arun_stage("before_llm_request", {
                        "request_input": request_input,
                        "tools": tools,
                        "options": dict(options),
                        "invocation_id": state.invocation_id,
                        "round_number": state.round_number,
                    })
                    if hook.blocked:
                        raise HookExecutionError(
                            hook.message or "Request blocked at before_llm_request",
                            stage="before_llm_request", error_type=hook.error_type,
                            metadata={"audit": hook.audit},
                        )
                    request_input = hook.payload.get("request_input", request_input)
                    tools = hook.payload.get("tools", tools)
                    options = dict(hook.payload.get("options", options))
                await services.context_manager.avalidate_input(
                    request_input, tools=tools, options=options,
                    history_epoch=services.context_manager.history_epoch + int(
                        compaction is not None and compaction.was_compacted
                    ),
                )
                self._check_stop(services)
                if compaction is not None and compaction.was_compacted:
                    services.context_manager.commit_compaction(compaction)
                    self._publish_compaction(services, state, compaction)
                await services.context_manager.aobserve_prepared_input(
                    request_input, tools=tools, options=options, query=state.query,
                )
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
                            terminal = self._select_terminal_event(terminal, provider_event)
                except Exception as exc:
                    self._publish(
                        services,
                        RuntimeEventType.LLM_INVOKE_FAILED,
                        state,
                        {"llm_invoke_id": llm_id, **self._error_payload(exc)},
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
                usage = services.llm.extract_usage_metrics({"usage": terminal.get("usage")})
                response_replay = deepcopy(services.llm.assistant_message_to_replay(
                    content=content, tool_calls=tool_calls or None,
                    thinking=thinking if services.config.persist_reasoning_history else None,
                ))
                services.context_manager.observe_model_result(usage, response_replay,
                    use_output=self._usage_output_is_retained(services, canonical_output))
                self._publish(
                    services,
                    RuntimeEventType.LLM_INVOKE_COMPLETED,
                    state,
                    {
                        "llm_invoke_id": llm_id,
                        "output": canonical_output,
                        "usage": usage,
                    },
                )
                terminal = self._apply_after_llm_hook(services, terminal)
                if not isinstance(terminal, dict):
                    raise TypeError("after_llm_response hook must keep a streamed response as a dict")
                content = str(terminal.get("content") or content)
                thinking = str(terminal.get("thinking") or thinking)
                tool_calls = list(terminal.get("tool_calls") or tool_calls)
                retained_replay = services.llm.assistant_message_to_replay(
                    content=content, tool_calls=tool_calls or None,
                    thinking=thinking if services.config.persist_reasoning_history else None,
                )
                if retained_replay != response_replay:
                    services.context_manager.observe_model_result(usage, retained_replay, use_output=False)
                if tool_calls:
                    services.context_manager.append_assistant(
                        content=content,
                        tool_calls=tool_calls,
                        thinking=thinking if services.config.persist_reasoning_history else None,
                    )
                    state.phase = AgentInvocationPhase.TOOL
                    for index, tool_call in enumerate(tool_calls):
                        tool_name, arguments, tool_id = services.tool_registry.parse_model_call(tool_call, f"call_{round_number}_{index}")
                        sequence += 1
                        yield self._stream_event(
                            services,
                            state,
                            AgentStreamEventType.TOOL_CALL,
                            sequence,
                            data={"tool_name": tool_name, "arguments": arguments, "tool_call_id": tool_id},
                        )
                        executed_name, executed_arguments, executed_id, result = await self.aexecute_tool_call(
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
                    **self._error_payload(exc),
                    "interrupted": isinstance(exc, (ToolInterruption, AgentStopRequested)),
                },
            )
            raise
        except BaseException as exc:
            self._fail(services, state, exc)
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
