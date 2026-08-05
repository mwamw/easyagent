"""Tool loop engine interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, AsyncGenerator, Optional
from agent import BasicAgent
from core.Exception import (
    LLMInvokeError,
    ToolExecutionError,
    ToolInterruption,
    ToolRegistryError,
    AgentStopRequested,
)
from core.request_input import ReplayRequestInput

logger = logging.getLogger(__name__)


class BaseToolLoopEngine(ABC):
    """Abstract engine for tool-calling loops."""

    @abstractmethod
    def invoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def ainvoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def astream_invoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if False: yield {} 
        pass


class DefaultToolLoopEngine(BaseToolLoopEngine):
    """Default tool loop engine that preserves current BasicAgent behavior."""

    @staticmethod
    def _extract_tool_id(tool_call: Any, fallback: str = "unknown") -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("call_id") or tool_call.get("id") or fallback)
        return str(
            getattr(tool_call, "call_id", None)
            or getattr(tool_call, "id", None)
            or fallback
        )

    @staticmethod
    def _normalize_stream_assistant_replay(assistant_items: Any) -> list[Any]:
        if assistant_items is None:
            return []
        if isinstance(assistant_items, list):
            return list(assistant_items)
        if isinstance(assistant_items, dict):
            return [assistant_items]
        raise TypeError(
            f"流式 assistant_items 必须是 dict 或 list[dict]，实际收到: {type(assistant_items).__name__}"
        )

    @staticmethod
    def _inject_tool_ephemeral_context(
        agent: BasicAgent,
        *,
        tool_name: str,
        tool_result: Any,
        messages: ReplayRequestInput,
        ephemeral_replay: list[Any],
    ) -> None:
        agent._maybe_inject_tool_ephemeral_context(
            tool_name=tool_name,
            tool_result=tool_result,
            messages=messages, # type: ignore
        )
        agent._maybe_inject_tool_ephemeral_context(
            tool_name=tool_name,
            tool_result=tool_result,
            messages=ephemeral_replay,
        )

    @staticmethod
    def _inject_runtime_skill_context(
        agent: BasicAgent,
        *,
        tool_name: str,
        messages: ReplayRequestInput,
        ephemeral_replay: list[Any],
    ) -> None:
        agent._maybe_inject_runtime_skill_context(
            tool_name=tool_name,
            messages=messages, # type: ignore
        )
        agent._maybe_inject_runtime_skill_context(
            tool_name=tool_name,
            messages=ephemeral_replay,
        )

    def invoke(
        self,
        agent:BasicAgent,
        query: str,
        # messages: ReplayRequestInput,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> str:
        resume_from_history = bool(kwargs.pop("resume_from_history", False))
        agent.enable_tool = True
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()
        agent._raise_if_stop_requested()

        if agent.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")

        raw_query = trace_query if trace_query is not None else (query or "[resume_pending_tool_interrupt]")
        if not resume_from_history:
            agent._append_query_history(query)
        agent.compact_persistent_history_if_needed()
        ephemeral_replay: list[Any] = []
        messages = agent._build_start_messages(
            query,
            include_query=False,
            extra_replay_entries=ephemeral_replay,
        )
        needs_rebuild = False
        final_response: Optional[str] = None
        response: Any = None
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        iteration_count = 0
        agent_run_id = agent._observe_agent_run_start(
            query if not resume_from_history else "[resume_pending_tool_interrupt]",
            mode="tool",
            stream=False,
            metadata={
                "entrypoint": "tool_loop_engine.invoke",
                "resumed": resume_from_history,
            },
        )

        try:
            while max_iter > 0:
                agent._raise_if_stop_requested()
                iteration_count += 1
                logger.debug(f"工具调用迭代 {iteration_count}")

                try:
                    if needs_rebuild:
                        messages = agent._build_start_messages(
                            query,
                            include_query=False,
                            extra_replay_entries=ephemeral_replay,
                        )
                        needs_rebuild = False
                    messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                        messages,
                        request_kind="tool_invoke",
                        temperature=temperature,
                        reasoning=agent.reasoning,
                        stream=False,
                        tools_enabled=True,
                        kwargs=kwargs,
                    )
                    llm_observation_id = agent._observe_llm_request_start(
                        turn_id=turn_id,
                        request_kind="tool_invoke",
                        messages=messages,
                        reasoning=request_reasoning,
                        stream=False,
                        tools_enabled=True,
                        metadata={"round": iteration_count},
                    )
                    agent.callback_manager.on_llm_start(messages)
                    try:
                        response = agent.llm.invoke_with_tools(
                            messages,
                            agent.get_provider_tools(),
                            temperature=request_temperature,
                            reasoning=request_reasoning,
                            **llm_kwargs,
                        )
                    except Exception as exc:
                        agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                        raise
                    try:
                        response = agent._run_after_llm_response(
                            response,
                            messages=messages,
                            request_kind="tool_invoke",
                            stream=False,
                            tools_enabled=True,
                            hook_audit=llm_hook_audit,
                        )
                    except Exception as exc:
                        agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                        raise
                    agent._observe_llm_request_end(
                        llm_observation_id,
                        response=response,
                        success=True,
                        final_text=agent.llm.get_response_content(response),
                        final_thinking=agent.llm.get_thinking_content(response),
                        metadata={"round": iteration_count},
                    )
                    agent.callback_manager.on_llm_end(response)
                    agent._raise_if_stop_requested()

                    if response is None:
                        raise LLMInvokeError("LLM 返回了空响应!")
                except LLMInvokeError:
                    raise
                except AgentStopRequested:
                    raise
                except Exception as e:
                    logger.error(f"智能体调用失败: {str(e)[:500]}")
                    raise ToolExecutionError(f"智能体调用失败: {str(e)[:500]}") from e

                thinking_content = agent.llm.get_thinking_content(response)
                logger.info(f"思考内容: {thinking_content}")
                reasoning_event_id: Optional[str] = None
                if thinking_content:
                    reasoning_event_id = agent._set_round_reasoning(
                        thinking_content,
                        turn_id=turn_id,
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                    )

                if agent.llm.has_tool_calls(response):
                    response_canonical = agent.llm.response_to_canonical(
                        response,
                        include_reasoning=True,
                    )
                    response_replay = agent.llm.response_to_replay(
                        response,
                        include_reasoning=True,
                    )
                    messages.extend_replay(response_replay)
                    agent._set_pending_step_state(
                        assistant_canonical=response_canonical,
                        assistant_replay=response_replay,
                        tool_calls=agent.llm.get_tool_calls(response),
                        round_number=iteration_count,
                    )
                    assistant_parent_id = (
                        reasoning_event_id
                        or agent._get_last_turn_event_id(turn_id, exclude_types={"turn_end"})
                        or turn_root_event_id
                    )
                    assistant_event_id = agent._record_assistant_trace(
                        turn_id,
                        agent.llm.get_response_content(response),
                        parent_id=assistant_parent_id,
                        stage="pre_tool",
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                        allow_empty=True,
                    )

                    for tool_call in agent.llm.get_tool_calls(response):
                        agent._raise_if_stop_requested()
                        tool_name = "unknown_tool"
                        tool_args: dict[str, Any] = {}
                        tool_id = self._extract_tool_id(tool_call)

                        try:
                            tool_name = agent._safe_get_tool_name(tool_call)
                        except Exception as e:
                            logger.warning(f"获取工具名称失败: {e}，使用默认名称")
                            if hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                                tool_name = tool_call.function.name or "unknown_tool"
                            elif hasattr(tool_call, "name"):
                                tool_name = tool_call.name or "unknown_tool"

                        try:
                            tool_args = agent._safe_parse_tool_args(tool_call)
                            logger.info(f"{agent.name}执行工具: {tool_name}，参数: {tool_args}")
                            tool_call_event_id = agent._record_tool_call(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                            )
                            tool_result_obj = agent._safe_execute_tool_result(
                                tool_name,
                                tool_args,
                                turn_id=turn_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                            )
                            tool_result = tool_result_obj.to_display_string()
                            tool_canonical = agent.llm.tool_result_to_canonical(tool_result, tool_id, tool_name)
                            tool_replay = agent.llm.tool_result_to_replay(tool_result, tool_id, tool_name)
                            if (
                                tool_result_obj.status == "needs_confirmation"
                                and agent.config.interrupt_on_confirmation
                            ):
                                interrupt_error = agent._finalize_tool_interrupt(
                                    turn_id=turn_id,
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tool_id=tool_id,
                                    round_number=iteration_count,
                                    tool_result=tool_result_obj,
                                    parent_id=tool_call_event_id,
                                    mode="tool",
                                    stream=False,
                                    tool_canonical=tool_canonical,
                                    tool_replay=tool_replay,
                                )
                                raise interrupt_error
                            agent._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                tool_result,
                                parent_id=tool_call_event_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=tool_result_obj.status == "success",
                                tool_result_obj=tool_result_obj,
                            )
                            agent._append_pending_tool_result(
                                tool_canonical=tool_canonical,
                                tool_replay=tool_replay,
                                ephemeral_context=tool_result_obj.ephemeral_context,
                                tool_name=tool_name,
                            )
                            messages.extend_replay(tool_replay)
                            self._inject_tool_ephemeral_context(
                                agent=agent,
                                tool_name=tool_name,
                                tool_result=tool_result_obj,
                                messages=messages,
                                ephemeral_replay=ephemeral_replay,
                            )
                            self._inject_runtime_skill_context(
                                agent=agent,
                                tool_name=tool_name,
                                messages=messages,
                                ephemeral_replay=ephemeral_replay,
                            )
                        except ToolExecutionError as e:
                            logger.error(f"工具 '{tool_name}' 执行失败: {e}")
                            error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                            agent._record_tool_result(
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
                            tool_canonical = agent.llm.tool_result_to_canonical(error_msg, tool_id, tool_name)
                            tool_replay = agent.llm.tool_result_to_replay(error_msg, tool_id, tool_name)
                            agent._append_pending_tool_result(
                                tool_canonical=tool_canonical,
                                tool_replay=tool_replay,
                                tool_name=tool_name,
                            )
                            messages.extend_replay(tool_replay)
                        except ToolInterruption as e:
                            raise e
                        except Exception as e:
                            logger.error(f"处理工具 '{tool_name}' 调用时发生未知错误: {e}")
                            error_msg = f"工具 '{tool_name}' 处理失败: {str(e)}"
                            agent._record_tool_result(
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
                            tool_canonical = agent.llm.tool_result_to_canonical(error_msg, tool_id, tool_name)
                            tool_replay = agent.llm.tool_result_to_replay(error_msg, tool_id, tool_name)
                            agent._append_pending_tool_result(
                                tool_canonical=tool_canonical,
                                tool_replay=tool_replay,
                                tool_name=tool_name,
                            )
                            messages.extend_replay(tool_replay)
                    agent._commit_pending_step_state()
                    needs_rebuild = agent.compact_persistent_history_if_needed()
                else:
                    content = agent.llm.get_response_content(response) or getattr(response, "content", None)
                    if content is not None:
                        final_response = content
                        break
                    else:
                        logger.warning("LLM 响应中没有内容,触发纠错")
                        messages.extend_replay(agent.llm.assistant_message_to_replay(content=" ",tool_calls=None,thinking=agent.llm.get_thinking_content(response)))
                        messages.append_replay(agent.llm.query_to_replay("System Error: You must output a valid tool call via standard JSON or provide a valid final response, do not just output reasoning text."))
                max_iter -= 1

            if final_response is None:
                logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
                final_response = "超过最大迭代次数，智能体调用失败!"

            final_response = agent.skill_manager.on_after_invoke(query, final_response)
            provider_content = agent.llm.get_response_content(response) if response is not None else None
            if response is not None and final_response == provider_content:
                agent._append_response_history(response, include_reasoning=True)
            else:
                agent._append_assistant_message_history(
                    content=final_response,
                    thinking=agent.llm.get_thinking_content(response) if response is not None else None,
                )
            agent.compact_persistent_history_if_needed()
            final_event_id = agent._record_assistant_trace(
                turn_id,
                final_response,
                parent_id=agent._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                stage="final",
                round_number=iteration_count or 1,
                mode="tool",
                stream=False,
            )
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="tool",
                stream=False,
            )
            agent._observe_agent_run_end(
                agent_run_id,
                output=final_response,
                success=True,
                turn_id=turn_id,
            )
            return final_response
        except Exception as exc:
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=exc, turn_id=turn_id)
            raise
        finally:
            agent._clear_ephemeral_skill_state()

    async def ainvoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> str:
        resume_from_history = bool(kwargs.pop("resume_from_history", False))
        agent.enable_tool = True
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()
        agent._raise_if_stop_requested()

        if agent.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")

        raw_query = trace_query if trace_query is not None else (query or "[resume_pending_tool_interrupt]")
        if not resume_from_history:
            agent._append_query_history(query)
        await agent.acompact_persistent_history_if_needed()
        ephemeral_replay: list[Any] = []
        messages = agent._build_start_messages(
            query,
            include_query=False,
            extra_replay_entries=ephemeral_replay,
        )
        needs_rebuild = False
        final_response: Optional[str] = None
        response: Any = None
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        iteration_count = 0
        agent_run_id = agent._observe_agent_run_start(
            query if not resume_from_history else "[resume_pending_tool_interrupt]",
            mode="tool",
            stream=False,
            metadata={
                "entrypoint": "tool_loop_engine.ainvoke",
                "resumed": resume_from_history,
            },
        )

        try:
            while max_iter > 0:
                agent._raise_if_stop_requested()
                iteration_count += 1
                logger.debug(f"异步工具调用迭代 {iteration_count}")

                try:
                    if needs_rebuild:
                        messages = agent._build_start_messages(
                            query,
                            include_query=False,
                            extra_replay_entries=ephemeral_replay,
                        )
                        needs_rebuild = False
                    messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                        messages,
                        request_kind="tool_ainvoke",
                        temperature=temperature,
                        reasoning=agent.reasoning,
                        stream=False,
                        tools_enabled=True,
                        kwargs=kwargs,
                    )
                    llm_observation_id = agent._observe_llm_request_start(
                        turn_id=turn_id,
                        request_kind="tool_ainvoke",
                        messages=messages,
                        reasoning=request_reasoning,
                        stream=False,
                        tools_enabled=True,
                        metadata={"round": iteration_count},
                    )
                    agent.callback_manager.on_llm_start(messages)
                    try:
                        response = await agent.llm.ainvoke_with_tools(
                            messages,
                            agent.get_provider_tools(),
                            reasoning=request_reasoning,
                            temperature=request_temperature,
                            **llm_kwargs,
                        )
                    except Exception as exc:
                        agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                        raise
                    try:
                        response = agent._run_after_llm_response(
                            response,
                            messages=messages,
                            request_kind="tool_ainvoke",
                            stream=False,
                            tools_enabled=True,
                            hook_audit=llm_hook_audit,
                        )
                    except Exception as exc:
                        agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                        raise
                    agent._observe_llm_request_end(
                        llm_observation_id,
                        response=response,
                        success=True,
                        final_text=agent.llm.get_response_content(response),
                        final_thinking=agent.llm.get_thinking_content(response),
                        metadata={"round": iteration_count},
                    )
                    agent.callback_manager.on_llm_end(response)
                    agent._raise_if_stop_requested()

                    if response is None:
                        raise LLMInvokeError("LLM 返回了空响应!")
                except LLMInvokeError:
                    raise
                except AgentStopRequested:
                    raise
                except Exception as e:
                    logger.error(f"异步智能体调用失败: {str(e)[:500]}")
                    raise ToolExecutionError(f"智能体调用失败: {str(e)[:500]}") from e

                thinking_content = agent.llm.get_thinking_content(response)
                reasoning_event_id: Optional[str] = None
                if thinking_content:
                    reasoning_event_id = agent._set_round_reasoning(
                        thinking_content,
                        turn_id=turn_id,
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                    )

                if agent.llm.has_tool_calls(response):
                    response_canonical = agent.llm.response_to_canonical(
                        response,
                        include_reasoning=True,
                    )
                    response_replay = agent.llm.response_to_replay(
                        response,
                        include_reasoning=True,
                    )
                    messages.extend_replay(response_replay)
                    tool_calls = agent.llm.get_tool_calls(response)
                    agent._set_pending_step_state(
                        assistant_canonical=response_canonical,
                        assistant_replay=response_replay,
                        tool_calls=tool_calls,
                        round_number=iteration_count,
                    )
                    assistant_parent_id = (
                        reasoning_event_id
                        or agent._get_last_turn_event_id(turn_id, exclude_types={"turn_end"})
                        or turn_root_event_id
                    )
                    assistant_event_id = agent._record_assistant_trace(
                        turn_id,
                        agent.llm.get_response_content(response),
                        parent_id=assistant_parent_id,
                        stage="pre_tool",
                        round_number=iteration_count,
                        mode="tool",
                        stream=False,
                        allow_empty=True,
                    )

                    async def _process_single_tool(tool_call) -> dict[str, Any]:
                        tool_name = "unknown_tool"
                        tool_args: dict[str, Any] = {}
                        tool_id = self._extract_tool_id(tool_call)

                        try:
                            tool_name = agent._safe_get_tool_name(tool_call)
                        except Exception as e:
                            logger.warning(f"获取工具名称失败: {e}，使用默认名称")
                            if hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                                tool_name = tool_call.function.name or "unknown_tool"
                            elif hasattr(tool_call, "name"):
                                tool_name = tool_call.name or "unknown_tool"

                        try:
                            tool_args = agent._safe_parse_tool_args(tool_call)
                            logger.info(f"{agent.name} 并发异步执行工具: {tool_name}，参数: {tool_args}")
                            tool_call_event_id = agent._record_tool_call(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                parent_id=assistant_event_id or assistant_parent_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                            )
                            tool_result_obj = await agent._async_safe_execute_tool_result(
                                tool_name,
                                tool_args,
                                turn_id=turn_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                            )
                            tool_result = tool_result_obj.to_display_string()
                            tool_canonical = agent.llm.tool_result_to_canonical(tool_result, tool_id, tool_name)
                            tool_replay = agent.llm.tool_result_to_replay(tool_result, tool_id, tool_name)
                            if (
                                tool_result_obj.status == "needs_confirmation"
                                and agent.config.interrupt_on_confirmation
                            ):
                                interrupt_error = agent._finalize_tool_interrupt(
                                    turn_id=turn_id,
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tool_id=tool_id,
                                    round_number=iteration_count,
                                    tool_result=tool_result_obj,
                                    parent_id=tool_call_event_id,
                                    mode="tool",
                                    stream=False,
                                    tool_canonical=tool_canonical,
                                    tool_replay=tool_replay,
                                )
                                raise interrupt_error
                            agent._record_tool_result(
                                turn_id,
                                tool_name,
                                tool_args,
                                tool_id,
                                tool_result,
                                parent_id=tool_call_event_id,
                                round_number=iteration_count,
                                mode="tool",
                                stream=False,
                                success=tool_result_obj.status == "success",
                                tool_result_obj=tool_result_obj,
                            )
                            return {
                                "tool_name": tool_name,
                                "tool_result": tool_result_obj,
                                "tool_canonical": tool_canonical,
                                "tool_replay": tool_replay,
                                "tool_ephemeral_context": tool_result_obj.ephemeral_context,
                            }
                        except ToolExecutionError as e:
                            logger.error(f"工具 '{tool_name}' 执行失败: {e}")
                            error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                            agent._record_tool_result(
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
                            return {
                                "tool_name": tool_name,
                                "tool_result": None,
                                "tool_canonical": agent.llm.tool_result_to_canonical(error_msg, tool_id, tool_name),
                                "tool_replay": agent.llm.tool_result_to_replay(error_msg, tool_id, tool_name),
                                "tool_ephemeral_context": None,
                            }
                        except ToolInterruption:
                            raise
                        except Exception as e:
                            logger.error(f"处理工具 '{tool_name}' 调用时发生未知错误: {e}")
                            error_msg = f"工具 '{tool_name}' 处理失败: {str(e)}"
                            agent._record_tool_result(
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
                            return {
                                "tool_name": tool_name,
                                "tool_result": None,
                                "tool_canonical": agent.llm.tool_result_to_canonical(error_msg, tool_id, tool_name),
                                "tool_replay": agent.llm.tool_result_to_replay(error_msg, tool_id, tool_name),
                                "tool_ephemeral_context": None,
                            }

                    tasks = [_process_single_tool(tc) for tc in tool_calls]
                    tool_payloads = await asyncio.gather(*tasks)
                    for payload in tool_payloads:
                        agent._append_pending_tool_result(
                            tool_canonical=payload["tool_canonical"],
                            tool_replay=payload["tool_replay"],
                            ephemeral_context=payload.get("tool_ephemeral_context"),
                            tool_name=payload["tool_name"],
                        )
                        messages.extend_replay(payload["tool_replay"])
                        result_obj = payload.get("tool_result")
                        if result_obj is not None:
                            self._inject_tool_ephemeral_context(
                                agent=agent,
                                tool_name=payload["tool_name"],
                                tool_result=result_obj,
                                messages=messages,
                                ephemeral_replay=ephemeral_replay,
                            )
                    if any(agent._safe_get_tool_name(tc) == "skill_tool" for tc in tool_calls):
                        self._inject_runtime_skill_context(
                            agent=agent,
                            tool_name="skill_tool",
                            messages=messages,
                            ephemeral_replay=ephemeral_replay,
                        )
                    agent._commit_pending_step_state()
                    needs_rebuild = await agent.acompact_persistent_history_if_needed()
                else:
                    content = agent.llm.get_response_content(response) or getattr(response, "content", None)
                    if content is not None:
                        final_response = content
                        break
                    else:
                        logger.warning("LLM 响应中没有内容,触发纠错")
                        messages.extend_replay(agent.llm.assistant_message_to_replay(content=" ",tool_calls=None,thinking=agent.llm.get_thinking_content(response)))
                        messages.append_replay(agent.llm.query_to_replay("System Error: You must output a valid tool call via standard JSON or provide a valid final response, do not just output reasoning text."))
                max_iter -= 1

            if final_response is None:
                logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
                final_response = "超过最大迭代次数，智能体调用失败!"

            final_response = agent.skill_manager.on_after_invoke(query, final_response)
            provider_content = agent.llm.get_response_content(response) if response is not None else None
            if response is not None and final_response == provider_content:
                agent._append_response_history(response, include_reasoning=True)
            else:
                thinking = agent.llm.get_thinking_content(response) if response is not None else None
                agent._append_assistant_message_history(
                    content=final_response,
                    thinking=thinking,
                )
            await agent.acompact_persistent_history_if_needed()
            final_event_id = agent._record_assistant_trace(
                turn_id,
                final_response,
                parent_id=agent._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                stage="final",
                round_number=iteration_count or 1,
                mode="tool",
                stream=False,
            )
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="tool",
                stream=False,
            )
            agent._observe_agent_run_end(
                agent_run_id,
                output=final_response,
                success=True,
                turn_id=turn_id,
            )
            return final_response
        except Exception as exc:
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=exc, turn_id=turn_id)
            raise
        finally:
            agent._clear_ephemeral_skill_state()

    async def astream_invoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        resume_from_history = bool(kwargs.pop("resume_from_history", False))
        if not resume_from_history:
            agent._validate_invoke_params(query, max_iter, temperature)
        agent._raise_if_stop_requested()
        raw_query = trace_query if trace_query is not None else (query or "[resume_pending_tool_interrupt]")
        agent._current_query = query
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()
        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)

        if agent.tool_registry is None:
            error = ToolRegistryError("工具调用需要提供 ToolRegistry!")
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=error)
            raise error

        if not resume_from_history:
            agent._append_query_history(query)
        await agent.acompact_persistent_history_if_needed()
        ephemeral_replay: list[Any] = []
        messages = agent._build_start_messages(
            query,
            include_query=False,
            extra_replay_entries=ephemeral_replay,
        )
        needs_rebuild = False
        final_response = ""
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        round_index = 0
        agent_run_id = agent._observe_agent_run_start(
            query if not resume_from_history else "[resume_pending_tool_interrupt]",
            mode="tool",
            stream=True,
            metadata={
                "entrypoint": "tool_loop_engine.astream_invoke",
                "resumed": resume_from_history,
            },
        )

        try:
            while max_iter > 0:
                agent._raise_if_stop_requested()
                round_index += 1

                if needs_rebuild:
                    messages = agent._build_start_messages(
                        query,
                        include_query=False,
                        extra_replay_entries=ephemeral_replay,
                    )
                    needs_rebuild = False
                messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                    messages,
                    request_kind="tool_astream_invoke",
                    temperature=temperature,
                    reasoning=agent.reasoning,
                    stream=True,
                    tools_enabled=True,
                    kwargs=kwargs,
                )
                llm_observation_id = agent._observe_llm_request_start(
                    turn_id=turn_id,
                    request_kind="tool_astream_invoke",
                    messages=messages,
                    reasoning=request_reasoning,
                    stream=True,
                    tools_enabled=True,
                    metadata={"round": round_index},
                )
                llm_observation_closed = False
                agent.callback_manager.on_llm_start(messages)
                llm_stream = agent.llm.astream_with_tools(
                    messages,
                    agent.get_provider_tools(),
                    temperature=request_temperature,
                    reasoning=request_reasoning,
                    **llm_kwargs,
                )
                should_continue = False
                streamed_thinking = ""
                streamed_content = ""
                llm_stream_error: Optional[BaseException] = None
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
                            agent._set_round_reasoning(
                                streamed_thinking,
                                turn_id=turn_id,
                                round_number=round_index,
                                mode="tool",
                                stream=True,
                            )
                            if agent.verbose_thinking:
                                yield event
                            continue

                        if event_type == "tool_calls":
                            event = agent._run_after_llm_response(
                                dict(event),
                                messages=messages,
                                request_kind="tool_astream_invoke",
                                stream=True,
                                tools_enabled=True,
                                hook_audit=llm_hook_audit,
                            )
                            if not isinstance(event, dict):
                                raise ToolExecutionError("after_llm_response 在流式工具轮次中必须返回 dict 事件。")
                            agent._observe_llm_request_end(
                                llm_observation_id,
                                response=event,
                                success=True,
                                final_text=event.get("content"),
                                final_thinking=event.get("thinking") or streamed_thinking,
                                metadata={"round": round_index},
                            )
                            llm_observation_closed = True
                            thinking_suffix = agent._stream_snapshot_suffix(
                                streamed_thinking,
                                event.get("thinking", "") or "",
                            )
                            if thinking_suffix:
                                streamed_thinking += thinking_suffix
                                agent._set_round_reasoning(
                                    streamed_thinking,
                                    turn_id=turn_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                                if agent.verbose_thinking:
                                    yield {
                                        "type": "thinking_delta",
                                        "delta": thinking_suffix,
                                    }

                            content_suffix = agent._stream_snapshot_suffix(
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
                                assistant_replay = self._normalize_stream_assistant_replay(assistant_items)
                                assistant_canonical = agent.llm.replay_to_canonical_history(assistant_replay)
                            else:
                                assistant_canonical = agent.llm.assistant_message_to_canonical(
                                    content=event.get("content"),
                                    tool_calls=event.get("tool_calls"),
                                    thinking=streamed_thinking or None,
                                )
                                assistant_replay = agent.llm.assistant_message_to_replay(
                                    content=event.get("content"),
                                    tool_calls=event.get("tool_calls"),
                                    thinking=streamed_thinking or None,
                                )
                            messages.extend_replay(assistant_replay)
                            agent._set_pending_step_state(
                                assistant_canonical=assistant_canonical,
                                assistant_replay=assistant_replay,
                                tool_calls=event.get("tool_calls", []),
                                round_number=round_index,
                            )

                            agent.callback_manager.on_llm_end(event)
                            agent._raise_if_stop_requested()
                            reasoning_event_id = None
                            if streamed_thinking:
                                reasoning_event_id = agent._set_round_reasoning(
                                    streamed_thinking,
                                    turn_id=turn_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                            assistant_parent_id = (
                                reasoning_event_id
                                or agent._get_last_turn_event_id(turn_id, exclude_types={"turn_end"})
                                or turn_root_event_id
                            )
                            assistant_event_id = agent._record_assistant_trace(
                                turn_id,
                                event.get("content"),
                                parent_id=assistant_parent_id,
                                stage="pre_tool",
                                round_number=round_index,
                                mode="tool",
                                stream=True,
                                allow_empty=True,
                            )

                            for tool_call in event.get("tool_calls", []):
                                agent._raise_if_stop_requested()
                                tool_name = agent._safe_get_tool_name(tool_call)
                                tool_args = agent._safe_parse_tool_args(tool_call)
                                tool_id = self._extract_tool_id(tool_call)
                                yield {
                                    "type": "tool_call",
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "tool_id": tool_id,
                                }
                                tool_call_event_id = agent._record_tool_call(
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
                                    tool_result_obj = await agent._async_safe_execute_tool_result(
                                        tool_name,
                                        tool_args,
                                        turn_id=turn_id,
                                        round_number=round_index,
                                        mode="tool",
                                        stream=True,
                                    )
                                    tool_result = tool_result_obj.to_display_string()
                                    tool_success = tool_result_obj.status == "success"
                                except Exception as e:
                                    logger.error(f"流式工具 '{tool_name}' 执行失败: {e}")
                                    tool_result = f"工具 '{tool_name}' 执行失败: {e}"
                                    tool_result_obj = None
                                    tool_success = False
                                yield {
                                    "type": "tool_result",
                                    "tool_name": tool_name,
                                    "tool_id": tool_id,
                                    "tool_args": tool_args,
                                    "content": tool_result,
                                    "status": getattr(tool_result_obj, "status", None),
                                    "structured_data": getattr(tool_result_obj, "structured_data", None),
                                    "result_metadata": getattr(tool_result_obj, "metadata", None),
                                    "error_type": getattr(tool_result_obj, "error_type", None),
                                }
                                tool_canonical = agent.llm.tool_result_to_canonical(tool_result, tool_id, tool_name)
                                tool_replay = agent.llm.tool_result_to_replay(tool_result, tool_id, tool_name)
                                if (
                                    tool_result_obj is not None
                                    and tool_result_obj.status == "needs_confirmation"
                                    and agent.config.interrupt_on_confirmation
                                ):
                                    interrupt_error = agent._finalize_tool_interrupt(
                                        turn_id=turn_id,
                                        tool_name=tool_name,
                                        tool_args=tool_args,
                                        tool_id=tool_id,
                                        round_number=round_index,
                                        tool_result=tool_result_obj,
                                        parent_id=tool_call_event_id,
                                        mode="tool",
                                        stream=True,
                                        tool_canonical=tool_canonical,
                                        tool_replay=tool_replay,
                                    )
                                    agent.callback_manager.on_agent_end(
                                        agent.name,
                                        "",
                                        success=False,
                                        error=interrupt_error,
                                    )
                                    agent._observe_agent_run_end(
                                        agent_run_id,
                                        output="",
                                        success=False,
                                        error=interrupt_error,
                                        turn_id=turn_id,
                                    )
                                    yield {
                                        "type": "interruption",
                                        "reason": "needs_confirmation",
                                        "content": tool_result,
                                        "tool_name": tool_name,
                                        "tool_id": tool_id,
                                        "tool_args": tool_args,
                                        "payload": interrupt_error.to_payload(),
                                    }
                                    return

                                agent._record_tool_result(
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
                                    tool_result_obj=tool_result_obj,
                                )
                                agent._append_pending_tool_result(
                                    tool_canonical=tool_canonical,
                                    tool_replay=tool_replay,
                                    ephemeral_context=(
                                        tool_result_obj.ephemeral_context
                                        if tool_result_obj is not None else None
                                    ),
                                    tool_name=tool_name,
                                )
                                messages.extend_replay(tool_replay)
                                if tool_result_obj is not None:
                                    self._inject_tool_ephemeral_context(
                                        agent=agent,
                                        tool_name=tool_name,
                                        tool_result=tool_result_obj,
                                        messages=messages,
                                        ephemeral_replay=ephemeral_replay,
                                    )
                                self._inject_runtime_skill_context(
                                    agent=agent,
                                    tool_name=tool_name,
                                    messages=messages,
                                    ephemeral_replay=ephemeral_replay,
                                )

                            agent._commit_pending_step_state()
                            needs_rebuild = await agent.acompact_persistent_history_if_needed()
                            max_iter -= 1
                            should_continue = True
                            break

                        if event_type == "final_response":
                            event = agent._run_after_llm_response(
                                dict(event),
                                messages=messages,
                                request_kind="tool_astream_invoke",
                                stream=True,
                                tools_enabled=True,
                                hook_audit=llm_hook_audit,
                            )
                            if not isinstance(event, dict):
                                raise ToolExecutionError("after_llm_response 在流式最终响应中必须返回 dict 事件。")
                            agent._observe_llm_request_end(
                                llm_observation_id,
                                response=event,
                                success=True,
                                final_text=event.get("content"),
                                final_thinking=event.get("thinking") or streamed_thinking,
                                metadata={"round": round_index},
                            )
                            llm_observation_closed = True
                            final_response = event.get("content", "") or ""
                            agent.callback_manager.on_llm_end(event)
                            if event.get("thinking"):
                                agent._set_round_reasoning(
                                    event.get("thinking", "") or "",
                                    turn_id=turn_id,
                                    round_number=round_index,
                                    mode="tool",
                                    stream=True,
                                )
                            final_response = agent.skill_manager.on_after_invoke(query, final_response)
                            assistant_items = event.get("assistant_items")
                            if assistant_items and final_response == (event.get("content", "") or ""):
                                assistant_replay = self._normalize_stream_assistant_replay(assistant_items)
                                assistant_canonical = agent.llm.replay_to_canonical_history(assistant_replay)
                            else:
                                assistant_canonical = agent.llm.assistant_message_to_canonical(
                                        content=final_response,
                                        thinking=event.get("thinking", "") or None,
                                    )
                                assistant_replay = agent.llm.assistant_message_to_replay(
                                    content=final_response,
                                    thinking=event.get("thinking", "") or None,
                                )
                            agent._append_dual_history(assistant_canonical, assistant_replay)
                            await agent.acompact_persistent_history_if_needed()
                            final_event_id = agent._record_assistant_trace(
                                turn_id,
                                final_response,
                                parent_id=agent._get_last_turn_event_id(
                                    turn_id,
                                    exclude_types={"turn_end"},
                                ),
                                stage="final",
                                round_number=round_index,
                                mode="tool",
                                stream=True,
                            )
                            agent._record_turn_end(
                                turn_id,
                                final_event_id=final_event_id,
                                mode="tool",
                                stream=True,
                            )
                            agent.callback_manager.on_agent_end(
                                agent.name,
                                final_response,
                                success=True,
                            )
                            agent._observe_agent_run_end(
                                agent_run_id,
                                output=final_response,
                                success=True,
                                turn_id=turn_id,
                            )
                            yield {
                                "type": "final",
                                "content": final_response,
                                "thinking": event.get("thinking", ""),
                            }
                            return
                except Exception as exc:
                    llm_stream_error = exc
                    if not llm_observation_closed:
                        agent._observe_llm_request_end(
                            llm_observation_id,
                            success=False,
                            error=exc,
                            metadata={"round": round_index},
                        )
                        llm_observation_closed = True
                    raise
                finally:
                    if not llm_observation_closed and llm_stream_error is None:
                        agent._observe_llm_request_end(
                            llm_observation_id,
                            success=True,
                            final_text=streamed_content or None,
                            final_thinking=streamed_thinking or None,
                            metadata={"round": round_index, "terminatedWithoutFinalEvent": True},
                        )
                    await llm_stream.aclose()

                if should_continue:
                    continue
                break

            if not final_response:
                final_response = "超过最大迭代次数，智能体调用失败!"
            final_response = agent.skill_manager.on_after_invoke(query, final_response)
            agent._append_assistant_message_history(content=final_response)
            await agent.acompact_persistent_history_if_needed()
            final_event_id = agent._record_assistant_trace(
                turn_id,
                final_response,
                parent_id=agent._get_last_turn_event_id(turn_id, exclude_types={"turn_end"}),
                stage="final",
                round_number=round_index or 1,
                mode="tool",
                stream=True,
            )
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="tool",
                stream=True,
            )
            agent.callback_manager.on_agent_end(agent.name, final_response, success=True)
            agent._observe_agent_run_end(
                agent_run_id,
                output=final_response,
                success=True,
                turn_id=turn_id,
            )
            yield {
                "type": "final",
                "content": final_response,
                "thinking": "",
            }
        except Exception as e:
            agent._clear_pending_step_state()
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e, turn_id=turn_id)
            yield {
                "type": "error",
                "error": str(e),
            }
            raise
        finally:
            agent._clear_ephemeral_skill_state()
