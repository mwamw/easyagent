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
    def _extend_turn_history(
        turn_canonical_history: list[Any],
        turn_replay_history: list[Any],
        canonical_entries: list[Any],
        replay_entries: list[Any],
    ) -> None:
        turn_canonical_history.extend(canonical_entries)
        turn_replay_history.extend(replay_entries)

    @staticmethod
    def _append_request_replay(messages: Any, replay_entries: list[Any]) -> None:
        messages.extend_replay(replay_entries)

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
        agent.enable_tool = True
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()

        if agent.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")

        raw_query = trace_query if trace_query is not None else query
        messages = agent._build_start_messages(query)

        final_response: Optional[str] = None
        response: Any = None
        turn_canonical_history = agent.llm.query_to_canonical(query)
        turn_replay_history = agent.llm.query_to_replay(query)
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        iteration_count = 0

        try:
            while max_iter > 0:
                iteration_count += 1
                logger.debug(f"工具调用迭代 {iteration_count}")

                try:
                    messages = agent.compact_request_input_if_needed(
                        messages,
                        tools=agent.tool_registry.get_openai_tools(),
                        reasoning=agent.reasoning,
                    )
                    agent._capture_context_usage(
                        messages,
                        label="invoke_tool",
                        tools=agent.tool_registry.get_openai_tools(),
                        reasoning=agent.reasoning,
                    )
                    agent.callback_manager.on_llm_start(messages)
                    response = agent.llm.invoke_with_tools(
                        messages,
                        agent.tool_registry.get_openai_tools(),
                        temperature=temperature,
                        reasoning=agent.reasoning,
                        **kwargs,
                    )
                    agent.callback_manager.on_llm_end(getattr(response, "content", "") or "")

                    if response is None:
                        raise LLMInvokeError("LLM 返回了空响应!")
                except LLMInvokeError:
                    raise
                except Exception as e:
                    logger.error(f"智能体调用失败: {str(e)[:500]}")
                    final_response = f"智能体调用失败: {str(e)[:500]}"
                    break

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
                    self._append_request_replay(messages, response_replay)
                    self._extend_turn_history(
                        turn_canonical_history,
                        turn_replay_history,
                        response_canonical,
                        response_replay,
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
                            tool_result_obj = agent._safe_execute_tool_result(tool_name, tool_args)
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
                                    turn_canonical_history=turn_canonical_history,
                                    turn_replay_history=turn_replay_history,
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
                            )
                            self._append_request_replay(messages, tool_replay)
                            self._extend_turn_history(
                                turn_canonical_history,
                                turn_replay_history,
                                tool_canonical,
                                tool_replay,
                            )
                            agent._maybe_inject_tool_ephemeral_context(
                                tool_name=tool_name,
                                tool_result=tool_result_obj,
                                messages=messages,
                            )
                            agent._maybe_inject_runtime_skill_context(
                                tool_name=tool_name,
                                messages=messages,
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
                            self._append_request_replay(messages, tool_replay)
                            self._extend_turn_history(
                                turn_canonical_history,
                                turn_replay_history,
                                tool_canonical,
                                tool_replay,
                            )
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
                            self._append_request_replay(messages, tool_replay)
                            self._extend_turn_history(
                                turn_canonical_history,
                                turn_replay_history,
                                tool_canonical,
                                tool_replay,
                            )
                else:
                    content = agent.llm.get_response_content(response) or getattr(response, "content", None)
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

            final_response = agent.skill_manager.on_after_invoke(query, final_response)
            provider_content = agent.llm.get_response_content(response) if response is not None else None
            if response is not None and final_response == provider_content:
                self._extend_turn_history(
                    turn_canonical_history,
                    turn_replay_history,
                    agent.llm.response_to_canonical(response, include_reasoning=True),
                    agent.llm.response_to_replay(response, include_reasoning=True),
                )
            else:
                self._extend_turn_history(
                    turn_canonical_history,
                    turn_replay_history,
                    agent.llm.assistant_message_to_canonical(
                        content=final_response,
                        thinking=agent.llm.get_thinking_content(response) if response is not None else None,
                    ),
                    agent.llm.assistant_message_to_replay(
                        content=final_response,
                        thinking=agent.llm.get_thinking_content(response) if response is not None else None,
                    ),
                )
            agent._append_dual_history(turn_canonical_history, turn_replay_history)
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
            return final_response
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
        agent.enable_tool = True
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()

        if agent.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")

        raw_query = trace_query if trace_query is not None else query
        messages = agent._build_start_messages(query)

        final_response: Optional[str] = None
        response: Any = None
        turn_canonical_history = agent.llm.query_to_canonical(query)
        turn_replay_history = agent.llm.query_to_replay(query)
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        iteration_count = 0

        try:
            while max_iter > 0:
                iteration_count += 1
                logger.debug(f"异步工具调用迭代 {iteration_count}")

                try:
                    messages = await agent.acompact_request_input_if_needed(
                        messages,
                        tools=agent.tool_registry.get_openai_tools(),
                        reasoning=agent.reasoning,
                    )
                    agent._capture_context_usage(
                        messages,
                        label="ainvoke_tool",
                        tools=agent.tool_registry.get_openai_tools(),
                        reasoning=agent.reasoning,
                    )
                    agent.callback_manager.on_llm_start(messages)
                    response = await agent.llm.ainvoke_with_tools(
                        messages,
                        agent.tool_registry.get_openai_tools(),
                        reasoning=agent.reasoning,
                        temperature=temperature,
                        **kwargs,
                    )
                    agent.callback_manager.on_llm_end(getattr(response, "content", "") or "")

                    if response is None:
                        raise LLMInvokeError("LLM 返回了空响应!")
                except LLMInvokeError:
                    raise
                except Exception as e:
                    logger.error(f"异步智能体调用失败: {str(e)[:500]}")
                    final_response = f"智能体调用失败: {str(e)[:500]}"
                    break

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
                    tool_calls = agent.llm.get_tool_calls(response)
                    self._append_request_replay(messages, response_replay)
                    self._extend_turn_history(
                        turn_canonical_history,
                        turn_replay_history,
                        response_canonical,
                        response_replay,
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
                                    turn_canonical_history=turn_canonical_history,
                                    turn_replay_history=turn_replay_history,
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
                            )
                            return {
                                "tool_name": tool_name,
                                "tool_result": tool_result_obj,
                                "tool_canonical": tool_canonical,
                                "tool_replay": tool_replay,
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
                            }

                    tasks = [_process_single_tool(tc) for tc in tool_calls]
                    tool_payloads = await asyncio.gather(*tasks)
                    for payload in tool_payloads:
                        self._append_request_replay(messages, payload["tool_replay"])
                        self._extend_turn_history(
                            turn_canonical_history,
                            turn_replay_history,
                            payload["tool_canonical"],
                            payload["tool_replay"],
                        )
                        result_obj = payload.get("tool_result")
                        if result_obj is not None:
                            agent._maybe_inject_tool_ephemeral_context(
                                tool_name=payload["tool_name"],
                                tool_result=result_obj,
                                messages=messages,
                            )
                    if any(agent._safe_get_tool_name(tc) == "skill_tool" for tc in tool_calls):
                        agent._maybe_inject_runtime_skill_context(
                            tool_name="skill_tool",
                            messages=messages,
                        )
                else:
                    content = agent.llm.get_response_content(response) or getattr(response, "content", None)
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

            final_response = agent.skill_manager.on_after_invoke(query, final_response)
            provider_content = agent.llm.get_response_content(response) if response is not None else None
            if response is not None and final_response == provider_content:
                self._extend_turn_history(
                    turn_canonical_history,
                    turn_replay_history,
                    agent.llm.response_to_canonical(response, include_reasoning=True),
                    agent.llm.response_to_replay(response, include_reasoning=True),
                )
            else:
                thinking = agent.llm.get_thinking_content(response) if response is not None else None
                self._extend_turn_history(
                    turn_canonical_history,
                    turn_replay_history,
                    agent.llm.assistant_message_to_canonical(content=final_response, thinking=thinking),
                    agent.llm.assistant_message_to_replay(content=final_response, thinking=thinking),
                )
            agent._append_dual_history(turn_canonical_history, turn_replay_history)
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
            return final_response
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
        agent._validate_invoke_params(query, max_iter, temperature)
        raw_query = trace_query if trace_query is not None else query
        agent._current_query = query
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()
        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)

        if agent.tool_registry is None:
            error = ToolRegistryError("工具调用需要提供 ToolRegistry!")
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=error)
            raise error

        messages = agent._build_start_messages(query)
        final_response = ""
        turn_canonical_history = agent.llm.query_to_canonical(query)
        turn_replay_history = agent.llm.query_to_replay(query)
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        round_index = 0

        try:
            while max_iter > 0:
                round_index += 1

                messages = await agent.acompact_request_input_if_needed(
                    messages,
                    tools=agent.tool_registry.get_openai_tools(),
                    reasoning=agent.reasoning,
                )
                agent._capture_context_usage(
                    messages,
                    label="astream_invoke_tool",
                    tools=agent.tool_registry.get_openai_tools(),
                    reasoning=agent.reasoning,
                )
                agent.callback_manager.on_llm_start(messages)
                llm_stream = agent.llm.astream_with_tools(
                    messages,
                    agent.tool_registry.get_openai_tools(),
                    temperature=temperature,
                    reasoning=agent.reasoning,
                    **kwargs,
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
                            self._append_request_replay(messages, assistant_replay)
                            self._extend_turn_history(
                                turn_canonical_history,
                                turn_replay_history,
                                assistant_canonical,
                                assistant_replay,
                            )

                            agent.callback_manager.on_llm_end(event.get("content", "") or "")
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
                                        turn_canonical_history=turn_canonical_history,
                                        turn_replay_history=turn_replay_history,
                                        tool_canonical=tool_canonical,
                                        tool_replay=tool_replay,
                                    )
                                    agent.callback_manager.on_agent_end(
                                        agent.name,
                                        "",
                                        success=False,
                                        error=interrupt_error,
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
                                )
                                self._append_request_replay(messages, tool_replay)
                                self._extend_turn_history(
                                    turn_canonical_history,
                                    turn_replay_history,
                                    tool_canonical,
                                    tool_replay,
                                )
                                if tool_result_obj is not None:
                                    agent._maybe_inject_tool_ephemeral_context(
                                        tool_name=tool_name,
                                        tool_result=tool_result_obj,
                                        messages=messages,
                                    )
                                agent._maybe_inject_runtime_skill_context(
                                    tool_name=tool_name,
                                    messages=messages,
                                )

                            max_iter -= 1
                            should_continue = True
                            break

                        if event_type == "final_response":
                            final_response = event.get("content", "") or ""
                            agent.callback_manager.on_llm_end(final_response)
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
                            self._extend_turn_history(
                                turn_canonical_history,
                                turn_replay_history,
                                assistant_canonical,
                                assistant_replay,
                            )
                            agent._append_dual_history(turn_canonical_history, turn_replay_history)
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
            final_response = agent.skill_manager.on_after_invoke(query, final_response)
            self._extend_turn_history(
                turn_canonical_history,
                turn_replay_history,
                agent.llm.assistant_message_to_canonical(content=final_response),
                agent.llm.assistant_message_to_replay(content=final_response),
            )
            agent._append_dual_history(turn_canonical_history, turn_replay_history)
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
            yield {
                "type": "final",
                "content": final_response,
                "thinking": "",
            }
        except Exception as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            yield {
                "type": "error",
                "error": str(e),
            }
            raise
        finally:
            agent._clear_ephemeral_skill_state()
