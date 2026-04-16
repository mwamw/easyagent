"""Tool loop engine interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

from core.Exception import (
    LLMInvokeError,
    ToolExecutionError,
    ToolInterruption,
    ToolRegistryError,
)
from core.Message import UserMessage

logger = logging.getLogger(__name__)


class BaseToolLoopEngine(ABC):
    """Abstract engine for tool-calling loops."""

    @abstractmethod
    def invoke(
        self,
        agent: Any,
        query: str,
        messages: list[Any],
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def ainvoke(
        self,
        agent: Any,
        query: str,
        messages: list[Any],
        max_iter: int = 10,
        temperature: float = 0.7,
        trace_query: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def astream_invoke(
        self,
        agent: Any,
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

    def invoke(
        self,
        agent: Any,
        query: str,
        messages: list[Any],
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
        turn_history: list[Any] = [UserMessage(query)]
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        iteration_count = 0

        try:
            while max_iter > 0:
                iteration_count += 1
                logger.debug(f"工具调用迭代 {iteration_count}")

                try:
                    agent._capture_context_usage(messages, label="invoke_tool")
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
                    logger.error(f"智能体调用失败: {e}")
                    final_response = f"智能体调用失败: {str(e)}"
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
                    formatted_response = agent.llm.format_assistant_response(
                        response,
                        include_reasoning=True,
                    )
                    if isinstance(formatted_response, list):
                        messages.extend(formatted_response)
                    else:
                        messages.append(formatted_response)
                    turn_history.extend(agent._as_history_entries(formatted_response))
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
                        tool_id = (
                            getattr(tool_call, "call_id", None)
                            or getattr(tool_call, "id", "unknown")
                        )

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
                            tool_msg = agent.llm.format_tool_result(tool_result, tool_id, tool_name)
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
                                    turn_history=turn_history,
                                    tool_message=tool_msg,
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
                            messages.append(tool_msg)
                            turn_history.append(tool_msg)
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
                            tool_msg = agent.llm.format_tool_result(error_msg, tool_id, tool_name)
                            messages.append(tool_msg)
                            turn_history.append(tool_msg)
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
                            tool_msg = agent.llm.format_tool_result(error_msg, tool_id, tool_name)
                            messages.append(tool_msg)
                            turn_history.append(tool_msg)
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
            turn_history.extend(
                agent._build_assistant_history_entries_from_response(
                    response,
                    include_reasoning=True,
                    fallback_content=final_response,
                )
            )
            agent.add_messages(turn_history)
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
        agent: Any,
        query: str,
        messages: list[Any],
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
        turn_history: list[Any] = [UserMessage(query)]
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        iteration_count = 0

        try:
            while max_iter > 0:
                iteration_count += 1
                logger.debug(f"异步工具调用迭代 {iteration_count}")

                try:
                    agent._capture_context_usage(messages, label="ainvoke_tool")
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
                    logger.error(f"异步智能体调用失败: {e}")
                    final_response = f"智能体调用失败: {str(e)}"
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
                    formatted_response = agent.llm.format_assistant_response(
                        response,
                        include_reasoning=True,
                    )
                    tool_calls = agent.llm.get_tool_calls(response)
                    if isinstance(formatted_response, list):
                        messages.extend(formatted_response)
                    else:
                        messages.append(formatted_response)
                    turn_history.extend(agent._as_history_entries(formatted_response))
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
                        tool_id = (
                            getattr(tool_call, "call_id", None)
                            or getattr(tool_call, "id", "unknown")
                        )

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
                            tool_message = agent.llm.format_tool_result(tool_result, tool_id, tool_name)
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
                                    turn_history=turn_history,
                                    tool_message=tool_message,
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
                                "tool_message": tool_message,
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
                                "tool_message": agent.llm.format_tool_result(error_msg, tool_id, tool_name),
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
                                "tool_message": agent.llm.format_tool_result(error_msg, tool_id, tool_name),
                            }

                    tasks = [_process_single_tool(tc) for tc in tool_calls]
                    tool_payloads = await asyncio.gather(*tasks)
                    tool_msgs = [payload["tool_message"] for payload in tool_payloads]
                    messages.extend(tool_msgs)
                    turn_history.extend(tool_msgs)
                    for payload in tool_payloads:
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
            turn_history.extend(
                agent._build_assistant_history_entries_from_response(
                    response,
                    include_reasoning=True,
                    fallback_content=final_response,
                )
            )
            agent.add_messages(turn_history)
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
        agent: Any,
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
        turn_history: list[Any] = [UserMessage(query)]
        turn_id, turn_root_event_id = agent._begin_trace_turn(raw_query)
        round_index = 0

        try:
            while max_iter > 0:
                round_index += 1

                agent._capture_context_usage(messages, label="astream_invoke_tool")
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
                                assistant_message = assistant_items
                            else:
                                assistant_message = agent.llm.format_assistant_message(
                                    content=event.get("content"),
                                    tool_calls=event.get("tool_calls"),
                                    thinking=streamed_thinking or None,
                                )
                            if isinstance(assistant_message, list):
                                messages.extend(assistant_message)
                            else:
                                messages.append(assistant_message)
                            turn_history_entries = agent._as_history_entries(assistant_message)
                            turn_history.extend(turn_history_entries)

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
                                tool_id = (
                                    tool_call.get("id", "unknown")
                                    if isinstance(tool_call, dict)
                                    else getattr(tool_call, "call_id", None)
                                    or getattr(tool_call, "id", "unknown")
                                )
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
                                tool_message = agent.llm.format_tool_result(tool_result, tool_id, tool_name)
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
                                        turn_history=turn_history,
                                        tool_message=tool_message,
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
                                messages.append(tool_message)
                                turn_history.append(tool_message)
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
                                turn_history.extend(agent._as_history_entries(assistant_items))
                            else:
                                turn_history.extend(
                                    agent._build_assistant_history_entries(
                                        content=final_response,
                                        thinking=event.get("thinking", "") or None,
                                    )
                                )
                            agent.add_messages(turn_history)
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
            turn_history.extend(
                agent._build_assistant_history_entries(
                    content=final_response,
                )
            )
            agent.add_messages(turn_history)
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
