"""Invocation runner interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any

from core.Exception import LLMInvokeError
from core.Message import UserMessage

logger = logging.getLogger(__name__)


class BaseInvocationRunner(ABC):
    """Abstract runner for public agent invocation entrypoints."""

    @abstractmethod
    def invoke(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def stream_invoke(
        self,
        agent: Any,
        query: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def stream_invoke_with_tool(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ):
        pass

    @abstractmethod
    async def ainvoke(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
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
        **kwargs,
    ) -> str:
        pass


class DefaultInvocationRunner(BaseInvocationRunner):
    """Default invocation runner that preserves current BasicAgent behavior."""

    def invoke(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        agent._validate_invoke_params(query, max_iter, temperature)
        original_query = query
        agent._current_query = query
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()

        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)

        messages: list[Any] = []

        if agent.enable_tool:
            logger.info("使用工具模式调用智能体")
            try:
                result = agent.tool_loop_engine.invoke(
                    agent,
                    query,
                    messages,
                    max_iter,
                    temperature,
                    trace_query=original_query,
                    **kwargs,
                )
                agent.callback_manager.on_agent_end(agent.name, result, success=True)
                return result
            except Exception as e:
                agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
                raise
            finally:
                agent._clear_ephemeral_skill_state()

        logger.info("使用普通模式调用智能体")
        try:
            messages = agent._build_start_messages(query)
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)

            agent._capture_context_usage(messages, label="invoke_plain")
            agent.callback_manager.on_llm_start(messages)
            response_obj = agent.llm.invoke_raw(
                messages,
                temperature=temperature,
                reasoning=agent.reasoning,
                **kwargs,
            )
            response = agent.llm.get_response_content(response_obj)
            agent.callback_manager.on_llm_end(response or "")

            if response is None:
                raise LLMInvokeError("LLM 返回了空响应!")

            if not isinstance(response, str):
                logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                response = str(response)

            response = agent.skill_manager.on_after_invoke(query, response)
            agent.add_message(UserMessage(query))
            agent.add_messages(
                agent._build_assistant_history_entries_from_response(
                    response_obj,
                    include_reasoning=True,
                    fallback_content=response,
                )
            )
            final_event_id = agent._record_assistant_trace(
                turn_id,
                response,
                parent_id=last_trace_event_id,
                stage="final",
                round_number=1,
                mode="plain",
                stream=False,
            ) or last_trace_event_id
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="plain",
                stream=False,
            )
            agent.callback_manager.on_agent_end(agent.name, response, success=True)
            return response
        except LLMInvokeError as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            raise
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            raise LLMInvokeError(f"LLM 调用失败: {e}") from e
        finally:
            agent._clear_ephemeral_skill_state()

    def stream_invoke(
        self,
        agent: Any,
        query: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        original_query = query
        agent._current_query = query
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()
        if agent.enable_tool:
            logger.info("使用工具模式流式调用智能体")
            display_state = agent._new_stream_display_state()
            final_result = ""
            try:
                for event in self.stream_invoke_with_tool(
                    agent,
                    query,
                    temperature=temperature,
                    trace_query=original_query,
                    **kwargs,
                ):
                    agent._display_stream_event(display_state, event)
                    if event["type"] == "interruption":
                        raise agent.tool_interrupt_controller.interruption_from_payload(
                            event["payload"]
                        )
                    if event["type"] == "final":
                        final_result = event["content"]
                        agent._print_stream_final(display_state, final_result)
                return final_result
            finally:
                agent._clear_ephemeral_skill_state()

        agent._validate_invoke_params(query, 1, temperature)
        original_query = query
        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)
        messages = agent._build_start_messages(query)
        final_results = []
        streamed_thinking = ""
        final_content = ""
        display_state = agent._new_stream_display_state()
        try:
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)
            agent._capture_context_usage(messages, label="stream_invoke_plain")
            agent.callback_manager.on_llm_start(messages)
            for event in agent.llm.stream_events(
                messages,
                temperature=temperature,
                reasoning=agent.reasoning,
                **kwargs,
            ):
                event_type = event.get("type")
                if event_type == "thinking_delta":
                    streamed_thinking += event.get("delta", "") or ""
                    if agent.verbose_thinking:
                        agent._display_stream_event(display_state, event)
                    continue
                if event_type == "text_delta":
                    final_results.append(event.get("delta", "") or "")
                    agent._display_stream_event(display_state, event)
                    continue
                if event_type == "final_response":
                    final_content = event.get("content", "") or final_content
                    streamed_thinking = event.get("thinking", "") or streamed_thinking
            result = "".join(final_results) or final_content
            agent.callback_manager.on_llm_end(result)
            result = agent.skill_manager.on_after_invoke(query, result)
            agent.add_message(UserMessage(query))
            agent.add_messages(
                agent._build_assistant_history_entries(
                    content=result,
                    thinking=streamed_thinking or None,
                )
            )
            if streamed_thinking:
                agent._set_round_reasoning(
                    streamed_thinking,
                    turn_id=turn_id,
                    round_number=1,
                    mode="plain",
                    stream=True,
                )
            final_event_id = agent._record_assistant_trace(
                turn_id,
                result,
                parent_id=last_trace_event_id,
                stage="final",
                round_number=1,
                mode="plain",
                stream=True,
            ) or last_trace_event_id
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="plain",
                stream=True,
            )
            agent.callback_manager.on_agent_end(agent.name, result, success=True)
            agent._print_stream_final(display_state, result)
            return result
        except Exception as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            raise
        finally:
            agent._clear_ephemeral_skill_state()

    def stream_invoke_with_tool(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ):
        async_gen = agent.tool_loop_engine.astream_invoke(
            agent,
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

    async def ainvoke(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        agent._validate_invoke_params(query, max_iter, temperature)
        original_query = query
        agent._current_query = query
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()

        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)

        if agent.enable_tool:
            logger.info("使用异步工具模式调用智能体")
            try:
                result = await agent.tool_loop_engine.ainvoke(
                    agent,
                    query,
                    [],
                    max_iter,
                    temperature,
                    trace_query=original_query,
                    **kwargs,
                )
                agent.callback_manager.on_agent_end(agent.name, result, success=True)
                return result
            except Exception as e:
                agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
                raise
            finally:
                agent._clear_ephemeral_skill_state()

        logger.info("使用异步普通模式调用智能体")
        try:
            messages = agent._build_start_messages(query)
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)

            agent._capture_context_usage(messages, label="ainvoke_plain")
            agent.callback_manager.on_llm_start(messages)
            response_obj = await agent.llm.ainvoke_raw(
                messages,
                temperature=temperature,
                reasoning=agent.reasoning,
                **kwargs,
            )
            response = agent.llm.get_response_content(response_obj)
            agent.callback_manager.on_llm_end(response or "")

            if response is None:
                raise LLMInvokeError("LLM 返回了空响应!")

            if not isinstance(response, str):
                logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                response = str(response)

            response = agent.skill_manager.on_after_invoke(query, response)
            agent.add_message(UserMessage(query))
            agent.add_messages(
                agent._build_assistant_history_entries_from_response(
                    response_obj,
                    include_reasoning=True,
                    fallback_content=response,
                )
            )
            final_event_id = agent._record_assistant_trace(
                turn_id,
                response,
                parent_id=last_trace_event_id,
                stage="final",
                round_number=1,
                mode="plain",
                stream=False,
            ) or last_trace_event_id
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="plain",
                stream=False,
            )
            agent.callback_manager.on_agent_end(agent.name, response, success=True)
            return response
        except LLMInvokeError as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            raise
        except Exception as e:
            logger.error(f"LLM 异步调用失败: {e}")
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            raise LLMInvokeError(f"LLM 异步调用失败: {e}") from e
        finally:
            agent._clear_ephemeral_skill_state()

    async def astream_invoke(
        self,
        agent: Any,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        original_query = query
        agent._current_query = query
        agent._clear_ephemeral_skill_state()
        if agent.enable_tool:
            display_state = agent._new_stream_display_state()
            final_result = ""
            try:
                async for event in agent.tool_loop_engine.astream_invoke(
                    agent,
                    query,
                    max_iter=max_iter,
                    temperature=temperature,
                    trace_query=original_query,
                    **kwargs,
                ):
                    agent._display_stream_event(display_state, event)
                    if event["type"] == "final":
                        final_result = event["content"]
                        agent._print_stream_final(display_state, final_result)
                return final_result
            finally:
                agent._clear_ephemeral_skill_state()

        agent._validate_invoke_params(query, 1, temperature)
        original_query = query
        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)
        messages = agent._build_start_messages(query)
        final_results = []
        streamed_thinking = ""
        final_content = ""
        display_state = agent._new_stream_display_state()
        try:
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)
            agent._capture_context_usage(messages, label="astream_invoke_plain")
            agent.callback_manager.on_llm_start(messages)
            async for event in agent.llm.astream_events(
                messages,
                temperature=temperature,
                reasoning=agent.reasoning,
                **kwargs,
            ):
                event_type = event.get("type")
                if event_type == "thinking_delta":
                    streamed_thinking += event.get("delta", "") or ""
                    if agent.verbose_thinking:
                        agent._display_stream_event(display_state, event)
                    continue
                if event_type == "text_delta":
                    final_results.append(event.get("delta", "") or "")
                    agent._display_stream_event(display_state, event)
                    continue
                if event_type == "final_response":
                    final_content = event.get("content", "") or final_content
                    streamed_thinking = event.get("thinking", "") or streamed_thinking

            result = "".join(final_results) or final_content
            agent.callback_manager.on_llm_end(result)
            result = agent.skill_manager.on_after_invoke(query, result)
            agent.add_message(UserMessage(query))
            agent.add_messages(
                agent._build_assistant_history_entries(
                    content=result,
                    thinking=streamed_thinking or None,
                )
            )
            if streamed_thinking:
                agent._set_round_reasoning(
                    streamed_thinking,
                    turn_id=turn_id,
                    round_number=1,
                    mode="plain",
                    stream=True,
                )
            final_event_id = agent._record_assistant_trace(
                turn_id,
                result,
                parent_id=last_trace_event_id,
                stage="final",
                round_number=1,
                mode="plain",
                stream=True,
            ) or last_trace_event_id
            agent._record_turn_end(
                turn_id,
                final_event_id=final_event_id,
                mode="plain",
                stream=True,
            )
            agent.callback_manager.on_agent_end(agent.name, result, success=True)
            agent._print_stream_final(display_state, result)
            return result
        except Exception as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            raise
        finally:
            agent._clear_ephemeral_skill_state()
