"""Invocation runner interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any

from core.Exception import LLMInvokeError

logger = logging.getLogger(__name__)
from agent import BasicAgent

class BaseInvocationRunner(ABC):
    """Abstract runner for public agent invocation entrypoints."""

    @abstractmethod
    def invoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def stream_invoke(
        self,
        agent:BasicAgent,
        query: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def stream_invoke_with_tool(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    )->Any:
        pass

    @abstractmethod
    async def ainvoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
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
        **kwargs,
    ) -> str:
        pass


class DefaultInvocationRunner(BaseInvocationRunner):
    """Default invocation runner that preserves current BasicAgent behavior."""

    @staticmethod
    def _drain_loop_awaitable(loop: asyncio.AbstractEventLoop, awaitable: Any) -> None:
        try:
            if loop.is_closed():
                return
            loop.run_until_complete(awaitable)
        except Exception:
            close = getattr(awaitable, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def invoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        agent._validate_invoke_params(query, max_iter, temperature)
        agent._raise_if_stop_requested()
        original_query = query
        agent._current_query = query
        agent._clear_last_tool_interrupt()
        agent._clear_ephemeral_skill_state()

        query = agent.skill_manager.on_before_invoke(query)
        agent.callback_manager.on_agent_start(agent.name, query)

        # messages: list[Any] = []

        if agent.enable_tool:
            logger.info("使用工具模式调用智能体")
            try:
                result = agent.tool_loop_engine.invoke(
                    agent,
                    query,
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
        agent_run_id = agent._observe_agent_run_start(
            query,
            mode="plain",
            stream=False,
            metadata={"entrypoint": "invoke"},
        )
        try:
            agent._raise_if_stop_requested()
            agent._append_query_history(query)
            agent.compact_persistent_history_if_needed()
            messages = agent._build_start_messages(query, include_query=False)
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)

            messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                messages,
                request_kind="plain_invoke",
                temperature=temperature,
                reasoning=agent.reasoning,
                stream=False,
                tools_enabled=False,
                kwargs=kwargs,
            )
            llm_observation_id = agent._observe_llm_request_start(
                turn_id=turn_id,
                request_kind="plain_invoke",
                messages=messages,
                reasoning=request_reasoning,
                stream=False,
                tools_enabled=False,
            )
            agent.callback_manager.on_llm_start(messages)
            try:
                response_obj = agent.llm.invoke_raw(
                    messages,
                    temperature=request_temperature,
                    reasoning=request_reasoning,
                    **llm_kwargs,
                )
            except Exception as exc:
                agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                raise
            try:
                response_obj = agent._run_after_llm_response(
                    response_obj,
                    messages=messages,
                    request_kind="plain_invoke",
                    stream=False,
                    tools_enabled=False,
                    hook_audit=llm_hook_audit,
                )
            except Exception as exc:
                agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                raise
            provider_content = agent.llm.get_response_content(response_obj)
            response = provider_content
            agent._observe_llm_request_end(
                llm_observation_id,
                response=response_obj,
                success=True,
                final_text=provider_content,
                final_thinking=agent.llm.get_thinking_content(response_obj),
            )
            agent.callback_manager.on_llm_end(response_obj)
            agent._raise_if_stop_requested()

            if response is None:
                raise LLMInvokeError("LLM 返回了空响应!")

            if not isinstance(response, str):
                logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                response = str(response)

            response = agent.skill_manager.on_after_invoke(query, response)
            if response == provider_content:
                agent._append_response_history(response_obj, include_reasoning=True)
            else:
                agent._append_assistant_message_history(
                    content=response,
                    thinking=agent.llm.get_thinking_content(response_obj),
                )
            agent.compact_persistent_history_if_needed()
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
            agent._observe_agent_run_end(agent_run_id, output=response, success=True, turn_id=turn_id)
            return response
        except LLMInvokeError as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e)
            raise
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e)
            raise LLMInvokeError(f"LLM 调用失败: {e}") from e
        finally:
            agent._clear_ephemeral_skill_state()

    def stream_invoke(
        self,
        agent:BasicAgent,
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
        agent_run_id = agent._observe_agent_run_start(
            query,
            mode="plain",
            stream=True,
            metadata={"entrypoint": "stream_invoke"},
        )
        agent.callback_manager.on_agent_start(agent.name, query)
        agent._append_query_history(query)
        agent.compact_persistent_history_if_needed()
        messages = agent._build_start_messages(query, include_query=False)
        final_results = []
        streamed_thinking = ""
        final_content = ""
        final_event_payload: dict[str, Any] | None = None
        display_state = agent._new_stream_display_state()
        llm_observation_id: str | None = None
        try:
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)
            messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                messages,
                request_kind="plain_stream_invoke",
                temperature=temperature,
                reasoning=agent.reasoning,
                stream=True,
                tools_enabled=False,
                kwargs=kwargs,
            )
            llm_observation_id = agent._observe_llm_request_start(
                turn_id=turn_id,
                request_kind="plain_stream_invoke",
                messages=messages,
                reasoning=request_reasoning,
                stream=True,
                tools_enabled=False,
            )
            agent.callback_manager.on_llm_start(messages)
            for event in agent.llm.stream_events(
                messages,
                temperature=request_temperature,
                reasoning=request_reasoning,
                **llm_kwargs,
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
                    final_event_payload = dict(event)
                    final_content = event.get("content", "") or final_content
                    streamed_thinking = event.get("thinking", "") or streamed_thinking
            result = "".join(final_results) or final_content
            response_event = dict(final_event_payload or {})
            response_event.setdefault("type", "final_response")
            response_event["content"] = result
            response_event["thinking"] = streamed_thinking
            stream_response = agent._run_after_llm_response(
                response_event,
                messages=messages,
                request_kind="plain_stream_invoke",
                stream=True,
                tools_enabled=False,
                hook_audit=llm_hook_audit,
            )
            if isinstance(stream_response, dict):
                result = str(stream_response.get("content", result) or result)
                streamed_thinking = str(stream_response.get("thinking", streamed_thinking) or streamed_thinking)
            agent._observe_llm_request_end(
                llm_observation_id,
                response=stream_response,
                success=True,
                final_text=result,
                final_thinking=streamed_thinking,
            )
            agent.callback_manager.on_llm_end(stream_response)
            result = agent.skill_manager.on_after_invoke(query, result)
            agent._append_assistant_message_history(
                content=result,
                thinking=streamed_thinking or None,
            )
            agent.compact_persistent_history_if_needed()
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
            agent._observe_agent_run_end(agent_run_id, output=result, success=True, turn_id=turn_id)
            agent._print_stream_final(display_state, result)
            return result
        except Exception as e:
            agent._observe_llm_request_end(llm_observation_id, success=False, error=e)
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e)
            raise
        finally:
            agent._clear_ephemeral_skill_state()

    def stream_invoke_with_tool(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "stream_invoke_with_tool cannot run inside an active event loop; "
                "use `await agent.astream_invoke(...)` instead."
            )
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
                self._drain_loop_awaitable(loop, async_gen.aclose())
            except Exception:
                pass
            try:
                self._drain_loop_awaitable(loop, loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                self._drain_loop_awaitable(loop, loop.shutdown_default_executor())
            except Exception:
                pass
            asyncio.set_event_loop(previous_loop)
            loop.close()

    async def ainvoke(
        self,
        agent:BasicAgent,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        agent._validate_invoke_params(query, max_iter, temperature)
        agent._raise_if_stop_requested()
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
        agent_run_id = agent._observe_agent_run_start(
            query,
            mode="plain",
            stream=False,
            metadata={"entrypoint": "ainvoke"},
        )
        try:
            agent._raise_if_stop_requested()
            agent._append_query_history(query)
            await agent.acompact_persistent_history_if_needed()
            messages = agent._build_start_messages(query, include_query=False)
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)

            messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                messages,
                request_kind="plain_ainvoke",
                temperature=temperature,
                reasoning=agent.reasoning,
                stream=False,
                tools_enabled=False,
                kwargs=kwargs,
            )
            llm_observation_id = agent._observe_llm_request_start(
                turn_id=turn_id,
                request_kind="plain_ainvoke",
                messages=messages,
                reasoning=request_reasoning,
                stream=False,
                tools_enabled=False,
            )
            agent.callback_manager.on_llm_start(messages)
            try:
                response_obj = await agent.llm.ainvoke_raw(
                    messages,
                    temperature=request_temperature,
                    reasoning=request_reasoning,
                    **llm_kwargs,
                )
            except Exception as exc:
                agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                raise
            try:
                response_obj = agent._run_after_llm_response(
                    response_obj,
                    messages=messages,
                    request_kind="plain_ainvoke",
                    stream=False,
                    tools_enabled=False,
                    hook_audit=llm_hook_audit,
                )
            except Exception as exc:
                agent._observe_llm_request_end(llm_observation_id, success=False, error=exc)
                raise
            provider_content = agent.llm.get_response_content(response_obj)
            response = provider_content
            agent._observe_llm_request_end(
                llm_observation_id,
                response=response_obj,
                success=True,
                final_text=provider_content,
                final_thinking=agent.llm.get_thinking_content(response_obj),
            )
            agent.callback_manager.on_llm_end(response_obj)
            agent._raise_if_stop_requested()

            if response is None:
                raise LLMInvokeError("LLM 返回了空响应!")

            if not isinstance(response, str):
                logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                response = str(response)

            response = agent.skill_manager.on_after_invoke(query, response)
            if response == provider_content:
                agent._append_response_history(response_obj, include_reasoning=True)
            else:
                agent._append_assistant_message_history(
                    content=response,
                    thinking=agent.llm.get_thinking_content(response_obj),
                )
            await agent.acompact_persistent_history_if_needed()
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
            agent._observe_agent_run_end(agent_run_id, output=response, success=True, turn_id=turn_id)
            return response
        except LLMInvokeError as e:
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e)
            raise
        except Exception as e:
            logger.error(f"LLM 异步调用失败: {e}")
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e)
            raise LLMInvokeError(f"LLM 异步调用失败: {e}") from e
        finally:
            agent._clear_ephemeral_skill_state()

    async def astream_invoke(
        self,
        agent:BasicAgent,
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
        agent_run_id = agent._observe_agent_run_start(
            query,
            mode="plain",
            stream=True,
            metadata={"entrypoint": "astream_invoke"},
        )
        agent.callback_manager.on_agent_start(agent.name, query)
        agent._append_query_history(query)
        await agent.acompact_persistent_history_if_needed()
        messages = agent._build_start_messages(query, include_query=False)
        final_results = []
        streamed_thinking = ""
        final_content = ""
        final_event_payload: dict[str, Any] | None = None
        display_state = agent._new_stream_display_state()
        llm_observation_id: str | None = None
        try:
            turn_id, last_trace_event_id = agent._begin_trace_turn(original_query)
            messages, request_temperature, request_reasoning, llm_kwargs, llm_hook_audit = agent._run_before_llm_request(
                messages,
                request_kind="plain_astream_invoke",
                temperature=temperature,
                reasoning=agent.reasoning,
                stream=True,
                tools_enabled=False,
                kwargs=kwargs,
            )
            llm_observation_id = agent._observe_llm_request_start(
                turn_id=turn_id,
                request_kind="plain_astream_invoke",
                messages=messages,
                reasoning=request_reasoning,
                stream=True,
                tools_enabled=False,
            )
            agent.callback_manager.on_llm_start(messages)
            async for event in agent.llm.astream_events(
                messages,
                temperature=request_temperature,
                reasoning=request_reasoning,
                **llm_kwargs,
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
                    final_event_payload = dict(event)
                    final_content = event.get("content", "") or final_content
                    streamed_thinking = event.get("thinking", "") or streamed_thinking

            result = "".join(final_results) or final_content
            response_event = dict(final_event_payload or {})
            response_event.setdefault("type", "final_response")
            response_event["content"] = result
            response_event["thinking"] = streamed_thinking
            stream_response = agent._run_after_llm_response(
                response_event,
                messages=messages,
                request_kind="plain_astream_invoke",
                stream=True,
                tools_enabled=False,
                hook_audit=llm_hook_audit,
            )
            if isinstance(stream_response, dict):
                result = str(stream_response.get("content", result) or result)
                streamed_thinking = str(stream_response.get("thinking", streamed_thinking) or streamed_thinking)
            agent._observe_llm_request_end(
                llm_observation_id,
                response=stream_response,
                success=True,
                final_text=result,
                final_thinking=streamed_thinking,
            )
            agent.callback_manager.on_llm_end(stream_response)
            result = agent.skill_manager.on_after_invoke(query, result)
            agent._append_assistant_message_history(
                content=result,
                thinking=streamed_thinking or None,
            )
            await agent.acompact_persistent_history_if_needed()
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
            agent._observe_agent_run_end(agent_run_id, output=result, success=True, turn_id=turn_id)
            agent._print_stream_final(display_state, result)
            return result
        except Exception as e:
            agent._observe_llm_request_end(llm_observation_id, success=False, error=e)
            agent.callback_manager.on_agent_end(agent.name, "", success=False, error=e)
            agent._observe_agent_run_end(agent_run_id, output="", success=False, error=e)
            raise
        finally:
            agent._clear_ephemeral_skill_state()
