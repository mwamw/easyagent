from typing_extensions import override
from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from typing import Optional, Any, AsyncGenerator, TYPE_CHECKING
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
import asyncio
import json
import logging
from core.Exception import *


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
        self.thinking_history: list[str] = []  # 记录思考过程
        self._current_query: str = ""  # 当前查询（供 get_enhanced_prompt 使用）
        self.history_via_context_manager = history_via_context_manager

        logger.info(f"BasicAgent '{name}' 初始化完成，工具调用: {'启用' if enable_tool else '禁用'}，provider: {llm.provider_name}")

    def _get_serializable_state(self) -> dict[str, Any]:
        return {
            "verbose_thinking": self.verbose_thinking,
            "history_via_context_manager": self.history_via_context_manager,
            "thinking_history": self.thinking_history,
        }

    def _restore_serializable_state(self, state: Optional[dict[str, Any]]) -> None:
        state = state or {}
        self.verbose_thinking = state.get("verbose_thinking", False)
        self.thinking_history = list(state.get("thinking_history", []))

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
        self._current_query = query  # 供 get_enhanced_prompt 使用
        
        # Skill 前置拦截
        query = self.skill_manager.on_before_invoke(query)
        
        self.callback_manager.on_agent_start(self.name, query)
        
        messages: list[Message | dict[str, str]] = []
 
        if self.enable_tool :
            logger.info("使用工具模式调用智能体")
            try:
                result = self.invoke_with_tool(query, messages, max_iter, temperature, **kwargs)
                self.callback_manager.on_agent_end(self.name, result, success=True)
                return result
            except Exception as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
        else:
            logger.info("使用普通模式调用智能体")
            try:
                messages = self._build_start_messages(query)
                
                self.callback_manager.on_llm_start(messages)
                response = self.llm.invoke(messages, temperature=temperature, **kwargs)
                self.callback_manager.on_llm_end(response or "")
                
                # 验证响应
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                if not isinstance(response, str):
                    logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                    response = str(response)
                
                self.add_message(UserMessage(query))
                self.add_message(AssistantMessage(response))
                # Skill 后置拦截
                response = self.skill_manager.on_after_invoke(query, response)
                self.callback_manager.on_agent_end(self.name, response, success=True)
                return response
                
            except LLMInvokeError as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise LLMInvokeError(f"LLM 调用失败: {e}") from e

    def stream_invoke(self,query: str,temperature: float = 0.7, **kwargs):
        self._current_query = query
        if self.enable_tool:
            logger.info("使用工具模式流式调用智能体")
            final_result = ""
            for event in self.stream_invoke_with_tool(query, temperature=temperature, **kwargs):
                if event["type"] == "text_delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "final":
                    final_result = event["content"]
            return final_result
        else:
            messages = self._build_start_messages(query)
            final_results=[]
            for chunk in self.llm.think(messages, temperature=temperature, **kwargs):
                final_results.append(chunk)
                
            self.add_message(UserMessage(query))
            self.add_message(AssistantMessage("".join(final_results)))
            return "".join(final_results)

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
        
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        messages = self._build_start_messages(query)
        
        final_response: Optional[str] = None
        iteration_count = 0
        
        while max_iter > 0:
            iteration_count += 1
            logger.debug(f"工具调用迭代 {iteration_count}")
            
            try:
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

                formatted_response = self.llm.format_assistant_response(response)
                # print(f"formatted_response:{formatted_response}")
                if isinstance(formatted_response, list):
                    messages.extend(formatted_response)
                else:
                    messages.append(formatted_response)

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
            if thinking_content:
                self.thinking_history.append(thinking_content)
                if self.verbose_thinking:
                    logger.info(f"💭 模型思考: {thinking_content}")
                # messages.append(AssistantMessage(thinking_content))

            if self.llm.has_tool_calls(response):
                for tool_call in self.llm.get_tool_calls(response):
                    tool_name = "unknown_tool"
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
                        tool_result = self._safe_execute_tool(tool_name, tool_args)
                        tool_msg = self.llm.format_tool_result(tool_result, tool_id, tool_name)
                        messages.append(tool_msg)

                    except ToolExecutionError as e:
                        logger.error(f"工具 '{tool_name}' 执行失败: {e}")
                        error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                        tool_msg = self.llm.format_tool_result(error_msg, tool_id, tool_name)
                        messages.append(tool_msg)
                    except Exception as e:
                        logger.error(f"处理工具 '{tool_name}' 调用时发生未知错误: {e}")
                        error_msg = f"工具 '{tool_name}' 处理失败: {str(e)}"
                        tool_msg = self.llm.format_tool_result(error_msg, tool_id, tool_name)
                        messages.append(tool_msg)
            else:
                content = self.llm.get_response_content(response) or getattr(response, 'content', None)
                if content is not None:
                    messages.append(AssistantMessage(content))
                    final_response = content
                else:
                    logger.warning("LLM 响应中没有内容")
                    final_response = ""
                break
            
            max_iter -= 1
        
        if final_response is None:
            logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
            final_response = "超过最大迭代次数，智能体调用失败!"
        
        self.add_message(UserMessage(query))
        self.add_message(AssistantMessage(final_response))
        return final_response

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
        self._current_query = query
        
        # Skill 前置拦截
        query = self.skill_manager.on_before_invoke(query)
        
        self.callback_manager.on_agent_start(self.name, query)
        
        if self.enable_tool:
            logger.info("使用异步工具模式调用智能体")
            try:
                result = await self.ainvoke_with_tool(query, [], max_iter, temperature, **kwargs)
                self.callback_manager.on_agent_end(self.name, result, success=True)
                return result
            except Exception as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
        else:
            logger.info("使用异步普通模式调用智能体")
            try:
                messages = self._build_start_messages(query)
                
                self.callback_manager.on_llm_start(messages)
                response = await self.llm.ainvoke(messages, temperature=temperature, **kwargs)
                self.callback_manager.on_llm_end(response or "")
                
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                if not isinstance(response, str):
                    logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                    response = str(response)
                
                self.add_message(UserMessage(query))
                self.add_message(AssistantMessage(response))
                response = self.skill_manager.on_after_invoke(query, response)
                self.callback_manager.on_agent_end(self.name, response, success=True)
                return response
                
            except LLMInvokeError as e:
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise
            except Exception as e:
                logger.error(f"LLM 异步调用失败: {e}")
                self.callback_manager.on_agent_end(self.name, "", success=False, error=e)
                raise LLMInvokeError(f"LLM 异步调用失败: {e}") from e

    async def ainvoke_with_tool(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        原生异步工具调用模式
        
        与 invoke_with_tool 对称，但 LLM 调用和工具执行均为异步。
        """
        self.enable_tool = True
        
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        messages = self._build_start_messages(query)
        
        final_response: Optional[str] = None
        iteration_count = 0
        
        while max_iter > 0:
            iteration_count += 1
            logger.debug(f"异步工具调用迭代 {iteration_count}")
            
            try:
                print("now messages:",messages[1:])
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

                formatted_response = self.llm.format_assistant_response(response)
                if isinstance(formatted_response, list):
                    messages.extend(formatted_response)
                else:
                    messages.append(formatted_response)

            except LLMInvokeError:
                raise
            except Exception as e:
                logger.error(f"异步智能体调用失败: {e}")
                final_response = f"智能体调用失败: {str(e)}"
                break

            # 捕获 LLM 的思考过程
            thinking_content = self.llm.get_thinking_content(response)
            if thinking_content:
                self.thinking_history.append(thinking_content)
                if self.verbose_thinking:
                    logger.info(f"💭 模型思考: {thinking_content}")
                messages.append(AssistantMessage(thinking_content))

            # 检查是否有工具调用
            if self.llm.has_tool_calls(response):
                async def _process_single_tool(tool_call) -> Message | dict:
                    tool_name = "unknown_tool"
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
                        tool_result = await self._async_safe_execute_tool(tool_name, tool_args)
                        return self.llm.format_tool_result(tool_result, tool_id, tool_name)

                    except ToolExecutionError as e:
                        logger.error(f"工具 '{tool_name}' 执行失败: {e}")
                        error_msg = f"工具 '{tool_name}' 执行失败: {str(e)}"
                        return self.llm.format_tool_result(error_msg, tool_id, tool_name)
                    except Exception as e:
                        logger.error(f"处理工具 '{tool_name}' 调用时发生未知错误: {e}")
                        error_msg = f"工具 '{tool_name}' 处理失败: {str(e)}"
                        return self.llm.format_tool_result(error_msg, tool_id, tool_name)

                # 并发执行所有工具调用
                tasks = [
                    _process_single_tool(tc) 
                    for tc in self.llm.get_tool_calls(response)
                ]
                tool_msgs = await asyncio.gather(*tasks)
                messages.extend(tool_msgs)

            else:
                content = self.llm.get_response_content(response) or getattr(response, 'content', None)
                if content is not None:
                    messages.append(AssistantMessage(content))
                    final_response = content
                else:
                    logger.warning("LLM 响应中没有内容")
                    final_response = ""
                break
            
            max_iter -= 1
        
        if final_response is None:
            logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
            final_response = "超过最大迭代次数，智能体调用失败!"
        
        self.add_message(UserMessage(query))
        self.add_message(AssistantMessage(final_response))
        return final_response

    async def astream_invoke(self, query: str, temperature: float = 0.7, **kwargs) -> str:
        """
        异步流式调用智能体
        
        Args:
            query: 用户输入
            temperature: 温度参数
            
        Returns:
            完整响应文本
        """
        self._current_query = query
        if self.enable_tool:
            final_result = ""
            async for event in self.astream_invoke_with_tool(query, temperature=temperature, **kwargs):
                if event["type"] == "text_delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "final":
                    final_result = event["content"]
            return final_result
        
        messages = self._build_start_messages(query)
        final_results = []
        async for chunk in self.llm.astream(messages, temperature=temperature, **kwargs):
            print(chunk, end="", flush=True)
            final_results.append(chunk)
        
        result = "".join(final_results)
        self.add_message(UserMessage(query))
        self.add_message(AssistantMessage(result))
        return result

    async def astream_invoke_with_tool(
        self,
        query: str,
        max_iter: int = 10,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式工具调用，逐步产出文本、工具与最终结果事件。"""
        if self.verbose_thinking:
            kwargs.setdefault("reasoning", {"effort": "medium", "summary": "auto"})
        self._validate_invoke_params(query, max_iter, temperature)
        self._current_query = query
        query = self.skill_manager.on_before_invoke(query)
        self.callback_manager.on_agent_start(self.name, query)

        if self.tool_registry is None:
            error = ToolRegistryError("工具调用需要提供 ToolRegistry!")
            self.callback_manager.on_agent_end(self.name, "", success=False, error=error)
            raise error

        messages = self._build_start_messages(query)
        final_response = ""

        try:
            while max_iter > 0:
                # print("now messages:",messages[1:])

                self.callback_manager.on_llm_start(messages)
                llm_stream = self.llm.astream_with_tools(
                    messages,
                    self.tool_registry.get_openai_tools(),
                    temperature=temperature,
                    **kwargs
                )
                should_continue = False
                try:
                    async for event in llm_stream:
                        event_type = event.get("type")

                        if event_type == "stream_end":
                            continue

                        if event_type == "text_delta":
                            yield event
                            continue

                        if event_type == "thinking_delta":
                            delta = event.get("delta", "")
                            if delta:
                                self.thinking_history.append(delta)
                            if self.verbose_thinking:
                                yield event
                            continue

                        if event_type == "tool_calls":
                            assistant_message = self.llm.format_assistant_message(
                                content=event.get("content"),
                                tool_calls=event.get("tool_calls"),
                            )
                            if isinstance(assistant_message, list):
                                messages.extend(assistant_message)
                            else:
                                messages.append(assistant_message)

                            self.callback_manager.on_llm_end(event.get("content", "") or "")

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
                                try:
                                    tool_result = await self._async_safe_execute_tool(tool_name, tool_args)
                                except Exception as e:
                                    logger.error(f"流式工具 '{tool_name}' 执行失败: {e}")
                                    tool_result = f"工具 '{tool_name}' 执行失败: {e}"
                                yield {
                                    "type": "tool_result",
                                    "tool_name": tool_name,
                                    "tool_id": tool_id,
                                    "tool_args": tool_args,
                                    "content": tool_result,
                                }
                                messages.append(
                                    self.llm.format_tool_result(tool_result, tool_id, tool_name)
                                )

                            max_iter -= 1
                            should_continue = True
                            break

                        if event_type == "final_response":
                            final_response = event.get("content", "") or ""
                            self.callback_manager.on_llm_end(final_response)
                            self.add_message(UserMessage(query))
                            self.add_message(AssistantMessage(final_response))
                            final_response = self.skill_manager.on_after_invoke(query, final_response)
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
            self.add_message(UserMessage(query))
            self.add_message(AssistantMessage(final_response))
            final_response = self.skill_manager.on_after_invoke(query, final_response)
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

 
    @override
    def get_enhanced_prompt(self) -> str:

        """
        获取增强后的系统提示词
        
        Returns:
            增强后的系统提示词
        """
        thinking_prompt="""在申请工具调用或者回复的同时，需要给出思考过程,但输出最终结果时不要含有思考内容"""
        if not self.enable_tool or not self.tool_registry:
            return self.system_prompt or "你是一个有用的AI助手，帮助用户回答问题，完成任务。"
        
        try:
            tool_descriptions = self.tool_registry.get_tools_description()
        except Exception as e:
            logger.error(f"获取工具描述失败: {e}")
            tool_descriptions = "（工具描述获取失败）"
        
        enhanced_prompt = f"""你是一个智能助手，具备使用工具解决问题的能力。

            ## 核心原则
            1. **先思考，再行动**：在调用工具前，先分析用户需求，确定是否需要使用工具
            2. **选择合适的工具**：根据任务需求选择最适合的工具
            3. **正确传递参数**：确保传递给工具的参数格式正确、内容准确
            4. **处理工具结果**：根据工具返回的结果并分析，继续推理或给出最终答案
            5. {thinking_prompt}
            ## 工具使用指南
            - 当用户问题可以直接回答时，不必使用工具
            - 当需要获取实时信息、执行计算或操作外部系统时，使用工具
            - 可以连续同时调用多个工具来完成复杂任务
            - 如果工具调用失败，分析原因并尝试其他方案
            - 当收集到足够的信息后回答用户问题

            ## 可用工具
            {tool_descriptions}

            ## 响应格式
            - 如果不需要工具，直接回答用户问题
            - 工具返回结果后，基于结果给出清晰的回答

            {self.system_prompt or ''}\n
            """
            
        # 注入记忆系统提示和 Working Memory 便签本
        # enhanced_prompt += self._build_memory_prompt()
        
        # 注入所有激活 Skill 的 prompt
        enhanced_prompt += self._build_skills_prompt()

        return enhanced_prompt

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
                return self.context_manager.build_messages(
                    query=query,
                    history=self.history,
                    system_prompt=system_prompt,
                    include_history=True,
                    include_query=True,
                )
            except Exception as e:
                logger.warning(f"ContextManager 构建 messages 失败，回退默认拼接: {e}")

        messages: list[Any] = [SystemMessage(system_prompt)]
        self._append_runtime_history(messages)
        messages.append(UserMessage(query))
        return messages

        

    def get_thinking_history(self) -> list[str]:
        """
        获取思考历史
        
        Returns:
            思考过程列表
        """
        return self.thinking_history.copy()
    
    def clear_thinking_history(self) -> None:
        """清空思考历史"""
        self.thinking_history.clear()
    
    def get_last_thinking(self) -> Optional[str]:
        """
        获取最后一次思考内容
        
        Returns:
            最后一次思考内容，如果没有则返回 None
        """
        return self.thinking_history[-1] if self.thinking_history else None
