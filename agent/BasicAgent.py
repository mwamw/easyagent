from typing_extensions import override
from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from typing import Optional, Any, TYPE_CHECKING
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from Tool.AsyncToolExecutor import AsyncToolExecutor
import asyncio
import json
import logging
from core.Exception import *

if TYPE_CHECKING:
    from memory.base import BaseMemory

# 配置日志
logger = logging.getLogger(__name__)

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
        memory: Optional["BaseMemory"] = None,
        enable_memory: bool = False,
        verbose_thinking: bool = False,
        enable_async_tool: bool = False,
        async_max_workers: int = 4,
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
            memory: 记忆系统实例（可选）
            verbose_thinking: 是否显示 LLM 的思考过程（默认 True）
            enable_memory: 是否启用记忆
            enable_async_tool: 是否启用异步工具执行（并行执行多个工具）
            async_max_workers: 异步执行器线程池大小
            
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
            memory=memory,
            enable_memory=enable_memory,
            enable_tool=enable_tool,
            tool_registry=tool_registry,
            enable_async_tool=enable_async_tool,
            async_max_workers=async_max_workers,
        )
        

        self.verbose_thinking = verbose_thinking
        self.thinking_history: list[str] = []  # 记录思考过程

        logger.info(f"BasicAgent '{name}' 初始化完成，工具调用: {'启用' if enable_tool else '禁用'}，异步执行: {'启用' if enable_async_tool else '禁用'}，记忆: {'启用' if self.enable_memory else '禁用'}，provider: {llm.provide}")


    @override
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
        # 参数验证
        self._validate_invoke_params(query, max_iter, temperature)
        
        messages: list[Message | dict[str, str]] = []
 
        if self.enable_tool :
            logger.info("使用工具模式调用智能体")
            return self.invoke_with_tool(query, messages, max_iter, temperature)
        else:
            logger.info("使用普通模式调用智能体")
            try:
                messages.append(SystemMessage(self.get_enhanced_prompt()))
                for message in self.history:
                    messages.append(message)
                messages.append(UserMessage(query))
                
                response = self.llm.invoke(messages, temperature=temperature, **kwargs)
                
                # 验证响应
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                if not isinstance(response, str):
                    logger.warning(f"LLM 响应类型不是字符串: {type(response).__name__}，尝试转换...")
                    response = str(response)
                
                self.history.append(UserMessage(query))
                self.history.append(AssistantMessage(response))
                return response
                
            except LLMInvokeError:
                raise
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                raise LLMInvokeError(f"LLM 调用失败: {e}") from e

    def stream_invoke(self,query: str,temperature: float = 0.7, **kwargs):
        if self.enable_tool:
            logger.info("使用工具模式不支持流式调用智能体")
            raise NotImplementedError("工具模式不支持流式调用智能体")
        else:
            messages=[]
            messages.append(SystemMessage(self.get_enhanced_prompt()))
            for message in self.history:
                messages.append(message)
            messages.append(UserMessage(query))
            final_results=[]
            for chunk in self.llm.think(messages, temperature=temperature, **kwargs):
                final_results.append(chunk)
                
            self.history.append(UserMessage(query))
            self.history.append(AssistantMessage("".join(final_results)))
            return "".join(final_results)

    async def invoke_async(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """
        异步调用智能体，支持并行工具执行
        
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
        # 参数验证
        self._validate_invoke_params(query, max_iter, temperature)
        
        messages: list[Message | dict[str, str]] = []
        
        if self.enable_tool:
            if self.enable_async_tool and self.async_executor:
                logger.info("使用异步工具模式调用智能体")
                return await self.invoke_with_tool_async(query, messages, max_iter, temperature)
            else:
                logger.info("使用同步工具模式调用智能体（异步包装）")
                return self.invoke_with_tool(query, messages, max_iter, temperature)
        else:
            logger.info("使用普通模式调用智能体")
            try:
                messages.append(SystemMessage(self.get_enhanced_prompt()))
                for message in self.history:
                    messages.append(message)
                messages.append(UserMessage(query))
                
                response = self.llm.invoke(messages, temperature=temperature, **kwargs)
                
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                if not isinstance(response, str):
                    response = str(response)
                
                self.history.append(UserMessage(query))
                self.history.append(AssistantMessage(response))
                return response
                
            except LLMInvokeError:
                raise
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                raise LLMInvokeError(f"LLM 调用失败: {e}") from e






    def invoke_with_tool(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7
    ) -> str:
        """
        使用工具调用模式调用智能体
        
        Args:
            query: 用户输入
            messages: 消息列表
            max_iter: 最大迭代次数
            temperature: 温度参数
            
        Returns:
            智能体返回结果
            
        Raises:
            ToolRegistryError: 工具注册表未配置
            LLMInvokeError: LLM 调用失败
        """
        self.enable_tool = True
        
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        enhanced_prompt = self.get_enhanced_prompt()
        messages.append(SystemMessage(enhanced_prompt))
        
        for message in self.history:
            messages.append(message)
        messages.append(UserMessage(query))
        
        final_response: Optional[str] = None
        iteration_count = 0
        
        while max_iter > 0:
            iteration_count += 1
            logger.debug(f"工具调用迭代 {iteration_count}")
            
            try:
                response = self.llm.invoke_with_tools(
                    messages,
                    self.tool_registry.get_openai_tools(),
                    temperature=temperature
                )
                
                # 验证响应对象
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                # 格式化 assistant 响应（处理不同 Provider 的格式差异）
                formatted_response = self.llm.format_assistant_response(response)
                messages.append(formatted_response)
            except LLMInvokeError:
                raise
            except Exception as e:
                logger.error(f"智能体调用失败: {e}")
                final_response = f"智能体调用失败: {str(e)}"
                break
            
            # 捕获 LLM 的思考过程（content 字段）
            thinking_content = getattr(response, 'reasoning_content', None)
            if thinking_content and hasattr(response, 'tool_calls') and response.tool_calls:
                # LLM 在调用工具前输出了思考过程
                self.thinking_history.append(thinking_content)
                if self.verbose_thinking:
                    logger.info(f"💭 思考: {thinking_content}")
                # 将思考内容添加到消息历史
                messages.append(AssistantMessage(thinking_content))
            
            # 检查是否有工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    # 先尝试获取工具名称，确保后续异常处理中可用
                    tool_name = "unknown_tool"
                    tool_id = getattr(tool_call, 'id', 'unknown')
                    
                    try:
                        tool_name = self._safe_get_tool_name(tool_call)
                    except Exception as e:
                        logger.warning(f"获取工具名称失败: {e}，使用默认名称")
                        # 尝试从 function.name 直接获取
                        if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                            tool_name = tool_call.function.name or "unknown_tool"
                    
                    try:
                        tool_args = self._safe_parse_tool_args(tool_call)
                        
                        logger.info(f"执行工具: {tool_name}，参数: {tool_args}")
                        tool_result = self._safe_execute_tool(tool_name, tool_args)
                        
                        # 使用 LLM 的 format_tool_result 方法获取正确格式
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
                # 没有工具调用，获取最终响应
                content = getattr(response, 'content', None)
                if content is not None:
                    messages.append(AssistantMessage(content))
                    final_response = content
                else:
                    logger.warning("LLM 响应中没有内容")
                    final_response = ""
                break
            
            max_iter -= 1
        
        # 检查是否超过最大迭代次数
        if final_response is None:
            logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
            final_response = "超过最大迭代次数，智能体调用失败!"
        
        self.history.append(UserMessage(query))
        self.history.append(AssistantMessage(final_response))
        return final_response

    async def invoke_with_tool_async(
        self,
        query: str,
        messages: list[Message | dict[str, str]],
        max_iter: int = 10,
        temperature: float = 0.7
    ) -> str:
        """
        异步使用工具调用模式调用智能体，支持并行执行多个工具
        
        Args:
            query: 用户输入
            messages: 消息列表
            max_iter: 最大迭代次数
            temperature: 温度参数
            
        Returns:
            智能体返回结果
            
        Raises:
            ToolRegistryError: 工具注册表未配置或异步执行器未初始化
            LLMInvokeError: LLM 调用失败
        """
        self.enable_tool = True
        
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        if self.async_executor is None:
            raise ToolRegistryError("异步工具执行器未初始化！请启用 enable_async_tool")
        
        enhanced_prompt = self.get_enhanced_prompt()
        messages.append(SystemMessage(enhanced_prompt))
        
        for message in self.history:
            messages.append(message)
        messages.append(UserMessage(query))
        
        final_response: Optional[str] = None
        iteration_count = 0
        
        while max_iter > 0:
            iteration_count += 1
            logger.debug(f"异步工具调用迭代 {iteration_count}")
            
            try:
                response = self.llm.invoke_with_tools(
                    messages,
                    self.tool_registry.get_openai_tools(),
                    temperature=temperature
                )
                
                if response is None:
                    raise LLMInvokeError("LLM 返回了空响应!")
                
                formatted_response = self.llm.format_assistant_response(response)
                messages.append(formatted_response)
            except LLMInvokeError:
                raise
            except Exception as e:
                logger.error(f"智能体调用失败: {e}")
                final_response = f"智能体调用失败: {str(e)}"
                break
            
            # 捕获 LLM 的思考过程
            thinking_content = getattr(response, 'reasoning_content', None)
            if thinking_content and hasattr(response, 'tool_calls') and response.tool_calls:
                self.thinking_history.append(thinking_content)
                if self.verbose_thinking:
                    logger.info(f"💭 思考: {thinking_content}")
                messages.append(AssistantMessage(thinking_content))
            
            # 检查是否有工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 准备并行执行的任务
                tasks_to_execute = []
                tool_metadata = []  # 保存 tool_id 和 tool_name 用于结果处理
                
                for tool_call in response.tool_calls:
                    tool_name = "unknown_tool"
                    tool_id = getattr(tool_call, 'id', 'unknown')
                    
                    try:
                        tool_name = self._safe_get_tool_name(tool_call)
                        tool_args = self._safe_parse_tool_args(tool_call)
                        
                        tasks_to_execute.append({
                            "tool_name": tool_name,
                            "parameters": tool_args
                        })
                        tool_metadata.append({
                            "tool_id": tool_id,
                            "tool_name": tool_name
                        })
                        logger.info(f"准备执行工具: {tool_name}，参数: {tool_args}")
                        
                    except Exception as e:
                        logger.error(f"解析工具调用失败: {e}")
                        # 即使解析失败也添加错误结果
                        error_msg = f"工具 '{tool_name}' 解析失败: {str(e)}"
                        tool_msg = self.llm.format_tool_result(error_msg, tool_id, tool_name)
                        messages.append(tool_msg)
                
                # 并行执行所有工具
                if tasks_to_execute:
                    logger.info(f"🚀 并行执行 {len(tasks_to_execute)} 个工具...")
                    try:
                        results = await self.async_executor.execute_tools_parallel(tasks_to_execute)
                        
                        # 将结果添加到消息历史
                        for i, result in enumerate(results):
                            meta = tool_metadata[i]
                            tool_msg = self.llm.format_tool_result(
                                str(result), 
                                meta["tool_id"], 
                                meta["tool_name"]
                            )
                            messages.append(tool_msg)
                            logger.info(f"✅ 工具 '{meta['tool_name']}' 执行完成")
                            
                    except Exception as e:
                        logger.error(f"并行执行工具失败: {e}")
                        # 为所有任务添加错误结果
                        for meta in tool_metadata:
                            error_msg = f"工具执行失败: {str(e)}"
                            tool_msg = self.llm.format_tool_result(
                                error_msg, 
                                meta["tool_id"], 
                                meta["tool_name"]
                            )
                            messages.append(tool_msg)
            else:
                # 没有工具调用，获取最终响应
                content = getattr(response, 'content', None)
                if content is not None:
                    messages.append(AssistantMessage(content))
                    final_response = content
                else:
                    logger.warning("LLM 响应中没有内容")
                    final_response = ""
                break
            
            max_iter -= 1
        
        # 检查是否超过最大迭代次数
        if final_response is None:
            logger.warning(f"超过最大迭代次数 ({iteration_count})，智能体调用失败")
            final_response = "超过最大迭代次数，智能体调用失败!"
        
        self.history.append(UserMessage(query))
        self.history.append(AssistantMessage(final_response))
        return final_response

 
    @override
    def get_enhanced_prompt(self) -> str:

        """
        获取增强后的系统提示词
        
        Returns:
            增强后的系统提示词
        """
        thinking_prompt:str=""
        if self.verbose_thinking:
            thinking_prompt="""
            如何需要使用工具，请同时给出思考内容和工具调用内容
            """
        else:
            thinking_prompt="""
            如何需要使用工具，请直接调用工具
            """
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
            4. **处理工具结果**：根据工具返回的结果，继续推理或给出最终答案

            ## 工具使用指南
            - 当用户问题可以直接回答时，不必使用工具
            - 当需要获取实时信息、执行计算或操作外部系统时，使用工具
            - 可以连续调用多个工具来完成复杂任务
            - 如果工具调用失败，分析原因并尝试其他方案
            - 当收集到足够的信息后回答用户问题

            ## 可用工具
            {tool_descriptions}

            ## 响应格式
            - {thinking_prompt}
            - 如果不需要工具，直接回答用户问题
            - 工具返回结果后，基于结果给出清晰的回答

            {self.system_prompt or ''}
            """
        return enhanced_prompt

        

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
