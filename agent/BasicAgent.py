from typing_extensions import override
from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from typing import Optional, Any, TYPE_CHECKING
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
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
            enable_memory=enable_memory
        )
        self.enable_tool = enable_tool
        
        if enable_tool and not tool_registry:
            raise ToolRegistryError("启用工具调用时必须提供 ToolRegistry!")
        
        if tool_registry is not None and not isinstance(tool_registry, ToolRegistry):
            raise ParameterValidationError(f"tool_registry 必须是 ToolRegistry 类型，收到: {type(tool_registry).__name__}")
        
        self.tool_registry = tool_registry
        self.verbose_thinking = verbose_thinking
        self.thinking_history: list[str] = []  # 记录思考过程
        
        logger.info(f"BasicAgent '{name}' 初始化完成，工具调用: {'启用' if enable_tool else '禁用'}，记忆: {'启用' if self.enable_memory else '禁用'}，provider: {llm.provide}")

    def _validate_invoke_params(self, query: str, max_iter: int, temperature: float) -> None:
        """
        验证 invoke 方法的参数
        
        Args:
            query: 用户输入
            max_iter: 最大迭代次数
            temperature: 温度参数
            
        Raises:
            ParameterValidationError: 参数验证失败
        """
        if not query or not isinstance(query, str):
            raise ParameterValidationError("用户输入 'query' 必须是非空字符串!")
        
        if query.strip() == "":
            raise ParameterValidationError("用户输入 'query' 不能只包含空白字符!")
        
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ParameterValidationError(f"max_iter 必须是正整数，收到: {max_iter}")
        
        if max_iter > 100:
            logger.warning(f"max_iter 设置过大 ({max_iter})，可能导致过长的执行时间")
        
        if not isinstance(temperature, (int, float)):
            raise ParameterValidationError(f"temperature 必须是数值类型，收到: {type(temperature).__name__}")
        
        if temperature < 0 or temperature > 2:
            raise ParameterValidationError(f"temperature 必须在 0 到 2 之间，收到: {temperature}")

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


    def set_enable_tool(self, enabled: bool) -> None:
        """
        设置是否启用工具调用
        
        Args:
            enabled: 是否启用
            
        Raises:
            ToolRegistryError: 启用工具但未配置 ToolRegistry
        """
        if not isinstance(enabled, bool):
            raise ParameterValidationError(f"enabled 参数必须是布尔类型，收到: {type(enabled).__name__}")
        
        if enabled and self.tool_registry is None:
            raise ToolRegistryError("启用工具调用时必须先设置 ToolRegistry!")
        
        self.enable_tool = enabled
        logger.info(f"工具调用已{'启用' if enabled else '禁用'}")

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

    def _safe_get_tool_name(self, tool_call: Any) -> str:
        """
        安全获取工具名称
        
        Args:
            tool_call: 工具调用对象
            
        Returns:
            工具名称
            
        Raises:
            ToolExecutionError: 无法获取工具名称
        """
        try:
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                name = tool_call.function.name
                if name and isinstance(name, str):
                    return name
            raise ToolExecutionError("工具调用对象中没有有效的工具名称")
        except Exception as e:
            raise ToolExecutionError(f"获取工具名称失败: {e}") from e

    def _safe_parse_tool_args(self, tool_call: Any) -> dict:
        """
        安全解析工具参数
        
        Args:
            tool_call: 工具调用对象
            
        Returns:
            解析后的参数字典
            
        Raises:
            ToolExecutionError: 参数解析失败
        """
        try:
            if not hasattr(tool_call, 'function') or not hasattr(tool_call.function, 'arguments'):
                raise ToolExecutionError("工具调用对象中没有 arguments 属性")
            
            arguments = tool_call.function.arguments
            
            # 处理不同类型的参数
            if arguments is None or arguments == "":
                return {}
            
            if isinstance(arguments, dict):
                return arguments
            
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    if not isinstance(parsed, dict):
                        raise ToolExecutionError(f"工具参数解析结果不是字典类型: {type(parsed).__name__}")
                    return parsed
                except json.JSONDecodeError as e:
                    raise ToolExecutionError(f"工具参数 JSON 解析失败: {e}") from e
            
            raise ToolExecutionError(f"不支持的参数类型: {type(arguments).__name__}")
            
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"解析工具参数时发生错误: {e}") from e

    def _safe_execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        安全执行工具
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ToolExecutionError: 工具执行失败
        """
        if self.tool_registry is None:
            raise ToolExecutionError("工具注册表未配置!")
        
        try:
            result = self.tool_registry.executeTool(tool_name, tool_args)
            
            # 确保返回字符串
            if result is None:
                return "工具执行完成，无返回结果"
            
            if not isinstance(result, str):
                return str(result)
            
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"工具 '{tool_name}' 执行失败: {e}") from e

    def executeTool(self, tool_name: str, tool_args: dict) -> str:
        """
        执行工具（保留原有接口）
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ToolRegistryError: 工具注册表未配置
            ToolExecutionError: 工具执行失败
        """
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        if not tool_name or not isinstance(tool_name, str):
            raise ParameterValidationError("工具名称必须是非空字符串!")
        
        if not isinstance(tool_args, dict):
            raise ParameterValidationError(f"工具参数必须是字典类型，收到: {type(tool_args).__name__}")
        
        return self._safe_execute_tool(tool_name, tool_args)

    def addTool(self, tool: Tool) -> None:
        """
        添加工具
        
        Args:
            tool: 工具实例
            
        Raises:
            ToolRegistryError: 工具注册表未配置
            ParameterValidationError: 参数验证失败
        """
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")
        
        if tool is None:
            raise ParameterValidationError("工具实例不能为空!")
        
        if not isinstance(tool, Tool):
            raise ParameterValidationError(f"tool 必须是 Tool 类型，收到: {type(tool).__name__}")
        
        try:
            self.tool_registry.registerTool(tool)
            logger.info(f"成功添加工具: {getattr(tool, 'name', 'unknown')}")
        except Exception as e:
            raise ToolRegistryError(f"添加工具失败: {e}") from e

    def get_tools_description(self) :
        """
        获取工具描述
        
        Returns:
            工具描述字符串
            
        Raises:
            ToolRegistryError: 工具注册表未配置或工具未启用
        """
        if self.tool_registry is None:
            raise ToolRegistryError("工具注册表未配置!")
        
        if not self.enable_tool:
            raise ToolRegistryError("工具调用未启用!")
        
        try:
            return self.tool_registry.get_tools_description()
        except Exception as e:
            raise ToolRegistryError(f"获取工具描述失败: {e}") from e

    def get_openai_tools(self) -> list:
        """
        获取 OpenAI 格式的工具列表
        
        Returns:
            OpenAI 格式的工具列表
            
        Raises:
            ToolRegistryError: 工具注册表未配置
        """
        if self.tool_registry is None:
            raise ToolRegistryError("工具注册表未配置!")
        
        try:
            return self.tool_registry.get_openai_tools()
        except Exception as e:
            raise ToolRegistryError(f"获取 OpenAI 工具列表失败: {e}") from e

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

## 可用工具
{tool_descriptions}

## 响应格式
- {thinking_prompt}
- 如果不需要工具，直接回答用户问题
- 工具返回结果后，基于结果给出清晰的回答

{self.system_prompt or ''}
"""
        return enhanced_prompt

    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()
        logger.info("对话历史已清空")

    def get_history_length(self) -> int:
        """
        获取对话历史长度
        
        Returns:
            对话历史条数
        """
        return len(self.history)

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