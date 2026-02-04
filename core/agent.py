"""
Agent 基类模块
"""
from core.Exception import ToolExecutionError
from .Message import Message
from typing import Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from .Config import Config
from .llm import EasyLLM
from Tool.ToolRegistry import ToolRegistry
if TYPE_CHECKING:
    from memory.base import BaseMemory
import json
from Tool.BaseTool import Tool
from .Exception import *
from Tool.AsyncToolExecutor import AsyncToolExecutor
import logging
logger = logging.getLogger(__name__)
class BaseAgent(ABC):
    """
    Agent 抽象基类
    
    所有 Agent 实现都应该继承此类。
    提供可选的记忆系统支持。
    
    Attributes:
        name: Agent 名称
        llm: LLM 实例
        system_prompt: 系统提示词
        description: Agent 描述
        config: 配置
        history: 对话历史（简单列表）
        memory: 记忆系统（可选）
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        memory: Optional["BaseMemory"] = None,
        enable_memory: bool = False,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        enable_async_tool: bool = False,
        async_max_workers: int = 4,
    ):
        """
        初始化 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            description: Agent 描述
            config: 配置
            memory: 记忆系统实例（可选）
            enable_memory: 是否启用记忆（如果 memory 为 None 则使用默认记忆）
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.description = description
        self.config = config or Config.from_env()
        self.history = []
        
        # 记忆系统
        self.enable_memory = enable_memory or (memory is not None)
        self._memory = memory
        
        # 工具系统
        if enable_tool and not tool_registry:
            raise ToolRegistryError("启用工具调用时必须提供 ToolRegistry!")
        
        if tool_registry is not None and not isinstance(tool_registry, ToolRegistry):
            raise ParameterValidationError(f"tool_registry 必须是 ToolRegistry 类型，收到: {type(tool_registry).__name__}")
        
        self.enable_tool = enable_tool or (tool_registry is not None)
        self.tool_registry = tool_registry
        
                
        # 异步工具执行配置
        self.enable_async_tool = enable_async_tool
        self.async_max_workers = async_max_workers
        self.async_executor: Optional[AsyncToolExecutor] = None
        
        if enable_async_tool and tool_registry:
            self.async_executor = AsyncToolExecutor(tool_registry, max_workers=async_max_workers)
        

        if self.enable_memory and self._memory is None:
            # 延迟导入，避免循环依赖
            from memory.buffer import ConversationBufferMemory
            self._memory = ConversationBufferMemory(max_messages=20)
    
    @property
    def memory(self) -> Optional["BaseMemory"]:
        """获取记忆系统"""
        return self._memory
    
    @memory.setter
    def memory(self, value: Optional["BaseMemory"]) -> None:
        """设置记忆系统"""
        self._memory = value
        self.enable_memory = value is not None
    
    def with_tool(self,tool_registry: Optional[ToolRegistry]) -> None:
        """设置工具注册表"""
        if(self.tool_registry is not None):
            logger.warning("工具注册表已存在!")
        self.tool_registry = tool_registry
        self.enable_tool = tool_registry is not None
    
    @abstractmethod
    def invoke(self, query: str, max_iter: int, temperature: float, **kwargs) -> str:
        """执行 Agent"""
        pass
    
    def add_message(self, message: Message) -> None:
        """添加消息到历史"""
        self.history.append(message)
        if len(self.history) > self.config.max_history_length:
            self.history.pop(0)
        
        # 同时添加到记忆系统
        if self.enable_memory and self._memory is not None:
            self._memory.add_message(message)
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        from .Message import UserMessage
        self.add_message(UserMessage(content))
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        from .Message import AssistantMessage
        self.add_message(AssistantMessage(content))
    
    def get_memory_context(self, query: Optional[str] = None) -> str:
        """
        获取记忆上下文
        
        Args:
            query: 查询（用于向量记忆的语义检索）
            
        Returns:
            格式化的记忆上下文
        """
        if not self.enable_memory or self._memory is None:
            return ""
        
        # 检查是否支持 query 参数（向量记忆）
        try:
            return self._memory.get_context(query=query)
        except TypeError:
            return self._memory.get_context()
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()
        if self.enable_memory and self._memory is not None:
            self._memory.clear()
        logger.info("对话历史已清空")
    def get_history(self):
        """获取对话历史"""
        return self.history

    def get_history_length(self) -> int:
        """
        获取对话历史长度
        
        Returns:
            对话历史条数
        """
        return len(self.history)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, description={self.description})"
    

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
    @abstractmethod
    def get_enhanced_prompt(self) -> str:
        pass
    

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
    def set_async_tool_mode(self, enabled: bool, max_workers: int = 4) -> None:
        """
        设置是否启用异步工具执行
        
        Args:
            enabled: 是否启用异步执行
            max_workers: 线程池大小
            
        Raises:
            ToolRegistryError: 启用异步但未配置 ToolRegistry
        """
        if not isinstance(enabled, bool):
            raise ParameterValidationError(f"enabled 参数必须是布尔类型，收到: {type(enabled).__name__}")
        
        if enabled and self.tool_registry is None:
            raise ToolRegistryError("启用异步工具执行时必须先设置 ToolRegistry!")
        
        self.enable_async_tool = enabled
        self.async_max_workers = max_workers
        
        if enabled and self.tool_registry:
            self.async_executor = AsyncToolExecutor(self.tool_registry, max_workers=max_workers)
        else:
            self.async_executor = None
        
        logger.info(f"异步工具执行已{'启用' if enabled else '禁用'}")