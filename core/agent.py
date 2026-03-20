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
from memory.V2.MemoryManage import MemoryManage
from context.manager import ContextManager
from context.source.base import BaseContextSource
import json
import threading
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
        memory_manage: V2 记忆系统（可选）
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        enable_async_tool: bool = False,
        async_max_workers: int = 4,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
    ):
        """
        初始化 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            description: Agent 描述
            config: 配置
            memory_manage: V2 记忆管理实例（可选）
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.description = description
        self.config = config or Config.from_env()
        self.history = []
        
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
        
        # V2 记忆系统 (MemoryManage)
        self.memory_manage = memory_manage
        self._unextracted_msg_count = 0
        self._memory_lock = threading.Lock()  # 保护后台提炼对 MemoryManage 的并发访问
        
        # 上下文工程管理器（可选）
        self.context_manager = context_manager
        
        # 自动注册 V2 记忆系统工具
        if self.memory_manage and self.tool_registry:
            self._register_v2_memory_tools()

    def _register_v2_memory_tools(self) -> None:
        if self.memory_manage and self.tool_registry:
            try:
                from Tool.builtin.memorytool import register_memory_tools
                register_memory_tools(self.memory_manage, self.tool_registry)
                logger.info("已自动注册 V2 记忆系统工具")
            except ImportError as e:
                logger.warning(f"未能导入 register_memory_tools: {e}")
                
    def with_memory(self, memory_manage: "MemoryManage") -> "BaseAgent":
        """
        方便地将 V2 版本的 MemoryManage 记忆系统绑定到 Agent
        """
        self.memory_manage = memory_manage
        if self.tool_registry is not None:
            self._register_v2_memory_tools()
        if self.context_manager is not None:
            from context.source.memory_source import MemoryContextSource
            memory_source = MemoryContextSource(memory_manage=memory_manage)
            self.context_manager.add_source(memory_source)
        return self

    def with_context(self, context_manager: "ContextManager") -> "BaseAgent":
        """绑定上下文管理器"""
        self.context_manager = context_manager
        if self.memory_manage is not None:
            from context.source.memory_source import MemoryContextSource
            memory_source = MemoryContextSource(memory_manage=self.memory_manage)
            self.context_manager.add_source(memory_source)
        return self
    
    def with_tool(self,tool_registry: Optional[ToolRegistry]) -> None:
        """设置工具注册表"""
        if(self.tool_registry is not None):
            logger.warning("工具注册表已存在!")
        self.tool_registry = tool_registry
        self.enable_tool = tool_registry is not None
    
    @abstractmethod
    def invoke(self, query: str, max_iter: int=10, temperature: float=0.7, **kwargs) -> str:
        """执行 Agent"""
        pass
    
    def add_message(self, message: Message) -> None:
        """添加消息到历史"""
        self.history.append(message)
        if len(self.history) > self.config.max_history_length:
            self.history.pop(0)
            
        # 触发后台记忆提炼
        self._check_and_trigger_background_memory()
        
    def _check_and_trigger_background_memory(self) -> None:
        """检查并触发后台记忆提炼"""
        if self.memory_manage is None:
            return
            
        # 设定阈值：例如每新增 5 条消息触发一次提炼
        trigger_threshold = 10
        self._unextracted_msg_count += 1

        if self._unextracted_msg_count >= trigger_threshold:
            self._unextracted_msg_count = 0
            
            # 提取需要提炼的对话内容
            recent_msgs = self.history[-trigger_threshold:]
            dialogue_text = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_msgs])
            
            # 使用独立线程异步处理，不阻塞主流程
            threading.Thread(
                target=self._extract_background_memory,
                args=(dialogue_text,),
                daemon=True
            ).start()
            
    def _extract_background_memory(self, dialogue_text: str) -> None:
        """后台异步执行语义/情景记忆提炼（线程安全）"""
        if not self.memory_manage or not self.tool_registry:
            return
        
        with self._memory_lock:
            try:
                logger.info("启动后台记忆提炼 (Background Memory Extraction)...")
                
                from agent.BasicAgent import BasicAgent
                from Tool.ToolRegistry import ToolRegistry
                
                # 使用一个独立的、无上下文包袱的 Agent 进行记忆提炼与保存
                bg_registry = ToolRegistry()
                add_memory_tool=self.tool_registry.get_Tool("add_memory_tool")
                if add_memory_tool:
                    bg_registry.registerTool(add_memory_tool)
                bg_agent = BasicAgent(
                    name="MemoryExtractor",
                    llm=self.llm,
                    enable_tool=True,
                    tool_registry=bg_registry,
                    system_prompt="你是一个专门负责后台记忆整理的AI核心。\n你的任务是分析这段多轮对话记录，提炼出重要的客观事实、用户的习惯与偏好以及发生的重要事件。\n你必须自己调用工具（如 add_memory_tool 等）将这些信息结构化地保存到记忆系统（semantic 和 episodic 面向长期，working 面向任务状态）中。\n保存完毕后只需回复'提取完成'，不需要啰嗦。"
                )
                summary_prompt = f"请提炼并保存以下对话记录到记忆系统中:\n{dialogue_text}"
                
                # 由于当前已经在独立线程中，调用 invoke 阻塞是可以接受的
                bg_agent.invoke(query=summary_prompt)
                
                logger.info("后台记忆提炼完成，对话已被 LLM 自主归档。")
                
            except Exception as e:
                logger.error(f"后台记忆提炼失败: {e}")
    
    def _build_memory_prompt(self) -> str:
        """构建记忆系统相关的 prompt 片段（供子类在 get_enhanced_prompt 中调用）
        
        包含：
        1. 记忆系统使用说明
        2. Working Memory 便签本内容全量注入
        
        Returns:
            记忆相关的 prompt 文本，无记忆系统时返回空字符串
        """
        if not getattr(self, "memory_manage", None):
            return ""
        
        prompt = """
## 你的记忆系统 (The Memory System)
你拥有一个高级记忆管理系统，能够跨越长期和短期存储知识。
- **必要性原则**：仅当当前对话上下文（History）不足以回答问题时，才考虑使用搜索工具。禁止对不需要记忆系统，显而易见或刚讨论过的信息进行重复搜索。
- **工作记忆 (Working Memory)**：遇到关键的约束、当前任务大纲或中间状态，请主动调用工具写入。这是你的“即时贴”，用于存放当前任务最核心的信息。
- **长期记忆搜索 (Search)**：遇到不知道的事实或历史脉络，请使用 search_memory_tool 到 semantic (语义) 或 episodic (情景) 记忆中检索。
- **记忆持久化**：只有你主动存入，未来的你才能回想起现在的经历和设定。
- **按需使用**：其他记忆工具（如删除、更新）仅在信息过时或用户要求时使用。
"""
        
        # 若 context_manager 已挂载 memory source，避免重复注入 Working Memory 全量文本
        context_has_memory_source = False
        if getattr(self, "context_manager", None) is not None:
            try:
                source_names = set(self.context_manager.builder.source_names)
                context_has_memory_source = "memory" in source_names
            except Exception:
                context_has_memory_source = False

        # 注入 Working Memory 便签本
        if "working" in self.memory_manage.memory_types:
            if context_has_memory_source:
                prompt += (
                    "\n\n【当前工作便签本（Working Memory）】见【记忆上下文】，已由上下文管理器注入】"
                    
                    "\n(注: 当任务结束或话题转变时，务必主动调用 memory_maintenance_tool 清理无用便签或用 remove_memory_tool 删除指定id内容)"
                    "\n(注: 遇到复杂任务时，请主动调用 add_memory_tool 记录约束条件和中间结论)"
                )
            else:
                working_memories = self.memory_manage.memory_types["working"].get_all_memories()
                if working_memories:
                    wm_texts = [f"- id:{m.id}: {m.content}" for m in working_memories]
                    wm_str = "\n".join(wm_texts)
                    prompt += f"\n\n【当前工作便签本（Working Memory）】:\n{wm_str}"
                else:
                    prompt += f"\n\n【当前工作便签本（Working Memory）】:\n(空)"
                prompt += "\n(注: 遇到复杂任务时，请主动调用 add_memory_tool 记录约束条件和中间结论)"
                prompt += "\n(注: 当任务结束或话题转变时，务必主动调用 memory_maintenance_tool 清理无用便签或用 remove_memory_tool 删除指定id内容)\n"
        
        return prompt
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        from .Message import UserMessage
        self.add_message(UserMessage(content))
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        from .Message import AssistantMessage
        self.add_message(AssistantMessage(content))
    
    def add_context_source(self, source:BaseContextSource) -> None:
        """添加上下文来源"""
        if self.context_manager is None:
            raise ParameterValidationError("上下文管理器未配置，无法添加上下文来源!")
        self.context_manager.add_source(source)
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()
        self._unextracted_msg_count = 0
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

    def addTool(self, tool) -> None:
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
        
        try:
            self.tool_registry.registry(tool)
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