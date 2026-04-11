"""
Agent 基类模块
"""
from core.Exception import ToolExecutionError
from .Message import Message
from typing import Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from datetime import datetime
from .Config import Config
from .llm import EasyLLM
from Tool.ToolRegistry import ToolRegistry
from context.manager import ContextManager
from context.source.base import BaseContextSource
from context.token.counter import TokenCounter
from .callbacks import CallbackManager
from skill.manager import SkillManager
from prompt import build_memory_prompt_section
import json
import asyncio
import threading
import concurrent.futures
from Tool.BaseTool import Tool
from .Exception import *
import logging

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage

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
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        callback_manager: Optional["CallbackManager"] = None,
        skill_manager: Optional["SkillManager"] = None,
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
            callback_manager: 回调管理器（可选）
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.description = description
        self.config = config or Config.from_env()
        self.history: list[Any] = []
        
        # 回调系统
        self.callback_manager = callback_manager or CallbackManager()
        
        # 工具系统
        if enable_tool and not tool_registry:
            raise ToolRegistryError("启用工具调用时必须提供 ToolRegistry!")
        
        if tool_registry is not None and not isinstance(tool_registry, ToolRegistry):
            raise ParameterValidationError(f"tool_registry 必须是 ToolRegistry 类型，收到: {type(tool_registry).__name__}")
        
        self.enable_tool = enable_tool or (tool_registry is not None)
        self.tool_registry = tool_registry
        
        # V2 记忆系统 (MemoryManage)
        self.memory_manage = memory_manage
        self._unextracted_msg_count = 0
        self._memory_lock = threading.Lock()  # 保护后台提炼对 MemoryManage 的并发访问
        
        # 上下文工程管理器（可选）
        self.context_manager = context_manager
        self._last_context_usage: dict[str, Any] = {}
        self._context_usage_counter = TokenCounter()
        
        # Skill 管理器
        self.skill_manager = skill_manager or SkillManager()
        self.skill_manager.bind_agent(self)
        
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
        方便地将 V2 版本的 MemoryManage 记忆系统绑定到 Agent。

        内部会自动创建 MemorySkill 并注册到 SkillManager。
        如果已经通过 with_skill 手动注册了 MemorySkill，则跳过自动注册。
        """
        self.memory_manage = memory_manage
        
        # 通过 MemorySkill 实现（新路径）
        if not self.skill_manager.has_skill("memory"):
            try:
                from skill.builtin.memory_skill import MemorySkill
                # 如果有 context_manager，MemorySkill 会自动提供 context_source
                include_ctx = self.context_manager is not None
                skill = MemorySkill(
                    memory_manage=memory_manage,
                    include_context_source=include_ctx,
                )
                self.skill_manager.register(skill)
                logger.info("已通过 MemorySkill 注册 V2 记忆系统")
            except ImportError:
                # 回退到旧方式
                logger.warning("MemorySkill 导入失败，使用旧方式注册记忆工具")
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
    
    def with_tool(self, tool_registry: Optional[ToolRegistry]=None) -> None:
        """设置工具注册表"""
        if(self.tool_registry is not None):
            logger.warning("工具注册表已存在!")
            return
        if(tool_registry is None):
            logger.warning("工具注册表为空!")
            self.tool_registry=ToolRegistry()
            self.enable_tool = True
            return
        self.tool_registry = tool_registry
        self.enable_tool = tool_registry is not None

    # ==================== Skill 管理 API ====================

    def with_skill(self, skill) -> "BaseAgent":
        """
        添加并激活一个 Skill
        
        Args:
            skill: BaseSkill 实例
            
        Returns:
            self（支持链式调用）
        """
        # 确保有 ToolRegistry
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()
            self.enable_tool = True
        
        self.skill_manager.register(skill)
        return self

    def remove_skill(self, name: str) -> None:
        """移除一个 Skill（先停用再注销）"""
        self.skill_manager.unregister(name)

    def activate_skill(self, name: str) -> None:
        """激活指定 Skill"""
        self.skill_manager.activate(name)

    def deactivate_skill(self, name: str) -> None:
        """停用指定 Skill"""
        self.skill_manager.deactivate(name)

    def _build_skills_prompt(self, exclude_names: Optional[set[str]] = None) -> str:
        """构建所有激活 Skill 的 prompt"""
        return self.skill_manager.build_skills_prompt(exclude_names=exclude_names)

    def _get_active_memory_skill(self) -> Any | None:
        """获取激活中的 MemorySkill（如果存在）。"""
        try:
            for skill in self.skill_manager.get_active_skills():
                if skill.name == "memory":
                    return skill
        except Exception:
            return None
        return None
    
    @abstractmethod
    def invoke(self, query: str, max_iter: int=10, temperature: float=0.7, **kwargs) -> str:
        """同步执行 Agent"""
        pass
    
    async def ainvoke(self, query: str, max_iter: int=10, temperature: float=0.7, **kwargs) -> str:
        """异步执行 Agent（子类可覆写，默认回退到同步）"""
        return self.invoke(query, max_iter=max_iter, temperature=temperature, **kwargs)
    
    def _append_history_entry(self, message: Any) -> None:
        """向 history 追加一条消息，支持 Message 或 provider-specific dict。"""
        self.history.append(message)
        if len(self.history) > self.config.max_history_length:
            self.history.pop(0)

        # 触发后台记忆提炼
        self._check_and_trigger_background_memory()

    def add_message(self, message: Any) -> None:
        """添加消息到历史"""
        self._append_history_entry(message)

    def add_messages(self, messages: list[Any]) -> None:
        """批量添加消息到历史。"""
        for message in messages:
            self._append_history_entry(message)

    @staticmethod
    def _history_entry_to_role_content(message: Any) -> tuple[str, str]:
        """提取 history 条目的 role/content，用于摘要与调试。"""
        if isinstance(message, dict):
            role = message.get("role") or message.get("type") or "unknown"
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "unknown")
            content = getattr(message, "content", "")

        if isinstance(content, str):
            return str(role), content
        return str(role), json.dumps(content, ensure_ascii=False, default=str)
        
    def _check_and_trigger_background_memory(self) -> None:
        """检查并触发后台记忆提炼"""
        if self.memory_manage is None:
            return
            
        # 设定阈值：例如每新增 5 条消息触发一次提炼
        trigger_threshold = self.config.trigger_threshold
        self._unextracted_msg_count += 1

        if self._unextracted_msg_count >= trigger_threshold:
            self._unextracted_msg_count = 0
            
            # 提取需要提炼的对话内容
            recent_msgs = self.history[-trigger_threshold:]
            dialogue_text = "\n".join(
                [
                    f"{role}: {content}"
                    for role, content in (
                        self._history_entry_to_role_content(msg) for msg in recent_msgs
                    )
                ]
            )
            
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
                add_memory_tool=self.tool_registry.get_tool("add_memory_tool")
                if add_memory_tool:
                    bg_registry.register_tool(add_memory_tool)
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
        memory_manage = getattr(self, "memory_manage", None)
        memory_skill = self._get_active_memory_skill()
        if memory_manage is None and memory_skill is not None:
            memory_manage = getattr(memory_skill, "memory_manage", None)

        if not memory_manage:
            return ""

        # 若 context_manager 已挂载 memory source，避免重复注入 Working Memory 全量文本
        context_has_memory_source = False
        if getattr(self, "context_manager", None) is not None:
            try:
                source_names = set(self.context_manager.builder.source_names) #type: ignore
                context_has_memory_source = "memory" in source_names
            except Exception:
                context_has_memory_source = False

        supported_memory_types: list[str] | None = None
        try:
            supported_memory_types = list(getattr(memory_manage, "memory_types", {}).keys())
        except Exception:
            supported_memory_types = None

        working_memory_entries: list[str] = []
        has_working_memory = False
        try:
            if "working" in memory_manage.memory_types: #type: ignore[attr-defined]
                has_working_memory = True
                if not context_has_memory_source:
                    working_memories = memory_manage.memory_types["working"].get_all_memories() #type: ignore[index]
                    working_memory_entries = [
                        f"- id:{memory.id}: {memory.content}"
                        for memory in working_memories
                    ]
        except Exception:
            working_memory_entries = ["(读取失败)"]
            has_working_memory = True

        return build_memory_prompt_section(
            supported_memory_types=supported_memory_types,
            working_memory_entries=working_memory_entries,
            working_memory_managed_by_context=context_has_memory_source,
            include_working_memory=has_working_memory,
        )
    
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

    def _get_serializable_state(self) -> dict[str, Any]:
        """返回子类需要补充持久化的状态。"""
        return {
            "last_context_usage": self.get_context_usage(),
        }

    def _restore_serializable_state(self, state: Optional[dict[str, Any]]) -> None:
        """恢复子类持久化状态。"""
        state = state or {}
        self._set_last_context_usage(state.get("last_context_usage") or {})
        return None

    @classmethod
    def _supports_session_restore(cls) -> bool:
        """当前 Agent 类型是否支持从会话快照恢复。"""
        return True

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    def _build_session_snapshot(self) -> dict[str, Any]:
        tool_names = []
        if self.tool_registry is not None:
            try:
                tool_names = self.tool_registry.get_tool_names()
            except Exception:
                tool_names = []

        registered_skills: list[str] = []
        active_skills: list[str] = []
        try:
            registered_skills = [item["name"] for item in self.skill_manager.list_skills()]
            active_skills = [skill.name for skill in self.skill_manager.get_active_skills()]
        except Exception:
            registered_skills = []
            active_skills = []

        return {
            "schema_version": 1,
            "agent_type": self.__class__.__name__,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "config": self.config.to_dict(),
            "enable_tool": self.enable_tool,
            "llm": self._make_json_safe(
                {
                    "provider_name": getattr(self.llm, "provider_name", None),
                    "model": getattr(self.llm, "model", None),
                    "base_url": getattr(self.llm, "base_url", None),
                }
            ),
            "tool_names": tool_names,
            "registered_skills": registered_skills,
            "active_skills": active_skills,
            "has_memory_manage": self.memory_manage is not None,
            "has_context_manager": self.context_manager is not None,
            "state": self._make_json_safe(self._get_serializable_state()),
        }

    @classmethod
    def _build_base_constructor_kwargs(
        cls,
        snapshot: dict[str, Any],
        llm: EasyLLM,
        tool_registry: Optional["ToolRegistry"] = None,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        callback_manager: Optional["CallbackManager"] = None,
        skill_manager: Optional["SkillManager"] = None,
    ) -> dict[str, Any]:
        config_data = snapshot.get("config") or {}
        config = Config(**config_data) if config_data else None
        requested_enable_tool = bool(snapshot.get("enable_tool", False))
        effective_enable_tool = requested_enable_tool and tool_registry is not None

        return {
            "name": snapshot["name"],
            "llm": llm,
            "system_prompt": snapshot.get("system_prompt"),
            "enable_tool": effective_enable_tool,
            "tool_registry": tool_registry,
            "description": snapshot.get("description"),
            "config": config,
            "memory_manage": memory_manage,
            "context_manager": context_manager,
            "callback_manager": callback_manager,
            "skill_manager": skill_manager,
        }

    @classmethod
    def _build_constructor_kwargs_from_snapshot(
        cls,
        snapshot: dict[str, Any],
        llm: EasyLLM,
        tool_registry: Optional["ToolRegistry"] = None,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        callback_manager: Optional["CallbackManager"] = None,
        skill_manager: Optional["SkillManager"] = None,
    ) -> dict[str, Any]:
        return cls._build_base_constructor_kwargs(
            snapshot,
            llm=llm,
            tool_registry=tool_registry,
            memory_manage=memory_manage,
            context_manager=context_manager,
            callback_manager=callback_manager,
            skill_manager=skill_manager,
        )

    @classmethod
    def _iter_agent_subclasses(cls) -> list[type["BaseAgent"]]:
        result: list[type["BaseAgent"]] = []
        for subclass in cls.__subclasses__():
            result.append(subclass)
            result.extend(subclass._iter_agent_subclasses())
        return result

    @classmethod
    def _resolve_agent_class(cls, agent_type: str) -> type["BaseAgent"]:
        if cls is not BaseAgent:
            return cls

        try:
            __import__("agent")
        except Exception:
            pass

        for candidate in [BaseAgent] + BaseAgent._iter_agent_subclasses():
            if candidate.__name__ == agent_type:
                return candidate

        raise SessionSerializationError(f"无法解析 Agent 类型: {agent_type}")

    @staticmethod
    def _resolve_session_store(store: Any = None):
        from db.session_store import SessionStore

        if store is None:
            return SessionStore()
        if isinstance(store, SessionStore):
            return store
        if isinstance(store, str):
            return SessionStore(db_path=store)
        raise SessionSerializationError(f"不支持的 store 类型: {type(store).__name__}")

    @classmethod
    def list_sessions(
        cls,
        *,
        store: Any = None,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        session_store = cls._resolve_session_store(store)
        return session_store.list_sessions(limit=limit, include_expired=include_expired)

    @classmethod
    def delete_session(cls, session_id: str, *, store: Any = None) -> bool:
        session_store = cls._resolve_session_store(store)
        return session_store.delete_session(session_id)

    @classmethod
    def cleanup_expired_sessions(
        cls,
        *,
        store: Any = None,
        now: Optional[datetime] = None,
    ) -> int:
        session_store = cls._resolve_session_store(store)
        return session_store.cleanup_expired_sessions(now=now)

    def save_session(
        self,
        session_id: str,
        *,
        store: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> str:
        if not session_id or not isinstance(session_id, str):
            raise SessionSerializationError("session_id 必须是非空字符串")

        from db.conversation_store import ConversationStore

        session_store = self._resolve_session_store(store)
        conversation_store = ConversationStore(db_path=session_store.db_path)

        snapshot = self._build_session_snapshot()
        session_metadata = self._make_json_safe(metadata or {})

        session_store.create_or_update_session(
            session_id=session_id,
            agent_type=self.__class__.__name__,
            agent_name=self.name,
            snapshot=snapshot,
            metadata=session_metadata,
            expires_at=expires_at,
        )
        conversation_store.replace_messages(session_id, self.history)
        logger.info("会话已保存: %s", session_id)
        return session_id

    @classmethod
    def load_session(
        cls,
        session_id: str,
        *,
        llm: EasyLLM,
        store: Any = None,
        tool_registry: Optional["ToolRegistry"] = None,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        callback_manager: Optional["CallbackManager"] = None,
        skill_manager: Optional["SkillManager"] = None,
    ) -> "BaseAgent":
        if not session_id or not isinstance(session_id, str):
            raise SessionSerializationError("session_id 必须是非空字符串")

        from db.conversation_store import ConversationStore

        session_store = cls._resolve_session_store(store)
        record = session_store.get_session(session_id)
        if record is None:
            raise SessionNotFoundError(f"会话不存在: {session_id}")

        snapshot = record["snapshot"]
        target_cls = cls._resolve_agent_class(snapshot["agent_type"])

        if cls is not BaseAgent and target_cls is not cls:
            raise SessionSerializationError(
                f"会话 {session_id} 属于 {target_cls.__name__}，无法按 {cls.__name__} 恢复"
            )
        if not target_cls._supports_session_restore():
            raise SessionSerializationError(
                f"{target_cls.__name__} 暂不支持自动恢复，请手动重建实例"
            )

        init_kwargs = target_cls._build_constructor_kwargs_from_snapshot(
            snapshot,
            llm=llm,
            tool_registry=tool_registry,
            memory_manage=memory_manage,
            context_manager=context_manager,
            callback_manager=callback_manager,
            skill_manager=skill_manager,
        )
        agent = target_cls(**init_kwargs)
        agent._restore_serializable_state(snapshot.get("state") or {})

        conversation_store = ConversationStore(db_path=session_store.db_path)
        agent.history = conversation_store.load_messages(session_id)

        missing_tools = []
        expected_tools = snapshot.get("tool_names") or []
        if expected_tools:
            if tool_registry is None:
                missing_tools = expected_tools
            else:
                missing_tools = [name for name in expected_tools if not tool_registry.has_tool(name)]
        if missing_tools:
            logger.warning("恢复会话时缺少工具实现: %s", missing_tools)

        expected_skills = snapshot.get("active_skills") or []
        if expected_skills:
            if skill_manager is None:
                logger.warning("恢复会话时未提供 skill_manager，以下 Skill 需手动恢复: %s", expected_skills)
            else:
                missing_skills = [name for name in expected_skills if not skill_manager.has_skill(name)]
                if missing_skills:
                    logger.warning("恢复会话时缺少 Skill 实现: %s", missing_skills)

        if snapshot.get("enable_tool") and tool_registry is None:
            logger.warning("会话原本启用了工具，但恢复时未注入 ToolRegistry，已降级为无工具模式")

        logger.info("会话已恢复: %s", session_id)
        return agent

    def get_history(self):
        """获取对话历史"""
        return self.history

    def get_context_usage(self) -> dict[str, Any]:
        """获取最近一次 invoke 构造出的可见上下文 token 使用快照。"""
        return self._make_json_safe(self._last_context_usage)

    def get_history_length(self) -> int:
        """
        获取对话历史长度
        
        Returns:
            对话历史条数
        """
        return len(self.history)

    def _capture_context_usage(
        self,
        messages: list[Any],
        *,
        label: str = "visible_messages",
    ) -> dict[str, Any]:
        """记录最近一次发送给 LLM 的可见上下文使用情况。"""
        if self.context_manager is not None:
            usage = self.context_manager.update_last_usage(messages, label=label)
            self._set_last_context_usage(usage)
            return usage

        normalized = self._normalize_context_messages(messages)
        max_tokens = self._resolve_context_budget_max_tokens()
        used_tokens = self._context_usage_counter.count_messages(normalized)
        remaining_tokens = max(0, max_tokens - used_tokens) if max_tokens is not None else None
        usage = {
            "label": label,
            "message_count": len(normalized),
            "used_tokens": used_tokens,
            "remaining_tokens": remaining_tokens,
            "max_tokens": max_tokens,
            "history_compacted": False,
            "tracked_at": datetime.now().isoformat(),
        }
        self._set_last_context_usage(usage)
        return usage

    def _set_last_context_usage(self, usage: Optional[dict[str, Any]]) -> None:
        normalized = self._make_json_safe(usage or {})
        self._last_context_usage = normalized
        if self.context_manager is not None:
            try:
                self.context_manager.set_last_usage(normalized)
            except Exception:
                logger.debug("同步 context usage 到 ContextManager 失败", exc_info=True)

    def _resolve_context_budget_max_tokens(self) -> Optional[int]:
        if self.context_manager is not None:
            return self.context_manager.budget.max_tokens
        if self.config.max_tokens is not None:
            return self.config.max_tokens
        return getattr(self.llm, "max_tokens", None)

    def _normalize_context_messages(self, messages: Optional[list[Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages or []:
            if hasattr(message, "to_dict"):
                payload = message.to_dict()
            elif isinstance(message, dict):
                payload = self._make_json_safe(message)
            elif hasattr(message, "role") and hasattr(message, "content"):
                payload = {
                    "role": str(getattr(message, "role", "user")),
                    "content": getattr(message, "content", ""),
                }
            else:
                payload = {"role": "user", "content": str(message)}

            if not isinstance(payload, dict):
                payload = {"role": "user", "content": str(payload)}
            normalized.append(payload)
        return normalized

    def __str__(self) -> str:
        return f"Agent(name={self.name}, description={self.description})"
    

    def _safe_get_tool_name(self, tool_call: Any) -> str:
        """
        安全获取工具名称

        支持两种 API 格式：
          - Chat API:      tool_call.function.name
          - Responses API: tool_call.name  (扶平化结构)

        Args:
            tool_call: 工具调用对象

        Returns:
            工具名称

        Raises:
            ToolExecutionError: 无法获取工具名称
        """
        try:
            if isinstance(tool_call, dict):
                if isinstance(tool_call.get("function"), dict) and tool_call["function"].get("name"):
                    return tool_call["function"]["name"]
                name = tool_call.get("name")
                if name and isinstance(name, str):
                    return name

            # Chat API: tool_call.function.name
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                name = tool_call.function.name
                if name and isinstance(name, str):
                    return name

            # Responses API: tool_call.name (flat structure)
            if hasattr(tool_call, 'name'):
                name = tool_call.name
                if name and isinstance(name, str):
                    return name

            raise ToolExecutionError("工具调用对象中没有有效的工具名称")
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"获取工具名称失败: {e}") from e

    def _safe_parse_tool_args(self, tool_call: Any) -> dict:
        """
        安全解析工具参数

        支持两种 API 格式：
          - Chat API:      tool_call.function.arguments  (JSON 字符串)
          - Responses API: tool_call.arguments           (字符串或字典)

        Args:
            tool_call: 工具调用对象

        Returns:
            解析后的参数字典

        Raises:
            ToolExecutionError: 参数解析失败
        """
        try:
            if isinstance(tool_call, dict):
                if isinstance(tool_call.get("function"), dict):
                    arguments = tool_call["function"].get("arguments")
                else:
                    arguments = tool_call.get("arguments")
            else:
                arguments = None

            # Chat API: tool_call.function.arguments
            if arguments is None and hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                arguments = tool_call.function.arguments
            # Responses API: tool_call.arguments (flat structure)
            elif arguments is None and hasattr(tool_call, 'arguments'):
                arguments = tool_call.arguments
            elif arguments is None:
                raise ToolExecutionError("工具调用对象中没有 arguments 属性")

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
        
        self.callback_manager.on_tool_start(tool_name, tool_args)
        
        try:
            result = self.tool_registry.execute_tool(tool_name, tool_args)
            
            # 确保返回字符串
            if result is None:
                result = "工具执行完成，无返回结果"
            
            if not isinstance(result, str):
                result = str(result)
            
            self.callback_manager.on_tool_end(tool_name, result, success=True)
            return result
            
        except Exception as e:
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=e)
            raise ToolExecutionError(f"工具 '{tool_name}' 执行失败: {e}") from e

    async def _async_safe_execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        异步安全执行工具
        
        工具本身是同步的 tool.run()，通过独立线程池执行以避免阻塞事件循环。
        这里避免使用默认线程池，实测在严格 asyncio 测试环境下可能导致关闭阶段挂起。
        """
        if self.tool_registry is None:
            raise ToolExecutionError("工具注册表未配置!")
        
        self.callback_manager.on_tool_start(tool_name, tool_args)
        
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.tool_registry.execute_tool,
                    tool_name,
                    tool_args,
                )
            
            if result is None:
                result = "工具执行完成，无返回结果"
            if not isinstance(result, str):
                result = str(result)
            
            self.callback_manager.on_tool_end(tool_name, result, success=True)
            return result
            
        except Exception as e:
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=e)
            raise ToolExecutionError(f"工具 '{tool_name}' 执行失败: {e}") from e

    def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        执行工具
        
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

    def add_tool(self, tool) -> None:
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

    # ==================== 向后兼容别名 ====================

    def executeTool(self, tool_name: str, tool_args: dict) -> str:
        """向后兼容：请改用 execute_tool"""
        return self.execute_tool(tool_name, tool_args)

    def addTool(self, tool) -> None:
        """向后兼容：请改用 add_tool"""
        return self.add_tool(tool)

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
