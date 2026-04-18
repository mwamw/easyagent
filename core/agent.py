"""
Agent 基类模块
"""
from core.Exception import ToolExecutionError
from .history import (
    CanonicalMessage,
    ReplayHistoryState,
    _json_safe,
    canonical_text_content,
    coerce_canonical_message,
)
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
from Tool.BaseTool import Tool, ToolResult
from .Exception import *
import logging

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage

logger = logging.getLogger(__name__)


def _build_history_context_usage_payload(
    *,
    max_tokens: Optional[int],
    history_budget_tokens: Optional[int],
    history_tokens: int,
    system_prompt_tokens: int,
    tool_schema_tokens: int,
    stable_context_tokens: int,
    canonical_history_messages: int,
    replay_history_messages: int,
    compaction: Optional[dict[str, Any]] = None,
    pending_step_active: bool = False,
) -> dict[str, Any]:
    remaining = (max_tokens - stable_context_tokens) if max_tokens is not None else None
    history_remaining = (history_budget_tokens - stable_context_tokens) if history_budget_tokens is not None else None
    compaction = dict(compaction or {})
    return {
        "max_tokens": max_tokens,
        "history_budget_tokens": history_budget_tokens,
        "history_tokens": history_tokens,
        "system_prompt_tokens": system_prompt_tokens,
        "tool_schema_tokens": tool_schema_tokens,
        "stable_context_tokens": stable_context_tokens,
        "remaining_tokens_for_sources_and_query": remaining,
        "history_budget_remaining_tokens": history_remaining,
        "history_compacted": bool(compaction.get("was_compacted", False)),
        "last_history_compaction": compaction or {},
        "canonical_history_messages": canonical_history_messages,
        "replay_history_messages": replay_history_messages,
        "pending_step_active": pending_step_active,
        "tracked_at": datetime.now().isoformat(),
    }


def _build_history_compaction_state(
    *,
    was_compacted: bool,
    compaction_possible: bool,
    tokens_before: int,
    tokens_after: int,
    max_tokens: Optional[int],
) -> dict[str, Any]:
    return {
        "was_compacted": was_compacted,
        "compaction_possible": compaction_possible,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "budget": max_tokens,
        "tracked_at": datetime.now().isoformat(),
    }


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
        reasoning: Optional[dict[str, Any]] = None,
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
        self.reasoning = reasoning
        self.llm = llm
        self.system_prompt = system_prompt
        self.description = description
        self.config = config or Config.from_env()
        self._history: list[Any] = []
        self.replay_history: list[Any] = []
        self.replay_history_provider_name: Optional[str] = getattr(llm, "provider_name", None)
        
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
        self._last_history_compaction: dict[str, Any] = {}
        self._pending_step_state: Optional[dict[str, Any]] = None
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
        try:
            self.skill_manager.register(skill)
        except Exception as e:
            logger.error(f"注册 Skill 失败: {e}")
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

    def _append_dual_history(
        self,
        canonical_entries: list[Any],
        replay_entries: list[Any],
    ) -> None:
        self._assert_replay_history_ready_for_current_provider()
        provider_name = getattr(self.llm, "provider_name", None)
        for entry in canonical_entries:
            self._history.append(entry)
        while len(self._history) > self.config.max_history_length:
            self._history.pop(0)

        for entry in replay_entries:
            self.llm.append_replay_entry(self.replay_history, entry, provider_name)
        while len(self.replay_history) > self.config.max_history_length:
            self.replay_history.pop(0)
        self.replay_history_provider_name = provider_name
        self._check_and_trigger_background_memory()

    def _append_query_history(self, query: str) -> None:
        self._append_dual_history(
            self.llm.query_to_canonical(query),
            self.llm.query_to_replay(query),
        )

    def _append_response_history(
        self,
        response: Any,
        *,
        include_reasoning: bool = True,
    ) -> None:
        self._append_dual_history(
            self.llm.response_to_canonical(response, include_reasoning=include_reasoning),
            self.llm.response_to_replay(response, include_reasoning=include_reasoning),
        )

    def _append_assistant_message_history(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> None:
        self._append_dual_history(
            self.llm.assistant_message_to_canonical(
                content=content,
                tool_calls=tool_calls,
                thinking=thinking,
            ),
            self.llm.assistant_message_to_replay(
                content=content,
                tool_calls=tool_calls,
                thinking=thinking,
            ),
        )

    def _append_tool_result_history(self, content: str, tool_id: str, tool_name: str) -> None:
        self._append_dual_history(
            self.llm.tool_result_to_canonical(content, tool_id, tool_name),
            self.llm.tool_result_to_replay(content, tool_id, tool_name),
        )

    def _set_pending_step_state(
        self,
        *,
        assistant_canonical: list[Any],
        assistant_replay: list[Any],
        tool_calls: Optional[list[dict[str, Any]]] = None,
        round_number: Optional[int] = None,
    ) -> None:
        self._pending_step_state = {
            "assistant_canonical": list(assistant_canonical or []),
            "assistant_replay": list(assistant_replay or []),
            "tool_results_canonical": [],
            "tool_results_replay": [],
            "tool_calls": _json_safe(tool_calls or []),
            "round_number": round_number,
            "provider_name": getattr(self.llm, "provider_name", None),
        }

    def _append_pending_tool_result(
        self,
        *,
        tool_canonical: list[Any],
        tool_replay: list[Any],
    ) -> None:
        if self._pending_step_state is None:
            return
        self._pending_step_state["tool_results_canonical"].extend(list(tool_canonical or []))
        self._pending_step_state["tool_results_replay"].extend(list(tool_replay or []))

    def _commit_pending_step_state(self) -> bool:
        if not self._pending_step_state:
            return False
        canonical_entries = [
            *list(self._pending_step_state.get("assistant_canonical") or []),
            *list(self._pending_step_state.get("tool_results_canonical") or []),
        ]
        replay_entries = [
            *list(self._pending_step_state.get("assistant_replay") or []),
            *list(self._pending_step_state.get("tool_results_replay") or []),
        ]
        self._append_dual_history(canonical_entries, replay_entries)
        self._pending_step_state = None
        return True

    def _clear_pending_step_state(self) -> None:
        self._pending_step_state = None

    def get_pending_step_state(self) -> Optional[dict[str, Any]]:
        if self._pending_step_state is None:
            return None
        return self._make_json_safe(self._pending_step_state)
    
    def _append_history_entries(self, messages: list[Any]) -> None:
        """向 canonical history 与 replay history 批量追加消息。"""
        self._assert_replay_history_ready_for_current_provider()
        canonical_entries: list[Any] = []
        for message in list(messages or []):
            canonical_entries.extend(self.llm.history_entry_to_canonical(message))
        replay_entries = self._build_replay_entries(canonical_entries)
        self._append_dual_history(canonical_entries, replay_entries)

    def _build_replay_entries(self, message: Any) -> list[Any]:
        messages = message if isinstance(message, list) else [message]
        return self.llm.canonical_to_replay_history(
            list(messages),
            getattr(self.llm, "provider_name", None),
        )

    @staticmethod
    def _serialize_replay_entry(message: Any) -> Any:
        if hasattr(message, "to_dict"):
            payload = message.to_dict()
            if isinstance(payload, dict):
                return payload
        if isinstance(message, dict):
            return _json_safe(message)
        return message

    @staticmethod
    def _deserialize_replay_entry(payload: Any) -> Any:
        return _json_safe(payload)

    def add_message(self, message: Any) -> None:
        """添加消息到历史"""
        self._append_history_entries([message])

    def add_messages(self, messages: list[Any]) -> None:
        """批量添加消息到历史。"""
        self._append_history_entries(messages)

    @staticmethod
    def _history_entry_to_role_content(message: Any) -> tuple[str, str]:
        """提取 history 条目的 role/content，用于摘要与调试。"""
        canonical = coerce_canonical_message(message)
        if canonical is not None:
            return str(canonical.role), canonical.text_content()
        if isinstance(message, dict):
            if message.get("record_type", message.get("schema")) == "canonical_message":
                canonical = CanonicalMessage.model_validate(message)
                return str(canonical.role), canonical.text_content()
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
        self._append_query_history(content)
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self._append_assistant_message_history(content=content)
    
    def add_context_source(self, source:BaseContextSource) -> None:
        """添加上下文来源"""
        if self.context_manager is None:
            raise ParameterValidationError("上下文管理器未配置，无法添加上下文来源!")
        self.context_manager.add_source(source)
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self._history.clear()
        self.replay_history.clear()
        self.replay_history_provider_name = getattr(self.llm, "provider_name", None)
        self._clear_pending_step_state()
        self._last_history_compaction = {}
        self._unextracted_msg_count = 0
        logger.info("对话历史已清空")

    def _request_budget_max_tokens(self) -> Optional[int]:
        if self.context_manager is not None:
            return self.context_manager.budget.max_tokens
        if self.config.max_tokens is not None:
            return self.config.max_tokens
        return getattr(self.llm, "max_tokens", None)

    def _history_budget_max_tokens(self) -> Optional[int]:
        if self.context_manager is not None:
            budget = self.context_manager.budget.get_budget("history")
            if budget > 0:
                return budget
        return self._request_budget_max_tokens()

    def _stable_system_prompt(self) -> Optional[str]:
        return self.get_enhanced_prompt()

    def _stable_tools(self) -> Optional[list[dict[str, Any]]]:
        if self.tool_registry is None:
            return None
        return self.tool_registry.get_openai_tools()

    def _apply_history_compaction_result(self, result: Any) -> bool:
        self._last_history_compaction = _build_history_compaction_state(
            was_compacted=result.was_compacted,
            compaction_possible=result.compaction_possible,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            max_tokens=result.budget,
        )
        if not result.was_compacted:
            return False
        self._history = list(result.canonical_history)
        self.replay_history = list(result.replay_history)
        self.replay_history_provider_name = getattr(self.llm, "provider_name", None)
        return True

    def compact_persistent_history_if_needed(self) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        result = self.context_manager.compact_persistent_history(
            self._history,
            self.replay_history,
            provider_name=getattr(self.llm, "provider_name", None),
            token_counter=self._context_usage_counter,
            system_prompt=self._stable_system_prompt(),
            tools=self._stable_tools(),
            reasoning=self.reasoning,
            max_tokens=budget,
        )
        return self._apply_history_compaction_result(result)

    async def acompact_persistent_history_if_needed(self) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        result = await self.context_manager.acompact_persistent_history(
            self._history,
            self.replay_history,
            provider_name=getattr(self.llm, "provider_name", None),
            token_counter=self._context_usage_counter,
            system_prompt=self._stable_system_prompt(),
            tools=self._stable_tools(),
            reasoning=self.reasoning,
            max_tokens=budget,
        )
        return self._apply_history_compaction_result(result)

    def compact_history(self, max_tokens: Optional[int] = None) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = max_tokens if max_tokens is not None else self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        result = self.context_manager.compact_persistent_history(
            self._history,
            self.replay_history,
            provider_name=getattr(self.llm, "provider_name", None),
            token_counter=self._context_usage_counter,
            system_prompt=self._stable_system_prompt(),
            tools=self._stable_tools(),
            reasoning=self.reasoning,
            max_tokens=budget,
            force=True,
        )
        return self._apply_history_compaction_result(result)

    async def acompact_history(self, max_tokens: Optional[int] = None) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = max_tokens if max_tokens is not None else self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        result = await self.context_manager.acompact_persistent_history(
            self._history,
            self.replay_history,
            provider_name=getattr(self.llm, "provider_name", None),
            token_counter=self._context_usage_counter,
            system_prompt=self._stable_system_prompt(),
            tools=self._stable_tools(),
            reasoning=self.reasoning,
            max_tokens=budget,
            force=True,
        )
        return self._apply_history_compaction_result(result)

    def _get_serializable_state(self) -> dict[str, Any]:
        """返回子类需要补充持久化的状态。"""
        return {
            "last_history_compaction": self._make_json_safe(self._last_history_compaction),
            "pending_step_state": self._make_json_safe(self._pending_step_state),
            "replay_history_state": ReplayHistoryState(
                provider_name=self.replay_history_provider_name,
                messages=[
                    self._serialize_replay_entry(message)
                    for message in self.replay_history
                ],
            ).to_dict(),
        }

    def _restore_serializable_state(self, state: Optional[dict[str, Any]]) -> None:
        """恢复子类持久化状态。"""
        state = state or {}
        self._last_history_compaction = self._make_json_safe(state.get("last_history_compaction") or {})
        pending_state = state.get("pending_step_state")
        self._pending_step_state = self._make_json_safe(pending_state) if pending_state is not None else None
        replay_state = state.get("replay_history_state") or {}
        provider_name = replay_state.get("provider_name")
        messages = [
            self._deserialize_replay_entry(message)
            for message in list(replay_state.get("messages") or [])
        ]
        if provider_name and provider_name == getattr(self.llm, "provider_name", None):
            self.replay_history = messages
            self.replay_history_provider_name = provider_name
        else:
            self.replay_history = []
            self.replay_history_provider_name = getattr(self.llm, "provider_name", None)
        return None

    @classmethod
    def _supports_session_restore(cls) -> bool:
        """当前 Agent 类型是否支持从会话快照恢复。"""
        return True

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        return _json_safe(value)

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
        restored_history = conversation_store.load_messages(session_id)
        agent._set_history_entries(restored_history, rebuild_replay=not bool(agent.replay_history))
        if agent.replay_history_provider_name != getattr(agent.llm, "provider_name", None):
            agent.rebuild_replay_history()

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
        """获取当前 provider 的 replay/raw history（向后兼容）。"""
        return self.replay_history

    def get_raw_history(self):
        """获取当前 provider 的 replay/raw history。"""
        return self.replay_history

    def get_canonical_history(self):
        """获取 canonical history。"""
        return self._history

    @property
    def raw_history(self) -> list[Any]:
        return self.replay_history

    def get_context_usage(self) -> dict[str, Any]:
        """获取当前稳定上下文的 token 使用情况。"""
        max_tokens = self._request_budget_max_tokens()
        history_budget = self._history_budget_max_tokens()
        system_prompt = self._stable_system_prompt()
        tools = self._stable_tools()
        history_tokens = self.llm.count_request_tokens(
            self._context_usage_counter,
            self.replay_history,
        )
        stable_context_tokens = self.llm.count_request_tokens(
            self._context_usage_counter,
            self.replay_history,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=self.reasoning,
        )
        usage = _build_history_context_usage_payload(
            max_tokens=max_tokens,
            history_budget_tokens=history_budget,
            history_tokens=history_tokens,
            system_prompt_tokens=self._context_usage_counter.count(system_prompt or ""),
            tool_schema_tokens=self._context_usage_counter.count(tools or []),
            stable_context_tokens=stable_context_tokens,
            canonical_history_messages=len(self._history),
            replay_history_messages=len(self.replay_history),
            compaction=self._last_history_compaction,
            pending_step_active=self._pending_step_state is not None,
        )
        return self._make_json_safe(usage)

    def get_history_length(self) -> int:
        """
        获取对话历史长度
        
        Returns:
            对话历史条数
        """
        return len(self.replay_history)

    def rebuild_replay_history(self) -> list[Any]:
        self.replay_history = self.llm.canonical_to_replay_history(
            self._history,
            getattr(self.llm, "provider_name", None),
        )
        self.replay_history_provider_name = getattr(self.llm, "provider_name", None)
        return self.replay_history

    def prepare_replay_history(self, messages: list[Any], provider_name: Optional[str] = None) -> list[Any]:
        target_provider = provider_name or getattr(self.llm, "provider_name", None)
        return self.llm.canonical_to_replay_history(messages, target_provider)

    def _set_history_entries(self, messages: list[Any], *, rebuild_replay: bool = True) -> None:
        canonical_entries: list[Any] = []
        for message in list(messages or []):
            canonical_entries.extend(self.llm.history_entry_to_canonical(message))
        self._history = canonical_entries[-self.config.max_history_length :]
        if rebuild_replay:
            self.rebuild_replay_history()

    def _assert_replay_history_ready_for_current_provider(self) -> None:
        current_provider = getattr(self.llm, "provider_name", None)
        if self._history and self.replay_history_provider_name != current_provider:
            raise SessionError(
                "当前 LLM provider 已变更，但 replay_history 仍属于旧 provider。请调用 change_model() 完成模型切换。"
            )

    @property
    def history(self) -> list[Any]:
        return self._history

    @history.setter
    def history(self, messages: list[Any]) -> None:
        self._set_history_entries(messages, rebuild_replay=True)

    def change_model(
        self,
        *,
        llm: Optional[EasyLLM] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> EasyLLM:
        current_llm = self.llm
        if llm is None:
            llm_kwargs = dict(getattr(current_llm, "kwargs", {}) or {})
            llm_kwargs.update(kwargs)
            llm = EasyLLM(
                model=model or getattr(current_llm, "model", None),
                provider=provider or getattr(current_llm, "provider_name", None) or "auto",
                api_key=api_key or getattr(current_llm, "api_key", None),
                base_url=base_url if base_url is not None else getattr(current_llm, "base_url", None),
                temperature=temperature if temperature is not None else getattr(current_llm, "temperature", None),
                max_tokens=max_tokens if max_tokens is not None else getattr(current_llm, "max_tokens", None),
                timeout=timeout if timeout is not None else getattr(current_llm, "timeout", None),
                **llm_kwargs,
            )

        self.llm = llm
        self.rebuild_replay_history()
        self._clear_pending_step_state()

        if current_llm is not llm:
            close = getattr(current_llm, "close", None)
            if callable(close):
                close()
        return llm

    def _resolve_context_budget_max_tokens(self) -> Optional[int]:
        if self.context_manager is not None:
            return self.context_manager.budget.max_tokens
        if self.config.max_tokens is not None:
            return self.config.max_tokens
        return getattr(self.llm, "max_tokens", None)


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

    def _safe_execute_tool_result(self, tool_name: str, tool_args: dict) -> ToolResult:
        """
        安全执行工具并返回结构化结果。
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            工具执行结果协议对象
            
        Raises:
            ToolExecutionError: 工具执行失败
        """
        if self.tool_registry is None:
            raise ToolExecutionError("工具注册表未配置!")
        
        self.callback_manager.on_tool_start(tool_name, tool_args)
        
        try:
            result = self.tool_registry.execute_tool_result(tool_name, tool_args)
            display_result = result.to_display_string()
            success = result.status == "success"
            self.callback_manager.on_tool_end(
                tool_name,
                display_result,
                success=success,
            )
            return result
            
        except Exception as e:
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=e)
            raise ToolExecutionError(f"工具 '{tool_name}' 执行失败: {e}") from e

    def _safe_execute_tool(self, tool_name: str, tool_args: dict) -> str:
        result = self._safe_execute_tool_result(tool_name, tool_args)
        return result.to_display_string()

    async def _async_safe_execute_tool_result(self, tool_name: str, tool_args: dict) -> ToolResult:
        """
        异步安全执行工具并返回结构化结果。
        
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
                    self.tool_registry.execute_tool_result,
                    tool_name,
                    tool_args,
                )
            display_result = result.to_display_string()
            success = result.status == "success"
            self.callback_manager.on_tool_end(
                tool_name,
                display_result,
                success=success,
            )
            return result
            
        except Exception as e:
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=e)
            raise ToolExecutionError(f"工具 '{tool_name}' 执行失败: {e}") from e

    async def _async_safe_execute_tool(self, tool_name: str, tool_args: dict) -> str:
        result = await self._async_safe_execute_tool_result(tool_name, tool_args)
        return result.to_display_string()

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

    def execute_tool_result(self, tool_name: str, tool_args: dict) -> ToolResult:
        """
        执行工具并返回结构化 ToolResult。
        """
        if self.tool_registry is None:
            raise ToolRegistryError("工具调用需要提供 ToolRegistry!")

        if not tool_name or not isinstance(tool_name, str):
            raise ParameterValidationError("工具名称必须是非空字符串!")

        if not isinstance(tool_args, dict):
            raise ParameterValidationError(f"工具参数必须是字典类型，收到: {type(tool_args).__name__}")

        return self._safe_execute_tool_result(tool_name, tool_args)

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
