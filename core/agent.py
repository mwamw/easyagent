"""
Agent 基类模块
"""
from core.Exception import ToolExecutionError
from .history import (
    CanonicalBlock,
    CanonicalMessage,
    ReplayHistoryState,
    _json_safe,
    canonical_text_content,
    coerce_canonical_message,
)
from typing import Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
import os
from prompt import PromptBlock
from .Config import Config
from .execution_mode import ExecutionMode, ModeController
from .llm import EasyLLM
from Tool.ToolRegistry import ToolRegistry
from context.manager import ContextManager
from context.source.base import BaseContextSource
from context.token.counter import TokenCounter
from .callbacks import CallbackManager
from .guardrails import build_default_hook_manager
from .hooks import HookManager
from .permissions import PermissionContext, PermissionEngine, PermissionMode, PermissionRule
from .request_input import ReplayRequestInput
from .runtime_reminders import (
    BaseRuntimeReminderSource,
    StaticRuntimeReminderSource,
    collect_runtime_reminder_prompt_blocks,
)
from .session import SessionRestoreReport
from observability import InMemoryObservabilityRecorder
from skill.manager import SkillManager
from prompt import build_memory_prompt_section
from core.providers import create_codec
import json
import asyncio
import threading
import concurrent.futures
from Tool.BaseTool import Tool, ToolResult, ToolSpec
from .Exception import *
import logging

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage

logger = logging.getLogger(__name__)


def _build_context_usage_report_v2(
    *,
    max_tokens: Optional[int],
    history_budget_tokens: Optional[int],
    history_tokens: int,
    system_tokens: int,
    tool_tokens: int,
    reasoning_tokens: int,
    estimated_request_tokens: int,
    request_estimate_source: str,
    request_estimate_metadata: Optional[dict[str, Any]] = None,
    request_layers: Optional[dict[str, Any]] = None,
    canonical_history_messages: int,
    replay_history_messages: int,
    compaction: Optional[dict[str, Any]] = None,
    compaction_estimated_request_tokens: Optional[int] = None,
    compaction_token_source: Optional[str] = None,
    compaction_metadata: Optional[dict[str, Any]] = None,
    cache: Optional[dict[str, Any]] = None,
    pending_step_active: bool = False,
) -> dict[str, Any]:
    remaining = (max_tokens - estimated_request_tokens) if max_tokens is not None else None
    history_remaining = (history_budget_tokens - estimated_request_tokens) if history_budget_tokens is not None else None
    compaction = dict(compaction or {})
    return {
        "version": 2,
        "budget": {
            "maxTokens": max_tokens,
            "historyBudgetTokens": history_budget_tokens,
            "remainingTokens": remaining,
            "historyRemainingTokens": history_remaining,
        },
        "requestEstimate": {
            "estimatedRequestTokens": estimated_request_tokens,
            "source": request_estimate_source,
            "metadata": dict(request_estimate_metadata or {}),
        },
        "requestLayers": dict(request_layers or {}),
        "tokenBreakdown": {
            "historyTokens": history_tokens,
            "systemTokens": system_tokens,
            "toolTokens": tool_tokens,
            "reasoningTokens": reasoning_tokens,
        },
        "history": {
            "canonicalMessages": canonical_history_messages,
            "replayMessages": replay_history_messages,
            "pendingStepActive": pending_step_active,
        },
        "compaction": {
            "last": compaction or {},
            "estimatedRequestTokens": compaction_estimated_request_tokens,
            "tokenSource": compaction_token_source,
            "metadata": dict(compaction_metadata or {}),
        },
        "cache": dict(cache or {}),
        "trackedAt": datetime.now().isoformat(),
    }


def _build_history_compaction_state(
    *,
    was_compacted: bool,
    compaction_possible: bool,
    tokens_before: int,
    tokens_after: int,
    max_tokens: Optional[int],
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "was_compacted": was_compacted,
        "compaction_possible": compaction_possible,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "budget": max_tokens,
        "metadata": dict(metadata or {}),
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
        permission_engine: Optional["PermissionEngine"] = None,
        permission_context: Optional["PermissionContext"] = None,
        hook_manager: Optional["HookManager"] = None,
        task_service: Optional[Any] = None,
        agent_runtime: Optional[Any] = None,
        team_manager: Optional[Any] = None,
        execution_context: Optional[Any] = None,
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
        self.hook_manager = hook_manager or build_default_hook_manager()
        
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
        self._history_usage_anchor: Optional[dict[str, Any]] = None
        self._pending_response_usage_anchor: Optional[dict[str, Any]] = None
        self._pending_step_state: Optional[dict[str, Any]] = None
        self._last_cache_signature: Optional[dict[str, Any]] = None
        self._last_cache_usage: Optional[dict[str, Any]] = None
        self._last_cache_break: Optional[dict[str, Any]] = None
        self._context_usage_counter = TokenCounter()
        
        # Skill 管理器
        self.skill_manager = skill_manager or SkillManager()
        self.skill_manager.bind_agent(self)
        self.mode_controller = ModeController()
        self.permission_engine = permission_engine or PermissionEngine()
        self.permission_context = permission_context or PermissionContext()
        self.current_task_id: Optional[str] = None
        self.task_service = task_service
        self.agent_runtime = agent_runtime
        self.team_manager = team_manager or getattr(agent_runtime, "team_manager", None)
        self._runtime_reminder_sources: list[BaseRuntimeReminderSource] = []
        self._stop_requested = False
        self._stop_reason: Optional[str] = None
        self._task_tools_registered = False
        if self.permission_context.mode == PermissionMode.PLAN:
            self.mode_controller.enter_plan_mode()
        else:
            self.mode_controller.exit_plan_mode()
        self.execution_context = execution_context
        self.last_restore_report: Optional[SessionRestoreReport] = None
        self.last_close_report: Optional[dict[str, Any]] = None
        self.observability_recorder = InMemoryObservabilityRecorder(agent_name=self.name)

        if self.memory_manage:
            self._install_memory_runtime(self.memory_manage)
        if self.task_service and self.tool_registry:
            self._register_task_tools()
        if self.enable_tool and self.tool_registry:
            self._ensure_deferred_tool_schema_tool()
        self._refresh_execution_context()

    def add_runtime_reminder_source(
        self,
        source: BaseRuntimeReminderSource,
    ) -> BaseRuntimeReminderSource:
        if not isinstance(source, BaseRuntimeReminderSource):
            raise ParameterValidationError(
                f"runtime_reminder_source 必须是 BaseRuntimeReminderSource 类型，收到: {type(source).__name__}"
            )
        self._runtime_reminder_sources.append(source)
        return source

    def extend_runtime_reminder_sources(
        self,
        sources: list[BaseRuntimeReminderSource],
    ) -> None:
        for source in sources:
            self.add_runtime_reminder_source(source)

    def remove_runtime_reminder_source(
        self,
        source_or_name: BaseRuntimeReminderSource | str,
    ) -> bool:
        if isinstance(source_or_name, BaseRuntimeReminderSource):
            try:
                self._runtime_reminder_sources.remove(source_or_name)
                return True
            except ValueError:
                return False
        target_name = str(source_or_name).strip()
        for index, source in enumerate(list(self._runtime_reminder_sources)):
            if getattr(source, "name", source.__class__.__name__) == target_name:
                del self._runtime_reminder_sources[index]
                return True
        return False

    def list_runtime_reminder_sources(self) -> list[BaseRuntimeReminderSource]:
        return list(self._runtime_reminder_sources)

    def with_runtime_reminder(
        self,
        *,
        name: str,
        content: str,
        stable: bool = True,
        cacheable: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "BaseAgent":
        self.add_runtime_reminder_source(
            StaticRuntimeReminderSource(
                name=name,
                content=content,
                stable=stable,
                cacheable=cacheable,
                metadata=metadata,
            )
        )
        return self

    def build_runtime_reminder_prompt_blocks(
        self,
        *,
        start_order: int,
    ) -> list[PromptBlock]:
        return collect_runtime_reminder_prompt_blocks(
            self,
            self._runtime_reminder_sources,
            start_order=start_order,
        )

    def _register_v2_memory_tools(self) -> None:
        if self.memory_manage and self.tool_registry:
            try:
                from Tool.builtin.memorytool import register_memory_tools
                register_memory_tools(self.memory_manage, self.tool_registry)
                logger.info("已自动注册 V2 记忆系统工具")
            except ImportError as e:
                logger.warning(f"未能导入 register_memory_tools: {e}")

    def _ensure_memory_context_source(self, memory_manage: "MemoryManage") -> None:
        if self.context_manager is None:
            self.context_manager = ContextManager()
        try:
            from context.source.memory_source import MemoryContextSource

            self.context_manager.add_source(MemoryContextSource(memory_manage=memory_manage))
        except ImportError as exc:
            logger.warning("未能导入 MemoryContextSource: %s", exc)

    def _install_memory_runtime(self, memory_manage: "MemoryManage") -> None:
        """Bind memory as tools plus request context, without dynamic system-prompt injection."""
        self.memory_manage = memory_manage
        self._ensure_memory_context_source(memory_manage)
        if self.skill_manager.has_skill("memory"):
            try:
                if not self.skill_manager.is_active("memory"):
                    self.skill_manager.activate("memory")
            except Exception as exc:
                logger.warning("激活已注册的 MemorySkill 失败: %s", exc)
            return
        try:
            from skill.builtin.memory_skill import MemorySkill

            self.skill_manager.register(
                MemorySkill(memory_manage=memory_manage, include_context_source=True)
            )
            logger.info("已通过 MemorySkill 注册 V2 记忆系统")
        except ImportError:
            logger.warning("MemorySkill 导入失败，使用旧方式注册记忆工具")
            if self.tool_registry is not None:
                self._register_v2_memory_tools()
    
    def enable_multi_agent_system(
        self, 
        workspace_root: str,
        storage_dir: str,
        max_background_tasks: int = 4
        ) -> None:
        """一键启用 Agent 的所有多节点协作调度能力及配套工具。"""
        from Tool.builtin import (
            register_agent_tool, 
            register_agent_runtime_tools, 
            register_send_message_tool, 
            register_mailbox_tools, 
            register_team_create_tool, 
            register_team_delete_tool,
            register_worktree_tools,
            register_task_tools,
            register_task_output_tool
        )
        from runtime import TeamManager
        if not self.tool_registry:
            raise ToolRegistryError("启用多智能体系统需要提供 ToolRegistry!")
        registry = self.tool_registry
        # 1. 注册核心任务与沙盒工具 (基础依赖)
        if hasattr(self, "task_service") and self.task_service:
            register_task_tools(registry, service=self.task_service)
            register_task_output_tool(registry)
        
        if getattr(self.config, "enable_worktree", False):
            from Tool.runtime import WorktreeManager
            try:
                repo_root = WorktreeManager.detect_repo_root(workspace_root)
                manager = WorktreeManager(repo_root=repo_root, original_cwd=workspace_root)
                register_worktree_tools(registry, worktree_manager=manager)
            except Exception:
                pass

        # 2. 注册主 Agent 工具并获取 runtime
        agent_tool = register_agent_tool(
            registry,
            parent_agent=self,
            workspace_root=workspace_root,
            allowed_roots=(workspace_root,),
            storage_dir=storage_dir,
            max_background_tasks=max_background_tasks,
        )

        # 3. 初始化 Team 机制
        team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
        agent_tool.agent_runtime.bind_team_manager(team_manager)

        # 4. 批量挂载其他协作类工具
        register_agent_runtime_tools(registry, agent_runtime=agent_tool.agent_runtime, parent_agent=self)
        register_send_message_tool(registry, agent_runtime=agent_tool.agent_runtime, parent_agent=self)
        register_mailbox_tools(registry, agent_runtime=agent_tool.agent_runtime, parent_agent=self)
        register_team_create_tool(registry, team_manager=team_manager, parent_agent=self)
        register_team_delete_tool(registry, team_manager=team_manager)

        # 5. 最后让主层级 Agent 绑定这个 runtime
        self.bind_runtime(
            agent_runtime=agent_tool.agent_runtime,
            team_manager=team_manager,
        )


    def with_memory(self, memory_manage: "MemoryManage") -> "BaseAgent":
        """
        方便地将 V2 版本的 MemoryManage 记忆系统绑定到 Agent。

        内部会自动创建 MemorySkill 并注册到 SkillManager，同时挂载 MemoryContextSource。
        如果已经通过 with_skill 手动注册了 MemorySkill，则跳过自动注册。
        """
        self._install_memory_runtime(memory_manage)
        return self

    def _register_task_tools(self) -> None:
        if self.tool_registry is None or self.task_service is None:
            return
        if self._task_tools_registered:
            self._rebind_todo_write_tool()
            return
        required_tools = ("TaskCreate", "TaskGet", "TaskUpdate", "TaskList")
        if all(self.tool_registry.has_tool(name) for name in required_tools):
            self._task_tools_registered = True
            self._rebind_todo_write_tool()
            return
        from Tool.builtin.task_tools import register_task_tools

        register_task_tools(self.tool_registry, service=self.task_service)
        self._task_tools_registered = True
        self._rebind_todo_write_tool()
        logger.info("已自动注册结构化任务工具")

    def _rebind_todo_write_tool(self) -> None:
        if self.tool_registry is None or self.task_service is None:
            return
        if not self.tool_registry.has_tool("TodoWrite"):
            return
        from Tool.builtin import register_todo_write_tool

        register_todo_write_tool(
            self.tool_registry,
            service=self.task_service,
            scope_key=f"agent:{self.name}",
            owner=self.name,
        )

    def _ensure_deferred_tool_schema_tool(self) -> None:
        if self.tool_registry is None:
            return
        if getattr(self.config, "tool_schema_mode", "full") != "deferred":
            return
        if self.tool_registry.has_tool("tool_schema_tool"):
            return
        from Tool.builtin import register_tool_schema_tool

        register_tool_schema_tool(self.tool_registry)

    def with_task_service(self, task_service: Any) -> "BaseAgent":
        self.task_service = task_service
        self._task_tools_registered = False
        if self.tool_registry is not None:
            self._register_task_tools()
        self._refresh_execution_context()
        return self

    def bind_runtime(
        self,
        *,
        agent_runtime: Optional[Any] = None,
        team_manager: Optional[Any] = None,
        execution_context: Optional[Any] = None,
    ) -> "BaseAgent":
        if agent_runtime is not None:
            self.agent_runtime = agent_runtime
        if team_manager is not None:
            self.team_manager = team_manager
        elif self.agent_runtime and getattr(self.agent_runtime, "team_manager", None) is not None:
            self.team_manager = self.agent_runtime.team_manager
        if execution_context is not None:
            self.execution_context = execution_context
        else:
            self._refresh_execution_context()
        return self

    def _refresh_execution_context(self) -> None:
        try:
            from runtime import ExecutionContext
        except Exception:
            return

        current = getattr(self, "execution_context", None)
        metadata = dict(getattr(current, "metadata", {}) or {})
        self.execution_context = ExecutionContext.from_agent(
            self,
            metadata=metadata,
            worktree_path=getattr(current, "worktree_path", None),
            worktree_branch=getattr(current, "worktree_branch", None),
        )

    def _get_runtime_agent_id(self) -> Optional[str]:
        execution_context = getattr(self, "execution_context", None)
        metadata = dict(getattr(execution_context, "metadata", {}) or {})
        raw_value = metadata.get("agentId") or metadata.get("agent_id")
        if raw_value is None:
            return None
        value = str(raw_value).strip()
        return value or None

    @staticmethod
    def _find_worktree_manager(tool_registry: Optional["ToolRegistry"]) -> Optional[Any]:
        if tool_registry is None:
            return None
        candidates: list[Any] = []
        seen: set[int] = set()
        for tool_name in ("EnterWorktree", "ExitWorktree", "Agent"):
            tool = tool_registry.get_tool(tool_name)
            manager = getattr(tool, "worktree_manager", None) if tool is not None else None
            if manager is None:
                continue
            marker = id(manager)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(manager)
        for manager in candidates:
            active_session_getter = getattr(manager, "get_active_session", None)
            if not callable(active_session_getter):
                continue
            try:
                if active_session_getter() is not None:
                    return manager
            except Exception:
                continue
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _find_codeintel_managers(tool_registry: Optional["ToolRegistry"]) -> list[Any]:
        if tool_registry is None:
            return []
        managers: list[Any] = []
        seen: set[int] = set()
        list_surfaces = getattr(tool_registry, "list_runtime_surfaces", None)
        if callable(list_surfaces):
            try:
                surfaces = list_surfaces("codeintel_manager")
                for manager in surfaces.values():
                    marker = id(manager)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    managers.append(manager)
            except Exception:
                pass
        if managers:
            return managers
        for tool_name in (
            "CodeIntelStatus",
            "CodeIntelCacheStatus",
            "CodeIntelPrewarmWorkspace",
            "FindDefinition",
            "FindReferences",
            "GetDocumentSymbols",
            "GetWorkspaceSymbols",
            "GetDiagnostics",
        ):
            tool = tool_registry.get_tool(tool_name)
            manager = getattr(tool, "codeintel_manager", None) if tool is not None else None
            if manager is None:
                continue
            marker = id(manager)
            if marker in seen:
                continue
            seen.add(marker)
            managers.append(manager)
        return managers

    @staticmethod
    def _find_mcp_hub(tool_registry: Optional["ToolRegistry"]) -> Optional[Any]:
        if tool_registry is None:
            return None
        list_surfaces = getattr(tool_registry, "list_runtime_surfaces", None)
        if callable(list_surfaces):
            try:
                hubs = list_surfaces("mcp_hub")
                if hubs:
                    return next(iter(hubs.values()))
            except Exception:
                pass
        for tool in tool_registry.get_visible_tools(scope="all"):
            hub = getattr(tool, "hub", None)
            if hub is not None:
                return hub
        return None

    @staticmethod
    def _find_mcp_managers(tool_registry: Optional["ToolRegistry"]) -> list[Any]:
        if tool_registry is None:
            return []
        managers: list[Any] = []
        seen: set[int] = set()
        list_surfaces = getattr(tool_registry, "list_runtime_surfaces", None)
        if callable(list_surfaces):
            try:
                surfaces = list_surfaces("mcp_manager")
                for manager in surfaces.values():
                    marker = id(manager)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    managers.append(manager)
            except Exception:
                pass
        if managers:
            return managers
        for tool in tool_registry.get_visible_tools(scope="all"):
            manager = getattr(tool, "manager", None)
            if manager is None or getattr(manager, "source_identifier", None) is None:
                continue
            marker = id(manager)
            if marker in seen:
                continue
            seen.add(marker)
            managers.append(manager)
        return managers

    def _run_before_llm_request(
        self,
        messages: Any,
        *,
        request_kind: str,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        stream: bool = False,
        tools_enabled: bool = False,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, Optional[float], Optional[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        payload = {
            "agent": self,
            "messages": messages,
            "request_kind": request_kind,
            "temperature": temperature,
            "reasoning": reasoning,
            "stream": stream,
            "tools_enabled": tools_enabled,
            "provider_name": getattr(self.llm, "provider_name", None),
            "kwargs": dict(kwargs or {}),
        }
        outcome = self.hook_manager.before_llm_request(payload)
        if outcome.blocked:
            raise LLMInvokeError(outcome.message or "LLM 请求被 hook 阻断。")
        updated = outcome.payload
        return (
            updated.get("messages", messages),
            updated.get("temperature", temperature),
            updated.get("reasoning", reasoning),
            dict(updated.get("kwargs") or {}),
            outcome.audit,
        )

    def _run_after_llm_response(
        self,
        response: Any,
        *,
        messages: Any,
        request_kind: str,
        stream: bool,
        tools_enabled: bool,
        hook_audit: Optional[list[dict[str, Any]]] = None,
    ) -> Any:
        payload = {
            "agent": self,
            "messages": messages,
            "request_kind": request_kind,
            "stream": stream,
            "tools_enabled": tools_enabled,
            "provider_name": getattr(self.llm, "provider_name", None),
            "response": response,
            "hook_audit": list(hook_audit or []),
        }
        outcome = self.hook_manager.after_llm_response(payload)
        if outcome.blocked:
            raise LLMInvokeError(outcome.message or "LLM 响应被 hook 阻断。")
        return outcome.payload.get("response", response)

    def _run_before_tool_use(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], ToolSpec | None]:
        tool_spec = self.tool_registry.get_tool_spec(tool_name) if self.tool_registry is not None else None
        payload = {
            "agent": self,
            "tool_name": tool_name,
            "tool_args": dict(tool_args or {}),
            "tool_spec": tool_spec,
            "execution_context": self.execution_context,
            "permission_mode": getattr(self.permission_context.mode, "value", self.permission_context.mode),
        }
        outcome = self.hook_manager.before_tool_use(payload)
        if outcome.blocked:
            blocked_result = ToolResult.error(
                outcome.message or f"工具 '{tool_name}' 被 hook 阻断。",
                error_type=outcome.error_type,
                metadata={
                    "tool_name": tool_name,
                    "tool_args": dict(tool_args or {}),
                    "hook_audit": outcome.audit,
                },
            )
            raise HookExecutionError(
                blocked_result.content,
                stage="before_tool_use",
                error_type=blocked_result.error_type or "hook_blocked",
                metadata=blocked_result.metadata,
            )
        updated = outcome.payload
        return dict(updated.get("tool_args") or tool_args), outcome.audit, updated.get("tool_spec", tool_spec)

    def _run_after_tool_use(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolResult,
        *,
        tool_spec: ToolSpec | None = None,
        hook_audit: Optional[list[dict[str, Any]]] = None,
    ) -> ToolResult:
        payload = {
            "agent": self,
            "tool_name": tool_name,
            "tool_args": dict(tool_args or {}),
            "tool_spec": tool_spec,
            "tool_result": result,
            "execution_context": self.execution_context,
            "hook_audit": list(hook_audit or []),
        }
        outcome = self.hook_manager.after_tool_use(payload)
        if outcome.blocked:
            blocked_result = ToolResult.error(
                outcome.message or f"工具 '{tool_name}' 的结果被 hook 阻断。",
                error_type=outcome.error_type,
                metadata={
                    "tool_name": tool_name,
                    "tool_args": dict(tool_args or {}),
                    "hook_audit": [*(hook_audit or []), *outcome.audit],
                },
            )
            return blocked_result
        updated_result = outcome.payload.get("tool_result", result)
        if not isinstance(updated_result, ToolResult):
            raise ToolExecutionError(
                f"after_tool_use hook 必须返回 ToolResult，实际收到: {type(updated_result).__name__}"
            )
        merged_audit = [*(hook_audit or []), *outcome.audit]
        if merged_audit:
            updated_result.metadata = dict(updated_result.metadata)
            updated_result.metadata.setdefault("hook_audit", []).extend(merged_audit)
        return updated_result

    def _run_before_compaction(
        self,
        *,
        operation: str,
        max_tokens: Optional[int],
        force: bool = False,
        tokens_before: Optional[int] = None,
    ) -> Optional[int]:
        '''在历史压缩前运行 hook，允许动态调整压缩策略或阻断压缩'''
        payload = {
            "agent": self,
            "operation": operation,
            "max_tokens": max_tokens,
            "force": force,
            "tokens_before": tokens_before,
            "provider_name": getattr(self.llm, "provider_name", None),
            "history": list(self._history),
            "replay_history": list(self.replay_history),
            "pending_step_state": self.get_pending_step_state(),
        }
        outcome = self.hook_manager.before_compaction(payload)
        if outcome.blocked:
            self._last_history_compaction = {
                "was_compacted": False,
                "compaction_possible": True,
                "budget": max_tokens,
                "hook_blocked": True,
                "hook_message": outcome.message,
                "hook_audit": outcome.audit,
                "tracked_at": datetime.now().isoformat(),
            }
            return None
        updated = outcome.payload
        return updated.get("max_tokens", max_tokens)

    def _run_after_session_restore_hook(
        self,
        *,
        session_id: str,
        restore_report: SessionRestoreReport,
        snapshot: dict[str, Any],
    ) -> SessionRestoreReport:
        payload = {
            "agent": self,
            "session_id": session_id,
            "restore_report": restore_report,
            "snapshot": snapshot,
        }
        outcome = self.hook_manager.after_session_restore(payload)
        if outcome.blocked:
            restore_report.add_issue(
                component="hooks",
                code="after_session_restore_blocked",
                message=outcome.message or "after_session_restore hook 阻断了恢复后处理。",
                metadata={"hook_audit": outcome.audit},
            )
            return restore_report
        updated_report = outcome.payload.get("restore_report", restore_report)
        if not isinstance(updated_report, SessionRestoreReport):
            raise SessionSerializationError(
                f"after_session_restore hook 必须返回 SessionRestoreReport，实际收到: {type(updated_report).__name__}"
            )
        if outcome.audit:
            updated_report.extend_component(
                "hooks",
                {
                    "status": "applied",
                    "appliedHooks": outcome.audit,
                },
            )
        return updated_report

    def _build_mailbox_prompt(self) -> str:
        agent_runtime = getattr(self, "agent_runtime", None)
        runtime_agent_id = self._get_runtime_agent_id()
        if agent_runtime is None or not runtime_agent_id:
            return ""

        try:
            messages = agent_runtime.read_mailbox(
                runtime_agent_id,
                include_consumed=False,
                include_expired=False,
                mark_delivered=True,
            )
        except Exception as exc:
            logger.debug("构建 mailbox prompt 失败: %s", exc)
            return ""

        if not messages:
            return ""

        payload = [message.to_dict() for message in messages]
        return (
            "## 协作邮箱\n"
            "你当前有尚未确认的 mailbox 消息。这些消息已经投递到当前 agent，并在本轮系统提示中自动展示。\n"
            "- 这些消息可能来自 manager、team 广播或 task 广播，默认都应被视为当前执行约束的一部分。\n"
            "- 如果消息改变了目标、边界、优先级或输出格式，应先调整当前计划，再继续调用工具。\n"
            "- 如果需要重新读取完整结构化消息，调用 `MailboxRead`。\n"
            "- 当你已经阅读并把某条消息纳入执行后，调用 `MailboxAck` 把它标记为 consumed。\n"
            "- 不要假设 `SendMessage` 只用于提示；对协作型子 agent 来说，这些消息就是运行时输入。\n\n"
            "未确认消息:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        )

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
            if self.task_service is not None:
                self._register_task_tools()
            self._refresh_execution_context()
            return
        self.tool_registry = tool_registry
        self.enable_tool = tool_registry is not None
        if self.task_service is not None:
            self._register_task_tools()
            self._rebind_todo_write_tool()
        self._refresh_execution_context()

    def set_permission_mode(self, mode: PermissionMode | str) -> None:
        permission_mode = PermissionMode(mode)
        self.permission_context.set_mode(permission_mode)
        if permission_mode == PermissionMode.PLAN:
            self.mode_controller.enter_plan_mode(
                allowed_actions=self.mode_controller.state.allowed_actions
            )
        else:
            self.mode_controller.exit_plan_mode(
                allowed_actions=self.mode_controller.state.allowed_actions
            )
        self._refresh_execution_context()

    def add_permission_rule(
        self,
        rule: PermissionRule,
        *,
        source: str | None = None,
        priority: int | None = None,
    ) -> None:
        self.permission_context.add_rule(rule, source=source, priority=priority)
        self._refresh_execution_context()

    def set_permission_rules(
        self,
        source: str,
        rules: list[PermissionRule],
        *,
        priority: int | None = None,
    ) -> None:
        self.permission_context.set_source_rules(source, rules, priority=priority)
        self._refresh_execution_context()

    def clear_permission_rules(self, *, source: str | None = None) -> None:
        self.permission_context.clear_rules(source=source)
        self._refresh_execution_context()

    def enter_plan_mode(self, *, allowed_actions: Optional[list[str]] = None) -> None:
        self.mode_controller.enter_plan_mode(allowed_actions=allowed_actions)
        self.permission_context.set_mode(PermissionMode.PLAN)
        self._refresh_execution_context()

    def request_exit_plan_mode(self, *, allowed_actions: Optional[list[str]] = None) -> None:
        self.mode_controller.request_exit(allowed_actions=allowed_actions)

    def exit_plan_mode(self, *, permission_mode: PermissionMode | str = PermissionMode.DEFAULT) -> None:
        self.mode_controller.exit_plan_mode()
        self.permission_context.set_mode(permission_mode)
        self._refresh_execution_context()

    def get_execution_mode(self) -> ExecutionMode:
        return self.mode_controller.mode

    def set_current_task(self, task_id: Optional[str]) -> None:
        self.current_task_id = task_id
        self._refresh_execution_context()

    def request_stop(self, reason: str = "") -> None:
        self._stop_requested = True
        clean_reason = str(reason or "").strip()
        self._stop_reason = clean_reason or "Agent 已收到停止请求。"

    def clear_stop_request(self) -> None:
        self._stop_requested = False
        self._stop_reason = None

    def is_stop_requested(self) -> bool:
        return bool(self._stop_requested)

    def get_stop_reason(self) -> Optional[str]:
        return self._stop_reason

    def _raise_if_stop_requested(self) -> None:
        if self._stop_requested:
            raise AgentStopRequested(self._stop_reason or "Agent 已收到停止请求。")

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
        *,
        usage_anchor_replay_count: Optional[int] = None,
    ) -> None:
        self._assert_replay_history_ready_for_current_provider()
        canonical_entries, replay_entries = self._prepare_history_entries_for_persistence(
            canonical_entries,
            replay_entries,
        )
        provider_name = getattr(self.llm, "provider_name", None)
        for entry in canonical_entries:
            self._history.append(entry)
        canonical_trimmed = 0
        while len(self._history) > self.config.max_history_length:
            self._history.pop(0)
            canonical_trimmed += 1

        replay_anchor_index: Optional[int] = None
        if (
            self._pending_response_usage_anchor is not None
            and replay_entries
            and self._contains_assistant_history_entry(canonical_entries)
        ):
            anchor_count = (
                len(replay_entries)
                if usage_anchor_replay_count is None
                else max(0, min(int(usage_anchor_replay_count), len(replay_entries)))
            )
            if anchor_count > 0:
                replay_anchor_index = len(self.replay_history) + anchor_count
        for entry in replay_entries:
            self.llm.append_replay_entry(self.replay_history, entry, provider_name)
        replay_trimmed = 0
        while len(self.replay_history) > self.config.max_history_length:
            self.replay_history.pop(0)
            replay_trimmed += 1
        self.replay_history_provider_name = provider_name
        if replay_anchor_index is not None:
            self._install_history_usage_anchor(
                replay_index=replay_anchor_index - replay_trimmed,
                canonical_index=len(self._history),
            )
        elif self._pending_response_usage_anchor is not None and not self._contains_assistant_history_entry(canonical_entries):
            self._pending_response_usage_anchor = None
        elif replay_trimmed and self._history_usage_anchor is not None:
            self._shift_history_usage_anchor(replay_trimmed)
        self._check_and_trigger_background_memory()

    def sanitize_replay_history(self) -> int:
        """Drop provider-invalid replay entries before they reach the next request."""
        self._assert_replay_history_ready_for_current_provider()
        before = list(self.replay_history)
        sanitized = self.llm.prepare_messages_for_request(before)
        if sanitized == before:
            return 0
        self.replay_history = sanitized
        self.replay_history_provider_name = getattr(self.llm, "provider_name", None)
        self._invalidate_history_usage_anchor()
        return max(0, len(before) - len(sanitized))

    @staticmethod
    def _contains_assistant_history_entry(entries: list[Any]) -> bool:
        for entry in entries or []:
            canonical = coerce_canonical_message(entry)
            if canonical is not None and canonical.role == "assistant":
                return True
            if isinstance(entry, dict) and entry.get("role") == "assistant":
                return True
        return False

    @staticmethod
    def _usage_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _request_context_signature(self) -> str:
        payload = {
            "provider_name": getattr(self.llm, "provider_name", None),
            "system_prompt": self._stable_system_prompt(),
            "tools": self._stable_tools(),
            "reasoning": self.reasoning,
        }
        try:
            return json.dumps(self._make_json_safe(payload), ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(self._make_json_safe(payload))

    def _usage_context_tokens(self, usage: dict[str, Any]) -> Optional[int]:
        input_tokens = self._usage_int(usage.get("inputTokens"))
        output_tokens = self._usage_int(usage.get("outputTokens"))
        total_tokens = self._usage_int(usage.get("totalTokens"))
        if total_tokens is not None:
            return total_tokens
        if input_tokens is not None or output_tokens is not None:
            return int(input_tokens or 0) + int(output_tokens or 0)
        return None

    def _capture_response_usage_for_history_anchor(self, usage: dict[str, Any]) -> None:
        if usage.get("usageSource") != "provider":
            return
        context_tokens = self._usage_context_tokens(usage)
        if context_tokens is None:
            return
        self._pending_response_usage_anchor = {
            "provider_name": getattr(self.llm, "provider_name", None),
            "context_tokens": context_tokens,
            "usage": self._make_json_safe(usage),
            "context_signature": self._request_context_signature(),
            "tracked_at": datetime.now().isoformat(),
        }

    def _install_history_usage_anchor(
        self,
        *,
        replay_index: int,
        canonical_index: int,
    ) -> None:
        pending = self._pending_response_usage_anchor
        self._pending_response_usage_anchor = None
        if pending is None or replay_index <= 0:
            self._history_usage_anchor = None
            return
        self._history_usage_anchor = {
            **pending,
            "replay_index": replay_index,
            "canonical_index": max(0, canonical_index),
        }

    def _shift_history_usage_anchor(self, replay_trimmed: int) -> None:
        if self._history_usage_anchor is None:
            return
        replay_index = int(self._history_usage_anchor.get("replay_index") or 0) - int(replay_trimmed or 0)
        if replay_index <= 0:
            self._history_usage_anchor = None
            return
        self._history_usage_anchor = {
            **self._history_usage_anchor,
            "replay_index": replay_index,
        }

    def _invalidate_history_usage_anchor(self) -> None:
        self._history_usage_anchor = None
        self._pending_response_usage_anchor = None

    def _should_persist_reasoning_history(self) -> bool:
        return bool(getattr(self.config, "persist_reasoning_history", True))

    @staticmethod
    def _strip_signature_metadata(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: BaseAgent._strip_signature_metadata(item)
                for key, item in value.items()
                if key not in {"signature", "thought_signature", "thoughtSignature"}
            }
        if isinstance(value, list):
            return [BaseAgent._strip_signature_metadata(item) for item in value]
        if isinstance(value, tuple):
            return [BaseAgent._strip_signature_metadata(item) for item in value]
        return value

    def _sanitize_canonical_entry_for_persistence(self, entry: Any) -> Any:
        canonical = coerce_canonical_message(entry)
        if canonical is None:
            return entry
        blocks: list[CanonicalBlock] = []
        for block in canonical.content:
            if block.type == "reasoning":
                continue
            blocks.append(
                block.model_copy(
                    update={
                        "signature": None,
                        "payload": self._strip_signature_metadata(block.payload),
                        "metadata": self._strip_signature_metadata(block.metadata),
                    }
                )
            )
        return canonical.model_copy(
            update={
                "content": blocks,
                "metadata": self._strip_signature_metadata(canonical.metadata),
            }
        )

    def _prepare_history_entries_for_persistence(
        self,
        canonical_entries: list[Any],
        replay_entries: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        if self._should_persist_reasoning_history():
            return list(canonical_entries or []), list(replay_entries or [])
        sanitized_canonical = [
            self._sanitize_canonical_entry_for_persistence(entry)
            for entry in list(canonical_entries or [])
        ]
        sanitized_replay = self.llm.canonical_to_replay_history(
            sanitized_canonical,
            getattr(self.llm, "provider_name", None),
        )
        return sanitized_canonical, sanitized_replay

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
            "tool_ephemeral_contexts": [],
            "tool_calls": _json_safe(tool_calls or []),
            "round_number": round_number,
            "provider_name": getattr(self.llm, "provider_name", None),
        }

    def _append_pending_tool_result(
        self,
        *,
        tool_canonical: list[Any],
        tool_replay: list[Any],
        ephemeral_context: Any = None,
        tool_name: Optional[str] = None,
    ) -> None:
        if self._pending_step_state is None:
            return
        self._pending_step_state["tool_results_canonical"].extend(list(tool_canonical or []))
        self._pending_step_state["tool_results_replay"].extend(list(tool_replay or []))
        if ephemeral_context is not None:
            self._pending_step_state["tool_ephemeral_contexts"].append(
                _json_safe(
                    {
                        "tool_name": tool_name,
                        "context": ephemeral_context,
                    }
                )
            )

    def _commit_pending_step_state(self) -> bool:
        if not self._pending_step_state:
            return False
        assistant_canonical = list(self._pending_step_state.get("assistant_canonical") or [])
        assistant_replay = list(self._pending_step_state.get("assistant_replay") or [])
        tool_results_canonical = list(self._pending_step_state.get("tool_results_canonical") or [])
        tool_results_replay = list(self._pending_step_state.get("tool_results_replay") or [])
        if assistant_canonical or assistant_replay:
            self._append_dual_history(assistant_canonical, assistant_replay)
        if tool_results_canonical or tool_results_replay:
            self._append_dual_history(tool_results_canonical, tool_results_replay)
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
        """构建记忆系统静态策略提示，不在 system prompt 中注入动态记忆内容。"""
        memory_manage = getattr(self, "memory_manage", None)
        memory_skill = self._get_active_memory_skill()
        if memory_manage is None and memory_skill is not None:
            memory_manage = getattr(memory_skill, "memory_manage", None)

        if not memory_manage:
            return ""

        supported_memory_types: list[str] | None = None
        try:
            supported_memory_types = list(getattr(memory_manage, "memory_types", {}).keys())
        except Exception:
            supported_memory_types = None

        return build_memory_prompt_section(
            supported_memory_types=supported_memory_types,
            include_working_memory=False,
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
        self._history_usage_anchor = None
        self._pending_response_usage_anchor = None
        self._last_cache_signature = None
        self._last_cache_usage = None
        self._last_cache_break = None
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

    def get_system_prompt_blocks(self) ->  list[PromptBlock]:
        """获取系统提示的 PromptBlock 列表，供历史压缩时使用。默认实现返回空列表，子类可覆写以提供更丰富的提示结构。"""
        return []
    def _stable_system_prompt(self) -> Optional[str]:
        try:
            from core.request_compiler import compile_prompt_blocks

            compiled = compile_prompt_blocks(
                self.get_system_prompt_blocks(),
                cache_policy=getattr(self.config, "cache_policy", None),
                cache_dynamic_memory=bool(getattr(self.config, "cache_dynamic_memory", False)),
                cache_dynamic_mailbox=bool(getattr(self.config, "cache_dynamic_mailbox", False)),
                cache_turn_skills=bool(getattr(self.config, "cache_turn_skills", False)),
            )
            return compiled.system_prompt
        except Exception:
            return self.get_enhanced_prompt()

    def _stable_tools(self) -> Optional[Any]:
        if self.tool_registry is None:
            return None
        return self.get_provider_tools()

    def _cache_signature_for_messages(
        self,
        messages: Any = None,
        *,
        reasoning: Optional[dict[str, Any]] = None,
        tools_enabled: bool = False,
    ) -> dict[str, Any]:
        if isinstance(messages, ReplayRequestInput) and isinstance(messages.cache_signature, dict):
            return self._make_json_safe(messages.cache_signature)
        from core.cache_policy import stable_hash

        return {
            "provider": getattr(self.llm, "provider_name", None),
            "model": getattr(self.llm, "model", None),
            "system_hash": stable_hash(self._stable_system_prompt()),
            "tools_hash": stable_hash(self._stable_tools() if tools_enabled else None),
            "reasoning_hash": stable_hash(reasoning if reasoning is not None else self.reasoning),
            "extra_hash": stable_hash(None),
            "cache_policy_hash": stable_hash(getattr(self.config, "cache_policy", None)),
        }

    @staticmethod
    def _cache_signature_changed_fields(
        previous: Optional[dict[str, Any]],
        current: Optional[dict[str, Any]],
    ) -> list[str]:
        if not previous or not current:
            return []
        fields: list[str] = []
        for key in sorted(set(previous) | set(current)):
            if previous.get(key) != current.get(key):
                fields.append(str(key))
        return fields

    def _record_cache_break(
        self,
        *,
        reason: str,
        changed_fields: Optional[list[str]] = None,
        previous_signature: Optional[dict[str, Any]] = None,
        current_signature: Optional[dict[str, Any]] = None,
        previous_cache_read_tokens: Optional[int] = None,
        current_cache_read_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if not bool(getattr(self.config, "record_cache_breaks", True)):
            return None
        event = self.observability_recorder.record_cache_break(
            reason=reason,
            changed_fields=changed_fields,
            previous_signature=previous_signature,
            current_signature=current_signature,
            previous_cache_read_tokens=previous_cache_read_tokens,
            current_cache_read_tokens=current_cache_read_tokens,
            metadata=metadata,
        )
        self._last_cache_break = self._make_json_safe(event)
        return event

    def _maybe_record_cache_signature_change(
        self,
        signature: Optional[dict[str, Any]],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not signature:
            return
        previous = self._last_cache_signature
        changed_fields = self._cache_signature_changed_fields(previous, signature)
        if previous and changed_fields:
            self._record_cache_break(
                reason="cache_signature_changed",
                changed_fields=changed_fields,
                previous_signature=previous,
                current_signature=signature,
                metadata=metadata,
            )
        self._last_cache_signature = self._make_json_safe(signature)

    def _cache_read_tokens_from_usage(self, usage: dict[str, Any]) -> Optional[int]:
        cache_read = self._usage_int(usage.get("cacheReadTokens"))
        cached_input = self._usage_int(usage.get("cachedInputTokens"))
        if cache_read is None and cached_input is None:
            return None
        if cache_read is not None:
            return int(cache_read)
        return int(cached_input or 0)

    def _maybe_record_cache_read_drop(self, usage: dict[str, Any]) -> None:
        current_read = self._cache_read_tokens_from_usage(usage)
        previous_read = None
        if isinstance(self._last_cache_usage, dict):
            previous_read = self._usage_int(self._last_cache_usage.get("cacheReadTokensForBreakDetection"))
        if previous_read is not None and current_read is not None:
            drop = int(previous_read) - int(current_read)
            if drop > 0 and drop >= max(2000, int(previous_read * 0.05)):
                self._record_cache_break(
                    reason="cache_read_drop",
                    changed_fields=[],
                    previous_signature=self._last_cache_signature,
                    current_signature=self._last_cache_signature,
                    previous_cache_read_tokens=previous_read,
                    current_cache_read_tokens=current_read,
                    metadata={"dropTokens": drop, "usage": self._make_json_safe(usage)},
                )
        self._last_cache_usage = {
            **self._make_json_safe(usage),
            "cacheReadTokensForBreakDetection": current_read,
        }

    def _apply_history_compaction_result(self, result: Any) -> bool:
        previous_signature = self._last_cache_signature
        previous_replay_messages = len(self.replay_history)
        self._last_history_compaction = _build_history_compaction_state(
            was_compacted=result.was_compacted,
            compaction_possible=result.compaction_possible,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            max_tokens=result.budget, 
            metadata=getattr(result, "metadata", None),
        )
        if not result.was_compacted:
            return False
        self._history = list(result.canonical_history)
        self.replay_history = list(result.replay_history)
        self.replay_history_provider_name = getattr(self.llm, "provider_name", None)
        self._invalidate_history_usage_anchor()
        self._record_cache_break(
            reason="history_compacted",
            changed_fields=["history.compacted", "replay_history"],
            previous_signature=previous_signature,
            current_signature=None,
            metadata={
                "previousReplayMessages": previous_replay_messages,
                "currentReplayMessages": len(self.replay_history),
                "compaction": self._make_json_safe(self._last_history_compaction),
            },
        )
        self._last_cache_signature = None
        return True

    def _local_history_compaction_tokens(self) -> Optional[int]:
        if not self.replay_history:
            return 0
        try:
            codec = create_codec(getattr(self.llm, "provider_name", None))
            return codec.count_request_tokens(
                self._context_usage_counter,
                self.replay_history,
                system_prompt=self._stable_system_prompt(),
                tools=self._stable_tools(),
                reasoning=self.reasoning,
            )
        except Exception:
            return None

    def _estimate_history_compaction_token_state(self) -> dict[str, Any]:
        local_tokens = self._local_history_compaction_tokens()
        anchor = self._history_usage_anchor
        if not anchor:
            return {
                "tokens": local_tokens,
                "source": "local_request_estimate",
                "local_tokens": local_tokens,
                "metadata": {"source": "local_request_estimate"},
            }
        if anchor.get("provider_name") != getattr(self.llm, "provider_name", None):
            self._history_usage_anchor = None
            return {
                "tokens": local_tokens,
                "source": "local_request_estimate",
                "local_tokens": local_tokens,
                "metadata": {"source": "local_request_estimate", "anchor_invalidated": "provider_changed"},
            }
        if anchor.get("context_signature") != self._request_context_signature():
            return {
                "tokens": local_tokens,
                "source": "local_request_estimate",
                "local_tokens": local_tokens,
                "metadata": {"source": "local_request_estimate", "anchor_invalidated": "request_context_changed"},
            }
        replay_index = self._usage_int(anchor.get("replay_index"))
        context_tokens = self._usage_int(anchor.get("context_tokens"))
        if replay_index is None or context_tokens is None or replay_index < 0 or replay_index > len(self.replay_history):
            return {
                "tokens": local_tokens,
                "source": "local_request_estimate",
                "local_tokens": local_tokens,
                "metadata": {"source": "local_request_estimate", "anchor_invalidated": "index_out_of_range"},
            }
        delta_replay = self.replay_history[replay_index:]
        try:
            codec = create_codec(getattr(self.llm, "provider_name", None))
            delta_tokens = codec.count_request_tokens(
                self._context_usage_counter,
                delta_replay,
            ) if delta_replay else 0
        except Exception:
            return {
                "tokens": local_tokens,
                "source": "local_request_estimate",
                "local_tokens": local_tokens,
                "metadata": {"source": "local_request_estimate", "anchor_invalidated": "delta_estimate_failed"},
            }
        tokens = int(context_tokens) + int(delta_tokens)
        return {
            "tokens": tokens,
            "source": "provider_usage_plus_delta_estimate",
            "local_tokens": local_tokens,
            "metadata": {
                "source": "provider_usage_plus_delta_estimate",
                "provider_usage_tokens": int(context_tokens),
                "delta_tokens": int(delta_tokens),
                "delta_replay_messages": len(delta_replay),
                "anchor_replay_index": replay_index,
                "local_tokens": local_tokens,
                "usage": self._make_json_safe(anchor.get("usage") or {}),
            },
        }

    def _precheck_history_compaction(
        self,
        *,
        max_tokens: Optional[int],
        force: bool = False,
    ) -> tuple[bool, Optional[int], dict[str, Any]]:
        if max_tokens is None or max_tokens <= 0:
            return False, None, {}
        token_state = self._estimate_history_compaction_token_state()
        tokens_before = token_state.get("tokens")
        metadata = {"token_estimate": token_state.get("metadata") or {}}
        if tokens_before is None:
            return True, None, metadata
        if force or tokens_before > max_tokens:
            return True, tokens_before, metadata
        self._last_history_compaction = _build_history_compaction_state(
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return False, tokens_before, metadata

    def compact_persistent_history_if_needed(self) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        should_compact, tokens_before, token_metadata = self._precheck_history_compaction(
            max_tokens=budget,
            force=False,
        )
        if not should_compact:
            return False
        budget = self._run_before_compaction(
            operation="compact_persistent_history_if_needed",
            max_tokens=budget,
            force=False,
            tokens_before=tokens_before,
        )
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
            tokens_before_override=tokens_before,
            metadata=token_metadata,
        )
        return self._apply_history_compaction_result(result)

    async def acompact_persistent_history_if_needed(self) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        should_compact, tokens_before, token_metadata = self._precheck_history_compaction(
            max_tokens=budget,
            force=False,
        )
        if not should_compact:
            return False
        budget = self._run_before_compaction(
            operation="acompact_persistent_history_if_needed",
            max_tokens=budget,
            force=False,
            tokens_before=tokens_before,
        )
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
            tokens_before_override=tokens_before,
            metadata=token_metadata,
        )
        return self._apply_history_compaction_result(result)

    def compact_history(self, max_tokens: Optional[int] = None) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = max_tokens if max_tokens is not None else self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        _, tokens_before, token_metadata = self._precheck_history_compaction(
            max_tokens=budget,
            force=True,
        )
        budget = self._run_before_compaction(
            operation="compact_history",
            max_tokens=budget,
            force=True,
            tokens_before=tokens_before,
        )
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
            tokens_before_override=tokens_before,
            metadata=token_metadata,
        )
        return self._apply_history_compaction_result(result)

    async def acompact_history(self, max_tokens: Optional[int] = None) -> bool:
        if self.context_manager is None or not self._history:
            return False
        budget = max_tokens if max_tokens is not None else self._history_budget_max_tokens()
        if budget is None or budget <= 0:
            return False
        _, tokens_before, token_metadata = self._precheck_history_compaction(
            max_tokens=budget,
            force=True,
        )
        budget = self._run_before_compaction(
            operation="acompact_history",
            max_tokens=budget,
            force=True,
            tokens_before=tokens_before,
        )
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
            tokens_before_override=tokens_before,
            metadata=token_metadata,
        )
        return self._apply_history_compaction_result(result)

    def _get_serializable_state(self) -> dict[str, Any]:
        """返回子类需要补充持久化的状态。"""
        return {
            "last_history_compaction": self._make_json_safe(self._last_history_compaction),
            "history_usage_anchor": self._make_json_safe(self._history_usage_anchor),
            "pending_response_usage_anchor": self._make_json_safe(self._pending_response_usage_anchor),
            "last_cache_signature": self._make_json_safe(self._last_cache_signature),
            "last_cache_usage": self._make_json_safe(self._last_cache_usage),
            "last_cache_break": self._make_json_safe(self._last_cache_break),
            "pending_step_state": self._make_json_safe(self._pending_step_state),
            "mode_state": self.mode_controller.export_state(),
            "permission_context": self.permission_context.export_state(),
            "current_task_id": self.current_task_id,
            "replay_history_state": ReplayHistoryState(
                provider_name=self.replay_history_provider_name,
                messages=[
                    self._serialize_replay_entry(message)
                    for message in self.replay_history
                ],
            ).to_dict(),
            "observability_state": self.observability_recorder.export_state(),
        }

    def _restore_serializable_state(self, state: Optional[dict[str, Any]]) -> None:
        """恢复子类持久化状态。"""
        state = state or {}
        self._last_history_compaction = self._make_json_safe(state.get("last_history_compaction") or {})
        self._history_usage_anchor = self._make_json_safe(state.get("history_usage_anchor") or None)
        self._pending_response_usage_anchor = self._make_json_safe(state.get("pending_response_usage_anchor") or None)
        self._last_cache_signature = self._make_json_safe(state.get("last_cache_signature") or None)
        self._last_cache_usage = self._make_json_safe(state.get("last_cache_usage") or None)
        self._last_cache_break = self._make_json_safe(state.get("last_cache_break") or None)
        pending_state = state.get("pending_step_state")
        self._pending_step_state = self._make_json_safe(pending_state) if pending_state is not None else None
        self.mode_controller.restore_state(state.get("mode_state"))
        self.permission_context.restore_state(state.get("permission_context"))
        self.current_task_id = state.get("current_task_id")
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
        self.observability_recorder.restore_state(state.get("observability_state"))
        self.observability_recorder.set_agent_name(self.name)
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

        context_manager_snapshot = None
        if self.context_manager is not None:
            context_manager_snapshot = {
                "max_tokens": self.context_manager.budget.max_tokens,
                "source_names": list(getattr(self.context_manager.builder, "source_names", [])),
                "history_compactor": self.context_manager.history_compactor.__class__.__name__,
                "formatter": self.context_manager.builder.formatter.__class__.__name__,
            }

        task_service_snapshot = None
        if self.task_service is not None:
            store = getattr(self.task_service, "store", None)
            task_service_snapshot = {
                "backend": "custom",
            }
            db_path = getattr(store, "db_path", None)
            if isinstance(db_path, str) and db_path:
                task_service_snapshot = {
                    "backend": "sqlite",
                    "db_path": db_path,
                }
            elif store is not None and store.__class__.__name__ == "InMemoryTaskStore":
                task_service_snapshot = {
                    "backend": "memory",
                }

        execution_context_snapshot = None
        execution_context = getattr(self, "execution_context", None)
        if execution_context is not None and hasattr(execution_context, "to_dict"):
            execution_context_snapshot = execution_context.to_dict()

        collaboration_runtime_snapshot = None
        if self.agent_runtime is not None or self.team_manager is not None:
            collaboration_runtime_snapshot = {
                "agent_runtime": (
                    self._make_json_safe(self.agent_runtime.export_state())
                    if self.agent_runtime is not None and hasattr(self.agent_runtime, "export_state")
                    else None
                ),
                "teams": (
                    self._make_json_safe(self.team_manager.export_state())
                    if self.team_manager is not None and hasattr(self.team_manager, "export_state")
                    else None
                ),
            }
        worktree_runtime_snapshot = None
        worktree_manager = self._find_worktree_manager(self.tool_registry)
        if worktree_manager is not None and hasattr(worktree_manager, "export_state"):
            try:
                worktree_runtime_snapshot = self._make_json_safe(worktree_manager.export_state())
            except Exception as exc:
                logger.warning("导出 worktree runtime 状态失败: %s", exc)

        mcp_runtime_snapshot = None
        mcp_hub = self._find_mcp_hub(self.tool_registry)
        mcp_managers = self._find_mcp_managers(self.tool_registry)
        if mcp_hub is not None and hasattr(mcp_hub, "export_state"):
            try:
                mcp_runtime_snapshot = {
                    "hub": self._make_json_safe(mcp_hub.export_state()),
                }
            except Exception as exc:
                logger.warning("导出 MCP hub 状态失败: %s", exc)
        elif mcp_managers:
            try:
                mcp_runtime_snapshot = {
                    "managers": [
                        self._make_json_safe(manager.export_state())
                        for manager in mcp_managers
                        if hasattr(manager, "export_state")
                    ],
                }
            except Exception as exc:
                logger.warning("导出 MCP manager 状态失败: %s", exc)

        codeintel_runtime_snapshot = None
        codeintel_managers = self._find_codeintel_managers(self.tool_registry)
        if codeintel_managers:
            try:
                codeintel_runtime_snapshot = {
                    "managers": [
                        self._make_json_safe(manager.export_state())
                        for manager in codeintel_managers
                        if hasattr(manager, "export_state")
                    ],
                }
            except Exception as exc:
                logger.warning("导出 codeintel manager 状态失败: %s", exc)

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
            "has_task_service": self.task_service is not None,
            "context_manager": self._make_json_safe(context_manager_snapshot),
            "task_service": self._make_json_safe(task_service_snapshot),
            "execution_context": self._make_json_safe(execution_context_snapshot),
            "collaboration_runtime": self._make_json_safe(collaboration_runtime_snapshot),
            "worktree_runtime": self._make_json_safe(worktree_runtime_snapshot),
            "mcp_runtime": self._make_json_safe(mcp_runtime_snapshot),
            "codeintel_runtime": self._make_json_safe(codeintel_runtime_snapshot),
            "state": self._make_json_safe(self._get_serializable_state()),
        }

    @staticmethod
    def _snapshot_config(snapshot: dict[str, Any]) -> Config:
        config_data = snapshot.get("config") or {}
        return Config(**config_data) if config_data else Config.from_env()

    @classmethod
    def _auto_restore_context_manager(
        cls,
        snapshot: dict[str, Any],
        *,
        context_manager: Optional["ContextManager"] = None,
        config: Optional[Config] = None,
    ) -> Optional["ContextManager"]:
        if context_manager is not None:
            return context_manager
        context_snapshot = snapshot.get("context_manager") or {}
        if not context_snapshot and not snapshot.get("has_context_manager"):
            return None
        max_tokens = context_snapshot.get("max_tokens")
        if max_tokens is None:
            max_tokens = getattr(config, "max_tokens", None) or 8000
        return ContextManager(max_tokens=int(max_tokens))

    @classmethod
    def _auto_restore_task_service(
        cls,
        snapshot: dict[str, Any],
        *,
        task_service: Optional[Any] = None,
    ) -> Optional[Any]:
        if task_service is not None:
            return task_service

        task_snapshot = snapshot.get("task_service") or {}
        has_task_tools = any(
            name in {"TaskCreate", "TaskGet", "TaskUpdate", "TaskList"}
            for name in list(snapshot.get("tool_names") or [])
        )
        if not task_snapshot and not snapshot.get("has_task_service") and not has_task_tools:
            return None

        try:
            from task import InMemoryTaskStore, SQLiteTaskStore, TaskService
        except Exception as exc:
            logger.warning("自动恢复 TaskService 失败: %s", exc)
            return None

        backend = str(task_snapshot.get("backend") or "sqlite").lower()
        try:
            if backend == "memory":
                return TaskService(InMemoryTaskStore())
            if backend == "sqlite":
                db_path = str(task_snapshot.get("db_path") or "db/easyagent_tasks.db")
                return TaskService(SQLiteTaskStore(db_path))
        except Exception as exc:
            logger.warning("自动恢复 TaskService 失败: %s", exc)
            return None

        logger.warning("TaskService backend '%s' 暂不支持自动恢复，请手动注入。", backend)
        return None

    @classmethod
    def _auto_restore_skill_manager(
        cls,
        snapshot: dict[str, Any],
        *,
        skill_manager: Optional["SkillManager"] = None,
    ) -> tuple[Optional["SkillManager"], list[str]]:
        if skill_manager is not None:
            return skill_manager, []

        registered_skills = list(snapshot.get("registered_skills") or [])
        active_skills = list(snapshot.get("active_skills") or [])
        if not registered_skills and not active_skills:
            return None, []

        try:
            from skill.registry import SkillRegistry
        except Exception as exc:
            logger.warning("自动恢复 SkillManager 失败: %s", exc)
            return None, []

        manager = SkillManager()
        registry = SkillRegistry.instance()
        manager.bind_registry(registry)

        missing: list[str] = []
        for name in registered_skills:
            if not registry.has(name):
                missing.append(name)
                continue
            try:
                manager.register(registry.create(name), auto_activate=False)
            except Exception as exc:
                logger.warning("自动注册 Skill '%s' 失败: %s", name, exc)

        if missing:
            logger.warning("以下 Skill 未在 SkillRegistry 中注册，无法自动恢复: %s", missing)

        auto_activate = [name for name in active_skills if manager.has_skill(name)]
        return manager, auto_activate

    @classmethod
    def _auto_restore_tool_registry(
        cls,
        snapshot: dict[str, Any],
        *,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[Config] = None,
        task_service: Optional[Any] = None,
        mcp_client_overrides: Optional[dict[str, Any]] = None,
    ) -> Optional["ToolRegistry"]:
        if tool_registry is not None:
            return tool_registry

        expected_tools = list(snapshot.get("tool_names") or [])
        if not expected_tools and not snapshot.get("enable_tool"):
            return None

        from Tool.ToolRegistry import ToolRegistry

        registry = ToolRegistry()
        config = config or cls._snapshot_config(snapshot)
        workspace_root = os.path.abspath(config.workspace_root or os.getcwd())
        allowed_roots = config.get_allowed_roots()
        cwd = workspace_root

        shell_tools = {"Bash", "TaskOutput", "TaskStop"}
        process_manager = None
        if any(name in shell_tools for name in expected_tools):
            try:
                from Tool.runtime import ProcessManager

                process_manager = ProcessManager(
                    shell=config.shell,
                    max_background_tasks=config.max_background_tasks,
                )
            except Exception as exc:
                logger.warning("创建 ProcessManager 失败，相关 shell 工具可能无法自动恢复: %s", exc)

        worktree_manager = None
        if any(name in {"EnterWorktree", "ExitWorktree"} for name in expected_tools):
            try:
                from Tool.runtime import WorktreeManager

                repo_root = WorktreeManager.detect_repo_root(
                    workspace_root,
                    git_binary=config.git_binary,
                )
                worktree_manager = WorktreeManager(
                    repo_root,
                    git_binary=config.git_binary,
                    original_cwd=workspace_root,
                )
            except Exception as exc:
                logger.warning("自动恢复 WorktreeManager 失败: %s", exc)

        try:
            from Tool.builtin import (
                register_ask_user_question_tool,
                register_bash_tool,
                register_calculator_tool,
                register_codeintel_tools,
                register_config_tool,
                register_enter_plan_mode_tool,
                register_enter_worktree_tool,
                register_exit_plan_mode_tool,
                register_exit_worktree_tool,
                register_file_edit_tool,
                register_file_read_tool,
                register_file_write_tool,
                register_glob_tool,
                register_grep_tool,
                register_notebook_edit_tool,
                register_search_tool,
                register_task_tools,
                register_task_output_tool,
                register_task_stop_tool,
                register_todo_write_tool,
                register_web_fetch_tool,
            )
        except Exception as exc:
            logger.warning("导入 builtin tool 注册器失败，无法自动恢复 ToolRegistry: %s", exc)
            return registry

        for tool_name in expected_tools:
            if registry.has_tool(tool_name):
                continue
            try:
                if (
                    tool_name in {"TaskCreate", "TaskGet", "TaskUpdate", "TaskList"}
                    and task_service is not None
                ):
                    register_task_tools(registry, service=task_service)
                    continue
                if tool_name == "WebSearch":
                    register_search_tool(registry)
                elif tool_name == "Calculator":
                    register_calculator_tool(registry)
                elif tool_name == "FileRead":
                    register_file_read_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd)
                elif tool_name == "Glob":
                    register_glob_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd)
                elif tool_name == "Grep":
                    register_grep_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd)
                elif tool_name == "FileWrite":
                    register_file_write_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd)
                elif tool_name == "FileEdit":
                    register_file_edit_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd)
                elif tool_name == "Bash":
                    register_bash_tool(
                        registry,
                        workspace_root,
                        allowed_roots=allowed_roots,
                        cwd=cwd,
                        shell=config.shell,
                        command_timeout_ms=config.command_timeout_ms,
                        max_background_tasks=config.max_background_tasks,
                        process_manager=process_manager,
                    )
                elif tool_name == "TaskOutput":
                    register_task_output_tool(
                        registry,
                        workspace_root,
                        allowed_roots=allowed_roots,
                        cwd=cwd,
                        shell=config.shell,
                        command_timeout_ms=config.command_timeout_ms,
                        max_background_tasks=config.max_background_tasks,
                        process_manager=process_manager,
                    )
                elif tool_name == "TaskStop":
                    register_task_stop_tool(
                        registry,
                        workspace_root,
                        allowed_roots=allowed_roots,
                        cwd=cwd,
                        shell=config.shell,
                        command_timeout_ms=config.command_timeout_ms,
                        max_background_tasks=config.max_background_tasks,
                        process_manager=process_manager,
                    )
                elif tool_name == "WebFetch":
                    register_web_fetch_tool(registry)
                elif tool_name == "TodoWrite":
                    register_todo_write_tool(
                        registry,
                        service=task_service,
                    )
                elif tool_name == "NotebookEdit":
                    register_notebook_edit_tool(registry, workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
                elif tool_name == "AskUserQuestion":
                    register_ask_user_question_tool(registry)
                elif tool_name == "EnterPlanMode":
                    register_enter_plan_mode_tool(registry)
                elif tool_name == "ExitPlanMode":
                    register_exit_plan_mode_tool(registry)
                elif tool_name == "Config":
                    register_config_tool(registry, config=config)
                elif tool_name == "EnterWorktree" and worktree_manager is not None:
                    register_enter_worktree_tool(registry, worktree_manager=worktree_manager)
                elif tool_name == "ExitWorktree" and worktree_manager is not None:
                    register_exit_worktree_tool(registry, worktree_manager=worktree_manager)
            except Exception as exc:
                logger.warning("自动恢复工具 '%s' 失败: %s", tool_name, exc)

        mcp_runtime_snapshot = snapshot.get("mcp_runtime") or {}
        if mcp_runtime_snapshot:
            try:
                from Tool.builtin.mcp_tool import MCPToolManager, register_mcp_resource_hub_tools
                from Emcp import MCPHub

                overrides = dict(mcp_client_overrides or {})
                hub_payload = mcp_runtime_snapshot.get("hub")
                standalone_manager_payloads = list(mcp_runtime_snapshot.get("managers") or [])

                if hub_payload:
                    hub = MCPHub()
                    register_surface = getattr(registry, "register_runtime_surface", None)
                    if callable(register_surface):
                        register_surface("mcp_hub", "default", hub)
                    for item in list(hub_payload.get("servers") or []):
                        server_name = str(item.get("serverName") or "").strip()
                        manager_state = dict(item.get("manager") or {})
                        client = overrides.get(server_name)
                        try:
                            manager = MCPToolManager.from_state(manager_state, client=client)
                            manager.register_to_registry(
                                registry,
                                include_resources=manager.include_resources,
                                resource_tool_prefix=manager.resource_tool_prefix,
                                hub=hub,
                                server_name=server_name or manager.registry_server_name,
                                legacy_resource_tools=False,
                            )
                        except Exception as exc:
                            logger.warning("自动恢复 MCP server '%s' 失败: %s", server_name, exc)
                    if any(name in expected_tools for name in {"ListMcpResources", "ReadMcpResource"}):
                        register_mcp_resource_hub_tools(registry, hub)

                for manager_state in standalone_manager_payloads:
                    state_payload = dict(manager_state or {})
                    server_name = str(state_payload.get("registryServerName") or state_payload.get("serverName") or "").strip()
                    client = overrides.get(server_name)
                    try:
                        manager = MCPToolManager.from_state(state_payload, client=client)
                        manager.register_to_registry(
                            registry,
                            include_resources=manager.include_resources,
                            resource_tool_prefix=manager.resource_tool_prefix,
                        )
                    except Exception as exc:
                        logger.warning("自动恢复 MCP manager '%s' 失败: %s", server_name, exc)
            except Exception as exc:
                logger.warning("自动恢复 MCP runtime 失败: %s", exc)

        codeintel_runtime_snapshot = snapshot.get("codeintel_runtime") or {}
        expected_codeintel_tools = {
            "CodeIntelStatus",
            "CodeIntelCacheStatus",
            "CodeIntelPrewarmWorkspace",
            "FindDefinition",
            "FindReferences",
            "GetDocumentSymbols",
            "GetWorkspaceSymbols",
            "GetDiagnostics",
        }
        if codeintel_runtime_snapshot or any(name in expected_tools for name in expected_codeintel_tools):
            try:
                from codeintel import CodeIntelManager

                manager_payloads = list(codeintel_runtime_snapshot.get("managers") or [])
                manager = None
                if manager_payloads:
                    manager = CodeIntelManager.from_state(
                        manager_payloads[0],
                        workspace_root=workspace_root,
                        allowed_roots=allowed_roots,
                    )
                register_codeintel_tools(
                    registry,
                    manager=manager,
                    workspace_root=workspace_root,
                    allowed_roots=tuple(allowed_roots or (workspace_root,)),
                )
            except Exception as exc:
                logger.warning("自动恢复 codeintel runtime 失败: %s", exc)

        if not registry.get_tool_names():
            if expected_tools:
                return registry
            return None
        return registry

    @classmethod
    def _restore_post_init_dependencies(
        cls,
        agent: "BaseAgent",
        snapshot: dict[str, Any],
        *,
        auto_activate_skill_names: Optional[list[str]] = None,
        restore_report: Optional[SessionRestoreReport] = None,
    ) -> None:
        if restore_report is None:
            restore_report = SessionRestoreReport(
                session_id="unknown",
                agent_type=snapshot.get("agent_type") or agent.__class__.__name__,
            )
        for skill_name in list(auto_activate_skill_names or []):
            if not agent.skill_manager.has_skill(skill_name):
                continue
            try:
                agent.skill_manager.activate(skill_name)
            except Exception as exc:
                logger.warning("恢复会话时激活 Skill '%s' 失败: %s", skill_name, exc)

        expected_tools = list(snapshot.get("tool_names") or [])
        runtime_snapshot = snapshot.get("collaboration_runtime") or {}
        execution_context_snapshot = snapshot.get("execution_context") or {}
        worktree_snapshot = snapshot.get("worktree_runtime") or {}
        collaboration_tools = {
            "Agent",
            "AgentGet",
            "AgentList",
            "AgentWait",
            "AgentStop",
            "SendMessage",
            "MailboxRead",
            "MailboxAck",
            "TeamCreate",
            "TeamDelete",
        }
        wants_collaboration_runtime = bool(runtime_snapshot) or any(
            name in collaboration_tools for name in expected_tools
        )

        restored_execution_context = None
        try:
            from runtime import ExecutionContext

            restored_execution_context = ExecutionContext.from_dict(execution_context_snapshot)
        except Exception:
            restored_execution_context = None
        if restored_execution_context is not None:
            restore_report.execution_context_restored = True
            restore_report.extend_component(
                "execution_context",
                {
                    "status": "restored",
                    "restoredItems": ["execution_context"],
                    "metadata": {"currentTaskId": restored_execution_context.current_task_id},
                },
            )
        elif execution_context_snapshot:
            restore_report.add_issue(
                component="execution_context",
                code="execution_context_restore_failed",
                message="execution_context 快照存在，但恢复失败。",
                metadata={"snapshot": execution_context_snapshot},
            )

        if wants_collaboration_runtime:
            runtime = None
            team_manager = None
            helper_tool = None
            workspace_root = getattr(agent.config, "workspace_root", None)
            allowed_roots = agent.config.get_allowed_roots() if agent.config is not None else None

            try:
                from Tool.builtin.agent_tool import AgentTool

                existing_tool = None
                if agent.tool_registry is not None and agent.tool_registry.has_tool("Agent"):
                    existing_tool = agent.tool_registry.get_tool("Agent")
                shared_worktree_manager = cls._find_worktree_manager(agent.tool_registry)
                if isinstance(existing_tool, AgentTool):
                    helper_tool = existing_tool
                else:
                    helper_tool = AgentTool(
                        parent_agent=agent,
                        worktree_manager=shared_worktree_manager,
                        workspace_root=workspace_root,
                        allowed_roots=allowed_roots,
                    )
                runtime = helper_tool.agent_runtime
            except Exception as exc:
                logger.warning("创建协作运行时失败: %s", exc)
                runtime = None
                restore_report.add_issue(
                    component="agent_runtime",
                    code="runtime_creation_failed",
                    message=f"创建协作运行时失败: {exc}",
                )

            if runtime is not None:
                try:
                    from runtime import TeamManager

                    team_manager = getattr(runtime, "team_manager", None) or TeamManager(agent_runtime=runtime)
                    runtime.bind_team_manager(team_manager)
                except Exception as exc:
                    logger.warning("创建 TeamManager 失败: %s", exc)
                    team_manager = None
                    restore_report.add_issue(
                        component="team_runtime",
                        code="team_manager_creation_failed",
                        message=f"创建 TeamManager 失败: {exc}",
                    )

                try:
                    if team_manager is not None and runtime_snapshot.get("teams"):
                        team_report = team_manager.restore_state(runtime_snapshot.get("teams"))
                        restore_report.extend_component("team_runtime", team_report)
                except Exception as exc:
                    logger.warning("恢复 team 状态失败: %s", exc)
                    restore_report.add_issue(
                        component="team_runtime",
                        code="team_restore_failed",
                        message=f"恢复 team 状态失败: {exc}",
                    )

                try:
                    if runtime_snapshot.get("agent_runtime"):
                        runtime_report = runtime.restore_state(runtime_snapshot.get("agent_runtime"))
                        restore_report.extend_component("agent_runtime", runtime_report)
                except Exception as exc:
                    logger.warning("恢复 agent runtime 状态失败: %s", exc)
                    restore_report.add_issue(
                        component="agent_runtime",
                        code="runtime_restore_failed",
                        message=f"恢复 agent runtime 状态失败: {exc}",
                    )

                try:
                    agent.bind_runtime(
                        agent_runtime=runtime,
                        team_manager=team_manager,
                        execution_context=restored_execution_context,
                    )
                except Exception as exc:
                    logger.warning("绑定协作运行时失败: %s", exc)
                    restore_report.add_issue(
                        component="agent_runtime",
                        code="runtime_bind_failed",
                        message=f"绑定协作运行时失败: {exc}",
                    )

                if agent.tool_registry is not None:
                    try:
                        from Tool.builtin import (
                            register_agent_tool,
                            register_agent_runtime_tools,
                            register_mailbox_tools,
                            register_send_message_tool,
                            register_team_create_tool,
                            register_team_delete_tool,
                        )

                        if "Agent" in expected_tools:
                            register_agent_tool(
                                agent.tool_registry,
                                parent_agent=agent,
                                agent_runtime=runtime,
                                worktree_manager=getattr(helper_tool, "worktree_manager", None),
                                workspace_root=workspace_root,
                                allowed_roots=allowed_roots,
                                storage_dir=getattr(runtime, "storage_dir", None),
                                max_background_tasks=getattr(getattr(runtime, "subagent_manager", None), "max_background_tasks", 4),
                            )
                        if any(name in expected_tools for name in {"AgentGet", "AgentList", "AgentWait", "AgentStop"}):
                            register_agent_runtime_tools(
                                agent.tool_registry,
                                agent_runtime=runtime,
                                parent_agent=agent,
                            )
                        if "SendMessage" in expected_tools:
                            register_send_message_tool(
                                agent.tool_registry,
                                agent_runtime=runtime,
                                parent_agent=agent,
                            )
                        if any(name in expected_tools for name in {"MailboxRead", "MailboxAck"}):
                            register_mailbox_tools(
                                agent.tool_registry,
                                agent_runtime=runtime,
                                parent_agent=agent,
                            )
                        if "TeamCreate" in expected_tools and team_manager is not None:
                            register_team_create_tool(
                                agent.tool_registry,
                                team_manager=team_manager,
                                parent_agent=agent,
                            )
                        if "TeamDelete" in expected_tools and team_manager is not None:
                            register_team_delete_tool(
                                agent.tool_registry,
                                team_manager=team_manager,
                            )
                    except Exception as exc:
                        logger.warning("恢复协作工具失败: %s", exc)
                        restore_report.add_issue(
                            component="tools",
                            code="collaboration_tool_restore_failed",
                            message=f"恢复协作工具失败: {exc}",
                        )
        elif restored_execution_context is not None:
            try:
                agent.bind_runtime(execution_context=restored_execution_context)
            except Exception as exc:
                logger.warning("恢复 execution_context 失败: %s", exc)
                restore_report.add_issue(
                    component="execution_context",
                    code="execution_context_bind_failed",
                    message=f"恢复 execution_context 失败: {exc}",
                )

        worktree_manager = cls._find_worktree_manager(agent.tool_registry)
        if worktree_snapshot:
            if worktree_manager is None:
                restore_report.add_issue(
                    component="worktree_runtime",
                    code="worktree_manager_missing",
                    message="会话包含 worktree runtime 状态，但恢复时没有可用的 WorktreeManager。",
                )
            elif hasattr(worktree_manager, "restore_state"):
                try:
                    worktree_report = worktree_manager.restore_state(worktree_snapshot)
                    restore_report.extend_component("worktree_runtime", worktree_report)
                except Exception as exc:
                    logger.warning("恢复 worktree runtime 状态失败: %s", exc)
                    restore_report.add_issue(
                        component="worktree_runtime",
                        code="worktree_restore_failed",
                        message=f"恢复 worktree runtime 状态失败: {exc}",
                    )

        mcp_runtime_snapshot = snapshot.get("mcp_runtime") or {}
        expected_mcp_servers: list[str] = []
        hub_snapshot = mcp_runtime_snapshot.get("hub") or {}
        for item in list(hub_snapshot.get("servers") or []):
            server_name = str(item.get("serverName") or "").strip()
            if server_name:
                expected_mcp_servers.append(server_name)
        for item in list(mcp_runtime_snapshot.get("managers") or []):
            server_name = str(
                item.get("registryServerName")
                or item.get("serverName")
                or ""
            ).strip()
            if server_name:
                expected_mcp_servers.append(server_name)
        expected_mcp_servers = sorted(set(expected_mcp_servers))

        if mcp_runtime_snapshot or expected_mcp_servers:
            current_hub = cls._find_mcp_hub(agent.tool_registry)
            current_managers = cls._find_mcp_managers(agent.tool_registry)
            current_servers = sorted(
                {
                    str(getattr(manager, "registry_server_name", None) or getattr(manager, "server_label", "")).strip()
                    for manager in current_managers
                    if str(getattr(manager, "registry_server_name", None) or getattr(manager, "server_label", "")).strip()
                }
            )
            missing_servers = [name for name in expected_mcp_servers if name not in current_servers]
            restore_report.extend_component(
                "mcp_runtime",
                {
                    "status": "restored" if not missing_servers else "degraded",
                    "restoredItems": current_servers,
                    "degradedItems": missing_servers,
                    "metadata": {
                        "serverCount": len(current_servers),
                        "hasHub": current_hub is not None,
                    },
                },
            )
            if missing_servers:
                restore_report.add_issue(
                    component="mcp_runtime",
                    code="mcp_runtime_partial_restore",
                    message=f"MCP runtime 仅部分恢复，缺少 server: {missing_servers}",
                    metadata={"missingServers": missing_servers},
                )

        codeintel_runtime_snapshot = snapshot.get("codeintel_runtime") or {}
        expected_codeintel_tools = {
            "CodeIntelStatus",
            "CodeIntelCacheStatus",
            "CodeIntelPrewarmWorkspace",
            "FindDefinition",
            "FindReferences",
            "GetDocumentSymbols",
            "GetWorkspaceSymbols",
            "GetDiagnostics",
        }
        if codeintel_runtime_snapshot or any(name in expected_tools for name in expected_codeintel_tools):
            codeintel_managers = cls._find_codeintel_managers(agent.tool_registry)
            for manager in codeintel_managers:
                bind_parent = getattr(manager, "bind_parent_agent", None)
                if callable(bind_parent):
                    try:
                        bind_parent(agent)
                    except Exception as exc:
                        logger.warning("绑定 codeintel manager 的 parent_agent 失败: %s", exc)
            manager_payloads = list(codeintel_runtime_snapshot.get("managers") or [])
            current_workspace_roots: list[str] = []
            cache_workspace_roots: list[str] = []
            for manager in codeintel_managers:
                workspace_root = str(getattr(manager, "workspace_root", "") or "").strip()
                if workspace_root:
                    current_workspace_roots.append(workspace_root)
                get_cache_status = getattr(manager, "get_cache_status", None)
                if callable(get_cache_status):
                    try:
                        cache_status = dict(get_cache_status() or {})
                        cache_workspace_root = str(cache_status.get("workspaceRoot") or "").strip()
                        if cache_workspace_root:
                            cache_workspace_roots.append(cache_workspace_root)
                    except Exception:
                        pass
            missing_managers = max(0, len(manager_payloads) - len(codeintel_managers))
            restore_report.extend_component(
                "codeintel_runtime",
                {
                    "status": "restored" if missing_managers == 0 else "degraded",
                    "restoredItems": current_workspace_roots or cache_workspace_roots,
                    "degradedItems": ["codeintel_manager"] * missing_managers if missing_managers else [],
                    "metadata": {
                        "managerCount": len(codeintel_managers),
                        "workspaceRoots": sorted(set(current_workspace_roots or cache_workspace_roots)),
                    },
                },
            )
            if missing_managers:
                restore_report.add_issue(
                    component="codeintel_runtime",
                    code="codeintel_runtime_partial_restore",
                    message=f"codeintel runtime 仅部分恢复，缺少 {missing_managers} 个 manager。",
                    metadata={"expectedManagerCount": len(manager_payloads), "actualManagerCount": len(codeintel_managers)},
                )

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
        permission_engine: Optional["PermissionEngine"] = None,
        permission_context: Optional["PermissionContext"] = None,
        hook_manager: Optional["HookManager"] = None,
        task_service: Optional[Any] = None,
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
            "permission_engine": permission_engine,
            "permission_context": permission_context,
            "hook_manager": hook_manager,
            "task_service": task_service,
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
        permission_engine: Optional["PermissionEngine"] = None,
        permission_context: Optional["PermissionContext"] = None,
        hook_manager: Optional["HookManager"] = None,
        task_service: Optional[Any] = None,
    ) -> dict[str, Any]:
        return cls._build_base_constructor_kwargs(
            snapshot,
            llm=llm,
            tool_registry=tool_registry,
            memory_manage=memory_manage,
            context_manager=context_manager,
            callback_manager=callback_manager,
            skill_manager=skill_manager,
            permission_engine=permission_engine,
            permission_context=permission_context,
            hook_manager=hook_manager,
            task_service=task_service,
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
        permission_engine: Optional["PermissionEngine"] = None,
        permission_context: Optional["PermissionContext"] = None,
        hook_manager: Optional["HookManager"] = None,
        task_service: Optional[Any] = None,
        mcp_client_overrides: Optional[dict[str, Any]] = None,
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
        snapshot_config = cls._snapshot_config(snapshot)
        task_service = cls._auto_restore_task_service(
            snapshot,
            task_service=task_service,
        )
        context_manager = cls._auto_restore_context_manager(
            snapshot,
            context_manager=context_manager,
            config=snapshot_config,
        )
        skill_manager, auto_activate_skill_names = cls._auto_restore_skill_manager(
            snapshot,
            skill_manager=skill_manager,
        )
        tool_registry = cls._auto_restore_tool_registry(
            snapshot,
            tool_registry=tool_registry,
            config=snapshot_config,
            task_service=task_service,
            mcp_client_overrides=mcp_client_overrides,
        )

        if cls is not BaseAgent and target_cls is not cls:
            raise SessionSerializationError(
                f"会话 {session_id} 属于 {target_cls.__name__}，无法按 {cls.__name__} 恢复"
            )
        if not target_cls._supports_session_restore():
            raise SessionSerializationError(
                f"{target_cls.__name__} 暂不支持自动恢复，请手动重建实例"
            )

        restore_report = SessionRestoreReport(
            session_id=session_id,
            agent_type=target_cls.__name__,
            metadata={"storePath": getattr(session_store, "db_path", None)},
        )
        init_kwargs = target_cls._build_constructor_kwargs_from_snapshot(
            snapshot,
            llm=llm,
            tool_registry=tool_registry,
            memory_manage=memory_manage,
            context_manager=context_manager,
            callback_manager=callback_manager,
            skill_manager=skill_manager,
            permission_engine=permission_engine,
            permission_context=permission_context,
            hook_manager=hook_manager,
            task_service=task_service,
        )
        agent = target_cls(**init_kwargs)
        target_cls._restore_post_init_dependencies(
            agent,
            snapshot,
            auto_activate_skill_names=auto_activate_skill_names,
            restore_report=restore_report,
        )
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
            restore_report.note_missing_tools(missing_tools)

        expected_skills = snapshot.get("active_skills") or []
        if expected_skills:
            if skill_manager is None:
                logger.warning("恢复会话时未提供 skill_manager，以下 Skill 需手动恢复: %s", expected_skills)
                restore_report.note_missing_skills(list(expected_skills))
            else:
                missing_skills = [name for name in expected_skills if not skill_manager.has_skill(name)]
                if missing_skills:
                    logger.warning("恢复会话时缺少 Skill 实现: %s", missing_skills)
                    restore_report.note_missing_skills(missing_skills)

        if snapshot.get("enable_tool") and tool_registry is None:
            logger.warning("会话原本启用了工具，但恢复时未注入 ToolRegistry，已降级为无工具模式")
            restore_report.add_issue(
                component="tools",
                code="tool_registry_missing",
                message="会话原本启用了工具，但恢复时未注入 ToolRegistry，已降级为无工具模式。",
            )

        restore_report = agent._run_after_session_restore_hook(
            session_id=session_id,
            restore_report=restore_report,
            snapshot=snapshot,
        )
        agent.last_restore_report = restore_report

        logger.info("会话已恢复: %s", session_id)
        return agent

    def get_last_restore_report(self) -> Optional[dict[str, Any]]:
        report = getattr(self, "last_restore_report", None)
        if report is None:
            return None
        if hasattr(report, "to_dict"):
            return report.to_dict()
        return report

    def close(
        self,
        *,
        close_runtime: bool = True,
        close_worktree: bool = True,
        worktree_action: str = "keep",
        discard_worktree_changes: bool = False,
        close_llm: bool = True,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "closed",
            "metadata": {
                "agentName": self.name,
                "agentType": self.__class__.__name__,
            },
            "components": {},
            "issues": [],
        }

        def _merge_component_status(status: str) -> None:
            normalized = str(status or "").strip().lower()
            if normalized == "failed":
                report["status"] = "failed"
            elif normalized == "degraded" and report["status"] == "closed":
                report["status"] = "degraded"

        def _normalize_component_payload(
            name: str,
            payload: Any,
            *,
            default_status: str = "closed",
        ) -> dict[str, Any]:
            if isinstance(payload, dict):
                component = dict(payload)
                if "status" not in component:
                    component = {
                        "status": default_status,
                        "metadata": component,
                        "issues": [],
                    }
            else:
                component = {
                    "status": default_status,
                    "metadata": {"value": payload},
                    "issues": [],
                }
            component.setdefault("status", default_status)
            component.setdefault("metadata", {})
            component.setdefault("issues", [])
            report["components"][name] = component
            _merge_component_status(component.get("status", default_status))
            return component

        if close_runtime and self.agent_runtime is not None:
            runtime_close = getattr(self.agent_runtime, "close", None)
            if callable(runtime_close):
                try:
                    runtime_report = runtime_close()
                    _normalize_component_payload("agent_runtime", runtime_report)
                except Exception as exc:
                    report["status"] = "failed"
                    issue = {
                        "component": "agent_runtime",
                        "code": "runtime_close_failed",
                        "message": f"关闭协作运行时失败: {exc}",
                        "severity": "error",
                    }
                    report["issues"].append(issue)
                    report["components"]["agent_runtime"] = {
                        "status": "failed",
                        "metadata": {},
                        "issues": [issue],
                    }

        if close_worktree:
            worktree_manager = self._find_worktree_manager(self.tool_registry)
            if worktree_manager is not None:
                worktree_close = getattr(worktree_manager, "close", None)
                if callable(worktree_close):
                    try:
                        worktree_report = worktree_close(
                            action=worktree_action,
                            discard_changes=discard_worktree_changes,
                        )
                        if worktree_report is None:
                            worktree_report = {
                                "status": "closed",
                                "metadata": {"hadActiveSession": False},
                                "issues": [],
                            }
                        _normalize_component_payload("worktree_runtime", worktree_report)
                    except Exception as exc:
                        report["status"] = "failed"
                        issue = {
                            "component": "worktree_runtime",
                            "code": "worktree_close_failed",
                            "message": f"关闭 worktree runtime 失败: {exc}",
                            "severity": "error",
                        }
                        report["issues"].append(issue)
                        report["components"]["worktree_runtime"] = {
                            "status": "failed",
                            "metadata": {},
                            "issues": [issue],
                        }

        mcp_hub = self._find_mcp_hub(self.tool_registry)
        mcp_managers = self._find_mcp_managers(self.tool_registry)
        if mcp_hub is not None:
            hub_close = getattr(mcp_hub, "close", None)
            if callable(hub_close):
                try:
                    hub_report = hub_close()
                    _normalize_component_payload("mcp_runtime", hub_report)
                except Exception as exc:
                    report["status"] = "failed"
                    issue = {
                        "component": "mcp_runtime",
                        "code": "mcp_hub_close_failed",
                        "message": f"关闭 MCP hub 失败: {exc}",
                        "severity": "error",
                    }
                    report["issues"].append(issue)
                    report["components"]["mcp_runtime"] = {
                        "status": "failed",
                        "metadata": {},
                        "issues": [issue],
                    }
        elif mcp_managers:
            mcp_component = {
                "status": "closed",
                "metadata": {"managerCount": len(mcp_managers)},
                "issues": [],
            }
            for manager in mcp_managers:
                close_fn = getattr(manager, "close", None)
                if not callable(close_fn):
                    continue
                try:
                    close_fn()
                except Exception as exc:
                    mcp_component["status"] = "degraded"
                    mcp_component["issues"].append(
                        {
                            "component": "mcp_runtime",
                            "code": "mcp_manager_close_failed",
                            "message": f"关闭 MCP manager 失败: {exc}",
                            "severity": "warning",
                        }
                    )
                    if report["status"] == "closed":
                        report["status"] = "degraded"
            report["components"]["mcp_runtime"] = mcp_component

        codeintel_managers = self._find_codeintel_managers(self.tool_registry)
        if codeintel_managers:
            codeintel_component = {
                "status": "closed",
                "metadata": {"managerCount": len(codeintel_managers)},
                "issues": [],
            }
            for manager in codeintel_managers:
                close_fn = getattr(manager, "close", None)
                if not callable(close_fn):
                    continue
                try:
                    close_fn()
                except Exception as exc:
                    codeintel_component["status"] = "degraded"
                    codeintel_component["issues"].append(
                        {
                            "component": "codeintel",
                            "code": "codeintel_close_failed",
                            "message": f"关闭 codeintel manager 失败: {exc}",
                            "severity": "warning",
                        }
                    )
                    if report["status"] == "closed":
                        report["status"] = "degraded"
            report["components"]["codeintel"] = codeintel_component

        if close_llm and self.llm is not None:
            llm_close = getattr(self.llm, "close", None)
            if callable(llm_close):
                try:
                    llm_close()
                    _normalize_component_payload(
                        "llm",
                        {
                            "status": "closed",
                            "metadata": {
                                "providerName": getattr(self.llm, "provider_name", None),
                                "model": getattr(self.llm, "model", None),
                            },
                            "issues": [],
                        },
                    )
                except Exception as exc:
                    report["status"] = "failed"
                    issue = {
                        "component": "llm",
                        "code": "llm_close_failed",
                        "message": f"关闭 LLM 连接失败: {exc}",
                        "severity": "error",
                    }
                    report["issues"].append(issue)
                    report["components"]["llm"] = {
                        "status": "failed",
                        "metadata": {},
                        "issues": [issue],
                    }

        self.last_close_report = report
        return report

    def get_last_close_report(self) -> Optional[dict[str, Any]]:
        report = getattr(self, "last_close_report", None)
        if report is None:
            return None
        return dict(report)

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
        from core.cache_policy import build_cache_signature
        from core.request_compiler import compile_prompt_blocks
        from core.request_input import ReplayRequestInput

        compiled = compile_prompt_blocks(
            self.get_system_prompt_blocks(),
            cache_policy=getattr(self.config, "cache_policy", None),
            cache_dynamic_memory=bool(getattr(self.config, "cache_dynamic_memory", False)),
            cache_dynamic_mailbox=bool(getattr(self.config, "cache_dynamic_mailbox", False)),
            cache_turn_skills=bool(getattr(self.config, "cache_turn_skills", False)),
        )
        request_input = ReplayRequestInput(
            provider_name=getattr(self.llm, "provider_name", None),
            replay_history=list(self.replay_history),
            system_prompt=compiled.system_prompt,
            system_prompt_blocks=compiled.system_prompt_blocks,
            runtime_reminder_blocks=compiled.runtime_reminder_blocks,
            cache_policy=compiled.cache_policy,
        )
        request_input.apply_runtime_layers()
        system_prompt = request_input.render_system_prompt()
        tools = self._stable_tools()
        request_input.cache_signature = build_cache_signature(
            provider=getattr(self.llm, "provider_name", None),
            model=getattr(self.llm, "model", None),
            system_blocks=[
                *request_input.system_prompt_blocks,
                *request_input.runtime_reminder_blocks,
            ],
            tools=tools,
            reasoning=self.reasoning,
            extra={
                "runtime_reminders": [
                    block.to_dict() for block in request_input.runtime_reminder_blocks
                    if block.cacheable
                ],
            },
            cache_policy=request_input.cache_policy,
        ).to_dict()
        reminder_text = request_input.render_runtime_reminders()
        expansion_text = request_input.render_on_demand_expansions()
        dynamic_tail_text = request_input.render_dynamic_tail()
        breakdown = self._estimate_request_token_breakdown(
            replay_history=request_input.replay_history,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=self.reasoning,
        )
        compaction_token_state = self._estimate_history_compaction_token_state()
        estimate_tokens = compaction_token_state.get("tokens")
        estimate_source = compaction_token_state.get("source") or "local_request_estimate"
        estimate_metadata = dict(compaction_token_state.get("metadata") or {})
        if estimate_tokens is None:
            estimate_tokens = breakdown["estimated_request_tokens"]
            estimate_source = "local_request_estimate"
            estimate_metadata = {"source": "local_request_estimate"}

        def _count_replay_tokens(messages: list[Any]) -> int:
            try:
                return int(self.llm.count_request_tokens(self._context_usage_counter, messages))
            except Exception:
                return 0

        persistent_replay_tokens = _count_replay_tokens(request_input.persistent_replay_history)
        prepended_replay_tokens = _count_replay_tokens(request_input.prepended_replay_history)
        appended_replay_tokens = _count_replay_tokens(request_input.appended_replay_history)
        runtime_layer_tokens = max(0, breakdown["history_tokens"] - persistent_replay_tokens)
        anchor = self._history_usage_anchor if isinstance(self._history_usage_anchor, dict) else None
        pending_anchor = (
            self._pending_response_usage_anchor
            if isinstance(self._pending_response_usage_anchor, dict)
            else None
        )
        capability = getattr(self.llm.provider, "get_cache_capability", lambda: None)()
        if hasattr(capability, "to_dict"):
            capability = capability.to_dict()
        cache_state = {
            "enabled": bool(getattr(getattr(self, "config", None), "cache_policy", None).enabled if getattr(getattr(self, "config", None), "cache_policy", None) is not None else False),
            "signature": anchor.get("context_signature") if anchor else None,
            "lastSignature": self._make_json_safe(self._last_cache_signature),
            "requestPrefixSignature": self._make_json_safe(request_input.cache_signature),
            "lastBreak": self._make_json_safe(self._last_cache_break),
            "lastCacheUsage": self._make_json_safe(self._last_cache_usage),
            "providerCapability": self._make_json_safe(capability),
            "requestCacheMetadata": self._make_json_safe(getattr(request_input, "cache_metadata", None)),
            "anchorActive": anchor is not None,
            "anchorProvider": anchor.get("provider_name") if anchor else None,
            "anchorReplayIndex": anchor.get("replay_index") if anchor else None,
            "pendingAnchorActive": pending_anchor is not None,
        }
        usage = _build_context_usage_report_v2(
            max_tokens=max_tokens,
            history_budget_tokens=history_budget,
            history_tokens=breakdown["history_tokens"],
            system_tokens=breakdown["system_tokens"],
            tool_tokens=breakdown["tool_tokens"],
            reasoning_tokens=breakdown["reasoning_tokens"],
            estimated_request_tokens=int(estimate_tokens),
            request_estimate_source=str(estimate_source),
            request_estimate_metadata=estimate_metadata,
            request_layers={
                "systemBlocks": [block.name for block in request_input.system_prompt_blocks],
                "runtimeReminderBlocks": [block.name for block in request_input.runtime_reminder_blocks],
                "onDemandExpansionBlocks": [block.name for block in request_input.on_demand_expansion_blocks],
                "dynamicTailBlocks": [block.name for block in request_input.dynamic_tail_blocks],
                "runtimeReminderTokens": self._context_usage_counter.count(reminder_text),
                "onDemandExpansionTokens": self._context_usage_counter.count(expansion_text),
                "dynamicTailTokens": self._context_usage_counter.count(dynamic_tail_text),
                "effectiveReplayTokens": breakdown["history_tokens"],
                "persistentReplayTokens": persistent_replay_tokens,
                "prependedReplayTokens": prepended_replay_tokens,
                "appendedReplayTokens": appended_replay_tokens,
                "runtimeLayerTokens": runtime_layer_tokens,
                "tokenBreakdownSource": "local_request_estimate",
            },
            canonical_history_messages=len(self._history),
            replay_history_messages=len(request_input.replay_history),
            compaction=self._last_history_compaction,
            compaction_estimated_request_tokens=compaction_token_state.get("tokens"),
            compaction_token_source=compaction_token_state.get("source"),
            compaction_metadata=compaction_token_state.get("metadata") or {},
            cache=cache_state,
            pending_step_active=self._pending_step_state is not None,
        )
        return self._make_json_safe(usage)

    def _estimate_request_token_breakdown(
        self,
        *,
        replay_history: list[Any],
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        history_tokens = self.llm.count_request_tokens(
            self._context_usage_counter,
            replay_history,
        )
        history_plus_system_tokens = self.llm.count_request_tokens(
            self._context_usage_counter,
            replay_history,
            system_prompt=system_prompt,
        )
        history_plus_system_plus_tools_tokens = self.llm.count_request_tokens(
            self._context_usage_counter,
            replay_history,
            system_prompt=system_prompt,
            tools=tools,
        )
        estimated_request_tokens = self.llm.count_request_tokens(
            self._context_usage_counter,
            replay_history,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=reasoning,
        )
        system_tokens = max(0, history_plus_system_tokens - history_tokens)
        tool_tokens = max(0, history_plus_system_plus_tools_tokens - history_plus_system_tokens)
        reasoning_tokens = max(0, estimated_request_tokens - history_plus_system_plus_tools_tokens)
        return {
            "history_tokens": int(history_tokens),
            "system_tokens": int(system_tokens),
            "tool_tokens": int(tool_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "estimated_request_tokens": int(estimated_request_tokens),
        }

    def _estimate_llm_request_tokens(
        self,
        messages: Any,
        *,
        reasoning: Optional[dict[str, Any]] = None,
        tools_enabled: bool = False,
    ) -> Optional[int]:
        try:
            request_input = self.llm._prepare_request_input(messages)
            tools = self._stable_tools() if tools_enabled and self.tool_registry is not None else None
            return self.llm.count_request_tokens(
                self._context_usage_counter,
                request_input.replay_history,
                system_prompt=request_input.system_prompt,
                tools=tools,
                reasoning=reasoning,
            )
        except Exception:
            return None

    def _estimate_llm_output_tokens(
        self,
        *,
        response: Any = None,
        final_text: Optional[str] = None,
        final_thinking: Optional[str] = None,
    ) -> dict[str, Any]:
        usage = self.llm.extract_usage_metrics(response)
        text = final_text
        thinking = final_thinking
        if text is None:
            text = self.llm.get_response_content(response) if response is not None else None
        if thinking is None:
            thinking = self.llm.get_thinking_content(response) if response is not None else None
        if text is None and response is not None and self.llm.has_tool_calls(response):
            try:
                text = json.dumps(self._make_json_safe(self.llm.get_tool_calls(response)), ensure_ascii=False)
            except Exception:
                text = None
        if usage.get("outputTokens") is None:
            estimated_text = str(text or "")
            estimated_thinking = str(thinking or "")
            usage["outputTokens"] = self._context_usage_counter.count(estimated_text) + self._context_usage_counter.count(estimated_thinking)
            usage.setdefault("usageSource", "estimated")
        if usage.get("totalTokens") is None and usage.get("inputTokens") is not None and usage.get("outputTokens") is not None:
            usage["totalTokens"] = int(usage["inputTokens"]) + int(usage["outputTokens"])
        return usage

    def _observe_agent_run_start(
        self,
        query: str,
        *,
        mode: str,
        stream: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.observability_recorder.begin_agent_run(
            query=query,
            mode=mode,
            stream=stream,
            metadata=metadata,
        )

    def _observe_agent_run_end(
        self,
        event_id: Optional[str],
        *,
        output: str,
        success: bool,
        error: Optional[BaseException] = None,
        turn_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not event_id:
            return
        self.observability_recorder.end_agent_run(
            event_id,
            output=output,
            success=success,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            turn_id=turn_id,
            metadata=metadata,
        )

    def _observe_llm_request_start(
        self,
        *,
        turn_id: Optional[str],
        request_kind: str,
        messages: Any,
        reasoning: Optional[dict[str, Any]] = None,
        stream: bool,
        tools_enabled: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        cache_signature = self._cache_signature_for_messages(
            messages,
            reasoning=reasoning,
            tools_enabled=tools_enabled,
        )
        cache_metadata = {
            **dict(metadata or {}),
            "cacheSignature": self._make_json_safe(cache_signature),
        }
        self._maybe_record_cache_signature_change(
            cache_signature,
            metadata={
                "turnId": turn_id,
                "requestKind": request_kind,
                "stream": stream,
                "toolsEnabled": tools_enabled,
            },
        )
        return self.observability_recorder.begin_llm_request(
            turn_id=turn_id,
            request_kind=request_kind,
            stream=stream,
            tools_enabled=tools_enabled,
            provider_name=getattr(self.llm, "provider_name", None),
            model=getattr(self.llm, "model", None),
            input_tokens=self._estimate_llm_request_tokens(
                messages,
                reasoning=reasoning,
                tools_enabled=tools_enabled,
            ),
            metadata=cache_metadata,
        )

    def _observe_llm_request_end(
        self,
        event_id: Optional[str],
        *,
        response: Any = None,
        success: bool,
        error: Optional[BaseException] = None,
        final_text: Optional[str] = None,
        final_thinking: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not event_id:
            return
        usage = self._estimate_llm_output_tokens(
            response=response,
            final_text=final_text,
            final_thinking=final_thinking,
        )
        if success:
            self._capture_response_usage_for_history_anchor(usage)
            self._maybe_record_cache_read_drop(usage)
        self.observability_recorder.end_llm_request(
            event_id,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            total_tokens=usage.get("totalTokens"),
            cached_input_tokens=usage.get("cachedInputTokens"),
            reasoning_tokens=usage.get("reasoningTokens"),
            cache_read_tokens=usage.get("cacheReadTokens"),
            cache_creation_tokens=usage.get("cacheCreationTokens"),
            tool_use_prompt_tokens=usage.get("toolUsePromptTokens"),
            usage_source=usage.get("usageSource"),
            success=success,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            cost_usd=usage.get("costUsd"),
            metadata={**dict(metadata or {}), **usage},
        )

    def _observe_tool_execution_start(
        self,
        *,
        turn_id: Optional[str],
        tool_name: str,
        tool_args: dict[str, Any],
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
        round_number: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        tool_spec = self.tool_registry.get_tool_spec(tool_name) if self.tool_registry is not None else None
        spec_metadata = {}
        if tool_spec is not None:
            spec_metadata = {
                "sideEffectLevel": tool_spec.side_effect_level,
                "visibilityScope": tool_spec.visibility_scope,
                "resourceScope": list(tool_spec.resource_scope or []),
                "toolSource": tool_spec.source,
            }
        return self.observability_recorder.begin_tool_execution(
            turn_id=turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            mode=mode,
            stream=stream,
            round_number=round_number,
            metadata={**spec_metadata, **dict(metadata or {})},
        )

    def _observe_tool_execution_end(
        self,
        event_id: Optional[str],
        *,
        result: Optional[ToolResult] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not event_id:
            return
        self.observability_recorder.end_tool_execution(
            event_id,
            success=(result.status == "success") if result is not None else False,
            result_status=(result.status if result is not None else "error"),
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            metadata={
                **dict(metadata or {}),
                **(dict(result.metadata or {}) if result is not None else {}),
            },
        )

    def get_observability_summary(self) -> dict[str, Any]:
        return self._make_json_safe(self.observability_recorder.get_summary())

    def get_recent_observability_events(
        self,
        *,
        limit: int = 20,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return self._make_json_safe(
            self.observability_recorder.get_recent_events(limit=limit, event_type=event_type)
        )

    def get_trace_summary(self, *, limit_turns: int = 5) -> list[dict[str, Any]]:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        return self._make_json_safe(
            self.observability_recorder.get_trace_summary(trace_history, limit_turns=limit_turns)
        )

    def list_agent_runs(self) -> list[dict[str, Any]]:
        getter = getattr(self.observability_recorder, "list_agent_runs", None)
        if not callable(getter):
            return []
        return self._make_json_safe(getter())

    def get_agent_run(self, run_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        getter = getattr(self.observability_recorder, "get_agent_run", None)
        if not callable(getter):
            return None
        run = getter(run_id)
        return self._make_json_safe(run) if run is not None else None

    def export_run_record(
        self,
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> Optional[dict[str, Any]]:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_run_record", None)
        if not callable(exporter):
            return None
        payload = exporter(trace_history, run_id=run_id, redact=redact)
        return self._make_json_safe(payload) if payload is not None else None

    def export_eval_trace(
        self,
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> Optional[dict[str, Any]]:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_eval_trace", None)
        if not callable(exporter):
            return None
        payload = exporter(trace_history, run_id=run_id, redact=redact)
        return self._make_json_safe(payload) if payload is not None else None

    def export_run_record_jsonl(
        self,
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_run_record_jsonl", None)
        if not callable(exporter):
            return ""
        return str(exporter(trace_history, run_id=run_id, redact=redact) or "")

    def export_eval_trace_jsonl(
        self,
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_eval_trace_jsonl", None)
        if not callable(exporter):
            return ""
        return str(exporter(trace_history, run_id=run_id, redact=redact) or "")

    def export_training_examples(
        self,
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_training_examples", None)
        if not callable(exporter):
            return []
        return self._make_json_safe(exporter(trace_history, run_id=run_id, redact=redact))

    def export_training_examples_jsonl(
        self,
        *,
        run_id: Optional[str] = None,
        redact: bool = False,
    ) -> str:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_training_examples_jsonl", None)
        if not callable(exporter):
            return ""
        return str(exporter(trace_history, run_id=run_id, redact=redact) or "")

    def export_sft_dataset(
        self,
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        redact: bool = False,
        example_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_sft_dataset", None)
        if not callable(exporter):
            return []
        return self._make_json_safe(
            exporter(
                trace_history,
                run_id=run_id,
                run_ids=run_ids,
                redact=redact,
                example_types=example_types,
            )
        )

    def export_sft_dataset_jsonl(
        self,
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        redact: bool = False,
        example_types: Optional[list[str]] = None,
    ) -> str:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_sft_dataset_jsonl", None)
        if not callable(exporter):
            return ""
        return str(
            exporter(
                trace_history,
                run_id=run_id,
                run_ids=run_ids,
                redact=redact,
                example_types=example_types,
            )
            or ""
        )

    def export_preference_pairs(
        self,
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        chosen_run_ids: Optional[list[str]] = None,
        rejected_run_ids: Optional[list[str]] = None,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_preference_pairs", None)
        if not callable(exporter):
            return []
        return self._make_json_safe(
            exporter(
                trace_history,
                run_id=run_id,
                run_ids=run_ids,
                chosen_run_ids=chosen_run_ids,
                rejected_run_ids=rejected_run_ids,
                redact=redact,
            )
        )

    def export_preference_pairs_jsonl(
        self,
        *,
        run_id: Optional[str] = None,
        run_ids: Optional[list[str]] = None,
        chosen_run_ids: Optional[list[str]] = None,
        rejected_run_ids: Optional[list[str]] = None,
        redact: bool = False,
    ) -> str:
        trace_history_getter = getattr(self, "get_trace_history", None)
        trace_history = trace_history_getter() if callable(trace_history_getter) else []
        exporter = getattr(self.observability_recorder, "export_preference_pairs_jsonl", None)
        if not callable(exporter):
            return ""
        return str(
            exporter(
                trace_history,
                run_id=run_id,
                run_ids=run_ids,
                chosen_run_ids=chosen_run_ids,
                rejected_run_ids=rejected_run_ids,
                redact=redact,
            )
            or ""
        )

    def label_run_outcome(
        self,
        *,
        run_id: Optional[str] = None,
        status: str,
        success: bool,
        failure_stage: Optional[str] = None,
        root_cause_tags: Optional[list[str]] = None,
        changed_files: Optional[list[str]] = None,
        tools_used: Optional[list[str]] = None,
        tests_attempted: Optional[list[str]] = None,
        tests_passed: Optional[list[str]] = None,
        tests_failed: Optional[list[str]] = None,
        user_approval_count: Optional[int] = None,
        user_deny_count: Optional[int] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self._make_json_safe(
            self.observability_recorder.label_run_outcome(
                run_id=run_id,
                status=status,
                success=success,
                failure_stage=failure_stage,
                root_cause_tags=root_cause_tags,
                changed_files=changed_files,
                tools_used=tools_used,
                tests_attempted=tests_attempted,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                user_approval_count=user_approval_count,
                user_deny_count=user_deny_count,
                notes=notes,
                metadata=metadata,
            )
        )

    def clear_observability(self) -> None:
        self.observability_recorder.clear()

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
        self._invalidate_history_usage_anchor()
        self._last_cache_signature = None
        return self.replay_history

    def prepare_replay_history(self, messages: list[Any], provider_name: Optional[str] = None) -> list[Any]:
        target_provider = provider_name or getattr(self.llm, "provider_name", None)
        return self.llm.canonical_to_replay_history(messages, target_provider)

    def _set_history_entries(self, messages: list[Any], *, rebuild_replay: bool = True) -> None:
        canonical_entries: list[Any] = []
        for message in list(messages or []):
            canonical_entries.extend(self.llm.history_entry_to_canonical(message))
        self._history = canonical_entries[-self.config.max_history_length :]
        if not rebuild_replay:
            self._pending_response_usage_anchor = None
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
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'): # type: ignore
                name = tool_call.function.name # type: ignore
                if name and isinstance(name, str):
                    return name

            # Responses API: tool_call.name (flat structure)
            if hasattr(tool_call, 'name'):
                name = tool_call.name # type: ignore
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
            if arguments is None and hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'): # type: ignore
                arguments = tool_call.function.arguments # type: ignore
            # Responses API: tool_call.arguments (flat structure)
            elif arguments is None and hasattr(tool_call, 'arguments'):
                arguments = tool_call.arguments # type: ignore
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

    def _safe_execute_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        *,
        turn_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> ToolResult:
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

        observe_id: Optional[str] = None
        try:
            effective_args, before_audit, tool_spec = self._run_before_tool_use(tool_name, tool_args)
        except HookExecutionError as exc:
            metadata = dict(exc.metadata)
            blocked_result = ToolResult.error(
                str(exc),
                error_type=exc.error_type,
                metadata=metadata,
            )
            observe_id = self._observe_tool_execution_start(
                turn_id=turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                mode=mode,
                stream=stream,
                round_number=round_number,
            )
            self._observe_tool_execution_end(observe_id, result=blocked_result, error=exc)
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=exc)
            return blocked_result

        observe_id = self._observe_tool_execution_start(
            turn_id=turn_id,
            tool_name=tool_name,
            tool_args=effective_args,
            mode=mode,
            stream=stream,
            round_number=round_number,
        )
        self.callback_manager.on_tool_start(tool_name, effective_args)

        try:
            result = self.tool_registry.execute_tool_result(
                tool_name,
                effective_args,
                permission_context=self.permission_context,
                permission_engine=self.permission_engine,
            )
            result = self._run_after_tool_use(
                tool_name,
                effective_args,
                result,
                tool_spec=tool_spec,
                hook_audit=before_audit,
            )
            display_result = result.to_display_string()
            success = result.status == "success"
            self.callback_manager.on_tool_end(
                tool_name,
                display_result,
                success=success,
            )
            self._observe_tool_execution_end(observe_id, result=result)
            return result
            
        except Exception as e:
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=e)
            self._observe_tool_execution_end(observe_id, error=e)
            raise ToolExecutionError(f"工具 '{tool_name}' 执行失败: {e}") from e

    def _safe_execute_tool(self, tool_name: str, tool_args: dict) -> str:
        result = self._safe_execute_tool_result(tool_name, tool_args)
        return result.to_display_string()

    async def _async_safe_execute_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        *,
        turn_id: Optional[str] = None,
        round_number: Optional[int] = None,
        mode: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> ToolResult:
        """
        异步安全执行工具并返回结构化结果。
        
        工具本身是同步的 tool.run()，通过独立线程池执行以避免阻塞事件循环。
        这里避免使用默认线程池，实测在严格 asyncio 测试环境下可能导致关闭阶段挂起。
        """
        if self.tool_registry is None:
            raise ToolExecutionError("工具注册表未配置!")

        observe_id: Optional[str] = None
        try:
            effective_args, before_audit, tool_spec = self._run_before_tool_use(tool_name, tool_args)
        except HookExecutionError as exc:
            metadata = dict(exc.metadata)
            blocked_result = ToolResult.error(
                str(exc),
                error_type=exc.error_type,
                metadata=metadata,
            )
            observe_id = self._observe_tool_execution_start(
                turn_id=turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                mode=mode,
                stream=stream,
                round_number=round_number,
            )
            self._observe_tool_execution_end(observe_id, result=blocked_result, error=exc)
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=exc)
            return blocked_result

        observe_id = self._observe_tool_execution_start(
            turn_id=turn_id,
            tool_name=tool_name,
            tool_args=effective_args,
            mode=mode,
            stream=stream,
            round_number=round_number,
        )
        self.callback_manager.on_tool_start(tool_name, effective_args)

        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor,
                    partial(
                        self.tool_registry.execute_tool_result,
                        tool_name,
                        effective_args,
                        permission_context=self.permission_context,
                        permission_engine=self.permission_engine,
                    ),
                )
            result = self._run_after_tool_use(
                tool_name,
                effective_args,
                result,
                tool_spec=tool_spec,
                hook_audit=before_audit,
            )
            display_result = result.to_display_string()
            success = result.status == "success"
            self.callback_manager.on_tool_end(
                tool_name,
                display_result,
                success=success,
            )
            self._observe_tool_execution_end(observe_id, result=result)
            return result
            
        except Exception as e:
            self.callback_manager.on_tool_end(tool_name, "", success=False, error=e)
            self._observe_tool_execution_end(observe_id, error=e)
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

    # def executeTool(self, tool_name: str, tool_args: dict) -> str:
    #     """向后兼容：请改用 execute_tool"""
    #     return self.execute_tool(tool_name, tool_args)

    # def addTool(self, tool) -> None:
    #     """向后兼容：请改用 add_tool"""
    #     return self.add_tool(tool)

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

    def get_provider_tools(self, provider_name: Optional[str] = None) -> Any:
        if self.tool_registry is None:
            raise ToolRegistryError("工具注册表未配置!")

        target_provider = provider_name or getattr(self.llm, "provider_name", None)
        tool_schema_mode = getattr(self.config, "tool_schema_mode", "full")
        try:
            return self.tool_registry.export_tools(
                target_provider or "openai",
                mode=tool_schema_mode,
            )
        except Exception as e:
            raise ToolRegistryError(f"获取 provider 工具列表失败: {e}") from e

    # def get_openai_tools(self) -> list:
    #     """
    #     获取 OpenAI 格式的工具列表
        
    #     Returns:
    #         OpenAI 格式的工具列表
            
    #     Raises:
    #         ToolRegistryError: 工具注册表未配置
    #     """
    #     if self.tool_registry is None:
    #         raise ToolRegistryError("工具注册表未配置!")
        
    #     try:
    #         return self.tool_registry.export_tools("openai")
    #     except Exception as e:
    #         raise ToolRegistryError(f"获取 OpenAI 工具列表失败: {e}") from e
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
