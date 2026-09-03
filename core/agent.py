"""Small Agent base class and explicit module composition APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
from typing import Any, AsyncIterator, Iterator

from agent.components.conversation_history import ConversationHistory
from agent.components.executor import BaseAgentExecutor
from agent.components.prompt_composer import (
    BaseSystemPromptComposer,
    PromptBuildContext,
    SystemPromptComposer,
)
from context.manager import ContextManager
from core.callbacks import CallbackManager
from core.Config import Config
from core.Exception import (
    AgentStopRequested,
    ExecutionModeError,
    ParameterValidationError,
    SessionNotFoundError,
    SessionSerializationError,
    ToolExecutionError,
    ToolRegistryError,
)
from core.guardrails import build_default_hook_manager
from core.hooks import HookManager
from core.llm import EasyLLM
from core.permissions import PermissionContext, PermissionEngine, PermissionMode, PermissionRule
from core.session import SessionRestoreReport
from metamessage import (
    AgentMetaMessageHistoryPort,
    BaseMetaMessageManager,
    MetaMessage,
    MetaMessageContext,
    MetaMessageLifecycle,
    MetaMessageManager,
)
from plan import BasePlanMode, ExecutionMode, PlanModeConfig, PlanModeManager
from prompt import PromptBlock, SystemPromptTemplate
from runtime import (
    AgentStreamEvent,
    BaseMultiAgentRuntime,
    ExecutionContext,
    MultiAgentRuntime,
    RuntimeEvent,
    RuntimeEventBus,
)
from runtime.subscribers import CallbackEventSubscriber
from skill.manager import SkillManager
from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry


class BaseAgent(ABC):
    """Agent identity, fixed history, and explicit optional-module composition."""

    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: str | None = None,
        description: str | None = None,
        config: Config | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ParameterValidationError("name must be a non-empty string")
        if not isinstance(llm, EasyLLM):
            raise ParameterValidationError(f"llm must be EasyLLM, got {type(llm).__name__}")
        if config is not None and not isinstance(config, Config):
            raise ParameterValidationError(f"config must be Config, got {type(config).__name__}")

        self.name = name.strip()
        self.llm = llm
        self.system_prompt = system_prompt
        self.description = description
        self.config = config or Config.from_env()
        self.reasoning: dict[str, Any] | None = None

        self.history_store = ConversationHistory(llm)
        self.event_bus = RuntimeEventBus()
        self.prompt_composer: BaseSystemPromptComposer = SystemPromptComposer()
        self.skill_manager: SkillManager | None = None
        self.callback_manager = CallbackManager()
        self.hook_manager = build_default_hook_manager()
        self.permission_engine = PermissionEngine()
        self.permission_context = PermissionContext()
        self.metamessage_manager: BaseMetaMessageManager = MetaMessageManager()

        workspace_root = os.path.abspath(self.config.workspace_root or os.getcwd())
        allowed_roots = tuple(os.path.abspath(item) for item in self.config.get_allowed_roots()) or (workspace_root,)
        self.execution_context = ExecutionContext(
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            metadata={"agentId": self.name},
        )

        self.tool_registry: ToolRegistry | None = None
        self.context_manager: ContextManager | None = None
        self.task_service: Any = None
        self.current_task_id: str | None = None
        self.memory_manage: Any = None
        self.plan: BasePlanMode | None = None
        self.observability: Any = None
        self.multi_agent: BaseMultiAgentRuntime | None = None
        self.mcp_managers: list[Any] = []
        self.codeintel_manager: Any = None
        self.worktree_manager: Any = None
        self.executor: BaseAgentExecutor | None = None
        self.interrupt_controller: Any = None

        self._stop_requested = False
        self._stop_reason: str | None = None
        self._closed = False
        self.last_restore_report: SessionRestoreReport | None = None
        self.last_close_report: dict[str, Any] | None = None
        self._callback_subscription = self.event_bus.subscribe(
            CallbackEventSubscriber(self.callback_manager)
        )
        self._metamessage_subscription = self.event_bus.subscribe(
            self._handle_runtime_metamessage
        )
        self._bind_metamessage_manager()

    @property
    def enable_tool(self) -> bool:
        return self.tool_registry is not None and bool(self.tool_registry.get_tool_names())

    @property
    def agent_runtime(self) -> Any:
        return getattr(self.multi_agent, "agent_runtime", None)

    @property
    def team_manager(self) -> Any:
        return getattr(self.multi_agent, "team_manager", None)

    def _metamessage_context(self) -> MetaMessageContext:
        return MetaMessageContext(
            execution_mode=self.get_execution_mode().value,
            permission_mode=self.permission_context.mode.value,
            current_task_id=self.current_task_id,
        )

    def _bind_metamessage_manager(self) -> None:
        self.metamessage_manager.bind(
            history_port=AgentMetaMessageHistoryPort(self.history_store),
            context_provider=self._metamessage_context,
        )

    def _install_metamessage_manager(
        self,
        manager: BaseMetaMessageManager,
    ) -> None:
        """Install runtime infrastructure while rebuilding an Agent."""

        if not isinstance(manager, BaseMetaMessageManager):
            raise TypeError("manager must extend BaseMetaMessageManager")
        self.metamessage_manager = manager
        self._bind_metamessage_manager()
        if self.plan is not None:
            self._bind_plan()

    def _handle_runtime_metamessage(self, event: RuntimeEvent) -> None:
        self.metamessage_manager.publish(event.type.value, event.data)

    def _refresh_execution_context(self) -> None:
        self.execution_context.execution_mode = self.get_execution_mode().value
        self.execution_context.permission_mode = self.permission_context.mode.value
        self.execution_context.current_task_id = self.current_task_id
        self.execution_context.mcp_servers = tuple(
            sorted(
                {
                    str(getattr(manager, "registry_server_name", None) or getattr(manager, "server_label", ""))
                    for manager in self.mcp_managers
                    if getattr(manager, "registry_server_name", None) or getattr(manager, "server_label", None)
                }
            )
        )
        if self.worktree_manager is not None:
            get_active_session = getattr(self.worktree_manager, "get_active_session", None)
            session = get_active_session() if callable(get_active_session) else None
            worktree = getattr(session, "worktree", None)
            self.execution_context.worktree_path = getattr(worktree, "path", None)
            self.execution_context.worktree_branch = getattr(worktree, "branch", None)

    def build_prompt_context(self, query: str = "") -> PromptBuildContext:
        self._refresh_execution_context()
        return PromptBuildContext(
            agent_name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            query=query,
            config=self.config,
            execution_context=self.execution_context,
            tool_registry=self.tool_registry,
            skill_manager=self.skill_manager,
            memory=self.memory_manage,
            task_service=self.task_service,
            plan=self.plan,
        )

    def with_prompt(self, composer: BaseSystemPromptComposer) -> "BaseAgent":
        if not isinstance(composer, BaseSystemPromptComposer):
            raise TypeError("composer must extend BaseSystemPromptComposer")
        self.prompt_composer = composer
        return self

    def with_tool(self, tool_registry: ToolRegistry | None = None) -> "BaseAgent":
        registry = tool_registry or ToolRegistry()
        if not isinstance(registry, ToolRegistry):
            raise TypeError("tool_registry must be ToolRegistry")
        if self.tool_registry is not None and self.tool_registry is not registry:
            raise ToolRegistryError("A different ToolRegistry is already installed")
        self.tool_registry = registry
        self._ensure_deferred_tool_schema_tool()
        if self.task_service is not None:
            self._register_task_tools()
        if self.plan is not None:
            self._bind_plan()
            getattr(self.plan, "install_tools", lambda: None)()
        self._refresh_execution_context()
        return self

    def with_context(self, context_manager: ContextManager) -> "BaseAgent":
        if not isinstance(context_manager, ContextManager):
            raise TypeError("context_manager must be ContextManager")
        self.context_manager = context_manager
        if self.memory_manage is not None:
            self._install_memory_context(self.memory_manage)
        return self

    def with_permissions(
        self,
        engine: PermissionEngine | None = None,
        context: PermissionContext | None = None,
    ) -> "BaseAgent":
        resolved_engine = engine or PermissionEngine()
        resolved_context = context or PermissionContext()
        if not isinstance(resolved_engine, PermissionEngine):
            raise TypeError("engine must be PermissionEngine")
        if not isinstance(resolved_context, PermissionContext):
            raise TypeError("context must be PermissionContext")
        self.permission_engine = resolved_engine
        self.permission_context = resolved_context
        if self.plan is not None:
            self._bind_plan()
        self._refresh_execution_context()
        return self

    def with_hooks(self, hook_manager: HookManager) -> "BaseAgent":
        if not isinstance(hook_manager, HookManager):
            raise TypeError("hook_manager must be HookManager")
        self.hook_manager = hook_manager
        return self

    def with_callbacks(self, callback_manager: CallbackManager) -> "BaseAgent":
        if not isinstance(callback_manager, CallbackManager):
            raise TypeError("callback_manager must be CallbackManager")
        self.event_bus.unsubscribe(self._callback_subscription)
        self.callback_manager = callback_manager
        self._callback_subscription = self.event_bus.subscribe(
            CallbackEventSubscriber(callback_manager)
        )
        return self

    def emit_metamessage(self, message: MetaMessage) -> MetaMessage:
        """Module-facing SPI for runtime context injection."""

        if not isinstance(message, MetaMessage):
            raise TypeError("message must be MetaMessage")
        return self.metamessage_manager.emit(message)

    def with_skill(
        self,
        *directories: str | os.PathLike[str],
        manager: SkillManager | None = None,
    ) -> "BaseAgent":
        """Install directory-based Skills and return this Agent for chaining."""

        if manager is not None and not isinstance(manager, SkillManager):
            raise TypeError("manager must be SkillManager")
        if self.skill_manager is not None and manager not in {None, self.skill_manager}:
            raise ValueError("A different SkillManager is already installed")
        resolved = self.skill_manager or manager or SkillManager()
        if directories:
            resolved.add_directories(directories)
        elif not resolved.skill_names:
            raise ValueError("with_skill requires at least one Skill directory")
        if self.tool_registry is None:
            self.with_tool()
        resolved.bind_agent(self)
        resolved.install_tool()
        self.skill_manager = resolved
        return self

    def with_plan(
        self,
        plan: BasePlanMode | None = None,
        *,
        config: PlanModeConfig | None = None,
    ) -> "BaseAgent":
        if plan is not None and config is not None:
            raise ValueError("plan and config cannot be provided together")
        resolved = plan or PlanModeManager(config)
        if not isinstance(resolved, BasePlanMode):
            raise TypeError("plan must extend BasePlanMode")
        if self.plan is not None and self.plan is not resolved:
            raise ExecutionModeError("A different Plan module is already installed")
        self.plan = resolved
        if getattr(getattr(resolved, "config", None), "register_tools", False) and self.tool_registry is None:
            self.with_tool()
        self._bind_plan()
        getattr(resolved, "install_tools", lambda: None)()
        self._refresh_execution_context()
        return self

    def _bind_plan(self) -> None:
        if self.plan is None:
            return
        self.plan.bind(
            permission_context=self.permission_context,
            metamessage_manager=self.metamessage_manager,
            tool_registry=self.tool_registry,
            runtime_refresher=self._refresh_execution_context,
        )

    def with_task_service(self, task_service: Any) -> "BaseAgent":
        from task import TaskService

        if not isinstance(task_service, TaskService):
            raise TypeError("task_service must be TaskService")
        self.task_service = task_service
        if self.tool_registry is None:
            self.with_tool()
        self._register_task_tools()
        self._refresh_execution_context()
        return self

    def _register_task_tools(self) -> None:
        if self.task_service is None or self.tool_registry is None:
            return
        from Tool.builtin import register_task_tools, register_todo_write_tool

        if not all(self.tool_registry.has_tool(name) for name in ("TaskCreate", "TaskGet", "TaskUpdate", "TaskList")):
            register_task_tools(self.tool_registry, service=self.task_service)
        register_todo_write_tool(
            self.tool_registry,
            service=self.task_service,
            scope_key=f"agent:{self.name}",
            owner=self.name,
        )

    def with_observability(
        self,
        manager: Any = None,
        *,
        path: str | None = None,
        store: Any = None,
    ) -> "BaseAgent":
        from observability import (
            BaseObservabilityManager,
            BaseObservabilityStore,
            ObservabilityManager,
            SQLiteObservabilityStore,
        )

        if self.observability is not None:
            raise ValueError("Observability module is already installed")
        if manager is not None and (path is not None or store is not None):
            raise ValueError("manager cannot be combined with path or store")
        if store is not None and not isinstance(store, BaseObservabilityStore):
            raise TypeError("store must extend BaseObservabilityStore")
        if manager is None:
            if store is None:
                resolved_path = path or os.path.join(
                    self.execution_context.workspace_root,
                    ".easyagent",
                    "observability.sqlite3",
                )
                store = SQLiteObservabilityStore(resolved_path)
            manager = ObservabilityManager(store)
        if not isinstance(manager, BaseObservabilityManager):
            raise TypeError("manager must extend BaseObservabilityManager")
        self.observability = manager.bind(agent_id=self.name, event_bus=self.event_bus)
        return self

    def with_multi_agent(
        self,
        runtime: BaseMultiAgentRuntime | None = None,
        *,
        workspace_root: str | None = None,
        storage_dir: str | None = None,
        max_background_tasks: int | None = None,
    ) -> "BaseAgent":
        if self.multi_agent is not None:
            raise ValueError("MultiAgent module is already installed")
        if self.tool_registry is None:
            self.with_tool()
        resolved = runtime or MultiAgentRuntime(
            workspace_root=workspace_root or self.execution_context.workspace_root,
            storage_dir=storage_dir,
            max_background_tasks=max_background_tasks or self.config.max_background_tasks,
        )
        if not isinstance(resolved, BaseMultiAgentRuntime):
            raise TypeError("runtime must extend BaseMultiAgentRuntime")
        self.multi_agent = resolved.install(self)
        return self

    def with_memory(self, memory_manage: Any) -> "BaseAgent":
        from memory.V2.MemoryManage import MemoryManage

        if not isinstance(memory_manage, MemoryManage):
            raise TypeError("memory_manage must be MemoryManage")
        self.memory_manage = memory_manage
        if self.context_manager is None:
            self.with_context(ContextManager())
        self._install_memory_context(memory_manage)
        if self.tool_registry is None:
            self.with_tool()
        from Tool.builtin.memorytool import register_memory_tools

        memory_tool_names = {
            "add_memory_tool",
            "search_memory_tool",
            "get_memory_tool",
            "update_memory_tool",
            "remove_memory_tool",
            "memory_maintenance_tool",
        }
        if not memory_tool_names.intersection(self.tool_registry.get_tool_names()):
            register_memory_tools(memory_manage, self.tool_registry)
        return self

    def _install_memory_context(self, memory_manage: Any) -> None:
        if self.context_manager is None:
            return
        from context.source.memory_source import MemoryContextSource

        self.context_manager.add_source(MemoryContextSource(memory_manage=memory_manage))

    def with_mcp(
        self,
        manager: Any = None,
        *,
        server_source: Any = None,
        **manager_kwargs: Any,
    ) -> "BaseAgent":
        from Emcp import MCPRuntimeManager
        from Tool.builtin import MCPToolManager

        if manager is not None and server_source is not None:
            raise ValueError("manager and server_source cannot be provided together")
        if manager is None:
            if server_source is None:
                raise ValueError("manager or server_source is required")
            manager = MCPToolManager(server_source=server_source, **manager_kwargs)
        elif manager_kwargs:
            raise ValueError("manager_kwargs can only be used with server_source")
        if not isinstance(manager, MCPRuntimeManager):
            raise TypeError("manager must be MCPRuntimeManager")
        if not isinstance(manager, MCPToolManager):
            raise TypeError("Agent MCP integration requires MCPToolManager")
        if self.tool_registry is None:
            self.with_tool()
        manager.register_to_registry(self.tool_registry)
        self.mcp_managers.append(manager)
        self._refresh_execution_context()
        return self

    def with_codeintel(self, manager: Any = None, *, provider: Any = None) -> "BaseAgent":
        from codeintel import CodeIntelManager, CodeIntelProvider
        from Tool.builtin import register_codeintel_tools

        if manager is not None and provider is not None:
            raise ValueError("manager and provider cannot be provided together")
        if manager is not None and not isinstance(manager, CodeIntelManager):
            raise TypeError("manager must be CodeIntelManager")
        if provider is not None and not isinstance(provider, CodeIntelProvider):
            raise TypeError("provider must extend CodeIntelProvider")
        if self.tool_registry is None:
            self.with_tool()
        self.codeintel_manager = register_codeintel_tools(
            self.tool_registry,
            manager=manager,
            provider=provider,
            parent_agent=self,
            workspace_root=self.execution_context.workspace_root,
            allowed_roots=self.execution_context.allowed_roots,
        )
        return self

    def with_worktree(self, manager: Any = None) -> "BaseAgent":
        from Tool.builtin import register_worktree_tools
        from Tool.runtime import WorktreeManager

        if manager is None:
            repo_root = WorktreeManager.detect_repo_root(
                self.execution_context.workspace_root,
                git_binary=self.config.git_binary,
            )
            manager = WorktreeManager(
                repo_root,
                git_binary=self.config.git_binary,
                original_cwd=self.execution_context.workspace_root,
            )
        if not isinstance(manager, WorktreeManager):
            raise TypeError("manager must be WorktreeManager")
        if self.tool_registry is None:
            self.with_tool()
        register_worktree_tools(self.tool_registry, worktree_manager=manager)
        self.worktree_manager = manager
        self._refresh_execution_context()
        return self

    def with_executor(self, executor: BaseAgentExecutor) -> "BaseAgent":
        if not isinstance(executor, BaseAgentExecutor):
            raise TypeError("executor must extend BaseAgentExecutor")
        self.executor = executor
        return self

    def with_interruptions(self, controller: Any) -> "BaseAgent":
        from agent.components.tool_interrupt_controller import BaseToolInterruptController

        if not isinstance(controller, BaseToolInterruptController):
            raise TypeError("controller must extend BaseToolInterruptController")
        self.interrupt_controller = controller
        return self

    def get_pending_interruption(self) -> dict[str, Any] | None:
        if self.interrupt_controller is None:
            return None
        return self.interrupt_controller.get_last_interrupt()

    def resolve_pending_interruption(
        self,
        *,
        content: str,
        ephemeral_context: Any = None,
    ) -> dict[str, Any]:
        payload = self.get_pending_interruption()
        if payload is None:
            raise ToolExecutionError("No pending tool interruption")
        tool_name = str(payload.get("tool_name") or "").strip()
        tool_id = str(payload.get("tool_id") or "").strip()
        if not tool_name or not tool_id:
            raise ToolExecutionError("Pending interruption is missing tool_name or tool_id")
        text = str(content)
        self.history_store.append_tool_result(text, tool_id, tool_name)
        if ephemeral_context is not None:
            rendered = (
                ephemeral_context
                if isinstance(ephemeral_context, str)
                else json.dumps(ephemeral_context, ensure_ascii=False, default=str, indent=2)
            )
            self.metamessage_manager.emit(
                MetaMessage(
                    name=f"tool_context:{tool_name}",
                    content=f"Runtime context produced by tool `{tool_name}`:\n{rendered}",
                    lifecycle=MetaMessageLifecycle.INVOCATION,
                    metadata={"source": "tool_result", "toolName": tool_name},
                )
            )
        self.interrupt_controller.clear_last_interrupt()
        return {
            "toolName": tool_name,
            "toolCallId": tool_id,
            "status": "resolved",
        }

    def _ensure_deferred_tool_schema_tool(self) -> None:
        if self.tool_registry is None or self.config.tool_schema_mode != "deferred":
            return
        if not self.tool_registry.has_tool("tool_schema_tool"):
            from Tool.builtin import register_tool_schema_tool

            register_tool_schema_tool(self.tool_registry)

    def add_tool(self, tool: Tool) -> "BaseAgent":
        if not isinstance(tool, Tool):
            raise TypeError("tool must extend Tool")
        if self.tool_registry is None:
            self.with_tool()
        self.tool_registry.register_tool(tool)
        return self

    def execute_tool_result(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.tool_registry is None:
            raise ToolRegistryError("Tool module is not installed")
        return self.tool_registry.execute_tool_result(
            tool_name,
            arguments,
            permission_context=self.permission_context,
            permission_engine=self.permission_engine,
        )

    def get_provider_tools(self) -> Any:
        if self.tool_registry is None:
            return None
        return self.tool_registry.export_tools(
            self.llm.provider_name or "openai",
            mode=self.config.tool_schema_mode,
        )

    def get_system_prompt_blocks(self, query: str = "") -> list[PromptBlock]:
        return self.prompt_composer.compose(self.build_prompt_context(query))

    def get_system_prompt_template(self, query: str = "") -> SystemPromptTemplate:
        return SystemPromptTemplate(self.get_system_prompt_blocks(query))

    def get_enhanced_prompt(self, query: str = "") -> str:
        return self.get_system_prompt_template(query).render_system()

    def set_permission_mode(self, mode: PermissionMode | str) -> None:
        resolved = PermissionMode(mode)
        if resolved == PermissionMode.PLAN:
            if self.plan is None:
                self.with_plan()
            self.plan.enter()
        elif self.plan is not None and self.get_execution_mode() == ExecutionMode.PLAN:
            self.plan.exit(permission_mode=resolved)
        else:
            self.permission_context.set_mode(resolved)
        self._refresh_execution_context()

    def add_permission_rule(
        self,
        rule: PermissionRule,
        *,
        source: str | None = None,
        priority: int | None = None,
    ) -> None:
        self.permission_context.add_rule(rule, source=source, priority=priority)

    def enter_plan_mode(self, *, allowed_actions: list[str] | None = None) -> None:
        if self.plan is None:
            raise ExecutionModeError("Plan module is not installed")
        self.plan.enter(allowed_actions=allowed_actions)

    def request_exit_plan_mode(self, *, allowed_actions: list[str] | None = None) -> None:
        if self.plan is None or not hasattr(self.plan, "request_exit"):
            raise ExecutionModeError("Plan module is not installed")
        self.plan.request_exit(allowed_actions=allowed_actions)

    def exit_plan_mode(self, *, permission_mode: PermissionMode | str = PermissionMode.DEFAULT) -> None:
        if self.plan is None:
            raise ExecutionModeError("Plan module is not installed")
        self.plan.exit(permission_mode=permission_mode)

    def get_execution_mode(self) -> ExecutionMode:
        return self.plan.mode if self.plan is not None else ExecutionMode.EXECUTE

    def set_current_task(self, task_id: str | None) -> None:
        self.current_task_id = task_id
        self._refresh_execution_context()

    def request_stop(self, reason: str = "") -> None:
        self._stop_requested = True
        self._stop_reason = reason.strip() or "Agent stop requested"

    def clear_stop_request(self) -> None:
        self._stop_requested = False
        self._stop_reason = None

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    def get_stop_reason(self) -> str | None:
        return self._stop_reason

    def stop_reason_if_requested(self) -> str | None:
        return self._stop_reason if self._stop_requested else None

    def add_message(self, message: Any) -> None:
        self.history_store.add(message)

    def add_messages(self, messages: list[Any]) -> None:
        self.history_store.add_many(messages)

    def add_user_message(self, content: str) -> None:
        self.history_store.append_query(content)

    def add_assistant_message(self, content: str) -> None:
        self.history_store.append_assistant(content=content)

    def clear_history(self) -> None:
        self.history_store.clear()

    @property
    def history(self) -> list[Any]:
        return self.history_store.canonical

    @history.setter
    def history(self, messages: list[Any]) -> None:
        self.history_store.replace(messages)

    @property
    def replay_history(self) -> list[Any]:
        return self.history_store.replay

    def get_history(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.history_store.canonical]

    def get_raw_history(self) -> list[Any]:
        return self.history_store.replay

    def get_canonical_history(self) -> list[Any]:
        return self.history_store.canonical

    def get_history_length(self) -> int:
        return len(self.history_store)

    def rebuild_replay_history(self) -> list[Any]:
        return self.history_store.rebuild_replay()

    def change_model(self, llm: EasyLLM) -> None:
        if not isinstance(llm, EasyLLM):
            raise TypeError("llm must be EasyLLM")
        self.llm = llm
        self.history_store.change_llm(llm)

    def get_trace_history(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.event_bus.history()]

    @property
    def trace_history(self) -> list[dict[str, Any]]:
        return self.get_trace_history()

    def clear_trace_history(self) -> None:
        self.event_bus.clear_history()

    def get_context_usage(self) -> dict[str, Any]:
        blocks = self.get_system_prompt_blocks()
        system_prompt = SystemPromptTemplate(blocks).render_system()
        tools = self.get_provider_tools()
        counter = (
            self.context_manager.counter
            if self.context_manager is not None
            else None
        )
        if counter is None:
            from context.token.counter import TokenCounter

            counter = TokenCounter()
        estimated = self.llm.count_request_tokens(
            counter,
            self.replay_history,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=self.reasoning,
        )
        return {
            "estimatedRequestTokens": estimated,
            "canonicalMessages": len(self.history_store),
            "replayMessages": len(self.replay_history),
            "provider": self.llm.provider_name,
        }

    def _build_session_snapshot(self) -> dict[str, Any]:
        def module_state(module: Any, state: Any = None) -> dict[str, Any]:
            return {
                "implementation": module.__class__.__name__,
                "state": state,
            }

        modules: dict[str, Any] = {
            "permissions": module_state(
                self.permission_context,
                self.permission_context.export_state(),
            ),
            "metamessage": module_state(
                self.metamessage_manager,
                getattr(self.metamessage_manager, "export_state", lambda: {})(),
            ),
            "executionContext": self.execution_context.to_dict(),
            "prompt": module_state(
                self.prompt_composer,
                getattr(self.prompt_composer, "export_state", lambda: None)(),
            ),
            "executor": module_state(self.executor),
            "runtimeEvents": module_state(
                self.event_bus,
                {"events": self.get_trace_history()},
            ),
        }
        if self.skill_manager is not None:
            modules["skills"] = module_state(
                self.skill_manager,
                self.skill_manager.export_state(),
            )
        if self.interrupt_controller is not None:
            modules["interruptions"] = module_state(
                self.interrupt_controller,
                self.interrupt_controller.export_state(),
            )
        if self.tool_registry is not None:
            modules["tools"] = module_state(
                self.tool_registry,
                {"names": self.tool_registry.get_tool_names()},
            )
        if self.context_manager is not None:
            modules["context"] = module_state(self.context_manager)
        if self.task_service is not None:
            modules["tasks"] = module_state(self.task_service)
        if self.memory_manage is not None:
            modules["memory"] = module_state(self.memory_manage)
        if self.plan is not None and hasattr(self.plan, "export_state"):
            modules["plan"] = module_state(self.plan, self.plan.export_state())
        if self.observability is not None and hasattr(self.observability, "export_state"):
            modules["observability"] = module_state(
                self.observability,
                self.observability.export_state(),
            )
        if self.multi_agent is not None and hasattr(self.multi_agent, "export_state"):
            modules["multiAgent"] = module_state(
                self.multi_agent,
                self.multi_agent.export_state(),
            )
        if self.mcp_managers:
            modules["mcp"] = [
                {
                    "implementation": manager.__class__.__name__,
                    "identity": str(
                        getattr(manager, "registry_server_name", None)
                        or getattr(manager, "server_label", index)
                    ),
                    "state": manager.export_state(),
                }
                for index, manager in enumerate(self.mcp_managers)
            ]
        if self.codeintel_manager is not None:
            modules["codeintel"] = module_state(
                self.codeintel_manager,
                self.codeintel_manager.export_state(),
            )
        if self.worktree_manager is not None:
            modules["worktree"] = module_state(
                self.worktree_manager,
                self.worktree_manager.export_state(),
            )
        return {
            "schemaVersion": 3,
            "agentType": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "systemPrompt": self.system_prompt,
            "config": self.config.model_dump(mode="python"),
            "reasoning": self.reasoning,
            "history": self.history_store.export_state(),
            "currentTaskId": self.current_task_id,
            "modules": modules,
        }

    @staticmethod
    def _resolve_session_store(store: Any = None):
        from db import SessionStore

        if store is None:
            return SessionStore()
        if isinstance(store, SessionStore):
            return store
        if isinstance(store, str):
            return SessionStore(store)
        raise SessionSerializationError(f"Unsupported session store: {type(store).__name__}")

    def save_session(
        self,
        session_id: str,
        *,
        store: Any = None,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise SessionSerializationError("session_id must be a non-empty string")
        session_store = self._resolve_session_store(store)
        session_store.create_or_update_session(
            session_id=session_id,
            agent_type=self.__class__.__name__,
            agent_name=self.name,
            snapshot=self._build_session_snapshot(),
            metadata=dict(metadata or {}),
            expires_at=expires_at,
        )
        return session_id

    @classmethod
    def load_session(
        cls,
        session_id: str,
        *,
        llm: EasyLLM,
        store: Any = None,
        tool_registry: ToolRegistry | None = None,
        context_manager: ContextManager | None = None,
        callback_manager: CallbackManager | None = None,
        skill_manager: SkillManager | None = None,
        permission_engine: PermissionEngine | None = None,
        permission_context: PermissionContext | None = None,
        hook_manager: HookManager | None = None,
        task_service: Any = None,
        plan: BasePlanMode | None = None,
        prompt_composer: BaseSystemPromptComposer | None = None,
        metamessage_manager: BaseMetaMessageManager | None = None,
        observability_manager: Any = None,
        multi_agent_runtime: BaseMultiAgentRuntime | None = None,
        mcp_managers: list[Any] | None = None,
        codeintel_manager: Any = None,
        worktree_manager: Any = None,
        memory_manage: Any = None,
        executor: BaseAgentExecutor | None = None,
        interruption_controller: Any = None,
    ) -> "BaseAgent":
        session_store = cls._resolve_session_store(store)
        record = session_store.get_session(session_id)
        if record is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        snapshot = dict(record.get("snapshot") or {})
        if int(snapshot.get("schemaVersion") or 0) != 3:
            raise SessionSerializationError("Only SessionSnapshotV3 is supported")
        target_cls = cls
        if cls is BaseAgent:
            if snapshot.get("agentType") != "BasicAgent":
                raise SessionSerializationError(f"Unsupported Agent type: {snapshot.get('agentType')}")
            from agent import BasicAgent

            target_cls = BasicAgent
        agent = target_cls(
            name=str(snapshot.get("name") or record.get("agent_name") or "agent"),
            llm=llm,
            system_prompt=snapshot.get("systemPrompt"),
            description=snapshot.get("description"),
            config=Config.model_validate(snapshot.get("config") or {}),
        )
        report = SessionRestoreReport(session_id=session_id, agent_type=target_cls.__name__)
        modules = dict(snapshot.get("modules") or {})
        saved_provider = str(dict(snapshot.get("history") or {}).get("providerName") or "")
        current_provider = str(llm.provider_name or "")
        if saved_provider and current_provider and saved_provider != current_provider:
            report.add_issue(
                component="history",
                code="provider_drift",
                message=(
                    f"Session history was recorded for {saved_provider}; replay history was rebuilt "
                    f"for {current_provider} from canonical messages"
                ),
                metadata={
                    "savedProvider": saved_provider,
                    "currentProvider": current_provider,
                },
            )

        def state_of(name: str) -> Any:
            value = modules.get(name)
            if not isinstance(value, dict):
                return None
            return value.get("state")

        def implementation_of(name: str) -> str:
            value = modules.get(name)
            if not isinstance(value, dict):
                return ""
            return str(value.get("implementation") or "")

        def note_explicit_module(name: str, parameter: str) -> None:
            report.add_issue(
                component=name,
                code=f"{name}_implementation_missing",
                message=(
                    f"Session used {implementation_of(name) or name}; supply `{parameter}` "
                    "to restore that module implementation"
                ),
            )

        supplied_prompt = prompt_composer is not None
        if supplied_prompt:
            agent.with_prompt(prompt_composer)
        elif implementation_of("prompt") not in {"", "SystemPromptComposer"}:
            note_explicit_module("prompt", "prompt_composer")
        if (supplied_prompt or implementation_of("prompt") in {"", "SystemPromptComposer"}) and hasattr(
            agent.prompt_composer, "restore_state"
        ):
            agent.prompt_composer.restore_state(state_of("prompt"))
        if callback_manager is not None:
            agent.with_callbacks(callback_manager)
        if hook_manager is not None:
            agent.with_hooks(hook_manager)
        restored_permission_context = permission_context or PermissionContext()
        restored_permission_context.restore_state(state_of("permissions"))
        agent.with_permissions(permission_engine, restored_permission_context)
        supplied_metamessage = metamessage_manager is not None
        if supplied_metamessage:
            agent._install_metamessage_manager(metamessage_manager)
        elif implementation_of("metamessage") not in {"", "MetaMessageManager"}:
            note_explicit_module("metamessage", "metamessage_manager")
        if (
            supplied_metamessage
            or implementation_of("metamessage") in {"", "MetaMessageManager"}
        ) and hasattr(agent.metamessage_manager, "restore_state"):
            agent.metamessage_manager.restore_state(state_of("metamessage"))
        tool_names = list(dict(state_of("tools") or {}).get("names") or [])
        if tool_registry is not None:
            agent.with_tool(tool_registry)
        if context_manager is not None:
            agent.with_context(context_manager)
        elif modules.get("context"):
            report.add_issue(component="context", code="context_manager_missing", message="ContextManager must be supplied explicitly")
        saved_skill_state = dict(state_of("skills") or {})
        saved_skill_directories = [
            str(path) for path in list(saved_skill_state.get("directories") or []) if path
        ]
        saved_skill_names = [
            str(item.get("name"))
            for item in list(saved_skill_state.get("skills") or [])
            if isinstance(item, dict) and item.get("name")
        ]
        if skill_manager is not None or saved_skill_directories:
            try:
                agent.with_skill(*saved_skill_directories, manager=skill_manager)
                report.note_missing_skills(
                    [
                        name
                        for name in saved_skill_names
                        if agent.skill_manager is None or not agent.skill_manager.has_skill(name)
                    ]
                )
                if agent.skill_manager is not None:
                    report.extend_component(
                        "skills",
                        agent.skill_manager.restore_state(saved_skill_state),
                    )
            except Exception as exc:
                report.add_issue(
                    component="skills",
                    code="skill_directories_unavailable",
                    message=str(exc),
                    metadata={"directories": saved_skill_directories},
                )
        if task_service is not None:
            agent.with_task_service(task_service)
        elif modules.get("tasks"):
            note_explicit_module("tasks", "task_service")
        if memory_manage is not None:
            agent.with_memory(memory_manage)
        elif modules.get("memory"):
            note_explicit_module("memory", "memory_manage")
        if plan is not None:
            plan_state = state_of("plan")
            if hasattr(plan, "restore_state"):
                plan.restore_state(plan_state)
            agent.with_plan(plan)
        elif modules.get("plan") and implementation_of("plan") == "PlanModeManager":
            restored_plan = PlanModeManager()
            restored_plan.restore_state(state_of("plan"))
            agent.with_plan(restored_plan)
        elif modules.get("plan"):
            note_explicit_module("plan", "plan")
        if observability_manager is not None:
            agent.with_observability(observability_manager)
        elif modules.get("observability"):
            if implementation_of("observability") == "ObservabilityManager":
                from observability import InMemoryObservabilityStore, ObservabilityManager

                agent.with_observability(ObservabilityManager(InMemoryObservabilityStore()))
            else:
                note_explicit_module("observability", "observability_manager")
        if agent.observability is not None and state_of("observability") is not None:
            agent.observability.restore_state(state_of("observability"))
        if worktree_manager is not None:
            agent.with_worktree(worktree_manager)
            if modules.get("worktree"):
                report.extend_component("worktree", worktree_manager.restore_state(state_of("worktree")))
        elif modules.get("worktree"):
            report.add_issue(component="worktree", code="worktree_manager_missing", message="WorktreeManager must be supplied explicitly")
        if multi_agent_runtime is not None:
            agent.with_multi_agent(multi_agent_runtime)
            if modules.get("multiAgent") and hasattr(agent.multi_agent, "restore_state"):
                report.extend_component("multi_agent", agent.multi_agent.restore_state(state_of("multiAgent")))
        elif modules.get("multiAgent"):
            report.add_issue(component="multi_agent", code="runtime_missing", message="MultiAgent runtime must be supplied explicitly")

        saved_mcp = list(modules.get("mcp") or [])
        unused_mcp = set(range(len(saved_mcp)))
        for manager_index, manager in enumerate(list(mcp_managers or [])):
            identities = {
                str(getattr(manager, "registry_server_name", "") or ""),
                str(getattr(manager, "server_label", "") or ""),
                str(getattr(manager, "source_identifier", "") or ""),
            }
            matched_index = next(
                (
                    index
                    for index in unused_mcp
                    if str(dict(saved_mcp[index] or {}).get("identity") or "") in identities
                    or str(dict(dict(saved_mcp[index] or {}).get("state") or {}).get("sourceIdentifier") or "") in identities
                ),
                manager_index if manager_index in unused_mcp else None,
            )
            if matched_index is not None:
                saved = dict(saved_mcp[matched_index] or {})
                unused_mcp.discard(matched_index)
                restore = getattr(manager, "restore_state", None)
                if callable(restore):
                    identity = str(saved.get("identity") or manager_index)
                    report.extend_component(f"mcp:{identity}", restore(saved.get("state")))
            agent.with_mcp(manager)
        if saved_mcp and not mcp_managers:
            report.add_issue(component="mcp", code="mcp_managers_missing", message="MCP managers must be supplied explicitly")
        elif unused_mcp:
            report.add_issue(
                component="mcp",
                code="mcp_managers_incomplete",
                message="Not every MCP server in the session had a supplied manager",
                metadata={
                    "servers": [
                        str(dict(saved_mcp[index] or {}).get("identity") or index)
                        for index in sorted(unused_mcp)
                    ]
                },
            )
        if codeintel_manager is not None:
            agent.with_codeintel(codeintel_manager)
            if modules.get("codeintel"):
                report.extend_component("codeintel", codeintel_manager.restore_state(state_of("codeintel")))
        elif modules.get("codeintel") and implementation_of("codeintel") == "CodeIntelManager":
            from codeintel import CodeIntelManager

            restored_codeintel = CodeIntelManager.from_state(state_of("codeintel"), parent_agent=agent)
            agent.with_codeintel(restored_codeintel)
            report.extend_component(
                "codeintel",
                restored_codeintel.restore_state(state_of("codeintel")),
            )
        elif modules.get("codeintel"):
            note_explicit_module("codeintel", "codeintel_manager")
        if executor is not None:
            agent.with_executor(executor)
        elif implementation_of("executor") not in {"", "DefaultAgentExecutor"}:
            note_explicit_module("executor", "executor")
        if interruption_controller is not None:
            agent.with_interruptions(interruption_controller)
        elif modules.get("interruptions") and implementation_of("interruptions") != "InMemoryToolInterruptController":
            note_explicit_module("interruptions", "interruption_controller")
        if (
            agent.interrupt_controller is not None
            and state_of("interruptions") is not None
            and (
                interruption_controller is not None
                or implementation_of("interruptions") == "InMemoryToolInterruptController"
            )
        ):
            agent.interrupt_controller.restore_state(state_of("interruptions"))

        if tool_names:
            missing_tools = (
                list(tool_names)
                if agent.tool_registry is None
                else [name for name in tool_names if not agent.tool_registry.has_tool(name)]
            )
            report.note_missing_tools(missing_tools)

        execution_context = ExecutionContext.from_dict(modules.get("executionContext"))
        if execution_context is not None:
            agent.execution_context = execution_context
            report.execution_context_restored = True
        agent.current_task_id = snapshot.get("currentTaskId")
        agent.reasoning = snapshot.get("reasoning")
        agent.history_store.restore_state(snapshot.get("history"))
        runtime_event_state = dict(state_of("runtimeEvents") or {})
        agent.event_bus.restore_history(list(runtime_event_state.get("events") or []))
        reconcile = getattr(agent.metamessage_manager, "reconcile_history", None)
        if callable(reconcile):
            removed = int(reconcile() or 0)
            if removed:
                report.add_issue(
                    component="metamessage",
                    code="stale_injections_removed",
                    message="MetaMessage injection records missing from restored history were removed",
                    metadata={"count": removed},
                )
        agent._refresh_execution_context()
        hook_result = agent.hook_manager.after_session_restore(
            {
                "session_id": session_id,
                "restore_report": report,
                "snapshot": snapshot,
            }
        )
        if hook_result.blocked:
            report.add_issue(component="hooks", code=hook_result.error_type, message=hook_result.message, severity="error")
        resolved_report = hook_result.payload.get("restore_report", report)
        if not isinstance(resolved_report, SessionRestoreReport):
            raise TypeError("after_session_restore hook must keep restore_report as SessionRestoreReport")
        report = resolved_report
        agent.last_restore_report = report
        return agent

    @classmethod
    def list_sessions(cls, *, store: Any = None, limit: int = 100, include_expired: bool = False) -> list[dict[str, Any]]:
        return cls._resolve_session_store(store).list_sessions(limit=limit, include_expired=include_expired)

    @classmethod
    def delete_session(cls, session_id: str, *, store: Any = None) -> bool:
        return cls._resolve_session_store(store).delete_session(session_id)

    @classmethod
    def cleanup_expired_sessions(cls, *, store: Any = None, now: datetime | None = None) -> int:
        return cls._resolve_session_store(store).cleanup_expired_sessions(now=now)

    def get_last_restore_report(self) -> dict[str, Any] | None:
        return self.last_restore_report.to_dict() if self.last_restore_report is not None else None

    def close(
        self,
        *,
        worktree_action: str = "keep",
        discard_worktree_changes: bool = False,
        close_llm: bool = True,
    ) -> dict[str, Any]:
        if self._closed:
            return dict(self.last_close_report or {"status": "closed", "components": {}})
        report: dict[str, Any] = {"status": "closed", "components": {}, "issues": []}

        def close_component(name: str, target: Any, **kwargs: Any) -> None:
            if target is None:
                return
            try:
                value = target.close(**kwargs)
                report["components"][name] = value if isinstance(value, dict) else {"status": "closed"}
            except Exception as exc:
                report["status"] = "degraded"
                report["issues"].append({"component": name, "error": str(exc)})

        close_component("multiAgent", self.multi_agent)
        for index, manager in enumerate(reversed(self.mcp_managers)):
            close_component(f"mcp[{index}]", manager)
        close_component("codeintel", self.codeintel_manager)
        close_component(
            "worktree",
            self.worktree_manager,
            action=worktree_action,
            discard_changes=discard_worktree_changes,
        )
        close_component("observability", self.observability)
        close_component("skills", self.skill_manager)
        if close_llm:
            close_component("llm", self.llm)
        self._closed = True
        self.last_close_report = report
        return dict(report)

    def get_last_close_report(self) -> dict[str, Any] | None:
        return dict(self.last_close_report) if self.last_close_report is not None else None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tools={self.enable_tool})"

    @abstractmethod
    def invoke(self, query: str, max_iter: int = 10, temperature: float | None = None, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    async def ainvoke(self, query: str, max_iter: int = 10, temperature: float | None = None, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, query: str, max_iter: int = 10, temperature: float | None = None, **kwargs: Any) -> Iterator[AgentStreamEvent]:
        raise NotImplementedError

    @abstractmethod
    def astream(self, query: str, max_iter: int = 10, temperature: float | None = None, **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
        raise NotImplementedError


__all__ = ["BaseAgent"]
