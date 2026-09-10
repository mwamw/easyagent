"""Small Agent base class and explicit module composition APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
from uuid import uuid4
from threading import Lock
from typing import Any, AsyncIterator, Iterator

from agent.components.conversation_history import ConversationHistory
from agent.components.executor import BaseAgentExecutor
from agent.components.prompt_composer import (
    BaseSystemPromptComposer,
    PromptBuildContext,
    SystemPromptComposer,
)
from context.manager import ContextManager
from context.base import BaseContextManager
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
        from context.usage import ContextUsageTracker
        self._context_usage = ContextUsageTracker()
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
        self.memory: Any = None
        self.session_id = f"session_{uuid4().hex}"
        self._memory_tools: list[Any] = []
        self._memory_hooks: list[Any] = []
        self._memory_emitter = None
        self._owns_memory = False
        self._invocation_lock = Lock()
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

        from core.session.persistence import SessionPersistence
        self._sessions = SessionPersistence(self)
        self._owns_session_store = False
        for name, attribute, forkable in (
            ("permissions", "permission_context", True), ("metamessage", "metamessage_manager", True),
            ("skills", "skill_manager", True), ("context", "context_manager", True),
            ("memory", "memory", True), ("plan", "plan", True),
            ("tasks", "task_service", False), ("interruptions", "interrupt_controller", False),
            ("runtimeEvents", "event_bus", False), ("observability", "observability", False),
            ("multiAgent", "multi_agent", False), ("codeintel", "codeintel_manager", False),
            ("worktree", "worktree_manager", False),
        ):
            self.register_session_state(name, lambda attribute=attribute: getattr(self, attribute), forkable=forkable)

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
            memory=self.memory,
            task_service=self.task_service,
            plan=self.plan,
            context_manager=self.context_manager,
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

    def with_context(self, context_manager: BaseContextManager) -> "BaseAgent":
        if not isinstance(context_manager, BaseContextManager):
            raise TypeError("context_manager must implement BaseContextManager")
        self.context_manager = context_manager
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
        if hook_manager is not self.hook_manager:
            for hook in self._memory_hooks:
                self.hook_manager.remove_hook(hook)
                hook_manager.add_hook(hook)
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

    def with_memory(self, memory: Any, *, owned: bool = False, replace: bool = False) -> "BaseAgent":
        from core.hooks import BaseHook
        from memory.base import BaseMemory, MemoryServices
        from core.scope import RuntimeScope
        from metamessage.emitter import MetaMessageEmitter

        if not isinstance(memory, BaseMemory):
            raise TypeError("memory must implement the BaseMemory module contract")
        if self._closed or self._invocation_lock.locked():
            raise RuntimeError("Memory can only be installed on an idle, open Agent")
        if self.memory is memory:
            return self
        if self.memory is not None and not replace:
            raise ValueError("Memory is already installed; pass replace=True explicitly")
        emitter = MetaMessageEmitter(self.metamessage_manager)
        memory.bind(MemoryServices(emitter, lambda: RuntimeScope(
            session_id=self.session_id,
            workspace_root=self.execution_context.workspace_root,
            user_id=getattr(memory, "user_id", None),
        )))
        try:
            tools = list(memory.get_tools())
            hooks = list(memory.get_hooks())
            if any(not isinstance(tool, Tool) for tool in tools):
                raise TypeError("Memory tools must be Tool instances")
            if any(not isinstance(hook, BaseHook) for hook in hooks):
                raise TypeError("Memory hooks must be BaseHook instances")
            if len({tool.name for tool in tools}) != len(tools):
                raise ToolRegistryError("Memory provided duplicate tool names")
            if tools and self.tool_registry is None:
                self.with_tool()
            for tool in tools:
                existing = self.tool_registry.get_tool(tool.name)
                if existing is not None and not any(existing is old for old in self._memory_tools):
                    raise ToolRegistryError(f"Memory tool conflicts with existing tool: {tool.name}")
        except BaseException:
            emitter.close()
            memory.unbind()
            raise
        try:
            self._detach_memory()
        except BaseException:
            emitter.close()
            memory.unbind()
            raise
        self.memory = memory
        self._memory_emitter = emitter
        self._owns_memory = owned
        self._memory_tools = tools
        self._memory_hooks = hooks
        for tool in tools:
            self.tool_registry.register_tool(tool, conflict_policy="error", expose_in_deferred=True)
        self.hook_manager.extend(hooks)
        return self

    def _detach_memory(self) -> None:
        memory = self.memory
        for tool in self._memory_tools:
            if self.tool_registry is not None and self.tool_registry.get_tool(tool.name) is tool:
                self.tool_registry.unregister_tool(tool.name)
        for hook in self._memory_hooks:
            self.hook_manager.remove_hook(hook)
        if self._memory_emitter is not None:
            self._memory_emitter.close()
        self._memory_tools = []
        self._memory_hooks = []
        self._memory_emitter = None
        self.memory = None
        if memory is not None:
            try:
                if self._owns_memory:
                    memory.close()
            finally:
                memory.unbind()
        self._owns_memory = False

    def with_mcp(
        self,
        manager: Any = None,
        *,
        server_source: Any = None,
        hub: Any = None,
        server_name: str | None = None,
        legacy_resource_tools: bool | None = None,
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
        registration = {}
        if hub is not None:
            registration["hub"] = hub
        if server_name is not None:
            registration["server_name"] = server_name
        if legacy_resource_tools is not None:
            registration["legacy_resource_tools"] = legacy_resource_tools
        manager.register_to_registry(self.tool_registry, **registration)
        self.mcp_managers.append(manager)
        identity = str(getattr(manager, "registry_server_name", None) or getattr(manager, "server_label", len(self.mcp_managers)))
        self.register_session_state(f"mcp:{identity}", manager, forkable=False)
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
                    content=f"<system-reminder>\nRuntime context produced by tool `{tool_name}`:\n{rendered}\n</system-reminder>",
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

    @property
    def context_usage_revision(self):
        return (self.history_store.revision, self._context_usage.revision,
                id(self.llm), self.llm.model, id(self.context_manager),
                id(self.prompt_composer), self.system_prompt, self.session_id)

    def get_context_usage(self) -> dict[str, Any]:
        from context.builder import ContextBuilder
        from core.request_compiler import compile_prompt_blocks

        tracker = self._context_usage
        manager = self.context_manager or ContextManager(builder=ContextBuilder(counter=tracker.counter))
        compiled = compile_prompt_blocks(
            self.get_system_prompt_blocks(tracker.last_query), cache_policy=self.config.cache_policy,
        )
        tools = self.get_provider_tools() or None
        request = manager.build_request_input(
            query="", include_query=False, replay_history=self.replay_history,
            provider_name=self.llm.provider_name, system_prompt=compiled.system_prompt,
            system_prompt_blocks=compiled.system_prompt_blocks,
            system_reminder_blocks=compiled.system_reminder_blocks,
            dynamic_tail_blocks=compiled.dynamic_tail_blocks,
            on_demand_expansion_blocks=compiled.on_demand_expansion_blocks,
            cache_policy=compiled.cache_policy, tools=tools, reasoning=self.reasoning,
        )
        estimate = manager.estimate_request(
            request, usage_tracker=tracker, llm=self.llm,
            history_epoch=self.history_store.context_epoch,
            tools=tools, reasoning=self.reasoning, options=tracker.last_options,
        )
        limit = manager.budget.max_tokens
        return {
            "estimatedRequestTokens": estimate.tokens,
            "localEstimatedRequestTokens": estimate.local_tokens,
            "request_estimate_source": estimate.source,
            "requestEstimate": {
                "estimatedRequestTokens": estimate.tokens, "source": estimate.source,
                "localEstimatedRequestTokens": estimate.local_tokens, "metadata": estimate.metadata,
            },
            "maxTokens": limit,
            "budget": {"maxTokens": limit, "remainingTokens": None if limit is None else max(0, limit - estimate.tokens)},
            "canonicalMessages": len(self.history_store),
            "replayMessages": len(self.replay_history),
            "provider": self.llm.provider_name,
        }

    def register_session_state(self, name, component, *, version=1, required=False, forkable=True):
        from core.session.persistence import SessionStateParticipant
        self._sessions.register(SessionStateParticipant(name, component, version, required, forkable))
        return self

    def with_session(self, store, *, session_id=None, owned=False):
        if self._invocation_lock.locked() or self._closed:
            raise RuntimeError("Session bindings can only change on an idle, open Agent")
        self._sessions.store = self._sessions.resolve_store(store)
        self._owns_session_store = owned
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("session_id must be non-empty")
            self.session_id = session_id
        return self

    def _build_session_snapshot(self):
        return self._sessions.snapshot()

    def save_session(self, session_id=None, *, store=None, metadata=None, expires_at=None):
        return self._sessions.save(session_id, store=store, metadata=metadata, expires_at=expires_at)

    def restore_session(self, session_id, *, store=None):
        return self._sessions.restore(session_id, store=store)

    def fork_session(self, session_id=None, *, store=None, metadata=None):
        return self._sessions.fork(session_id, store=store, metadata=metadata)

    async def asave_session(self, session_id=None, **kwargs):
        return await self._sessions.asave(session_id, **kwargs)

    async def arestore_session(self, session_id, **kwargs):
        return await self._sessions.arestore(session_id, **kwargs)

    async def afork_session(self, session_id=None, **kwargs):
        return await self._sessions.afork(session_id, **kwargs)

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
        if self._owns_session_store:
            close_component("session_store", self._sessions.store)
        try:
            self._detach_memory()
        except Exception as exc:
            report["status"] = "degraded"
            report["issues"].append({"component": "memory", "error": str(exc)})
        if close_llm:
            close_component("llm", self.llm)
        self._closed = True
        self.last_close_report = report
        return dict(report)

    def get_last_close_report(self) -> dict[str, Any] | None:
        return dict(self.last_close_report) if self.last_close_report is not None else None

    async def aclose(self, **kwargs):
        """Close owned async memory/storage before the existing product cleanup."""
        import asyncio
        if self._closed:
            return self.get_last_close_report()
        if self._invocation_lock.locked() or self._sessions.busy:
            raise RuntimeError("Close an Agent after its active operation completes")
        issues = []
        for name, component, flag in (
            ("memory", self.memory, "_owns_memory"),
            ("session_store", self._sessions.store, "_owns_session_store"),
        ):
            if component is not None and getattr(self, flag):
                try:
                    await component.aclose()
                except Exception as exc:
                    issues.append({"component": name, "error": str(exc)})
                finally:
                    setattr(self, flag, False)
        report = await asyncio.to_thread(self.close, **kwargs)
        if issues:
            report["status"] = "degraded"
            report.setdefault("issues", []).extend(issues)
            self.last_close_report = report
        return report

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
