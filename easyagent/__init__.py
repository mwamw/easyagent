"""Stable public SDK facade for EasyAgent."""

from .agents import (
    BaseAgent,
    BasicAgent,
    ConversationalAgent,
    PlanningAgent,
    ReactAgent,
    StructuredOutputAgent,
)
from .callbacks import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    LoggingCallback,
    MetricsCallback,
    StreamingCallback,
)
from .codeintel import CachedFileEntry, CachedQueryEntry, CodeIntelManager, CodeIntelProvider, WorkspaceCodeIntelCache
from .config import Config
from .context import ContextManager
from .guardrails import (
    DangerousCommandGuardrail,
    PromptInjectionGuardrail,
    SecretLeakGuardrail,
    build_default_hook_manager,
    install_default_guardrails,
)
from .hooks import BaseHook, HookAction, HookDecision, HookExecutionResult, HookManager
from .llms import EasyLLM
from .mcp import (
    MCPAuthConfig,
    MCPCapabilitySnapshot,
    MCPClient,
    MCPConnectionManager,
    MCPConnectionState,
    MCPHub,
    MCPPolicyContext,
    MCPPolicyDecision,
    MCPPolicyError,
    MCPPolicyRule,
    MCPRuntimeManager,
    MCPServerCache,
    MCPToolManager,
    register_mcp_tools,
)
from .observability import BaseObservabilityRecorder, InMemoryObservabilityRecorder
from .permissions import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
    PermissionEngine,
    PermissionMode,
    PermissionRule,
    PermissionStore,
    RiskCategory,
)
from .prompting import BasePromptComposer, DefaultPromptComposer, PromptBlock, SystemPromptTemplate
from .reminders import BaseRuntimeReminderSource, RuntimeReminder, StaticRuntimeReminderSource
from .runtime import ExecutionContext, TeamManager
from .session import ConversationStore, SessionRestoreReport, SessionStore
from .skills import BaseSkill, SkillManager, SkillRegistry
from .tasks import TaskRecord, TaskService, TaskStatus
from .tools import Tool, ToolConflictPolicy, ToolRegistry, ToolResult, ToolSpec
from .worktree import GitWorktreeInfo, GitWorktreeSession, WorktreeManager
from .version import __version__

__all__ = [
    "__version__",
    "BaseAgent",
    "BaseCallback",
    "BaseHook",
    "BasePromptComposer",
    "BaseRuntimeReminderSource",
    "BaseSkill",
    "BasicAgent",
    "CachedFileEntry",
    "CachedQueryEntry",
    "CallbackEvent",
    "CallbackManager",
    "CodeIntelManager",
    "CodeIntelProvider",
    "Config",
    "ContextManager",
    "ConversationStore",
    "ConversationalAgent",
    "DangerousCommandGuardrail",
    "DefaultPromptComposer",
    "EasyLLM",
    "ExecutionContext",
    "HookAction",
    "HookDecision",
    "HookExecutionResult",
    "HookManager",
    "MCPAuthConfig",
    "MCPCapabilitySnapshot",
    "MCPClient",
    "MCPConnectionManager",
    "MCPConnectionState",
    "MCPHub",
    "MCPPolicyContext",
    "MCPPolicyDecision",
    "MCPPolicyError",
    "MCPPolicyRule",
    "MCPRuntimeManager",
    "MCPServerCache",
    "MCPToolManager",
    "BaseObservabilityRecorder",
    "InMemoryObservabilityRecorder",
    "PermissionBehavior",
    "PermissionContext",
    "PermissionDecision",
    "PermissionEngine",
    "PermissionMode",
    "PermissionRule",
    "PermissionStore",
    "PlanningAgent",
    "LoggingCallback",
    "MetricsCallback",
    "PromptInjectionGuardrail",
    "PromptBlock",
    "ReactAgent",
    "RiskCategory",
    "RuntimeReminder",
    "SecretLeakGuardrail",
    "SessionRestoreReport",
    "SessionStore",
    "SkillManager",
    "SkillRegistry",
    "StructuredOutputAgent",
    "TaskRecord",
    "TaskService",
    "TaskStatus",
    "TeamManager",
    "Tool",
    "ToolConflictPolicy",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "GitWorktreeInfo",
    "GitWorktreeSession",
    "StaticRuntimeReminderSource",
    "StreamingCallback",
    "SystemPromptTemplate",
    "WorktreeManager",
    "WorkspaceCodeIntelCache",
    "build_default_hook_manager",
    "install_default_guardrails",
    "register_mcp_tools",
]
