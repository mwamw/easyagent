"""Stable public SDK facade for EasyAgent."""

from .agents import (
    BaseAgent,
    BasicAgent,
    ConversationalAgent,
    PlanningAgent,
    ReactAgent,
    StructuredOutputAgent,
)
from .codeintel import CachedFileEntry, CachedQueryEntry, CodeIntelManager, CodeIntelProvider, WorkspaceCodeIntelCache
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
from .runtime import ExecutionContext, TeamManager
from .session import ConversationStore, SessionRestoreReport, SessionStore
from .skills import BaseSkill, SkillManager, SkillRegistry
from .tasks import TaskRecord, TaskService, TaskStatus
from .tools import Tool, ToolConflictPolicy, ToolRegistry, ToolResult, ToolSpec
from .version import __version__

__all__ = [
    "__version__",
    "BaseAgent",
    "BaseHook",
    "BaseSkill",
    "BasicAgent",
    "CachedFileEntry",
    "CachedQueryEntry",
    "CodeIntelManager",
    "CodeIntelProvider",
    "ContextManager",
    "ConversationStore",
    "ConversationalAgent",
    "DangerousCommandGuardrail",
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
    "PromptInjectionGuardrail",
    "ReactAgent",
    "RiskCategory",
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
    "WorkspaceCodeIntelCache",
    "build_default_hook_manager",
    "install_default_guardrails",
    "register_mcp_tools",
]
