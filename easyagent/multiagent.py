"""Stable public API for replaceable multi-agent modules."""

from multiagent import BaseMultiAgent, ClaudeCodeMultiAgent
from multiagent.default import (
    AgentHandle,
    AgentRuntimeManager,
    BackgroundAgentHandle,
    CompletionRecord,
    MailboxMessage,
    SubagentManager,
    SubagentRequest,
    SubagentSnapshot,
    TeamHandle,
    TeamManager,
)
from multiagent.default.factory import DefaultSubagentFactory
from multiagent.default.tool_binding import ChildToolFactory
from multiagent.default.permissions import ChildPermissionFactory
from multiagent.default.definitions import (
    AgentDefinition,
    AgentDefinitionRegistry,
    build_default_agent_definitions,
)

__all__ = [
    "AgentHandle",
    "AgentDefinition",
    "AgentDefinitionRegistry",
    "AgentRuntimeManager",
    "BackgroundAgentHandle",
    "BaseMultiAgent",
    "ClaudeCodeMultiAgent",
    "CompletionRecord",
    "ChildToolFactory",
    "ChildPermissionFactory",
    "DefaultSubagentFactory",
    "MailboxMessage",
    "SubagentManager",
    "SubagentRequest",
    "SubagentSnapshot",
    "TeamHandle",
    "TeamManager",
    "build_default_agent_definitions",
]
