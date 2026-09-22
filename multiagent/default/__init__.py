"""Built-in Claude Code-style multi-agent harness."""

from importlib import import_module


_EXPORTS = {
    "AgentHandle": (".agents", "AgentHandle"),
    "AgentRuntimeManager": (".agents", "AgentRuntimeManager"),
    "BackgroundAgentHandle": (".agents", "BackgroundAgentHandle"),
    "ClaudeCodeMultiAgent": (".module", "ClaudeCodeMultiAgent"),
    "CompletionRecord": (".agents", "CompletionRecord"),
    "AgentDefinition": (".definitions", "AgentDefinition"),
    "AgentDefinitionRegistry": (".definitions", "AgentDefinitionRegistry"),
    "build_default_agent_definitions": (".definitions", "build_default_agent_definitions"),
    "MailboxMessage": (".agents", "MailboxMessage"),
    "SubagentManager": (".subagents", "SubagentManager"),
    "SubagentRequest": (".subagents", "SubagentRequest"),
    "SubagentSnapshot": (".subagents", "SubagentSnapshot"),
    "TeamHandle": (".teams", "TeamHandle"),
    "TeamManager": (".teams", "TeamManager"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
