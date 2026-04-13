"""Claude Code compatibility layer for EasyAgent tools."""

from .catalog import CLAUDE_TOOL_MODELS, CLAUDE_TOOL_ORDER, ClaudeToolDefinition, get_claude_tool_definition
from .wrappers import ClaudeCompatDelegatingTool, ClaudeCompatTool

__all__ = [
    "ClaudeCompatTool",
    "ClaudeCompatDelegatingTool",
    "ClaudeToolDefinition",
    "CLAUDE_TOOL_MODELS",
    "CLAUDE_TOOL_ORDER",
    "get_claude_tool_definition",
]
