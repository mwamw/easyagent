# Tool module for EasyAgent
from .BaseTool import Tool, ToolResult, ToolSpec
from .ToolRegistry import ToolRegistry
from .claude_compat import (
    CLAUDE_TOOL_MODELS,
    CLAUDE_TOOL_ORDER,
    ClaudeCompatDelegatingTool,
    ClaudeCompatTool,
    ClaudeToolDefinition,
    get_claude_tool_definition,
)
from mcp import MCPClient
from .runtime import (
    FilesystemAccessError,
    FilesystemGuard,
    GitWorktreeInfo,
    PathResolutionError,
    PathResolver,
    ProcessExecutionResult,
    ProcessManager,
    WorktreeManager,
)
from .builtin import (
    WebSearchTool,
    CalculatorTool,
    register_search_tool,
    register_calculator_tool,
    MCPToolManager,
    MCPWrappedTool,
    MCPListResourcesTool,
    MCPReadResourceTool,
    register_mcp_tools,
    mcptool,
)

__all__ = [
    "Tool",
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
    "ClaudeCompatTool",
    "ClaudeCompatDelegatingTool",
    "ClaudeToolDefinition",
    "CLAUDE_TOOL_MODELS",
    "CLAUDE_TOOL_ORDER",
    "get_claude_tool_definition",
    "PathResolver",
    "PathResolutionError",
    "FilesystemGuard",
    "FilesystemAccessError",
    "ProcessManager",
    "ProcessExecutionResult",
    "WorktreeManager",
    "GitWorktreeInfo",
    # Builtin tools
    "WebSearchTool",
    "CalculatorTool",
    "register_search_tool",
    "register_calculator_tool",
    "MCPClient",
    "MCPToolManager",
    "MCPWrappedTool",
    "MCPListResourcesTool",
    "MCPReadResourceTool",
    "register_mcp_tools",
    "mcptool",
]
