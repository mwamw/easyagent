# Tool module for EasyAgent
from .BaseTool import Tool
from .ToolRegistry import ToolRegistry
from mcp import MCPClient
from .builtin import (
    WebSearchTool,
    CalculatorTool,
    register_search_tool,
    register_calculator_tool,
    MCPToolManager,
    MCPWrappedTool,
    register_mcp_tools,
    mcptool,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    # Builtin tools
    "WebSearchTool",
    "CalculatorTool",
    "register_search_tool",
    "register_calculator_tool",
    "MCPClient",
    "MCPToolManager",
    "MCPWrappedTool",
    "register_mcp_tools",
    "mcptool",
]
