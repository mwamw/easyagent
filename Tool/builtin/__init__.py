"""
预置工具模块
"""
from .search import WebSearchTool, register_search_tool
from .calculator import CalculatorTool, register_calculator_tool
from .mcp_tool import MCPToolManager, MCPWrappedTool, register_mcp_tools, mcptool

__all__ = [
    "WebSearchTool",
    "CalculatorTool",
    "register_search_tool",
    "register_calculator_tool",
    "MCPToolManager",
    "MCPWrappedTool",
    "register_mcp_tools",
    "mcptool",
]
