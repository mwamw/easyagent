# Tool module for EasyAgent
from .BaseTool import Tool
from .ToolRegistry import ToolRegistry
from .builtin import (
    WebSearchTool,
    CalculatorTool,
    register_search_tool,
    register_calculator_tool,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    # Builtin tools
    "WebSearchTool",
    "CalculatorTool",
    "register_search_tool",
    "register_calculator_tool",
]
