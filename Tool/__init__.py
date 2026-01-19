# Tool module for EasyAgent
from .BaseTool import Tool
from .ToolRegistry import ToolRegistry
from .memory_tools import (
    MemorySearchTool,
    MemorySaveTool,
    RAGSearchTool,
    register_memory_tools,
    register_rag_tool,
)
from .builtin import (
    WebSearchTool,
    CalculatorTool,
    register_search_tool,
    register_calculator_tool,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "MemorySearchTool",
    "MemorySaveTool",
    "RAGSearchTool",
    "register_memory_tools",
    "register_rag_tool",
    # Builtin tools
    "WebSearchTool",
    "CalculatorTool",
    "register_search_tool",
    "register_calculator_tool",
]
