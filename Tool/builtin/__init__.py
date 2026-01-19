"""
预置工具模块
"""
from .search import WebSearchTool, register_search_tool
from .calculator import CalculatorTool, register_calculator_tool

__all__ = [
    "WebSearchTool",
    "CalculatorTool",
    "register_search_tool",
    "register_calculator_tool",
]
