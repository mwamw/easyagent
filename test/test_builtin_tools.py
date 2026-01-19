"""
预置工具单元测试
"""
import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tool.builtin.calculator import CalculatorTool, safe_eval
from Tool.builtin.search import WebSearchTool
from Tool.ToolRegistry import ToolRegistry


class TestCalculatorTool(unittest.TestCase):
    """CalculatorTool 测试"""
    
    def setUp(self):
        self.calculator = CalculatorTool()
    
    def test_basic_arithmetic(self):
        """测试基本四则运算"""
        self.assertEqual(self.calculator.run({"expression": "2 + 3"}), "5")
        self.assertEqual(self.calculator.run({"expression": "10 - 4"}), "6")
        self.assertEqual(self.calculator.run({"expression": "3 * 4"}), "12")
        self.assertEqual(self.calculator.run({"expression": "15 / 3"}), "5")
    
    def test_complex_expressions(self):
        """测试复杂表达式"""
        self.assertEqual(self.calculator.run({"expression": "2 + 3 * 4"}), "14")
        self.assertEqual(self.calculator.run({"expression": "(2 + 3) * 4"}), "20")
        self.assertEqual(self.calculator.run({"expression": "2 ** 10"}), "1024")
    
    def test_math_functions(self):
        """测试数学函数"""
        self.assertEqual(self.calculator.run({"expression": "sqrt(16)"}), "4")
        self.assertEqual(self.calculator.run({"expression": "abs(-5)"}), "5")
        self.assertEqual(self.calculator.run({"expression": "pow(2, 3)"}), "8")
    
    def test_constants(self):
        """测试数学常量"""
        result = float(self.calculator.run({"expression": "pi"}))
        self.assertAlmostEqual(result, 3.14159, places=4)
    
    def test_division_by_zero(self):
        """测试除零错误"""
        result = self.calculator.run({"expression": "1 / 0"})
        self.assertIn("零", result)
    
    def test_invalid_expression(self):
        """测试无效表达式"""
        result = self.calculator.run({"expression": "import os"})
        self.assertIn("错误", result)
    
    def test_chinese_symbols(self):
        """测试中文符号转换"""
        self.assertEqual(self.calculator.run({"expression": "（2＋3）×4"}), "20")
    
    def test_empty_expression(self):
        """测试空表达式"""
        result = self.calculator.run({"expression": ""})
        self.assertIn("错误", result)


class TestSafeEval(unittest.TestCase):
    """safe_eval 函数测试"""
    
    def test_allowed_operations(self):
        """测试允许的操作"""
        self.assertEqual(safe_eval("1 + 1"), 2)
        self.assertEqual(safe_eval("sqrt(4)"), 2.0)
    
    def test_blocked_operations(self):
        """测试阻止的操作"""
        with self.assertRaises(ValueError):
            safe_eval("__import__('os')")
        
        with self.assertRaises(ValueError):
            safe_eval("open('file.txt')")


class TestWebSearchTool(unittest.TestCase):
    """WebSearchTool 测试"""
    
    def test_initialization(self):
        """测试初始化"""
        tool = WebSearchTool(backend="duckduckgo")
        self.assertEqual(tool.name, "web_search")
        self.assertEqual(tool.backend, "duckduckgo")
    
    def test_auto_backend_selection(self):
        """测试自动后端选择"""
        # 没有 API Key 时应该选择 duckduckgo
        tool = WebSearchTool(api_key=None, backend="auto")
        self.assertEqual(tool.backend, "duckduckgo")
    
    def test_empty_query(self):
        """测试空查询"""
        tool = WebSearchTool()
        result = tool.run({"query": ""})
        self.assertIn("错误", result)
    
    def test_openai_schema(self):
        """测试 OpenAI Schema 生成"""
        tool = WebSearchTool()
        schema = tool.get_openai_schema()
        
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "web_search")
        self.assertIn("parameters", schema["function"])


class TestToolRegistration(unittest.TestCase):
    """工具注册测试"""
    
    def test_register_calculator(self):
        """测试注册计算器工具"""
        from Tool.builtin import register_calculator_tool
        
        registry = ToolRegistry()
        tool = register_calculator_tool(registry)
        
        self.assertIn("calculator", registry.tools)
        self.assertEqual(tool.name, "calculator")
    
    def test_register_search(self):
        """测试注册搜索工具"""
        from Tool.builtin import register_search_tool
        
        registry = ToolRegistry()
        tool = register_search_tool(registry)
        
        self.assertIn("web_search", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
