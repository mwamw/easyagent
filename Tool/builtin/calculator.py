"""
计算器工具

安全执行数学表达式。
"""
import math
import operator
import ast
import logging
from typing import Any, Dict
from pydantic import BaseModel, Field

from ..BaseTool import Tool
from ..ToolRegistry import ToolRegistry

logger = logging.getLogger(__name__)


class CalculatorParams(BaseModel):
    """计算器参数"""
    expression: str = Field(description="数学表达式，如 '2 + 3 * 4' 或 'sqrt(16) + pow(2, 3)'")


# 允许的操作符和函数
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

ALLOWED_FUNCTIONS = {
    # 基础数学函数
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    
    # math 模块函数
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "radians": math.radians,
    "degrees": math.degrees,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
}


class SafeEvaluator(ast.NodeVisitor):
    """
    安全的表达式求值器
    
    只允许基本数学运算和预定义的函数。
    """
    
    def visit(self, node: ast.AST) -> Any:
        """访问 AST 节点"""
        return super().visit(node)
    
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)
    
    def visit_Constant(self, node: ast.Constant) -> Any:
        """处理常量（数字）"""
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"不允许的常量类型: {type(node.value)}")
    
    def visit_Num(self, node: ast.Num) -> Any:
        """处理数字（Python 3.7 兼容）"""
        return node.n
    
    def visit_Name(self, node: ast.Name) -> Any:
        """处理变量名（只允许预定义常量）"""
        name = node.id
        if name in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[name]
        raise ValueError(f"不允许的变量名: {name}")
    
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """处理二元运算"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"不允许的运算符: {op_type.__name__}")
        
        return ALLOWED_OPERATORS[op_type](left, right)
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """处理一元运算"""
        operand = self.visit(node.operand)
        op_type = type(node.op)
        
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"不允许的运算符: {op_type.__name__}")
        
        return ALLOWED_OPERATORS[op_type](operand)
    
    def visit_Call(self, node: ast.Call) -> Any:
        """处理函数调用"""
        if not isinstance(node.func, ast.Name):
            raise ValueError("不允许的函数调用方式")
        
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"不允许的函数: {func_name}")
        
        args = [self.visit(arg) for arg in node.args]
        return ALLOWED_FUNCTIONS[func_name](*args)
    
    def visit_Tuple(self, node: ast.Tuple) -> Any:
        """处理元组"""
        return tuple(self.visit(el) for el in node.elts)
    
    def visit_List(self, node: ast.List) -> Any:
        """处理列表"""
        return [self.visit(el) for el in node.elts]
    
    def generic_visit(self, node: ast.AST) -> Any:
        """未知节点类型"""
        raise ValueError(f"不允许的表达式类型: {type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """
    安全地求值数学表达式
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        计算结果
        
    Raises:
        ValueError: 表达式不合法或包含不允许的操作
    """
    try:
        tree = ast.parse(expression, mode='eval')
        evaluator = SafeEvaluator()
        return evaluator.visit(tree)
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}")


class CalculatorTool(Tool):
    """
    安全计算器工具
    
    支持基本数学运算和常用数学函数。
    使用 AST 解析确保安全，防止代码注入。
    
    支持的运算：
    - 基本运算: +, -, *, /, //, %, **
    - 函数: sqrt, sin, cos, tan, log, exp, pow, abs, round, min, max 等
    - 常量: pi, e
    
    Example:
        >>> tool = CalculatorTool()
        >>> tool.run({"expression": "2 + 3 * 4"})
        '14'
        >>> tool.run({"expression": "sqrt(16) + pow(2, 3)"})
        '12.0'
    """
    
    def __init__(self):
        """初始化计算器工具"""
        super().__init__(
            name="calculator",
            description="安全的数学计算器，支持基本运算(+,-,*,/,**,%,//)和数学函数(sqrt,sin,cos,log,pow等)。输入数学表达式。",
            parameters=CalculatorParams
        )
    
    def run(self, parameters: dict) -> str:
        """执行计算"""
        expression = parameters.get("expression", "")
        
        if not expression:
            return "错误：表达式不能为空"
        
        # 预处理表达式（替换中文符号等）
        expression = self._preprocess(expression)
        
        try:
            result = safe_eval(expression)
            
            # 格式化结果
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                else:
                    # 保留合理的小数位数
                    return f"{result:.10g}"
            
            return str(result)
            
        except ValueError as e:
            return f"计算错误: {e}"
        except ZeroDivisionError:
            return "错误: 除数不能为零"
        except OverflowError:
            return "错误: 结果超出范围"
        except Exception as e:
            logger.error(f"计算失败: {e}")
            return f"计算失败: {str(e)}"
    
    def _preprocess(self, expression: str) -> str:
        """预处理表达式"""
        # 替换中文符号
        replacements = {
            "（": "(",
            "）": ")",
            "×": "*",
            "÷": "/",
            "＋": "+",
            "－": "-",
            "，": ",",
            "^": "**",  # 支持 ^ 作为幂运算
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression.strip()
    
    def get_help(self) -> str:
        """获取帮助信息"""
        return """计算器工具支持：

运算符：
  + - * /    基本四则运算
  //         整除
  %          取余
  **         幂运算

函数：
  sqrt(x)    平方根
  pow(x, y)  幂运算
  abs(x)     绝对值
  round(x)   四舍五入
  sin/cos/tan 三角函数
  log(x)     自然对数
  log10(x)   以10为底的对数
  exp(x)     e的x次方
  ceil/floor 上取整/下取整

常量：
  pi         圆周率 3.14159...
  e          自然常数 2.71828...

示例：
  2 + 3 * 4        => 14
  sqrt(16) + 2**3  => 12
  sin(pi/2)        => 1"""


def register_calculator_tool(registry: ToolRegistry) -> CalculatorTool:
    """
    注册计算器工具到 ToolRegistry
    
    Args:
        registry: 工具注册表
        
    Returns:
        创建的 CalculatorTool 实例
    """
    tool = CalculatorTool()
    registry.registerTool(tool)
    return tool
