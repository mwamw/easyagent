"""
输出解析器基类模块
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic

T = TypeVar('T')


class BaseOutputParser(ABC, Generic[T]):
    """
    输出解析器抽象基类
    
    用于将 LLM 的文本输出解析为结构化数据。
    
    所有解析器实现必须继承此类并实现以下方法：
    - parse: 解析输出文本
    - get_format_instructions: 获取格式说明（注入到提示词）
    """
    
    @abstractmethod
    def parse(self, output: str) -> T:
        """
        解析 LLM 输出
        
        Args:
            output: LLM 输出的文本
            
        Returns:
            解析后的结构化数据
            
        Raises:
            OutputParseError: 解析失败时抛出
        """
        pass
    
    @abstractmethod
    def get_format_instructions(self) -> str:
        """
        获取格式说明
        
        返回的说明将被注入到提示词中，指导 LLM 生成符合格式的输出。
        
        Returns:
            格式说明字符串
        """
        pass
    
    def parse_with_prompt(self, output: str, prompt: str) -> T:
        """
        带提示词的解析（用于错误恢复）
        
        Args:
            output: LLM 输出的文本
            prompt: 原始提示词
            
        Returns:
            解析后的结构化数据
        """
        return self.parse(output)
    
    def __call__(self, output: str) -> T:
        """允许直接调用解析器"""
        return self.parse(output)


class OutputParseError(Exception):
    """输出解析异常"""
    
    def __init__(self, message: str, output: Optional[str] = None):
        super().__init__(message)
        self.output = output
    
    def __str__(self):
        if self.output:
            return f"{self.args[0]}\n原始输出:\n{self.output[:500]}..."
        return self.args[0]
