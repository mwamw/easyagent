"""
JSON 输出解析器
"""
import json
import re
from typing import Any, Optional, Dict
from .base import BaseOutputParser, OutputParseError


class JsonOutputParser(BaseOutputParser[Dict[str, Any]]):
    """
    JSON 输出解析器
    
    从 LLM 输出中提取并解析 JSON 数据。
    支持多种 JSON 格式：
    - 纯 JSON 字符串
    - Markdown 代码块中的 JSON
    - 混合文本中的 JSON
    
    Example:
        >>> parser = JsonOutputParser()
        >>> result = parser.parse('```json\\n{"name": "张三", "age": 25}\\n```')
        >>> print(result)
        {'name': '张三', 'age': 25}
    """
    
    def __init__(self, ensure_ascii: bool = False):
        """
        初始化 JSON 解析器
        
        Args:
            ensure_ascii: JSON 编码时是否确保 ASCII（默认 False，支持中文）
        """
        self.ensure_ascii = ensure_ascii
    
    def parse(self, output: str) -> Dict[str, Any]:
        """
        解析 LLM 输出中的 JSON
        
        Args:
            output: LLM 输出的文本
            
        Returns:
            解析后的字典
            
        Raises:
            OutputParseError: JSON 解析失败
        """
        if not output or not output.strip():
            raise OutputParseError("输出为空", output)
        
        # 尝试多种方式提取 JSON
        json_str = self._extract_json(output)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise OutputParseError(f"JSON 解析失败: {e}", output)
    
    def _extract_json(self, text: str) -> str:
        """
        从文本中提取 JSON 字符串
        
        尝试多种模式：
        1. Markdown 代码块 ```json ... ```
        2. 普通代码块 ``` ... ```
        3. 花括号包围的 JSON 对象
        4. 方括号包围的 JSON 数组
        5. 整个文本作为 JSON
        """
        text = text.strip()
        
        # 1. 尝试匹配 ```json ... ``` 代码块
        json_block = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if json_block:
            return json_block.group(1).strip()
        
        # 2. 尝试匹配普通 ``` ... ``` 代码块
        code_block = re.search(r'```\s*([\s\S]*?)\s*```', text)
        if code_block:
            content = code_block.group(1).strip()
            if content.startswith('{') or content.startswith('['):
                return content
        
        # 3. 尝试匹配花括号包围的 JSON 对象
        obj_match = re.search(r'\{[\s\S]*\}', text)
        if obj_match:
            return obj_match.group(0)
        
        # 4. 尝试匹配方括号包围的 JSON 数组
        arr_match = re.search(r'\[[\s\S]*\]', text)
        if arr_match:
            return arr_match.group(0)
        
        # 5. 返回原始文本
        return text
    
    def get_format_instructions(self) -> str:
        """获取 JSON 格式说明"""
        return """请以 JSON 格式输出结果。

输出格式要求：
1. 使用有效的 JSON 格式
2. 可以用 ```json 代码块包裹
3. 确保所有字符串使用双引号
4. 不要包含注释

示例：
```json
{
  "key1": "value1",
  "key2": 123
}
```"""


class JsonListOutputParser(BaseOutputParser[list]):
    """
    JSON 列表输出解析器
    
    专门用于解析 JSON 数组格式的输出。
    """
    
    def __init__(self):
        self._json_parser = JsonOutputParser()
    
    def parse(self, output: str) -> list:
        """解析 JSON 数组"""
        result = self._json_parser.parse(output)
        if not isinstance(result, list):
            # 如果返回的是对象，尝试提取其中的列表
            if isinstance(result, dict):
                for value in result.values():
                    if isinstance(value, list):
                        return value
            raise OutputParseError("期望 JSON 数组，但收到其他类型", output)
        return result
    
    def get_format_instructions(self) -> str:
        """获取 JSON 数组格式说明"""
        return """请以 JSON 数组格式输出结果。

示例：
```json
[
  {"item": "项目1"},
  {"item": "项目2"}
]
```"""
