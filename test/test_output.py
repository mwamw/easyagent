"""
Output 输出解析器模块单元测试
"""
import unittest
import sys
import os
from pydantic import BaseModel, Field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output.json_parser import JsonOutputParser
from output.pydantic_parser import PydanticOutputParser, ListPydanticOutputParser
from output.base import OutputParseError


class TestJsonOutputParser(unittest.TestCase):
    """JsonOutputParser 测试"""
    
    def setUp(self):
        self.parser = JsonOutputParser()
    
    def test_parse_simple_json(self):
        """测试解析简单 JSON"""
        result = self.parser.parse('{"name": "张三", "age": 25}')
        self.assertEqual(result["name"], "张三")
        self.assertEqual(result["age"], 25)
    
    def test_parse_json_in_markdown(self):
        """测试解析 Markdown 代码块中的 JSON"""
        text = """
这是一些说明文字

```json
{"name": "李四", "age": 30}
```

更多文字
"""
        result = self.parser.parse(text)
        self.assertEqual(result["name"], "李四")
    
    def test_parse_json_with_extra_text(self):
        """测试从混合文本中提取 JSON"""
        text = '根据我的分析，结果是：{"status": "success", "value": 42}'
        result = self.parser.parse(text)
        self.assertEqual(result["status"], "success")
    
    def test_parse_array(self):
        """测试解析 JSON 数组"""
        result = self.parser.parse('[1, 2, 3, 4]')
        self.assertEqual(result, [1, 2, 3, 4])
    
    def test_parse_invalid_json(self):
        """测试解析无效 JSON"""
        with self.assertRaises(OutputParseError):
            self.parser.parse("这不是有效的 JSON")


# 定义测试用的 Pydantic 模型
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


class Task(BaseModel):
    title: str = Field(description="任务标题")
    priority: int = Field(default=1, ge=1, le=5, description="优先级 1-5")
    completed: bool = Field(default=False, description="是否完成")


class TestPydanticOutputParser(unittest.TestCase):
    """PydanticOutputParser 测试"""
    
    def test_parse_valid_json(self):
        """测试解析有效 JSON 到 Pydantic 模型"""
        parser = PydanticOutputParser(Person)
        result = parser.parse('{"name": "张三", "age": 25}')
        
        self.assertIsInstance(result, Person)
        self.assertEqual(result.name, "张三")
        self.assertEqual(result.age, 25)
    
    def test_parse_with_defaults(self):
        """测试带默认值的解析"""
        parser = PydanticOutputParser(Task)
        result = parser.parse('{"title": "完成报告"}')
        
        self.assertEqual(result.title, "完成报告")
        self.assertEqual(result.priority, 1)
        self.assertFalse(result.completed)
    
    def test_parse_invalid_type(self):
        """测试类型错误"""
        parser = PydanticOutputParser(Person)
        with self.assertRaises(OutputParseError):
            parser.parse('{"name": "张三", "age": "not a number"}')
    
    def test_get_format_instructions(self):
        """测试获取格式说明"""
        parser = PydanticOutputParser(Person)
        instructions = parser.get_format_instructions()
        
        self.assertIn("JSON", instructions)
        self.assertIn("name", instructions)
        self.assertIn("age", instructions)
    
    def test_get_schema(self):
        """测试获取 Schema"""
        parser = PydanticOutputParser(Person)
        schema = parser.get_schema()
        
        self.assertIn("properties", schema)
        self.assertIn("name", schema["properties"])


class TestListPydanticOutputParser(unittest.TestCase):
    """ListPydanticOutputParser 测试"""
    
    def test_parse_list(self):
        """测试解析列表"""
        parser = ListPydanticOutputParser(Person)
        # LLM 通常以 markdown 代码块格式输出
        result = parser.parse('```json\n[{"name": "张三", "age": 25}, {"name": "李四", "age": 30}]\n```')
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "张三")
        self.assertEqual(result[1].name, "李四")
    
    def test_parse_nested_list(self):
        """测试从对象中提取列表"""
        parser = ListPydanticOutputParser(Person)
        result = parser.parse('{"people": [{"name": "王五", "age": 35}]}')
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "王五")


if __name__ == "__main__":
    unittest.main(verbosity=2)
