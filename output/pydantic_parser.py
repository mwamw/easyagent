"""
Pydantic 输出解析器
"""
import json
from typing import Type, TypeVar, Generic
from pydantic import BaseModel, ValidationError
from .base import BaseOutputParser, OutputParseError
from .json_parser import JsonOutputParser

T = TypeVar('T', bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T], Generic[T]):
    """
    Pydantic 模型输出解析器
    
    将 LLM 输出解析为 Pydantic 模型实例。
    自动生成格式说明和进行数据验证。
    
    Example:
        >>> from pydantic import BaseModel, Field
        >>> 
        >>> class Person(BaseModel):
        ...     name: str = Field(description="姓名")
        ...     age: int = Field(description="年龄")
        >>> 
        >>> parser = PydanticOutputParser(Person)
        >>> result = parser.parse('{"name": "张三", "age": 25}')
        >>> print(result.name)
        张三
    """
    
    def __init__(self, pydantic_model: Type[T]):
        """
        初始化 Pydantic 解析器
        
        Args:
            pydantic_model: Pydantic 模型类
        """
        self.model = pydantic_model
        self._json_parser = JsonOutputParser()
    
    def parse(self, output: str) -> T:
        """
        解析 LLM 输出为 Pydantic 模型
        
        Args:
            output: LLM 输出的文本
            
        Returns:
            Pydantic 模型实例
            
        Raises:
            OutputParseError: 解析或验证失败
        """
        try:
            # 先解析为 JSON
            data = self._json_parser.parse(output)
            
            # 再验证为 Pydantic 模型
            return self.model.model_validate(data)
        
        except OutputParseError:
            raise
        except ValidationError as e:
            raise OutputParseError(f"Pydantic 验证失败: {e}", output)
        except Exception as e:
            raise OutputParseError(f"解析失败: {e}", output)
    
    def get_format_instructions(self) -> str:
        """获取格式说明（包含 JSON Schema）"""
        schema = self.model.model_json_schema()
        
        # 提取字段描述
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        field_descriptions = []
        for name, info in properties.items():
            desc = info.get("description", "")
            field_type = info.get("type", "any")
            is_required = "必填" if name in required else "可选"
            field_descriptions.append(f"  - {name} ({field_type}, {is_required}): {desc}")
        
        fields_str = "\n".join(field_descriptions) if field_descriptions else "  (无字段描述)"
        
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        return f"""请以 JSON 格式输出，符合以下结构：

字段说明：
{fields_str}

JSON Schema:
```json
{schema_str}
```

请确保输出的 JSON 符合上述 Schema 定义。"""
    
    def get_schema(self) -> dict:
        """获取 Pydantic 模型的 JSON Schema"""
        return self.model.model_json_schema()


class ListPydanticOutputParser(BaseOutputParser[list[T]], Generic[T]):
    """
    Pydantic 列表输出解析器
    
    将 LLM 输出解析为 Pydantic 模型列表。
    """
    
    def __init__(self, pydantic_model: Type[T]):
        """
        初始化列表解析器
        
        Args:
            pydantic_model: 列表元素的 Pydantic 模型类
        """
        self.model = pydantic_model
        self._json_parser = JsonOutputParser()
    
    def parse(self, output: str) -> list[T]:
        """解析为 Pydantic 模型列表"""
        try:
            data = self._json_parser.parse(output)
            
            if not isinstance(data, list):
                # 尝试从对象中提取列表
                if isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, list):
                            data = value
                            break
                    else:
                        raise OutputParseError("期望列表格式", output)
                else:
                    raise OutputParseError("期望列表格式", output)
            
            return [self.model.model_validate(item) for item in data]
        
        except OutputParseError:
            raise
        except ValidationError as e:
            raise OutputParseError(f"Pydantic 验证失败: {e}", output)
        except Exception as e:
            raise OutputParseError(f"解析失败: {e}", output)
    
    def get_format_instructions(self) -> str:
        """获取格式说明"""
        schema = self.model.model_json_schema()
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        return f"""请以 JSON 数组格式输出，每个元素符合以下结构：

```json
{schema_str}
```

示例输出：
```json
[
  {{"...": "..."}},
  {{"...": "..."}}
]
```"""
