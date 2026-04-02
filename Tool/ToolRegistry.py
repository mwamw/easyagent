import json
from typing import Callable
from pydantic import BaseModel
from .BaseTool import Tool
from typing import Type
from functools import wraps
class ToolRegistry:
    def __init__(self):
        self.tools:dict[str,Tool]={}

    
    def register_tool(self,tool:Tool):
        self.tools[tool.name]=tool

    def registry(self, item):
        """兼容注册入口：支持 Tool 实例或带 register_to_registry 的对象。"""
        if isinstance(item, Tool):
            self.register_tool(item)
            return item

        register_fn = getattr(item, "register_to_registry", None)
        if callable(register_fn):
            return register_fn(self)

        raise ValueError("registry(...) 仅支持 Tool 实例或可注册对象")

    def tool(self, name: str, description: str, parameters: Type[BaseModel]):
        """装饰器：注册函数为工具"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 注册到 registry
            class FunctionTool(Tool):
                def run(self, parameters: dict):
                    return func(**parameters)

            tool_instance = FunctionTool(name, description, parameters)
            self.register_tool(tool_instance)
            
            return wrapper
        return decorator

    def get_tools_description(self)->list[dict]:
        result=[]
        for tool in self.tools.values():
            description={"type":"tool","name":tool.name,"description":tool.description,"parameters":tool.parameters.model_json_schema()}
            result.append(description)

        return result

    def execute_tool(self,name:str,parameters:dict):
        if name in self.tools:
            try:
                result= self.tools[name](parameters)
                if isinstance(result, (dict, list)):
                    return json.dumps(result, ensure_ascii=False, indent=2)
                return str(result)
            except Exception as e:
                raise ValueError(f"Invalid parameters: {e}")
        else:
            raise ValueError(f"Tool {name} not found")

    def get_openai_tools(self)->list[dict]:
        result=[]
        for tool in self.tools.values():
            result.append(tool.get_openai_schema())
        return result

    def unregister_tool(self,name:str):
        if name in self.tools:
            del self.tools[name]
        else:
            print(f"Tool {name} not found")

    def register_tools(self, tools: list) -> None:
        """批量注册多个工具"""
        for tool in tools:
            self.register_tool(tool)

    def unregister_tools(self, names: list) -> None:
        """批量移除多个工具"""
        for name in names:
            self.unregister_tool(name)

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self.tools

    def get_tool_names(self) -> list:
        """获取所有已注册工具名称"""
        return list(self.tools.keys())

    def get_tool(self,name:str):
        return self.tools.get(name)

    # ==================== 向后兼容别名 ====================

    def registerTool(self, tool: Tool):
        """向后兼容：请改用 register_tool"""
        return self.register_tool(tool)

    def executeTool(self, name: str, parameters: dict):
        """向后兼容：请改用 execute_tool"""
        return self.execute_tool(name, parameters)

    def get_Tool(self, name: str):
        """向后兼容：请改用 get_tool"""
        return self.get_tool(name)

    def disregister_tool(self, name: str):
        """向后兼容：请改用 unregister_tool"""
        return self.unregister_tool(name)
