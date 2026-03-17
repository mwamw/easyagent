import json
from typing import Callable

from pydantic import BaseModel
from .BaseTool import Tool
from typing import Type
from functools import wraps
class ToolRegistry:
    def __init__(self):
        self.tools:dict[str,Tool]={}

    
    def registerTool(self,tool:Tool):
        self.tools[tool.name]=tool

    def tool(self, name: str, description: str, parameters: Type[BaseModel]):
        """装饰器：注册函数为工具"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 注册到 registry
            class FunctionTool(Tool):
                def run(self, params: dict):
                    return func(**params)
            
            tool_instance = FunctionTool(name, description, parameters)
            self.registerTool(tool_instance)
            
            return wrapper
        return decorator

    def get_tools_description(self)->list[dict]:
        result=[]
        for tool in self.tools.values():
            description={"type":"tool","name":tool.name,"description":tool.description,"parameters":tool.parameters.model_json_schema()}
            result.append(description)

        return result

    def executeTool(self,name:str,parameters:dict):
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
    
    def disregister_tool(self,name:str):
        if name in self.tools:
            del self.tools[name]
        else:
            print(f"Tool {name} not found")

    def get_Tool(self,name:str):
        return self.tools.get(name)