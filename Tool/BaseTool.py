
from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Type

class Tool(ABC):
    def __init__(self,name:str,description:str,parameters:Type[BaseModel]):
        self.name=name
        self.description=description
        self.parameters=parameters
    
    @abstractmethod
    def run(self,parameters:dict):
        pass

    def get_openai_schema(self) ->dict:
        # 获取 Pydantic 生成的 schema
        schema = self.parameters.model_json_schema()
        
        # 清理 schema (OpenAI 不需要 'title' 字段，移除以节省 Token)
        if "title" in schema:
            del schema["title"]
        for field in schema.get("properties", {}).values():
            if "title" in field:
                del field["title"]

        # 组装成 OpenAI 需要的格式
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema
        }
    }


    def __call__(self,parameters:dict):
        #检查参数是否合法
        try:
            self.parameters.model_validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}")
        return self.run(parameters)