
from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Any, Type

class Tool(ABC):
    def __init__(self,name:str,description:str,parameters:Type[BaseModel]):
        self.name=name
        self.description=description
        self.parameters=parameters
    
    @abstractmethod
    def run(self,parameters:dict) -> Any:
        pass

    def get_openai_schema(self) ->dict:
        # 获取 Pydantic 生成的 schema
        schema = self.parameters.model_json_schema()
        defs = schema.pop("$defs", {})
        
        def resolve_schema(schema_node):
            if isinstance(schema_node, dict):
                if "$ref" in schema_node:
                    ref_key = schema_node["$ref"].split("/")[-1]
                    resolved = defs.get(ref_key, {}).copy()
                    for k, v in schema_node.items():
                        if k != "$ref" and k not in resolved:
                            resolved[k] = v
                    return resolve_schema(resolved)
                
                if "anyOf" in schema_node:
                    non_null_schemas = [s for s in schema_node["anyOf"] if s.get("type") != "null"]
                    if len(non_null_schemas) == 1:
                        merged = {k: v for k, v in schema_node.items() if k != "anyOf" and k != "default"}
                        resolved_child = resolve_schema(non_null_schemas[0])
                        for k, v in resolved_child.items():
                            merged[k] = v
                        return merged
                    else:
                        schema_node["anyOf"] = [resolve_schema(s) for s in schema_node["anyOf"]]
                
                # 清理 schema (移除 title)
                if "title" in schema_node:
                    del schema_node["title"]
                    
                for k, v in schema_node.items():
                    schema_node[k] = resolve_schema(v)
                    
            elif isinstance(schema_node, list):
                return [resolve_schema(item) for item in schema_node]
                
            return schema_node

        cleaned_schema = resolve_schema(schema)

        # 组装成 OpenAI 需要的格式
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": cleaned_schema
        }
    }


    def __call__(self,parameters:dict):
        #检查参数是否合法
        try:
            self.parameters.model_validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}")
        return self.run(parameters)