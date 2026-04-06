import hashlib
from Tool.BaseTool import Tool
from pydantic import BaseModel, Field

class HashParams(BaseModel):
    text: str = Field(description="需要计算哈希的明文字符串")

class HashCalculatorTool(Tool):
    def __init__(self):
        super().__init__("hash_calculator", "计算字符串的 SHA-256 哈希值", HashParams)
        
    def run(self, params: dict) -> str:
        text = params.get("text", "")
        # 计算 SHA-256 哈希值
        sha256_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f"  [Tool执行] 计算文本 '{text}' 的 SHA-256 结果为: {sha256_hash}")
        return sha256_hash

def get_tools():
    """暴露给 FolderSkillLoader 用于获取该目录涵盖的所有工具对象"""
    return [HashCalculatorTool()]
