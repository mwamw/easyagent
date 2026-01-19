"""
文档类定义
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    文档类
    
    表示一个文本文档片段，包含内容和元数据。
    
    Attributes:
        page_content: 文档内容
        metadata: 文档元数据（来源、页码等）
    """
    page_content: str = Field(description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    
    def __str__(self) -> str:
        return self.page_content
    
    def __repr__(self) -> str:
        content_preview = self.page_content[:50] + "..." if len(self.page_content) > 50 else self.page_content
        return f"Document(content='{content_preview}', metadata={self.metadata})"
    
    @property
    def source(self) -> Optional[str]:
        """获取文档来源"""
        return self.metadata.get("source")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建文档"""
        return cls(
            page_content=data.get("page_content", ""),
            metadata=data.get("metadata", {})
        )
