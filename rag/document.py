from pydantic import BaseModel
from typing import Dict, Any, Optional

class Document(BaseModel):
    document_id: Optional[str] 
    document_path: Optional[str] 
    content: str 
    metadata: Dict[str, Any] 
    document_type:str
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{content_preview}', metadata={self.metadata})"
    
    @property
    def source(self) -> Optional[str]:
        """获取文档来源"""
        return self.metadata.get("source")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "document_id": self.document_id,
            "document_path": self.document_path,
            "content": self.content,
            "metadata": self.metadata,
            "document_type":self.document_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建文档"""
        return cls(
            document_id=data.get("document_id",""),
            document_path=data.get("document_path",""),
            document_type=data.get("document_type","text"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {})
        )

class Document_Chunk(BaseModel):
    document_id:str
    document_path:str
    chunk_id:str
    content:str
    metadata:Dict[str,Any]
    chunk_index:int
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document_Chunk(content='{content_preview}', metadata={self.metadata})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "document_id": self.document_id,
            "document_path": self.document_path,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document_Chunk":
        """从字典创建文档块"""
        return cls(
            document_id=data.get("document_id",""),
            document_path=data.get("document_path",""),
            chunk_id=data.get("chunk_id",""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            chunk_index=data.get("chunk_index",0)
        )

    
