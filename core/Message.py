from typing import Optional, Any, Literal, overload
from datetime import datetime
from typing_extensions import override
from pydantic import BaseModel, Field

MessageRole = Literal['system', 'user', 'assistant', 'tool','function']

class Message(BaseModel):
    role: MessageRole
    content: str
    time: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}
    
    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class UserMessage(Message):
    role: MessageRole = "user"  # 设置默认值
    
    def __init__(self, content: str, **kwargs):
        super().__init__(role="user", content=content, **kwargs)


class AssistantMessage(Message):
    role: MessageRole = "assistant"
    
    def __init__(self, content: str, **kwargs):
        super().__init__(role="assistant", content=content, **kwargs)


class SystemMessage(Message):
    role: MessageRole = "system"
    
    def __init__(self, content: str, **kwargs):
        super().__init__(role="system", content=content, **kwargs)


class ToolMessage(Message):
    role: MessageRole = "tool"
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # 工具名称 - Google API需要
    
    def __init__(self, content: str, tool_call_id: Optional[str] = None, name: Optional[str] = None, **kwargs):
        # 把所有字段传给Pydantic
        kwargs['tool_call_id'] = tool_call_id
        kwargs['name'] = name
        super().__init__(role="tool", content=content, **kwargs)

    @override
    def to_dict(self):
        result = {"role": self.role, "content": self.content, "tool_call_id": self.tool_call_id}
        if self.name:
            result["name"] = self.name  # Google API需要name字段
        return result

class FunctionMessage(Message):
    role: MessageRole = "function"
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # 工具名称 - Google API需要
    
    def __init__(self, content: str, tool_call_id: Optional[str] = None, name: Optional[str] = None, **kwargs):
        # 把所有字段传给Pydantic
        kwargs['tool_call_id'] = tool_call_id
        kwargs['name'] = name
        super().__init__(role="function", content=content, **kwargs)

    @override
    def to_dict(self):
        result = {"role": self.role, "content": self.content, "tool_call_id": self.tool_call_id}
        if self.name:
            result["name"] = self.name  # Google API需要name字段
        return result