"""
多 Agent 协作 — Agent 间消息格式

AgentMessage 是多 Agent 系统中所有通信的标准载体。
它记录了消息的发送者、接收者、内容、类型和元数据，
为编排器提供完整的审计链与调试能力。
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


MessageType = Literal["task", "result", "handoff", "feedback", "system"]


@dataclass
class AgentMessage:
    """
    Agent 间通信消息
    
    Attributes:
        sender:    发送者 Agent 名称（"user" 表示用户，"orchestrator" 表示编排器）
        receiver:  接收者 Agent 名称（"all" 表示广播）
        content:   消息正文
        msg_type:  消息类型
                   - "task":     任务指派
                   - "result":   任务结果
                   - "handoff":  任务交接
                   - "feedback": 反馈/评审
                   - "system":   系统消息
        timestamp: 创建时间
        metadata:  扩展数据（如 token_count, duration_ms, error 等）
    """
    sender: str
    receiver: str
    content: str
    msg_type: MessageType = "task"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.msg_type.upper()}] {self.sender} → {self.receiver}: {self.content[:80]}{'…' if len(self.content) > 80 else ''}"

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于日志 / JSON 输出）"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "msg_type": self.msg_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
