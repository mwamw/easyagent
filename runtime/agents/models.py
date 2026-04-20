"""Public models for the unified agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from runtime.context import ExecutionContext


@dataclass(slots=True)
class MailboxMessage:
    message_id: str
    sender_id: Optional[str]
    recipient_type: str
    recipient_id: str
    content: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messageId": self.message_id,
            "senderId": self.sender_id,
            "recipientType": self.recipient_type,
            "recipientId": self.recipient_id,
            "content": self.content,
            "createdAt": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MailboxMessage":
        data = dict(payload or {})
        return cls(
            message_id=str(data.get("messageId") or ""),
            sender_id=data.get("senderId"),
            recipient_type=str(data.get("recipientType") or "agent"),
            recipient_id=str(data.get("recipientId") or ""),
            content=str(data.get("content") or ""),
            created_at=float(data.get("createdAt") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class AgentHandle:
    agent_id: str
    status: str
    description: str
    prompt: str
    output_file: str
    workspace_root: str
    allowed_roots: tuple[str, ...]
    execution_context: ExecutionContext
    agent_type: Optional[str] = None
    name: Optional[str] = None
    team_name: Optional[str] = None
    team_id: Optional[str] = None
    mode: Optional[str] = None
    isolation: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    started_at: float = 0.0
    finished_at: Optional[float] = None
    content: str = ""
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    total_duration_ms: int = 0
    total_tool_use_count: int = 0
    total_tokens: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    mailbox: list[MailboxMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "status": self.status,
            "description": self.description,
            "prompt": self.prompt,
            "outputFile": self.output_file,
            "workspaceRoot": self.workspace_root,
            "allowedRoots": list(self.allowed_roots),
            "executionContext": self.execution_context.to_dict(),
            "agentType": self.agent_type,
            "name": self.name,
            "teamName": self.team_name,
            "teamId": self.team_id,
            "mode": self.mode,
            "isolation": self.isolation,
            "worktreePath": self.worktree_path,
            "worktreeBranch": self.worktree_branch,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "content": self.content,
            "error": self.error,
            "stopReason": self.stop_reason,
            "totalDurationMs": self.total_duration_ms,
            "totalToolUseCount": self.total_tool_use_count,
            "totalTokens": self.total_tokens,
            "usage": dict(self.usage),
            "mailbox": [message.to_dict() for message in self.mailbox],
            "mailboxCount": len(self.mailbox),
            "metadata": dict(self.metadata),
        }

    def to_tool_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["content"] = (
            [{"type": "text", "text": self.content}]
            if self.content
            else []
        )
        return payload


@dataclass(slots=True)
class BackgroundAgentHandle(AgentHandle):
    is_background: bool = True
    can_wait: bool = True
    can_stop: bool = True
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = AgentHandle.to_dict(self)
        payload.update(
            {
                "isBackground": self.is_background,
                "canWait": self.can_wait,
                "canStop": self.can_stop,
                "stopRequested": self.stop_requested,
            }
        )
        return payload
