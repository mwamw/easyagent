"""Models for runtime team management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TeamHandle:
    team_id: str
    name: str
    description: str = ""
    member_agent_ids: tuple[str, ...] = ()
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "teamId": self.team_id,
            "name": self.name,
            "description": self.description,
            "memberAgentIds": list(self.member_agent_ids),
            "memberCount": len(self.member_agent_ids),
            "createdAt": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TeamHandle":
        data = dict(payload or {})
        return cls(
            team_id=str(data.get("teamId") or ""),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            member_agent_ids=tuple(str(item) for item in list(data.get("memberAgentIds") or [])),
            created_at=float(data.get("createdAt") or 0.0),
            metadata=dict(data.get("metadata") or {}),
        )
