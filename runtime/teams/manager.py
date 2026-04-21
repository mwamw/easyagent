"""Team registry for multi-agent collaboration."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional
from uuid import uuid4

from .models import TeamHandle


class TeamManager:
    def __init__(self, *, agent_runtime: Any | None = None):
        self._lock = threading.RLock()
        self._teams: dict[str, TeamHandle] = {}
        self._team_names: dict[str, str] = {}
        self.agent_runtime = None
        self.last_restore_report: Optional[dict[str, Any]] = None
        if agent_runtime is not None:
            self.bind_agent_runtime(agent_runtime)

    def bind_agent_runtime(self, agent_runtime: Any) -> None:
        self.agent_runtime = agent_runtime

    def create_team(
        self,
        *,
        name: str,
        description: str = "",
        member_agent_ids: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        team_id: Optional[str] = None,
    ) -> TeamHandle:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("name 不能为空。")
        with self._lock:
            if clean_name in self._team_names:
                raise ValueError(f"团队名称已存在: {clean_name}")
            handle = TeamHandle(
                team_id=team_id or f"team_{uuid4().hex[:12]}",
                name=clean_name,
                description=str(description or ""),
                member_agent_ids=tuple(member_agent_ids or ()),
                created_at=time.time(),
                metadata=dict(metadata or {}),
            )
            self._teams[handle.team_id] = handle
            self._team_names[handle.name] = handle.team_id
        return handle

    def get_team(self, identifier: str) -> TeamHandle:
        with self._lock:
            handle = self._teams.get(identifier)
            if handle is not None:
                return handle
            team_id = self._team_names.get(identifier)
            if team_id is not None and team_id in self._teams:
                return self._teams[team_id]
        raise KeyError(f"团队不存在: {identifier}")

    def list_teams(self) -> list[TeamHandle]:
        with self._lock:
            return list(self._teams.values())

    def add_member(self, identifier: str, agent_id: str) -> TeamHandle:
        handle = self.get_team(identifier)
        members = list(handle.member_agent_ids)
        if agent_id not in members:
            members.append(agent_id)
        updated = TeamHandle(
            team_id=handle.team_id,
            name=handle.name,
            description=handle.description,
            member_agent_ids=tuple(members),
            created_at=handle.created_at,
            metadata=dict(handle.metadata),
        )
        with self._lock:
            self._teams[handle.team_id] = updated
        return updated

    def remove_member(self, identifier: str, agent_id: str) -> TeamHandle:
        handle = self.get_team(identifier)
        updated = TeamHandle(
            team_id=handle.team_id,
            name=handle.name,
            description=handle.description,
            member_agent_ids=tuple(member for member in handle.member_agent_ids if member != agent_id),
            created_at=handle.created_at,
            metadata=dict(handle.metadata),
        )
        with self._lock:
            self._teams[handle.team_id] = updated
        return updated

    def delete_team(self, identifier: str) -> TeamHandle:
        handle = self.get_team(identifier)
        with self._lock:
            self._teams.pop(handle.team_id, None)
            self._team_names.pop(handle.name, None)
        return handle

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            teams = [handle.to_dict() for handle in self._teams.values()]
        return {
            "version": 1,
            "teams": teams,
        }

    def restore_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(state or {})
        teams = [TeamHandle.from_dict(item) for item in list(data.get("teams") or [])]
        with self._lock:
            self._teams = {handle.team_id: handle for handle in teams}
            self._team_names = {handle.name: handle.team_id for handle in teams if handle.name}
        report = {
            "status": "restored",
            "restoredItems": [handle.team_id for handle in teams if handle.team_id],
            "degradedItems": [],
            "missingItems": [],
            "metadata": {"teamCount": len(teams)},
            "issues": [],
        }
        self.last_restore_report = report
        return report


__all__ = ["TeamManager"]
