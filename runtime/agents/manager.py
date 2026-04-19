"""Unified runtime manager for subagents, handles, and mailboxes."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from Tool.runtime import SubagentManager, SubagentRequest
from runtime.context import ExecutionContext

from .models import AgentHandle, MailboxMessage


class AgentRuntimeManager:
    def __init__(
        self,
        *,
        agent_factory: Callable[[SubagentRequest], Any],
        storage_dir: Optional[str] = None,
        max_background_tasks: int = 4,
        subagent_manager: Optional[SubagentManager] = None,
        team_manager: Any | None = None,
    ):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".easyagent-agents"))
        self.subagent_manager = subagent_manager or SubagentManager(
            agent_factory=agent_factory,
            storage_dir=self.storage_dir,
            max_background_tasks=max_background_tasks,
        )
        self._lock = threading.RLock()
        self._requests: dict[str, SubagentRequest] = {}
        self._contexts: dict[str, ExecutionContext] = {}
        self._mailboxes: dict[str, list[MailboxMessage]] = {}
        self._team_assignments: dict[str, str] = {}
        self.team_manager = None
        if team_manager is not None:
            self.bind_team_manager(team_manager)

    def bind_team_manager(self, team_manager: Any) -> None:
        self.team_manager = team_manager
        bind = getattr(team_manager, "bind_agent_runtime", None)
        if callable(bind):
            bind(self)

    def _normalize_context(
        self,
        request: SubagentRequest,
        execution_context: Optional[ExecutionContext],
    ) -> ExecutionContext:
        if execution_context is not None:
            return execution_context
        workspace_root = os.path.abspath(request.workspace_root or os.getcwd())
        return ExecutionContext(
            workspace_root=workspace_root,
            allowed_roots=tuple(request.allowed_roots or (workspace_root,)),
            execution_mode="plan" if request.mode == "plan" else "execute",
            permission_mode=request.mode or "default",
            worktree_path=request.worktree_path,
            worktree_branch=request.worktree_branch,
            metadata=dict(request.metadata or {}),
        )

    def _remember_registration(
        self,
        agent_id: str,
        request: SubagentRequest,
        execution_context: ExecutionContext,
    ) -> None:
        with self._lock:
            self._requests[agent_id] = request
            self._contexts[agent_id] = execution_context
            self._mailboxes.setdefault(agent_id, [])

    def _attach_team_membership(self, agent_id: str, request: SubagentRequest) -> None:
        if self.team_manager is None or not request.team_name:
            return
        try:
            team = self.team_manager.get_team(request.team_name)
        except KeyError:
            return
        self.team_manager.add_member(team.team_id, agent_id)
        with self._lock:
            self._team_assignments[agent_id] = team.team_id

    def run(
        self,
        request: SubagentRequest,
        *,
        execution_context: Optional[ExecutionContext] = None,
        run_in_background: bool = False,
    ) -> AgentHandle:
        context = self._normalize_context(request, execution_context)
        snapshot = (
            self.subagent_manager.launch_background(request)
            if run_in_background
            else self.subagent_manager.run(request)
        )
        self._remember_registration(snapshot.agent_id, request, context)
        self._attach_team_membership(snapshot.agent_id, request)
        return self.get_handle(snapshot.agent_id)

    def get_handle(self, agent_id: str) -> AgentHandle:
        snapshot = self.subagent_manager.get_snapshot(agent_id)
        with self._lock:
            request = self._requests.get(agent_id)
            context = self._contexts.get(agent_id)
            mailbox = list(self._mailboxes.get(agent_id, ()))
            team_id = self._team_assignments.get(agent_id)
        if request is None or context is None:
            raise KeyError(f"子 agent 运行时上下文不存在: {agent_id}")
        return AgentHandle(
            agent_id=snapshot.agent_id,
            status=snapshot.status,
            description=snapshot.description,
            prompt=snapshot.prompt,
            output_file=snapshot.output_file,
            workspace_root=os.path.abspath(request.workspace_root or context.workspace_root),
            allowed_roots=tuple(request.allowed_roots or context.allowed_roots),
            execution_context=context,
            agent_type=snapshot.agent_type,
            name=snapshot.name,
            team_name=snapshot.team_name,
            team_id=team_id,
            mode=snapshot.mode,
            isolation=snapshot.isolation,
            worktree_path=snapshot.worktree_path,
            worktree_branch=snapshot.worktree_branch,
            started_at=snapshot.started_at,
            finished_at=snapshot.finished_at,
            content=snapshot.content,
            error=snapshot.error,
            total_duration_ms=snapshot.total_duration_ms,
            total_tool_use_count=snapshot.total_tool_use_count,
            total_tokens=snapshot.total_tokens,
            usage=dict(snapshot.usage),
            mailbox=mailbox,
            metadata=dict(request.metadata or {}),
        )

    def list_handles(self) -> list[AgentHandle]:
        snapshots = self.subagent_manager.list_snapshots()
        return [self.get_handle(snapshot.agent_id) for snapshot in snapshots]

    def list_mailbox(self, agent_id: str) -> list[MailboxMessage]:
        self.get_handle(agent_id)
        with self._lock:
            return list(self._mailboxes.get(agent_id, ()))

    def send_message(
        self,
        *,
        recipient_type: str,
        recipient_id: str,
        content: str,
        sender_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[MailboxMessage]:
        text = str(content).strip()
        if not text:
            raise ValueError("content 不能为空。")

        recipient_type = str(recipient_type).strip().lower()
        if recipient_type == "agent":
            target_ids = [recipient_id]
        elif recipient_type == "team":
            if self.team_manager is None:
                raise ValueError("当前未绑定 TeamManager，无法向团队发送消息。")
            team = self.team_manager.get_team(recipient_id)
            target_ids = list(team.member_agent_ids)
            if not target_ids:
                raise ValueError(f"团队没有成员: {team.name}")
        else:
            raise ValueError(f"不支持的 recipient_type: {recipient_type}")

        deliveries: list[MailboxMessage] = []
        base_metadata = dict(metadata or {})
        created_at = time.time()
        for target_id in target_ids:
            self.get_handle(target_id)
            message = MailboxMessage(
                message_id=f"msg_{uuid4().hex[:12]}",
                sender_id=sender_id,
                recipient_type="agent",
                recipient_id=target_id,
                content=text,
                created_at=created_at,
                metadata={
                    **base_metadata,
                    "originalRecipientType": recipient_type,
                    "originalRecipientId": recipient_id,
                },
            )
            with self._lock:
                self._mailboxes.setdefault(target_id, []).append(message)
            deliveries.append(message)
        return deliveries

    def close(self) -> None:
        self.subagent_manager.close()


__all__ = ["AgentRuntimeManager"]
