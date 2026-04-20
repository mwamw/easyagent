"""Unified runtime manager for subagents, handles, and mailboxes."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from Tool.runtime import SubagentManager, SubagentRequest
from runtime.context import ExecutionContext
from runtime.teams.manager import TeamManager

from .models import AgentHandle, BackgroundAgentHandle, MailboxMessage


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
        self._background_agent_ids: set[str] = set()
        self.team_manager = None
        if team_manager is not None:
            self.bind_team_manager(team_manager)

    def bind_team_manager(self, team_manager: Any) -> None:
        if not team_manager:
            self.team_manager = TeamManager(agent_runtime=self)
        else:
            self.team_manager = team_manager
        bind = getattr(self.team_manager, "bind_agent_runtime", None)
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
        *,
        background: bool = False,
    ) -> None:
        with self._lock:
            self._requests[agent_id] = request
            self._contexts[agent_id] = execution_context
            self._mailboxes.setdefault(agent_id, [])
            if background:
                self._background_agent_ids.add(agent_id)
            else:
                self._background_agent_ids.discard(agent_id)

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
        self._remember_registration(
            snapshot.agent_id,
            request,
            context,
            background=run_in_background,
        )
        self._attach_team_membership(snapshot.agent_id, request)
        return self.get_handle(snapshot.agent_id)

    def get_handle(self, agent_id: str) -> AgentHandle:
        snapshot = self.subagent_manager.get_snapshot(agent_id)
        with self._lock:
            request = self._requests.get(agent_id)
            context = self._contexts.get(agent_id)
            mailbox = list(self._mailboxes.get(agent_id, ()))
            team_id = self._team_assignments.get(agent_id)
            is_background = agent_id in self._background_agent_ids
        if request is None or context is None:
            raise KeyError(f"子 agent 运行时上下文不存在: {agent_id}")
        handle_cls = BackgroundAgentHandle if is_background else AgentHandle
        base_kwargs = dict(
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
            stop_reason=snapshot.stop_reason,
            total_duration_ms=snapshot.total_duration_ms,
            total_tool_use_count=snapshot.total_tool_use_count,
            total_tokens=snapshot.total_tokens,
            usage=dict(snapshot.usage),
            mailbox=mailbox,
            metadata=dict(request.metadata or {}),
        )
        if handle_cls is BackgroundAgentHandle:
            return handle_cls(
                **base_kwargs,
                stop_requested=snapshot.status == "stop_requested",
                can_wait=True,
                can_stop=snapshot.status not in {"completed", "error", "stopped", "cancelled", "interrupted"},
            )
        return handle_cls(**base_kwargs)

    def list_handles(
        self,
        *,
        status: Optional[str] = None,
        team_id: Optional[str] = None,
        team_name: Optional[str] = None,
        current_task_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[AgentHandle]:
        snapshots = self.subagent_manager.list_snapshots()
        handles = [self.get_handle(snapshot.agent_id) for snapshot in snapshots]
        if status is not None:
            handles = [handle for handle in handles if handle.status == status]
        if team_id is not None:
            handles = [handle for handle in handles if handle.team_id == team_id]
        if team_name is not None:
            handles = [handle for handle in handles if handle.team_name == team_name]
        if current_task_id is not None:
            handles = [
                handle for handle in handles
                if handle.execution_context.current_task_id == current_task_id
            ]
        if limit is not None:
            handles = handles[: max(int(limit), 0)]
        return handles

    def wait(self, agent_id: str, *, timeout_ms: Optional[int] = None) -> AgentHandle:
        self.subagent_manager.wait(agent_id, timeout_ms=timeout_ms)
        return self.get_handle(agent_id)

    def stop(
        self,
        agent_id: str,
        *,
        reason: str = "",
        wait: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> AgentHandle:
        self.subagent_manager.stop(
            agent_id,
            reason=reason,
            wait=wait,
            timeout_ms=timeout_ms,
        )
        return self.get_handle(agent_id)

    def delete_handle(
        self,
        agent_id: str,
        *,
        remove_output_file: bool = False,
        remove_mailbox: bool = True,
    ) -> AgentHandle:
        handle = self.get_handle(agent_id)
        self.subagent_manager.delete(agent_id, remove_output_file=remove_output_file)
        with self._lock:
            self._requests.pop(agent_id, None)
            self._contexts.pop(agent_id, None)
            self._background_agent_ids.discard(agent_id)
            team_id = self._team_assignments.pop(agent_id, None)
            if remove_mailbox:
                self._mailboxes.pop(agent_id, None)
        if team_id and self.team_manager is not None:
            try:
                self.team_manager.remove_member(team_id, agent_id)
            except Exception:
                pass
        return handle

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
        elif recipient_type == "task":
            target_ids = [
                handle.agent_id
                for handle in self.list_handles(current_task_id=recipient_id)
            ]
            if not target_ids:
                raise ValueError(f"没有 agent 绑定到任务: {recipient_id}")
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

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            requests = {agent_id: request.to_dict() for agent_id, request in self._requests.items()}
            contexts = {
                agent_id: context.to_dict()
                for agent_id, context in self._contexts.items()
            }
            mailboxes = {
                agent_id: [message.to_dict() for message in messages]
                for agent_id, messages in self._mailboxes.items()
            }
            team_assignments = dict(self._team_assignments)
            background_agent_ids = sorted(self._background_agent_ids)
        return {
            "version": 1,
            "storageDir": self.storage_dir,
            "subagents": self.subagent_manager.export_state(),
            "requests": requests,
            "contexts": contexts,
            "mailboxes": mailboxes,
            "teamAssignments": team_assignments,
            "backgroundAgentIds": background_agent_ids,
        }

    def restore_state(self, state: dict[str, Any] | None) -> None:
        data = dict(state or {})
        self.subagent_manager.restore_state(data.get("subagents"))
        with self._lock:
            self._requests = {
                agent_id: SubagentRequest.from_dict(payload)
                for agent_id, payload in dict(data.get("requests") or {}).items()
            }
            self._contexts = {
                agent_id: context
                for agent_id, payload in dict(data.get("contexts") or {}).items()
                if (context := ExecutionContext.from_dict(payload)) is not None
            }
            self._mailboxes = {
                agent_id: [
                    MailboxMessage.from_dict(message)
                    for message in list(messages or [])
                ]
                for agent_id, messages in dict(data.get("mailboxes") or {}).items()
            }
            self._team_assignments = {
                str(agent_id): str(team_id)
                for agent_id, team_id in dict(data.get("teamAssignments") or {}).items()
                if agent_id and team_id
            }
            self._background_agent_ids = {
                str(agent_id)
                for agent_id in list(data.get("backgroundAgentIds") or [])
                if agent_id
            }


__all__ = ["AgentRuntimeManager"]
