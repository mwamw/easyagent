"""Per-Agent runtime state and framework-owned execution facilities."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from threading import Lock
from _thread import LockType
from typing import Any, Iterator, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .events import RuntimeEvent, RuntimeEventBus
    from .hooks import HookManager
    from .interruptions import BaseToolInterruptController
    from .invocation import AgentInvocationState
    from .metamessage import MetaMessageManager


@dataclass(eq=False)
class AgentRuntime:
    """One lifetime per Agent. Modules receive this same instance at assembly."""

    workspace_root: str
    allowed_roots: tuple[str, ...] = ()
    agent_id: str = field(default_factory=lambda: f"agent_{uuid4().hex}")
    session_id: str = field(default_factory=lambda: f"session_{uuid4().hex}")
    mcp_servers: tuple[str, ...] = ()
    execution_mode: str = "execute"
    permission_mode: str = "ask_for_approval"
    plan_allowed_actions: tuple[str, ...] = ()
    current_task_id: str | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    closed: bool = field(default=False, init=False)
    stop_requested: bool = field(default=False, init=False)
    stop_reason: str | None = field(default=None, init=False)
    active_invocation: AgentInvocationState | None = field(default=None, init=False)
    last_invocation: AgentInvocationState | None = field(default=None, init=False)
    invocation_lock: LockType = field(default_factory=Lock, init=False, repr=False)
    _event_bus: RuntimeEventBus = field(init=False, repr=False)
    _metamessages: MetaMessageManager = field(init=False, repr=False)
    _hooks: HookManager = field(init=False, repr=False)
    interruptions: BaseToolInterruptController = field(init=False, repr=False)
    _owner: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from .events import RuntimeEventBus
        from .hooks import HookManager
        from .interruptions import InMemoryToolInterruptController
        from .metamessage import MetaMessageManager

        self.workspace_root = os.path.abspath(self.workspace_root)
        self.allowed_roots = tuple(os.path.abspath(root) for root in self.allowed_roots) or (self.workspace_root,)
        self.metadata = deepcopy(self.metadata)
        self.metadata.setdefault("agentId", self.agent_id)
        self._event_bus = RuntimeEventBus()
        self._metamessages = MetaMessageManager()
        self._hooks = HookManager()
        self.interruptions = InMemoryToolInterruptController()
        self.event_bus.subscribe(self._dispatch_metamessage)

    @property
    def event_bus(self) -> RuntimeEventBus:
        return self._event_bus

    @property
    def metamessages(self) -> MetaMessageManager:
        return self._metamessages

    @property
    def hooks(self) -> HookManager:
        return self._hooks

    def claim(self, owner: object) -> None:
        if self._owner is not None and self._owner is not owner:
            raise RuntimeError("An AgentRuntime cannot be shared between Agent instances")
        self.ensure_idle()
        self._owner = owner

    def _dispatch_metamessage(self, event: RuntimeEvent) -> None:
        self.metamessages.publish(event.type.value, event.data)

    def ensure_idle(self) -> None:
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        if self.invocation_lock.locked():
            raise RuntimeError("Operation requires an idle Agent")

    def begin_invocation(self, query: str, *, stream: bool) -> AgentInvocationState:
        from .invocation import AgentInvocationPhase, AgentInvocationState

        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        if not self.invocation_lock.acquire(blocking=False):
            raise RuntimeError("An Agent supports one active invocation at a time")
        state = AgentInvocationState(f"invoke_{uuid4().hex}", query, stream,
                                     phase=AgentInvocationPhase.STARTED)
        self.active_invocation = state
        self.last_invocation = state
        return state

    def end_invocation(self) -> None:
        if self.active_invocation is not None:
            self.active_invocation = None
            self.invocation_lock.release()

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Reserve the runtime during session I/O or other exclusive changes."""
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        if not self.invocation_lock.acquire(blocking=False):
            raise RuntimeError("Operation requires an idle Agent")
        try:
            yield
        finally:
            self.invocation_lock.release()

    def request_stop(self, reason: str = "") -> None:
        self.stop_requested = True
        self.stop_reason = reason.strip() or "Agent stop requested"

    def clear_stop_request(self) -> None:
        self.stop_requested = False
        self.stop_reason = None

    def stop_reason_if_requested(self) -> str | None:
        return self.stop_reason if self.stop_requested else None

    def set_execution_mode(self, mode: str, *, allowed_actions: tuple[str, ...] = ()) -> None:
        self.execution_mode = mode
        self.plan_allowed_actions = tuple(allowed_actions) if mode == "plan" else ()

    def derive(self, *, workspace_root: str | None = None,
               allowed_roots: tuple[str, ...] | None = None,
               agent_id: str | None = None, session_id: str | None = None,
               worktree_path: str | None = None, worktree_branch: str | None = None,
               metadata: dict[str, Any] | None = None, execution_mode: str = "execute",
               permission_mode: str | None = None, current_task_id: str | None = None) -> AgentRuntime:
        identity = agent_id or f"agent_{uuid4().hex}"
        details = deepcopy(self.metadata)
        details.update(deepcopy(metadata or {}))
        details.update(agentId=identity, parentAgentId=self.agent_id)
        return AgentRuntime(
            agent_id=identity, session_id=session_id or f"session_{uuid4().hex}",
            workspace_root=workspace_root or self.workspace_root,
            allowed_roots=allowed_roots if allowed_roots is not None else self.allowed_roots,
            execution_mode=execution_mode, permission_mode=permission_mode or self.permission_mode,
            current_task_id=current_task_id, worktree_path=worktree_path,
            worktree_branch=worktree_branch, metadata=details,
            mcp_servers=tuple(self.mcp_servers), reasoning=deepcopy(self.reasoning),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize environment data only; never serialize live infrastructure."""
        return {
            "agentId": self.agent_id, "sessionId": self.session_id,
            "workspaceRoot": self.workspace_root, "allowedRoots": list(self.allowed_roots),
            "mcpServers": list(self.mcp_servers), "executionMode": self.execution_mode,
            "permissionMode": self.permission_mode, "currentTaskId": self.current_task_id,
            "worktreePath": self.worktree_path, "worktreeBranch": self.worktree_branch,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> AgentRuntime | None:
        data = dict(payload or {})
        if not data.get("workspaceRoot"):
            return None
        return cls(
            workspace_root=data["workspaceRoot"], allowed_roots=tuple(data.get("allowedRoots") or ()),
            agent_id=data.get("agentId") or f"agent_{uuid4().hex}",
            session_id=data.get("sessionId") or f"session_{uuid4().hex}",
            mcp_servers=tuple(data.get("mcpServers") or ()),
            execution_mode=data.get("executionMode") or "execute",
            permission_mode=data.get("permissionMode") or "ask_for_approval",
            current_task_id=data.get("currentTaskId"), worktree_path=data.get("worktreePath"),
            worktree_branch=data.get("worktreeBranch"), metadata=data.get("metadata") or {},
        )

    def close(self) -> None:
        self.ensure_idle()
        self.closed = True
        self.hooks.clear()
        self.event_bus.close()
        self.metamessages.close()
