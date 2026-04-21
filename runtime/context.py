"""Shared execution context for agent runtime workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def _normalize_roots(values: Iterable[str] | None, fallback_root: str) -> tuple[str, ...]:
    roots = tuple(
        os.path.abspath(item)
        for item in (values or ())
        if item
    )
    return roots or (os.path.abspath(fallback_root),)


@dataclass(slots=True)
class ExecutionContext:
    workspace_root: str
    allowed_roots: tuple[str, ...]
    mcp_servers: tuple[str, ...] = field(default_factory=tuple)
    execution_mode: str = "execute"
    permission_mode: str = "default"
    current_task_id: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        *,
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        worktree_path: Optional[str] = None,
        worktree_branch: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        execution_mode: Optional[str] = None,
        permission_mode: Optional[str] = None,
        current_task_id: Optional[str] = None,
    ) -> "ExecutionContext":
        config = getattr(agent, "config", None)
        resolved_workspace = os.path.abspath(
            workspace_root
            or getattr(config, "workspace_root", None)
            or os.getcwd()
        )
        roots = _normalize_roots(
            allowed_roots
            or (getattr(config, "get_allowed_roots", lambda: [])() if config is not None else None),
            resolved_workspace,
        )
        mode_controller = getattr(agent, "mode_controller", None)
        permission_context = getattr(agent, "permission_context", None)
        mode_value = execution_mode or getattr(getattr(mode_controller, "mode", None), "value", None) or "execute"
        permission_value = (
            permission_mode
            or getattr(getattr(permission_context, "mode", None), "value", None)
            or "default"
        )
        task_id = current_task_id if current_task_id is not None else getattr(agent, "current_task_id", None)
        merged_metadata = dict(getattr(getattr(agent, "execution_context", None), "metadata", {}) or {})
        merged_metadata.update(dict(metadata or {}))

        visible_mcp_servers: list[str] = []
        tool_registry = getattr(agent, "tool_registry", None)
        if tool_registry is not None:
            list_surfaces = getattr(tool_registry, "list_runtime_surfaces", None)
            if callable(list_surfaces):
                try:
                    visible_mcp_servers.extend(str(name) for name in list_surfaces("mcp_manager").keys())
                except Exception:
                    pass
            if not visible_mcp_servers:
                try:
                    for tool in tool_registry.get_visible_tools(scope="all"):
                        spec = tool.get_spec()
                        server_name = str(spec.metadata.get("mcp_server") or "").strip()
                        if server_name:
                            visible_mcp_servers.append(server_name)
                except Exception:
                    pass
        normalized_mcp_servers = tuple(
            sorted({value for value in visible_mcp_servers if value})
        )
        return cls(
            workspace_root=resolved_workspace,
            allowed_roots=roots,
            mcp_servers=normalized_mcp_servers,
            execution_mode=str(mode_value),
            permission_mode=str(permission_value),
            current_task_id=task_id,
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            metadata=merged_metadata,
        )

    def copy_for_workspace(
        self,
        *,
        workspace_root: str,
        allowed_roots: Optional[Iterable[str]] = None,
        worktree_path: Optional[str] = None,
        worktree_branch: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        execution_mode: Optional[str] = None,
        permission_mode: Optional[str] = None,
        current_task_id: Optional[str] = None,
    ) -> "ExecutionContext":
        merged_metadata = dict(self.metadata)
        merged_metadata.update(dict(metadata or {}))
        resolved_workspace = os.path.abspath(workspace_root)
        return ExecutionContext(
            workspace_root=resolved_workspace,
            allowed_roots=_normalize_roots(allowed_roots or self.allowed_roots, resolved_workspace),
            mcp_servers=tuple(self.mcp_servers),
            execution_mode=execution_mode or self.execution_mode,
            permission_mode=permission_mode or self.permission_mode,
            current_task_id=current_task_id if current_task_id is not None else self.current_task_id,
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspaceRoot": self.workspace_root,
            "allowedRoots": list(self.allowed_roots),
            "mcpServers": list(self.mcp_servers),
            "executionMode": self.execution_mode,
            "permissionMode": self.permission_mode,
            "currentTaskId": self.current_task_id,
            "worktreePath": self.worktree_path,
            "worktreeBranch": self.worktree_branch,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ExecutionContext" | None:
        data = dict(payload or {})
        workspace_root = data.get("workspaceRoot")
        if not workspace_root:
            return None
        return cls(
            workspace_root=os.path.abspath(str(workspace_root)),
            allowed_roots=_normalize_roots(
                data.get("allowedRoots"),
                str(workspace_root),
            ),
            mcp_servers=tuple(
                str(item).strip()
                for item in list(data.get("mcpServers") or [])
                if str(item).strip()
            ),
            execution_mode=str(data.get("executionMode") or "execute"),
            permission_mode=str(data.get("permissionMode") or "default"),
            current_task_id=data.get("currentTaskId"),
            worktree_path=data.get("worktreePath"),
            worktree_branch=data.get("worktreeBranch"),
            metadata=dict(data.get("metadata") or {}),
        )
