"""Structured session restore reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RestoreIssue:
    component: str
    code: str
    message: str
    severity: str = "warning"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ComponentRestoreReport:
    name: str
    status: str = "restored"
    restored_items: list[str] = field(default_factory=list)
    degraded_items: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[RestoreIssue] = field(default_factory=list)

    def add_issue(
        self,
        *,
        code: str,
        message: str,
        severity: str = "warning",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(
            RestoreIssue(
                component=self.name,
                code=code,
                message=message,
                severity=severity,
                metadata=dict(metadata or {}),
            )
        )
        if severity in {"warning", "error"} and self.status == "restored":
            self.status = "degraded"
        if severity == "error":
            self.status = "failed"

    def extend_from_payload(self, payload: dict[str, Any] | None) -> None:
        data = dict(payload or {})
        payload_status = str(data.get("status") or "").strip()
        if payload_status:
            if self.status == "restored" or payload_status == "failed":
                self.status = payload_status
        self.restored_items.extend(str(item) for item in list(data.get("restoredItems") or []) if item)
        self.degraded_items.extend(str(item) for item in list(data.get("degradedItems") or []) if item)
        self.missing_items.extend(str(item) for item in list(data.get("missingItems") or []) if item)
        metadata = dict(data.get("metadata") or {})
        self.metadata.update(metadata)
        for issue in list(data.get("issues") or []):
            issue_data = dict(issue or {})
            self.add_issue(
                code=str(issue_data.get("code") or "restore_issue"),
                message=str(issue_data.get("message") or ""),
                severity=str(issue_data.get("severity") or "warning"),
                metadata=dict(issue_data.get("metadata") or {}),
            )
        if self.degraded_items and self.status == "restored":
            self.status = "degraded"
        if self.missing_items and self.status == "restored":
            self.status = "degraded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "restoredItems": list(self.restored_items),
            "degradedItems": list(self.degraded_items),
            "missingItems": list(self.missing_items),
            "metadata": dict(self.metadata),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(slots=True)
class SessionRestoreReport:
    session_id: str
    agent_type: str
    status: str = "restored"
    execution_context_restored: bool = False
    missing_tools: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    issues: list[RestoreIssue] = field(default_factory=list)
    components: dict[str, ComponentRestoreReport] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_component(self, name: str) -> ComponentRestoreReport:
        component = self.components.get(name)
        if component is None:
            component = ComponentRestoreReport(name=name)
            self.components[name] = component
        return component

    def add_issue(
        self,
        *,
        component: str,
        code: str,
        message: str,
        severity: str = "warning",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        issue = RestoreIssue(
            component=component,
            code=code,
            message=message,
            severity=severity,
            metadata=dict(metadata or {}),
        )
        self.issues.append(issue)
        self.ensure_component(component).issues.append(issue)
        if severity in {"warning", "error"} and self.status == "restored":
            self.status = "degraded"
        if severity == "error":
            self.status = "failed"

    def extend_component(self, name: str, payload: dict[str, Any] | None) -> None:
        component = self.ensure_component(name)
        component.extend_from_payload(payload)
        if component.status == "degraded" and self.status == "restored":
            self.status = "degraded"
        if component.status == "failed":
            self.status = "failed"

    def note_missing_tools(self, tool_names: list[str]) -> None:
        items = [str(name) for name in tool_names if name]
        if not items:
            return
        self.missing_tools.extend(items)
        self.add_issue(
            component="tools",
            code="missing_tools",
            message=f"恢复会话时缺少工具实现: {items}",
            metadata={"tools": items},
        )

    def note_missing_skills(self, skill_names: list[str]) -> None:
        items = [str(name) for name in skill_names if name]
        if not items:
            return
        self.missing_skills.extend(items)
        self.add_issue(
            component="skills",
            code="missing_skills",
            message=f"恢复会话时缺少 Skill 实现: {items}",
            metadata={"skills": items},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "agentType": self.agent_type,
            "status": self.status,
            "executionContextRestored": self.execution_context_restored,
            "missingTools": list(self.missing_tools),
            "missingSkills": list(self.missing_skills),
            "issues": [issue.to_dict() for issue in self.issues],
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
            "metadata": dict(self.metadata),
        }
