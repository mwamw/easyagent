"""Optional structural protocols for session participants, not an AgentModule base."""
from typing import Any, Protocol, TypedDict, runtime_checkable
from .report import ComponentRestoreReport

SessionStateData = dict[str, Any] | None


class RestoreReportPayload(TypedDict, total=False):
    status: str
    restoredItems: list[str]
    degradedItems: list[str]
    missingItems: list[str]
    metadata: dict[str, Any]
    issues: list[dict[str, Any]]


@runtime_checkable
class SessionState(Protocol):
    def export_state(self) -> SessionStateData: ...
    def restore_state(self, state: SessionStateData) -> ComponentRestoreReport | RestoreReportPayload | None: ...


@runtime_checkable
class ValidatableSessionState(Protocol):
    def validate_state(self, state: SessionStateData) -> None: ...


@runtime_checkable
class ResettableSessionState(Protocol):
    def reset_session_state(self) -> None: ...


@runtime_checkable
class ForkableSessionState(Protocol):
    def fork_state(self, state: SessionStateData) -> SessionStateData: ...
