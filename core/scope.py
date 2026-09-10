"""Read-only identity scope shared by runtime extensions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeScope:
    session_id: str
    workspace_root: str = ""
    user_id: str | None = None
