"""Permission context models and helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .types import PermissionMode, PermissionRule


class PermissionContext(BaseModel):
    mode: PermissionMode = Field(default=PermissionMode.DEFAULT)
    rules: list[PermissionRule] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_rule(self, rule: PermissionRule) -> None:
        self.rules.append(rule)

    def extend_rules(self, rules: list[PermissionRule]) -> None:
        self.rules.extend(list(rules or []))

    def clear_rules(self) -> None:
        self.rules.clear()

    def set_mode(self, mode: PermissionMode | str) -> None:
        self.mode = PermissionMode(mode)

    def export_state(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def restore_state(self, state: dict[str, Any] | None) -> None:
        restored = PermissionContext.model_validate(state or {})
        self.mode = restored.mode
        self.rules = list(restored.rules)
        self.metadata = dict(restored.metadata)

