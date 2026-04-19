"""Permission context models and helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .store import PermissionStore
from .types import PermissionMode, PermissionRule


class PermissionContext(BaseModel):
    mode: PermissionMode = Field(default=PermissionMode.DEFAULT)
    rules: list[PermissionRule] = Field(default_factory=list)
    store: PermissionStore = Field(default_factory=PermissionStore)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.store.sources:
            self.rules = self.store.get_rules()
            return
        if self.rules:
            self.store.set_source_rules("session", self.rules)
            self.rules = self.store.get_rules()

    def sync_rules_from_store(self) -> None:
        self.rules = self.store.get_rules()

    def iter_rules(self) -> list[PermissionRule]:
        return list(self.rules or self.store.get_rules())

    def add_rule(
        self,
        rule: PermissionRule,
        *,
        source: str | None = None,
        priority: int | None = None,
    ) -> None:
        self.store.add_rule(rule, source=source, priority=priority)
        self.sync_rules_from_store()

    def extend_rules(
        self,
        rules: list[PermissionRule],
        *,
        source: str | None = None,
        priority: int | None = None,
    ) -> None:
        self.store.extend_rules(rules, source=source, priority=priority)
        self.sync_rules_from_store()

    def set_source_rules(
        self,
        source: str,
        rules: list[PermissionRule],
        *,
        priority: int | None = None,
    ) -> None:
        self.store.set_source_rules(source, rules, priority=priority)
        self.sync_rules_from_store()

    def clear_rules(self, *, source: str | None = None) -> None:
        if source is None:
            self.store.clear()
        else:
            self.store.clear_source(source)
        self.sync_rules_from_store()

    def set_mode(self, mode: PermissionMode | str) -> None:
        self.mode = PermissionMode(mode)

    def export_state(self) -> dict[str, Any]:
        self.sync_rules_from_store()
        return self.model_dump(mode="python")

    def restore_state(self, state: dict[str, Any] | None) -> None:
        restored = PermissionContext.model_validate(state or {})
        self.mode = restored.mode
        self.store.restore_state(restored.store.export_state())
        self.metadata = dict(restored.metadata)

        if self.store.sources:
            self.sync_rules_from_store()
            return

        legacy_rules = list(restored.rules or [])
        if legacy_rules:
            self.store.set_source_rules("session", legacy_rules)
            self.sync_rules_from_store()
            return

        self.rules = []
