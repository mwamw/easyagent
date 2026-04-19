"""Permission rule storage with explicit sources and priorities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .types import PermissionRule


_DEFAULT_SOURCE_PRIORITIES: dict[str, int] = {
    "system": 10,
    "workspace": 20,
    "project": 30,
    "session": 40,
    "user": 50,
    "runtime": 60,
}


class PermissionStore(BaseModel):
    sources: dict[str, list[PermissionRule]] = Field(default_factory=dict)
    source_priorities: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def _normalize_source(source: str | None) -> str:
        value = str(source or "session").strip()
        return value or "session"

    def _priority_for_source(self, source: str) -> int:
        return int(
            self.source_priorities.get(
                source,
                _DEFAULT_SOURCE_PRIORITIES.get(source, 100),
            )
        )

    def _normalized_rule(self, rule: PermissionRule | dict[str, Any], source: str) -> PermissionRule:
        candidate = PermissionRule.model_validate(rule)
        if candidate.source != source:
            candidate = candidate.model_copy(update={"source": source})
        return candidate

    def set_source_rules(
        self,
        source: str,
        rules: list[PermissionRule | dict[str, Any]],
        *,
        priority: int | None = None,
    ) -> None:
        normalized_source = self._normalize_source(source)
        self.sources[normalized_source] = [
            self._normalized_rule(rule, normalized_source)
            for rule in list(rules or [])
        ]
        if priority is not None or normalized_source not in self.source_priorities:
            self.source_priorities[normalized_source] = int(
                priority if priority is not None else self._priority_for_source(normalized_source)
            )

    def add_rule(
        self,
        rule: PermissionRule | dict[str, Any],
        *,
        source: str | None = None,
        priority: int | None = None,
    ) -> PermissionRule:
        normalized_source = self._normalize_source(source)
        candidate = self._normalized_rule(rule, normalized_source)
        bucket = list(self.sources.get(normalized_source, []))
        bucket.append(candidate)
        self.sources[normalized_source] = bucket
        if priority is not None or normalized_source not in self.source_priorities:
            self.source_priorities[normalized_source] = int(
                priority if priority is not None else self._priority_for_source(normalized_source)
            )
        return candidate

    def extend_rules(
        self,
        rules: list[PermissionRule | dict[str, Any]],
        *,
        source: str | None = None,
        priority: int | None = None,
    ) -> list[PermissionRule]:
        normalized_source = self._normalize_source(source)
        added: list[PermissionRule] = []
        for rule in list(rules or []):
            added.append(self.add_rule(rule, source=normalized_source, priority=priority))
        return added

    def clear_source(self, source: str) -> None:
        normalized_source = self._normalize_source(source)
        self.sources.pop(normalized_source, None)
        self.source_priorities.pop(normalized_source, None)

    def clear(self) -> None:
        self.sources.clear()
        self.source_priorities.clear()

    def get_rules(self) -> list[PermissionRule]:
        ordered: list[tuple[int, int, PermissionRule]] = []
        for source, rules in self.sources.items():
            priority = self._priority_for_source(source)
            for index, rule in enumerate(list(rules or [])):
                ordered.append((priority, index, rule))
        ordered.sort(key=lambda item: (item[0], item[1]))
        return [rule for _, _, rule in ordered]

    def export_state(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def restore_state(self, state: dict[str, Any] | None) -> None:
        restored = PermissionStore.model_validate(state or {})
        self.sources = {
            source: list(rules)
            for source, rules in restored.sources.items()
        }
        self.source_priorities = dict(restored.source_priorities)
        self.metadata = dict(restored.metadata)


__all__ = ["PermissionStore"]
