"""Policy data and explicit file loading for the default permission engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..models import PermissionBehavior


@dataclass(frozen=True)
class PolicyRule:
    behavior: PermissionBehavior | str
    tool_name: str = "*"
    matcher: Mapping[str, Any] = field(default_factory=dict)
    source: str = "session"
    description: str = ""
    pattern: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "behavior", PermissionBehavior(self.behavior))
        if not self.tool_name or not isinstance(self.tool_name, str):
            raise ValueError("PolicyRule.tool_name must be a non-empty string")
        if not isinstance(self.matcher, Mapping):
            raise TypeError("PolicyRule.matcher must be a mapping")
        object.__setattr__(self, "matcher", deepcopy(dict(self.matcher)))
        if self.pattern is not None and not self.pattern.strip():
            raise ValueError("PolicyRule.pattern must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name, "behavior": self.behavior.value,
            "matcher": deepcopy(dict(self.matcher)), "source": self.source,
            "description": self.description, "pattern": self.pattern,
        }

    @classmethod
    def from_value(cls, value: PolicyRule | Mapping[str, Any]) -> PolicyRule:
        return cls(**(value.to_dict() if isinstance(value, cls) else dict(value)))


@dataclass
class PolicySet:
    rules: list[PolicyRule] = field(default_factory=list)
    revision: int = 0

    def __post_init__(self) -> None:
        self.rules = [PolicyRule.from_value(rule) for rule in self.rules]

    def extend(self, rules: Iterable[PolicyRule]) -> None:
        self.rules.extend(PolicyRule.from_value(rule) for rule in rules)
        self.revision += 1

    def clear(self, source: str | None = None) -> None:
        self.rules = [rule for rule in self.rules if source is not None and rule.source != source]
        self.revision += 1

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "rules": [rule.to_dict() for rule in self.rules]}

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None) -> PolicySet:
        payload = dict(value or {})
        return cls(rules=payload.get("rules", []), revision=int(payload.get("revision", 0)))


def load_policy_file(path: str | Path, *, source: str | None = None) -> PolicySet:
    """Read only a caller-selected file. Invalid policy never silently allows access."""
    location = Path(path).expanduser()
    try:
        data = json.loads(location.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PolicySet()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load permission policy {location}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Permission policy {location} must contain a JSON object")
    payload = data.get("permissions", data)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid permissions object in {location}")
    rules = []
    for raw in payload.get("rules", []):
        rule = PolicyRule.from_value(raw)
        fields = rule.to_dict()
        if source:
            fields["source"] = source
        rules.append(PolicyRule(**fields))
    for behavior in ("deny", "ask", "allow"):
        entries = payload.get(behavior, [])
        if not isinstance(entries, list):
            raise ValueError(f"permissions.{behavior} must be a list in {location}")
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"permissions.{behavior} entries must be non-empty strings")
            rules.append(PolicyRule(behavior, pattern=entry, source=source or str(location)))
    return PolicySet(rules)


def policy_locations(*, policy_directory=None, policy_paths=()) -> tuple[Path, ...]:
    paths = []
    if policy_directory is not None:
        directory = Path(policy_directory).expanduser()
        paths.extend((directory / "settings.json", directory / "settings.local.json"))
    paths.extend(Path(path).expanduser() for path in policy_paths or ())
    return tuple(dict.fromkeys(path.resolve() for path in paths))
