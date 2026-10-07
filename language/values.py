"""Checked recipe values, never live components or executable Python."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

from .syntax import LanguageError, Span


@dataclass(frozen=True)
class Environment:
    name: str
    default: str | None = None


@dataclass(frozen=True)
class DeferredPath:
    value: Environment
    directory: Path


@dataclass(frozen=True)
class ParameterHole:
    """Symbolic parameter for checking a template without creating an instance."""
    name: str


@dataclass(frozen=True)
class Recipe:
    kind: str
    implementation: str
    fields: dict[str, Value]
    span: Span
    enabled: bool = True


@dataclass(frozen=True)
class Value:
    kind: str
    data: str | int | float | bool | None | Environment | DeferredPath | ParameterHole | Recipe | dict[str, Value] | tuple[Value, ...]
    span: Span


Plain: TypeAlias = str | int | float | bool | None | list["Plain"] | dict[str, "Plain"]


def resolve_path(value: str, directory: Path, span: Span) -> str:
    try:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else directory / path).resolve())
    except (OSError, ValueError, RuntimeError):
        raise LanguageError(span, "Invalid or unresolvable filesystem path", code="EA202") from None


def plain(value: Value, *, environ: Mapping[str, str] | None = None, active: bool = True) -> Plain:
    """environ=None is static checking, not environment lookup."""
    data = value.data
    if isinstance(data, ParameterHole):
        return {"String": "<parameter>", "Path": "/<parameter>", "Integer": 1,
                "Number": 1.0, "Boolean": True}.get(value.kind)
    if isinstance(data, (Environment, DeferredPath)):
        env = data.value if isinstance(data, DeferredPath) else data
        if environ is None:
            return "<environment>"
        if not active:
            return ""
        result = environ.get(env.name, env.default)
        if result is None:
            raise LanguageError(value.span, f"Required environment variable {env.name} is missing", code="EA301")
        if not isinstance(result, str):
            raise LanguageError(value.span, "Environment values must be text", code="EA301")
        if isinstance(data, DeferredPath):
            result = resolve_path(result, data.directory, value.span)
        return result
    if isinstance(data, dict):
        enabled = data.get("enabled")
        active = active and not (enabled is not None and enabled.data is False)
        return {key: plain(item, environ=environ, active=active) for key, item in data.items()}
    if isinstance(data, tuple):
        return [plain(item, environ=environ, active=active) for item in data]
    if isinstance(data, Recipe):
        raise LanguageError(value.span, "A component reference cannot be used as a configuration value", code="EA202")
    return data


def contains_symbol(value: Value) -> bool:
    if isinstance(value.data, (Environment, DeferredPath, ParameterHole)):
        return True
    if isinstance(value.data, dict):
        return any(contains_symbol(v) for v in value.data.values())
    if isinstance(value.data, tuple):
        return any(contains_symbol(v) for v in value.data)
    return False


def at_location(value: Value, location: tuple[str | int, ...]) -> Value:
    for part in location:
        if isinstance(value.data, dict) and part in value.data:
            value = value.data[part]
        elif isinstance(value.data, tuple) and isinstance(part, int) and 0 <= part < len(value.data):
            value = value.data[part]
        else:
            break
    return value
