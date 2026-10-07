"""Source-located syntax for EasyAgent Language; no runtime dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class Span:
    file: Path
    line: int
    column: int


class LanguageError(ValueError):
    def __init__(self, span: Span, message: str, *, code: str = "EA100") -> None:
        self.span, self.message, self.code = span, message, code
        # Do not include source lines: they can contain literal credentials.
        rendered = f"{span.file}:{span.line}:{span.column}: {code}: {message}"
        super().__init__("".join(c if ord(c) >= 32 and not 127 <= ord(c) <= 159
                                else f"\\x{ord(c):02x}" for c in rendered))


@dataclass(frozen=True)
class Literal:
    value: str | int | float | bool | None
    span: Span


@dataclass(frozen=True)
class Reference:
    name: str
    span: Span


@dataclass(frozen=True)
class Argument:
    name: str | None
    value: Expression
    span: Span


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[Argument, ...]
    span: Span


@dataclass(frozen=True)
class Field:
    name: str
    value: Expression
    span: Span


@dataclass(frozen=True)
class Object:
    fields: tuple[Field, ...]
    span: Span


@dataclass(frozen=True)
class Array:
    items: tuple[Expression, ...]
    span: Span


@dataclass(frozen=True)
class Component:
    kind: str
    implementation: str
    body: Object
    span: Span


@dataclass(frozen=True)
class ToolCall:
    call: Call
    enabled: bool = True


@dataclass(frozen=True)
class Collection:
    kind: str
    calls: tuple[ToolCall, ...]
    span: Span


Expression: TypeAlias = Literal | Reference | Call | Object | Array | Component | Collection


@dataclass(frozen=True)
class Declaration:
    kind: str
    name: str
    value: Expression
    span: Span


@dataclass(frozen=True)
class Parameter:
    name: str
    type_name: str
    default: Expression | None
    span: Span


@dataclass(frozen=True)
class Definition:
    name: str
    parameters: tuple[Parameter, ...]
    return_type: str
    body: Expression
    span: Span


@dataclass(frozen=True)
class Import:
    names: tuple[tuple[str, str], ...]
    source: str
    span: Span


@dataclass(frozen=True)
class Run:
    agent: str
    ui: str | None
    options: Object
    span: Span


@dataclass(frozen=True)
class Module:
    file: Path
    declarations: tuple[Declaration | Definition, ...]
    imports: tuple[Import, ...]
    run: Run | None


KINDS = frozenset({
    "model", "context", "permission", "memory", "prompt", "executor", "plan",
    "tasks", "session", "observability", "hooks", "callbacks", "codeintel",
    "worktree", "multi_agent", "agent",
})
TYPE_KINDS = {
    "Model": "model", "Context": "context", "Permission": "permission", "Memory": "memory",
    "Prompt": "prompt", "Executor": "executor", "Plan": "plan", "Tasks": "tasks",
    "Session": "session", "Observability": "observability", "Hooks": "hooks",
    "Callbacks": "callbacks", "CodeIntel": "codeintel", "Worktree": "worktree",
    "MultiAgent": "multi_agent", "Agent": "agent",
}
VALUE_TYPES = frozenset({"String", "Path", "Integer", "Number", "Boolean"})
