"""Bounded recipe evaluation: lexical references and typed template expansion."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import re

from .parser import parse
from .syntax import (
    Array, Call, Collection, Component, Declaration, Definition, Expression, LanguageError,
    Literal, Module, Object, Reference, Span, TYPE_KINDS, VALUE_TYPES,
)
from .values import DeferredPath, Environment, ParameterHole, Recipe, Value, resolve_path


RESERVED = frozenset({"path", "env", "directory", "true", "false", "none", "null", "disabled", "return"})


class Evaluator:
    def __init__(self) -> None:
        self.modules: dict[Path, Module] = {}
        self.symbols: dict[Path, dict[str, tuple[Path, Declaration | Definition]]] = {}
        self.cache: dict[tuple[Path, str], Value] = {}
        self.loading: set[Path] = set()
        self.resolving: list[tuple[Path, str]] = []
        self.calls: list[tuple[Path, str]] = []
        self.steps = 0

    def load(self, file: Path, span: Span | None = None) -> Module:
        location = span or Span(file, 1, 1)
        file = Path(resolve_path(str(file), Path.cwd(), location))
        if file in self.loading:
            raise LanguageError(location, "Cyclic source import", code="EA201")
        if file in self.modules:
            return self.modules[file]
        if len(self.modules) + len(self.loading) >= 128 or len(self.loading) >= 32:
            raise LanguageError(location, "Import graph exceeds the file/depth limit", code="EA201")
        if file.suffix != ".ea":
            raise LanguageError(location, "Imports must reference local .ea source files", code="EA201")
        try:
            with file.open(encoding="utf-8") as stream:
                text = stream.read(2_000_001)
        except (OSError, UnicodeError):
            raise LanguageError(location, "Cannot read source file", code="EA201") from None
        module = parse(text, source=file)
        self.loading.add(file)
        table: dict[str, tuple[Path, Declaration | Definition]] = {}

        def bind(name: str, target: tuple[Path, Declaration | Definition], at: Span) -> None:
            if name in RESERVED or name in table:
                raise LanguageError(at, f"Duplicate or reserved declaration name {name}", code="EA201")
            table[name] = target

        try:
            for declaration in module.declarations:
                bind(declaration.name, (file, declaration), declaration.span)
            for imported in module.imports:
                if "://" in imported.source:
                    raise LanguageError(imported.span, "Network imports are not supported", code="EA201")
                dependency = self.load(file.parent / imported.source, imported.span)
                for original, alias in imported.names:
                    if original not in self.symbols[dependency.file]:
                        raise LanguageError(imported.span, f"Imported declaration {original} does not exist", code="EA201")
                    bind(alias, self.symbols[dependency.file][original], imported.span)
            self.symbols[file] = table
            self.modules[file] = module
            return module
        finally:
            self.loading.remove(file)

    def lookup(self, file: Path, name: str, span: Span) -> tuple[Path, Declaration | Definition]:
        if name not in self.symbols[file]:
            raise LanguageError(span, f"Unknown name {name}", code="EA201")
        return self.symbols[file][name]

    def resolve(self, file: Path, name: str, span: Span) -> Value:
        owner, declaration = self.lookup(file, name, span)
        if isinstance(declaration, Definition):
            raise LanguageError(span, f"{name} is a template; call it with arguments", code="EA202")
        key = (owner, declaration.name)
        if key in self.resolving:
            raise LanguageError(span, "Cyclic component reference", code="EA201")
        if key not in self.cache:
            self.resolving.append(key)
            try:
                value = self.evaluate(declaration.value, owner, {})
                self.require(value, declaration.kind, span)
                self.cache[key] = value
            finally:
                self.resolving.pop()
        return self.cache[key]

    @staticmethod
    def require(value: Value, kind: str, span: Span) -> None:
        if value.kind != kind and not (kind == "Number" and value.kind == "Integer"):
            raise LanguageError(span, f"Expected {kind}, received {value.kind}", code="EA202")

    def fields(self, obj: Object, file: Path, local: dict[str, Value]) -> dict[str, Value]:
        values = {}
        for field in obj.fields:
            if field.name in values:
                raise LanguageError(field.span, f"Duplicate field {field.name}", code="EA202")
            values[field.name] = self.evaluate(field.value, file, local)
        return values

    def evaluate(self, expression: Expression, file: Path, local: dict[str, Value]) -> Value:
        self.steps += 1
        if self.steps > 100000:
            raise LanguageError(expression.span, "Template expansion exceeds the evaluation limit", code="EA202")
        if isinstance(expression, Literal):
            kind = {str: "String", bool: "Boolean", int: "Integer", float: "Number", type(None): "None"}[type(expression.value)]
            return Value(kind, expression.value, expression.span)
        if isinstance(expression, Reference):
            value = local[expression.name] if expression.name in local else self.resolve(file, expression.name, expression.span)
            return replace(value, span=expression.span)
        if isinstance(expression, Object):
            return Value("Object", self.fields(expression, file, local), expression.span)
        if isinstance(expression, Array):
            return Value("List", tuple(self.evaluate(v, file, local) for v in expression.items), expression.span)
        if isinstance(expression, Component):
            fields = self.fields(expression.body, file, local)
            return Value(expression.kind, Recipe(expression.kind, expression.implementation, fields, expression.span), expression.span)
        if isinstance(expression, Collection):
            values = []
            for item in expression.calls:
                call = item.call
                if expression.kind == "skills":
                    if call.name != "directory" or len(call.arguments) != 1 or call.arguments[0].name is not None:
                        raise LanguageError(call.span, "Skill entries must use directory(path(...))", code="EA202")
                    value = self.evaluate(call.arguments[0].value, file, local)
                    self.require(value, "Path", call.span)
                    if not item.enabled:
                        raise LanguageError(call.span, "Disable Skills with skills = none", code="EA202")
                    values.append(value)
                else:
                    fields = self.named_arguments(call, file, local)
                    values.append(Value("tool", Recipe("tool", call.name, fields, call.span, item.enabled), call.span))
            return Value(expression.kind, tuple(values), expression.span)
        if isinstance(expression, Call):
            if expression.name in ("env", "path"):
                return self.builtin(expression, file, local)
            owner, definition = self.lookup(file, expression.name, expression.span)
            if not isinstance(definition, Definition):
                raise LanguageError(expression.span, "Only templates and language builtins are callable", code="EA202")
            args = self.named_arguments(expression, file, local)
            return self.expand(owner, definition, args, expression.span)
        raise AssertionError("Unhandled expression")

    def named_arguments(self, call: Call, file: Path, local: dict[str, Value]) -> dict[str, Value]:
        result = {}
        for argument in call.arguments:
            if argument.name is None:
                raise LanguageError(argument.span, "Use named arguments: name = value", code="EA202")
            if argument.name in result:
                raise LanguageError(argument.span, f"Duplicate argument {argument.name}", code="EA202")
            result[argument.name] = self.evaluate(argument.value, file, local)
        return result

    def builtin(self, call: Call, file: Path, local: dict[str, Value]) -> Value:
        if not call.arguments or call.arguments[0].name is not None:
            raise LanguageError(call.span, f"{call.name} requires a positional string argument", code="EA202")
        value = self.evaluate(call.arguments[0].value, file, local)
        self.require(value, "String", call.span)
        if call.name == "path":
            if len(call.arguments) != 1:
                raise LanguageError(call.span, "path accepts exactly one argument", code="EA202")
            if isinstance(value.data, ParameterHole):
                return Value("Path", value.data, call.span)
            if isinstance(value.data, Environment):
                return Value("Path", DeferredPath(value.data, file.parent), call.span)
            assert isinstance(value.data, str)
            return Value("Path", resolve_path(value.data, file.parent, call.span), call.span)
        if not isinstance(value.data, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value.data) is None:
            raise LanguageError(call.span, "env requires a literal portable environment variable name", code="EA202")
        default = None
        if len(call.arguments) == 2 and call.arguments[1].name == "default":
            item = self.evaluate(call.arguments[1].value, file, local)
            if item.kind != "String" or not isinstance(item.data, str):
                raise LanguageError(item.span, "env default must be literal text", code="EA202")
            default = item.data
        elif len(call.arguments) != 1:
            raise LanguageError(call.span, "Use env(\"NAME\", default = \"optional value\")", code="EA202")
        return Value("String", Environment(value.data, default), call.span)

    def expand(self, file: Path, definition: Definition, supplied: dict[str, Value], span: Span, *, symbolic: bool = False) -> Value:
        key = (file, definition.name)
        if key in self.calls or len(self.calls) >= 64:
            raise LanguageError(span, "Recursive or excessively nested template expansion", code="EA202")
        known = set()
        local = {}
        self.calls.append(key)
        try:
            for parameter in definition.parameters:
                if parameter.name in known or parameter.name in RESERVED:
                    raise LanguageError(parameter.span, "Duplicate or reserved parameter name", code="EA202")
                known.add(parameter.name)
                kind = TYPE_KINDS.get(parameter.type_name, parameter.type_name)
                if parameter.type_name not in TYPE_KINDS and parameter.type_name not in VALUE_TYPES:
                    raise LanguageError(parameter.span, f"Unknown parameter type {parameter.type_name}", code="EA202")
                # Validate defaults even when an argument overrides them.
                default = self.evaluate(parameter.default, file, local) if parameter.default is not None else None
                if default is not None:
                    self.require(default, kind, parameter.span)
                value = supplied.get(parameter.name, default)
                if symbolic:
                    value = Value(kind, ParameterHole(parameter.name), parameter.span)
                if value is None:
                    raise LanguageError(span, f"Missing argument {parameter.name}", code="EA202")
                self.require(value, kind, span)
                local[parameter.name] = value
            extra = set(supplied) - known
            if extra:
                raise LanguageError(span, "Unknown arguments: " + ", ".join(sorted(extra)), code="EA202")
            if definition.return_type not in TYPE_KINDS and definition.return_type not in VALUE_TYPES:
                raise LanguageError(definition.span, f"Unknown return type {definition.return_type}", code="EA202")
            result = self.evaluate(definition.body, file, local)
            self.require(result, TYPE_KINDS.get(definition.return_type, definition.return_type), definition.body.span)
            return result
        finally:
            self.calls.pop()
