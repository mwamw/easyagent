"""Recursive-descent parser for declarations, recipes, and local imports."""
from __future__ import annotations

from pathlib import Path

from .lexer import Token, tokenize
from .values import resolve_path
from .syntax import (
    Argument, Array, Call, Collection, Component, Declaration, Definition, Expression,
    Field, Import, KINDS, LanguageError, Literal, Module, Object, Parameter, Reference,
    Run, Span, ToolCall,
)


class Parser:
    def __init__(self, text: str, file: Path) -> None:
        self.tokens = tokenize(text, file)
        self.index = 0
        self.file = file

    @property
    def token(self) -> Token:
        return self.tokens[self.index]

    def take(self, kind: str, value: str | None = None) -> Token:
        token = self.token
        if token.kind != kind or (value is not None and token.value != value):
            raise LanguageError(token.span, f"Expected {value or kind}")
        self.index += 1
        return token

    def accept(self, kind: str, value: str | None = None) -> bool:
        if self.token.kind == kind and (value is None or self.token.value == value):
            self.index += 1
            return True
        return False

    def separators(self) -> None:
        while self.accept(";") or self.accept(","):
            pass

    def parse(self) -> Module:
        declarations, imports = [], []
        run = None
        while self.token.kind != "EOF":
            self.separators()
            if self.token.kind == "EOF":
                break
            start = self.take("NAME")
            if start.value == "import":
                names = []
                self.take("{")
                while not self.accept("}"):
                    name = str(self.take("NAME").value)
                    alias = str(self.take("NAME").value) if self.accept("NAME", "as") else name
                    names.append((name, alias))
                    if not self.accept(","):
                        self.take("}")
                        break
                self.take("NAME", "from")
                imports.append(Import(tuple(names), str(self.take("STRING").value), start.span))
            elif start.value == "define":
                name = str(self.take("NAME").value)
                self.take("(")
                params = []
                while not self.accept(")"):
                    p = self.take("NAME")
                    self.take(":")
                    type_name = str(self.take("NAME").value)
                    default = self.expression() if self.accept("=") else None
                    params.append(Parameter(str(p.value), type_name, default, p.span))
                    if not self.accept(","):
                        self.take(")")
                        break
                self.take("->")
                result_type = str(self.take("NAME").value)
                self.take("{")
                self.accept("NAME", "return")
                body = self.expression()
                self.separators()
                self.take("}")
                declarations.append(Definition(name, tuple(params), result_type, body, start.span))
            elif start.value == "run":
                if run is not None:
                    raise LanguageError(start.span, "Only one run declaration is allowed per source file")
                name = str(self.take("NAME").value)
                ui = str(self.take("NAME").value) if self.accept("NAME", "with") else None
                options = self.object() if self.token.kind == "{" else Object((), start.span)
                run = Run(name, ui, options, start.span)
            elif start.value in KINDS:
                name = str(self.take("NAME").value)
                value = self.expression() if self.accept("=") else self.component(start)
                declarations.append(Declaration(str(start.value), name, value, start.span))
            else:
                raise LanguageError(start.span, "Expected a component, agent, define, import, or run declaration")
        return Module(self.file, tuple(declarations), tuple(imports), run)

    def component(self, kind: Token) -> Component:
        implementation = str(self.take("NAME").value) if self.accept("NAME", "using") else "default"
        if kind.value == "agent" and implementation != "default":
            raise LanguageError(kind.span, "Agent composes components; it does not select an implementation")
        return Component(str(kind.value), implementation, self.object(agent=kind.value == "agent"), kind.span)

    def object(self, *, agent: bool = False) -> Object:
        span = self.take("{").span
        fields = []
        while not self.accept("}"):
            key = self.token
            if key.kind not in ("NAME", "STRING"):
                raise LanguageError(key.span, "Expected a field name or closing brace")
            self.index += 1
            name = str(key.value)
            if self.accept("=") or self.accept(":"):
                value = self.expression()
            elif agent and name in KINDS - {"agent"}:
                value = self.component(key)
            elif agent and name in ("tools", "skills"):
                self.take("{")
                calls = []
                while not self.accept("}"):
                    enabled = not self.accept("NAME", "disabled")
                    call = self.expression()
                    if not isinstance(call, Call):
                        raise LanguageError(call.span, "Expected a named tool or directory(...) call")
                    calls.append(ToolCall(call, enabled))
                    self.separators()
                value = Collection(name, tuple(calls), key.span)
            else:
                raise LanguageError(self.token.span, "Expected '=' after field name")
            fields.append(Field(name, value, key.span))
            self.separators()
        return Object(tuple(fields), span)

    def expression(self) -> Expression:
        token = self.token
        if token.kind in ("STRING", "NUMBER"):
            self.index += 1
            return Literal(token.value, token.span)
        if self.accept("["):
            items = []
            while not self.accept("]"):
                items.append(self.expression())
                if not self.accept(","):
                    self.take("]")
                    break
            return Array(tuple(items), token.span)
        if token.kind == "{":
            return self.object()
        if token.kind == "NAME":
            self.index += 1
            if token.value in ("true", "false", "none", "null"):
                return Literal({"true": True, "false": False, "none": None, "null": None}[token.value], token.span)
            if self.accept("("):
                args = []
                while not self.accept(")"):
                    start = self.token
                    name = None
                    if start.kind == "NAME" and self.tokens[self.index + 1].kind == "=":
                        name = str(self.take("NAME").value)
                        self.take("=")
                    args.append(Argument(name, self.expression(), start.span))
                    if not self.accept(","):
                        self.take(")")
                        break
                return Call(str(token.value), tuple(args), token.span)
            if token.value in KINDS and (self.token.kind == "{" or self.token.value == "using"):
                return self.component(token)
            return Reference(str(token.value), token.span)
        raise LanguageError(token.span, "Expected a literal, reference, collection, component, or function call")


def parse(text: str, *, source: str | Path = "<string>.ea") -> Module:
    file = Path(resolve_path(str(source), Path.cwd(), Span(Path(source), 1, 1)))
    if len(text) > 2_000_000:
        raise LanguageError(Span(file, 1, 1), "Source exceeds 2 million characters")
    try:
        return Parser(text, file).parse()
    except RecursionError:
        raise LanguageError(Span(file, 1, 1), "Syntax nesting is too deep") from None
