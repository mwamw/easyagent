"""Small non-executable lexer. Strings are JSON literals or raw triple quotes."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .syntax import LanguageError, Span


@dataclass(frozen=True)
class Token:
    kind: str
    value: str | int | float
    span: Span


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def tokenize(text: str, file: Path) -> tuple[Token, ...]:
    tokens: list[Token] = []
    position, line, column = 0, 1, 1

    def advance(fragment: str) -> None:
        nonlocal position, line, column
        position += len(fragment)
        count = fragment.count("\n")
        line += count
        column = len(fragment.rsplit("\n", 1)[-1]) + 1 if count else column + len(fragment)

    while position < len(text):
        span = Span(file, line, column)
        char = text[position]
        if char.isspace() or (position == 0 and char == "\ufeff"):
            advance(char)
            continue
        if char == "#" or text.startswith("//", position):
            end = text.find("\n", position)
            advance(text[position:end if end >= 0 else len(text)])
            continue
        if text.startswith('"""', position):
            end = text.find('"""', position + 3)
            if end < 0:
                raise LanguageError(span, "Unterminated multiline string")
            value = text[position + 3:end]
            advance(text[position:end + 3])
            tokens.append(Token("STRING", value, span))
        elif char == '"':
            try:
                value, end = json.JSONDecoder().raw_decode(text, position)
            except (ValueError, RecursionError):
                raise LanguageError(span, "Invalid quoted string; use JSON escapes") from None
            if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
                raise LanguageError(span, "Unpaired Unicode surrogate in string")
            advance(text[position:end])
            tokens.append(Token("STRING", value, span))
        elif text.startswith("->", position):
            tokens.append(Token("->", "->", span))
            advance("->")
        elif char in "{}[](),=:;":
            tokens.append(Token(char, char, span))
            advance(char)
        elif match := _IDENTIFIER.match(text, position):
            value = match.group()
            tokens.append(Token("NAME", value, span))
            advance(value)
        elif match := _NUMBER.match(text, position):
            raw = match.group()
            try:
                value = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                raise LanguageError(span, "Numeric literal is too large") from None
            if isinstance(value, float) and not math.isfinite(value):
                raise LanguageError(span, "Numeric literal must be finite")
            tokens.append(Token("NUMBER", value, span))
            advance(raw)
        else:
            raise LanguageError(span, "Unexpected character; expressions cannot contain executable code")
        if len(tokens) > 100000:
            raise LanguageError(span, "Source exceeds the 100000 token limit")
    tokens.append(Token("EOF", "", Span(file, line, column)))
    return tuple(tokens)
