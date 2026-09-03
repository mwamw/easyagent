"""Discovery and lazy loading for ``SKILL.md`` directories."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Iterable

import yaml

from .base import SkillManifest


SKILL_FILENAME = "SKILL.md"
_MAX_FRONTMATTER_CHARS = 64 * 1024


def discover_skill_files(directory: str | os.PathLike[str]) -> list[Path]:
    """Find a Skill directory or the direct child Skills in a collection."""

    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Skill path must be a directory: {root}")

    direct = root / SKILL_FILENAME
    if direct.is_file():
        return [direct.resolve(strict=True)]

    discovered = sorted(
        (
            child.joinpath(SKILL_FILENAME).resolve(strict=True)
            for child in root.iterdir()
            if child.is_dir() and child.joinpath(SKILL_FILENAME).is_file()
        ),
        key=lambda item: (item.parent.name, str(item)),
    )
    if not discovered:
        raise FileNotFoundError(
            f"No {SKILL_FILENAME} found in {root} or its direct child directories"
        )
    return discovered


def _read_frontmatter(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8-sig") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"{file_path} must begin with YAML frontmatter")
        lines: list[str] = []
        size = 0
        for line in handle:
            if line.strip() == "---":
                break
            size += len(line)
            if size > _MAX_FRONTMATTER_CHARS:
                raise ValueError(f"Skill frontmatter exceeds {_MAX_FRONTMATTER_CHARS} characters: {file_path}")
            lines.append(line)
        else:
            raise ValueError(f"Unclosed YAML frontmatter in {file_path}")

    payload = yaml.safe_load("".join(lines)) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Skill frontmatter must be a mapping: {file_path}")
    return dict(payload)


def _split_tool_entries(value: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    depth = 0
    for character in value:
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        if depth == 0 and (character == "," or character.isspace()):
            item = "".join(current).strip()
            if item:
                entries.append(item)
            current = []
            continue
        current.append(character)
    item = "".join(current).strip()
    if item:
        entries.append(item)
    return entries


def _string_list(value: Any, *, tool_entries: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = _split_tool_entries(value) if tool_entries else [value]
    elif isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        values = list(value)
    else:
        raise ValueError(f"Expected a string or list, got {type(value).__name__}")
    return [str(item).strip() for item in values if str(item).strip()]


def _expand_braces(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]
    expanded: list[str] = []
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    for alternative in match.group(1).split(","):
        expanded.extend(_expand_braces(prefix + alternative.strip() + suffix))
    return expanded


def _path_patterns(value: Any) -> list[str]:
    if value is None:
        return []
    values = (
        [value]
        if isinstance(value, str)
        else list(value)
        if isinstance(value, Iterable) and not isinstance(value, (dict, bytes))
        else None
    )
    if values is None:
        raise ValueError(f"Expected paths to be a string or list, got {type(value).__name__}")

    patterns: list[str] = []
    for raw_value in values:
        text = str(raw_value or "")
        current: list[str] = []
        depth = 0
        parts: list[str] = []
        for character in text:
            if character == "{":
                depth += 1
            elif character == "}":
                depth = max(0, depth - 1)
            if character == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(character)
        part = "".join(current).strip()
        if part:
            parts.append(part)
        for item in parts:
            normalized = item[:-3] if item.endswith("/**") else item
            patterns.extend(_expand_braces(normalized))

    normalized_patterns = [
        item.strip().replace("\\", "/")
        for item in patterns
        if item.strip()
    ]
    if normalized_patterns and all(item == "**" for item in normalized_patterns):
        return []
    return list(dict.fromkeys(normalized_patterns))


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off", ""}:
            return False
    return bool(value)


def load_skill_manifest(file_path: str | os.PathLike[str]) -> SkillManifest:
    """Parse only frontmatter and return the indexed Skill metadata."""

    path = Path(file_path).expanduser().resolve(strict=True)
    if path.name != SKILL_FILENAME or not path.is_file():
        raise ValueError(f"Skill definition must be a {SKILL_FILENAME} file: {path}")
    payload = _read_frontmatter(path)
    known_keys = {
        "name",
        "description",
        "when_to_use",
        "allowed-tools",
        "argument-hint",
        "context",
        "agent",
        "model",
        "paths",
        "disable-model-invocation",
    }
    return SkillManifest(
        name=payload.get("name", path.parent.name),
        description=payload.get("description", ""),
        when_to_use=payload.get("when_to_use", ""),
        directory=str(path.parent),
        file_path=str(path),
        allowed_tools=_string_list(
            payload.get("allowed-tools"),
            tool_entries=True,
        ),
        argument_hint=str(payload.get("argument-hint", "") or "").strip(),
        context=str(payload.get("context") or "inline").strip(),
        agent=str(payload["agent"]).strip() if payload.get("agent") is not None else None,
        model=str(payload["model"]).strip() if payload.get("model") is not None else None,
        paths=_path_patterns(payload.get("paths")),
        disable_model_invocation=_boolean(
            payload.get("disable-model-invocation", False)
        ),
        metadata={key: value for key, value in payload.items() if key not in known_keys},
    )


def load_skill_body(manifest: SkillManifest, args: str = "") -> str:
    """Read and render a Skill body at invocation time."""

    path = Path(manifest.file_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"{path} must begin with YAML frontmatter")
        for line in handle:
            if line.strip() == "---":
                break
        else:
            raise ValueError(f"Unclosed YAML frontmatter in {path}")
        body = handle.read().strip()

    if not body:
        raise ValueError(f"Skill body is empty: {path}")
    return body.replace("${SKILL_DIR}", manifest.directory).replace("$ARGUMENTS", str(args or ""))


__all__ = [
    "SKILL_FILENAME",
    "discover_skill_files",
    "load_skill_body",
    "load_skill_manifest",
]
