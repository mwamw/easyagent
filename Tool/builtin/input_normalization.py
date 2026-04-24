"""Helpers for normalizing common LLM-produced tool inputs."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse


_MARKDOWN_LINK_RE = re.compile(r"^\[[^\]]+\]\((?P<target>.+)\)$", re.S)
_CODE_FENCE_RE = re.compile(r"^```(?:[\w.+-]+)?\s*\n(?P<body>.*)\n```$", re.S)
_LINE_FRAGMENT_RE = re.compile(r"^(?P<path>.+?)#L\d+(?:-L\d+)?$")
_LINE_SUFFIX_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?$")


def _unwrap_wrappers(text: str) -> str:
    current = text.strip()
    for _ in range(4):
        changed = False
        code_match = _CODE_FENCE_RE.match(current)
        if code_match:
            current = code_match.group("body").strip()
            changed = True

        link_match = _MARKDOWN_LINK_RE.match(current)
        if link_match:
            current = link_match.group("target").strip()
            changed = True

        for left, right in (('"', '"'), ("'", "'"), ("`", "`"), ("<", ">")):
            if len(current) >= 2 and current.startswith(left) and current.endswith(right):
                current = current[1:-1].strip()
                changed = True
                break
        if not changed:
            break
    return current.strip()


def normalize_generic_input(value: object) -> str:
    """Strip obvious wrappers like quotes, markdown links, angle brackets and code fences."""
    return _unwrap_wrappers(str(value or ""))


def normalize_path_input(value: object) -> str:
    """Normalize path-like inputs that may include wrappers or line suffixes."""
    text = normalize_generic_input(value)
    if text.startswith("file://"):
        parsed = urlparse(text)
        text = unquote(parsed.path or "").strip()

    fragment_match = _LINE_FRAGMENT_RE.match(text)
    if fragment_match:
        text = fragment_match.group("path").strip()

    suffix_match = _LINE_SUFFIX_RE.match(text)
    if suffix_match:
        candidate = suffix_match.group("path").strip()
        if "/" in candidate or "." in candidate or candidate.startswith("~"):
            text = candidate

    return text.strip()


def normalize_path_with_line_hint(value: object) -> tuple[str, int | None]:
    """Normalize a path-like input and optionally extract a trailing line hint."""
    text = normalize_generic_input(value)
    if text.startswith("file://"):
        parsed = urlparse(text)
        text = unquote(parsed.path or "").strip()

    line_hint: int | None = None

    fragment_match = _LINE_FRAGMENT_RE.match(text)
    if fragment_match:
        line_fragment_match = re.search(r"#L(?P<line>\d+)", text)
        text = fragment_match.group("path").strip()
        if line_fragment_match:
            line_hint = int(line_fragment_match.group("line"))

    suffix_match = _LINE_SUFFIX_RE.match(text)
    if suffix_match:
        candidate = suffix_match.group("path").strip()
        if "/" in candidate or "." in candidate or candidate.startswith("~"):
            text = candidate
            line_hint = int(suffix_match.group("line"))

    return text.strip(), line_hint


def normalize_url_input(value: object) -> str:
    """Normalize URL-like inputs that may be wrapped by markdown or quotes."""
    return normalize_generic_input(value)


def normalize_domain_filter(value: object) -> str:
    """Normalize a host/domain filter that may be passed as a full URL."""
    text = normalize_generic_input(value).lower()
    if not text:
        return ""

    if "://" in text:
        parsed = urlparse(text)
        text = parsed.hostname or ""

    text = text.strip()
    if text.startswith("*."):
        text = text[2:]
    if "/" in text:
        text = text.split("/", 1)[0]
    if ":" in text and not text.startswith("["):
        host, _, maybe_port = text.rpartition(":")
        if host and maybe_port.isdigit():
            text = host
    return text.strip(".")


def glob_pattern_hints(*, original_pattern: str, normalized_pattern: str) -> list[str]:
    """Return actionable hints when a glob pattern looks suspicious."""
    hints: list[str] = []
    stripped_original = str(original_pattern or "").strip()

    if stripped_original and stripped_original != normalized_pattern:
        hints.append("已自动去掉外层引号、Markdown 包装或代码围栏。")

    if re.search(r"\{[^}]+,[^}]+\}", normalized_pattern):
        hints.append("Python glob 不支持 `*.{js,ts}` 这种 brace expansion；请拆成多个模式分别搜索。")

    if re.search(r"[\^\$\|\(\)]", normalized_pattern):
        hints.append("当前 pattern 看起来更像 regex；Glob 只支持 shell 风格通配符，如 `*`、`?`、`**`。")

    if "**" not in normalized_pattern and "/" not in normalized_pattern:
        hints.append("如果你想递归搜索子目录，请显式使用 `**/`，例如 `**/*.py`。")

    return hints


def grep_pattern_hints(
    *,
    original_pattern: str,
    normalized_pattern: str,
    file_glob: str | None = None,
) -> list[str]:
    """Return actionable hints when a grep pattern looks suspicious."""
    hints: list[str] = []
    stripped_original = str(original_pattern or "").strip()

    if stripped_original and stripped_original != normalized_pattern:
        hints.append("已自动去掉外层引号、Markdown 包装或代码围栏。")

    if normalized_pattern.startswith("*.") or normalized_pattern.startswith("**/") or re.search(r"\.[A-Za-z0-9]{1,6}$", normalized_pattern):
        hints.append("当前 pattern 看起来像文件名或 glob；如果你想限制文件范围，请使用 `glob` 参数，而不是把它写进内容 pattern。")

    if file_glob and str(file_glob).strip() != str(file_glob):
        hints.append("已自动整理 `glob` 参数中的首尾空白。")

    return hints


def format_no_match_message(title: str, *, scope_label: str, scope_value: str, query_label: str, query_value: str, hints: list[str] | None = None) -> str:
    """Build a richer no-match message that helps the model self-correct."""
    lines = [title, f"{scope_label}: {scope_value}", f"{query_label}: {query_value}"]
    normalized_hints = [hint for hint in (hints or []) if hint]
    if normalized_hints:
        lines.append("排查建议:")
        lines.extend(f"- {hint}" for hint in normalized_hints)
    return "\n".join(lines)
