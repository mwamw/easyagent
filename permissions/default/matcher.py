"""Conservative tool-specific matching; never execute or evaluate shell input."""

from __future__ import annotations

import fnmatch
from pathlib import Path
import re
import shlex
from urllib.parse import urlparse

from ..models import PermissionRequest
from .policy import PolicyRule


MATCHER_KEYS = frozenset({
    "pattern", "risk", "risk_categories", "command", "command_prefixes", "commands",
    "path", "path_prefixes", "paths", "host", "hosts", "mcp_servers",
    "param_equals", "param_contains",
})


def wildcard(value: str, pattern: str) -> bool:
    return re.fullmatch(re.escape(pattern).replace(r"\*", ".*"), value, flags=re.S) is not None


def shell_segments(command: str) -> tuple[list[str], bool]:
    """Return simple commands and whether parsing leaves unclassified effects.

    Expansion, substitution, redirection, shell syntax and embedded interpreters
    must not be blessed by matching one harmless prefix.
    """
    uncertain = any(token in command for token in ("$", "`", "\n", "\r", "<", ">", "(", ")", "{", "}"))
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        parts, current = [], []
        for token in lexer:
            if token and all(char in ";&|<>" for char in token):
                if current:
                    parts.append(shlex.join(current))
                    current = []
                uncertain |= token not in {";", "&&", "||", "|"}
            else:
                current.append(token)
        if current:
            parts.append(shlex.join(current))
        return parts, uncertain or not parts
    except ValueError:
        return [], True


def request_paths(request: PermissionRequest) -> list[Path]:
    root = Path(request.context.workspace_root or ".").resolve()
    values = [request.arguments[key] for key in ("path", "file_path", "notebook_path", "cwd")
              if isinstance(request.arguments.get(key), str)]
    values.extend(item for item in request.arguments.get("paths", []) if isinstance(item, str))
    paths = []
    for value in values:
        path = Path(value).expanduser()
        paths.append((path if path.is_absolute() else root / path).resolve())
    return paths


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def path_matches(value: Path, pattern: str, root: Path) -> bool:
    pattern = str(pattern)
    if pattern.startswith("//"):
        target = Path("/" + pattern[2:])
    elif pattern.startswith("~/"):
        target = Path(pattern).expanduser()
    else:
        target = Path(pattern)
        if not target.is_absolute():
            target = root / target
    if "*" not in str(target) and "?" not in str(target) and "[" not in str(target):
        return within(value, target.resolve())
    # Preserve path separators: a single star must not grant sibling subtrees.
    expression = re.escape(str(target)).replace(r"\*\*/", "(?:.*/)?")
    expression = expression.replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.fullmatch(expression, str(value)) is not None


class RuleMatcher:
    @staticmethod
    def tool_names(request):
        names = {request.tool_name}
        alias = {"FileRead": "Read", "FileWrite": "Write", "FileEdit": "Edit"}.get(request.tool_name)
        if alias:
            names.add(alias)
        return names

    @staticmethod
    def host(request):
        value = str(request.arguments.get("url", request.arguments.get("host", "")))
        return (urlparse(value if "://" in value else "//" + value).hostname or "").lower().rstrip(".")

    @staticmethod
    def domain(host, expected):
        expected = str(expected).lower().rstrip(".")
        if expected.startswith("*."):
            return host.endswith(expected[1:])
        return host == expected

    def pattern_matches(self, pattern: str, request: PermissionRequest, *, allow: bool) -> bool:
        found = re.fullmatch(r"([^()]+)(?:\((.*)\))?", pattern, flags=re.S)
        if not found:
            return False
        tool_pattern, specifier = found.groups()
        names = self.tool_names(request)
        if not any(fnmatch.fnmatchcase(name, tool_pattern) for name in names):
            # Bare MCP server names denote all tools exposed by that server.
            if not (tool_pattern.startswith("mcp__") and request.tool_name.startswith(tool_pattern + "__")):
                return False
        if specifier is None or specifier == "*":
            return True
        if "Bash" in names or "shell" in request.descriptor.capabilities:
            command = str(request.arguments.get("command", ""))
            segments, uncertain = shell_segments(command)
            if allow and uncertain:
                return False
            specifier = specifier.replace(":*", " *")
            matches = [wildcard(segment, specifier) or (specifier.endswith(" *") and segment == specifier[:-2])
                       for segment in segments]
            return bool(matches) and (all(matches) if allow else any(matches))
        if specifier.startswith("domain:"):
            return self.domain(self.host(request), specifier[7:])
        if names & {"Read", "Write", "Edit", "FileRead", "FileWrite", "FileEdit", "Glob", "Grep", "NotebookEdit"}:
            paths = request_paths(request)
            root = Path(request.context.workspace_root or ".").resolve()
            matches = [path_matches(path, specifier, root) for path in paths]
            return bool(matches) and (all(matches) if allow else any(matches))
        if not allow and ":" in specifier:
            name, value = specifier.split(":", 1)
            actual = request.arguments.get(name.strip())
            return actual is not None and wildcard(str(actual).lower() if isinstance(actual, bool) else str(actual), value.strip())
        if request.tool_name == "Agent":
            return wildcard(str(request.arguments.get("subagent_type", "")), specifier)
        return False

    def matches(self, rule: PolicyRule, request: PermissionRequest) -> bool:
        if not any(fnmatch.fnmatchcase(name, rule.tool_name) for name in self.tool_names(request)):
            return False
        allow = rule.behavior.value == "allow"
        matcher = dict(rule.matcher)
        if set(matcher) - MATCHER_KEYS:
            raise ValueError("Unsupported permission matcher: " + ", ".join(sorted(set(matcher) - MATCHER_KEYS)))
        pattern = rule.pattern or matcher.get("pattern")
        if pattern and not self.pattern_matches(pattern, request, allow=allow):
            return False
        risks = matcher.get("risk") or matcher.get("risk_categories")
        if risks and not set(risks).intersection(request.descriptor.risk_categories):
            return False
        commands = matcher.get("command_prefixes") or matcher.get("commands")
        if matcher.get("command"):
            commands = [matcher["command"]]
        if commands:
            parts, uncertain = shell_segments(str(request.arguments.get("command", request.arguments.get("cmd", ""))))
            if allow and uncertain:
                return False
            matches = [any(part == prefix or part.startswith(str(prefix) + " ") for prefix in commands) for part in parts]
            if not matches or not (all(matches) if allow else any(matches)):
                return False
        prefixes = matcher.get("path_prefixes") or matcher.get("paths")
        if matcher.get("path"):
            prefixes = [matcher["path"]]
        if prefixes:
            root = Path(request.context.workspace_root or ".").resolve()
            matches = [any(path_matches(path, prefix, root) for prefix in prefixes) for path in request_paths(request)]
            if not matches or not (all(matches) if allow else any(matches)):
                return False
        hosts = matcher.get("hosts") or ([matcher["host"]] if matcher.get("host") else [])
        if hosts:
            host = self.host(request)
            if not any(self.domain(host, domain) or host.endswith("." + str(domain)) for domain in hosts):
                return False
        servers = matcher.get("mcp_servers")
        if servers:
            metadata = request.descriptor.metadata
            server = metadata.get("mcp_server") or metadata.get("mcp_server_name") or metadata.get("server_name")
            uri = str(metadata.get("source_identifier", ""))
            if not server and uri.startswith("mcp://"):
                server = urlparse(uri).netloc
            if server not in servers:
                return False
        if any(request.arguments.get(key) != value for key, value in matcher.get("param_equals", {}).items()):
            return False
        if any(str(value) not in str(request.arguments.get(key, "")) for key, value in matcher.get("param_contains", {}).items()):
            return False
        return True
