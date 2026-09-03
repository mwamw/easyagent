"""Agent-local manager for directory-based Skills."""

from __future__ import annotations

from hashlib import sha256
import os
import re
from threading import RLock
from typing import Any, Iterable

from pathspec import GitIgnoreSpec

from core.permissions import PermissionBehavior, PermissionRule
from metamessage import MetaMessage, MetaMessageLifecycle
from runtime import RuntimeEvent, RuntimeEventType

from .base import SkillManifest
from .folder_loader import discover_skill_files, load_skill_body, load_skill_manifest


_TOOL_RULE_PATTERN = re.compile(r"^(?P<name>[^()]+?)(?:\((?P<rule>.*)\))?$")
_TERMINAL_EVENTS = {
    RuntimeEventType.AGENT_INVOKE_COMPLETED,
    RuntimeEventType.AGENT_INVOKE_FAILED,
    RuntimeEventType.AGENT_INVOKE_INTERRUPTED,
}
_SKILL_EVENTS = _TERMINAL_EVENTS | {RuntimeEventType.TOOL_INVOKE_COMPLETED}
_PATH_KEYS = {"file_path", "notebook_path", "path", "root", "relative_path"}
_PATH_COLLECTION_KEYS = {"paths", "files", "matches", "entries"}
_FILESYSTEM_TOOL_NAMES = {
    "fileedit",
    "fileread",
    "filewrite",
    "glob",
    "grep",
    "list",
    "notebookedit",
}


class SkillModelInvocationDisabledError(PermissionError):
    pass


class SkillNotActiveError(PermissionError):
    def __init__(self, manifest: SkillManifest) -> None:
        self.manifest = manifest
        super().__init__(
            f"Skill '{manifest.name}' is conditional and has not been activated by a "
            "matching workspace path"
        )


class SkillManager:
    """Indexes Skill directories and expands a selected Skill on demand."""

    def __init__(self) -> None:
        self._agent: Any = None
        self._directories: list[str] = []
        self._skills: dict[str, SkillManifest] = {}
        self._activated_path_skills: set[str] = set()
        self._active_keys: set[str] = set()
        self._permission_sources: set[str] = set()
        self._event_subscription: str | None = None
        self._lock = RLock()

    @property
    def directories(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._directories)

    @property
    def skill_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._skills))

    def bind_agent(self, agent: Any) -> None:
        """Bind this module to one Agent and its RuntimeEvent stream."""

        previous = self._agent
        if previous is not None and previous is not agent and self._event_subscription:
            previous.event_bus.unsubscribe(self._event_subscription)
            self._event_subscription = None
        self._agent = agent
        if self._event_subscription is None:
            self._event_subscription = agent.event_bus.subscribe(
                self._handle_runtime_event,
                event_types=_SKILL_EVENTS,
            )

    def install_tool(self) -> None:
        """Install the single model-facing Skill tool into the bound Agent."""

        if self._agent is None:
            raise RuntimeError("SkillManager must be bound before installing skill_tool")
        registry = getattr(self._agent, "tool_registry", None)
        if registry is None:
            raise RuntimeError("SkillManager requires an installed ToolRegistry")
        existing = registry.get_tool("skill_tool")
        if existing is not None:
            if getattr(existing, "manager", None) is self:
                return
            raise ValueError("Tool 'skill_tool' is already registered by another SkillManager")
        from .tool import SkillTool

        registry.register_tool(
            SkillTool(self),
            expose_in_deferred=True,
            conflict_policy="error",
        )

    def add_directories(
        self,
        directories: Iterable[str | os.PathLike[str]],
    ) -> "SkillManager":
        """Index one or more Skill roots as an all-or-nothing operation."""

        roots: list[str] = []
        candidates: list[SkillManifest] = []
        for directory in directories:
            root = str(os.path.realpath(os.path.expanduser(os.fspath(directory))))
            if root in roots:
                continue
            roots.append(root)
            for skill_file in discover_skill_files(root):
                manifest = load_skill_manifest(skill_file)
                if manifest.name != skill_file.parent.name:
                    raise ValueError(
                        f"Skill name '{manifest.name}' must match directory "
                        f"'{skill_file.parent.name}' in {skill_file}"
                    )
                if manifest.context != "fork" and (manifest.agent or manifest.model):
                    raise ValueError(
                        f"Skill '{manifest.name}' may declare agent/model only with context: fork"
                    )
                for allowed_tool in manifest.allowed_tools:
                    self._permission_rule(allowed_tool)
                candidates.append(manifest)

        with self._lock:
            proposed = dict(self._skills)
            for manifest in candidates:
                existing = proposed.get(manifest.name)
                if existing is not None and existing.file_path != manifest.file_path:
                    raise ValueError(
                        f"Duplicate Skill name '{manifest.name}' from "
                        f"{existing.file_path} and {manifest.file_path}"
                    )
                proposed[manifest.name] = manifest
            self._skills = proposed
            for root in roots:
                if root not in self._directories:
                    self._directories.append(root)
        return self

    def has_skill(self, name: str) -> bool:
        with self._lock:
            return str(name or "").strip() in self._skills

    def get_skill(self, name: str) -> SkillManifest:
        normalized = str(name or "").strip().lstrip("/")
        with self._lock:
            manifest = self._skills.get(normalized)
        if manifest is None:
            available = ", ".join(self.skill_names) or "none"
            raise KeyError(f"Unknown Skill '{normalized}'. Available Skills: {available}")
        return manifest

    def list_skills(self, *, model_invocable_only: bool = False) -> list[SkillManifest]:
        with self._lock:
            manifests = [
                self._skills[name]
                for name in sorted(self._skills)
                if not self._skills[name].paths or name in self._activated_path_skills
            ]
        if model_invocable_only:
            manifests = [item for item in manifests if not item.disable_model_invocation]
        return manifests

    def is_visible(self, name: str) -> bool:
        manifest = self.get_skill(name)
        with self._lock:
            return not manifest.paths or manifest.name in self._activated_path_skills

    def build_skill_listing_prompt(self) -> str:
        """Build the small discovery layer placed in a system reminder."""

        manifests = self.list_skills(model_invocable_only=True)
        if not manifests:
            return ""
        lines = [
            "<available_skills>",
            "The following Skills are available through `skill_tool`. Use a Skill when its "
            "description or usage guidance matches the task. Load it before following its workflow; "
            "do not guess instructions that have not been loaded.",
        ]
        for manifest in manifests:
            details = manifest.description
            if manifest.when_to_use:
                details += f" Use when: {manifest.when_to_use}"
            if manifest.argument_hint:
                details += f" Arguments: {manifest.argument_hint}"
            lines.append(f"- `{manifest.name}`: {details}")
        lines.extend(
            [
                "Skill instructions apply only to the current Agent invocation and are removed "
                "after it finishes. Call `skill_tool` again in a later invocation when needed.",
                "</available_skills>",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _permission_rule(value: str) -> PermissionRule:
        match = _TOOL_RULE_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"Invalid allowed-tools entry: {value}")
        tool_name = match.group("name").strip()
        expression = str(match.group("rule") or "").strip()
        matcher: dict[str, Any] = {}
        if expression:
            prefix = expression.rstrip("*").removesuffix(":").strip()
            if not prefix:
                raise ValueError(
                    f"allowed-tools matcher '{expression}' must not grant an empty prefix"
                )
            if tool_name.lower() == "bash":
                if any(character in prefix for character in "*?["):
                    raise ValueError(
                        f"Bash matcher '{expression}' must be a command prefix"
                    )
                matcher["command_prefixes"] = [prefix]
            elif expression.startswith("domain:"):
                domain = prefix.removeprefix("domain:").removeprefix("*.")
                if any(character in domain for character in "*?["):
                    raise ValueError(
                        f"Domain matcher '{expression}' must be a hostname suffix"
                    )
                matcher["hosts"] = [domain]
            elif expression.startswith("path:"):
                path_prefix = prefix.removeprefix("path:")
                if any(character in path_prefix for character in "*?["):
                    raise ValueError(
                        f"Path matcher '{expression}' must be a path prefix"
                    )
                matcher["path_prefixes"] = [path_prefix]
            elif tool_name.lower() in _FILESYSTEM_TOOL_NAMES:
                if any(character in prefix for character in "*?["):
                    raise ValueError(
                        f"Filesystem matcher '{expression}' must be a path prefix"
                    )
                matcher["path_prefixes"] = [prefix]
            else:
                raise ValueError(
                    f"Unsupported allowed-tools matcher '{expression}' for "
                    f"tool '{tool_name}'. Use path:, domain:, a Bash command prefix, "
                    "or grant the exact tool name without parentheses."
                )
        return PermissionRule(
            tool_name=tool_name,
            behavior=PermissionBehavior.ALLOW,
            matcher=matcher,
            description="Temporarily allowed by the active Skill",
        )

    def _activate_allowed_tools(self, manifest: SkillManifest) -> tuple[list[str], list[str]]:
        if self._agent is None or not manifest.allowed_tools:
            return [], []
        registry = getattr(self._agent, "tool_registry", None)
        available: list[str] = []
        unavailable: list[str] = []
        rules: list[PermissionRule] = []
        for item in manifest.allowed_tools:
            rule = self._permission_rule(item)
            if registry is None or not registry.has_tool(rule.tool_name):
                unavailable.append(item)
                continue
            rules.append(rule)
            available.append(item)
        if rules:
            source = f"skill:{manifest.name}"
            self._agent.permission_context.set_source_rules(source, rules, priority=55)
            self._permission_sources.add(source)
            registry.expand_deferred_tools(sorted({rule.tool_name for rule in rules}))
        return available, unavailable

    @staticmethod
    def _render_inline_content(manifest: SkillManifest, body: str, args: str) -> str:
        argument_text = args.strip() or "(none)"
        return (
            f'<skill name="{manifest.name}">\n'
            f"Skill directory: {manifest.directory}\n"
            f"Invocation arguments: {argument_text}\n\n"
            f"{body}\n"
            "</skill>"
        )

    def invoke(
        self,
        name: str,
        *,
        args: str = "",
        model_initiated: bool = True,
    ) -> dict[str, Any]:
        """Load and activate one Skill for the current Agent invocation."""

        manifest = self.get_skill(name)
        if model_initiated and manifest.disable_model_invocation:
            raise SkillModelInvocationDisabledError(
                f"Skill '{manifest.name}' has disable-model-invocation enabled"
            )
        if model_initiated and not self.is_visible(manifest.name):
            raise SkillNotActiveError(manifest)
        if self._agent is None:
            raise RuntimeError("SkillManager is not bound to an Agent")
        body = load_skill_body(manifest, args)
        if manifest.context == "fork":
            allowed, unavailable = self._activate_allowed_tools(manifest)
            try:
                return self._invoke_fork(
                    manifest,
                    body,
                    args,
                    allowed=allowed,
                    unavailable=unavailable,
                )
            finally:
                source = f"skill:{manifest.name}"
                self._agent.permission_context.clear_rules(source=source)
                self._permission_sources.discard(source)

        key = f"{manifest.name}:{sha256(str(args or '').encode('utf-8')).hexdigest()[:16]}"
        with self._lock:
            already_active = key in self._active_keys
            self._active_keys.add(key)
        allowed, unavailable = self._activate_allowed_tools(manifest)
        if not already_active:
            self._agent.emit_metamessage(
                MetaMessage(
                    name=f"skill:{manifest.name}",
                    content=self._render_inline_content(manifest, body, args),
                    lifecycle=MetaMessageLifecycle.INVOCATION,
                    dedup_key=f"skill:{key}",
                    metadata={
                        "source": "skill",
                        "skillName": manifest.name,
                        "skillPath": manifest.file_path,
                        "skillDirectory": manifest.directory,
                        "arguments": str(args or ""),
                        "allowedTools": allowed,
                    },
                )
            )
        return {
            "success": True,
            "skill": manifest.name,
            "status": "inline",
            "scope": MetaMessageLifecycle.INVOCATION.value,
            "alreadyActive": already_active,
            "instructionSource": manifest.file_path,
            "skillDirectory": manifest.directory,
            "allowedTools": allowed,
            "unavailableTools": unavailable,
            "model": manifest.model,
        }

    def _invoke_fork(
        self,
        manifest: SkillManifest,
        body: str,
        args: str,
        *,
        allowed: list[str],
        unavailable: list[str],
    ) -> dict[str, Any]:
        if self._agent is None:
            raise RuntimeError("SkillManager is not bound to an Agent")
        registry = getattr(self._agent, "tool_registry", None)
        if registry is None or not registry.has_tool("Agent"):
            raise RuntimeError(
                f"Skill '{manifest.name}' requires context: fork; install the multi-agent "
                "module with agent.with_multi_agent() first"
            )
        parameters: dict[str, Any] = {
            "description": f"Run Skill {manifest.name}",
            "prompt": self._render_inline_content(manifest, body, args),
            "run_in_background": False,
        }
        if manifest.agent:
            parameters["subagent_type"] = manifest.agent
        if manifest.model:
            parameters["model"] = manifest.model
        result = self._agent.execute_tool_result("Agent", parameters)
        if result.status != "success":
            raise RuntimeError(result.to_display_string())
        data = dict(result.structured_data or {})
        return {
            "success": True,
            "skill": manifest.name,
            "status": "forked",
            "instructionSource": manifest.file_path,
            "skillDirectory": manifest.directory,
            "allowedTools": allowed,
            "unavailableTools": unavailable,
            "model": manifest.model,
            "agentId": data.get("agentId"),
            "outputFile": data.get("outputFile"),
            "result": data.get("content") or result.content,
        }

    @staticmethod
    def _path_strings(value: Any, *, parent_key: str = "") -> list[str]:
        if isinstance(value, dict):
            paths: list[str] = []
            for key, item in value.items():
                normalized_key = str(key).lower()
                if normalized_key in _PATH_KEYS or normalized_key in _PATH_COLLECTION_KEYS:
                    paths.extend(
                        SkillManager._path_strings(item, parent_key=normalized_key)
                    )
                elif isinstance(item, (dict, list, tuple)):
                    paths.extend(SkillManager._path_strings(item, parent_key=normalized_key))
            return paths
        if isinstance(value, (list, tuple)):
            paths: list[str] = []
            for item in value:
                paths.extend(SkillManager._path_strings(item, parent_key=parent_key))
            return paths
        if not isinstance(value, str):
            return []
        candidate = value.strip()
        if not candidate:
            return []
        if parent_key in {"matches", "files"}:
            absolute_match = re.match(r"^(?P<path>/.*?)(?::\d+)?(?::.*)?$", candidate)
            return [absolute_match.group("path")] if absolute_match else []
        return [candidate] if parent_key in _PATH_KEYS or parent_key == "paths" else []

    def _workspace_paths_from_event(self, event: RuntimeEvent) -> list[str]:
        if self._agent is None:
            return []
        registry = getattr(self._agent, "tool_registry", None)
        tool_name = str(event.data.get("tool_name") or "")
        tool = registry.get_tool(tool_name) if registry is not None else None
        if tool is None:
            return []
        spec = tool.get_spec()
        if "filesystem" not in spec.tags and "notebook" not in spec.tags:
            return []

        result = event.data.get("result")
        payloads = [event.data.get("arguments") or {}]
        if result is not None:
            payloads.extend(
                [
                    getattr(result, "structured_data", None) or {},
                    getattr(result, "metadata", None) or {},
                ]
            )
        candidates: list[str] = []
        for payload in payloads:
            candidates.extend(self._path_strings(payload))
        return candidates

    def activate_for_paths(self, paths: Iterable[str | os.PathLike[str]]) -> list[str]:
        """Make conditional Skills visible after a workspace path is touched."""

        if self._agent is None:
            raise RuntimeError("SkillManager is not bound to an Agent")
        workspace_root = os.path.realpath(self._agent.execution_context.workspace_root)
        relative_paths: list[str] = []
        for value in paths:
            candidate = os.fspath(value).strip()
            if not candidate or "://" in candidate:
                continue
            absolute = os.path.realpath(
                candidate
                if os.path.isabs(candidate)
                else os.path.join(workspace_root, candidate)
            )
            try:
                relative = os.path.relpath(absolute, workspace_root)
            except ValueError:
                continue
            if relative == os.pardir or relative.startswith(os.pardir + os.sep):
                continue
            relative_paths.append(relative.replace(os.sep, "/"))

        activated: list[str] = []
        with self._lock:
            conditional = [
                manifest
                for manifest in self._skills.values()
                if manifest.paths and manifest.name not in self._activated_path_skills
            ]
            for manifest in conditional:
                matcher = GitIgnoreSpec.from_lines(manifest.paths)
                if any(matcher.match_file(path) for path in relative_paths):
                    self._activated_path_skills.add(manifest.name)
                    activated.append(manifest.name)
        return sorted(activated)

    def _handle_runtime_event(self, event: RuntimeEvent) -> None:
        if event.type == RuntimeEventType.TOOL_INVOKE_COMPLETED:
            self.activate_for_paths(self._workspace_paths_from_event(event))
            return
        if event.type not in _TERMINAL_EVENTS:
            return
        with self._lock:
            sources = list(self._permission_sources)
            self._permission_sources.clear()
            self._active_keys.clear()
        if self._agent is not None:
            for source in sources:
                self._agent.permission_context.clear_rules(source=source)

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            manifests = [self._skills[name] for name in sorted(self._skills)]
            activated = sorted(self._activated_path_skills)
        return {
            "version": 2,
            "directories": list(self.directories),
            "skills": [manifest.model_dump(mode="python") for manifest in manifests],
            "activatedPathSkills": activated,
        }

    def restore_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(state or {})
        requested = {
            str(name)
            for name in list(data.get("activatedPathSkills") or [])
            if str(name or "").strip()
        }
        with self._lock:
            restored = {
                name
                for name in requested
                if name in self._skills and bool(self._skills[name].paths)
            }
            self._activated_path_skills = restored
        missing = sorted(requested - restored)
        return {
            "status": "degraded" if missing else "restored",
            "restoredItems": [f"conditional-skill:{name}" for name in sorted(restored)],
            "missingItems": [f"conditional-skill:{name}" for name in missing],
            "metadata": {"activatedPathSkills": sorted(restored)},
        }

    def close(self) -> dict[str, Any]:
        with self._lock:
            sources = list(self._permission_sources)
            self._permission_sources.clear()
            self._active_keys.clear()
        if self._agent is not None:
            for source in sources:
                self._agent.permission_context.clear_rules(source=source)
            if self._event_subscription:
                self._agent.event_bus.unsubscribe(self._event_subscription)
        self._event_subscription = None
        self._agent = None
        return {"status": "closed", "directories": list(self.directories)}


__all__ = [
    "SkillManager",
    "SkillModelInvocationDisabledError",
    "SkillNotActiveError",
]
