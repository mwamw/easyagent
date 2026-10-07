"""Validate first, then assemble through the unchanged Python Agent API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from agent import BasicAgent
from core.Config import Config
from core.llm import EasyLLM

from .builtins import default_registry
from .errors import ConfigurationError
from .loader import read_configuration, validate_options
from .models import AgentConfiguration, ConfigurationPaths
from .registry import BuildContext, ConfigurationRegistry, FactoryEntry, close_resource


# This is installation order, not an execution graph. Runtime remains unchanged.
_BEFORE_TOOLS = ("prompt", "context", "executor", "permissions", "hooks", "callbacks")
_AFTER_TOOLS = ("tasks", "plan", "memory", "session", "observability", "codeintel", "worktree")


@dataclass
class _Prepared:
    document: AgentConfiguration
    paths: ConfigurationPaths
    config: Config
    prompt: str | None
    model_options: BaseModel
    model_factory: Callable[[BaseModel, ConfigurationPaths], EasyLLM]
    steps: dict[str, tuple[FactoryEntry, BaseModel]]


class AgentBuilder:
    def __init__(self, registry: ConfigurationRegistry | None = None) -> None:
        self.registry = registry if registry is not None else default_registry()

    def _prepare(self, path: str | Path, document: AgentConfiguration | None = None) -> _Prepared:
        source = Path(path).expanduser().resolve()
        document = (document.model_copy(deep=True) if document is not None else read_configuration(source))
        directory = source.parent
        workspace = Path(document.agent.workspace).expanduser()
        workspace = (workspace if workspace.is_absolute() else directory / workspace).resolve()
        if not workspace.is_dir():
            raise ConfigurationError("agent.workspace", "Workspace directory does not exist")
        paths = ConfigurationPaths(directory, workspace, ())
        roots = tuple(dict.fromkeys([str(workspace), *(paths.resolve(p) for p in document.agent.allowed_roots)]))
        for root in roots:
            if not Path(root).is_dir():
                raise ConfigurationError("agent.allowed_roots", "Allowed directory does not exist")
        paths = ConfigurationPaths(directory, workspace, roots)
        if {"workspace_root", "allowed_roots"} & document.agent.config.keys():
            raise ConfigurationError("agent.config", "Use agent.workspace / agent.allowed_roots")
        config = validate_options(Config, document.agent.config, "agent.config")
        config.workspace_root = str(workspace)
        config.allowed_roots = list(roots)

        model_name = document.model.implementation
        if model_name not in self.registry.models:
            raise ConfigurationError("model.implementation", f"Unknown implementation {model_name!r}")
        model_type, model_factory = self.registry.models[model_name]
        model_options = validate_options(model_type, document.model.config, "model.config")

        prompt = None
        if document.prompt:
            prompt = document.prompt.text
            if document.prompt.file is not None:
                try:
                    prompt = Path(paths.resolve(document.prompt.file)).read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    raise ConfigurationError("prompt.file", "Cannot read prompt file") from None
        steps = {}
        for kind in (*_BEFORE_TOOLS, *_AFTER_TOOLS, "multi_agent"):
            spec = getattr(document, kind)
            if spec is None or not spec.enabled:
                continue
            entry = self.registry.get(kind, spec.implementation)
            steps[kind] = (entry, entry.validate(spec.config, f"{kind}.config", paths))
        seen = set()
        declared_tools = set()
        for index, tool in enumerate(document.tools):
            if not tool.enabled:
                continue
            if tool.name in seen:
                raise ConfigurationError(f"tools[{index}]", "Duplicate tool loading entry")
            seen.add(tool.name)
            entry = self.registry.get("tools", tool.name)
            if declared_tools.intersection(entry.tool_names):
                raise ConfigurationError(f"tools[{index}]", "Selected tool groups declare conflicting tool names")
            declared_tools.update(entry.tool_names)
            steps[f"tools[{index}]"] = (entry, entry.validate(tool.config, f"tools[{index}].config", paths))

        if document.skills and document.skills.enabled:
            if not document.skills.directories:
                raise ConfigurationError("skills.directories", "At least one Skill directory is required")
            resolved = list(dict.fromkeys(paths.resolve(p) for p in document.skills.directories))
            for value in resolved:
                directory_path = Path(value)
                if not directory_path.is_dir() or not (
                    (directory_path / "SKILL.md").is_file()
                    or any(directory_path.glob("*/SKILL.md"))
                ):
                    raise ConfigurationError("skills.directories", "Directory must contain SKILL.md or direct child skills")
            document.skills.directories = resolved
            # Reuse Skill's own read-only manifest parser; do not duplicate its schema.
            from skill.folder_loader import discover_skill_files, load_skill_manifest
            skill_names = {}
            for value in resolved:
                for file in discover_skill_files(value):
                    try:
                        manifest = load_skill_manifest(file)
                        if manifest.name != file.parent.name or (
                            manifest.name in skill_names and skill_names[manifest.name] != file
                        ):
                            raise ValueError("Skill name mismatch or duplicate")
                        skill_names[manifest.name] = file
                    except Exception:
                        raise ConfigurationError("skills.directories", "Invalid or duplicate Skill manifest") from None
        if document.mcp and document.mcp.enabled:
            names = set()
            for server in document.mcp.servers:
                if not server.enabled:
                    continue
                if server.name in names:
                    raise ConfigurationError("mcp.servers", "Duplicate MCP server name")
                names.add(server.name)
                if server.source_type == "file":
                    server.source = paths.resolve(server.source)
                    if not Path(server.source).is_file():
                        raise ConfigurationError("mcp.servers.source", "MCP source file does not exist")
                elif server.source_type == "url" and not server.source.startswith(("https://", "http://")):
                    raise ConfigurationError("mcp.servers.source", "MCP URL must use http or https")
        return _Prepared(document, paths, config, prompt, model_options, model_factory, steps)

    def validate(self, path: str | Path) -> tuple[str, ...]:
        """Return selected installation steps, without clients, connections or writes."""
        try:
            prepared = self._prepare(path)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(str(path), f"Validation failed ({type(exc).__name__})") from None
        result = ["model", *prepared.steps]
        if prepared.document.skills and prepared.document.skills.enabled:
            result.append("skills")
        if prepared.document.mcp and prepared.document.mcp.enabled:
            result.append("mcp")
        return tuple(result)

    def build(self, path: str | Path) -> BasicAgent:
        try:
            prepared = self._prepare(path)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(str(path), f"Validation failed ({type(exc).__name__})") from None
        return self._assemble(prepared)

    def validate_document(self, document: AgentConfiguration, *, source: str | Path) -> tuple[str, ...]:
        """Validate an in-memory recipe; source supplies relative-path semantics.

        Like validate(), this checks local resources but creates no Agent/client.
        The caller's document is never mutated.
        """
        try:
            prepared = self._prepare(source, document)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(str(source), f"Validation failed ({type(exc).__name__})") from None
        selected = ["model", *prepared.steps]
        for name in ("skills", "mcp"):
            spec = getattr(prepared.document, name)
            if spec is not None and spec.enabled:
                selected.append(name)
        return tuple(selected)

    def build_document(self, document: AgentConfiguration, *, source: str | Path) -> BasicAgent:
        """Build through exactly the same assembly path as a configuration file."""
        try:
            prepared = self._prepare(source, document)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(str(source), f"Validation failed ({type(exc).__name__})") from None
        return self._assemble(prepared)

    def _assemble(self, prepared: _Prepared) -> BasicAgent:
        agent = None
        llm = None
        context = None
        location = "model"
        try:
            llm = prepared.model_factory(prepared.model_options, prepared.paths)
            if not isinstance(llm, EasyLLM):
                raise TypeError("Model factory must return EasyLLM")
            if "temperature" not in prepared.document.agent.config and isinstance(llm.temperature, (int, float)):
                prepared.config.temperature = llm.temperature
            location = "agent"
            agent = BasicAgent(
                prepared.document.agent.name, llm, system_prompt=prepared.prompt,
                description=prepared.document.agent.description, config=prepared.config,
            )
            context = BuildContext(prepared.paths, llm)

            def install(name: str) -> None:
                nonlocal location
                if name in prepared.steps:
                    location = name
                    entry, options = prepared.steps[name]
                    entry.install(agent, options, context)

            for name in _BEFORE_TOOLS:
                install(name)
            tool_modules = ("tasks", "plan", "memory", "codeintel", "worktree", "multi_agent")
            if (
                any(prepared.document.tools[i].enabled for i in range(len(prepared.document.tools)))
                or any(name in prepared.steps for name in tool_modules)
                or (prepared.document.skills and prepared.document.skills.enabled)
                or (prepared.document.mcp and prepared.document.mcp.enabled)
            ):
                from Tool import ToolRegistry
                if agent.tool_registry is None:
                    agent.with_tool(ToolRegistry(conflict_policy="error"))
            for index in range(len(prepared.document.tools)):
                install(f"tools[{index}]")
            for name in _AFTER_TOOLS:
                install(name)
            if prepared.document.skills and prepared.document.skills.enabled:
                location = "skills"
                agent.with_skill(*prepared.document.skills.directories)
            if prepared.document.mcp and prepared.document.mcp.enabled:
                from Tool.builtin import MCPToolManager
                for server in prepared.document.mcp.servers:
                    if not server.enabled:
                        continue
                    location = f"mcp.{server.name}"
                    manager = MCPToolManager(
                        server_source=server.source, server_args=server.args,
                        env=server.env, tool_prefix=server.tool_prefix,
                        include_resources=server.include_resources,
                        auto_connect=False,
                    )
                    try:
                        agent.with_mcp(manager, server_name=server.name)
                    except BaseException:
                        manager.close()
                        raise
            # Capture the completed parent's capabilities only at the end.
            install("multi_agent")
            return agent
        except BaseException as exc:
            cleanup_failed = False
            try:
                if agent is not None:
                    cleanup_failed = agent.close().get("status") == "degraded"
                elif llm is not None:
                    close_resource(llm)
            except Exception:
                cleanup_failed = True
            if context is not None and not context.rollback():
                cleanup_failed = True
            if not isinstance(exc, Exception):
                raise
            suffix = "; resource cleanup was incomplete" if cleanup_failed else ""
            # External exception messages can include credentials. Never echo them.
            raise ConfigurationError(location, f"Assembly failed ({type(exc).__name__}){suffix}") from None


def load_agent(path: str | Path, *, registry: ConfigurationRegistry | None = None) -> BasicAgent:
    return AgentBuilder(registry).build(path)


def validate_agent(path: str | Path, *, registry: ConfigurationRegistry | None = None) -> tuple[str, ...]:
    return AgentBuilder(registry).validate(path)
