"""Compiled source recipes and their bridge to the existing AgentBuilder."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from pathlib import Path
from typing import Mapping

from agent import BasicAgent
from configuration import AgentBuilder, AgentConfiguration, ConfigurationRegistry
from configuration.errors import ConfigurationError
from configuration.loader import validate_options
from ui.configuration import UIRegistry
from ui.base import BaseAgentUI

from .checking import Checker, COMPONENT_FIELDS, schema_check
from .syntax import LanguageError, Span
from .values import Recipe, Value, plain


def configuration_span(value: Value, location: str) -> Span:
    """Map Builder field paths back to the nearest original source expression."""
    parts = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+", location)
    if parts and parts[0] == "agent":
        parts.pop(0)
    if parts and parts[0] == "permissions":
        parts[0] = "permission"
    for part in parts:
        if isinstance(value.data, Recipe):
            if part in ("config", "implementation") and value.data.kind != "agent":
                continue
            fields = value.data.fields
            if part not in fields:
                break
            value = fields[part]
        elif isinstance(value.data, dict) and part in value.data:
            value = value.data[part]
        elif isinstance(value.data, tuple) and part.isdigit() and int(part) < len(value.data):
            value = value.data[int(part)]
        else:
            break
    return value.span


def document_for(value: Value, *, name: str, environ: Mapping[str, str]) -> AgentConfiguration:
    recipe = value.data
    assert isinstance(recipe, Recipe) and recipe.kind == "agent"
    fields = recipe.fields
    agent = {k: plain(v, environ=environ) for k, v in fields.items()
             if k in {"name", "description", "workspace", "allowed_roots", "config"}}
    agent.setdefault("name", name)
    agent.setdefault("workspace", str(recipe.span.file.parent))
    document = {"version": 1, "agent": agent}
    for key in COMPONENT_FIELDS:
        if key not in fields:
            continue
        value = fields[key]
        if value.kind == "None":
            continue
        item = value.data
        assert isinstance(item, Recipe)
        config = {k: plain(v, environ=environ) for k, v in item.fields.items()}
        spec = {"implementation": item.implementation, "config": config}
        if key == "prompt":
            for reserved in ("text", "file"):
                if reserved in config:
                    spec[reserved] = config.pop(reserved)
        document["permissions" if key == "permission" else key] = spec
    tools = fields.get("tools")
    if tools and tools.kind != "None":
        assert isinstance(tools.data, tuple)
        tool_specs = []
        for tool in tools.data:
            item = tool.data
            assert isinstance(item, Recipe)
            tool_specs.append({
                "name": item.implementation, "enabled": item.enabled,
                "config": {k: plain(v, environ=environ) for k, v in item.fields.items()} if item.enabled else {},
            })
        document["tools"] = tool_specs
    skills = fields.get("skills")
    if skills and skills.kind != "None":
        document["skills"] = {"directories": plain(skills, environ=environ)}
    mcp = fields.get("mcp")
    if mcp and mcp.kind != "None":
        document["mcp"] = plain(mcp, environ=environ)
    return validate_options(AgentConfiguration, document, str(recipe.span.file))


@dataclass(frozen=True)
class Launch:
    agent: str
    ui: str | None
    options: Value
    span: Span


@dataclass
class CompiledProgram:
    source: Path
    agents: dict[str, Value] = field(repr=False)
    registry: ConfigurationRegistry = field(repr=False)
    ui_registry: UIRegistry = field(repr=False)
    launch: Launch | None = None
    sources: tuple[Path, ...] = ()

    def select(self, name: str | None = None) -> tuple[str, Value]:
        if name is None:
            name = self.launch.agent if self.launch else next(iter(self.agents)) if len(self.agents) == 1 else None
        if name is None:
            raise LanguageError(Span(self.source, 1, 1), "Select an Agent by name or add a run declaration", code="EA301")
        if name not in self.agents:
            raise LanguageError(Span(self.source, 1, 1), f"No Agent named {name}", code="EA301")
        return name, self.agents[name]

    def configuration(self, name: str | None = None, *, environ: Mapping[str, str] | None = None) -> AgentConfiguration:
        """Resolve secrets explicitly at build time. Returned config may contain secrets."""
        name, value = self.select(name)
        env = dict(os.environ if environ is None else environ)
        Checker(self.registry, self.ui_registry).check(value, runtime=True, environ=env)
        return document_for(value, name=name, environ=env)

    def build(self, name: str | None = None, *, environ: Mapping[str, str] | None = None) -> BasicAgent:
        name, value = self.select(name)
        document = self.configuration(name, environ=environ)
        try:
            return AgentBuilder(self.registry).build_document(document, source=self.source)
        except ConfigurationError as exc:
            raise LanguageError(configuration_span(value, exc.location), f"{exc.location}: {exc.message}", code="EA301") from None

    def validate_environment(self, name: str | None = None, *, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
        name, value = self.select(name)
        document = self.configuration(name, environ=environ)
        try:
            return AgentBuilder(self.registry).validate_document(document, source=self.source)
        except ConfigurationError as exc:
            raise LanguageError(configuration_span(value, exc.location), f"{exc.location}: {exc.message}", code="EA301") from None

    def create_ui(self, *, environ: Mapping[str, str] | None = None) -> BaseAgentUI:
        name = self.launch.ui if self.launch and self.launch.ui else "terminal"
        options = self.launch.options if self.launch else Value("Object", {}, Span(self.source, 1, 1))
        env = dict(os.environ if environ is None else environ)
        schema_check(self.ui_registry.config_model(name), options, runtime=True, environ=env)
        data = plain(options, environ=env)
        assert isinstance(data, dict)
        instance: BaseAgentUI = self.ui_registry.create(name, data)
        return instance

    def explain(self) -> dict[str, object]:
        """A value-free assembly preview: no secrets, factories, IO, or environment reads."""
        agents = {}
        for name, value in self.agents.items():
            recipe = value.data
            assert isinstance(recipe, Recipe)
            components = {key: item.data.implementation for key, item in recipe.fields.items()
                          if key in COMPONENT_FIELDS and isinstance(item.data, Recipe)}
            for key in ("context", "executor", "prompt"):
                components.setdefault(key, "default (Agent default)")
            tools = recipe.fields.get("tools")
            tool_names = []
            if tools and tools.kind != "None":
                assert isinstance(tools.data, tuple)
                for tool in tools.data:
                    item = tool.data
                    assert isinstance(item, Recipe)
                    if item.enabled:
                        tool_names.append(item.implementation)
            mcp = recipe.fields.get("mcp")
            mcp_enabled = mcp is not None and mcp.kind != "None"
            if mcp is not None and isinstance(mcp.data, dict):
                enabled = mcp.data.get("enabled")
                if enabled is not None and enabled.data is False:
                    mcp_enabled = False
            agents[name] = {"components": components, "tools": tool_names,
                            "skills": "skills" in recipe.fields and recipe.fields["skills"].kind != "None",
                            "mcp": mcp_enabled}
        return {"source": str(self.source), "sources": [str(p) for p in self.sources], "agents": agents,
                "run": {"agent": self.launch.agent, "ui": self.launch.ui} if self.launch else None,
                "checked": "syntax, references, types, registered schemas",
                "not_checked": "credentials, local resources, services, OS sandbox availability"}
