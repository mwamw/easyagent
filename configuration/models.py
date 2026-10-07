"""Configuration document shape, independent of module implementations."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ComponentSpec(Options):
    enabled: bool = True
    implementation: str = "default"
    config: dict[str, Any] = Field(default_factory=dict)


class RequiredComponentSpec(ComponentSpec):
    enabled: Literal[True] = True


class UISpec(ComponentSpec):
    implementation: str = "terminal"


class PromptSpec(RequiredComponentSpec):
    text: str | None = None
    file: str | None = None

    @model_validator(mode="after")
    def exclusive_content(self):
        if self.text is not None and self.file is not None:
            raise ValueError("Choose text or file, not both")
        return self


class AgentOptions(Options):
    name: str = Field(default="assistant", min_length=1)
    description: str | None = None
    workspace: str = "."
    allowed_roots: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class ToolSelection(Options):
    name: str
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class SkillsOptions(Options):
    enabled: bool = True
    directories: list[str] = Field(default_factory=list)


class MCPServerOptions(Options):
    name: str
    enabled: bool = True
    source: str
    source_type: Literal["command", "file", "url"] = "command"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    tool_prefix: str = ""
    include_resources: bool = False


class MCPOptions(Options):
    enabled: bool = True
    servers: list[MCPServerOptions] = Field(default_factory=list)


class AgentConfiguration(Options):
    version: Literal[1] = 1
    ui: UISpec | None = None
    agent: AgentOptions = Field(default_factory=AgentOptions)
    model: RequiredComponentSpec
    prompt: PromptSpec | None = None
    context: RequiredComponentSpec | None = None
    executor: RequiredComponentSpec | None = None
    tools: list[ToolSelection] = Field(default_factory=list)
    skills: SkillsOptions | None = None
    mcp: MCPOptions | None = None
    permissions: ComponentSpec | None = None
    memory: ComponentSpec | None = None
    plan: ComponentSpec | None = None
    tasks: ComponentSpec | None = None
    session: ComponentSpec | None = None
    observability: ComponentSpec | None = None
    hooks: ComponentSpec | None = None
    callbacks: ComponentSpec | None = None
    codeintel: ComponentSpec | None = None
    worktree: ComponentSpec | None = None
    multi_agent: ComponentSpec | None = None

    @field_validator("model", mode="before")
    @classmethod
    def model_shorthand(cls, value):
        if isinstance(value, dict) and "config" not in value:
            value = dict(value)
            return {
                "enabled": value.pop("enabled", True),
                "implementation": value.pop("implementation", "default"),
                "config": value,
            }
        return value


@dataclass(frozen=True)
class ConfigurationPaths:
    """Only path resolution; not an Agent/module registry."""
    directory: Path
    workspace: Path
    allowed_roots: tuple[str, ...]

    def resolve(self, value: str) -> str:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else self.directory / path).resolve())
