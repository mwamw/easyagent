"""Agent definitions owned by the Claude Code-style implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class AgentDefinition:
    name: str
    description: str
    system_prompt: str = ""
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    model: str | None = None
    permission_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AgentDefinition.name must not be empty")
        if not self.description.strip():
            raise ValueError("AgentDefinition.description must not be empty")

    def format_listing(self) -> str:
        if self.tools is not None:
            visible = [name for name in self.tools if name not in self.disallowed_tools]
            tool_text = ", ".join(visible) if visible else "None"
        elif self.disallowed_tools:
            tool_text = "All tools except " + ", ".join(self.disallowed_tools)
        else:
            tool_text = "All tools"
        return f"- {self.name}: {self.description} (Tools: {tool_text})"


class AgentDefinitionRegistry:
    def __init__(self, definitions: tuple[AgentDefinition, ...] = ()) -> None:
        self._definitions: dict[str, AgentDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: AgentDefinition) -> None:
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be AgentDefinition")
        self._definitions[definition.name] = definition

    def get(self, name: str | None) -> AgentDefinition:
        resolved = str(name or "general-purpose").strip()
        try:
            return self._definitions[resolved]
        except KeyError as exc:
            available = ", ".join(sorted(self._definitions))
            raise ValueError(
                f"Unknown subagent_type {resolved!r}; available types: {available}"
            ) from exc

    def list(self) -> tuple[AgentDefinition, ...]:
        return tuple(self._definitions.values())


def build_default_agent_definitions() -> AgentDefinitionRegistry:
    return AgentDefinitionRegistry(
        (
            AgentDefinition(
                name="general-purpose",
                description=(
                    "General-purpose agent for researching complex questions, "
                    "executing multi-step tasks, and making code changes."
                ),
                system_prompt=(
                    "Work autonomously on the assigned task. Use the available tools, "
                    "follow the requested scope, and return a concise result with concrete evidence."
                ),
            ),
            AgentDefinition(
                name="Explore",
                description=(
                    "Fast read-only agent for locating files, searching code, and answering "
                    "questions about an existing codebase."
                ),
                system_prompt=(
                    "You are a codebase exploration specialist. Search efficiently, remain read-only, "
                    "and report file paths and relevant symbols. Do not modify files."
                ),
                tools=("FileRead", "List", "Glob", "Grep", "WebFetch", "WebSearch"),
            ),
            AgentDefinition(
                name="Plan",
                description=(
                    "Read-only planning agent for investigating a codebase and producing an "
                    "implementation plan."
                ),
                system_prompt=(
                    "Investigate the requested change without editing files. Return an actionable "
                    "implementation plan grounded in the current codebase."
                ),
                tools=("FileRead", "List", "Glob", "Grep", "WebFetch", "WebSearch"),
                permission_mode="plan",
            ),
        )
    )


__all__ = [
    "AgentDefinition",
    "AgentDefinitionRegistry",
    "build_default_agent_definitions",
]
