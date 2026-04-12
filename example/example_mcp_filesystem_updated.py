"""Updated MCP filesystem example for EasyAgent.

This example uses a real MCP filesystem server configuration:

    tool = mcptool(
        server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
        tool_prefix="py_",
    )

It demonstrates the updated MCP module in four layers:
1. Runtime capability discovery (tools / resources / prompts)
2. ToolRegistry integration (remote tools + optional resource tools)
3. SkillRegistry integration for MCP prompts
4. Agent-side wiring with MetaSkill + MCPSkill

This file is intended to be read and optionally run manually.
There are no asserts; everything is printed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import mcptool
from agent.BasicAgent import BasicAgent
from skill.builtin import MCPSkill
from skill.meta_tools import MetaSkill
from skill.registry import SkillRegistry


class PreviewLLM:
    """A minimal placeholder LLM so we can inspect Agent wiring without real model calls."""

    def __init__(self) -> None:
        self.provide = "preview"
        self.provider_name = "preview"
        self.model = "preview-model"

    def invoke(self, messages, temperature: float | None = None, **kwargs) -> str:
        return "preview"

    def think(self, messages, temperature: float | None = None):
        yield "preview"


def build_filesystem_mcp_manager(workspace: str):
    """Build the MCP manager with the same shape you are currently using."""
    tool = mcptool(
        server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
        tool_prefix="py_",
    )
    return tool


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_items(title: str, items: Iterable[Any]) -> None:
    print(title)
    has_any = False
    for item in items:
        has_any = True
        print("-", item)
    if not has_any:
        print("(empty)")


def preview_runtime_capabilities(workspace: str) -> None:
    """Connect directly through MCPToolManager and print discovered capabilities."""
    manager = build_filesystem_mcp_manager(workspace)
    print_section("1. Runtime Capability Discovery")

    try:
        tools = manager.list_remote_tools()
        print_items("Remote tools", [tool["name"] for tool in tools])

        try:
            resources = manager.list_remote_resources()
        except Exception as exc:
            resources = []
            print("list_remote_resources() failed:", exc)
        print_items("Remote resources", [resource.get("uri", "") for resource in resources])

        try:
            prompts = manager.list_remote_prompts()
        except Exception as exc:
            prompts = []
            print("list_remote_prompts() failed:", exc)
        print_items("Remote prompts", [prompt["name"] for prompt in prompts])

        print("\nFull prompt metadata:")
        if prompts:
            for prompt in prompts:
                print(prompt)
        else:
            print("(filesystem server often has no prompts; this is normal)")
    finally:
        manager.close()


def preview_tool_registry_integration(workspace: str) -> None:
    """Register MCP tools/resources into ToolRegistry and print ToolSpec details."""
    manager = build_filesystem_mcp_manager(workspace)
    registry = ToolRegistry()
    print_section("2. ToolRegistry Integration")

    try:
        manager.register_to_registry(
            registry,
            include_resources=True,
            resource_tool_prefix="py_",
        )

        print_items("Registered tool names", registry.get_tool_names())

        print("\nTool specs:")
        for tool_name in registry.get_tool_names():
            spec = registry.get_tool_spec(tool_name)
            if spec is None:
                continue
            print(f"\n[{tool_name}]")
            print("description:", spec.description)
            print("read_only:", spec.read_only)
            print("destructive:", spec.destructive)
            print("requires_confirmation:", spec.requires_confirmation)
            print("supports_parallel:", spec.supports_parallel)
            print("source:", spec.source)
            print("metadata:", spec.metadata)
            print("schema_description:")
            print(spec.build_schema_description())
    finally:
        manager.close()


def preview_prompt_skill_integration(workspace: str) -> None:
    """Register MCP prompts into SkillRegistry as on-demand Skills and print manifests."""
    manager = build_filesystem_mcp_manager(workspace)
    registry = SkillRegistry()
    print_section("3. SkillRegistry Integration For MCP Prompts")

    try:
        skill_names = manager.register_prompt_skills(
            registry,
            skill_prefix="filesystem_",
        )
        print_items("Registered MCP prompt skills", skill_names)

        if not skill_names:
            print("\nThis filesystem server likely exposes no prompts.")
            print("The code path is still useful for MCP servers that do expose prompts.")
            return

        print("\nPrompt skill manifests:")
        for name in skill_names:
            manifest = registry.get_manifest(name)
            print(f"\n[{name}]")
            print("listing_description:", manifest.listing_description)
            print("when_to_use:", manifest.when_to_use)
            print("source_type:", manifest.source_type)
            print("source_path:", manifest.source_path)
            print("metadata:", manifest.metadata)

        first_name = skill_names[0]
        print(f"\nLoad body for first prompt skill: {first_name}")
        try:
            body = registry.load_body(first_name, prompt_arguments={"language": "中文"})
            print(body)
        except Exception as exc:
            print("load_body(...) failed:", exc)
            print("If the prompt requires different arguments, adjust prompt_arguments manually.")
    finally:
        manager.close()


def _extract_section(prompt: str, header: str, max_lines: int = 18) -> str:
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == header.strip():
            return "\n".join(lines[index:index + max_lines])
    return f"(section not found: {header})"


def preview_agent_wiring(workspace: str) -> None:
    """Show how the updated MCP module fits into the current Agent/Skill system."""
    print_section("4. Agent-Side Wiring")

    registry = ToolRegistry()
    skill_registry = SkillRegistry()

    # Path A: directly register MCP tools/resources into ToolRegistry.
    manager = build_filesystem_mcp_manager(workspace)
    try:
        manager.register_to_registry(
            registry,
            include_resources=True,
            resource_tool_prefix="py_",
        )
    finally:
        manager.close()

    agent = BasicAgent(
        name="mcp_filesystem_demo",
        llm=PreviewLLM(),
        enable_tool=True,
        tool_registry=registry,
    )

    # Path B: mount MetaSkill + MCPSkill so MCP prompts can become on-demand Skills.
    agent.with_skill(MetaSkill(skill_registry, agent.skill_manager))
    agent.with_skill(
        MCPSkill(
            server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
            tool_prefix="py_",
            include_resource_tools=True,
            resource_tool_prefix="py_",
            prompt_registry=skill_registry,
            register_prompt_skills=True,
        )
    )

    print_items("Agent tool names", agent.tool_registry.get_tool_names())
    print_items("On-demand skill names", [manifest.name for manifest in skill_registry.list_manifests()])

    prompt = agent.get_enhanced_prompt()
    print("\nPrompt excerpt: ## Skill 使用规则")
    print(_extract_section(prompt, "## Skill 使用规则"))
    print("\nPrompt excerpt: ## 可用 Skills")
    print(_extract_section(prompt, "## 可用 Skills"))


def main() -> None:
    workspace = os.path.abspath(".")
    print("workspace =", workspace)
    preview_runtime_capabilities(workspace)
    preview_tool_registry_integration(workspace)
    preview_prompt_skill_integration(workspace)
    preview_agent_wiring(workspace)


if __name__ == "__main__":
    main()
