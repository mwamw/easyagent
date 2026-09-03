"""Real MCP filesystem registration flow with first-class MCP runtime wiring."""

from __future__ import annotations

import os

from easyagent import BasicAgent, EasyLLM, ToolRegistry
from Tool.builtin import mcptool


def build_filesystem_manager(workspace: str):
    return mcptool(
        server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
        tool_prefix="fs_",
        include_resources=True,
        resource_tool_prefix="fs_",
    )


def main() -> None:
    workspace = os.path.abspath(".")
    registry = ToolRegistry()
    manager = build_filesystem_manager(workspace)
    agent = (
        BasicAgent(
            name="mcp-filesystem",
            llm=EasyLLM(
                provider="openai",
                base_url="http://127.0.0.1:5124/v1",
                api_key="122",
                model="qwen3.5-9b",
            ),
        )
        .with_tool(registry)
        .with_mcp(manager)
    )

    snapshot = manager.snapshot(refresh=True)
    print("MCP server:", manager.registry_server_name)
    print("Remote capabilities:", snapshot.to_dict())
    print("Agent tools:", agent.tool_registry.get_tool_names())
    print(agent.invoke("列出当前 MCP filesystem server 可访问的资源。"))
    agent.close()


if __name__ == "__main__":
    main()
