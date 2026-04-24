"""Another real MCP server example (structured output tools, stdio transport)."""

import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.mcp_server import MCPServer


def create_server() -> MCPServer:
    server = MCPServer(
        name="easyagent-real-python-structured-server",
        description="Real structured MCP server for integration tests",
    )

    def make_profile(name: str, age: int) -> str:
        """Build a simple JSON profile string."""
        return json.dumps({"name": name, "age": age}, ensure_ascii=False)

    def sum_list(numbers: list[int]) -> int:
        """Sum integer list."""
        return sum(numbers)

    def make_tags(topic: str) -> list[str]:
        """Generate simple tags for topic."""
        return [topic, f"{topic}-mcp", "easyagent"]

    server.add_tool(make_profile, name="make_profile", description="Build JSON profile")
    server.add_tool(sum_list, name="sum_list", description="Sum integer list")
    server.add_tool(make_tags, name="make_tags", description="Generate tags")
    return server


if __name__ == "__main__":
    app = create_server()
    app.run(transport="stdio")
