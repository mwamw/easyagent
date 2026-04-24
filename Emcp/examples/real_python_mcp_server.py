"""Real MCP server example for EasyAgent (stdio transport).

Run:
    python mcp/examples/real_python_mcp_server.py
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.mcp_server import MCPServer


def create_server() -> MCPServer:
    server = MCPServer(
        name="easyagent-real-python-server",
        description="Real MCP server used by EasyAgent examples",
    )

    def echo(text: str) -> str:
        """Echo input text."""
        return f"echo: {text}"

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def repeat(text: str, times: int = 2) -> str:
        """Repeat text N times."""
        return " ".join([text] * times)

    server.add_tool(echo, name="echo", description="Echo input text")
    server.add_tool(add, name="add", description="Add two integers")
    server.add_tool(repeat, name="repeat", description="Repeat text N times")
    return server


if __name__ == "__main__":
    app = create_server()
    print("[MCP] starting easyagent-real-python-server (stdio)")
    app.run(transport="stdio")
