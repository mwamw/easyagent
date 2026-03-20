"""Another real MCP server example (text utilities, stdio transport)."""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.mcp_server import MCPServer


def create_server() -> MCPServer:
    server = MCPServer(
        name="easyagent-real-python-text-server",
        description="Real text utility MCP server for integration tests",
    )

    def word_count(text: str) -> int:
        """Count words in text."""
        return len([w for w in text.split(" ") if w])

    def reverse_text(text: str) -> str:
        """Reverse the text string."""
        return text[::-1]

    def upper_text(text: str) -> str:
        """Convert text to upper case."""
        return text.upper()

    server.add_tool(word_count, name="word_count", description="Count words in text")
    server.add_tool(reverse_text, name="reverse_text", description="Reverse text")
    server.add_tool(upper_text, name="upper_text", description="Convert text to upper case")
    return server


if __name__ == "__main__":
    app = create_server()
    app.run(transport="stdio")
