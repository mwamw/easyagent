"""Real MCP usage example: connect to a Python MCP server script over stdio.

Run from project root:
    python mcp/examples/example_real_python_stdio_client.py
"""

import asyncio
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp import MCPClient


async def main() -> None:
    client = MCPClient(server_source=["python", "real_python_mcp_server.py"])

    try:
        await client.connect()

        tools = await client.list_tools()
        print("=== Discovered MCP tools (python stdio) ===")
        for tool in tools:
            print("-", tool["name"], "-", tool.get("description", ""),"-", tool.get("input_schema", []))

        print("\n=== Call echo ===")
        print(await client.call_tool("echo", {"text": "hello mcp"}))

        print("\n=== Call add ===")
        print(await client.call_tool("add", {"a": 7, "b": 9}))

        print("\n=== Call repeat ===")
        print(await client.call_tool("repeat", {"text": "easyagent", "times": 3}))
    finally:
        await client.disconnect()

    from Tool import MCPToolManager, MCPWrappedTool
    
if __name__ == "__main__":
    asyncio.run(main())
