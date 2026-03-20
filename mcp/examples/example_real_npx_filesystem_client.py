"""Real MCP usage example: connect to an npx MCP filesystem server.

Prerequisites:
- Node.js and npx available
- Internet access for first-time package fetch
- Node.js >= 18 (recommended)

Run from project root:
    python mcp/examples/example_real_npx_filesystem_client.py
"""

import asyncio
import os
import sys
import subprocess

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp import MCPClient


def _node_major_version() -> int:
    try:
        out = subprocess.check_output(["node", "-v"], text=True).strip()
        if out.startswith("v"):
            out = out[1:]
        return int(out.split(".")[0])
    except Exception:
        return 0


async def main() -> None:
    major = _node_major_version()
    if major < 18:
        print("Node.js version is too old for @modelcontextprotocol/server-filesystem.")
        print("Detected major version:", major)
        print("Please upgrade Node.js to >= 18 and rerun this example.")
        return

    workspace = os.path.abspath(".")
    client = MCPClient(
        server_source=[
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            workspace,
        ]
    )

    async with client:
        # await client.connect()
        tools = await client.list_tools()
        tool_names = [t["name"] for t in tools]

        print("=== Discovered MCP tools (npx filesystem) ===")
        for name in tool_names:
            print("-", name)

        # Common tool names in filesystem servers.
        candidate_calls = [
            ("list_directory", {"path": "."}),
            ("list_dir", {"path": "."}),
            ("read_file", {"path": "USAGE.md"}),
        ]

        print("\n=== Try one real filesystem tool call ===")
        for tool_name, params in candidate_calls:
            if tool_name in tool_names:
                print("calling:", tool_name, params)
                print(await client.call_tool(tool_name, params))
                break
        else:
            print("No known demo tool name found. Use listed tool names to call manually.")
    # finally:
    #     await client.disconnect()
    #     print("Client disconnected.")

def test_mcptoolmanage() -> None:

    from Tool import MCPToolManager, MCPWrappedTool
    workspace = os.path.abspath(".")
    manage = MCPToolManager(
        server_source=[
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            workspace,
        ]
    )

    print("=== Testing MCPToolManager ===")
    print(manage.list_remote_tools())
    # finally:
    #     await client.disconnect()
    #     print("Client disconnected.")
    
if __name__ == "__main__":
    test_mcptoolmanage()
