
import json
import os
import sys
from typing import Any
from dotenv import load_dotenv

example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import mcptool


def main() -> None:
    load_dotenv()

    llm = EasyLLM()
    registry = ToolRegistry()
    workspace = os.path.abspath(".")
    tool = mcptool(
        server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
        tool_prefix="py_",
    )

    agent = BasicAgent(
        name="mcp-agent",
        llm=llm,
        enable_tool=True,
        tool_registry=registry,
    )
    agent.add_tool(tool)
    print(registry.list_tool_specs())
    try:
        agent.stream_invoke("列出当前文件夹下的文件")
    finally:
        tool.close()

import asyncio

main()
