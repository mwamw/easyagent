
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
from Tool.builtin import mcptool


def main() -> None:
    load_dotenv()

    llm = EasyLLM()
    workspace = os.path.abspath(".")
    tool = mcptool(
        server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", workspace],
        tool_prefix="py_",
        # auto_connect=False,
        include_resources=True,
        resource_tool_prefix="py_",
    )
    # tool.connect()
    # print(tool.snapshot().tools)
    agent = BasicAgent(
        name="mcp-agent",
        llm=llm,
    ).with_mcp(tool)
    try:
        for event in agent.stream("列出当前文件夹下的文件"):
            print(event)
    finally:
        agent.close()

import asyncio

main()
