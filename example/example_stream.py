import sys
import os
example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup context
from dotenv import load_dotenv
load_dotenv(os.path.join(example_dir, ".env"))
from typing import Any, Optional
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent

from core import enable_logging

enable_logging("INFO")   # 或 "DEBUG"

llm=EasyLLM()
agent=BasicAgent(name="test",llm=llm)

def demonstrate_stream_with_tool(agent):
    agent.clear_history()
    from Tool.builtin import CalculatorTool
    agent.with_tool()
    agent.add_tool(CalculatorTool())
    for event in agent.stream("你好，请介绍一下你自己，调用工具帮我计算 4^12+6*412"):
        print(event)
    print()


demonstrate_stream_with_tool(agent)
