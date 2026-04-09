import os
import sys

example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(example_dir, ".env"))

from agent.BasicAgent import BasicAgent
from core import enable_logging
from core.llm import EasyLLM
from Tool.builtin import CalculatorTool

enable_logging("WARNING")
def main() -> None:

    llm = EasyLLM()
    agent = BasicAgent(
        name="stream-display-demo",
        llm=llm,
        verbose_thinking=True,
    )
    agent.with_tool()
    agent.add_tool(CalculatorTool())
    from Tool.builtin import WebSearchTool
    agent.add_tool(WebSearchTool())
    query = "计算 4^12 + 6*412，并查询graphrag的概念"
    agent.stream_invoke(query)
    

if __name__ == "__main__":
    main()
