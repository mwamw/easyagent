import os
import sys

from dotenv import load_dotenv
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
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
    )
    agent = BasicAgent(name="mcp-agent", llm=llm).with_mcp(tool)

    try:
        print(agent.invoke("查看现在文件夹下有哪些文件"))
    finally:
        tool.close()


if __name__ == "__main__":
    main()
