import os
from dotenv import load_dotenv
load_dotenv()
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_mcp_tools,mcptool

llm = EasyLLM()
registry = ToolRegistry()
workspace = os.path.abspath(".")

 
tool = mcptool(
    server_source=["npx","-y","@modelcontextprotocol/server-filesystem",workspace,],
    tool_prefix="py_",
)

agent = BasicAgent(
        name="mcp-agent",
        llm=llm,
        enable_tool=True,
        tool_registry=registry,
)
agent.addTool(tool)
# print(agent.tool_registry.get_tools_description())
# print(agent.invoke("请调用 py_add 计算 12 + 30"))
print(agent.invoke("查看现在文件夹下有哪些文件"))
tool.close()