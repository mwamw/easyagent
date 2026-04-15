import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录添加到路径，确保可以导入 EasyAgent
example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core import enable_logging
enable_logging()
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    register_filesystem_tools,
    register_file_edit_tool,
    register_file_write_tool,
    register_shell_tools,
    register_todo_write_tool
)

async def main():
    # 1. 加载环境变量 (需在 .env 中配置 LLM_API_KEY, LLM_MODEL 等)
    load_dotenv()
    
    # 2. 初始化引擎与工具
    llm = EasyLLM() 
    registry = ToolRegistry()
    
    # 注册核心编码工具集
    workspace = os.getcwd()
    register_filesystem_tools(registry, workspace_root=workspace)
    register_file_edit_tool(registry, workspace_root=workspace)
    register_file_write_tool(registry, workspace_root=workspace)
    register_shell_tools(registry, workspace_root=workspace)
    register_todo_write_tool(registry)

    # 3. 创建 Agent
    agent = BasicAgent(
        name="Coder",
        llm=llm,
        tool_registry=registry,
        verbose_thinking=True,
        enable_tool=True
    )

    # 4. 真实场景 Query
    query = """
   tmp_test_logic.py运行结果不对，是什么原因，你帮我修复一下。
    """

    print(f"🚀 开始任务: {query}\n")

    # 5. 使用 astream_invoke 进行流式调用
    await agent.astream_invoke(query)


if __name__ == "__main__":
    asyncio.run(main())
