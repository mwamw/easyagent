from ast import pattern
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
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    register_search_tool,
    register_calculator_tool,
    register_filesystem_tools,
    register_file_write_tool,
    register_file_edit_tool,
    register_shell_tools,
    register_task_output_tool,
    register_task_stop_tool,
    register_web_fetch_tool,
    register_todo_write_tool,
    register_agent_tool,
    register_notebook_edit_tool,
    register_ask_user_question_tool,
    register_exit_plan_mode_tool,
    register_config_tool,
    register_worktree_tools,
)
from Tool.runtime import WorktreeManager

enable_logging()

def create_agent(registry: ToolRegistry, name: str) -> BasicAgent:
    """辅助函数，用于创建配置好的 Agent 实例"""
    llm = EasyLLM(provider="openai") 
    return BasicAgent(
        name=name,
        llm=llm,
        tool_registry=registry,
        verbose_thinking=True,
        enable_tool=True
    )

async def example_calculator_tool():
    """示例：使用计算器工具"""
    print("\n" + "="*50)
    print("🚀 示例: Calculator Tool\n")
    registry = ToolRegistry()
    register_calculator_tool(registry)
    agent = create_agent(registry, "CalcAgent")
    
    query = "请帮我计算 3456 的平方根加上 128 乘以 99 的结果是多少"
    await agent.astream_invoke(query)

async def example_search_tool():
    """示例：使用网络搜索工具"""
    print("\n" + "="*50)
    print("🚀 示例: Web Search Tool\n")
    registry = ToolRegistry()
    register_search_tool(registry)
    agent = create_agent(registry, "SearchAgent")
    
    query = "请搜索一下 'EasyAgent AI 框架' 的相关信息，并总结一下"
    await agent.astream_invoke(query)

async def example_filesystem_tools():
    """示例：使用文件系统读取/搜索工具 (FileRead, Glob, Grep)"""
    print("\n" + "="*50)
    print("🚀 示例: Filesystem Tools\n")
    registry = ToolRegistry()
    workspace = os.path.dirname(__file__)
    register_filesystem_tools(registry, workspace_root=workspace)
    agent = create_agent(registry, "FSAgent")
    
    query = "请查看当前 example 目录下的所有 python 文件，并找到包含 'enable_logging' 的文件"
    await agent.astream_invoke(query)

async def example_file_write_tool():
    """示例：使用文件写入工具 (FileWrite)"""
    print("\n" + "="*50)
    print("🚀 示例: File Write Tool\n")
    registry = ToolRegistry()
    workspace = os.path.join(os.path.dirname(__file__), "scratch")
    os.makedirs(workspace, exist_ok=True)
    register_file_write_tool(registry, workspace_root=workspace)
    agent = create_agent(registry, "WriteAgent")
    
    query = "请在 scratch 目录下创建一个 hello.txt 文件，内容为 'Hello World from File Write Tool'"
    await agent.astream_invoke(query)

async def example_file_edit_tool():
    """示例：使用文件编辑工具 (FileEdit)"""
    print("\n" + "="*50)
    print("🚀 示例: File Edit Tool\n")
    registry = ToolRegistry()
    workspace = os.path.join(os.path.dirname(__file__), "scratch")
    os.makedirs(workspace, exist_ok=True)
    
    # 准备一个待编辑的文件
    test_file = os.path.join(workspace, "target_edit.py")
    with open(test_file, 'w') as f:
        f.write("def add(a, b):\n    return a + b\n")

    # 注册编辑工具，同时也注册文件系统工具以便 Agent 在编辑前先读取和定位文件
    register_file_edit_tool(registry, workspace_root=workspace)
    register_filesystem_tools(registry, workspace_root=workspace)
    agent = create_agent(registry, "EditAgent")
    
    query = "请修改 scratch 目录下的 target_edit.py，给 add 函数加上具体的类型注解"
    await agent.astream_invoke(query)

async def example_shell_tools():
    """示例：使用终端 Shell 工具 (Bash)"""
    print("\n" + "="*50)
    print("🚀 示例: Shell Tools\n")
    registry = ToolRegistry()
    workspace = os.path.dirname(__file__)
    register_shell_tools(registry, workspace_root=workspace)
    agent = create_agent(registry, "ShellAgent")
    
    query = "告诉我当前目录下有哪些内容"
    await agent.astream_invoke(query)

async def example_web_fetch_tool():
    """示例：使用抓取网页工具 (WebFetch)"""
    print("\n" + "="*50)
    print("🚀 示例: Web Fetch Tool\n")
    registry = ToolRegistry()
    register_web_fetch_tool(registry)
    agent = create_agent(registry, "FetchAgent")
    
    query = "请抓取 https://example.com 的页面内容，并提取页面上的标题和主要段落"
    await agent.astream_invoke(query)

async def example_todo_write_tool():
    """示例：使用 TODO 写入工具"""
    print("\n" + "="*50)
    print("🚀 示例: Todo Write Tool\n")
    registry = ToolRegistry()
    # 临时覆盖全局根目录以防污染主项目
    register_todo_write_tool(registry)
    agent = create_agent(registry, "TodoAgent")
    
    query = "我们在开发新功能，请帮我把 '重构数据库连接池', '完善单元测试' 记录到 TODO 中"
    await agent.astream_invoke(query)

async def example_task_management_tools():
    """示例：使用任务管理工具 (TaskOutput, TaskStop)"""
    print("\n" + "="*50)
    print("🚀 示例: Task Management Tools\n")
    registry = ToolRegistry()
    register_task_output_tool(registry)
    register_task_stop_tool(registry)
    agent = create_agent(registry, "TaskAgent")
    
    query = "请帮我总结一段关于 AI Agent 的优缺点，并使用任务输出工具输出最终结果。如果发现无法完成，请直接停止任务并陈述原因。"
    await agent.astream_invoke(query)

async def example_interaction_tools():
    """示例：使用交互工具 (AskUserQuestion, ExitPlanMode)"""
    print("\n" + "="*50)
    print("🚀 示例: Interaction Tools\n")
    registry = ToolRegistry()
    register_ask_user_question_tool(registry)
    register_exit_plan_mode_tool(registry)
    agent = create_agent(registry, "PlanAgent")
    
    query = "你现在处于计划模式中。我需要你处理一些模糊的需求。请主动调用询问用户工具来获取更多信息，并在问题得到解答后退出计划模式。"
    await agent.astream_invoke(query)

async def example_config_tool():
    """示例：使用配置访问工具 (ConfigTool)"""
    print("\n" + "="*50)
    print("🚀 示例: Config Tool\n")
    registry = ToolRegistry()
    register_config_tool(registry)
    agent = create_agent(registry, "ConfigAgent")
    
    query = "请使用配置工具读取一项名为 'model' 或 'api_key' 的配置信息，如果未知则尝试读取 'test_key' 看看有哪些可用配置。"
    await agent.astream_invoke(query)

async def example_worktree_tools():
    """示例：使用 Git Worktree 切换工具"""
    print("\n" + "="*50)
    print("🚀 示例: Worktree Tools\n")
    registry = ToolRegistry()
    
    workspace = os.path.dirname(__file__)
    manager = WorktreeManager(workspace)
    register_worktree_tools(registry, worktree_manager=manager)
    agent = create_agent(registry, "GitAgent")
    
    query = "请尝试查看当前是否在 git worktree 中"
    await agent.astream_invoke(query)

async def example_notebook_edit_tool():
    """示例：使用 Jupyter Notebook 编辑工具"""
    print("\n" + "="*50)
    print("🚀 示例: Notebook Edit Tool\n")
    registry = ToolRegistry()
    workspace = os.path.join(os.path.dirname(__file__), "scratch")
    os.makedirs(workspace, exist_ok=True)
    register_notebook_edit_tool(registry, workspace_root=workspace)
    agent = create_agent(registry, "NotebookAgent")
    
    query = "请在 scratch 目录下创建一个简单的 test.ipynb 文件，并为其添加一个打印 Hello 的代码 cell"
    await agent.astream_invoke(query)

async def example_agent_tool():
    """示例：使用 Agent 调用工具 (让 Agent 可以调用具有特定职能的子 Agent)"""
    print("\n" + "="*50)
    print("🚀 示例: Agent Tool\n")
    registry = ToolRegistry()
    # 假设我们有一个专门处理数学运算的子Agent工具
    register_agent_tool(registry)
    agent = create_agent(registry, "ManagerAgent")
    
    query = "请分配一个子 Agent 去帮我写一篇关于 Python 协程的文章大纲"
    await agent.astream_invoke(query)


async def main():
    load_dotenv()
    
    print("准备运行 EasyAgent 内置工具示例...\n")
    print("取消注释下方相应的行以运行对应工具的示例：\n")
    # from Tool.builtin import GrepTool
    # await example_calculator_tool()
    # await example_search_tool()
    # await example_filesystem_tools()
    # await example_file_write_tool()
    # await example_file_edit_tool()
    # await example_shell_tools()
    # await example_web_fetch_tool()
    # await example_todo_write_tool()
    # await example_task_management_tools()
    # await example_interaction_tools()
    # await example_config_tool()
    # await example_worktree_tools()
    # await example_notebook_edit_tool()
    await example_agent_tool()
    
    print("请打开 example/example_builtin_tools.py，取消 main() 中需要的注释后即可执行体验。")


if __name__ == "__main__":
    asyncio.run(main())
