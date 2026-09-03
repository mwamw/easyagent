import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录添加到路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from memory.V2.MemoryManage import MemoryManage
from memory import MemoryConfig
from Tool.builtin import (
    register_search_tool,
    register_web_fetch_tool,
)
from memory.V2 import EpisodicMemory
async def main():
    load_dotenv()
    
    llm = EasyLLM()
    registry = ToolRegistry()
    
    # 1. 初始化记忆后端
    memory_config = MemoryConfig()
    memory_manage = MemoryManage(memory_config,enable_episodic=False)
    # 2. 注册搜索、抓取和记忆工具
    register_search_tool(registry)
    register_web_fetch_tool(registry)

    agent = BasicAgent(
        name="Researcher",
        llm=llm,
    ).with_tool(registry).with_memory(memory_manage)
    # 真实场景 Query: 调研新技术并存入长期记忆
    query = """
    我想了解一下目前 Python 3.13 中关于 JIT (Just-In-Time) 编译器的最新进展：
    1. 搜索相关技术博客或官方文档（如 PEP 744）。
    2. 抓取 1 个核心网页的内容。
    3. 总结其实现原理（特别是 Copy-and-Patch 技术）。
    4. 将核心结论作为记忆保存
    5. 最后搜索一下记忆库，确认你已经成功记住了关于 'Python JIT' 的信息。
    """

    print(f"🕵️ 开始调研任务...\n")
    async for event in agent.astream(query):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
