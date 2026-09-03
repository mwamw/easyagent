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
from Tool.builtin import register_calculator_tool

async def main():
    load_dotenv()
    
    llm = EasyLLM()
    registry = ToolRegistry()
    
    # 注册计算器工具
    register_calculator_tool(registry)

    agent = BasicAgent(
        name="MathAnalyst",
        llm=llm,
    ).with_tool(registry)

    # 复杂计算场景
    query = """
    假设我有一个复杂的金融模型计算需求：
    1. 计算 (1.05 的 10 次方) 乘以 5000 的精确结果。
    2. 然后计算这个结果的平方根再乘以 pi (圆周率)。
    3. 最后告诉我：如果我每年复利 5%，10 年后 5000 元会变成多少？并给出精确到小数点后 4 位的计算结果。
    请务必调用计算器工具进行每一步计算，不要仅凭直觉。
    """

    print(f"🔢 开始数值计算...\n")
    async for event in agent.astream(query):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
