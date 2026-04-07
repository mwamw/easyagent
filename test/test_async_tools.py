import sys
import os
import time
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
import logging

logging.basicConfig(level=logging.INFO)

# ================= 准备工作：注册两个会产生延时的工具 =================

tool_registry = ToolRegistry()

class WeatherParams(BaseModel):
    city: str = Field(description="需要查询天气的城市名称")

@tool_registry.tool("get_weather", "查询某城市当前天气的工具", WeatherParams)
def get_weather(city: str) -> str:
    print(f"☁️ [天气工具] 开始查询 {city} 的天气... (预计耗时2秒)")
    # 模拟耗时网络请求（使用同步 sleep，框架底层的 _async_safe_execute_tool 会把它放入线程池执行）
    time.sleep(10)
    print(f"☁️ [天气工具] 查询完成！")
    return f"{city} 今天晴朗，气温 25°C"


class StockParams(BaseModel):
    symbol: str = Field(description="需要查询股票的名称或代码")

@tool_registry.tool("get_stock", "查询某只股票当前价格的工具", StockParams)
def get_stock(symbol: str) -> str:
    print(f"📈 [股票工具] 开始查询 {symbol} 的股价... (预计耗时2秒)")
    time.sleep(10)
    print(f"📈 [股票工具] 查询完成！")
    return f"{symbol} 当前股价为 158.5 USD，上涨 2.1%"

# ================= 核心测试代码 =================

async def main():
    print("="*60)
    print("🚀 开始多工具并发异步测试")
    print("="*60)
    
    llm = EasyLLM()
    # 创建 BasicAgent
    agent = BasicAgent(
        name="Concurrent Helper",
        llm=llm,
        tool_registry=tool_registry,
        system_prompt="你是一个全能助手，如果有多个需求，请同时并发调用相应的工具获取信息。然后在获得信息后汇总成一句话回答。",
        verbose_thinking=False
    )
    agent.set_enable_tool(True)
    
    query = "请同时告诉我北京的当前天气，以及苹果公司(AAPL)的股价情况。"
    
    print(f"🙋 User >> {query}")
    print("-" * 60)
    
    start_time = time.time()
    
    # 异步调用模型，触发多工具并发
    response = await agent.ainvoke(query)
    
    end_time = time.time()
    duration1 = end_time - start_time
    
    start_time1 = time.time()
    response1=agent.invoke(query)
    end_time1 = time.time()
    duration2 = end_time1 - start_time1
    print("-" * 60)
    print(f"✨ Agent_async >> {response}")
    print("="*60)
    print(f"✨ Agent_sync >> {response1}")
    print("="*60)
    
    print(f"⏱️ 耗时分析:")
    print(f"整个 ainvoke 流程理论上只花费了略微大于2秒的时间。")
    print(f"如果工具是串行执行的，单单是两个工具的等待时间加起来至少需要 4 秒。")
    print(f"async实测执行总耗时: {duration1:.2f} 秒")
    print(f"sync实测执行总耗时: {duration2:.2f} 秒")
    
    if duration1 < duration2:
        print("\n✅ 测试通过！成功验证了框架的多工具并发异步调度能力！")
    else:
        print("\n⚠️ 耗时似乎较长，请检查底层 LLM 是否确实进行了一次同时的并发工具调用 (tool_calls)。")

if __name__ == "__main__":
    asyncio.run(main())
