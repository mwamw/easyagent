import sys
import os  
import json
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from Tool.BaseTool import Tool
from agent.BasicAgent import BasicAgent
from pydantic import BaseModel,Field
import serpapi
import logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    llm=EasyLLM(model="gemini-2.5-flash")
    print(f"provider: {llm.provide}, model: {llm.model}, base_url: {llm.base_url}, api_key: {llm.api_key}")
    tool_registry=ToolRegistry()
    
    
    class SearchParameters(BaseModel):
        query: str=Field(description="搜索查询")
    @tool_registry.tool("search","搜索工具",SearchParameters)
    def search(query: str) -> str:
        """
        一个基于SerpApi的实战网页搜索引擎工具。
        它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
        """
        print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
        try:
            api_key = os.getenv("SERPAPI_API_KEY")
            if not api_key:
                return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

            client = serpapi.Client(api_key=api_key)
            results = client.search({
                "engine": "google",
                "q": query,
                "gl": "cn",  # 国家代码
                "hl": "zh-cn", # 语言代码
            })
            
            # 智能解析：优先寻找最直接的答案
            if "answer_box_list" in results:
                return "\n".join(results["answer_box_list"])
            if "answer_box" in results and "answer" in results["answer_box"]:
                return results["answer_box"]["answer"]
            if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                return results["knowledge_graph"]["description"]
            if "organic_results" in results and results["organic_results"]:
                # 如果没有直接答案，则返回前三个有机结果的摘要
                snippets = [
                    f"[{i+1}] {res.get('title', '')}\n{res.get('snipapet', '')}"
                    for i, res in enumerate(results["organic_results"][:3])
                ]
                print("搜索成功")
                return "\n\n".join(snippets)
            
            return f"对不起，没有找到关于 '{query}' 的信息。"

        except Exception as e:
            return f"搜索时发生错误: {e}"


    basic_agent=BasicAgent("搜索助手",llm,tool_registry=tool_registry,description="搜索助手",system_prompt="你是一个搜索的助手，请用中文回答",verbose_thinking=False,enable_async_tool=True)
    basic_agent.set_enable_tool(True)
    print(basic_agent.invoke("GraphRAG是什么"))
    basic_agent.clear_history()
    print("------------------------------异步执行----------------------------------------")
    print(asyncio.run(basic_agent.invoke_async("GraphRAG是什么")))