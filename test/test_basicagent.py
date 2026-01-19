import sys
import os  
import json
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import BaseAgent
from core.llm import EasyLLM
from core.Message import Message
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from Tool.BaseTool import Tool
from agent.BasicAgent import BasicAgent
from pydantic import BaseModel,Field
import serpapi


if __name__ == "__main__":
    llm=EasyLLM()
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
                    f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(results["organic_results"][:3])
                ]
                print("搜索成功")
                return "\n\n".join(snippets)
            
            return f"对不起，没有找到关于 '{query}' 的信息。"

        except Exception as e:
            return f"搜索时发生错误: {e}"


    basic_agent=BasicAgent("搜索助手",llm,tool_registry=tool_registry,description="搜索助手")
    # print(basic_agent.invoke("你好，我是迈克尔，我来自中国，我今年25岁，我是一名学生，我喜欢打篮球和游泳，你是谁？"))
    # print(basic_agent.invoke("你还记得我叫什么名字吗"))
    # print(basic_agent.get_history())
    # basic_agent.set_enable_tool(True)
    # print(basic_agent.invoke("现在最厉害的篮球明星是谁？"))
    # print(basic_agent.invoke("他多大了？"))
    # basic_agent.set_enable_tool(False)
    
    basic_agent.stream_invoke("graphrag是什么")    