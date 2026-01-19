import sys
import os  
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tool.ToolRegistry import ToolRegistry
from Tool.BaseTool import Tool
from pydantic import BaseModel,Field
import json
import serpapi
from core.llm import EasyLLM
class SearchParameters(BaseModel):
    query: str=Field(description="搜索查询")


llm=EasyLLM(model="gemini-2.5-pro")
registry=ToolRegistry()
@registry.tool("search","搜索工具",SearchParameters)
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
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

message=llm.invoke_with_tools([{"role":"system","content":"你是一个智能助手，能够帮助用户回答问题和完成任务,并拥有调用工具的能力,如果需要调用工具,请同时给出你的思考过程。"},{"role":"user","content":"解释一下LangGraph是什么"}],registry.get_openai_tools())
#查看message的所有key
print(message.dict().keys())
print(f"content:{message.content}")
if message.tool_calls:
    for function_call in message.tool_calls:
        print(f"function_call:{function_call}")