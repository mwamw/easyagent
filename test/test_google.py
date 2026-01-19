import os
import serpapi
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# 配置 API Key
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 定义搜索函数
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

        serpapi_client = serpapi.Client(api_key=api_key)
        results = serpapi_client.search({
            "engine": "google",
            "q": query,
            "gl": "cn",
            "hl": "zh-cn",
        })
        
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

# 使用新的 google.genai 包
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="查一下GraphRAG是什么",
    config=types.GenerateContentConfig(
        tools=[search],  # 直接传函数
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False  # 启用自动函数调用
        )
    )
)

print(f"\n✅ 最终回答:\n{response.text}")