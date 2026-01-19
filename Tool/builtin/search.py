"""
搜索工具

支持 SerpAPI 和简单的 DuckDuckGo 搜索。
"""
import os
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

from ..BaseTool import Tool
from ..ToolRegistry import ToolRegistry

logger = logging.getLogger(__name__)


class SearchParams(BaseModel):
    """搜索参数"""
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, description="返回结果数量，默认5条")


class WebSearchTool(Tool):
    """
    网络搜索工具
    
    支持多种搜索后端：
    - SerpAPI（需要 API Key）
    - DuckDuckGo（免费，无需 API Key）
    
    Example:
        >>> tool = WebSearchTool()
        >>> result = tool.run({"query": "Python 教程", "num_results": 3})
        >>> print(result)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        backend: str = "auto"
    ):
        """
        初始化搜索工具
        
        Args:
            api_key: SerpAPI Key（可选，默认从环境变量读取）
            backend: 搜索后端 ("serpapi", "duckduckgo", "auto")
        """
        super().__init__(
            name="web_search",
            description="在互联网上搜索信息。输入搜索关键词，返回相关网页摘要。",
            parameters=SearchParams
        )
        
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        self.backend = backend
        
        # 自动选择后端
        if backend == "auto":
            if self.api_key:
                self.backend = "serpapi"
            else:
                self.backend = "duckduckgo"
        
        logger.info(f"WebSearchTool 初始化完成，后端: {self.backend}")
    
    def run(self, parameters: dict) -> str:
        """执行搜索"""
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 5)
        
        if not query:
            return "错误：搜索关键词不能为空"
        
        try:
            if self.backend == "serpapi":
                return self._search_serpapi(query, num_results)
            else:
                return self._search_duckduckgo(query, num_results)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return f"搜索失败: {str(e)}"
    
    def _search_serpapi(self, query: str, num_results: int) -> str:
        """使用 SerpAPI 搜索"""
        try:
            import requests
        except ImportError:
            return "错误：需要安装 requests 库"
        
        if not self.api_key:
            return "错误：SerpAPI 需要 API Key，请设置 SERPAPI_API_KEY 环境变量"
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 提取搜索结果
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        
        if not results:
            return "未找到相关结果"
        
        # 格式化输出
        output_lines = [f"搜索「{query}」的结果：\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(f"{i}. {r['title']}")
            output_lines.append(f"   链接: {r['link']}")
            output_lines.append(f"   摘要: {r['snippet']}\n")
        
        return "\n".join(output_lines)
    
    def _search_duckduckgo(self, query: str, num_results: int) -> str:
        """使用 DuckDuckGo 搜索（无需 API Key）"""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            # 如果没有安装 duckduckgo_search，使用简单的 HTTP 请求
            return self._search_duckduckgo_lite(query, num_results)
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        
        if not results:
            return "未找到相关结果"
        
        output_lines = [f"搜索「{query}」的结果：\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(f"{i}. {r['title']}")
            output_lines.append(f"   链接: {r['link']}")
            output_lines.append(f"   摘要: {r['snippet']}\n")
        
        return "\n".join(output_lines)
    
    def _search_duckduckgo_lite(self, query: str, num_results: int) -> str:
        """备用的简单 DuckDuckGo 搜索"""
        try:
            import requests
        except ImportError:
            return "错误：需要安装 requests 库"
        
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        
        # 获取摘要
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", query),
                "link": data.get("AbstractURL", ""),
                "snippet": data.get("Abstract", "")
            })
        
        # 获取相关话题
        for topic in data.get("RelatedTopics", [])[:num_results - len(results)]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:50],
                    "link": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")
                })
        
        if not results:
            return f"DuckDuckGo 未找到「{query}」的相关结果。建议安装 duckduckgo-search 获取更好的搜索结果。"
        
        output_lines = [f"搜索「{query}」的结果：\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(f"{i}. {r['title']}")
            if r['link']:
                output_lines.append(f"   链接: {r['link']}")
            output_lines.append(f"   摘要: {r['snippet']}\n")
        
        return "\n".join(output_lines)


def register_search_tool(
    registry: ToolRegistry,
    api_key: Optional[str] = None,
    backend: str = "auto"
) -> WebSearchTool:
    """
    注册搜索工具到 ToolRegistry
    
    Args:
        registry: 工具注册表
        api_key: SerpAPI Key（可选）
        backend: 搜索后端
        
    Returns:
        创建的 WebSearchTool 实例
    """
    tool = WebSearchTool(api_key=api_key, backend=backend)
    registry.registerTool(tool)
    return tool
