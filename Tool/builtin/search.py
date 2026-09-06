"""
搜索工具

支持 SerpAPI 和简单的 DuckDuckGo 搜索。
"""
import os
import logging
from urllib.parse import urlparse
from typing import Optional
from pydantic import BaseModel, Field

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .input_normalization import normalize_domain_filter, normalize_generic_input

logger = logging.getLogger(__name__)


class SearchParams(BaseModel):
    """Web search parameters."""
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results; defaults to 5")
    allowed_domains: list[str] = Field(default_factory=list, description="Domains to include")
    blocked_domains: list[str] = Field(default_factory=list, description="Domains to exclude")


SEARCH_TOOL_PROMPT = """Allows the agent to search the web and use the results to inform responses.
- Provides up-to-date information for current events and recent data.
- Returns search results with titles, URLs, and snippets.
- Use this tool for information beyond the model's knowledge or when current facts are needed.

CRITICAL REQUIREMENT:
- After answering a question using search results, include a "Sources:" section.
- List relevant URLs from the search results as Markdown hyperlinks: [Title](URL).
- Never invent sources or treat search snippets as automatically verified facts.

Usage notes:
- Domain filtering is supported through allowed_domains and blocked_domains.
- Set num_results to control the number of results.
- Use the current year when searching for recent documentation or current events. Refer to the current-date context, not an assumed year."""


def _format_search_output(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return "未找到相关结果"

    output_lines = [f"搜索「{query}」的结果：\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(f"{i}. {r.get('title', '')}")
        if r.get("link"):
            output_lines.append(f"   链接: {r['link']}")
        output_lines.append(f"   摘要: {r.get('snippet', '')}\n")
    return "\n".join(output_lines)


def _normalize_domain_filters(domains: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for domain in domains or []:
        value = normalize_domain_filter(domain)
        if not value:
            continue
        normalized.append(value)
    return normalized


def _hostname_from_url(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _domain_matches(hostname: str, domain: str) -> bool:
    if not hostname or not domain:
        return False
    return hostname == domain or hostname.endswith(f".{domain}")


def _filter_results_by_domains(
    results: list[dict[str, str]],
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> list[dict[str, str]]:
    allowed = _normalize_domain_filters(allowed_domains)
    blocked = _normalize_domain_filters(blocked_domains)
    filtered: list[dict[str, str]] = []
    seen_links: set[str] = set()

    for item in results:
        link = str(item.get("link", "") or "")
        hostname = _hostname_from_url(link)

        if link and link in seen_links:
            continue
        if blocked and hostname and any(_domain_matches(hostname, domain) for domain in blocked):
            continue
        if allowed:
            if not hostname:
                continue
            if not any(_domain_matches(hostname, domain) for domain in allowed):
                continue

        if link:
            seen_links.add(link)
        filtered.append(item)

    return filtered


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
            description="Searches the web and returns relevant page summaries.",
            parameters=SearchParams,
            guidance="Use for current external information; verify search results before relying on them.",
            prompt=SEARCH_TOOL_PROMPT,
            read_only=True,
            source="builtin",
            tags=["search", "web"],
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
    
    def run(self, parameters: dict) -> ToolResult:
        """执行搜索"""
        query = normalize_generic_input(parameters.get("query", ""))
        num_results = parameters.get("num_results", 5)
        allowed_domains = list(parameters.get("allowed_domains") or [])
        blocked_domains = list(parameters.get("blocked_domains") or [])
        
        if not query:
            return ToolResult.error("错误：搜索关键词不能为空", error_type="invalid_parameters")
        
        try:
            if self.backend == "serpapi":
                return self._search_serpapi(query, num_results, allowed_domains, blocked_domains)
            else:
                return self._search_duckduckgo(query, num_results, allowed_domains, blocked_domains)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return ToolResult.error(
                f"搜索失败: {str(e)}",
                error_type="search_failed",
                metadata={
                    "backend": self.backend,
                    "query": query,
                    "allowed_domains": allowed_domains,
                    "blocked_domains": blocked_domains,
                },
            )
    
    def _search_serpapi(
        self,
        query: str,
        num_results: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """使用 SerpAPI 搜索"""
        try:
            import requests
        except ImportError:
            return ToolResult.error("错误：需要安装 requests 库", error_type="missing_dependency")
        
        if not self.api_key:
            return ToolResult.error(
                "错误：SerpAPI 需要 API Key，请设置 SERPAPI_API_KEY 环境变量",
                error_type="missing_api_key",
            )
        
        url = "https://serpapi.com/search"
        requested_num = max(num_results, min(num_results * 4, 20)) if (allowed_domains or blocked_domains) else num_results
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": requested_num
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 提取搜索结果
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        results = _filter_results_by_domains(
            results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )[:num_results]
        
        return ToolResult.success(
            _format_search_output(query, results),
            structured_data=results,
            metadata={
                "backend": "serpapi",
                "query": query,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
            },
        )
    
    def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """使用 DuckDuckGo 搜索（无需 API Key）"""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            # 如果没有安装 duckduckgo_search，使用简单的 HTTP 请求
            return self._search_duckduckgo_lite(query, num_results, allowed_domains, blocked_domains)
        
        results = []
        requested_num = max(num_results, min(num_results * 4, 20)) if (allowed_domains or blocked_domains) else num_results
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=requested_num):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        results = _filter_results_by_domains(
            results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )[:num_results]
        
        return ToolResult.success(
            _format_search_output(query, results),
            structured_data=results,
            metadata={
                "backend": "duckduckgo",
                "query": query,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
            },
        )
    
    def _search_duckduckgo_lite(
        self,
        query: str,
        num_results: int,
        allowed_domains: list[str],
        blocked_domains: list[str],
    ) -> ToolResult:
        """备用的简单 DuckDuckGo 搜索"""
        try:
            import requests
        except ImportError:
            return ToolResult.error("错误：需要安装 requests 库", error_type="missing_dependency")
        
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
        results = _filter_results_by_domains(
            results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )[:num_results]
        
        if not results:
            return ToolResult.success(
                f"DuckDuckGo 未找到「{query}」的相关结果。建议安装 duckduckgo-search 获取更好的搜索结果。",
                structured_data=[],
                metadata={
                    "backend": "duckduckgo_lite",
                    "query": query,
                    "allowed_domains": allowed_domains,
                    "blocked_domains": blocked_domains,
                },
            )

        return ToolResult.success(
            _format_search_output(query, results),
            structured_data=results,
            metadata={
                "backend": "duckduckgo_lite",
                "query": query,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
            },
        )


def register_search_tool(
    registry: ToolRegistry,
    api_key: Optional[str] = None,
    backend: str = "auto",
    *,
    expose_in_deferred: bool | None = True,
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
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool
