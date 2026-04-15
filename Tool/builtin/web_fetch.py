"""Prompt-aware web fetch tool for Claude-style research workflows."""

from __future__ import annotations

import json
import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeWebFetchInput

logger = logging.getLogger(__name__)

MAX_WEBFETCH_CONTENT_CHARS = 40000
MAX_WEBFETCH_DISPLAY_CHARS = 12000
MAX_WEBFETCH_EXCERPTS = 8

WEB_FETCH_PROMPT = """用于抓取公开网页并根据 prompt 挑选最相关的正文片段。
- 适合在已经知道目标 URL 时读取网页正文，而不是搜索。
- prompt 应说明你想从页面里提取什么信息。
- 结果是网页正文摘录，不是自动验证过的事实。"""


try:
    import requests as _requests  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised via helper in tests
    _requests = None


def _import_requests():
    return _requests


def _clip_text(value: str, *, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    clipped = value[:max_chars].rstrip()
    return f"{clipped}\n\n...[truncated]", True


def _normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.split("\n")]
    normalized_lines: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            blank_pending = True
            continue
        if blank_pending and normalized_lines:
            normalized_lines.append("")
        normalized_lines.append(line)
        blank_pending = False
    return "\n".join(normalized_lines).strip()


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _prompt_terms(prompt: str) -> list[str]:
    ascii_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_]{3,}", prompt)]
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", prompt)
    ordered: list[str] = []
    seen: set[str] = set()
    for term in ascii_terms + cjk_terms:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def _score_paragraph(paragraph: str, prompt: str, terms: list[str]) -> int:
    lowered = paragraph.lower()
    score = 0
    prompt_lower = prompt.lower().strip()
    if prompt_lower and prompt_lower in lowered:
        score += 8
    for term in terms:
        if term in lowered:
            score += min(4, lowered.count(term))
    return score


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def _select_relevant_paragraphs(text: str, prompt: str, *, max_excerpts: int = MAX_WEBFETCH_EXCERPTS) -> list[str]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    terms = _prompt_terms(prompt)
    scored: list[tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        score = _score_paragraph(paragraph, prompt, terms)
        scored.append((score, index, paragraph))

    scored_matches = [item for item in scored if item[0] > 0]
    if scored_matches:
        top = sorted(scored_matches, key=lambda item: (-item[0], item[1]))[:max_excerpts]
        return [item[2] for item in sorted(top, key=lambda item: item[1])]

    return paragraphs[:max_excerpts]


class _SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if lowered == "title":
            self._in_title = True
        if lowered in {"p", "div", "section", "article", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lowered == "title":
            self._in_title = False
        if lowered in {"p", "div", "section", "article", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
        self._parts.append(data)

    def get_text(self) -> str:
        return _normalize_whitespace(unescape("".join(self._parts)))


def _extract_response_content(content_type: str, response_text: str) -> tuple[str, str]:
    lowered = (content_type or "").lower()
    if "html" in lowered:
        parser = _SimpleHTMLTextExtractor()
        parser.feed(response_text)
        return parser.title.strip(), parser.get_text()
    if "json" in lowered:
        try:
            parsed = json.loads(response_text)
            pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
            return "", pretty
        except json.JSONDecodeError:
            return "", response_text
    if lowered.startswith("text/") or "xml" in lowered or lowered == "":
        return "", _normalize_whitespace(response_text)
    raise ValueError(f"暂不支持的内容类型: {content_type or 'unknown'}")


def _format_web_fetch_output(
    *,
    url: str,
    final_url: str,
    title: str,
    prompt: str,
    excerpts: list[str],
    content_type: str,
) -> str:
    lines = [f"URL: {final_url}"]
    if final_url != url:
        lines.append(f"原始 URL: {url}")
    if title:
        lines.append(f"标题: {title}")
    if content_type:
        lines.append(f"内容类型: {content_type}")
    lines.append(f"提取目标: {prompt}")
    lines.append("")
    lines.append("相关正文摘录:")
    if not excerpts:
        lines.append("未提取到可用正文。")
    else:
        for index, excerpt in enumerate(excerpts, start=1):
            lines.append(f"{index}. {excerpt}")
    return "\n".join(lines).strip()


class WebFetchTool(Tool):
    """Fetch a public webpage and extract prompt-relevant text."""

    def __init__(
        self,
        *,
        request_timeout_s: int = 30,
        max_content_chars: int = MAX_WEBFETCH_CONTENT_CHARS,
        max_display_chars: int = MAX_WEBFETCH_DISPLAY_CHARS,
    ):
        self.request_timeout_s = request_timeout_s
        self.max_content_chars = max_content_chars
        self.max_display_chars = max_display_chars
        super().__init__(
            name="WebFetch",
            description="抓取公开网页正文，并根据 prompt 返回最相关的内容摘录。",
            parameters=ClaudeWebFetchInput,
            guidance="适合在已知 URL 的前提下读取网页正文。prompt 要明确说明想提取的主题、事实或片段。",
            prompt=WEB_FETCH_PROMPT,
            read_only=True,
            source="builtin",
            tags=["web", "fetch", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        url = str(parameters.get("url", "")).strip()
        prompt = str(parameters.get("prompt", "")).strip()

        if not url:
            return ToolResult.error("错误：URL 不能为空。", error_type="invalid_parameters")
        if not prompt:
            return ToolResult.error("错误：prompt 不能为空。", error_type="invalid_parameters")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResult.error("错误：仅支持 http/https URL。", error_type="invalid_parameters")

        requests_module = _import_requests()
        if requests_module is None:
            return ToolResult.error("错误：需要安装 requests 库", error_type="missing_dependency")

        try:
            response = requests_module.get(
                url,
                timeout=self.request_timeout_s,
                headers={
                    "User-Agent": "EasyAgent/1.0 (+https://github.com/openai/codex)",
                    "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1",
                },
            )
            response.raise_for_status()
        except Exception as exc:
            logger.error("网页抓取失败: %s", exc)
            return ToolResult.error(
                f"网页抓取失败: {exc}",
                error_type="web_fetch_failed",
                metadata={"url": url},
            )

        content_type = str(response.headers.get("Content-Type", "") or "")
        try:
            title, content = _extract_response_content(content_type, response.text)
        except ValueError as exc:
            return ToolResult.error(
                f"网页抓取失败: {exc}",
                error_type="unsupported_content_type",
                metadata={"url": url, "content_type": content_type},
            )

        if not content:
            return ToolResult.error(
                "网页抓取失败: 未提取到可用正文。",
                error_type="empty_content",
                metadata={"url": url, "content_type": content_type},
            )

        excerpts = _select_relevant_paragraphs(content, prompt)
        display_text = _format_web_fetch_output(
            url=url,
            final_url=str(getattr(response, "url", url) or url),
            title=title,
            prompt=prompt,
            excerpts=excerpts,
            content_type=content_type,
        )
        display_text, display_truncated = _clip_text(display_text, max_chars=self.max_display_chars)
        content, content_truncated = _clip_text(content, max_chars=self.max_content_chars)

        return ToolResult.success(
            display_text,
            structured_data={
                "url": url,
                "final_url": str(getattr(response, "url", url) or url),
                "hostname": _hostname(url),
                "title": title,
                "content_type": content_type,
                "status_code": getattr(response, "status_code", None),
                "prompt": prompt,
                "content": content,
                "excerpts": excerpts,
                "content_truncated": content_truncated,
                "display_truncated": display_truncated,
            },
            metadata={
                "url": url,
                "final_url": str(getattr(response, "url", url) or url),
                "content_type": content_type,
                "status_code": getattr(response, "status_code", None),
                "content_truncated": content_truncated,
                "display_truncated": display_truncated,
            },
        )


def register_web_fetch_tool(
    registry: ToolRegistry,
    *,
    request_timeout_s: int = 30,
    max_content_chars: int = MAX_WEBFETCH_CONTENT_CHARS,
    max_display_chars: int = MAX_WEBFETCH_DISPLAY_CHARS,
) -> WebFetchTool:
    tool = WebFetchTool(
        request_timeout_s=request_timeout_s,
        max_content_chars=max_content_chars,
        max_display_chars=max_display_chars,
    )
    registry.register_tool(tool)
    return tool


__all__ = [
    "WebFetchTool",
    "register_web_fetch_tool",
    "_select_relevant_paragraphs",
]
