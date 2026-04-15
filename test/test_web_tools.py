import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_web_fetch_tool
from Tool.builtin.search import WebSearchTool, _filter_results_by_domains
from Tool.builtin.web_fetch import WebFetchTool


class TestWebSearchDomainFiltering(unittest.TestCase):
    def test_filter_results_by_allowed_and_blocked_domains(self):
        results = [
            {"title": "A", "link": "https://docs.python.org/3/", "snippet": "Python docs"},
            {"title": "B", "link": "https://example.com/post", "snippet": "Example"},
            {"title": "C", "link": "https://blog.python.org/news", "snippet": "Python blog"},
        ]

        filtered = _filter_results_by_domains(
            results,
            allowed_domains=["python.org"],
            blocked_domains=["blog.python.org"],
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["link"], "https://docs.python.org/3/")

    def test_search_schema_includes_domain_filters(self):
        tool = WebSearchTool()
        schema = tool.get_openai_schema()
        properties = schema["function"]["parameters"]["properties"]

        self.assertIn("allowed_domains", properties)
        self.assertIn("blocked_domains", properties)


class FakeResponse:
    def __init__(self, text: str, *, url: str, status_code: int = 200, content_type: str = "text/html; charset=utf-8"):
        self.text = text
        self.url = url
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestWebFetchTool(unittest.TestCase):
    def test_web_fetch_extracts_relevant_html_content(self):
        html = """
        <html>
          <head><title>Example Article</title></head>
          <body>
            <article>
              <p>Intro paragraph about a product launch.</p>
              <p>The pricing section says the pro plan costs $20 per month.</p>
              <p>Contact details and footer text.</p>
            </article>
          </body>
        </html>
        """
        tool = WebFetchTool()

        with patch(
            "Tool.builtin.web_fetch._import_requests",
            return_value=SimpleNamespace(get=lambda *args, **kwargs: FakeResponse(html, url="https://example.com/article")),
        ):
            result = tool.run(
                {
                    "url": "https://example.com/article",
                    "prompt": "提取页面里的 pricing 和价格信息",
                }
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["title"], "Example Article")
        self.assertIn("pricing", result.to_display_string().lower())
        self.assertIn("$20", result.structured_data["content"])

    def test_web_fetch_supports_json_content(self):
        tool = WebFetchTool()
        response = FakeResponse('{"name":"demo","value":1}', url="https://api.example.com/data", content_type="application/json")

        with patch(
            "Tool.builtin.web_fetch._import_requests",
            return_value=SimpleNamespace(get=lambda *args, **kwargs: response),
        ):
            result = tool.run(
                {
                    "url": "https://api.example.com/data",
                    "prompt": "提取 name 字段",
                }
            )

        self.assertEqual(result.status, "success")
        self.assertIn('"name": "demo"', result.structured_data["content"])

    def test_web_fetch_rejects_invalid_url(self):
        tool = WebFetchTool()
        result = tool.run({"url": "file:///tmp/test.txt", "prompt": "读取内容"})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "invalid_parameters")

    def test_register_web_fetch_tool(self):
        registry = ToolRegistry()
        tool = register_web_fetch_tool(registry)

        self.assertIsInstance(tool, WebFetchTool)
        self.assertIn("WebFetch", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
