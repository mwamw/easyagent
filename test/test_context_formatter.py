"""
上下文格式化器测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.window import ContextItem
from context.formatter.plain import PlainFormatter
from context.formatter.xml import XMLFormatter
from context.formatter.markdown import MarkdownFormatter
from manual_test_runner import run_manual_tests, exit_with_status


def make_item(content, source="rag", metadata=None):
    return ContextItem(
        content=content, source=source,
        metadata=metadata or {}, token_count=10
    )


class TestPlainFormatter(unittest.TestCase):
    """PlainFormatter 测试"""

    def test_basic_format(self):
        """基本纯文本格式"""
        fmt = PlainFormatter()
        items = [make_item("内容一"), make_item("内容二")]
        result = fmt.format(items, source="rag")
        self.assertIn("【参考资料】", result)
        self.assertIn("1. 内容一", result)
        self.assertIn("2. 内容二", result)

    def test_no_numbering(self):
        """无编号模式"""
        fmt = PlainFormatter(numbered=False)
        items = [make_item("内容一")]
        result = fmt.format(items, source="rag")
        self.assertNotIn("1.", result)
        self.assertIn("内容一", result)

    def test_empty_items(self):
        """空列表"""
        fmt = PlainFormatter()
        result = fmt.format([], source="rag")
        self.assertEqual(result, "")

    def test_format_all(self):
        """多来源格式化"""
        fmt = PlainFormatter()
        groups = {
            "rag": [make_item("检索结果")],
            "memory": [make_item("记忆内容", source="memory")],
        }
        result = fmt.format_all(groups)
        self.assertIn("参考资料", result)
        self.assertIn("记忆上下文", result)


class TestXMLFormatter(unittest.TestCase):
    """XMLFormatter 测试"""

    def test_basic_format(self):
        """基本 XML 格式"""
        fmt = XMLFormatter()
        items = [make_item("内容一"), make_item("内容二")]
        result = fmt.format(items, source="rag")
        self.assertIn("<rag>", result)
        self.assertIn("</rag>", result)
        self.assertIn('<item index="1">', result)
        self.assertIn("内容一", result)

    def test_xml_escaping(self):
        """XML 特殊字符转义"""
        fmt = XMLFormatter()
        items = [make_item("a < b & c > d")]
        result = fmt.format(items, source="context")
        self.assertIn("&lt;", result)
        self.assertIn("&amp;", result)
        self.assertIn("&gt;", result)

    def test_source_attribute(self):
        """来源属性"""
        fmt = XMLFormatter()
        items = [make_item("内容", metadata={"source": "/docs/test.txt"})]
        result = fmt.format(items, source="rag")
        self.assertIn('source="/docs/test.txt"', result)

    def test_empty_items(self):
        """空列表"""
        fmt = XMLFormatter()
        result = fmt.format([], source="rag")
        self.assertEqual(result, "")

    def test_no_source_tag(self):
        """无来源标识时使用默认 tag"""
        fmt = XMLFormatter()
        items = [make_item("内容")]
        result = fmt.format(items)
        self.assertIn("<context>", result)
        self.assertIn("</context>", result)


class TestMarkdownFormatter(unittest.TestCase):
    """MarkdownFormatter 测试"""

    def test_basic_format(self):
        """基本 Markdown 格式"""
        fmt = MarkdownFormatter()
        items = [make_item("内容一"), make_item("内容二")]
        result = fmt.format(items, source="rag")
        self.assertIn("## 参考资料", result)
        self.assertIn("- 内容一", result)

    def test_custom_heading_level(self):
        """自定义标题级别"""
        fmt = MarkdownFormatter(heading_level=3)
        items = [make_item("内容")]
        result = fmt.format(items, source="rag")
        self.assertIn("### 参考资料", result)

    def test_source_info(self):
        """来源信息"""
        fmt = MarkdownFormatter()
        items = [make_item("内容", metadata={"source": "/test.txt"})]
        result = fmt.format(items, source="rag")
        self.assertIn("来源: /test.txt", result)

    def test_empty_items(self):
        """空列表"""
        fmt = MarkdownFormatter()
        result = fmt.format([])
        self.assertEqual(result, "")


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestPlainFormatter,
            TestXMLFormatter,
            TestMarkdownFormatter,
        ],
        title="Context Formatter Manual Test",
    )
    exit_with_status(ok)
