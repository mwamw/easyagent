"""Check local references, figures, code syntax, and generated reading artifacts."""

from __future__ import annotations

import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.images = []
        self.tables = []
        self._table = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs)
        if tag == "table":
            self._table = []
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def parse_markdown(path):
    result = subprocess.run(["pandoc", str(path), "--to=html5"], check=True, capture_output=True, text=True)
    page = Page()
    page.feed(result.stdout)
    return page


def check_links(page, path):
    assert len(page.ids) == len(set(page.ids)), f"Duplicate heading IDs in {path.name}"
    for href in page.links + [item["src"] for item in page.images]:
        parts = urlsplit(href)
        if parts.scheme or parts.netloc:
            continue
        target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
        assert target.exists(), f"Broken local reference: {path.name}: {href}"
        if parts.fragment:
            target_page = page if target == path else parse_markdown(target) if target.suffix == ".md" else None
            if target_page:
                assert unquote(parts.fragment) in target_page.ids, f"Broken anchor: {path.name}: {href}"


def main():
    for path in sorted(ROOT.glob("*.md")):
        page = parse_markdown(path)
        check_links(page, path)
        for example in re.findall(r"```python\n(.*?)\n```", path.read_text(), flags=re.S):
            ast.parse(example)
        print(f"PASS Markdown: {path.name}")
    for path in sorted((ROOT / "assets").glob("*.svg")):
        svg = ET.parse(path).getroot()
        assert svg.get("viewBox") and svg.get("role") == "img"
        assert svg.find("{http://www.w3.org/2000/svg}title") is not None
        assert svg.find("{http://www.w3.org/2000/svg}desc") is not None
        print(f"PASS Figure: {path.name}")
    html_path = ROOT / "easyagent-whitepaper.zh-CN.html"
    html = Page()
    html.feed(html_path.read_text())
    check_links(html, html_path)
    assert len(html.images) == 5
    assert all(item["src"].startswith("data:image/svg+xml") for item in html.images), "Figures must be embedded"
    assert all(item.get("alt") for item in html.images)
    print(f"PASS HTML: {len(html.images)} embedded figures; {len(html.ids)} anchors")
    assert len(html.tables) == 11, "Expected ten comparison/status tables and one responsibility table"
    for index, table in enumerate(html.tables, 1):
        assert table and len(table) > 1
        columns = len(table[0])
        assert all(len(row) == columns for row in table), f"Inconsistent columns in table {index}"
        assert all(cell for row in table for cell in row), f"Empty comparison cell in table {index}"
        if columns >= 4:
            assert "EasyAgent" in table[0][1], f"Expected EasyAgent in highlighted column: table {index}"
    dimension_tables = [table for table in html.tables if table[1][0].startswith("D01 ")]
    assert len(dimension_tables) == 1
    dimensions = [row[0].split()[0] for row in dimension_tables[0][1:]]
    assert dimensions == [f"D{index:02}" for index in range(1, 13)]
    source = (ROOT / "easyagent-whitepaper.zh-CN.md").read_text()
    assert len(re.findall(r"^\[S\d+\]:", source, flags=re.M)) == 26
    assert 'edition: "v0.3"' in source
    assert all(old not in html_path.read_text() for old in ("技术白皮书 v0.1", "技术白皮书 v0.2"))
    assert len(re.findall(r"```python\n", source)) == 7
    assert "从理念到实现：通过模块接口构建领域 Harness" in source
    print("PASS Matrices: 11 tables; 12 dimensions; 26 sources; 7 Python examples; edition v0.3")
    pdf = ROOT / "easyagent-whitepaper.zh-CN.pdf"
    if pdf.exists():
        result = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True)
        text = result.stdout
        assert "面向 Harness 编程" in text and "阅读与版本说明" in text
        assert "\ufffd" not in text, "Unexpected replacement characters in PDF text"
        pages = [page for page in text.split("\f") if page.strip()]
        for index, page in enumerate(pages, 1):
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            assert len(lines) >= 3, f"Suspicious empty PDF page: {index}"
            print(f"PDF {index:02}: {len(lines):2} lines | {lines[0][:65]}")
        print(f"PASS PDF: {len(pages)} nonempty pages; CJK text extractable")
        if result.stderr:
            print("NOTE pdftotext warning:", result.stderr.strip())


if __name__ == "__main__":
    main()
