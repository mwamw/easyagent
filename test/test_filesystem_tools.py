import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.filesystem import (
    FileReadTool,
    GlobTool,
    GrepTool,
    register_filesystem_tools,
)
from Tool.runtime import clear_file_read_timestamps, get_recorded_file_version


class FilesystemToolsTestCase(unittest.TestCase):
    def setUp(self):
        clear_file_read_timestamps()
        self.addCleanup(clear_file_read_timestamps)
        self.workspace = tempfile.TemporaryDirectory()
        self.outside = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.addCleanup(self.outside.cleanup)
        self.root = self.workspace.name

        os.makedirs(os.path.join(self.root, "src", "utils"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "docs"), exist_ok=True)

        with open(os.path.join(self.root, "notes.txt"), "w", encoding="utf-8") as handle:
            handle.write("alpha\nbeta\ngamma\ndelta\nepsilon\n")

        with open(os.path.join(self.root, "src", "main.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "def main():\n"
                "    # TODO: implement greeting\n"
                "    print('hello world')\n"
                "    # TODO: add cli\n"
            )

        with open(os.path.join(self.root, "src", "utils", "helper.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "def helper():\n"
                "    return 'helper'\n"
            )

        with open(os.path.join(self.root, "docs", "README.md"), "w", encoding="utf-8") as handle:
            handle.write("# EasyAgent\n\nTODO: document tools.\n")

        with open(os.path.join(self.outside.name, "outside.txt"), "w", encoding="utf-8") as handle:
            handle.write("secret\n")

    def test_file_read_reads_windowed_text(self):
        tool = FileReadTool(workspace_root=self.root)
        result = tool.run({"file_path": "notes.txt", "offset": 2, "limit": 2})

        self.assertEqual(result.status, "success")
        self.assertIn("2 | beta", result.to_display_string())
        self.assertIn("3 | gamma", result.to_display_string())
        self.assertEqual(result.structured_data["start_line"], 2)
        self.assertEqual(result.structured_data["returned_lines"], 2)
        self.assertEqual(result.metadata["file_path"], os.path.join(self.root, "notes.txt"))
        self.assertIsNotNone(get_recorded_file_version(os.path.join(self.root, "notes.txt")))

    def test_file_read_rejects_escape_path(self):
        tool = FileReadTool(workspace_root=self.root)
        outside_file = os.path.join(self.outside.name, "outside.txt")
        result = tool.run({"file_path": outside_file})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "invalid_path")
        self.assertIn("读取文件失败", result.to_display_string())

    def test_glob_matches_recursive_python_files(self):
        tool = GlobTool(workspace_root=self.root)
        result = tool.run({"pattern": "**/*.py", "path": "src"})

        self.assertEqual(result.status, "success")
        matches = result.structured_data["matches"]
        self.assertEqual(len(matches), 2)
        relative_paths = {item["relative_path"] for item in matches}
        self.assertEqual(relative_paths, {"main.py", os.path.join("utils", "helper.py")})

    def test_grep_returns_content_matches(self):
        tool = GrepTool(workspace_root=self.root)
        result = tool.run(
            {
                "pattern": "TODO",
                "path": self.root,
                "output_mode": "content",
                "head_limit": 10,
            }
        )

        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(result.structured_data["match_count"], 2)
        self.assertIn("TODO", result.to_display_string())
        self.assertIn("main.py", result.to_display_string())

    def test_grep_python_fallback_count_mode(self):
        tool = GrepTool(workspace_root=self.root)
        tool.rg_binary = None
        result = tool.run(
            {
                "pattern": "TODO",
                "path": self.root,
                "output_mode": "count",
                "glob": "**/*.py",
            }
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.metadata["engine"], "python")
        self.assertEqual(result.structured_data["match_count"], 2)
        self.assertEqual(result.structured_data["result_count"], 1)
        self.assertIn("main.py:2", result.to_display_string())

    def test_register_filesystem_tools(self):
        registry = ToolRegistry()
        tools = register_filesystem_tools(registry, workspace_root=self.root)

        self.assertEqual(len(tools), 3)
        self.assertIn("FileRead", registry.tools)
        self.assertIn("Glob", registry.tools)
        self.assertIn("Grep", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
