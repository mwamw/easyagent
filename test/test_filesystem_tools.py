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
    ListTool,
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

        with open(os.path.join(self.root, ".env.example"), "w", encoding="utf-8") as handle:
            handle.write("DEBUG=false\n")

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

    def test_file_read_normalizes_line_suffix_path(self):
        tool = FileReadTool(workspace_root=self.root)
        result = tool.run({"file_path": "notes.txt:2", "limit": 1})

        self.assertEqual(result.status, "success")
        self.assertIn("2 | beta", result.to_display_string())
        self.assertEqual(result.structured_data["start_line"], 2)

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

    def test_glob_normalizes_wrapped_pattern_and_uses_cwd_as_default_root(self):
        tool = GlobTool(workspace_root=self.root, cwd=os.path.join(self.root, "src"))
        result = tool.run({"pattern": "'**/*.py'"})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["root"], os.path.join(self.root, "src"))
        relative_paths = {item["relative_path"] for item in result.structured_data["matches"]}
        self.assertEqual(relative_paths, {"main.py", os.path.join("utils", "helper.py")})

    def test_glob_no_match_returns_diagnostics(self):
        tool = GlobTool(workspace_root=self.root)
        result = tool.run({"pattern": ".*\\.py$"})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["matches"], [])
        self.assertIn("regex", result.to_display_string().lower())

    def test_list_shows_directory_structure_and_hidden_entries(self):
        tool = ListTool(workspace_root=self.root)
        result = tool.run({"path": self.root})

        self.assertEqual(result.status, "success")
        names = {item["name"] for item in result.structured_data["entries"]}
        self.assertIn("src", names)
        self.assertIn("docs", names)
        self.assertIn(".env.example", names)
        self.assertIn("目录:", result.to_display_string())

    def test_list_supports_recursive_directory_only_view(self):
        tool = ListTool(workspace_root=self.root)
        result = tool.run(
            {
                "path": self.root,
                "recursive": True,
                "directories_only": True,
                "max_depth": 2,
            }
        )

        self.assertEqual(result.status, "success")
        entries = result.structured_data["entries"]
        self.assertTrue(all(item["is_dir"] for item in entries))
        relative_paths = {item["relative_path"] for item in entries}
        self.assertIn("src", relative_paths)
        self.assertIn(os.path.join("src", "utils"), relative_paths)

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

    def test_grep_normalizes_wrapped_pattern_and_glob(self):
        tool = GrepTool(workspace_root=self.root)
        result = tool.run(
            {
                "pattern": "'TODO'",
                "path": self.root,
                "output_mode": "content",
                "glob": "'**/*.py'",
            }
        )

        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(result.structured_data["match_count"], 2)
        self.assertEqual(result.structured_data["pattern"], "TODO")

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

        self.assertEqual(len(tools), 4)
        self.assertIn("FileRead", registry.tools)
        self.assertIn("List", registry.tools)
        self.assertIn("Glob", registry.tools)
        self.assertIn("Grep", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
