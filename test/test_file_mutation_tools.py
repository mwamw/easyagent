import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.filesystem import FileReadTool
from Tool.builtin.file_edit import FileEditTool
from Tool.builtin.file_write import FileWriteTool
from Tool.builtin import register_file_edit_tool, register_file_write_tool
from Tool.runtime import clear_file_read_timestamps


class FileMutationToolsTestCase(unittest.TestCase):
    def setUp(self):
        clear_file_read_timestamps()
        self.addCleanup(clear_file_read_timestamps)
        self.workspace = tempfile.TemporaryDirectory()
        self.outside = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.addCleanup(self.outside.cleanup)
        self.root = self.workspace.name
        self.reader = FileReadTool(workspace_root=self.root)

        os.makedirs(os.path.join(self.root, "src"), exist_ok=True)
        with open(os.path.join(self.root, "src", "target.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "def main():\n"
                "    print('hello')\n"
                "    print('hello')\n"
            )

    def _read(self, path: str) -> None:
        result = self.reader.run({"file_path": path})
        self.assertEqual(result.status, "success")

    def test_file_write_creates_new_file(self):
        tool = FileWriteTool(workspace_root=self.root)
        target = os.path.join(self.root, "src", "new_file.py")
        result = tool.run({"file_path": target, "content": "print('created')\n"})

        self.assertEqual(result.status, "success")
        self.assertTrue(os.path.exists(target))
        self.assertTrue(result.metadata["created"])
        self.assertIn("diff --git a/src/new_file.py b/src/new_file.py", result.structured_data["diff"]["unified"])
        self.assertIn("+++ b/src/new_file.py", result.structured_data["diff"]["unified"])
        self.assertIn("+print('created')", result.structured_data["diff"]["unified"])
        with open(target, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "print('created')\n")

    def test_file_write_rejects_outside_workspace(self):
        tool = FileWriteTool(workspace_root=self.root)
        outside_path = os.path.join(self.outside.name, "forbidden.py")
        result = tool.run({"file_path": outside_path, "content": "print('nope')\n"})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "invalid_path")
        self.assertEqual(result.structured_data["reason"], "invalid_path")

    def test_file_write_overwrite_requires_prior_read(self):
        tool = FileWriteTool(workspace_root=self.root)
        target = os.path.join(self.root, "src", "target.py")
        result = tool.run({"file_path": target, "content": "print('overwrite')\n"})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "read_required")
        self.assertEqual(result.structured_data["reason"], "file_not_read")

    def test_file_write_overwrite_after_read_succeeds(self):
        tool = FileWriteTool(workspace_root=self.root)
        target = os.path.join(self.root, "src", "target.py")
        self._read(target)
        result = tool.run({"file_path": target, "content": "print('overwrite')\n"})

        self.assertEqual(result.status, "success")
        self.assertIn("--- a/src/target.py", result.structured_data["diff"]["unified"])
        self.assertIn("+++ b/src/target.py", result.structured_data["diff"]["unified"])
        self.assertIn("+print('overwrite')", result.structured_data["diff"]["unified"])
        with open(target, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "print('overwrite')\n")

    def test_file_edit_replaces_unique_match(self):
        target = os.path.join(self.root, "src", "unique.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("value = 'hello'\n")

        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "'hello'",
                "new_string": "'world'",
            }
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["replacements"], 1)
        self.assertIn("-value = 'hello'", result.structured_data["diff"]["unified"])
        self.assertIn("+value = 'world'", result.structured_data["diff"]["unified"])
        with open(target, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "value = 'world'\n")

    def test_file_edit_requires_prior_read(self):
        target = os.path.join(self.root, "src", "target.py")
        tool = FileEditTool(workspace_root=self.root)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "print('hello')",
                "new_string": "print('world')",
                "replace_all": True,
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "read_required")
        self.assertEqual(result.structured_data["reason"], "file_not_read")

    def test_file_edit_reports_non_unique_match(self):
        target = os.path.join(self.root, "src", "target.py")
        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "print('hello')",
                "new_string": "print('world')",
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "non_unique_match")
        self.assertEqual(result.structured_data["reason"], "multiple_matches")
        self.assertEqual(result.structured_data["match_count"], 2)

    def test_file_edit_replace_all_updates_all_matches(self):
        target = os.path.join(self.root, "src", "target.py")
        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "print('hello')",
                "new_string": "print('world')",
                "replace_all": True,
            }
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["replacements"], 2)
        with open(target, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertEqual(content.count("print('world')"), 2)

    def test_file_edit_reports_no_match(self):
        target = os.path.join(self.root, "src", "target.py")
        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "print('missing')",
                "new_string": "print('world')",
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "no_match")
        self.assertEqual(result.structured_data["reason"], "no_match")

    def test_file_edit_detects_stale_read(self):
        target = os.path.join(self.root, "src", "target.py")
        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("# changed outside\n")

        result = tool.run(
            {
                "file_path": target,
                "old_string": "print('hello')",
                "new_string": "print('world')",
                "replace_all": True,
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "stale_file")
        self.assertEqual(result.structured_data["reason"], "stale_read")

    def test_file_edit_normalizes_line_endings(self):
        target = os.path.join(self.root, "src", "multiline.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("alpha = 1\nbeta = 2\n")

        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "alpha = 1\r\nbeta = 2",
                "new_string": "alpha = 10\r\nbeta = 20",
            }
        )

        self.assertEqual(result.status, "success")
        self.assertIn(result.structured_data["match_mode"], {"exact", "normalized"})
        with open(target, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "alpha = 10\nbeta = 20\n")

    def test_file_edit_rejects_large_file(self):
        from Tool.builtin.file_edit import MAX_EDIT_FILE_SIZE

        target = os.path.join(self.root, "src", "large.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("a" * (MAX_EDIT_FILE_SIZE + 1))

        tool = FileEditTool(workspace_root=self.root)
        self._read(target)
        result = tool.run(
            {
                "file_path": target,
                "old_string": "aaaa",
                "new_string": "bbbb",
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "file_too_large")
        self.assertEqual(result.structured_data["reason"], "file_too_large")

    def test_register_mutation_tools(self):
        registry = ToolRegistry()
        write_tool = register_file_write_tool(registry, workspace_root=self.root)
        edit_tool = register_file_edit_tool(registry, workspace_root=self.root)

        self.assertIn("FileWrite", registry.tools)
        self.assertIn("FileEdit", registry.tools)
        self.assertIsInstance(write_tool, FileWriteTool)
        self.assertIsInstance(edit_tool, FileEditTool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
