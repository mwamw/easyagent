import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import TodoWriteTool, register_todo_write_tool
from Tool.runtime import clear_todo_items, get_todo_items


class TestTodoWriteTool(unittest.TestCase):
    def setUp(self):
        clear_todo_items()

    def tearDown(self):
        clear_todo_items()

    def test_todo_write_updates_state_and_returns_old_and_new_todos(self):
        tool = TodoWriteTool()

        first = tool.run(
            {
                "todos": [
                    {
                        "content": "实现 TodoWrite",
                        "status": "in_progress",
                        "activeForm": "正在实现 TodoWrite",
                    },
                    {
                        "content": "补测试",
                        "status": "pending",
                        "activeForm": "正在补测试",
                    },
                ]
            }
        )
        second = tool.run(
            {
                "todos": [
                    {
                        "content": "实现 TodoWrite",
                        "status": "completed",
                        "activeForm": "正在实现 TodoWrite",
                    },
                    {
                        "content": "补测试",
                        "status": "in_progress",
                        "activeForm": "正在补测试",
                    },
                ]
            }
        )

        self.assertEqual(first.status, "success")
        self.assertEqual(first.structured_data["oldTodos"], [])
        self.assertEqual(len(first.structured_data["newTodos"]), 2)

        self.assertEqual(second.status, "success")
        self.assertEqual(len(second.structured_data["oldTodos"]), 2)
        self.assertEqual(second.structured_data["newTodos"][0]["status"], "completed")
        self.assertEqual(get_todo_items()[1].status, "in_progress")

    def test_todo_write_rejects_duplicate_content(self):
        tool = TodoWriteTool()
        result = tool.run(
            {
                "todos": [
                    {
                        "content": "同一个任务",
                        "status": "pending",
                        "activeForm": "正在处理同一个任务",
                    },
                    {
                        "content": "同一个任务",
                        "status": "completed",
                        "activeForm": "正在收尾同一个任务",
                    },
                ]
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "invalid_parameters")

    def test_todo_write_rejects_multiple_in_progress_items(self):
        tool = TodoWriteTool()
        result = tool.run(
            {
                "todos": [
                    {
                        "content": "任务 A",
                        "status": "in_progress",
                        "activeForm": "正在做任务 A",
                    },
                    {
                        "content": "任务 B",
                        "status": "in_progress",
                        "activeForm": "正在做任务 B",
                    },
                ]
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "invalid_parameters")

    def test_todo_write_sets_verification_nudge_when_no_verification_step_exists(self):
        tool = TodoWriteTool()
        result = tool.run(
            {
                "todos": [
                    {
                        "content": "实现工具",
                        "status": "completed",
                        "activeForm": "正在实现工具",
                    },
                    {
                        "content": "整理文档",
                        "status": "pending",
                        "activeForm": "正在整理文档",
                    },
                ]
            }
        )

        self.assertEqual(result.status, "success")
        self.assertTrue(result.structured_data["verificationNudgeNeeded"])
        self.assertIn("验证/测试步骤", result.to_display_string())

    def test_register_todo_write_tool(self):
        registry = ToolRegistry()
        tool = register_todo_write_tool(registry)

        self.assertIsInstance(tool, TodoWriteTool)
        self.assertIn("TodoWrite", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
