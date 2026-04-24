import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    AskUserQuestionTool,
    ConfigTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
    NotebookEditTool,
    register_ask_user_question_tool,
    register_config_tool,
    register_enter_plan_mode_tool,
    register_exit_plan_mode_tool,
    register_notebook_edit_tool,
)
from Tool.builtin.filesystem import FileReadTool
from core.Config import Config


def _write_notebook(path: str) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": ["# Title\n", "intro text\n"],
            },
            {
                "cell_type": "code",
                "id": "code-1",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": ["print('hello')\n"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
        handle.write("\n")


class TestNotebookEditTool(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = self.tempdir.name
        self.notebook_path = os.path.join(self.root, "demo.ipynb")
        _write_notebook(self.notebook_path)
        self.reader = FileReadTool(workspace_root=self.root)
        self.tool = NotebookEditTool(workspace_root=self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def _load_notebook(self):
        with open(self.notebook_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_notebook_edit_requires_recent_read(self):
        result = self.tool.run(
            {
                "notebook_path": self.notebook_path,
                "cell_id": "intro",
                "new_source": "# Updated\n",
                "edit_mode": "replace",
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "read_required")

    def test_notebook_edit_replace_and_insert_and_delete(self):
        self.reader.run({"file_path": self.notebook_path})

        replace_result = self.tool.run(
            {
                "notebook_path": self.notebook_path,
                "cell_id": "intro",
                "new_source": "# Updated\nmore text\n",
                "edit_mode": "replace",
            }
        )
        self.assertEqual(replace_result.status, "success")

        insert_result = self.tool.run(
            {
                "notebook_path": self.notebook_path,
                "cell_id": "intro",
                "new_source": "print('after intro')\n",
                "cell_type": "code",
                "edit_mode": "insert",
            }
        )
        self.assertEqual(insert_result.status, "success")
        inserted_cell_id = insert_result.structured_data["affectedCellId"]

        delete_result = self.tool.run(
            {
                "notebook_path": self.notebook_path,
                "cell_id": inserted_cell_id,
                "new_source": "",
                "edit_mode": "delete",
            }
        )
        self.assertEqual(delete_result.status, "success")

        notebook = self._load_notebook()
        self.assertEqual(notebook["cells"][0]["id"], "intro")
        self.assertEqual("".join(notebook["cells"][0]["source"]), "# Updated\nmore text\n")
        self.assertEqual(len(notebook["cells"]), 2)

    def test_notebook_edit_normalizes_wrapped_path(self):
        self.reader.run({"file_path": self.notebook_path})
        result = self.tool.run(
            {
                "notebook_path": f"`{self.notebook_path}`",
                "cell_id": "intro",
                "new_source": "# Wrapped\n",
                "edit_mode": "replace",
            }
        )

        self.assertEqual(result.status, "success")


class TestInteractionTools(unittest.TestCase):
    def test_ask_user_question_interrupts(self):
        tool = AskUserQuestionTool()
        result = tool.run(
            {
                "questions": [
                    {
                        "question": "选哪个方案？",
                        "header": "方案",
                        "options": [
                            {"label": "A", "description": "方案 A"},
                            {"label": "B", "description": "方案 B"},
                        ],
                    }
                ],
                "source": "planner",
            }
        )

        self.assertEqual(result.status, "needs_confirmation")
        self.assertEqual(result.error_type, "ask_user_question")
        self.assertEqual(result.structured_data["source"], "planner")

    def test_exit_plan_mode_interrupts(self):
        tool = ExitPlanModeTool()
        result = tool.run(
            {
                "allowedPrompts": [
                    {"tool": "Bash", "prompt": "允许执行测试命令"},
                ]
            }
        )

        self.assertEqual(result.status, "needs_confirmation")
        self.assertEqual(result.error_type, "exit_plan_mode_requested")
        self.assertEqual(result.structured_data["allowedPrompts"][0]["tool"], "Bash")

    def test_enter_plan_mode_interrupts(self):
        tool = EnterPlanModeTool()
        result = tool.run(
            {
                "reason": "先做方案分析",
                "allowedActions": ["read", "search"],
            }
        )

        self.assertEqual(result.status, "needs_confirmation")
        self.assertEqual(result.error_type, "enter_plan_mode_requested")
        self.assertEqual(result.structured_data["allowedActions"], ["read", "search"])
        self.assertEqual(result.structured_data["reason"], "先做方案分析")


class TestConfigTool(unittest.TestCase):
    def test_config_tool_reads_and_updates_live_config(self):
        config = Config(workspace_root="/tmp/demo", command_timeout_ms=5000)
        tool = ConfigTool(config=config)

        read_result = tool.run({"setting": "command_timeout_ms"})
        self.assertEqual(read_result.status, "success")
        self.assertEqual(read_result.structured_data["value"], 5000)

        write_result = tool.run({"setting": "command_timeout_ms", "value": 9000})
        self.assertEqual(write_result.status, "success")
        self.assertEqual(write_result.structured_data["newValue"], 9000)
        self.assertEqual(config.command_timeout_ms, 9000)

    def test_config_tool_splits_allowed_roots(self):
        config = Config()
        tool = ConfigTool(config=config)
        result = tool.run({"setting": "allowed_roots", "value": "/tmp/a:/tmp/b"})

        self.assertEqual(result.status, "success")
        self.assertEqual(config.allowed_roots, ["/tmp/a", "/tmp/b"])


class TestRegistration(unittest.TestCase):
    def test_register_low_frequency_tools(self):
        registry = ToolRegistry()
        notebook_tool = register_notebook_edit_tool(registry, workspace_root=os.getcwd())
        ask_tool = register_ask_user_question_tool(registry)
        enter_tool = register_enter_plan_mode_tool(registry)
        exit_tool = register_exit_plan_mode_tool(registry)
        config_tool = register_config_tool(registry, config=Config())

        self.assertIsInstance(notebook_tool, NotebookEditTool)
        self.assertIsInstance(ask_tool, AskUserQuestionTool)
        self.assertIsInstance(enter_tool, EnterPlanModeTool)
        self.assertIsInstance(exit_tool, ExitPlanModeTool)
        self.assertIsInstance(config_tool, ConfigTool)
        self.assertIn("NotebookEdit", registry.tools)
        self.assertIn("AskUserQuestion", registry.tools)
        self.assertIn("EnterPlanMode", registry.tools)
        self.assertIn("ExitPlanMode", registry.tools)
        self.assertIn("Config", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
