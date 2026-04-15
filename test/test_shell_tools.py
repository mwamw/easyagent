import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_shell_tools
from Tool.builtin.bash_tool import BashTool
from Tool.builtin.task_output import TaskOutputTool
from Tool.builtin.task_stop import TaskStopTool
from Tool.runtime import ProcessManager


class ShellToolsTestCase(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = self.workspace.name
        self.shell = os.getenv("SHELL") or "bash"
        self.manager = ProcessManager(shell=self.shell, max_background_tasks=4)
        self.addCleanup(self.manager.close)

        self.bash = BashTool(
            workspace_root=self.root,
            shell=self.shell,
            process_manager=self.manager,
            command_timeout_ms=1000,
        )
        self.task_output = TaskOutputTool(
            workspace_root=self.root,
            shell=self.shell,
            process_manager=self.manager,
        )
        self.task_stop = TaskStopTool(
            workspace_root=self.root,
            shell=self.shell,
            process_manager=self.manager,
        )

    def test_bash_foreground_success(self):
        result = self.bash.run({"command": "printf 'hello\\n'"})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["return_code"], 0)
        self.assertIn("hello", result.structured_data["stdout"])

    def test_bash_foreground_non_zero_exit_is_reported(self):
        result = self.bash.run({"command": "printf 'oops\\n' 1>&2; exit 7"})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.structured_data["return_code"], 7)
        self.assertIn("oops", result.structured_data["stderr"])

    def test_bash_background_and_task_output(self):
        start = self.bash.run(
            {
                "command": "printf 'start\\n'; sleep 0.2; printf 'done\\n'",
                "run_in_background": True,
            }
        )

        self.assertEqual(start.status, "success")
        task_id = start.structured_data["task_id"]
        output = self.task_output.run({"task_id": task_id, "block": True, "timeout": 1000})

        self.assertEqual(output.status, "success")
        self.assertEqual(output.structured_data["status"], "completed")
        self.assertIn("start", output.structured_data["stdout"])
        self.assertIn("done", output.structured_data["stdout"])

    def test_task_stop_terminates_running_task(self):
        start = self.bash.run(
            {
                "command": "sleep 5",
                "run_in_background": True,
            }
        )

        self.assertEqual(start.status, "success")
        task_id = start.structured_data["task_id"]
        stopped = self.task_stop.run({"task_id": task_id})

        self.assertEqual(stopped.status, "success")
        self.assertEqual(stopped.structured_data["status"], "terminated")

    def test_task_output_rejects_unknown_task(self):
        result = self.task_output.run({"task_id": "task_missing"})

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "task_not_found")

    def test_bash_rejects_unsupported_sandbox_flag(self):
        result = self.bash.run(
            {
                "command": "printf 'hello\\n'",
                "dangerouslyDisableSandbox": True,
            }
        )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "unsupported_option")

    def test_register_shell_tools(self):
        registry = ToolRegistry()
        tools = register_shell_tools(
            registry,
            workspace_root=self.root,
            shell=self.shell,
            process_manager=self.manager,
        )

        self.assertEqual(len(tools), 3)
        self.assertIn("Bash", registry.tools)
        self.assertIn("TaskOutput", registry.tools)
        self.assertIn("TaskStop", registry.tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
