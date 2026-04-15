import os
import sys
import tempfile
import time
import unittest
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    AgentTool,
    BashTool,
    EnterWorktreeTool,
    ExitWorktreeTool,
    FileReadTool,
    FileWriteTool,
    register_file_read_tool,
    register_file_write_tool,
    register_shell_tools,
)
from Tool.builtin.agent_tool import clone_tool_registry_for_workspace
from Tool.runtime import WorktreeManager


class FakeSubagent:
    def __init__(self, response_prefix: str = "handled", *, delay_s: float = 0.0):
        self.response_prefix = response_prefix
        self.delay_s = delay_s
        self.trace_history = [{"type": "tool_call", "tool_name": "WebSearch"}]

    def invoke(self, prompt: str) -> str:
        if self.delay_s:
            time.sleep(self.delay_s)
        return f"{self.response_prefix}:{prompt}"

    def get_context_usage(self) -> dict:
        return {"used_tokens": 42}

    def get_trace_history(self):
        return list(self.trace_history)


def _init_git_repo(root: str) -> str:
    repo = os.path.join(root, "repo")
    os.makedirs(repo, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
    with open(os.path.join(repo, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


class TestAgentTool(unittest.TestCase):
    def test_agent_tool_runs_subagent_synchronously(self):
        with tempfile.TemporaryDirectory() as tempdir:
            captured_requests = []

            def factory(request):
                captured_requests.append(request)
                return FakeSubagent()

            tool = AgentTool(
                agent_factory=factory,
                workspace_root=tempdir,
                storage_dir=os.path.join(tempdir, ".easyagent-agents"),
            )
            result = tool.run(
                {
                    "description": "编写测试",
                    "prompt": "补充单元测试",
                }
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(len(captured_requests), 1)
            self.assertEqual(captured_requests[0].workspace_root, os.path.abspath(tempdir))
            self.assertEqual(result.structured_data["status"], "completed")
            self.assertIn("handled:补充单元测试", result.structured_data["content"][0]["text"])
            self.assertEqual(result.structured_data["totalToolUseCount"], 1)
            self.assertEqual(result.structured_data["totalTokens"], 42)
            self.assertTrue(os.path.exists(result.structured_data["outputFile"]))
            tool.subagent_manager.close()

    def test_agent_tool_supports_background_launch(self):
        with tempfile.TemporaryDirectory() as tempdir:
            def factory(request):
                return FakeSubagent(delay_s=0.1)

            tool = AgentTool(
                agent_factory=factory,
                workspace_root=tempdir,
                storage_dir=os.path.join(tempdir, ".easyagent-agents"),
            )
            result = tool.run(
                {
                    "description": "后台任务",
                    "prompt": "在后台处理",
                    "run_in_background": True,
                }
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(result.structured_data["status"], "async_launched")
            agent_id = result.structured_data["agentId"]
            time.sleep(0.2)
            snapshot = tool.subagent_manager.get_snapshot(agent_id)
            self.assertEqual(snapshot.status, "completed")
            self.assertTrue(os.path.exists(snapshot.output_file))
            tool.subagent_manager.close()

    def test_agent_tool_uses_worktree_isolation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo = _init_git_repo(tempdir)
            storage = os.path.join(tempdir, "worktrees")
            manager = WorktreeManager(repo, storage_dir=storage, original_cwd=repo)
            captured_requests = []

            def factory(request):
                captured_requests.append(request)
                return FakeSubagent()

            tool = AgentTool(
                agent_factory=factory,
                workspace_root=repo,
                worktree_manager=manager,
                storage_dir=os.path.join(repo, ".easyagent-agents"),
            )
            result = tool.run(
                {
                    "description": "隔离修改",
                    "prompt": "在隔离环境里修改代码",
                    "isolation": "worktree",
                    "name": "feature-isolated",
                }
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(len(captured_requests), 1)
            request = captured_requests[0]
            self.assertNotEqual(request.workspace_root, repo)
            self.assertTrue(request.workspace_root.startswith(storage))
            self.assertTrue(os.path.isdir(request.workspace_root))
            self.assertEqual(request.allowed_roots, (request.workspace_root,))
            self.assertEqual(result.structured_data["worktreePath"], request.workspace_root)
            tool.subagent_manager.close()


class TestRegistryClone(unittest.TestCase):
    def test_clone_tool_registry_for_workspace_rebinds_workspace_tools(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source_root = os.path.join(tempdir, "source")
            target_root = os.path.join(tempdir, "target")
            os.makedirs(source_root, exist_ok=True)
            os.makedirs(target_root, exist_ok=True)

            registry = ToolRegistry()
            register_file_read_tool(registry, workspace_root=source_root)
            register_file_write_tool(registry, workspace_root=source_root)
            register_shell_tools(registry, workspace_root=source_root)

            cloned = clone_tool_registry_for_workspace(registry, workspace_root=target_root)

            self.assertIsInstance(cloned.get_tool("FileRead"), FileReadTool)
            self.assertIsInstance(cloned.get_tool("FileWrite"), FileWriteTool)
            self.assertIsInstance(cloned.get_tool("Bash"), BashTool)
            self.assertEqual(cloned.get_tool("FileRead").workspace_root, os.path.abspath(target_root))
            self.assertEqual(cloned.get_tool("FileWrite").workspace_root, os.path.abspath(target_root))
            self.assertEqual(cloned.get_tool("Bash").cwd, os.path.abspath(target_root))


class TestWorktreeTools(unittest.TestCase):
    def test_enter_and_exit_worktree_keep(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo = _init_git_repo(tempdir)
            storage = os.path.join(tempdir, "worktrees")
            manager = WorktreeManager(repo, storage_dir=storage, original_cwd=repo)
            enter_tool = EnterWorktreeTool(worktree_manager=manager)
            exit_tool = ExitWorktreeTool(worktree_manager=manager)

            enter_result = enter_tool.run({"name": "feature-keep"})
            worktree_path = enter_result.structured_data["worktreePath"]
            self.assertEqual(enter_result.status, "success")
            self.assertTrue(os.path.isdir(worktree_path))
            self.assertIsNotNone(manager.get_active_worktree())

            exit_result = exit_tool.run({"action": "keep", "discard_changes": False})
            self.assertEqual(exit_result.status, "success")
            self.assertEqual(exit_result.structured_data["action"], "keep")
            self.assertTrue(os.path.isdir(worktree_path))
            self.assertIsNone(manager.get_active_worktree())

    def test_exit_worktree_remove_discards_changes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo = _init_git_repo(tempdir)
            storage = os.path.join(tempdir, "worktrees")
            manager = WorktreeManager(repo, storage_dir=storage, original_cwd=repo)
            enter_tool = EnterWorktreeTool(worktree_manager=manager)
            exit_tool = ExitWorktreeTool(worktree_manager=manager)

            enter_result = enter_tool.run({"name": "feature-remove"})
            worktree_path = enter_result.structured_data["worktreePath"]
            with open(os.path.join(worktree_path, "temp.txt"), "w", encoding="utf-8") as handle:
                handle.write("dirty\n")

            exit_result = exit_tool.run({"action": "remove", "discard_changes": True})
            self.assertEqual(exit_result.status, "success")
            self.assertEqual(exit_result.structured_data["action"], "remove")
            self.assertGreaterEqual(exit_result.structured_data["discardedFiles"], 1)
            self.assertFalse(os.path.exists(worktree_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
