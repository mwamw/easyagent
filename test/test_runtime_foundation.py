import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.Config import Config
from Tool.runtime import FilesystemAccessError, FilesystemGuard, PathResolutionError, PathResolver, ProcessManager, WorktreeManager


class RuntimeFoundationTestCase(unittest.TestCase):
    def test_config_from_env_reads_runtime_fields(self):
        with patch.dict(
            os.environ,
            {
                "WORKSPACE_ROOT": "/tmp/workspace",
                "ALLOWED_ROOTS": f"/tmp/workspace{os.pathsep}/tmp/shared",
                "COMMAND_TIMEOUT_MS": "1234",
                "MAX_BACKGROUND_TASKS": "5",
                "GIT_BINARY": "git-custom",
                "ENABLE_WORKTREE": "true",
                "INTERRUPT_ON_CONFIRMATION": "false",
            },
            clear=False,
        ):
            config = Config.from_env()

        self.assertEqual(config.workspace_root, "/tmp/workspace")
        self.assertEqual(config.allowed_roots, ["/tmp/workspace", "/tmp/shared"])
        self.assertEqual(config.command_timeout_ms, 1234)
        self.assertEqual(config.max_background_tasks, 5)
        self.assertEqual(config.git_binary, "git-custom")
        self.assertTrue(config.enable_worktree)
        self.assertFalse(config.interrupt_on_confirmation)

    def test_path_resolver_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.makedirs(os.path.join(tempdir, "sub"), exist_ok=True)
            resolver = PathResolver(tempdir)
            inside = resolver.resolve("sub/file.txt", must_exist=False)
            self.assertTrue(inside.startswith(tempdir))

            with self.assertRaises(PathResolutionError):
                resolver.resolve("../outside.txt", must_exist=False)

    def test_filesystem_guard_write_requires_parent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            guard = FilesystemGuard(tempdir)
            target = guard.resolve_write_path("nested/file.txt", create_parents=True)
            self.assertTrue(target.startswith(tempdir))
            self.assertTrue(os.path.isdir(os.path.join(tempdir, "nested")))

            with self.assertRaises(FilesystemAccessError):
                guard.ensure_parent_writable(os.path.join(tempdir, "missing", "file.txt"))

    def test_process_manager_foreground_and_background(self):
        manager = ProcessManager(shell=os.getenv("SHELL", "bash"), max_background_tasks=2)
        result = manager.run([sys.executable, "-c", "print('hello')"], use_shell=False)
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout.strip(), "hello")

        snapshot = manager.start_background(
            [
                sys.executable,
                "-c",
                "import sys,time;print('start');sys.stdout.flush();time.sleep(0.2);print('done')",
            ],
            use_shell=False,
        )
        time.sleep(0.4)
        output = manager.get_output(snapshot.task_id)
        self.assertIn("start", output.stdout)
        self.assertIn("done", output.stdout)
        manager.close()

    def test_worktree_manager_round_trip(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo = os.path.join(tempdir, "repo")
            storage = os.path.join(tempdir, "worktrees")
            os.makedirs(repo, exist_ok=True)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
            with open(os.path.join(repo, "README.md"), "w", encoding="utf-8") as handle:
                handle.write("hello\n")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

            repo_root = WorktreeManager.detect_repo_root(repo)
            manager = WorktreeManager(repo_root, storage_dir=storage)
            info = manager.create_worktree("feature-one")

            self.assertTrue(os.path.isdir(info.path))
            self.assertTrue(any(item.path == info.path for item in manager.list_worktrees()))

            manager.remove_worktree(info.path, force=True)
            manager.prune()
            self.assertFalse(os.path.exists(info.path))
