import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import BasicAgent
from codeintel import CodeIntelManager, LSPCodeIntelProvider
from core.Config import Config
from core.llm import EasyLLM
from db import SessionStore
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_codeintel_tools


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 128
        self.closed = False

    def invoke(self, messages, temperature=None, **kwargs):
        return "mock-response"

    def prepare_messages_for_request(self, messages):
        return list(messages)

    def close(self):
        self.closed = True


class CodeIntelEnhancementTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = self.tempdir.name
        self.sample_path = os.path.join(self.workspace, "sample.py")
        self.helper_path = os.path.join(self.workspace, "helper.py")
        with open(self.sample_path, "w", encoding="utf-8") as handle:
            handle.write(
                "def demo_function():\n"
                "    return 1\n"
                "demo_function()\n"
            )
        with open(self.helper_path, "w", encoding="utf-8") as handle:
            handle.write(
                "from sample import demo_function\n"
                "demo_function()\n"
            )
        self.fake_server = os.path.join(ROOT, "test", "fake_lsp_server.py")
        self.missing_server = ["/definitely/missing/lsp-server"]
        self.db_path = os.path.join(self.workspace, "sessions.db")

    def tearDown(self):
        self.tempdir.cleanup()

    def _provider(self):
        return LSPCodeIntelProvider(
            server_command=[sys.executable, self.fake_server],
            request_timeout_ms=2000,
            diagnostics_wait_ms=800,
        )

    def test_prewarm_builds_cache_status_and_offline_index(self):
        registry = ToolRegistry()
        manager = register_codeintel_tools(
            registry,
            provider=self._provider(),
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )
        try:
            prewarm = registry.execute_tool_result(
                "CodeIntelPrewarmWorkspace",
                {"max_files": 10, "include_diagnostics": True, "force": False},
            )
            self.assertTrue(prewarm.structured_data["providerAvailable"])
            self.assertGreaterEqual(prewarm.structured_data["indexedFiles"], 1)
            self.assertTrue(prewarm.structured_data["offlineIndexAvailable"])

            cache_status = registry.execute_tool_result("CodeIntelCacheStatus", {})
            self.assertTrue(cache_status.structured_data["offlineIndexAvailable"])
            self.assertGreaterEqual(cache_status.structured_data["indexedFileCount"], 1)
            self.assertIsNotNone(cache_status.structured_data["lastPrewarmSummary"])
        finally:
            manager.close()

    def test_workspace_symbols_and_file_queries_fall_back_to_cache(self):
        manager = CodeIntelManager(
            provider=self._provider(),
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )
        try:
            prewarm = manager.prewarm_workspace(max_files=10, include_diagnostics=True)
            self.assertTrue(prewarm["offlineIndexAvailable"])

            healthy_provider = manager.provider
            manager.find_definition(file_path=self.sample_path, line=1, column=5)
            manager.find_references(file_path=self.sample_path, line=1, column=5)
            manager.provider = LSPCodeIntelProvider(server_command=self.missing_server)
            healthy_provider.close()

            workspace_symbols = manager.get_workspace_symbols(query="demo_function", limit=10)
            self.assertEqual(workspace_symbols.status, "ok")
            self.assertTrue(workspace_symbols.metadata["cacheHit"])
            self.assertEqual(workspace_symbols.metadata["cacheSource"], "offline_index")

            document_symbols = manager.get_document_symbols(file_path=self.sample_path)
            self.assertEqual(document_symbols.status, "ok")
            self.assertTrue(document_symbols.metadata["cacheHit"])
            self.assertEqual(document_symbols.metadata["cacheSource"], "document_symbols")

            diagnostics = manager.get_diagnostics(file_path=self.sample_path)
            self.assertEqual(diagnostics.status, "ok")
            self.assertTrue(diagnostics.metadata["cacheHit"])
            self.assertEqual(diagnostics.metadata["cacheSource"], "diagnostics")

            definition = manager.find_definition(file_path=self.sample_path, line=1, column=5)
            self.assertEqual(definition.status, "ok")
            self.assertTrue(definition.metadata["cacheHit"])
            self.assertEqual(definition.metadata["cacheSource"], "definition")

            references = manager.find_references(file_path=self.sample_path, line=1, column=5)
            self.assertEqual(references.status, "ok")
            self.assertTrue(references.metadata["cacheHit"])
            self.assertEqual(references.metadata["cacheSource"], "references")
        finally:
            manager.close()

    def test_manager_export_restore_preserves_workspace_cache_snapshot(self):
        manager = CodeIntelManager(
            provider=self._provider(),
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )
        try:
            manager.prewarm_workspace(max_files=10, include_diagnostics=True)
            state = manager.export_state()
        finally:
            manager.close()

        restored = CodeIntelManager.from_state(
            state,
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
            provider=LSPCodeIntelProvider(server_command=self.missing_server),
        )
        try:
            result = restored.get_workspace_symbols(query="demo_function", limit=10)
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.metadata["cacheHit"])
            self.assertEqual(result.metadata["cacheSource"], "offline_index")
        finally:
            restored.close()

    def test_session_restore_rebuilds_codeintel_cache_runtime(self):
        session_store = SessionStore(self.db_path)
        registry = ToolRegistry()
        agent = BasicAgent(
            name="codeintel-cache-agent",
            llm=DummyLLM(),
            enable_tool=True,
            tool_registry=registry,
            config=Config(
                workspace_root=self.workspace,
                allowed_roots=[self.workspace],
            ),
        )
        manager = register_codeintel_tools(
            registry,
            provider=self._provider(),
            parent_agent=agent,
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )
        try:
            registry.execute_tool_result(
                "CodeIntelPrewarmWorkspace",
                {"max_files": 10, "include_diagnostics": True, "force": False},
            )
            agent.save_session("codeintel-cache-session", store=session_store)
        finally:
            manager.close()

        restored = BasicAgent.load_session(
            "codeintel-cache-session",
            llm=DummyLLM(),
            store=session_store,
        )
        restored_manager = restored.tool_registry.get_runtime_surface("codeintel_manager", "default")
        self.assertIsNotNone(restored_manager)
        cache_status = restored.tool_registry.execute_tool_result("CodeIntelCacheStatus", {})
        self.assertGreaterEqual(cache_status.structured_data["indexedFileCount"], 1)

        healthy_provider = restored_manager.provider
        restored_manager.provider = LSPCodeIntelProvider(server_command=self.missing_server)
        healthy_provider.close()
        offline = restored.tool_registry.execute_tool_result(
            "GetWorkspaceSymbols",
            {"query": "demo_function", "limit": 10},
        )
        self.assertEqual(offline.structured_data["status"], "ok")
        self.assertEqual(offline.structured_data["metadata"]["cacheSource"], "offline_index")

        report = restored.get_last_restore_report()
        self.assertIn("codeintel_runtime", report["components"])

        restored.close(close_worktree=False)


if __name__ == "__main__":
    unittest.main()
