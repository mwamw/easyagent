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


class CodeIntelTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = self.tempdir.name
        self.sample_path = os.path.join(self.workspace, "sample.py")
        with open(self.sample_path, "w", encoding="utf-8") as handle:
            handle.write(
                "def demo_function():\n"
                "    return 1\n"
                "demo_function()\n"
            )
        self.fake_server = os.path.join(ROOT, "test", "fake_lsp_server.py")

    def tearDown(self):
        self.tempdir.cleanup()

    def _provider(self):
        return LSPCodeIntelProvider(
            server_command=[sys.executable, self.fake_server],
            request_timeout_ms=2000,
            diagnostics_wait_ms=800,
        )

    def test_lsp_codeintel_provider_round_trip_queries(self):
        manager = CodeIntelManager(
            provider=self._provider(),
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )

        status = manager.get_status(file_path=self.sample_path)
        self.assertTrue(status.available)
        self.assertEqual(status.provider_name, "lsp")

        definition = manager.find_definition(file_path=self.sample_path, line=1, column=5)
        self.assertEqual(definition.status, "ok")
        self.assertEqual(definition.items[0].path, os.path.abspath(self.sample_path))

        references = manager.find_references(file_path=self.sample_path, line=1, column=5)
        self.assertEqual(references.status, "ok")
        self.assertGreaterEqual(len(references.items), 2)

        symbols = manager.get_document_symbols(file_path=self.sample_path)
        self.assertEqual(symbols.status, "ok")
        self.assertEqual(symbols.items[0].name, "demo_function")

        workspace_symbols = manager.get_workspace_symbols(query="demo_function", limit=10)
        self.assertEqual(workspace_symbols.status, "ok")
        self.assertEqual(workspace_symbols.items[0].name, "demo_function")

        diagnostics = manager.get_diagnostics(file_path=self.sample_path)
        self.assertEqual(diagnostics.status, "ok")
        self.assertEqual(diagnostics.items[0].source, "fake-lsp")
        self.assertIn("fake diagnostic", diagnostics.items[0].message)

        manager.close()

    def test_codeintel_tools_return_structured_results(self):
        registry = ToolRegistry()
        manager = CodeIntelManager(
            provider=self._provider(),
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )
        register_codeintel_tools(registry, manager=manager)

        status_result = registry.execute_tool_result("CodeIntelStatus", {"file_path": "sample.py"})
        self.assertEqual(status_result.structured_data["providerName"], "lsp")
        self.assertTrue(status_result.structured_data["available"])

        definition_result = registry.execute_tool_result(
            "FindDefinition",
            {"file_path": "sample.py", "line": 1, "column": 5},
        )
        self.assertEqual(definition_result.structured_data["status"], "ok")
        self.assertIn("CodeIntel 返回", definition_result.to_display_string())

        diagnostics_result = registry.execute_tool_result("GetDiagnostics", {"file_path": "sample.py"})
        self.assertEqual(diagnostics_result.structured_data["status"], "ok")
        self.assertEqual(diagnostics_result.structured_data["items"][0]["source"], "fake-lsp")

        manager.close()

    def test_codeintel_tools_return_unavailable_fallback_when_server_missing(self):
        registry = ToolRegistry()
        manager = CodeIntelManager(
            provider=LSPCodeIntelProvider(server_command=["/definitely/missing/lsp-server"]),
            workspace_root=self.workspace,
            allowed_roots=(self.workspace,),
        )
        register_codeintel_tools(registry, manager=manager)

        result = registry.execute_tool_result(
            "FindDefinition",
            {"file_path": "sample.py", "line": 1, "column": 5},
        )

        self.assertEqual(result.structured_data["status"], "unavailable")
        self.assertEqual(result.structured_data["fallbackTools"], ["FileRead", "Grep", "Glob"])
        self.assertIn("FileRead", result.to_display_string())

        manager.close()

    def test_basic_agent_close_reports_codeintel_component(self):
        llm = DummyLLM()
        registry = ToolRegistry()
        agent = BasicAgent(
            name="codeintel-manager",
            llm=llm,
            config=Config(
                workspace_root=self.workspace,
                allowed_roots=[self.workspace],
            ),
        ).with_tool(registry)
        agent.with_codeintel(
            provider=self._provider(),
        )
        registry.execute_tool_result(
            "FindDefinition",
            {"file_path": "sample.py", "line": 1, "column": 5},
        )

        report = agent.close()

        self.assertEqual(report["components"]["codeintel"]["status"], "closed")
        self.assertTrue(llm.closed)


if __name__ == "__main__":
    unittest.main()
