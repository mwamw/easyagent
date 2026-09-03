"""Public SDK contract tests for Phase G."""

from __future__ import annotations

import os
import sys
import tomllib
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import easyagent
from agent import BasicAgent as InternalBasicAgent
from core.llm import EasyLLM as InternalEasyLLM
from observability import InMemoryObservabilityStore as InternalInMemoryObservabilityStore
from training import TrainingExporter as InternalTrainingExporter
from core.permissions import PermissionContext as InternalPermissionContext
from codeintel import WorkspaceCodeIntelCache as InternalWorkspaceCodeIntelCache
from task import TaskService as InternalTaskService
from Tool.ToolRegistry import ToolRegistry as InternalToolRegistry


class DummyLLM(InternalEasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256

    def invoke(self, messages, temperature=None, **kwargs):
        return "mock-response"

    def prepare_messages_for_request(self, messages):
        return list(messages)


class TestSDKPublicAPI(unittest.TestCase):
    def test_top_level_sdk_exports_map_to_internal_objects(self):
        self.assertIs(easyagent.BasicAgent, InternalBasicAgent)
        self.assertIs(easyagent.EasyLLM, InternalEasyLLM)
        self.assertIs(easyagent.ToolRegistry, InternalToolRegistry)
        self.assertIs(easyagent.PermissionContext, InternalPermissionContext)
        self.assertIs(easyagent.TaskService, InternalTaskService)
        self.assertIs(easyagent.WorkspaceCodeIntelCache, InternalWorkspaceCodeIntelCache)
        self.assertIs(easyagent.InMemoryObservabilityStore, InternalInMemoryObservabilityStore)
        self.assertIs(easyagent.TrainingExporter, InternalTrainingExporter)
        self.assertTrue(hasattr(easyagent, "__version__"))

    def test_submodule_imports_are_available(self):
        from easyagent.agents import BasicAgent
        from easyagent.llms import EasyLLM
        from easyagent.mcp import MCPPolicyContext, register_mcp_tools
        from easyagent.observability import InMemoryObservabilityStore
        from easyagent.permissions import PermissionBehavior, PermissionRule
        from easyagent.session import SessionRestoreReport
        from easyagent.tools import ToolRegistry
        from easyagent.training import TrainingExporter

        self.assertIs(BasicAgent, InternalBasicAgent)
        self.assertIs(EasyLLM, InternalEasyLLM)
        self.assertIs(InMemoryObservabilityStore, InternalInMemoryObservabilityStore)
        self.assertIs(TrainingExporter, InternalTrainingExporter)
        self.assertTrue(callable(register_mcp_tools))
        self.assertEqual(PermissionBehavior.ALLOW.value, "allow")
        self.assertTrue(hasattr(PermissionRule, "model_fields"))
        self.assertTrue(hasattr(SessionRestoreReport, "to_dict"))
        self.assertIs(ToolRegistry, InternalToolRegistry)
        self.assertTrue(hasattr(MCPPolicyContext, "authorize"))

    def test_sdk_agent_can_be_built_without_optional_subsystems(self):
        agent = easyagent.BasicAgent(
            name="sdk-agent",
            llm=DummyLLM(),
        )
        self.assertEqual(agent.name, "sdk-agent")
        self.assertIsInstance(agent.get_last_close_report(), type(None))
        self.assertEqual(agent.get_history(), [])

    def test_pyproject_defines_sdk_package_and_extras(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        with open(pyproject_path, "rb") as handle:
            data = tomllib.load(handle)

        project = data["project"]
        self.assertEqual(project["name"], "easyagent")
        self.assertIn("mcp", project["optional-dependencies"])
        self.assertIn("rag", project["optional-dependencies"])
        self.assertIn("memory", project["optional-dependencies"])
        self.assertIn("dev", project["optional-dependencies"])

        package_data = data["tool"]["setuptools"]["package-data"]
        self.assertIn("easyagent", package_data)
        self.assertIn("py.typed", package_data["easyagent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
