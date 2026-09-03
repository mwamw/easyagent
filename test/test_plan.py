from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.BasicAgent import BasicAgent
from core.Exception import ExecutionModeError
from core.history import coerce_canonical_message
from core.llm import EasyLLM
from core.permissions import PermissionMode
from plan import ExecutionMode, PlanModeConfig, PlanModeManager


class StubLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.client = None
        self._provider = SimpleNamespace()


def _plan_messages(agent: BasicAgent):
    return [
        coerce_canonical_message(message)
        for message in agent.history
        if coerce_canonical_message(message) is not None
        and coerce_canonical_message(message).metadata.get("source") == "plan"
    ]


def test_plan_module_can_be_imported_before_tool_package():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = project_root
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from plan import EnterPlanModeTool, ExitPlanModeTool, PlanModeManager",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_with_plan_installs_bound_module_and_tools_without_entering():
    agent = BasicAgent(name="planner", llm=StubLLM()).with_plan()

    assert isinstance(agent.plan, PlanModeManager)
    assert agent.get_execution_mode() == ExecutionMode.EXECUTE
    assert agent.permission_context.mode == PermissionMode.DEFAULT
    assert agent.tool_registry is not None
    assert agent.tool_registry.get_tool("EnterPlanMode").plan_manager is agent.plan
    assert agent.tool_registry.get_tool("ExitPlanMode").plan_manager is agent.plan


def test_plan_enter_and_exit_use_configured_permanent_metamessages():
    agent = BasicAgent(name="planner", llm=StubLLM()).with_plan(
        config=PlanModeConfig(
            enter_message="Only inspect and plan.",
            exit_message="Execute the accepted plan.",
            allowed_actions=["FileRead", "Grep"],
            register_tools=False,
        )
    )

    agent.plan.enter()
    assert agent.plan.is_active
    assert agent.permission_context.mode == PermissionMode.PLAN
    assert agent.execution_context.execution_mode == "plan"
    agent.metamessage_manager.flush()

    agent.plan.exit(permission_mode=PermissionMode.ACCEPT_EDITS)
    assert not agent.plan.is_active
    assert agent.permission_context.mode == PermissionMode.ACCEPT_EDITS
    assert agent.execution_context.execution_mode == "execute"
    agent.metamessage_manager.flush()

    messages = _plan_messages(agent)
    assert [message.metadata["mode"] for message in messages] == ["plan", "execute"]
    assert "Only inspect and plan." in messages[0].text_content()
    assert "FileRead, Grep" in messages[0].text_content()
    assert "Execute the accepted plan." in messages[1].text_content()


def test_plan_must_be_installed_and_cannot_be_installed_twice():
    agent = BasicAgent(name="plain", llm=StubLLM())

    with pytest.raises(ExecutionModeError, match="not installed"):
        agent.enter_plan_mode()

    agent.with_plan(config=PlanModeConfig(register_tools=False))
    with pytest.raises(ExecutionModeError, match="already installed"):
        agent.with_plan()


def test_exit_plan_tool_records_request_in_plan_module():
    agent = BasicAgent(name="planner", llm=StubLLM()).with_plan()
    agent.plan.enter()

    result = agent.tool_registry.execute_tool_result(
        "ExitPlanMode",
        {
            "allowedPrompts": [
                {"tool": "Bash", "prompt": "允许执行测试"},
            ]
        },
    )

    assert result.status == "needs_confirmation"
    assert agent.plan.state.exit_requested is True
    assert agent.plan.state.allowed_actions == ["Bash"]
