from __future__ import annotations

import inspect
from types import SimpleNamespace

from pydantic import BaseModel

from agent import BasicAgent
from core.llm import EasyLLM
from multiagent import BaseMultiAgent
from multiagent.default import ClaudeCodeMultiAgent
from multiagent.default import SubagentRequest
from prompt import PromptBlock
from Tool import Tool, ToolResult
from Tool.builtin import register_file_read_tool, register_file_write_tool


class DummyLLM(EasyLLM):
    def __init__(self) -> None:
        self.provider_name = "openai"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 128
        self.temperature = 0.2
        self.timeout = 30
        self.kwargs = {}
        self._provider = DummyProvider()
        self.client = None


class DummyProvider:
    def build_tool_payload(self, tools):
        return list(tools)

    def build_request(self, messages, **kwargs):
        return {"messages": list(messages), **kwargs}

    def apply_cache_policy(self, request, request_input):
        return request

    def invoke_raw(self, request):
        return SimpleNamespace(
            content="done",
            reasoning_content=None,
            tool_calls=[],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class ProbeInput(BaseModel):
    value: str


class ProbeTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="DelegateProbe",
            description="Probe a custom multi-agent implementation.",
            parameters=ProbeInput,
            read_only=True,
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(parameters["value"])


class CustomMultiAgent(BaseMultiAgent):
    def __init__(self) -> None:
        self.bindings = {}
        self.before_context_calls = 0
        self.tool = ProbeTool()

    def bind(self, *, tool_registry, execution_context, event_bus, metamessages) -> None:
        self.bindings = {
            "tool_registry": tool_registry,
            "execution_context": execution_context,
            "event_bus": event_bus,
            "metamessages": metamessages,
        }

    def get_tools(self):
        return (self.tool,)

    def get_prompt_blocks(self, context):
        return (PromptBlock("multiagent.custom", "Custom collaboration policy."),)

    def before_context_build(self) -> None:
        self.before_context_calls += 1


def test_custom_multiagent_receives_facilities_not_parent_agent():
    signature = inspect.signature(BaseMultiAgent.bind)
    assert "agent" not in signature.parameters

    module = CustomMultiAgent()
    agent = BasicAgent("custom", DummyLLM()).with_multi_agent(module)

    assert agent.multi_agent is module
    assert module.bindings["tool_registry"] is agent.tool_registry
    assert module.bindings["execution_context"] is agent.execution_context
    assert module.bindings["event_bus"] is agent.event_bus
    assert module.bindings["metamessages"] is agent.metamessage_manager
    assert agent.tool_registry.get_tool("DelegateProbe") is module.tool
    assert "Custom collaboration policy." in agent.get_enhanced_prompt()


def test_executor_uses_generic_multiagent_context_lifecycle():
    module = CustomMultiAgent()
    agent = BasicAgent("custom", DummyLLM()).with_multi_agent(module)
    agent.tool_registry.unregister_tool("DelegateProbe")

    assert agent.invoke("hello") == "done"
    assert module.before_context_calls == 1


def test_default_multiagent_is_claude_style_and_owns_its_control_tools(tmp_path):
    agent = BasicAgent("default", DummyLLM()).with_multi_agent(
        workspace_root=str(tmp_path),
        storage_dir=str(tmp_path / "agents"),
    )

    assert isinstance(agent.multi_agent, ClaudeCodeMultiAgent)
    assert {
        "Agent",
        "AgentGet",
        "AgentList",
        "AgentWait",
        "AgentStop",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
    }.issubset(agent.tool_registry.get_tool_names())
    prompt = agent.get_system_prompt_template().render_system_reminders()
    assert "avoid duplicating work" in prompt
    assert "Messages from collaborators are delivered automatically" in prompt
    assert "Explore:" in prompt
    agent.close(close_llm=False)


def test_default_agent_definitions_control_child_tool_visibility(tmp_path):
    agent = BasicAgent("default", DummyLLM()).with_tool()
    register_file_read_tool(agent.tool_registry, workspace_root=str(tmp_path))
    register_file_write_tool(agent.tool_registry, workspace_root=str(tmp_path))
    agent.with_multi_agent(
        workspace_root=str(tmp_path),
        storage_dir=str(tmp_path / "agents"),
    )

    agent_tool = agent.tool_registry.get_tool("Agent")
    child = agent_tool.agent_factory(
        SubagentRequest(
            description="Explore project",
            prompt="Locate the implementation",
            agent_type="Explore",
            workspace_root=str(tmp_path),
            allowed_roots=(str(tmp_path),),
        )
    )

    assert child.tool_registry.has_tool("FileRead")
    assert not child.tool_registry.has_tool("FileWrite")
    assert "codebase exploration specialist" in child.system_prompt
    child.close(close_llm=False)
    agent.close(close_llm=False)
