from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

from pydantic import BaseModel

import agent
from agent import BasicAgent
from agent.components.prompt_composer import BaseSystemPromptComposer, PromptBuildContext
from core.hooks import BaseHook, HookDecision, HookManager
from core.llm import EasyLLM
from metamessage import MetaMessage, MetaMessageLifecycle
from observability import InMemoryObservabilityStore
from plan import PlanModeConfig
from prompt import PromptBlock
from runtime import AgentRuntimeManager, AgentStreamEventType, ExecutionContext
from Tool import Tool, ToolRegistry
from Tool.runtime import SubagentRequest


def _response(
    content: str | None = None,
    *,
    tool_calls: list[SimpleNamespace] | None = None,
    input_tokens: int = 7,
    output_tokens: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=list(tool_calls or []),
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


class ScriptedProvider:
    def __init__(self, *, tool_flow: bool = False) -> None:
        self.tool_flow = tool_flow
        self.last_request: dict = {}

    def build_tool_payload(self, tools):
        return list(tools)

    def build_request(
        self,
        messages,
        *,
        system_prompt=None,
        tools=None,
        temperature=None,
        reasoning=None,
        stream=False,
        **kwargs,
    ):
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(list(messages))
        request = {
            "messages": request_messages,
            "tools": tools,
            "temperature": temperature,
            "reasoning": reasoning,
            "stream": stream,
            **kwargs,
        }
        self.last_request = request
        return request

    def apply_cache_policy(self, request, request_input):
        return request

    def invoke_raw(self, request):
        self.last_request = request
        if self.tool_flow and request.get("tools"):
            used_tool = any(item.get("role") == "tool" for item in request["messages"])
            if not used_tool:
                return _response(
                    tool_calls=[
                        SimpleNamespace(
                            id="call_echo",
                            function=SimpleNamespace(
                                name="Echo",
                                arguments='{"text":"hello"}',
                            ),
                        )
                    ]
                )
            return _response("tool-complete", input_tokens=9, output_tokens=4)
        return _response("plain-response")

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    def stream_raw(self, request):
        self.last_request = request
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="stream-",
                        reasoning_content=None,
                        reasoning=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="response",
                        reasoning_content=None,
                        reasoning=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ]
        )

    async def async_stream_raw(self, request):
        async def chunks():
            for item in self.stream_raw(request):
                yield item

        return chunks()

    def close(self):
        return None


class ScriptedLLM(EasyLLM):
    def __init__(self, *, tool_flow: bool = False) -> None:
        self.provider_name = "openai"
        self.model = "test-model"
        self.base_url = "http://test.local/v1"
        self.api_key = "test-key"
        self.max_tokens = 256
        self.temperature = 0.2
        self.timeout = 60
        self.kwargs = {}
        self._provider = ScriptedProvider(tool_flow=tool_flow)
        self.client = None


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="Echo",
            description="Return the supplied text.",
            parameters=EchoParams,
            read_only=True,
            side_effect_level="none",
            resource_scope=["runtime"],
        )

    def run(self, parameters: dict):
        return f"echo:{parameters['text']}"


class MinimalPrompt(BaseSystemPromptComposer):
    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return [
            PromptBlock("identity", f"Agent={context.agent_name}", placement="system"),
            PromptBlock("runtime", "temporary runtime rule", placement="system_reminder"),
        ]


class RewriteResponseHook(BaseHook):
    def after_llm_response(self, payload: dict):
        response = payload["response"]
        if isinstance(response, dict):
            return HookDecision.modify({"response": {**response, "content": "stream-rewritten"}})
        response.content = "invoke-rewritten"
        return HookDecision.modify({"response": response})


def test_basic_agent_has_light_defaults_and_only_final_invocation_api():
    signature = inspect.signature(BasicAgent)
    assert list(signature.parameters) == [
        "name",
        "llm",
        "system_prompt",
        "description",
        "config",
    ]

    instance = BasicAgent("minimal", ScriptedLLM())
    assert instance.tool_registry is None
    assert instance.context_manager is None
    assert instance.plan is None
    assert instance.observability is None
    assert instance.multi_agent is None
    assert not hasattr(instance, "invoke_with_tool")
    assert not hasattr(instance, "stream_invoke")
    assert not hasattr(agent, "ReactAgent")


def test_with_methods_install_dependencies_without_replacing_custom_modules():
    instance = BasicAgent("composed", ScriptedLLM())
    assert not hasattr(instance, "with_metamessage")
    registry = ToolRegistry()
    instance.with_tool(registry).with_plan(
        config=PlanModeConfig(register_tools=False)
    )

    assert instance.tool_registry is registry
    assert instance.plan is not None
    assert instance.with_multi_agent().tool_registry is registry
    expected = {
        "Agent",
        "AgentGet",
        "AgentList",
        "AgentWait",
        "AgentStop",
        "SendMessage",
        "MailboxRead",
        "MailboxAck",
        "TeamCreate",
        "TeamDelete",
    }
    assert expected.issubset(set(registry.get_tool_names()))


def test_custom_prompt_and_metamessage_share_history_without_agent_coupling():
    llm = ScriptedLLM()
    instance = BasicAgent("prompted", llm).with_prompt(MinimalPrompt())
    instance.metamessage_manager.subscribe(
        "agent.invoke.started",
        lambda _context: MetaMessage(
            name="invoke-rule",
            content="injected invocation rule",
            lifecycle=MetaMessageLifecycle.INVOCATION,
        ),
    )

    assert instance.invoke("question") == "plain-response"
    request = llm.provider.last_request
    assert request["messages"][0] == {"role": "system", "content": "Agent=prompted"}
    user_text = "\n".join(
        str(item.get("content") or "")
        for item in request["messages"]
        if item.get("role") == "user"
    )
    assert "temporary runtime rule" in user_text
    assert "injected invocation rule" in user_text
    assert not any(
        message.metadata.get("metaMessageLifecycle") == "invocation"
        for message in instance.get_canonical_history()
    )


def test_tool_execution_is_inferred_and_runtime_events_are_json_safe():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    instance = BasicAgent("tools", ScriptedLLM(tool_flow=True)).with_tool(registry)

    assert instance.invoke("use echo") == "tool-complete"
    events = instance.get_trace_history()
    event_types = [item["type"] for item in events]
    assert event_types == [
        "agent.invoke.started",
        "llm.invoke.started",
        "llm.invoke.completed",
        "tool.invoke.started",
        "tool.invoke.completed",
        "llm.invoke.started",
        "llm.invoke.completed",
        "agent.invoke.completed",
    ]
    assert events[3]["data"]["tool_name"] == "Echo"
    json.dumps(events)


def test_sync_async_and_stream_paths_share_response_hooks_and_events():
    hook_manager = HookManager([RewriteResponseHook()])

    sync_agent = BasicAgent("sync", ScriptedLLM()).with_hooks(hook_manager)
    assert sync_agent.invoke("one") == "invoke-rewritten"

    async_agent = BasicAgent("async", ScriptedLLM()).with_hooks(
        HookManager([RewriteResponseHook()])
    )
    assert asyncio.run(async_agent.ainvoke("two")) == "invoke-rewritten"

    stream_agent = BasicAgent("stream", ScriptedLLM()).with_hooks(
        HookManager([RewriteResponseHook()])
    )
    stream_events = list(stream_agent.stream("three"))
    assert stream_events[-1].type == AgentStreamEventType.FINAL
    assert stream_events[-1].content == "stream-rewritten"

    async_stream_agent = BasicAgent("astream", ScriptedLLM()).with_hooks(
        HookManager([RewriteResponseHook()])
    )

    async def collect():
        return [item async for item in async_stream_agent.astream("four")]

    async_events = asyncio.run(collect())
    assert async_events[-1].type == AgentStreamEventType.FINAL
    assert async_events[-1].content == "stream-rewritten"


def test_plan_transitions_are_permanent_metamessages():
    instance = BasicAgent("planner", ScriptedLLM()).with_plan(
        config=PlanModeConfig(
            enter_message="plan-only",
            exit_message="execute-now",
            register_tools=False,
        )
    )
    instance.enter_plan_mode(allowed_actions=["FileRead"])
    instance.invoke("inspect")
    instance.exit_plan_mode()
    instance.invoke("implement")

    plan_messages = [
        message
        for message in instance.get_canonical_history()
        if message.metadata.get("source") == "plan"
    ]
    assert [message.metadata["mode"] for message in plan_messages] == ["plan", "execute"]
    assert "FileRead" in plan_messages[0].text_content()


def test_observability_consumes_runtime_events_once():
    instance = BasicAgent("observed", ScriptedLLM()).with_observability(
        store=InMemoryObservabilityStore()
    )
    assert instance.invoke("observe") == "plain-response"

    record = instance.observability.latest()
    assert record is not None
    assert record.stats.success
    assert record.stats.llm_calls == 1
    assert record.stats.total_tokens == 10
    assert record.llm_invokes[0].options["provider"] == "openai"
    assert record.llm_invokes[0].options["model"] == "test-model"


class _RuntimeProbeAgent:
    def __init__(self, runtime: AgentRuntimeManager, request: SubagentRequest) -> None:
        self.runtime = runtime
        self.request = request

    def invoke(self, query: str) -> str:
        agent_id = self.request.metadata["agent_id"]
        messages = self.runtime.read_mailbox(agent_id)
        return "|".join(item.content for item in messages)

    def get_trace_history(self):
        return []

    def get_context_usage(self):
        return {"estimatedRequestTokens": 0}


def test_runtime_registers_before_factory_and_supports_per_launch_factory(tmp_path):
    runtime: AgentRuntimeManager

    def root_factory(request: SubagentRequest):
        raise AssertionError("per-launch factory was ignored")

    runtime = AgentRuntimeManager(
        agent_factory=root_factory,
        storage_dir=str(tmp_path / "agents"),
    )

    def current_factory(request: SubagentRequest):
        runtime.send_message(
            recipient_type="agent",
            recipient_id=request.metadata["agent_id"],
            sender_id="manager",
            content="registered-before-factory",
        )
        return _RuntimeProbeAgent(runtime, request)

    context = ExecutionContext(
        workspace_root=str(tmp_path),
        allowed_roots=(str(tmp_path),),
        metadata={"agentId": "parent"},
    )
    handle = runtime.run(
        SubagentRequest(
            description="probe",
            prompt="read mailbox",
            workspace_root=str(tmp_path),
            allowed_roots=(str(tmp_path),),
        ),
        execution_context=context,
        agent_factory=current_factory,
    )

    assert handle.status == "completed"
    assert handle.content == "registered-before-factory"
    assert handle.execution_context.metadata["agentId"] == handle.agent_id
    assert handle.output_file
