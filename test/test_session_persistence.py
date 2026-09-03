from __future__ import annotations

from datetime import datetime, timedelta
import os
from types import SimpleNamespace

from pydantic import BaseModel

from agent import BasicAgent
from agent.components.prompt_composer import BaseSystemPromptComposer, PromptBuildContext
from core.history import CanonicalBlock, CanonicalMessage
from core.llm import EasyLLM
from db import ConversationStore, SessionStore
from metamessage import MetaMessage, MetaMessageLifecycle
from observability import InMemoryObservabilityStore
from plan import PlanModeConfig
from prompt import PromptBlock
from Tool import Tool, ToolRegistry


class SessionProvider:
    def __init__(self) -> None:
        self.closed = False

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
        values = []
        if system_prompt:
            values.append({"role": "system", "content": system_prompt})
        values.extend(list(messages))
        return {"messages": values, "tools": tools, "stream": stream}

    def apply_cache_policy(self, request, request_input):
        return request

    def invoke_raw(self, request):
        return SimpleNamespace(
            content="session-response",
            reasoning_content=None,
            tool_calls=[],
            usage=SimpleNamespace(
                prompt_tokens=8,
                completion_tokens=2,
                total_tokens=10,
            ),
        )

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    def close(self):
        self.closed = True


class SessionLLM(EasyLLM):
    def __init__(self, provider_name: str = "openai") -> None:
        self.provider_name = provider_name
        self.model = "session-model"
        self.base_url = "http://session.local/v1"
        self.api_key = "session-key"
        self.max_tokens = 256
        self.temperature = 0.1
        self.timeout = 60
        self.kwargs = {}
        self._provider = SessionProvider()
        self.client = None


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="Echo",
            description="Echo a value.",
            parameters=EchoParams,
            read_only=True,
            side_effect_level="none",
        )

    def run(self, parameters: dict):
        return parameters["text"]


class CustomPrompt(BaseSystemPromptComposer):
    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return [PromptBlock("custom", "custom prompt")]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    return registry


def test_session_store_crud_and_expiration(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.sqlite3"))
    store.create_or_update_session(
        session_id="active",
        agent_type="BasicAgent",
        agent_name="assistant",
        snapshot={"schemaVersion": 3},
    )
    store.create_or_update_session(
        session_id="expired",
        agent_type="BasicAgent",
        agent_name="assistant",
        snapshot={"schemaVersion": 3},
        expires_at=datetime.now() - timedelta(seconds=1),
    )

    assert [item["session_id"] for item in store.list_sessions()] == ["active"]
    assert store.cleanup_expired_sessions() == 1
    assert store.get_session("expired") is None
    assert store.delete_session("active")


def test_conversation_store_round_trips_canonical_messages(tmp_path):
    path = str(tmp_path / "conversations.sqlite3")
    sessions = SessionStore(path)
    conversations = ConversationStore(path)
    sessions.create_or_update_session(
        session_id="conversation",
        agent_type="BasicAgent",
        agent_name="assistant",
        snapshot={"schemaVersion": 3},
    )
    messages = [
        CanonicalMessage(
            role="user",
            content=[CanonicalBlock(type="text", text="hello")],
            metadata={"source": "test"},
        ),
        CanonicalMessage(
            role="assistant",
            content=[CanonicalBlock(type="text", text="world")],
        ),
    ]

    conversations.replace_messages("conversation", messages)
    restored = conversations.load_messages("conversation")

    assert [item.role for item in restored] == ["user", "assistant"]
    assert restored[0].metadata["source"] == "test"
    assert restored[1].text_content() == "world"


def test_basic_agent_restores_explicit_standard_modules(tmp_path):
    store = SessionStore(str(tmp_path / "agent.sqlite3"))
    agent = (
        BasicAgent("session-agent", SessionLLM(), system_prompt="session prompt")
        .with_tool(_registry())
        .with_plan(config=PlanModeConfig(register_tools=False))
        .with_observability(store=InMemoryObservabilityStore())
    )
    agent.emit_metamessage(
        MetaMessage(
            name="permanent-rule",
            content="keep this rule",
            lifecycle=MetaMessageLifecycle.PERMANENT,
            dedup_key="permanent-rule",
        )
    )
    agent.enter_plan_mode(allowed_actions=["Echo"])
    assert agent.invoke("persist state") == "session-response"
    agent.save_session("session-v3", store=store)

    snapshot = store.get_session("session-v3")["snapshot"]
    assert snapshot["schemaVersion"] == 3
    assert snapshot["modules"]["tools"]["state"]["names"] == ["Echo"]
    assert snapshot["modules"]["plan"]["implementation"] == "PlanModeManager"
    assert snapshot["modules"]["observability"]["implementation"] == "ObservabilityManager"
    saved_events = snapshot["modules"]["runtimeEvents"]["state"]["events"]
    assert saved_events[-1]["type"] == "agent.invoke.completed"

    restored = BasicAgent.load_session(
        "session-v3",
        llm=SessionLLM(),
        store=store,
        tool_registry=_registry(),
    )

    assert restored.system_prompt == "session prompt"
    assert restored.plan is not None and restored.plan.is_active
    assert restored.observability is not None
    assert restored.observability.latest().query == "persist state"
    assert restored.get_trace_history() == saved_events
    assert [item.role for item in restored.get_canonical_history()][-1] == "assistant"
    assert any(
        item.metadata.get("metaMessageName") == "permanent-rule"
        for item in restored.get_canonical_history()
    )
    assert restored.get_last_restore_report()["status"] == "restored"


def test_session_restores_directory_skills_and_conditional_activation(tmp_path):
    skill_directory = tmp_path / "skills" / "python-review"
    skill_directory.mkdir(parents=True)
    (skill_directory / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: python-review",
                "description: Review Python modules",
                'paths: "src/**/*.py"',
                "---",
                "Review the Python file selected by the user.",
            ]
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(str(tmp_path / "skills.sqlite3"))
    agent = BasicAgent("skill-session", SessionLLM()).with_skill(tmp_path / "skills")
    agent.execution_context.workspace_root = str(workspace)
    assert agent.skill_manager.activate_for_paths(["src/package/module.py"]) == [
        "python-review"
    ]
    agent.save_session("skill-session", store=store)

    snapshot = store.get_session("skill-session")["snapshot"]
    skill_state = snapshot["modules"]["skills"]["state"]
    assert skill_state["activatedPathSkills"] == ["python-review"]
    assert [item["name"] for item in skill_state["skills"]] == ["python-review"]

    restored = BasicAgent.load_session(
        "skill-session",
        llm=SessionLLM(),
        store=store,
    )

    assert restored.skill_manager is not None
    assert restored.skill_manager.has_skill("python-review")
    assert "`python-review`" in restored.get_system_prompt_template().render_system_reminders()
    assert restored.metamessage_manager.list_pending() == []
    assert restored.get_last_restore_report()["components"]["skills"]["status"] == "restored"


def test_restore_reports_missing_custom_and_optional_modules(tmp_path):
    from task import InMemoryTaskStore, TaskService

    store = SessionStore(str(tmp_path / "missing.sqlite3"))
    agent = (
        BasicAgent("custom", SessionLLM())
        .with_prompt(CustomPrompt())
        .with_tool(_registry())
        .with_task_service(TaskService(InMemoryTaskStore()))
    )
    agent.save_session("missing", store=store)

    restored = BasicAgent.load_session(
        "missing",
        llm=SessionLLM(),
        store=store,
    )
    report = restored.get_last_restore_report()
    codes = {item["code"] for item in report["issues"]}

    assert report["status"] == "degraded"
    assert "prompt_implementation_missing" in codes
    assert "tasks_implementation_missing" in codes
    assert "missing_tools" in codes


def test_change_model_rebuilds_provider_replay_history():
    agent = BasicAgent("history", SessionLLM("openai"))
    agent.add_user_message("hello")
    agent.add_assistant_message("world")

    agent.change_model(SessionLLM("anthropic_native"))

    assert agent.history_store.provider_name == "anthropic_native"
    assert [item.role for item in agent.get_canonical_history()] == ["user", "assistant"]
    assert [item["role"] for item in agent.replay_history] == ["user", "assistant"]


def test_close_is_idempotent_and_closes_llm():
    llm = SessionLLM()
    agent = BasicAgent("closable", llm).with_observability(
        store=InMemoryObservabilityStore()
    )

    first = agent.close()
    second = agent.close()

    assert first["status"] == "closed"
    assert first["components"]["observability"]["status"] == "closed"
    assert first["components"]["llm"]["status"] == "closed"
    assert second == first
    assert llm.provider.closed
