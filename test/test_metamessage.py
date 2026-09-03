from __future__ import annotations

import pytest

from agent.BasicAgent import BasicAgent
from core.history import CanonicalMessage, coerce_canonical_message
from core.llm import EasyLLM
from metamessage import (
    MetaMessage,
    MetaMessageContext,
    MetaMessageLifecycle,
    MetaMessageManager,
)
from plan import PlanModeConfig


class MemoryHistoryPort:
    def __init__(self):
        self.messages: dict[str, CanonicalMessage] = {}

    def append(self, message: CanonicalMessage) -> str:
        handle = str(message.metadata["metaMessageInjectionId"])
        self.messages[handle] = message
        return handle

    def remove(self, history_handle: str) -> bool:
        return self.messages.pop(history_handle, None) is not None

    def contains(self, history_handle: str) -> bool:
        return history_handle in self.messages


class StubLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.client = None


def build_manager():
    state = {
        "execution_mode": "execute",
        "permission_mode": "default",
        "current_task_id": None,
    }
    history = MemoryHistoryPort()
    manager = MetaMessageManager(
        history_port=history,
        context_provider=lambda: MetaMessageContext(**state),
    )
    return state, history, manager


def test_permanent_message_is_inserted_once_and_retained():
    _, history, manager = build_manager()
    manager.register(
        MetaMessage(
            name="constraint",
            content="Keep this constraint.",
            lifecycle=MetaMessageLifecycle.PERMANENT,
        )
    )

    assert len(manager.flush()) == 1
    assert manager.flush() == []
    manager.publish("agent.invoke.completed")
    assert [item.text_content() for item in history.messages.values()] == [
        "Keep this constraint."
    ]


def test_runtime_scoped_messages_must_be_emitted_by_modules():
    _, _, manager = build_manager()
    for lifecycle in (MetaMessageLifecycle.INVOCATION, MetaMessageLifecycle.REQUEST):
        with pytest.raises(ValueError):
            manager.register(
                MetaMessage(name="invalid", content="invalid", lifecycle=lifecycle)
            )


def test_runtime_events_reclaim_request_then_invocation_messages():
    _, history, manager = build_manager()
    manager.publish("agent.invoke.started", {"query": "test"})
    manager.emit(
        MetaMessage(
            name="skill",
            content="Temporary skill body",
            lifecycle=MetaMessageLifecycle.INVOCATION,
        )
    )
    manager.emit(
        MetaMessage(
            name="request_hint",
            content="Use this once",
            lifecycle=MetaMessageLifecycle.REQUEST,
        )
    )
    manager.flush()
    assert len(history.messages) == 2

    manager.publish("llm.invoke.completed")
    assert [item.text_content() for item in history.messages.values()] == [
        "Temporary skill body"
    ]

    manager.publish("agent.invoke.completed")
    assert history.messages == {}
    assert manager.invocation_active is False


def test_conditional_message_tracks_false_true_false_transitions():
    state, history, manager = build_manager()
    manager.register(
        MetaMessage(
            name="plan_condition",
            content=lambda context: f"Mode: {context.execution_mode}",
            lifecycle=MetaMessageLifecycle.CONDITIONAL,
            condition=lambda context: context.execution_mode == "plan",
        )
    )

    assert manager.flush() == []
    state["execution_mode"] = "plan"
    assert len(manager.flush()) == 1
    assert manager.flush() == []
    assert len(history.messages) == 1

    state["execution_mode"] = "execute"
    assert manager.flush() == []
    assert history.messages == {}


def test_event_factory_and_dedup_key_deliver_one_permanent_message():
    _, history, manager = build_manager()
    manager.subscribe(
        "mailbox.received",
        lambda context: MetaMessage(
            name="mailbox",
            content=str(context.payload["content"]),
            lifecycle=MetaMessageLifecycle.PERMANENT,
            dedup_key=f"mailbox:{context.payload['message_id']}",
        ),
    )
    payload = {"message_id": "msg-1", "content": "Change the output format"}
    manager.publish("mailbox.received", payload)
    assert len(manager.flush()) == 1
    manager.publish("mailbox.received", payload)
    assert manager.flush() == []
    assert len(history.messages) == 1


def test_context_has_agent_state_but_no_runtime_ids():
    context = MetaMessageContext(
        execution_mode="plan",
        permission_mode="default",
        current_task_id="task-1",
        payload={"value": 1},
    )
    assert context.execution_mode == "plan"
    assert context.current_task_id == "task-1"
    assert not hasattr(context, "invocation_id")
    assert not hasattr(context, "request_id")


def test_session_state_does_not_persist_temporary_pending_messages():
    _, _, manager = build_manager()
    manager.emit(
        MetaMessage(
            name="skill",
            content="temporary",
            lifecycle=MetaMessageLifecycle.INVOCATION,
        )
    )
    manager.emit(
        MetaMessage(
            name="permanent",
            content="retained",
            lifecycle=MetaMessageLifecycle.PERMANENT,
        )
    )

    state = manager.export_state()
    assert [item["name"] for item in state["pending"]] == ["permanent"]


def test_basic_agent_plan_transitions_are_permanent_history():
    agent = BasicAgent(name="planner", llm=StubLLM()).with_plan(
        config=PlanModeConfig(register_tools=False)
    )
    agent.enter_plan_mode(allowed_actions=["FileRead", "Glob"])
    agent.metamessage_manager.flush()
    agent.exit_plan_mode()
    agent.metamessage_manager.flush()

    transitions = [
        message
        for raw in agent.history
        if (message := coerce_canonical_message(raw)) is not None
        and message.metadata.get("source") == "plan"
    ]
    assert [message.metadata["mode"] for message in transitions] == ["plan", "execute"]
    assert "FileRead" in transitions[0].text_content()
    assert "Plan mode has ended" in transitions[1].text_content()
