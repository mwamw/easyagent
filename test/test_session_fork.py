import pytest
from easyagent import BasicAgent
from easyagent.session import SQLiteSessionStore as SessionStore
from test_session_persistence import SessionLLM


def test_public_fork_excludes_execution_handles_and_preserves_source(tmp_path):
    store = SessionStore(str(tmp_path / "fork.sqlite"))
    agent = BasicAgent("product", SessionLLM())
    try:
        agent.add_user_message("source")
        agent.current_task_id = "active-task"
        agent.save_session("source", store=store)
        agent.fork_session("branch", store=store, metadata={"parent": "source"})
        source = store.get_session("source", touch=False)
        branch = store.get_session("branch", touch=False)
        assert source["snapshot"]["currentTaskId"] == "active-task"
        assert agent.current_task_id == "active-task"
        assert branch["snapshot"]["currentTaskId"] is None
        assert branch["snapshot"]["history"] == source["snapshot"]["history"]
        assert branch["metadata"]["parent"] == "source"
        assert branch["metadata"]["forked_from_session_id"] == "source"
        with pytest.raises(Exception, match="revision changed"):
            agent.fork_session("source", store=store)
        assert store.get_session("source", touch=False)["snapshot"] == source["snapshot"]
    finally:
        agent.close()


def test_fork_rejects_pending_interaction(tmp_path):
    store = SessionStore(str(tmp_path / "fork.sqlite"))
    agent = BasicAgent("product", SessionLLM())
    try:
        agent.interrupt_controller.restore_state({"last_tool_interrupt": {"tool_id": "pending", "tool_name": "Bash"}})
        with pytest.raises(Exception, match="pending interaction"):
            agent.fork_session("branch", store=store)
        assert store.get_session("branch", touch=False) is None
    finally:
        agent.close()
