from copy import deepcopy
import pytest

from easyagent import BasicAgent
from easyagent.session import InMemorySessionStore, SQLiteSessionStore, SessionConflictError
from easyagent.memory import LayeredMemory
from session.migration import migrate_session_v3
from test_session_persistence import SessionLLM


@pytest.fixture(params=["memory", "sqlite"])
def sessions(request, tmp_path):
    return InMemorySessionStore() if request.param == "memory" else SQLiteSessionStore(tmp_path / "sessions.db")


def test_restore_preserves_live_objects_and_detects_conflicting_saves(sessions):
    first = BasicAgent("first", SessionLLM()).with_session(sessions)
    first.add_user_message("original")
    session_id = first.save_session(metadata={"project": "a"})
    second = BasicAgent("second", SessionLLM()).with_session(sessions)
    components = (second.llm, second.executor, second.prompt_composer, second.hook_manager)
    second.restore_session(session_id)
    assert components == (second.llm, second.executor, second.prompt_composer, second.hook_manager)
    assert second.name == "second"
    first.add_user_message("new")
    first.save_session()
    with pytest.raises(SessionConflictError):
        second.save_session()
    assert len(sessions.get_session(session_id)["snapshot"]["history"]["canonical"]) == 2
    assert "snapshot" not in sessions.list_sessions()[0]


def test_restore_layered_session_state_without_copying_external_memory(sessions):
    first = BasicAgent("first", SessionLLM()).with_memory(LayeredMemory()).with_session(sessions)
    first.memory.add_memory("working fact", "working", .9)
    session_id = first.save_session()
    second = BasicAgent("second", SessionLLM()).with_memory(LayeredMemory()).with_session(sessions)
    second.restore_session(session_id)
    assert second.memory.memory_types["working"].get_all_memories()[0].content == "working fact"
    assert second.memory.services.scope.session_id == session_id


def test_custom_session_state_can_be_registered_without_agent_changes(sessions):
    class Counter:
        def __init__(self):
            self.value = 0
        def __call__(self):
            raise AssertionError("A callable component is not a component factory")
        def export_state(self):
            return {"value": self.value}
        def restore_state(self, state):
            self.value = state["value"]
    counter = Counter()
    first = BasicAgent("first", SessionLLM()).with_session(sessions)
    first.register_session_state("counter", counter, required=True)
    counter.value = 42
    session_id = first.save_session()
    second = BasicAgent("second", SessionLLM()).with_session(sessions)
    second.add_user_message("unchanged on failure")
    before = deepcopy(second.history)
    with pytest.raises(Exception, match="Required session component"):
        second.restore_session(session_id)
    assert second.history == before
    target = Counter()
    second.register_session_state("counter", target, required=True)
    second.restore_session(session_id)
    assert target.value == 42


def test_switching_to_session_without_memory_does_not_leak_working_state(sessions):
    without = BasicAgent("without", SessionLLM()).with_session(sessions)
    session_id = without.save_session()
    current = BasicAgent("with", SessionLLM()).with_session(sessions).with_memory(LayeredMemory())
    current.memory.add_memory("private to old session", "working", .9)
    current.restore_session(session_id)
    assert current.memory.memory_types["working"].get_all_memories() == []


@pytest.mark.asyncio
async def test_async_save_restore(sessions):
    first = BasicAgent("first", SessionLLM()).with_session(sessions)
    await first.ainvoke("hello")
    session_id = await first.asave_session()
    other = BasicAgent("other", SessionLLM()).with_session(sessions)
    await other.arestore_session(session_id)
    assert len(other.history) == 2


def test_session_migration_is_copy_only(tmp_path):
    from db import SessionStore as LegacyStore
    source, target = tmp_path / "old.db", tmp_path / "new.db"
    original = BasicAgent("legacy", SessionLLM())
    original.add_user_message("retained")
    LegacyStore(str(source)).create_or_update_session("old", "BasicAgent", "legacy", {
        "schemaVersion": 3, "history": original.context_manager.export_state(), "modules": {},
    }, metadata={"title": "old title"})
    before = source.read_bytes()
    assert migrate_session_v3(source, target)["count"] == 1
    assert not target.exists()
    migrate_session_v3(source, target, dry_run=False)
    restored = BasicAgent("new", SessionLLM()).with_session(SQLiteSessionStore(target))
    restored.restore_session("old")
    assert restored.history[0].text_content() == "retained"
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        migrate_session_v3(source, target, dry_run=False)


@pytest.mark.asyncio
async def test_async_only_session_backend_is_really_awaited():
    from easyagent.session import BaseSessionStore
    class AsyncStore(BaseSessionStore):
        def __init__(self):
            self.inner = InMemorySessionStore()
            self.reads = self.writes = 0
        async def aget_session(self, session_id):
            self.reads += 1
            return self.inner.get_session(session_id)
        async def asave_session(self, **kwargs):
            self.writes += 1
            return self.inner.create_or_update_session(**kwargs)
    store = AsyncStore()
    first = BasicAgent("first", SessionLLM()).with_session(store)
    first.add_user_message("native async storage")
    session_id = await first.asave_session()
    second = BasicAgent("second", SessionLLM()).with_session(store)
    await second.arestore_session(session_id)
    assert store.reads == store.writes == 1
    assert second.history[0].text_content() == "native async storage"
