"""Ownership, lifecycle and shared-state regressions for AgentRuntime."""

import asyncio

import pytest

from easyagent import AgentRuntime, BasicAgent
from easyagent.hooks import BaseHook, HookManager
from easyagent.plans import PlanModeConfig
from easyagent.session import InMemorySessionStore
from runtime.invocation import AgentInvocationPhase
from test_permission_boundaries import OfflineLLM, run_path
from test_session_persistence import SessionLLM


def test_agents_own_independent_facilities_and_derived_state(tmp_path):
    parent = BasicAgent("parent", SessionLLM())
    parent.runtime.metadata["nested"] = {"values": [1]}
    parent.request_stop("parent only")
    child = BasicAgent("child", parent.llm, runtime=parent.runtime.derive(
        workspace_root=str(tmp_path), agent_id="worker", execution_mode="plan",
    ))
    try:
        assert child.runtime.agent_id == "worker"
        assert child.runtime.session_id != parent.runtime.session_id
        assert child.runtime.workspace_root == str(tmp_path)
        assert child.runtime.metadata["parentAgentId"] == parent.runtime.agent_id
        child.runtime.metadata["nested"]["values"].append(2)
        assert parent.runtime.metadata["nested"]["values"] == [1]
        assert not child.runtime.stop_requested
        for field in ("event_bus", "hooks", "metamessages", "interruptions", "invocation_lock"):
            assert getattr(child.runtime, field) is not getattr(parent.runtime, field)
        assert child._sessions.runtime is child.runtime
        assert child.build_prompt_context().runtime is child.runtime
        with pytest.raises(RuntimeError, match="cannot be shared"):
            BasicAgent("invalid", parent.llm, runtime=parent.runtime)
        with pytest.raises(AttributeError):
            child.runtime = parent.runtime
    finally:
        child.close(close_llm=False)
        parent.close()


@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
def test_runtime_tracks_each_invocation_and_releases_it(path):
    llm = OfflineLLM()
    llm._provider.calls = 1
    agent = BasicAgent("runtime-loop", llm)
    observations = []
    agent.event_bus.subscribe(lambda event: observations.append(
        (event.invocation_id, agent.runtime.active_invocation)
    ))
    try:
        run_path(agent, path)
        first = agent.runtime.last_invocation
        assert first is not None
        assert first.phase == AgentInvocationPhase.COMPLETED
        assert all(state is first for _, state in observations)
        assert agent.runtime.active_invocation is None
        assert not agent.runtime.invocation_lock.locked()
        observations.clear()
        run_path(agent, path)
        assert agent.runtime.last_invocation is not first
        assert agent.runtime.last_invocation.invocation_id != first.invocation_id
    finally:
        agent.close()


def test_runtime_blocks_overlapping_operations_and_close():
    agent = BasicAgent("busy", SessionLLM()).with_session(InMemorySessionStore())
    state = agent.runtime.begin_invocation("active", stream=False)
    try:
        with pytest.raises(RuntimeError, match="idle"):
            agent.save_session()
        with pytest.raises(RuntimeError, match="idle"):
            agent.close()
        with pytest.raises(RuntimeError, match="active invocation"):
            agent.invoke("overlap")
        assert agent.runtime.active_invocation is state
    finally:
        agent.runtime.end_invocation()
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.runtime.begin_invocation("closed", stream=False)


@pytest.mark.asyncio
async def test_cancelled_session_io_releases_runtime_reservation():
    entered = asyncio.Event()

    class WaitingStore(InMemorySessionStore):
        async def asave_session(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    agent = BasicAgent("async-session", SessionLLM()).with_session(WaitingStore())
    task = asyncio.create_task(agent.asave_session())
    await entered.wait()
    try:
        with pytest.raises(RuntimeError):
            agent.invoke("during save")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not agent.runtime.invocation_lock.locked()
        assert not agent._sessions.busy
        assert agent.invoke("after cancel") == "session-response"
    finally:
        agent.close()


def test_plan_updates_shared_runtime_and_restore_preserves_facilities():
    store = InMemorySessionStore()
    agent = BasicAgent("plan", SessionLLM()).with_session(store).with_plan(
        config=PlanModeConfig(register_tools=False)
    )
    runtime = agent.runtime
    facilities = (runtime.event_bus, runtime.metamessages, runtime.hooks, runtime.interruptions)
    try:
        agent.plan.enter(allowed_actions=["Read"])
        assert runtime.execution_mode == "plan"
        assert agent.build_authorization_context().plan_allowed_actions == ("Read",)
        session = agent.save_session()
        agent.plan.exit()
        assert runtime.execution_mode == "execute"
        agent.restore_session(session)
        assert runtime.execution_mode == "plan"
        assert agent.runtime is runtime
        assert facilities == (runtime.event_bus, runtime.metamessages, runtime.hooks, runtime.interruptions)
    finally:
        agent.close()


def test_hook_registration_keeps_pipeline_identity():
    agent = BasicAgent("hooks", SessionLLM())
    pipeline = agent.runtime.hooks
    one, two = BaseHook(), BaseHook()
    agent.with_hooks(HookManager([one]))
    agent.with_hooks(HookManager([two]))
    assert agent.runtime.hooks is pipeline
    assert pipeline.hooks == [one, two]
    agent.close()
    assert pipeline.hooks == []


def test_multiagent_handle_references_the_workers_actual_runtime(tmp_path):
    from multiagent.default import AgentRuntimeManager, SubagentRequest

    workers = []

    def factory(request):
        worker = BasicAgent("worker", SessionLLM())
        workers.append(worker)
        return worker

    manager = AgentRuntimeManager(agent_factory=factory, storage_dir=str(tmp_path))
    try:
        handle = manager.run(SubagentRequest(description="probe", prompt="hello"))
        assert handle.status == "completed"
        assert handle.runtime is workers[0].runtime
        assert handle.runtime.agent_id == handle.agent_id
        assert handle.runtime.last_invocation is not None
    finally:
        manager.close()


@pytest.mark.parametrize("path", ["invoke", "ainvoke", "stream", "astream"])
def test_stop_releases_invocation_without_losing_runtime(path):
    from core.Exception import AgentStopRequested

    agent = BasicAgent("stop", OfflineLLM())
    runtime = agent.runtime
    try:
        runtime.request_stop("cancel test")
        with pytest.raises(AgentStopRequested, match="cancel test"):
            run_path(agent, path)
        assert runtime.active_invocation is None
        assert runtime.last_invocation.phase == AgentInvocationPhase.INTERRUPTED
        assert not runtime.invocation_lock.locked()
    finally:
        agent.clear_stop_request()
        assert not runtime.stop_requested
        agent.close()
