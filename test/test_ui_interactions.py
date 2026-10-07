"""Public reply/continuation APIs, exercised without terminal or model services."""
import asyncio
import json
from contextlib import aclosing
from types import SimpleNamespace
from threading import Event

import pytest

from agent import BasicAgent
from core.Exception import AgentStopRequested, ToolInterruption
from permissions import BasePermissionEngine, PermissionBehavior, PermissionDecision
from runtime.interactions import InteractionReply
from runtime import AgentStreamEventType
from Tool.builtin.interaction_tools import AskUserQuestionTool
from test_permission_boundaries import OfflineLLM, CountingWriteTool


async def interrupt(agent):
    with pytest.raises(ToolInterruption):
        async for _ in agent.astream("run task"):
            pass
    return agent.get_pending_interruption()


def tool_results(agent):
    return [
        block for message in agent.context_manager.canonical for block in message.content
        if block.type == "function_response"
    ]


@pytest.mark.parametrize("action, calls", [("approve", 1), ("deny", 0)])
def test_public_reply_and_continuation_do_not_inject_user_messages(action, calls):
    async def run():
        tool = CountingWriteTool()
        agent = BasicAgent("ui", OfflineLLM()).add_tool(tool).with_permissions()
        try:
            pending = await interrupt(agent)
            await agent.arespond_to_interruption(InteractionReply(pending["request_id"], action))
            result = [event async for event in agent.astream_continue()]
            assert result[-1].content == "done"
            assert tool.calls == calls
            assert agent.get_pending_interruption() is None
            assert len(tool_results(agent)) == 1
            assert len([m for m in agent.context_manager.canonical if m.role == "user"]) == 1
            assert agent.permission_engine.grants() == []
            with pytest.raises(ValueError):
                await agent.arespond_to_interruption(InteractionReply(pending["request_id"], action))
        finally:
            await agent.aclose()
    asyncio.run(run())


def test_approval_cannot_override_deny_added_after_prompt():
    async def run():
        tool = CountingWriteTool()
        agent = BasicAgent("ui", OfflineLLM()).add_tool(tool).with_permissions()
        try:
            pending = await interrupt(agent)
            agent.permission_engine.add_rule(tool_name=tool.name, behavior="deny")
            await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "approve"))
            assert tool.calls == 0
            assert agent.permission_engine.grants() == []
            assert len(tool_results(agent)) == 1
        finally:
            await agent.aclose()
    asyncio.run(run())


def test_stale_reply_preserves_pending_action():
    async def run():
        agent = BasicAgent("ui", OfflineLLM()).add_tool(CountingWriteTool()).with_permissions()
        try:
            pending = await interrupt(agent)
            with pytest.raises(ValueError):
                await agent.arespond_to_interruption(InteractionReply("wrong", "approve"))
            assert agent.get_pending_interruption() == pending
        finally:
            await agent.aclose()
    asyncio.run(run())


def test_multiple_calls_resume_without_reexecuting_previous_siblings():
    async def run():
        llm = OfflineLLM()
        original = llm._provider.invoke_raw

        def invoke(request):
            response = original(request)
            if response.tool_calls:
                response.tool_calls.append(SimpleNamespace(
                    id="write-2", function=SimpleNamespace(name="WriteProbe", arguments="{}"),
                ))
            return response

        llm._provider.invoke_raw = invoke
        def stream(request):
            response = invoke(request)
            calls = [SimpleNamespace(index=i, id=call.id, function=call.function)
                     for i, call in enumerate(response.tool_calls)]
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=response.content, tool_calls=calls,
                                      reasoning_content=None, reasoning=None),
                finish_reason="tool_calls" if calls else "stop",
            )])
        llm._provider.stream_raw = stream
        tool = CountingWriteTool()
        agent = BasicAgent("ui", llm).add_tool(tool).with_permissions()
        try:
            first = await interrupt(agent)
            with pytest.raises(ToolInterruption):
                await agent.arespond_to_interruption(InteractionReply(first["request_id"], "approve"))
            second = agent.get_pending_interruption()
            assert second["tool_id"] == "write-2"
            assert tool.calls == 1
            await agent.arespond_to_interruption(InteractionReply(second["request_id"], "approve"))
            assert tool.calls == 2
            assert len(tool_results(agent)) == 2
            assert agent.permission_engine.grants() == []
            assert [e async for e in agent.astream_continue()][-1].content == "done"
        finally:
            await agent.aclose()
    asyncio.run(run())


def redirect_model_tool(llm, name, arguments):
    original = llm._provider.invoke_raw

    def invoke(request):
        response = original(request)
        if response.tool_calls:
            response.tool_calls[0].function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
        return response
    llm._provider.invoke_raw = invoke


def test_question_answer_uses_result_not_reexecution():
    async def run():
        llm = OfflineLLM()
        redirect_model_tool(llm, "AskUserQuestion", {"questions": [{
            "question": "Which option?", "header": "Choice",
            "options": [{"label": "A", "description": "Option A"}, {"label": "B", "description": "Option B"}],
        }]})
        agent = BasicAgent("ui", llm).add_tool(AskUserQuestionTool())
        try:
            pending = await interrupt(agent)
            with pytest.raises(ValueError):
                await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "approve"))
            await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "answer", "B"))
            assert len(tool_results(agent)) == 1
            assert [e async for e in agent.astream_continue()][-1].content == "done"
        finally:
            await agent.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("name, expected", [("EnterPlanMode", "plan"), ("ExitPlanMode", "execute")])
def test_plan_reply_changes_plan_through_public_contract(name, expected):
    async def run():
        llm = OfflineLLM()
        redirect_model_tool(llm, name, {})
        agent = BasicAgent("ui", llm).with_plan().with_permissions()
        if name == "ExitPlanMode":
            agent.enter_plan_mode()
        try:
            pending = await interrupt(agent)
            await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "approve"))
            assert agent.runtime.execution_mode == expected
            assert agent.permission_engine.grants() == []
        finally:
            await agent.aclose()
    asyncio.run(run())


def test_custom_engine_without_approval_support_can_still_deny():
    class Engine(BasePermissionEngine):
        def authorize(self, request):
            return PermissionDecision(PermissionBehavior.ASK)

    async def run():
        agent = BasicAgent("ui", OfflineLLM()).add_tool(CountingWriteTool()).with_permissions(Engine())
        try:
            pending = await interrupt(agent)
            with pytest.raises(NotImplementedError):
                await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "approve"))
            assert agent.get_pending_interruption() == pending
            await agent.arespond_to_interruption(InteractionReply(pending["request_id"], "deny"))
            assert agent.get_pending_interruption() is None
        finally:
            await agent.aclose()
    asyncio.run(run())


def test_sync_tool_async_bridge_keeps_loop_responsive_and_stops_siblings():
    entered, release = Event(), Event()

    class SlowTool(CountingWriteTool):
        def run(self, arguments):
            entered.set()
            assert release.wait(3)
            return super().run(arguments)

    async def run():
        tool = SlowTool()
        agent = BasicAgent("ui", OfflineLLM()).add_tool(tool)

        async def consume():
            async for _ in agent.astream("run"):
                pass

        task = asyncio.create_task(consume())
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            agent.request_stop()
            assert not task.done()
            release.set()
            with pytest.raises(AgentStopRequested):
                await task
            assert not agent.runtime.invocation_lock.locked()
            assert len(tool_results(agent)) == 1
        finally:
            release.set()
            await agent.aclose()
    asyncio.run(run())


def test_closing_stream_after_tool_call_releases_runtime_and_pairs_history():
    async def run():
        tool = CountingWriteTool()
        agent = BasicAgent("ui", OfflineLLM()).add_tool(tool)
        try:
            async with aclosing(agent.astream("run")) as events:
                async for event in events:
                    if event.type == AgentStreamEventType.TOOL_CALL:
                        break
            assert not agent.runtime.invocation_lock.locked()
            assert tool.calls == 0
            assert len(tool_results(agent)) == 1
        finally:
            await agent.aclose()
    asyncio.run(run())
