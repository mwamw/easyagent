"""Regressions for API-first context sizing (no network or real model calls)."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from easyagent import BasicAgent
from easyagent.context import ContextBuilder, ContextManager, TokenBudget, BaseHistoryCompactor, ContextBudgetExceeded
from easyagent.hooks import BaseHook, HookManager
from easyagent.prompting import PromptBlock, SystemPromptComposer
from context.default.usage import ContextUsageTracker
from core.cache_policy import CacheableBlock
from core.history import CanonicalBlock, CanonicalMessage
from core.providers import AnthropicNativeProvider, create_codec
from test_session_persistence import SessionLLM, SessionProvider, EchoTool
from test_provider_usage_metrics import _openai_stream_chunk


class CharacterCounter:
    def count(self, value):
        return len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))


def context(limit=100000):
    return ContextManager(builder=ContextBuilder(budget=TokenBudget(limit), counter=CharacterCounter()))


def request(manager, history=None, **kwargs):
    return manager.build_request_input("", include_query=False, provider_name="openai",
        replay_history=history or [{"role": "user", "content": "question"}], system_prompt="system", **kwargs)


def observed(manager, *, usage=None, use_output=True, **kwargs):
    tracker = ContextUsageTracker()
    llm = SimpleNamespace(model="one", base_url="mock")
    initial = request(manager, **kwargs)
    response = [{"role": "assistant", "content": "answer"}]
    prepared = tracker.prepare(manager, initial, llm=llm, history_epoch=0)
    tracker.observe(manager, prepared, usage if usage is not None else {
        "inputTokens": 30, "outputTokens": 7, "totalTokens": 37, "usageSource": "provider",
    }, response_replay=response, use_output=use_output)
    after = deepcopy(initial)
    after.extend_replay(response)
    return tracker, llm, initial, after


def test_api_base_is_preferred_even_if_local_count_is_larger():
    manager = context(40)
    tracker, llm, _, after = observed(manager)
    assert manager.count_request(after) > 40
    estimate = manager.estimate_request(after, usage_tracker=tracker, llm=llm, history_epoch=0)
    assert estimate.tokens == 37
    assert estimate.source == "provider_usage_plus_delta_estimate"
    assert manager.validate_request(after, usage_tracker=tracker, llm=llm, history_epoch=0) == 37


def test_new_tool_results_and_expiring_reminders_use_signed_delta():
    manager = context()
    tracker, llm, _, after = observed(manager, extra_replay_entries=[{"role": "user", "content": "temporary " * 20}])
    baseline = manager.count_request(after)
    changed = deepcopy(after)
    changed.append_dynamic_tail_text("new result")
    estimate = tracker.estimate(manager, changed, llm=llm, history_epoch=0)
    assert estimate.tokens == 37 + manager.count_request(changed) - baseline
    # Removing a temporary layer is a delta, not an excuse to ignore valid usage.
    changed = request(manager)
    changed.extend_replay([{"role": "assistant", "content": "answer"}])
    estimate = tracker.estimate(manager, changed, llm=llm, history_epoch=0)
    assert estimate.source == "provider_usage_plus_delta_estimate"
    assert estimate.metadata["delta_tokens"] < 0


@pytest.mark.parametrize("change", ["model", "provider", "system", "reminder", "tools", "reasoning", "history", "counter"])
def test_incompatible_request_invalidates_usage_base(change):
    manager = context()
    tracker, llm, _, after = observed(manager)
    kwargs = dict(llm=llm, history_epoch=0)
    if change == "model":
        llm.model = "other"
    elif change == "provider":
        after.provider_name = "anthropic_native"
    elif change == "system":
        after.system_prompt = "different"
    elif change == "reminder":
        after.append_system_reminder_block(CacheableBlock("memory_policy", "new rules", placement="system_reminder"))
    elif change == "tools":
        kwargs["tools"] = [{"name": "new_tool"}]
    elif change == "reasoning":
        kwargs["reasoning"] = {"effort": "high"}
    elif change == "history":
        kwargs["history_epoch"] = 1
    elif change == "counter":
        manager = context()
    estimate = tracker.estimate(manager, after, **kwargs)
    assert estimate.source == "local_request_estimate"
    assert estimate.tokens == estimate.local_tokens


@pytest.mark.parametrize("usage", [{}, {"totalTokens": 3, "usageSource": "estimated"},
                                  {"inputTokens": -1, "usageSource": "provider"},
                                  {"inputTokens": 3.5, "usageSource": "provider"}])
def test_missing_or_invalid_usage_falls_back(usage):
    manager = context()
    tracker, llm, _, after = observed(manager, usage=usage)
    assert tracker.estimate(manager, after, llm=llm, history_epoch=0).source == "local_request_estimate"


def test_input_only_usage_and_nonreplayed_reasoning_do_not_invent_output_tokens():
    manager = context()
    for usage in ({"inputTokens": 30, "usageSource": "provider"},
                  {"inputTokens": 30, "outputTokens": 5000, "totalTokens": 5030, "reasoningTokens": 4990, "usageSource": "provider"}):
        tracker, llm, initial, after = observed(manager, usage=usage)
        estimate = tracker.estimate(manager, after, llm=llm, history_epoch=0)
        assert estimate.tokens == 30 + manager.count_request(after) - manager.count_request(initial)
        assert estimate.metadata["anchor_kind"] == "request_input"


def test_dynamic_reminder_changes_are_deltas_not_anchor_invalidations():
    manager = context()
    tracker, llm, _, after = observed(manager)
    baseline = manager.count_request(after)
    after.append_system_reminder_block(CacheableBlock("clock", "now", placement="system_reminder", partition="dynamic"))
    estimate = tracker.estimate(manager, after, llm=llm, history_epoch=0)
    assert estimate.source == "provider_usage_plus_delta_estimate"
    assert estimate.tokens == 37 + manager.count_request(after) - baseline


def test_input_base_can_be_derived_from_total_when_output_is_not_retained():
    manager = context()
    tracker, llm, initial, after = observed(manager, use_output=False, usage={
        "totalTokens": 5030, "outputTokens": 5000, "usageSource": "provider",
        "contextInputTokens": None, "contextTotalTokens": None,
    })
    estimate = tracker.estimate(manager, after, llm=llm, history_epoch=0)
    assert estimate.tokens == 30 + manager.count_request(after) - manager.count_request(initial)
    assert estimate.metadata["anchor_kind"] == "request_input"


class ScriptedProvider(SessionProvider):
    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.requests = []

    def invoke_raw(self, request):
        self.requests.append(deepcopy(request))
        reply = self.replies.pop(0)
        return SimpleNamespace(content=reply.get("content", "answer"), reasoning_content=reply.get("thinking"),
            tool_calls=reply.get("tool_calls", []), usage=reply.get("usage"))

    def stream_raw(self, request):
        result = self.invoke_raw(request)
        yield _openai_stream_chunk(content=result.content, finish_reason="stop")
        if result.usage is not None:
            yield _openai_stream_chunk(usage=result.usage)

    async def async_stream_raw(self, request):
        async def events():
            for item in self.stream_raw(request):
                await asyncio.sleep(0)
                yield item
        return events()


def api_usage(inputs=100, outputs=10):
    return {"prompt_tokens": inputs, "completion_tokens": outputs, "total_tokens": inputs + outputs}


def agent_with(replies, manager=None, llm=None):
    llm = llm or SessionLLM()
    llm._provider = ScriptedProvider(replies)
    agent = BasicAgent("usage", llm).with_prompt(SystemPromptComposer(
        [PromptBlock("identity", "system")], include_defaults=False,
    ))
    if manager is not None:
        agent.with_context(manager)
    return agent


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
def test_all_execution_paths_capture_usage_without_observability(mode):
    agent = agent_with([{"usage": api_usage()}])
    assert agent.observability is None
    if mode == "invoke":
        agent.invoke("question")
    elif mode == "ainvoke":
        asyncio.run(agent.ainvoke("question"))
    elif mode == "stream":
        list(agent.stream("question"))
    else:
        async def collect():
            return [event async for event in agent.astream("question")]
        asyncio.run(collect())
    usage = agent.get_context_usage()
    assert usage["estimatedRequestTokens"] == 110
    assert usage["request_estimate_source"] == "provider_usage_plus_delta_estimate"
    assert usage["requestEstimate"]["metadata"]["usage"]["inputTokens"] == 100


def test_history_restore_clear_and_model_change_do_not_reuse_stale_usage():
    for change in ("restore", "clear", "model"):
        agent = agent_with([{"usage": api_usage()}], context())
        agent.invoke("question")
        assert agent.get_context_usage()["request_estimate_source"].startswith("provider_usage")
        if change == "restore":
            agent.context_manager.restore_state(agent.context_manager.export_state())
        elif change == "clear":
            agent.clear_history()
        else:
            agent.change_model(SessionLLM())
        assert agent.get_context_usage()["request_estimate_source"] == "local_request_estimate"


def test_empty_tool_registry_matches_the_request_usage_signature():
    agent = agent_with([{"usage": api_usage()}], context()).with_tool()
    agent.invoke("question")
    assert agent.get_context_usage()["estimatedRequestTokens"] == 110
    assert agent.get_context_usage()["request_estimate_source"] == "provider_usage_plus_delta_estimate"


def test_shared_llm_has_agent_owned_context_and_usage_anchors():
    llm, manager = SessionLLM(), context()
    one = agent_with([{"usage": api_usage()}], manager, llm)
    one.invoke("question")
    with pytest.raises(RuntimeError, match="cannot be shared"):
        BasicAgent("invalid-sharing", llm).with_context(manager)
    two = agent_with([{"usage": api_usage(300)}], context(), llm)
    two.invoke("question")
    assert one.get_context_usage()["estimatedRequestTokens"] == 110
    assert two.get_context_usage()["estimatedRequestTokens"] == 310
    # A direct summarizer/model call must not replace either Agent's anchor.
    llm._provider.replies = [{"usage": api_usage(9000)}]
    llm.invoke_raw([{"role": "user", "content": "summary"}])
    assert one.get_context_usage()["estimatedRequestTokens"] == 110


@pytest.mark.parametrize("asynchronous", [False, True])
def test_api_usage_not_local_count_drives_compaction_and_rebases(asynchronous):
    manager = context(1800)
    captured = []
    class Compactor(BaseHistoryCompactor):
        recent_turns = 1
        def compact(self, history):
            return [CanonicalMessage(role="user", content=[CanonicalBlock(type="text", text="summary")])]
    class Observe(BaseHook):
        def before_compaction(self, payload):
            captured.append(payload)
    manager.set_history_compactor(Compactor())
    agent = agent_with([
        {"usage": api_usage(10000), "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "Echo", "arguments": '{"text":"ok"}'}}]},
        {"usage": api_usage(100)},
    ], manager)
    agent.add_tool(EchoTool())
    agent.with_hooks(HookManager([Observe()]))
    agent.add_user_message("old question")
    agent.add_assistant_message("old answer")
    result = asyncio.run(agent.ainvoke("new question")) if asynchronous else agent.invoke("new question")
    assert result == "answer"
    assert len(captured) == 1
    assert captured[0]["estimated_tokens"] > 10000
    assert captured[0]["token_estimate_source"] == "provider_usage_plus_delta_estimate"
    assert "summary" in repr(agent.history)
    assert agent.get_context_usage()["estimatedRequestTokens"] == 110


def test_final_validation_includes_hook_added_content_with_api_anchor():
    manager = context(1500)
    agent = agent_with([{"usage": api_usage()}], manager)
    agent.invoke("question")
    class Grow(BaseHook):
        def before_llm_request(self, payload):
            payload["request_input"].append_dynamic_tail_text("x" * 4000)
    agent.with_hooks(HookManager([Grow()]))
    with pytest.raises(ContextBudgetExceeded):
        agent.invoke("next")


def test_unpersisted_thinking_uses_actual_input_and_only_retained_output_delta():
    agent = agent_with([{"usage": api_usage(100, 5000), "thinking": "private reasoning"}], context())
    agent.config.persist_reasoning_history = False
    agent.invoke("question")
    metadata = agent.get_context_usage()["requestEstimate"]["metadata"]
    assert metadata["anchor_kind"] == "request_input"
    assert metadata["provider_usage_tokens"] == 100
    assert "private reasoning" not in repr(agent.history)


def test_modified_model_output_does_not_reuse_original_output_token_count():
    class Rewrite(BaseHook):
        def after_llm_response(self, payload):
            payload["response"].content = "rewritten"
    agent = agent_with([{"usage": api_usage(100, 5000)}], context())
    agent.with_hooks(HookManager([Rewrite()]))
    assert agent.invoke("question") == "rewritten"
    metadata = agent.get_context_usage()["requestEstimate"]["metadata"]
    assert metadata["anchor_kind"] == "request_input"
    assert metadata["provider_usage_tokens"] == 100


def test_anthropic_stream_retains_start_input_and_cache_counts():
    codec = create_codec("anthropic_native")
    events = list(codec.stream_events([
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "answer"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]))
    provider = AnthropicNativeProvider(model="test", api_key="test", base_url="http://mock.local", client=SimpleNamespace())
    usage = provider.get_usage_from_response(events[-1])
    assert usage["inputTokens"] == 10
    assert usage["contextInputTokens"] == 130
    assert usage["contextTotalTokens"] == 135
    assert usage["cacheReadTokens"] == 100


def test_openai_cached_input_is_not_counted_twice():
    from core.providers import OpenAIProvider
    provider = OpenAIProvider(model="test", api_key="test", base_url="http://mock.local", client=SimpleNamespace())
    usage = provider.get_usage_from_response({"usage": {"prompt_tokens": 100, "completion_tokens": 10,
        "total_tokens": 110, "prompt_tokens_details": {"cached_tokens": 90}}})
    assert usage["contextInputTokens"] == 100
    assert usage["contextTotalTokens"] == 110
    assert usage["cachedInputTokens"] == 90
