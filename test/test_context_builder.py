from copy import deepcopy

import pytest

from easyagent.context import ContextBuilder, ContextItem, ContextManager, ContextBudgetExceeded, TokenBudget, TokenCounter


def test_supplied_items_keep_order_and_are_not_mutated():
    items = [ContextItem("first", "memory", priority=.1), ContextItem("second", "docs", priority=.9)]
    original = deepcopy(items)
    result = ContextBuilder(TokenBudget(100)).build(items)
    assert [item.content for item in result.items] == ["first", "second"]
    assert items == original


def test_only_total_budget_no_source_registry():
    builder = ContextBuilder(TokenBudget(1), TokenCounter(chars_per_token=1))
    assert not hasattr(builder, "add_source")
    assert not hasattr(builder.budget, "allocations")
    with pytest.raises(ContextBudgetExceeded):
        builder.build([ContextItem("too much input", "anything")])


def test_unbounded_default_and_explicit_empty_budget():
    assert ContextBuilder().budget.max_tokens is None
    assert ContextBuilder(TokenBudget(0)).build([]).total_tokens == 0
    with pytest.raises(ContextBudgetExceeded):
        ContextBuilder(TokenBudget(0)).build([ContextItem("hello", "test")])


def test_formatted_extras_and_tools_count_in_final_request():
    manager = ContextManager(max_tokens=10)
    request = manager.build_request_input(query="question", provider_name="openai", context_items=[ContextItem("extra " * 50, "docs")])
    assert "extra" in repr(request.replay_history)
    with pytest.raises(ContextBudgetExceeded):
        manager.validate_request(request)


def test_provider_history_is_copied_without_coercion():
    history = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "text", "signature": "opaque"}]}]
    request = ContextBuilder().build_request_input(provider_name="anthropic", replay_history=history, include_query=False)
    assert request.replay_history == history
    request.replay_history[0]["content"][0]["signature"] = "changed"
    assert history[0]["content"][0]["signature"] == "opaque"
