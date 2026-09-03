from __future__ import annotations

from context.builder import ContextBuilder
from context.compressor.sliding_window import SlidingWindowCompressor
from context.compressor.token_budget import TokenBudgetCompressor
from context.formatter.markdown import MarkdownFormatter
from context.formatter.xml import XMLFormatter
from context.manager import ContextManager
from context.source.base import BaseContextSource
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
from context.window import ContextItem
from core.cache_policy import CacheableBlock
from core.request_input import ReplayRequestInput


class StubSource(BaseContextSource):
    def __init__(self, name: str, values: list[dict]):
        self._name = name
        self._values = values

    @property
    def source_name(self) -> str:
        return self._name

    def fetch(self, query: str, max_tokens: int = 0, **kwargs):
        return [
            ContextItem(
                content=item["content"],
                source=self._name,
                priority=item.get("priority", 0.5),
                token_count=item.get("token_count", 0),
            )
            for item in self._values
        ]


def test_builder_collects_weighted_sources_and_respects_budget():
    builder = ContextBuilder(
        budget=TokenBudget(max_tokens=30),
        counter=TokenCounter(chars_per_token=1.0),
    )
    builder.add_source(
        StubSource("high", [{"content": "high", "priority": 0.5, "token_count": 10}]),
        weight=2.0,
    )
    builder.add_source(
        StubSource("low", [{"content": "low", "priority": 0.5, "token_count": 10}]),
        weight=0.25,
    )

    window = builder.build("query")

    assert len(window) == 2
    assert window.items[0].source == "high"
    assert sum(item.token_count for item in window.items) <= 30


def test_builder_supports_source_and_global_compressors():
    values = [
        {"content": f"item-{index}", "priority": index / 10, "token_count": 10}
        for index in range(10)
    ]
    builder = ContextBuilder()
    builder.add_source(
        StubSource("records", values),
        compressor=SlidingWindowCompressor(max_items=5),
    )
    builder.set_compressor(TokenBudgetCompressor(max_tokens=20))

    window = builder.build("query")

    assert len(window) <= 2
    assert sum(item.token_count for item in window.items) <= 20


def test_builder_formats_external_context_without_history_ownership():
    builder = ContextBuilder().add_source(
        StubSource("rag", [{"content": "retrieved fact", "token_count": 5}])
    )
    builder.set_formatter(XMLFormatter())

    assert "<rag>" in builder.build_text("query")


def test_build_request_input_preserves_replay_and_applies_runtime_layers():
    builder = ContextBuilder().add_source(
        StubSource("rag", [{"content": "retrieved fact", "token_count": 5}])
    )
    request = builder.build_request_input(
        query="current question",
        provider_name="openai",
        system_prompt_blocks=[CacheableBlock("identity", "system", placement="system")],
        system_reminder_blocks=[
            CacheableBlock("runtime", "remember this", placement="system_reminder")
        ],
        replay_history=[{"role": "assistant", "content": "previous answer"}],
    )

    assert isinstance(request, ReplayRequestInput)
    assert request.render_system_prompt() == "system"
    assert request.persistent_replay_history == [
        {"role": "assistant", "content": "previous answer"}
    ]
    rendered = "\n".join(str(item.get("content") or "") for item in request.replay_history)
    assert "remember this" in rendered
    assert "retrieved fact" in rendered
    assert request.replay_history[-1] == {"role": "user", "content": "current question"}


def test_context_manager_delegates_request_building_and_chain_configuration():
    manager = ContextManager(max_tokens=200)
    source = StubSource("docs", [{"content": "module docs", "token_count": 5}])

    returned = manager.add_source(source).set_formatter(MarkdownFormatter())
    request = manager.build_request_input(
        "explain",
        provider_name="openai",
        system_prompt="assistant",
    )

    assert returned is manager
    assert request.system_prompt == "assistant"
    assert "module docs" in str(request.replay_history)
    assert manager.builder.source_names == ["docs"]


def test_memory_source_registration_replaces_existing_source():
    builder = ContextBuilder()
    first = StubSource("memory", [{"content": "old", "token_count": 5}])
    second = StubSource("memory", [{"content": "new", "token_count": 5}])

    builder.add_source(first).add_source(second)

    assert builder.source_names == ["memory"]
    assert "new" in builder.build_text("query")
    assert "old" not in builder.build_text("query")
