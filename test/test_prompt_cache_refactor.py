from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from agent.BasicAgent import BasicAgent
from core.cache_policy import CacheableBlock, PromptCachePolicy
from core.llm import EasyLLM
from core.providers.cache_adapter import create_cache_adapter
from core.runtime_reminders import BaseRuntimeReminderSource, RuntimeReminder
from observability.recorder import InMemoryObservabilityRecorder
from core.request_compiler import compile_prompt_blocks
from core.request_input import ReplayRequestInput
from prompt import PromptBlock
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.calculator import register_calculator_tool
from Tool.builtin.tool_schema_tool import register_tool_schema_tool


class EmptyParams(BaseModel):
    pass


class DummyTool(Tool):
    def __init__(self, name: str, *, source: str = "custom"):
        super().__init__(name=name, description=f"{name} tool", parameters=EmptyParams, source=source)

    def run(self, parameters: dict):
        return "ok"


class RepoPolicyReminder(BaseRuntimeReminderSource):
    def build_runtime_reminders(self, agent):
        return [
            RuntimeReminder(
                name="repo_policy",
                content="Always explain cache breaks before proposing a fix.",
            )
        ]


def test_prompt_compiler_moves_dynamic_blocks_out_of_system_prefix():
    compiled = compile_prompt_blocks(
        [
            PromptBlock("identity", "stable identity", metadata={"cache_partition": "static"}),
            PromptBlock("memory", "volatile memory", metadata={"cache_partition": "dynamic"}),
            PromptBlock("mailbox", "volatile mailbox", metadata={"cache_partition": "dynamic"}),
            PromptBlock("tool_inventory", "tool listing", metadata={"cache_partition": "session", "request_layer": "reminder"}),
        ]
    )

    assert compiled.system_prompt == "stable identity"
    assert [block.name for block in compiled.runtime_reminder_blocks] == ["tool_inventory"]
    assert [block.name for block in compiled.dynamic_context_blocks] == ["memory", "mailbox"]


def test_turn_skill_block_does_not_enter_cacheable_system_prefix():
    compiled = compile_prompt_blocks(
        [
            PromptBlock("identity", "stable identity", metadata={"cache_partition": "static"}),
            PromptBlock(
                "skills",
                "turn skill body",
                metadata={"cache_partition": "session", "skill_lifecycle": "turn"},
            ),
        ],
        cache_turn_skills=False,
    )

    assert compiled.system_prompt == "stable identity"
    assert [block.name for block in compiled.dynamic_context_blocks] == ["skills"]


def test_replay_request_input_renders_dynamic_context_as_replay_delta():
    request = ReplayRequestInput(
        provider_name="openai",
        system_prompt_blocks=[CacheableBlock("identity", "stable identity")],
        dynamic_context_blocks=[CacheableBlock("memory", "volatile memory", partition="dynamic", cacheable=False)],
    )

    assert request.render_system_prompt() == "stable identity"
    assert request.render_dynamic_context() == "volatile memory"


def test_replay_request_input_prepends_runtime_reminders_before_history():
    request = ReplayRequestInput(
        provider_name="openai",
        replay_history=[{"role": "user", "content": "persisted history"}],
        runtime_reminder_blocks=[CacheableBlock("tool_inventory", "tool listing", partition="session", cacheable=True)],
        dynamic_context_blocks=[CacheableBlock("memory", "volatile memory", partition="dynamic", cacheable=False)],
    )

    request.apply_runtime_layers()

    assert "<system-reminder" in request.replay_history[0]["content"]
    assert request.replay_history[1]["content"] == "persisted history"
    assert request.replay_history[-1]["content"] == "volatile memory"


def test_custom_runtime_reminder_source_enters_reminder_layer_not_system_prefix():
    agent = BasicAgent(
        name="reminder-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
        enable_tool=False,
    )
    agent.add_runtime_reminder_source(RepoPolicyReminder())

    compiled = compile_prompt_blocks(agent.get_system_prompt_blocks())

    assert "Always explain cache breaks before proposing a fix." not in (compiled.system_prompt or "")
    assert [block.name for block in compiled.runtime_reminder_blocks if block.name == "repo_policy"] == ["repo_policy"]


def test_runtime_reminder_renders_as_system_reminder_once_per_request():
    agent = BasicAgent(
        name="reminder-render-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
        enable_tool=False,
    ).with_runtime_reminder(
        name="product_context",
        content="You are running inside a product shell with slash commands.",
    )

    compiled = compile_prompt_blocks(agent.get_system_prompt_blocks())
    request = ReplayRequestInput(
        provider_name="openai",
        replay_history=[{"role": "user", "content": "hello"}],
        system_prompt=compiled.system_prompt,
        system_prompt_blocks=compiled.system_prompt_blocks,
        runtime_reminder_blocks=compiled.runtime_reminder_blocks,
    )
    request.apply_runtime_layers()

    reminder_messages = [
        item for item in request.replay_history
        if isinstance(item, dict) and "<system-reminder" in str(item.get("content", ""))
    ]
    assert len(reminder_messages) == 1
    assert "slash commands" in reminder_messages[0]["content"]


def test_tool_export_uses_stable_visibility_order():
    registry = ToolRegistry()
    registry.mount_turn_tool(DummyTool("z_turn"))
    registry.register_tool(DummyTool("b_builtin", source="builtin"))
    registry.mount_runtime_tool(DummyTool("a_runtime"))
    registry.register_tool(DummyTool("a_custom"))

    names = [item["function"]["name"] for item in registry.export_tools("openai")]

    assert names == ["b_builtin", "a_custom", "a_runtime", "z_turn"]


def test_tool_export_order_stable_after_runtime_tool_added():
    registry = ToolRegistry()
    registry.register_tool(DummyTool("resident_b"))
    registry.register_tool(DummyTool("resident_a"))
    before = [item["function"]["name"] for item in registry.export_tools("openai")]

    registry.mount_runtime_tool(DummyTool("runtime_a"))
    after = [item["function"]["name"] for item in registry.export_tools("openai")]

    assert before == ["resident_a", "resident_b"]
    assert after[:2] == before


def test_deferred_tool_export_only_exposes_loader_and_expanded_tools():
    registry = ToolRegistry()
    register_tool_schema_tool(registry)
    registry.register_tool(DummyTool("resident_a"))
    registry.register_tool(DummyTool("resident_b"))

    initial = [item["function"]["name"] for item in registry.export_tools("openai", mode="deferred")]
    registry.expand_deferred_tools(["resident_b"])
    expanded = [item["function"]["name"] for item in registry.export_tools("openai", mode="deferred")]

    assert initial == ["tool_schema_tool"]
    assert expanded == ["tool_schema_tool", "resident_b"]


def test_builtin_tools_are_exposed_in_deferred_mode_by_default():
    registry = ToolRegistry()
    register_calculator_tool(registry)

    names = [item["function"]["name"] for item in registry.export_tools("openai", mode="deferred")]

    assert names == ["calculator"]


def test_builtin_registration_can_override_deferred_exposure():
    registry = ToolRegistry()
    register_calculator_tool(registry, expose_in_deferred=False)

    names = [item["function"]["name"] for item in registry.export_tools("openai", mode="deferred")]

    assert names == []


def test_anthropic_cache_adapter_applies_system_tool_and_message_cache_markers():
    registry = ToolRegistry()
    registry.register_tool(DummyTool("alpha"))
    registry.register_tool(DummyTool("beta"))
    request_input = ReplayRequestInput(
        provider_name="anthropic_native",
        system_prompt_blocks=[
            CacheableBlock("identity", "identity", partition="static", cacheable=True),
            CacheableBlock("custom", "custom", partition="session", cacheable=True),
        ],
        replay_history=[
            {"role": "user", "content": "history"},
            {"role": "user", "content": "latest query"},
        ],
        cache_policy=PromptCachePolicy(enabled=True, ttl="1h"),
    )
    request = {
        "model": "test",
        "messages": list(request_input.replay_history),
        "tools": registry.export_tools("anthropic_native"),
        "system": request_input.render_system_prompt(),
    }

    updated = create_cache_adapter("anthropic_native").apply_cache_policy(request, request_input)

    assert isinstance(updated["system"], list)
    assert "cache_control" not in updated["system"][0]
    assert updated["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert updated["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert updated["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert request_input.cache_metadata["explicitSystemCacheApplied"] is True
    assert request_input.cache_metadata["explicitToolCacheApplied"] is True
    assert request_input.cache_metadata["explicitMessageCacheApplied"] is True


def test_anthropic_cache_adapter_shifts_message_marker_for_skip_write():
    request_input = ReplayRequestInput(
        provider_name="anthropic_native",
        replay_history=[
            {"role": "user", "content": "shared prefix"},
            {"role": "user", "content": "fork-only tail"},
        ],
        cache_policy=PromptCachePolicy(enabled=True, mode="skip_write"),
    )
    request = {"model": "test", "messages": list(request_input.replay_history)}

    updated = create_cache_adapter("anthropic_native").apply_cache_policy(request, request_input)

    assert updated["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert updated["messages"][1]["content"] == "fork-only tail"
    assert request_input.cache_metadata["messageCacheMarkerIndex"] == 0


def test_google_cache_adapter_applies_cached_content_name_from_request_metadata():
    request_input = ReplayRequestInput(
        provider_name="google_native",
        cache_metadata={"googleCachedContent": "cachedContents/demo"},
    )
    request = {"model": "gemini", "contents": [], "config": {"temperature": 0.1}}

    updated = create_cache_adapter("google_native").apply_cache_policy(request, request_input)

    assert updated["config"]["cached_content"] == "cachedContents/demo"
    assert request_input.cache_metadata["cachedContentApplied"] is True


def test_cache_signature_change_records_cache_break_for_reasoning_change():
    agent = BasicAgent(
        name="cache-break-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
        enable_tool=False,
        reasoning={"effort": "low"},
    )
    first = agent._cache_signature_for_messages(reasoning={"effort": "low"})
    second = agent._cache_signature_for_messages(reasoning={"effort": "high"})

    agent._maybe_record_cache_signature_change(first)
    agent._maybe_record_cache_signature_change(second)

    summary = agent.observability_recorder.get_summary()
    assert summary["cacheBreaks"] == 1
    assert summary["lastCacheBreak"]["reason"] == "cache_signature_changed"
    assert "reasoning_hash" in summary["lastCacheBreak"]["changedFields"]


def test_cache_read_drop_records_cache_break():
    agent = BasicAgent(
        name="cache-drop-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
        enable_tool=False,
    )
    agent._last_cache_signature = {"model": "m1"}
    agent._last_cache_usage = {"cacheReadTokensForBreakDetection": 5000}

    agent._maybe_record_cache_read_drop({"cacheReadTokens": 1000, "usageSource": "provider"})

    summary = agent.observability_recorder.get_summary()
    assert summary["cacheBreaks"] == 1
    assert summary["lastCacheBreak"]["reason"] == "cache_read_drop"
    assert summary["lastCacheBreak"]["previousCacheReadTokens"] == 5000
    assert summary["lastCacheBreak"]["currentCacheReadTokens"] == 1000


def test_history_compaction_records_cache_break_and_invalidates_signature():
    agent = BasicAgent(
        name="compact-cache-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
        enable_tool=False,
    )
    agent._last_cache_signature = {"model": "m1", "system_hash": "old"}
    agent.replay_history = [{"role": "user", "content": "old"}]
    result = SimpleNamespace(
        was_compacted=True,
        compaction_possible=True,
        tokens_before=100,
        tokens_after=20,
        budget=50,
        metadata={},
        canonical_history=[],
        replay_history=[{"role": "user", "content": "summary"}],
    )

    assert agent._apply_history_compaction_result(result) is True

    summary = agent.observability_recorder.get_summary()
    assert agent._last_cache_signature is None
    assert summary["lastCacheBreak"]["reason"] == "history_compacted"
    assert "history.compacted" in summary["lastCacheBreak"]["changedFields"]


def test_summary_normalizes_anthropic_cache_hit_ratio():
    recorder = InMemoryObservabilityRecorder(agent_name="ratio-agent")
    event_id = recorder.begin_llm_request(
        turn_id="t1",
        request_kind="invoke",
        stream=False,
        tools_enabled=False,
        provider_name="anthropic_native",
        model="claude",
        input_tokens=10,
        metadata={"cacheUsageSemantics": "anthropic_style"},
    )
    recorder.end_llm_request(
        event_id,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cache_read_tokens=90,
        cached_input_tokens=90,
        usage_source="provider",
        success=True,
        metadata={"cacheUsageSemantics": "anthropic_style"},
    )

    summary = recorder.get_summary()
    assert summary["promptTokensTotal"] == 100
    assert summary["promptTokensUncached"] == 10
    assert summary["promptTokensCached"] == 90
    assert summary["cacheHitTokenRatio"] == 0.9
