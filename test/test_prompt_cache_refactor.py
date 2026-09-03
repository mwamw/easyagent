from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent.BasicAgent import BasicAgent
from agent.components.prompt_composer import SystemPromptComposer
from core.Config import Config
from core.cache_policy import CacheableBlock, PromptCachePolicy
from core.llm import EasyLLM
from core.providers.cache_adapter import create_cache_adapter
from observability import InMemoryObservabilityStore
from core.request_compiler import compile_prompt_blocks
from core.request_input import ReplayRequestInput
from prompt import PromptBlock
from runtime import RuntimeEventType
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


def test_prompt_compiler_routes_blocks_only_by_placement():
    compiled = compile_prompt_blocks(
        [
            PromptBlock("identity", "stable identity", metadata={"cache_partition": "static"}),
            PromptBlock("memory", "volatile memory", metadata={"cache_partition": "dynamic"}),
            PromptBlock(
                "tool_inventory",
                "tool listing",
                placement="system_reminder",
                metadata={"cache_partition": "session"},
            ),
        ]
    )

    assert compiled.system_prompt == "stable identity\n\nvolatile memory"
    assert [block.name for block in compiled.system_reminder_blocks] == ["tool_inventory"]
    assert compiled.dynamic_context_blocks == []


def test_prompt_placement_is_preserved_in_cacheable_blocks():
    compiled = compile_prompt_blocks(
        [PromptBlock("policy", "policy", placement="system_reminder")]
    )

    assert compiled.system_reminder_blocks[0].placement == "system_reminder"
    assert compiled.system_reminder_blocks[0].to_dict()["placement"] == "system_reminder"


def test_replay_request_input_renders_dynamic_context_as_replay_delta():
    request = ReplayRequestInput(
        provider_name="openai",
        system_prompt_blocks=[CacheableBlock("identity", "stable identity")],
        dynamic_context_blocks=[CacheableBlock("memory", "volatile memory", partition="dynamic", cacheable=False)],
    )

    assert request.render_system_prompt() == "stable identity"
    assert request.render_dynamic_context() == "volatile memory"


def test_replay_request_input_prepends_system_reminders_before_history():
    request = ReplayRequestInput(
        provider_name="openai",
        replay_history=[{"role": "user", "content": "persisted history"}],
        system_reminder_blocks=[
            CacheableBlock(
                "tool_inventory",
                "tool listing",
                placement="system_reminder",
                partition="session",
                cacheable=True,
            )
        ],
        dynamic_context_blocks=[CacheableBlock("memory", "volatile memory", partition="dynamic", cacheable=False)],
    )

    request.apply_runtime_layers()

    assert "<system-reminder" in request.replay_history[0]["content"]
    assert request.replay_history[1]["content"] == "persisted history"
    assert request.replay_history[-1]["content"] == "volatile memory"


def test_replay_request_input_rejects_mismatched_prompt_placement():
    with pytest.raises(ValueError, match="placement='system_reminder'"):
        ReplayRequestInput(
            provider_name="openai",
            system_reminder_blocks=[CacheableBlock("policy", "content")],
        )


def test_custom_prompt_block_enters_reminder_layer_not_system_prefix():
    agent = BasicAgent(
        name="reminder-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
    ).with_prompt(
        SystemPromptComposer(
            [
                PromptBlock(
                    name="repo_policy",
                    content="Always explain cache breaks before proposing a fix.",
                    placement="system_reminder",
                )
            ]
        )
    )

    compiled = compile_prompt_blocks(agent.get_system_prompt_blocks())

    assert "Always explain cache breaks before proposing a fix." not in (compiled.system_prompt or "")
    assert [block.name for block in compiled.system_reminder_blocks if block.name == "repo_policy"] == ["repo_policy"]


def test_system_reminder_renders_once_per_request():
    agent = BasicAgent(
        name="reminder-render-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
    ).with_prompt(
        SystemPromptComposer(
            [
                PromptBlock(
                    name="product_context",
                    content="You are running inside a product shell with slash commands.",
                    placement="system_reminder",
                )
            ]
        )
    )

    compiled = compile_prompt_blocks(agent.get_system_prompt_blocks())
    request = ReplayRequestInput(
        provider_name="openai",
        replay_history=[{"role": "user", "content": "hello"}],
        system_prompt=compiled.system_prompt,
        system_prompt_blocks=compiled.system_prompt_blocks,
        system_reminder_blocks=compiled.system_reminder_blocks,
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


def test_agent_request_token_estimate_respects_deferred_tool_schema_mode():
    registry = ToolRegistry()
    register_tool_schema_tool(registry)
    registry.register_tool(DummyTool("resident_a"))
    registry.register_tool(DummyTool("resident_b"))
    agent = BasicAgent(
        name="deferred-estimate-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
        config=Config(tool_schema_mode="deferred"),
    ).with_tool(registry)
    agent.add_user_message("hello")
    deferred_tools = agent.get_provider_tools()
    estimate = agent.get_context_usage()["estimatedRequestTokens"]

    assert [item["function"]["name"] for item in deferred_tools] == ["tool_schema_tool"]
    assert estimate > 0


def test_anthropic_cache_adapter_applies_system_tool_and_message_cache_markers():
    registry = ToolRegistry()
    registry.register_tool(DummyTool("alpha"))
    registry.register_tool(DummyTool("beta"))
    request_input = ReplayRequestInput(
        provider_name="anthropic_native",
        system_prompt_blocks=[
            CacheableBlock("identity", "identity", partition="static", cacheable=True),
            CacheableBlock("custom", "custom", partition="session", cacheable=True),
            CacheableBlock("dynamic", "dynamic system", partition="dynamic", cacheable=False),
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
    assert updated["system"][2] == {"type": "text", "text": "dynamic system"}
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


def test_history_compaction_event_records_cache_break():
    agent = BasicAgent(
        name="compact-cache-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
    ).with_observability(store=InMemoryObservabilityStore())
    invoke_id = agent.observability.begin_agent_invoke(
        query="compact history",
        mode="plain",
        stream=False,
    )
    agent.event_bus.publish(
        RuntimeEventType.HISTORY_COMPACTED,
        agent_id=agent.name,
        invocation_id="manual-compaction",
        data={"tokens_before": 100, "tokens_after": 20},
    )

    agent.observability.end_agent_invoke(invoke_id, output=[], success=True)
    cache_break = agent.observability.latest().metadata["cache_breaks"][0]
    assert cache_break["reason"] == "history_compacted"
    assert "history.compacted" in cache_break["changed_fields"]


def test_llm_invoke_preserves_anthropic_cache_usage():
    agent = BasicAgent(
        name="ratio-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
    ).with_observability(store=InMemoryObservabilityStore())
    agent_id = agent.observability.begin_agent_invoke(query="ratio", mode="plain", stream=False)
    llm_id = agent.observability.begin_llm_invoke(
        input_messages=[],
        tools=[],
        options={"provider": "anthropic_native", "model": "claude"},
        estimated_input_tokens=10,
    )
    agent.observability.end_llm_invoke(
        llm_id,
        output=[],
        usage={
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
            "cacheReadTokens": 90,
            "cachedInputTokens": 90,
        },
        success=True,
    )
    agent.observability.end_agent_invoke(agent_id, output=[], success=True)

    stats = agent.observability.latest().llm_invokes[0].stats
    assert stats.input_tokens == 10
    assert stats.cache_read_tokens == 90
    assert stats.cached_input_tokens == 90


def test_llm_invoke_preserves_total_input_cache_usage():
    agent = BasicAgent(
        name="compat-ratio-agent",
        llm=EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="m1"),
    ).with_observability(store=InMemoryObservabilityStore())
    agent_id = agent.observability.begin_agent_invoke(query="ratio", mode="plain", stream=False)
    llm_id = agent.observability.begin_llm_invoke(
        input_messages=[],
        tools=[],
        options={"provider": "anthropic_native", "model": "claude-compatible"},
        estimated_input_tokens=100,
    )
    agent.observability.end_llm_invoke(
        llm_id,
        output=[],
        usage={
            "inputTokens": 100,
            "outputTokens": 5,
            "totalTokens": 105,
            "cacheReadTokens": 98,
            "cachedInputTokens": 98,
        },
        success=True,
    )
    agent.observability.end_agent_invoke(agent_id, output=[], success=True)

    stats = agent.observability.latest().llm_invokes[0].stats
    assert stats.input_tokens == 100
    assert stats.total_tokens == 105
    assert stats.cached_input_tokens == 98
