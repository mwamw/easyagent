from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.cache_policy import CacheableBlock, PromptCachePolicy, normalize_cache_policy


def _anthropic_cache_control(policy: PromptCachePolicy) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "ephemeral"}
    if policy.ttl:
        payload["ttl"] = policy.ttl
    if policy.scope == "global":
        payload["scope"] = "global"
    return payload


def _anthropic_system_blocks(
    blocks: list[CacheableBlock],
    policy: PromptCachePolicy,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cacheable_indexes: list[int] = []
    for block in blocks:
        if block.partition == "dynamic":
            continue
        text = block.render()
        if not text:
            continue
        index = len(items)
        item: dict[str, Any] = {"type": "text", "text": text}
        items.append(item)
        if block.cacheable:
            cacheable_indexes.append(index)

    if policy.enabled and policy.breakpoint_strategy != "none" and cacheable_indexes:
        items[cacheable_indexes[-1]]["cache_control"] = _anthropic_cache_control(policy)
    return items


def _anthropic_cache_markers_enabled(policy: PromptCachePolicy) -> bool:
    return bool(policy.enabled and policy.breakpoint_strategy != "none")


def _anthropic_message_marker_index(messages: list[Any], policy: PromptCachePolicy) -> int | None:
    if not messages:
        return None
    if policy.mode in {"read_only", "skip_write"} and len(messages) >= 2:
        return len(messages) - 2
    return len(messages) - 1


def _anthropic_find_message_block_index(content: list[Any]) -> int | None:
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if not isinstance(block, dict):
            return index
        block_type = block.get("type")
        if block_type in {"thinking", "redacted_thinking"}:
            continue
        return index
    return None


def _anthropic_apply_message_cache_marker(
    messages: list[Any],
    policy: PromptCachePolicy,
) -> tuple[list[Any], int | None, bool]:
    if not _anthropic_cache_markers_enabled(policy):
        return list(messages), None, False

    updated = [dict(message) if isinstance(message, dict) else message for message in list(messages)]
    marker_index = _anthropic_message_marker_index(updated, policy)
    if marker_index is None or marker_index < 0 or marker_index >= len(updated):
        return updated, marker_index, False

    message = updated[marker_index]
    if not isinstance(message, dict):
        return updated, marker_index, False

    content = message.get("content")
    cache_control = _anthropic_cache_control(policy)
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return updated, marker_index, True
    if not isinstance(content, list):
        return updated, marker_index, False

    content_blocks = [dict(block) if isinstance(block, dict) else block for block in content]
    block_index = _anthropic_find_message_block_index(content_blocks)
    if block_index is None:
        return updated, marker_index, False
    block = content_blocks[block_index]
    if isinstance(block, dict):
        content_blocks[block_index] = {**block, "cache_control": cache_control}
    else:
        content_blocks[block_index] = {"type": "text", "text": str(block), "cache_control": cache_control}
    message["content"] = content_blocks
    return updated, marker_index, True


def _anthropic_apply_tool_cache_marker(
    tools: Any,
    policy: PromptCachePolicy,
) -> tuple[Any, bool]:
    if not _anthropic_cache_markers_enabled(policy):
        return tools, False
    if not isinstance(tools, list) or not tools:
        return tools, False
    updated: list[Any] = []
    last_index: int | None = None
    for index, tool in enumerate(tools):
        if isinstance(tool, dict):
            updated.append(dict(tool))
            last_index = index
        else:
            updated.append(tool)
    if last_index is None:
        return updated, False
    tool_payload = updated[last_index]
    if not isinstance(tool_payload, dict):
        return updated, False
    tool_payload["cache_control"] = _anthropic_cache_control(policy)
    return updated, True


class ProviderCacheAdapter:
    provider_name = "generic"
    capability: "CacheCapability"

    def __init__(self) -> None:
        self.capability = CacheCapability()

    def apply_cache_policy(self, request: Any, request_input: Any) -> Any:
        return request


@dataclass(frozen=True)
class CacheCapability:
    supports_explicit_cache_control: bool = False
    supports_message_level_breakpoint: bool = False
    supports_tool_cache_marker: bool = False
    supports_usage_cache_fields: bool = False
    supports_cached_content_objects: bool = False
    supports_deferred_tools: bool = True
    usage_semantics: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_explicit_cache_control": self.supports_explicit_cache_control,
            "supports_message_level_breakpoint": self.supports_message_level_breakpoint,
            "supports_tool_cache_marker": self.supports_tool_cache_marker,
            "supports_usage_cache_fields": self.supports_usage_cache_fields,
            "supports_cached_content_objects": self.supports_cached_content_objects,
            "supports_deferred_tools": self.supports_deferred_tools,
            "usage_semantics": self.usage_semantics,
        }


class AnthropicCacheAdapter(ProviderCacheAdapter):
    provider_name = "anthropic_native"

    def __init__(self) -> None:
        self.capability = CacheCapability(
            supports_explicit_cache_control=True,
            supports_message_level_breakpoint=True,
            supports_tool_cache_marker=True,
            supports_usage_cache_fields=True,
            supports_cached_content_objects=False,
            supports_deferred_tools=True,
            usage_semantics="anthropic_style",
        )

    def apply_cache_policy(self, request: Any, request_input: Any) -> Any:
        if not isinstance(request, dict):
            return request
        blocks = list(getattr(request_input, "system_prompt_blocks", None) or [])
        policy = normalize_cache_policy(getattr(request_input, "cache_policy", None))
        updated = dict(request)
        metadata = dict(getattr(request_input, "cache_metadata", None) or {})

        system_blocks = _anthropic_system_blocks(blocks, policy)
        if system_blocks:
            updated["system"] = system_blocks

        tools_payload, tool_cache_applied = _anthropic_apply_tool_cache_marker(
            updated.get("tools"),
            policy,
        )
        if tool_cache_applied:
            updated["tools"] = tools_payload

        messages_payload, marker_index, message_cache_applied = _anthropic_apply_message_cache_marker(
            list(updated.get("messages") or []),
            policy,
        )
        if message_cache_applied:
            updated["messages"] = messages_payload

        metadata["providerCacheAdapter"] = self.provider_name
        metadata["explicitSystemCacheApplied"] = any("cache_control" in item for item in system_blocks)
        metadata["explicitToolCacheApplied"] = tool_cache_applied
        metadata["explicitMessageCacheApplied"] = message_cache_applied
        metadata["messageCacheMarkerIndex"] = marker_index
        metadata["skipCacheWrite"] = policy.mode in {"read_only", "skip_write"}
        metadata["explicitCacheApplied"] = bool(
            metadata["explicitSystemCacheApplied"] or tool_cache_applied or message_cache_applied
        )
        request_input.cache_metadata = metadata
        return updated


class NoopCacheAdapter(ProviderCacheAdapter):
    def __init__(self, *, usage_semantics: str = "unknown", supports_usage_cache_fields: bool = False, supports_cached_content_objects: bool = False) -> None:
        self.capability = CacheCapability(
            supports_usage_cache_fields=supports_usage_cache_fields,
            supports_cached_content_objects=supports_cached_content_objects,
            supports_deferred_tools=True,
            usage_semantics=usage_semantics,
        )


class GoogleCacheAdapter(NoopCacheAdapter):
    provider_name = "google_native"

    def __init__(self) -> None:
        super().__init__(
            usage_semantics="google_style",
            supports_usage_cache_fields=True,
            supports_cached_content_objects=True,
        )

    def apply_cache_policy(self, request: Any, request_input: Any) -> Any:
        if not isinstance(request, dict):
            return request
        metadata = dict(getattr(request_input, "cache_metadata", None) or {})
        cached_content = metadata.get("googleCachedContent") or metadata.get("cachedContent")
        if not cached_content:
            return request
        updated = dict(request)
        config = dict(updated.get("config") or {})
        config["cached_content"] = str(cached_content)
        updated["config"] = config
        metadata["providerCacheAdapter"] = self.provider_name
        metadata["cachedContentApplied"] = True
        request_input.cache_metadata = metadata
        return updated


def create_cache_adapter(provider_name: str | None) -> ProviderCacheAdapter:
    normalized = (provider_name or "").lower()
    if normalized in {"anthropic_native", "claude_native"}:
        return AnthropicCacheAdapter()
    if normalized in {"openai", "openai_responses"}:
        return NoopCacheAdapter(usage_semantics="openai_style", supports_usage_cache_fields=True)
    if normalized in {"google_native", "gemini_native"}:
        return GoogleCacheAdapter()
    return NoopCacheAdapter()


__all__ = ["CacheCapability", "ProviderCacheAdapter", "create_cache_adapter"]
