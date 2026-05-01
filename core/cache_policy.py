from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal, Optional


CacheMode = Literal["auto", "read_only", "write", "skip_write"]
CacheScope = Literal["session", "global"]
CachePartition = Literal["static", "session", "dynamic"]
BreakpointStrategy = Literal["stable_prefix", "last_cacheable_block", "none"]


@dataclass(slots=True)
class PromptCachePolicy:
    """Provider-neutral prompt cache policy.

    Provider adapters translate this intent into concrete request fields when
    the provider exposes explicit prompt-cache controls. Providers that do not
    expose such controls still use the policy for signature/debug metadata.
    """

    enabled: bool = True
    mode: CacheMode = "auto"
    ttl: Optional[str] = None
    scope: Optional[CacheScope] = "session"
    breakpoint_strategy: BreakpointStrategy = "stable_prefix"

    @classmethod
    def from_value(cls, value: Any) -> "PromptCachePolicy":
        if isinstance(value, PromptCachePolicy):
            return value
        if isinstance(value, dict):
            allowed = {"enabled", "mode", "ttl", "scope", "breakpoint_strategy"}
            return cls(**{key: value[key] for key in allowed if key in value})
        if value is None:
            return cls()
        raise TypeError(f"Unsupported prompt cache policy: {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "ttl": self.ttl,
            "scope": self.scope,
            "breakpoint_strategy": self.breakpoint_strategy,
        }


@dataclass(slots=True)
class CacheableBlock:
    name: str
    content: str
    partition: CachePartition = "static"
    cacheable: bool = True
    reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return str(self.content or "").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "partition": self.partition,
            "cacheable": self.cacheable,
            "reason": self.reason,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(slots=True)
class CacheSignature:
    provider: Optional[str]
    model: Optional[str]
    system_hash: str
    tools_hash: str
    reasoning_hash: str
    extra_hash: str
    cache_policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "system_hash": self.system_hash,
            "tools_hash": self.tools_hash,
            "reasoning_hash": self.reasoning_hash,
            "extra_hash": self.extra_hash,
            "cache_policy_hash": self.cache_policy_hash,
        }


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def normalize_cache_policy(value: Any) -> PromptCachePolicy:
    return PromptCachePolicy.from_value(value)


def prompt_block_to_cacheable(block: Any) -> CacheableBlock:
    metadata = dict(getattr(block, "metadata", None) or {})
    partition = str(metadata.get("cache_partition") or metadata.get("partition") or "static")
    if partition not in {"static", "session", "dynamic"}:
        partition = "dynamic"
    cacheable = bool(metadata.get("cacheable", partition != "dynamic"))
    return CacheableBlock(
        name=str(getattr(block, "name", "") or "prompt_block"),
        content=str(getattr(block, "render", lambda: getattr(block, "content", ""))() or ""),
        partition=partition,  # type: ignore[arg-type]
        cacheable=cacheable,
        reason=metadata.get("cache_reason"),
        metadata=metadata,
    )


def render_blocks(blocks: list[CacheableBlock], *, include_dynamic: bool = False) -> Optional[str]:
    selected = []
    for block in blocks:
        if not include_dynamic and block.partition == "dynamic":
            continue
        rendered = block.render()
        if rendered:
            selected.append(rendered)
    return "\n\n".join(selected) or None


def build_cache_signature(
    *,
    provider: Optional[str],
    model: Optional[str],
    system_blocks: list[CacheableBlock],
    tools: Any = None,
    reasoning: Any = None,
    extra: Any = None,
    cache_policy: PromptCachePolicy | dict[str, Any] | None = None,
) -> CacheSignature:
    policy = normalize_cache_policy(cache_policy)
    cacheable_system = [
        block.to_dict()
        for block in system_blocks
        if block.partition != "dynamic" and block.cacheable
    ]
    return CacheSignature(
        provider=provider,
        model=model,
        system_hash=stable_hash(cacheable_system),
        tools_hash=stable_hash(tools),
        reasoning_hash=stable_hash(reasoning),
        extra_hash=stable_hash(extra),
        cache_policy_hash=stable_hash(policy.to_dict()),
    )


__all__ = [
    "PromptCachePolicy",
    "CacheableBlock",
    "CacheSignature",
    "build_cache_signature",
    "normalize_cache_policy",
    "prompt_block_to_cacheable",
    "render_blocks",
    "stable_hash",
    "stable_json",
]
