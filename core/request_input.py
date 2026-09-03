from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .cache_policy import CacheableBlock, PromptCachePolicy, normalize_cache_policy, render_blocks
from .history import _json_safe
from .providers import create_codec


def _clone_payload(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            payload = value.to_dict()
            if isinstance(payload, dict):
                return _json_safe(payload)
        except Exception:
            pass
    if isinstance(value, dict):
        return _json_safe(value)
    return value


@dataclass
class ReplayRequestInput:
    """Current-provider request buffer.

    Stable system prompt stays out of replay history. Per-request system reminders and
    dynamic tail content are tracked separately, then materialized into the
    effective replay history view exposed through `replay_history`.
    """

    provider_name: Optional[str]
    replay_history: list[Any] = field(default_factory=list)
    persistent_replay_history: list[Any] = field(default_factory=list)
    prepended_replay_history: list[Any] = field(default_factory=list)
    appended_replay_history: list[Any] = field(default_factory=list)
    system_prompt: Optional[str] = None
    system_prompt_blocks: list[CacheableBlock] = field(default_factory=list)
    system_reminder_blocks: list[CacheableBlock] = field(default_factory=list)
    dynamic_context_blocks: list[CacheableBlock] = field(default_factory=list)
    dynamic_tail_blocks: list[CacheableBlock] = field(default_factory=list)
    on_demand_expansion_blocks: list[CacheableBlock] = field(default_factory=list)
    cache_policy: PromptCachePolicy = field(default_factory=PromptCachePolicy)
    cache_signature: Optional[dict[str, Any]] = None
    cache_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_policy = normalize_cache_policy(self.cache_policy)
        self.system_prompt_blocks = [
            item if isinstance(item, CacheableBlock) else CacheableBlock(**dict(item))
            for item in list(self.system_prompt_blocks or [])
        ]
        self.system_reminder_blocks = [
            item if isinstance(item, CacheableBlock) else CacheableBlock(**dict(item))
            for item in list(self.system_reminder_blocks or [])
        ]
        self.dynamic_context_blocks = [
            item if isinstance(item, CacheableBlock) else CacheableBlock(**dict(item))
            for item in list(self.dynamic_context_blocks or [])
        ]
        self.dynamic_tail_blocks = [
            item if isinstance(item, CacheableBlock) else CacheableBlock(**dict(item))
            for item in list(self.dynamic_tail_blocks or [])
        ]
        self.on_demand_expansion_blocks = [
            item if isinstance(item, CacheableBlock) else CacheableBlock(**dict(item))
            for item in list(self.on_demand_expansion_blocks or [])
        ]
        self._validate_prompt_placements()
        if not self.dynamic_tail_blocks and self.dynamic_context_blocks:
            self.dynamic_tail_blocks = [CacheableBlock(**block.to_dict()) for block in self.dynamic_context_blocks]
        if self.system_prompt is None and self.system_prompt_blocks:
            self.system_prompt = self.render_system_prompt()
        if not self.persistent_replay_history and self.replay_history:
            self.persistent_replay_history = [_clone_payload(item) for item in self.replay_history]
        self._rebuild_replay_history()

    def _validate_prompt_placements(self) -> None:
        invalid_system = [
            block.name for block in self.system_prompt_blocks
            if block.placement != "system"
        ]
        if invalid_system:
            names = ", ".join(invalid_system)
            raise ValueError(
                f"system_prompt_blocks must use placement='system': {names}"
            )

        invalid_reminders = [
            block.name for block in self.system_reminder_blocks
            if block.placement != "system_reminder"
        ]
        if invalid_reminders:
            names = ", ".join(invalid_reminders)
            raise ValueError(
                "system_reminder_blocks must use "
                f"placement='system_reminder': {names}"
            )

    def render_system_prompt(self) -> Optional[str]:
        if self.system_prompt_blocks:
            return render_blocks(self.system_prompt_blocks, include_dynamic=True)
        return self.system_prompt

    @staticmethod
    def _render_tagged_blocks(blocks: list[CacheableBlock], *, tag: str) -> Optional[str]:
        rendered: list[str] = []
        for block in blocks:
            content = block.render()
            if not content:
                continue
            rendered.append(
                f"<{tag} name=\"{block.name}\">\n{content}\n</{tag}>"
            )
        return "\n\n".join(rendered) or None

    def render_system_reminders(self) -> Optional[str]:
        return self._render_tagged_blocks(self.system_reminder_blocks, tag="system-reminder")

    def render_dynamic_tail(self) -> Optional[str]:
        return render_blocks(self.dynamic_tail_blocks, include_dynamic=True)

    def render_on_demand_expansions(self) -> Optional[str]:
        return self._render_tagged_blocks(self.on_demand_expansion_blocks, tag="on-demand-expansion")

    def render_dynamic_context(self) -> Optional[str]:
        return self.render_dynamic_tail()

    def as_legacy_messages(self) -> list[Any]:
        messages: list[Any] = []
        system_prompt = self.render_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.replay_history)
        return messages

    def __len__(self) -> int:
        return len(self.as_legacy_messages())

    def __bool__(self) -> bool:
        return bool(self.as_legacy_messages())

    def __iter__(self):
        return iter(self.as_legacy_messages())

    def __getitem__(self, index: int) -> Any:
        return self.as_legacy_messages()[index]

    def _rebuild_replay_history(self) -> None:
        self.replay_history = [
            *[_clone_payload(item) for item in self.prepended_replay_history],
            *[_clone_payload(item) for item in self.persistent_replay_history],
            *[_clone_payload(item) for item in self.appended_replay_history],
        ]

    def append_dynamic_context(self, text: str, *, name: str = "dynamic_context") -> None:
        content = str(text or "").strip()
        if not content:
            return
        self.dynamic_tail_blocks.append(
            CacheableBlock(
                name=name,
                content=content,
                partition="dynamic",
                cacheable=False,
                reason="runtime_context",
            )
        )
        self.append_dynamic_tail_text(content)

    def prepend_system_reminder_text(self, text: str) -> None:
        content = str(text or "").strip()
        if not content:
            return
        self.prepend_replay(create_codec(self.provider_name).query_to_replay(content))

    def append_system_reminder_block(self, block: CacheableBlock) -> None:
        resolved = block if isinstance(block, CacheableBlock) else CacheableBlock(**dict(block))
        if resolved.placement != "system_reminder":
            raise ValueError(
                "system reminder blocks must use placement='system_reminder'"
            )
        self.system_reminder_blocks.append(resolved)
        content = self._render_tagged_blocks([self.system_reminder_blocks[-1]], tag="system-reminder")
        if content:
            self.prepend_replay(create_codec(self.provider_name).query_to_replay(content))

    def append_on_demand_expansion_text(self, text: str) -> None:
        content = str(text or "").strip()
        if not content:
            return
        self.extend_replay(create_codec(self.provider_name).query_to_replay(content))

    def append_on_demand_expansion_block(self, block: CacheableBlock) -> None:
        self.on_demand_expansion_blocks.append(
            block if isinstance(block, CacheableBlock) else CacheableBlock(**dict(block))
        )
        content = self._render_tagged_blocks(
            [self.on_demand_expansion_blocks[-1]],
            tag="on-demand-expansion",
        )
        if content:
            self.append_on_demand_expansion_text(content)

    def append_dynamic_tail_text(self, text: str) -> None:
        content = str(text or "").strip()
        if not content:
            return
        self.extend_replay(create_codec(self.provider_name).query_to_replay(content))

    def append_dynamic_tail_block(self, block: CacheableBlock) -> None:
        self.dynamic_tail_blocks.append(
            block if isinstance(block, CacheableBlock) else CacheableBlock(**dict(block))
        )
        content = self.dynamic_tail_blocks[-1].render()
        if content:
            self.append_dynamic_tail_text(content)

    def prepend_replay(self, messages: Iterable[Any]) -> None:
        codec = create_codec(self.provider_name)
        for message in messages:
            entry_payload = _clone_payload(message)
            codec.append_replay_entry(self.prepended_replay_history, entry_payload)
        self._rebuild_replay_history()

    def append_replay(self, message: Any) -> None:
        self.extend_replay([message])

    def extend_replay(self, messages: Iterable[Any]) -> None:
        codec = create_codec(self.provider_name)
        for message in messages:
            entry_payload = _clone_payload(message)
            codec.append_replay_entry(self.appended_replay_history, entry_payload)
        self._rebuild_replay_history()

    def set_replay_history(self, replay_history: list[Any]) -> None:
        self.persistent_replay_history = [_clone_payload(item) for item in replay_history]
        self.prepended_replay_history = []
        self.appended_replay_history = []
        self._rebuild_replay_history()

    def set_persistent_replay_history(self, replay_history: list[Any]) -> None:
        self.persistent_replay_history = [_clone_payload(item) for item in replay_history]
        self._rebuild_replay_history()

    def apply_runtime_layers(self) -> None:
        self.prepended_replay_history = []
        reminders = self.render_system_reminders()
        if reminders:
            self.prepend_replay(create_codec(self.provider_name).query_to_replay(reminders))
        expansions = self.render_on_demand_expansions()
        if expansions:
            self.append_on_demand_expansion_text(expansions)
        dynamic_tail = self.render_dynamic_tail()
        if dynamic_tail:
            self.append_dynamic_tail_text(dynamic_tail)

    def clone(self) -> "ReplayRequestInput":
        return ReplayRequestInput(
            provider_name=self.provider_name,
            replay_history=[_clone_payload(item) for item in self.replay_history],
            persistent_replay_history=[_clone_payload(item) for item in self.persistent_replay_history],
            prepended_replay_history=[_clone_payload(item) for item in self.prepended_replay_history],
            appended_replay_history=[_clone_payload(item) for item in self.appended_replay_history],
            system_prompt=self.system_prompt,
            system_prompt_blocks=[
                CacheableBlock(**block.to_dict()) for block in self.system_prompt_blocks
            ],
            system_reminder_blocks=[
                CacheableBlock(**block.to_dict()) for block in self.system_reminder_blocks
            ],
            dynamic_context_blocks=[
                CacheableBlock(**block.to_dict()) for block in self.dynamic_tail_blocks
            ],
            dynamic_tail_blocks=[
                CacheableBlock(**block.to_dict()) for block in self.dynamic_tail_blocks
            ],
            on_demand_expansion_blocks=[
                CacheableBlock(**block.to_dict()) for block in self.on_demand_expansion_blocks
            ],
            cache_policy=PromptCachePolicy.from_value(self.cache_policy.to_dict()),
            cache_signature=_clone_payload(self.cache_signature) if self.cache_signature else None,
            cache_metadata=_clone_payload(self.cache_metadata),
        )
