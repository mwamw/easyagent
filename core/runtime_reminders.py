"""Public runtime reminder primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional

from prompt import PromptBlock

if TYPE_CHECKING:
    from core.agent import BaseAgent


@dataclass(slots=True)
class RuntimeReminder:
    """Single request-time reminder block.

    Runtime reminders are injected once per request as `<system-reminder>`
    messages. They are not persisted into canonical history.
    """

    name: str
    content: str
    order: int = 0
    stable: bool = True
    cacheable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return str(self.content or "").strip()

    def to_prompt_block(self, order: int) -> PromptBlock:
        metadata = dict(self.metadata or {})
        metadata.setdefault("request_layer", "reminder")
        metadata.setdefault("cache_partition", "session" if self.stable else "dynamic")
        metadata.setdefault("cacheable", bool(self.cacheable and self.stable))
        return PromptBlock(
            name=self.name,
            content=self.render(),
            order=self.order or order,
            metadata=metadata,
        )


class BaseRuntimeReminderSource:
    """Extensible source of request-time reminders."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def build_runtime_reminders(
        self,
        agent: "BaseAgent",
    ) -> Iterable[RuntimeReminder | PromptBlock]:
        return []


class StaticRuntimeReminderSource(BaseRuntimeReminderSource):
    """Always inject the same runtime reminder text."""

    def __init__(
        self,
        *,
        name: str,
        content: str,
        stable: bool = True,
        cacheable: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._reminder = RuntimeReminder(
            name=name,
            content=content,
            stable=stable,
            cacheable=cacheable,
            metadata=dict(metadata or {}),
        )

    @property
    def name(self) -> str:
        return self._reminder.name

    def build_runtime_reminders(
        self,
        agent: "BaseAgent",
    ) -> Iterable[RuntimeReminder | PromptBlock]:
        return [self._reminder]


def reminder_to_prompt_block(
    item: RuntimeReminder | PromptBlock,
    *,
    order: int,
) -> PromptBlock:
    if isinstance(item, PromptBlock):
        metadata = dict(item.metadata or {})
        metadata.setdefault("request_layer", "reminder")
        metadata.setdefault(
            "cache_partition",
            "session" if metadata.get("cacheable", True) else "dynamic",
        )
        if "cacheable" not in metadata:
            metadata["cacheable"] = metadata.get("cache_partition") != "dynamic"
        return PromptBlock(
            name=item.name,
            content=item.content,
            order=item.order or order,
            enabled=item.enabled,
            metadata=metadata,
        )
    return item.to_prompt_block(order)


def collect_runtime_reminder_prompt_blocks(
    agent: "BaseAgent",
    sources: Iterable[BaseRuntimeReminderSource],
    *,
    start_order: int,
) -> list[PromptBlock]:
    blocks: list[PromptBlock] = []
    next_order = start_order
    for source in sources:
        for item in source.build_runtime_reminders(agent) or []:
            block = reminder_to_prompt_block(item, order=next_order)
            if block.enabled and block.render():
                blocks.append(block)
                next_order = max(next_order + 10, block.order + 10)
    return blocks


__all__ = [
    "BaseRuntimeReminderSource",
    "RuntimeReminder",
    "StaticRuntimeReminderSource",
    "collect_runtime_reminder_prompt_blocks",
    "reminder_to_prompt_block",
]
