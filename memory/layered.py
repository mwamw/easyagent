"""The existing layered memory scheme as one opt-in framework module."""

from __future__ import annotations

from typing import Any

from .base import BaseMemory
from .V2.BaseMemory import MemoryConfig
from .V2.MemoryManage import MemoryManage


class LayeredMemory(MemoryManage, BaseMemory):
    """Model-directed memory access; defaults to working memory, with no auto-recall.

    Injected storage/embedding instances are borrowed, not closed by this module.
    """

    def __init__(self, config: MemoryConfig | None = None, **kwargs: Any):
        try:
            super().__init__(config=config or MemoryConfig(), **kwargs)
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for resource in reversed(getattr(self, "_owned_resources", [])):
            try:
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                errors.append(str(exc))
        self._owned_resources = []
        if errors:
            raise RuntimeError("Memory resource cleanup failed: " + "; ".join(errors))

    def get_prompt_blocks(self, context):
        from prompt import PromptBlock, build_memory_prompt_section

        return [PromptBlock(
            "memory.layered.rules",
            build_memory_prompt_section(
                supported_memory_types=list(self.memory_types),
                include_working_memory=True,
                language="en",
            ),
            placement="system_reminder",
            order=140,
        )]

    def get_tools(self):
        from Tool.builtin.memorytool import build_memory_tools

        return build_memory_tools(self, session_id_provider=lambda: self.services.scope.session_id)

    def export_state(self):
        working = self.memory_types.get("working")
        return {"user_id": self.user_id, "working": [item.model_dump(mode="json") for item in working.get_all_memories()] if working else []}

    def validate_state(self, state):
        from .V2.BaseMemory import MemoryItem
        if state.get("user_id") != self.user_id:
            raise ValueError("Cannot restore memory owned by another user")
        for item in state.get("working", []):
            MemoryItem.model_validate(item)

    def reset_session_state(self):
        working = self.memory_types.get("working")
        if working is not None:
            working.clear_memory()

    def restore_state(self, state):
        from .V2.BaseMemory import MemoryItem
        self.validate_state(state)
        working = self.memory_types.get("working")
        if working is not None:
            working.clear_memory()
            for item in state.get("working", []):
                working.add_memory(MemoryItem.model_validate(item))
