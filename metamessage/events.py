"""Runtime events consumed by the meta-message manager."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass(slots=True, frozen=True)
class MetaMessageEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise ValueError("MetaMessageEvent.name must be a non-empty string")


__all__ = ["MetaMessageEvent"]
