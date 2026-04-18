from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

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

    `replay_history` stores request-ready history entries for the active provider.
    `system_prompt` remains separate so providers do not need to scan history to
    extract it.
    """

    provider_name: Optional[str]
    replay_history: list[Any] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def append_replay(self, message: Any) -> None:
        self.extend_replay([message])

    def extend_replay(self, messages: Iterable[Any]) -> None:
        codec = create_codec(self.provider_name)
        for message in messages:
            entry_payload = _clone_payload(message)
            codec.append_replay_entry(self.replay_history, entry_payload)

    def set_replay_history(self, replay_history: list[Any]) -> None:
        self.replay_history = [_clone_payload(item) for item in replay_history]

    def clone(self) -> "ReplayRequestInput":
        return ReplayRequestInput(
            provider_name=self.provider_name,
            replay_history=[_clone_payload(item) for item in self.replay_history],
            system_prompt=self.system_prompt,
        )
