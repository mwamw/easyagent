"""UI-independent replies to suspended Agent actions."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class InteractionReply:
    request_id: str
    action: Literal["approve", "deny", "answer"]
    answer: str = ""
