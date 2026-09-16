"""One budget for the complete model input; no source-specific allocation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int | None = None

    def __post_init__(self):
        if self.max_tokens is not None and (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 0):
            raise ValueError("max_tokens must be a non-negative integer or None")

    def remaining(self, used: int) -> int | None:
        return None if self.max_tokens is None else max(0, self.max_tokens - used)


class ContextBudgetExceeded(ValueError):
    def __init__(self, tokens: int, max_tokens: int):
        self.tokens = tokens
        self.max_tokens = max_tokens
        super().__init__(f"Request requires approximately {tokens} input tokens; budget is {max_tokens}")
