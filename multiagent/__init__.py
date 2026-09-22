"""Replaceable multi-agent module and built-in implementations."""

from .base import BaseMultiAgent


def __getattr__(name: str):
    if name == "ClaudeCodeMultiAgent":
        from .default import ClaudeCodeMultiAgent

        return ClaudeCodeMultiAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseMultiAgent", "ClaudeCodeMultiAgent"]
