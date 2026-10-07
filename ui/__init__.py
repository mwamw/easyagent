"""Optional user interfaces. Importing the contract does not import Rich or a TUI."""
from .base import BaseAgentUI

__all__ = ["BaseAgentUI", "TerminalUI"]


def __getattr__(name: str):
    if name == "TerminalUI":
        from .terminal import TerminalUI
        return TerminalUI
    raise AttributeError(name)
