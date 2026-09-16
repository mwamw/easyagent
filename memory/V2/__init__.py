"""Internal layered-memory implementation; no eager optional backend imports."""
"""V2 layered memory: one complete BaseMemory implementation."""

__all__ = ["MemoryManage", "LayeredMemory"]


def __getattr__(name):
    if name in __all__:
        from .MemoryManage import MemoryManage
        return MemoryManage
    raise AttributeError(name)
