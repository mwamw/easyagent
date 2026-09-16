"""Replaceable memory modules; optional implementations are imported on demand."""

from .base import BaseMemory, MemoryServices, MetaMessageEmitterPort

__all__ = ["BaseMemory", "MemoryServices", "MetaMessageEmitterPort", "MemoryManage", "LayeredMemory", "MemoryConfig", "MemoryItem", "MemoryType", "ForgetType", "WorkingMemory", "BaseMemoryLayer", "migrate_memory_sqlite", "migrate_legacy_graph"]


def __getattr__(name):
    if name in {"MemoryManage", "LayeredMemory"}:
        from .V2.MemoryManage import MemoryManage
        return MemoryManage
    if name in {"MemoryConfig", "MemoryItem", "MemoryType", "ForgetType", "BaseMemoryLayer"}:
        from importlib import import_module
        return getattr(import_module(".V2.BaseMemory", __name__), name)
    if name == "WorkingMemory":
        from .V2.WorkingMemory import WorkingMemory
        return WorkingMemory
    if name in {"migrate_memory_sqlite", "migrate_legacy_graph"}:
        from . import migration
        return getattr(migration, name)
    raise AttributeError(name)
