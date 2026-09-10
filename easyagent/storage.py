"""Stable storage facade, independent of memory and RAG domain types."""

import storage as _storage

__all__ = _storage.__all__


def __getattr__(name):
    if name in __all__:
        return getattr(_storage, name)
    raise AttributeError(name)
