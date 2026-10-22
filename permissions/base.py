"""Minimal permission contracts used by Agent Runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator
from contextlib import contextmanager

if TYPE_CHECKING:
    from runtime.agent import AgentRuntime

from .models import PermissionDecision, PermissionRequest, PermissionGrant


class BasePermissionEngine(ABC):
    """Small stable boundary; policy storage and matching belong to implementations."""

    def bind(self, runtime: "AgentRuntime") -> None:
        """Receive request-independent runtime state; no Agent reference is needed."""

    def set_mode(self, mode: str) -> None:
        """Optional policy control. Implementations define their own mode names."""
        raise NotImplementedError("This permission engine does not support policy modes")

    @contextmanager
    def approval(self, grant: PermissionGrant) -> Iterator[None]:
        """Optional, scoped approval. Never bypass authorize or retain stale grants."""
        raise NotImplementedError("This permission engine does not support interactive approval")
        yield

    @abstractmethod
    def authorize(self, request: PermissionRequest) -> PermissionDecision:
        raise NotImplementedError

    async def aauthorize(self, request: PermissionRequest) -> PermissionDecision:
        import asyncio
        return await asyncio.to_thread(self.authorize, request)
