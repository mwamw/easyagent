"""Minimal permission contracts used by Agent Runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.agent import AgentRuntime

from .models import PermissionDecision, PermissionRequest


class BasePermissionEngine(ABC):
    """Small stable boundary; policy storage and matching belong to implementations."""

    def bind(self, runtime: "AgentRuntime") -> None:
        """Receive request-independent runtime state; no Agent reference is needed."""

    def set_mode(self, mode: str) -> None:
        """Optional policy control. Implementations define their own mode names."""
        raise NotImplementedError("This permission engine does not support policy modes")

    @abstractmethod
    def authorize(self, request: PermissionRequest) -> PermissionDecision:
        raise NotImplementedError

    async def aauthorize(self, request: PermissionRequest) -> PermissionDecision:
        import asyncio
        return await asyncio.to_thread(self.authorize, request)
