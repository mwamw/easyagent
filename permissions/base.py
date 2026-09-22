"""Minimal permission contracts used by Agent Runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import PermissionDecision, PermissionRequest


class BasePermissionEngine(ABC):
    """Small stable boundary; policy storage and matching belong to implementations."""

    @abstractmethod
    def authorize(self, request: PermissionRequest) -> PermissionDecision:
        raise NotImplementedError

    async def aauthorize(self, request: PermissionRequest) -> PermissionDecision:
        import asyncio
        return await asyncio.to_thread(self.authorize, request)
