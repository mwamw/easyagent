"""Application-facing UI contract, independent of terminal dependencies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import BasicAgent


class BaseAgentUI(ABC):
    @abstractmethod
    async def run(self, agent: BasicAgent) -> None:
        """Interact with a borrowed Agent; never implicitly take ownership of it."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release UI resources, not the supplied Agent."""
