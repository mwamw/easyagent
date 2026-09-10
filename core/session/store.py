"""Replaceable persistence contract. No Agent construction or product policy."""

from abc import ABC, abstractmethod
import asyncio


class SessionConflictError(RuntimeError):
    pass


class BaseSessionStore(ABC):
    def create_or_update_session(self, session_id, agent_type, agent_name, snapshot,
                                 metadata=None, expires_at=None, *, expected_revision=None):
        raise NotImplementedError("This session-store operation is not implemented; use its supported sync/async API")

    def get_session(self, session_id, touch=False):
        raise NotImplementedError("This session-store operation is not implemented; use its supported sync/async API")

    def list_sessions(self, limit=100, include_expired=False, *, offset=0, metadata_filter=None):
        raise NotImplementedError("This session-store operation is not implemented; use its supported sync/async API")

    def delete_session(self, session_id):
        raise NotImplementedError("This session-store operation is not implemented; use its supported sync/async API")

    def cleanup_expired_sessions(self, now=None):
        raise NotImplementedError("This session-store operation is not implemented; use its supported sync/async API")

    async def aget_session(self, session_id):
        return await asyncio.to_thread(self.get_session, session_id)

    async def asave_session(self, **kwargs):
        return await asyncio.to_thread(self.create_or_update_session, **kwargs)

    def close(self):
        pass

    async def aclose(self):
        await asyncio.to_thread(self.close)
