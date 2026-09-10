"""Thread-safe ephemeral implementation of the session persistence contract."""

from copy import deepcopy
from datetime import datetime
import json
from threading import RLock

from .store import BaseSessionStore, SessionConflictError
from storage.records import matches, validate_filter, validate_limit


class InMemorySessionStore(BaseSessionStore):
    def __init__(self):
        self._records = {}
        self._lock = RLock()

    def create_or_update_session(self, session_id, agent_type, agent_name, snapshot, metadata=None, expires_at=None, *, expected_revision=None):
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be a non-empty string")
        snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
        metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        with self._lock:
            old = self._records.get(session_id)
            if (old["revision"] if old else None) != expected_revision:
                raise SessionConflictError(f"Session revision changed: {session_id}")
            now = datetime.now()
            record = dict(session_id=session_id, agent_type=agent_type, agent_name=agent_name,
                          snapshot=snapshot, metadata=metadata, revision=(old["revision"] if old else 0) + 1,
                          created_at=old["created_at"] if old else now, updated_at=now,
                          last_accessed_at=old["last_accessed_at"] if old else now, expires_at=expires_at)
            self._records[session_id] = record
            return deepcopy(record)

    def get_session(self, session_id, touch=False):
        with self._lock:
            record = self._records.get(session_id)
            if record and touch:
                record["last_accessed_at"] = datetime.now()
            return deepcopy(record)

    def list_sessions(self, limit=100, include_expired=False, *, offset=0, metadata_filter=None):
        validate_limit(limit)
        validate_limit(offset)
        validate_filter(metadata_filter)
        with self._lock:
            records = [deepcopy(record) for record in self._records.values()
                       if (include_expired or record["expires_at"] is None or record["expires_at"] > datetime.now())
                       and matches(record["metadata"], metadata_filter)]
        records.sort(key=lambda record: record["updated_at"], reverse=True)
        for record in records:
            record.pop("snapshot")
        return records[offset:offset+limit]

    def delete_session(self, session_id):
        with self._lock:
            return self._records.pop(session_id, None) is not None

    def cleanup_expired_sessions(self, now=None):
        with self._lock:
            ids = [key for key, record in self._records.items() if record["expires_at"] is not None and record["expires_at"] <= (now or datetime.now())]
            for key in ids:
                del self._records[key]
            return len(ids)
