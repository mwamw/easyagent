"""Versioned SQLite snapshots; every update checks the caller's revision."""

from contextlib import closing, contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from .store import BaseSessionStore, SessionConflictError
from storage.records import matches, validate_filter, validate_limit


class SQLiteSessionStore(BaseSessionStore):
    def __init__(self, db_path="db/easyagent_sessions_v4.db"):
        self.db_path = str(db_path)
        if self.db_path == ":memory:":
            raise ValueError("Use InMemorySessionStore for ephemeral sessions")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "sessions" in tables and "session_records" not in tables:
                raise ValueError("Legacy session database: use python -m easyagent.migrate session with a separate V4 target")
            conn.execute("""CREATE TABLE IF NOT EXISTS session_records (
                session_id TEXT PRIMARY KEY, agent_type TEXT NOT NULL, agent_name TEXT NOT NULL,
                snapshot TEXT NOT NULL, metadata TEXT NOT NULL, revision INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_accessed_at TEXT NOT NULL,
                expires_at TEXT)""")

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    def create_or_update_session(self, session_id, agent_type, agent_name, snapshot,
                                 metadata=None, expires_at=None, *, expected_revision=None):
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be a non-empty string")
        data = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        meta = json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False)
        now = datetime.now().isoformat()
        expiry = expires_at.isoformat() if expires_at is not None else None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT revision,created_at FROM session_records WHERE session_id=?", (session_id,)).fetchone()
            revision = existing["revision"] if existing else None
            if revision != expected_revision:
                raise SessionConflictError(f"Session revision changed: {session_id}")
            next_revision = (revision or 0) + 1
            conn.execute("""INSERT INTO session_records VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET agent_type=excluded.agent_type,
                agent_name=excluded.agent_name,snapshot=excluded.snapshot,metadata=excluded.metadata,
                revision=excluded.revision,updated_at=excluded.updated_at,expires_at=excluded.expires_at""",
                (session_id, agent_type, agent_name, data, meta, next_revision,
                 existing["created_at"] if existing else now, now, now, expiry))
            row = conn.execute("SELECT * FROM session_records WHERE session_id=?", (session_id,)).fetchone()
        return self._record(row)

    @staticmethod
    def _record(row):
        data = dict(row)
        if "snapshot" in data:
            data["snapshot"] = json.loads(data["snapshot"])
        data["metadata"] = json.loads(data["metadata"])
        for key in ("created_at", "updated_at", "last_accessed_at", "expires_at"):
            data[key] = datetime.fromisoformat(data[key]) if data.get(key) else None
        return data

    def get_session(self, session_id, touch=False):
        with self._connect() as conn:
            if touch:
                conn.execute("UPDATE session_records SET last_accessed_at=? WHERE session_id=?", (datetime.now().isoformat(), session_id))
            row = conn.execute("SELECT * FROM session_records WHERE session_id=?", (session_id,)).fetchone()
        return self._record(row) if row else None

    def list_sessions(self, limit=100, include_expired=False, *, offset=0, metadata_filter=None):
        validate_limit(limit)
        validate_limit(offset)
        validate_filter(metadata_filter)
        if not limit:
            return []
        sql = "SELECT session_id,agent_type,agent_name,metadata,revision,created_at,updated_at,last_accessed_at,expires_at FROM session_records"
        params = []
        if not include_expired:
            sql += " WHERE expires_at IS NULL OR expires_at > ?"
            params.append(datetime.now().isoformat())
        sql += " ORDER BY updated_at DESC,session_id"
        result = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                record = self._record(row)
                if not matches(record["metadata"], metadata_filter):
                    continue
                if offset:
                    offset -= 1
                    continue
                result.append(record)
                if len(result) == limit:
                    break
        return result

    def delete_session(self, session_id):
        with self._connect() as conn:
            return conn.execute("DELETE FROM session_records WHERE session_id=?", (session_id,)).rowcount > 0

    def cleanup_expired_sessions(self, now=None):
        with self._connect() as conn:
            return conn.execute("DELETE FROM session_records WHERE expires_at IS NOT NULL AND expires_at<=?", ((now or datetime.now()).isoformat(),)).rowcount
