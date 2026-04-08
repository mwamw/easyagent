from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_SESSION_DB_PATH = "db/easyagent_sessions.db"


def _ensure_parent_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str]) -> Any:
    if not value:
        return None
    return json.loads(value)


def _dt_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


class SessionStore:
    """SQLite 会话元数据与快照存储。"""

    def __init__(self, db_path: str = DEFAULT_SESSION_DB_PATH):
        self.db_path = db_path
        _ensure_parent_dir(self.db_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    snapshot TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    expires_at TEXT
                )
                """
            )

    def create_or_update_session(
        self,
        session_id: str,
        agent_type: str,
        agent_name: str,
        snapshot: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        now = datetime.now()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now.isoformat()

            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, agent_type, agent_name, snapshot, metadata,
                    created_at, updated_at, last_accessed_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    agent_type = excluded.agent_type,
                    agent_name = excluded.agent_name,
                    snapshot = excluded.snapshot,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at,
                    last_accessed_at = excluded.last_accessed_at,
                    expires_at = excluded.expires_at
                """,
                (
                    session_id,
                    agent_type,
                    agent_name,
                    _json_dumps(snapshot),
                    _json_dumps(metadata or {}),
                    created_at,
                    now.isoformat(),
                    now.isoformat(),
                    _dt_to_iso(expires_at),
                ),
            )

        return self.get_session(session_id, touch=False) or {}

    def get_session(self, session_id: str, touch: bool = True) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, agent_type, agent_name, snapshot, metadata,
                       created_at, updated_at, last_accessed_at, expires_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                return None

            if touch:
                touched_at = datetime.now().isoformat()
                conn.execute(
                    "UPDATE sessions SET last_accessed_at = ? WHERE session_id = ?",
                    (touched_at, session_id),
                )
                row = conn.execute(
                    """
                    SELECT session_id, agent_type, agent_name, snapshot, metadata,
                           created_at, updated_at, last_accessed_at, expires_at
                    FROM sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()

        return self._row_to_record(row)

    def list_sessions(
        self,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT session_id, agent_type, agent_name, snapshot, metadata,
                   created_at, updated_at, last_accessed_at, expires_at
            FROM sessions
        """
        params: list[Any] = []
        if not include_expired:
            query += " WHERE expires_at IS NULL OR expires_at > ?"
            params.append(datetime.now().isoformat())
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_record(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        return cursor.rowcount > 0

    def cleanup_expired_sessions(self, now: Optional[datetime] = None) -> int:
        current = (now or datetime.now()).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (current,),
            )
        return cursor.rowcount

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "agent_type": row["agent_type"],
            "agent_name": row["agent_name"],
            "snapshot": _json_loads(row["snapshot"]) or {},
            "metadata": _json_loads(row["metadata"]) or {},
            "created_at": _iso_to_dt(row["created_at"]),
            "updated_at": _iso_to_dt(row["updated_at"]),
            "last_accessed_at": _iso_to_dt(row["last_accessed_at"]),
            "expires_at": _iso_to_dt(row["expires_at"]),
        }
