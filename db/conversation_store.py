from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.history import CanonicalMessage

from .session_store import DEFAULT_SESSION_DB_PATH


def _ensure_parent_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str]) -> Any:
    if not value:
        return None
    return json.loads(value)


class ConversationStore:
    """SQLite 对话消息存储。"""

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    time TEXT,
                    metadata TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    raw_message TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, position)
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "raw_message" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN raw_message TEXT")

    def replace_messages(self, session_id: str, messages: list[Any]) -> None:
        rows = [self._message_to_row(session_id, position, message) for position, message in enumerate(messages)]

        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO messages (
                        session_id, position, role, content, time, metadata, tool_call_id, name, raw_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def load_messages(self, session_id: str) -> list[Any]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, time, metadata, tool_call_id, name, raw_message
                FROM messages
                WHERE session_id = ?
                ORDER BY position ASC
                """,
                (session_id,),
            ).fetchall()

        return [self._row_to_message(row) for row in rows]

    def delete_messages(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
        return cursor.rowcount

    def _message_to_row(
        self,
        session_id: str,
        position: int,
        message: Any,
    ) -> tuple[Any, ...]:
        payload = message.to_dict() if hasattr(message, "to_dict") else message
        if not isinstance(payload, dict):
            payload = {"role": "unknown", "content": str(payload)}
        if hasattr(message, "metadata") and payload.get("metadata") is None:
            payload["metadata"] = getattr(message, "metadata", None)
        if hasattr(message, "time") and payload.get("time") is None:
            time_value = getattr(message, "time", None)
            payload["time"] = time_value.isoformat() if hasattr(time_value, "isoformat") else time_value
        if hasattr(message, "tool_call_id") and payload.get("tool_call_id") is None:
            payload["tool_call_id"] = getattr(message, "tool_call_id", None)
        if hasattr(message, "name") and payload.get("name") is None:
            payload["name"] = getattr(message, "name", None)
        role = str(payload.get("role") or payload.get("type") or "__raw__")
        content_value = payload.get("content", "")
        if isinstance(content_value, str):
            content = content_value
        else:
            content = _json_dumps(content_value)
        time_value = payload.get("time")
        time = time_value.isoformat() if hasattr(time_value, "isoformat") else None
        metadata = _json_dumps(payload.get("metadata") or {})
        tool_call_id = payload.get("tool_call_id")
        name = payload.get("name")
        raw_message = _json_dumps(payload)

        return (
            session_id,
            position,
            role,
            content,
            time,
            metadata,
            tool_call_id,
            name,
            raw_message,
        )

    def _row_to_message(self, row: sqlite3.Row) -> Any:
        metadata = _json_loads(row["metadata"]) or {}
        time_value = datetime.fromisoformat(row["time"]) if row["time"] else None
        if row["raw_message"]:
            payload = _json_loads(row["raw_message"])
            if isinstance(payload, dict) and payload.get("record_type", payload.get("schema")) == "canonical_message":
                return CanonicalMessage.model_validate(payload)
            if isinstance(payload, dict):
                if metadata and not payload.get("metadata"):
                    payload["metadata"] = metadata
                if time_value is not None and not payload.get("time"):
                    payload["time"] = time_value.isoformat()
                if row["tool_call_id"] and not payload.get("tool_call_id"):
                    payload["tool_call_id"] = row["tool_call_id"]
                if row["name"] and not payload.get("name"):
                    payload["name"] = row["name"]
                return payload
            return payload

        role = row["role"]
        payload: dict[str, Any] = {
            "role": role,
            "content": row["content"],
        }
        if metadata:
            payload["metadata"] = metadata
        if time_value is not None:
            payload["time"] = time_value
        if row["tool_call_id"]:
            payload["tool_call_id"] = row["tool_call_id"]
        if row["name"]:
            payload["name"] = row["name"]
        return payload
