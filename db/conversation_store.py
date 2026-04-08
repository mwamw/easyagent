from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.Message import (
    AssistantMessage,
    GoogleToolMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

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
        if isinstance(message, Message):
            role = message.role
            content = message.content
            time = message.time.isoformat() if message.time else None
            metadata = _json_dumps(message.metadata or {})
            tool_call_id = getattr(message, "tool_call_id", None)
            name = getattr(message, "name", None)
            raw_message = None
        else:
            payload = message.to_dict() if hasattr(message, "to_dict") else message
            if not isinstance(payload, dict):
                payload = {"role": "unknown", "content": str(payload)}
            role = str(payload.get("role") or payload.get("type") or "__raw__")
            content_value = payload.get("content", "")
            if isinstance(content_value, str):
                content = content_value
            else:
                content = _json_dumps(content_value)
            time = None
            metadata = _json_dumps({})
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
        if row["raw_message"]:
            return _json_loads(row["raw_message"])

        role = row["role"]
        kwargs = {
            "time": datetime.fromisoformat(row["time"]) if row["time"] else None,
            "metadata": _json_loads(row["metadata"]) or {},
        }

        if role == "user":
            return UserMessage(row["content"], **kwargs)
        if role == "assistant":
            return AssistantMessage(row["content"], **kwargs)
        if role == "system":
            return SystemMessage(row["content"], **kwargs)
        if role == "tool":
            return ToolMessage(
                row["content"],
                tool_call_id=row["tool_call_id"],
                name=row["name"],
                **kwargs,
            )
        if role == "function":
            return GoogleToolMessage(
                row["content"],
                tool_call_id=row["tool_call_id"],
                name=row["name"],
                **kwargs,
            )
        return Message(role=role, content=row["content"], **kwargs)
