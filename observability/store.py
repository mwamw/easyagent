"""Storage backends for finalized AgentInvoke records."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import sqlite3
import threading

from .models import AgentInvoke


class BaseObservabilityStore(ABC):
    @abstractmethod
    def save(self, invoke: AgentInvoke) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, invoke_id: str) -> AgentInvoke | None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[AgentInvoke]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class InMemoryObservabilityStore(BaseObservabilityStore):
    def __init__(self):
        self._records: dict[str, AgentInvoke] = {}
        self._lock = threading.RLock()

    def save(self, invoke: AgentInvoke) -> None:
        with self._lock:
            self._records[invoke.invoke_id] = invoke.model_copy(deep=True)

    def get(self, invoke_id: str) -> AgentInvoke | None:
        with self._lock:
            record = self._records.get(str(invoke_id))
            return record.model_copy(deep=True) if record is not None else None

    def list(self) -> list[AgentInvoke]:
        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda item: item.stats.started_at,
            )
            return [item.model_copy(deep=True) for item in records]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class SQLiteObservabilityStore(BaseObservabilityStore):
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or os.getcwd(), exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_invokes (
                invoke_id TEXT PRIMARY KEY,
                parent_invoke_id TEXT,
                agent_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_invokes_parent ON agent_invokes(parent_invoke_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_invokes_started ON agent_invokes(started_at)"
        )
        self._connection.commit()

    def save(self, invoke: AgentInvoke) -> None:
        payload = json.dumps(invoke.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO agent_invokes (
                    invoke_id, parent_invoke_id, agent_id, success, started_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(invoke_id) DO UPDATE SET
                    parent_invoke_id=excluded.parent_invoke_id,
                    agent_id=excluded.agent_id,
                    success=excluded.success,
                    started_at=excluded.started_at,
                    payload=excluded.payload
                """,
                (
                    invoke.invoke_id,
                    invoke.parent_invoke_id,
                    invoke.agent_id,
                    1 if invoke.stats.success else 0,
                    invoke.stats.started_at.isoformat(),
                    payload,
                ),
            )
            self._connection.commit()

    def get(self, invoke_id: str) -> AgentInvoke | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM agent_invokes WHERE invoke_id = ?",
                (str(invoke_id),),
            ).fetchone()
        if row is None:
            return None
        return AgentInvoke.model_validate_json(row[0])

    def list(self) -> list[AgentInvoke]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM agent_invokes ORDER BY started_at, invoke_id"
            ).fetchall()
        return [AgentInvoke.model_validate_json(row[0]) for row in rows]

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM agent_invokes")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True


__all__ = [
    "BaseObservabilityStore",
    "InMemoryObservabilityStore",
    "SQLiteObservabilityStore",
]
