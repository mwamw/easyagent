"""Explicit, copy-only migration from legacy SQLite snapshots."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3


def migrate_session_v3(source, target, *, dry_run=True):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Migration target must differ from the source")
    if not source.is_file():
        raise FileNotFoundError(source)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = list(connection.execute("SELECT * FROM sessions"))
    converted = []
    for row in rows:
        previous = json.loads(row["snapshot"])
        if previous.get("schemaVersion") != 3:
            raise ValueError(f"Unsupported legacy snapshot: {row['session_id']}")
        modules = {}
        for name, data in (previous.get("modules") or {}).items():
            if name in {"tools", "executor", "prompt", "executionContext"}:
                continue
            if name == "mcp":
                for index, component in enumerate(data or []):
                    modules[f"mcp:{component.get('identity', index)}"] = {**component, "version": 1, "required": False}
            elif isinstance(data, dict):
                modules[name] = {**data, "version": 1, "required": False}
        snapshot = {"schemaVersion": 4, "agentType": previous.get("agentType"), "name": previous.get("name"),
                    "history": previous.get("history", {}), "modules": modules,
                    "tools": previous.get("modules", {}).get("tools", {}).get("state", {}).get("names", []),
                    "reasoning": previous.get("reasoning"), "currentTaskId": previous.get("currentTaskId"),
                    "migration": {"sourceVersion": 3, "historyCompleteness": "legacy_snapshot"}}
        converted.append((row, snapshot))
    report = {"dry_run": dry_run, "count": len(converted), "source": str(source), "target": str(target),
              "history_completeness": "legacy_snapshot", "source_preserved": True}
    if dry_run:
        report["target_exists"] = target.exists()
        return report
    if target.exists():
        with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables != {"session_records"} or conn.execute("SELECT count(*) FROM session_records").fetchone()[0]:
                raise FileExistsError("Use a new or empty V4 target; existing data is never overwritten")
    from .stores.sqlite import SQLiteSessionStore
    store = SQLiteSessionStore(target)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT count(*) FROM session_records").fetchone()[0]:
            raise FileExistsError("Migration target is no longer empty")
        for row, snapshot in converted:
            conn.execute("INSERT INTO session_records VALUES (?,?,?,?,?,?,?,?,?,?)", (
                row["session_id"], row["agent_type"], row["agent_name"], json.dumps(snapshot, allow_nan=False),
                json.dumps(json.loads(row["metadata"] or "{}"), allow_nan=False), 1,
                row["created_at"], row["updated_at"], row["last_accessed_at"], row["expires_at"],
            ))
    return report
