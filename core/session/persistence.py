"""Data-only session capture/restore into explicitly assembled components."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, date
import json
import inspect
from threading import RLock
from uuid import uuid4

from core.Exception import SessionNotFoundError, SessionSerializationError
from .report import SessionRestoreReport
from .store import BaseSessionStore


def _state_json(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Session state must contain data, not {type(value).__name__}")


@dataclass
class SessionStateParticipant:
    name: str
    component: object
    version: int = 1
    required: bool = False
    forkable: bool = True

    def resolve(self):
        return self.component() if inspect.isfunction(self.component) or inspect.ismethod(self.component) else self.component


class SessionPersistence:
    def __init__(self, agent):
        self.agent = agent
        self.store = None
        self._participants = {}
        self._revisions = {}
        self._unknown = {}
        self._lock = RLock()
        self.busy = False
        self._metadata = {}
        self._stores = {}

    def register(self, participant):
        if not isinstance(participant, SessionStateParticipant) or not participant.name:
            raise TypeError("Expected a named SessionStateParticipant")
        if participant.version < 1:
            raise ValueError("Session state versions must be positive")
        self._participants[participant.name] = participant

    def resolve_store(self, store=None):
        store = store if store is not None else self.store
        if isinstance(store, str):
            from .sqlite import SQLiteSessionStore
            if self.store is None or getattr(self.store, "db_path", None) != store:
                self.store = SQLiteSessionStore(store)
            store = self.store
        if store is None:
            from .sqlite import SQLiteSessionStore
            self.store = store = SQLiteSessionStore()
        if not isinstance(store, BaseSessionStore):
            raise TypeError("Session store must implement BaseSessionStore")
        # Retain identities while revision/metadata caches use them as keys.
        self._stores[id(store)] = store
        return store

    def _idle(self):
        if self.busy:
            raise RuntimeError("A session operation is already in progress")
        if self.agent._closed:
            raise RuntimeError("Cannot persist or restore a closed Agent")
        if self.agent._invocation_lock.locked():
            raise RuntimeError("Session operations require an idle Agent")

    def snapshot(self, *, fork=False):
        modules = {} if fork else deepcopy(self._unknown)
        for name, binding in self._participants.items():
            if name == "context":
                continue
            component = binding.resolve()
            if component is None or (fork and not binding.forkable):
                continue
            exporter = getattr(component, "export_state", None)
            value = exporter() if callable(exporter) else None
            if fork and callable(getattr(component, "fork_state", None)):
                value = component.fork_state(deepcopy(value))
            modules[name] = {"version": binding.version, "required": binding.required,
                             "implementation": type(component).__name__, "state": value}
        snapshot = {"schemaVersion": 4, "agentType": type(self.agent).__name__, "name": self.agent.name,
                    "history": self.agent.context_manager.export_state(), "modules": modules,
                    "tools": self.agent.tool_registry.get_tool_names() if self.agent.tool_registry else [],
                    "reasoning": deepcopy(self.agent.reasoning),
                    "currentTaskId": None if fork else self.agent.current_task_id}
        # Fail at the boundary rather than silently stringify live Python objects.
        return json.loads(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, default=_state_json))

    def save(self, session_id=None, *, store=None, metadata=None, expires_at=None):
        with self._lock:
            self._idle()
            store = self.resolve_store(store)
            session_id = session_id or self.agent.session_id
            key = (id(store), session_id)
            record = store.create_or_update_session(session_id=session_id, agent_type=type(self.agent).__name__,
                agent_name=self.agent.name, snapshot=self.snapshot(), metadata=self._metadata.get(key, {}) if metadata is None else metadata, expires_at=expires_at,
                expected_revision=self._revisions.get(key))
            self._revisions[key] = record["revision"]
            self._metadata[key] = deepcopy(record["metadata"])
            self.agent.session_id = session_id
            return session_id

    def fork(self, session_id=None, *, store=None, metadata=None):
        with self._lock:
            self._idle()
            if self.agent.get_pending_interruption() is not None:
                raise SessionSerializationError("Resolve the pending interaction before forking")
            store = self.resolve_store(store)
            session_id = session_id or f"session_{uuid4().hex}"
            lineage = {**dict(metadata or {}), "forked_from_session_id": self.agent.session_id,
                       "forked_from_revision": self._revisions.get((id(store), self.agent.session_id))}
            store.create_or_update_session(session_id=session_id, agent_type=type(self.agent).__name__,
                agent_name=self.agent.name, snapshot=self.snapshot(fork=True), metadata=lineage)
            return session_id

    def restore(self, session_id, *, store=None):
        with self._lock:
            self._idle()
            store = self.resolve_store(store)
            return self._restore_record(session_id, store, store.get_session(session_id))

    async def asave(self, session_id=None, *, store=None, metadata=None, expires_at=None):
        self._idle()
        self.busy = True
        try:
            store = self.resolve_store(store)
            session_id = session_id or self.agent.session_id
            key = (id(store), session_id)
            record = await store.asave_session(
                session_id=session_id, agent_type=type(self.agent).__name__, agent_name=self.agent.name,
                snapshot=self.snapshot(), metadata=self._metadata.get(key, {}) if metadata is None else metadata,
                expires_at=expires_at, expected_revision=self._revisions.get(key),
            )
            self._revisions[key] = record["revision"]
            self._metadata[key] = deepcopy(record["metadata"])
            self.agent.session_id = session_id
            return session_id
        finally:
            self.busy = False

    async def arestore(self, session_id, *, store=None):
        self._idle()
        self.busy = True
        try:
            store = self.resolve_store(store)
            record = await store.aget_session(session_id)
            self._restore_record(session_id, store, record, notify=False)
            if self.agent.hook_manager is not None:
                hook = await self.agent.hook_manager.arun_stage("after_session_restore", {
                    "session_id": session_id, "snapshot": record["snapshot"],
                    "restore_report": self.agent.last_restore_report,
                })
                self._apply_report(hook, self.agent.last_restore_report)
            return self.agent
        finally:
            self.busy = False

    async def afork(self, session_id=None, *, store=None, metadata=None):
        self._idle()
        if self.agent.get_pending_interruption() is not None:
            raise SessionSerializationError("Resolve the pending interaction before forking")
        self.busy = True
        try:
            store = self.resolve_store(store)
            session_id = session_id or f"session_{uuid4().hex}"
            lineage = {**dict(metadata or {}), "forked_from_session_id": self.agent.session_id,
                       "forked_from_revision": self._revisions.get((id(store), self.agent.session_id))}
            await store.asave_session(session_id=session_id, agent_type=type(self.agent).__name__,
                agent_name=self.agent.name, snapshot=self.snapshot(fork=True), metadata=lineage)
            return session_id
        finally:
            self.busy = False

    def _apply_report(self, hook, report):
        report = hook.payload.get("restore_report", report)
        if not isinstance(report, SessionRestoreReport):
            raise TypeError("Restore hook must return a SessionRestoreReport")
        if hook.blocked:
            report.add_issue(component="hooks", code="restore_hook_blocked", message=hook.message, severity="error")
        self.agent.last_restore_report = report

    def _restore_record(self, session_id, store, record, *, notify=True):
        if record is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        snapshot = record["snapshot"]
        if snapshot.get("schemaVersion") != 4:
            raise SessionSerializationError("Explicitly migrate the legacy session to V4 first")
        report = SessionRestoreReport(session_id=session_id, agent_type=type(self.agent).__name__)
        history = snapshot.get("history")
        try:
            self.agent.context_manager.validate_state(history)
        except Exception as exc:
            raise SessionSerializationError("Invalid Context session state") from exc
        provider = history.get("providerName") if isinstance(history, dict) else None
        if provider and provider != self.agent.llm.provider_name:
            report.add_issue(component="history", code="provider_drift",
                message="Replay was rebuilt for a different provider",
                metadata={"savedProvider": provider, "currentProvider": self.agent.llm.provider_name})
        operations = []
        unknown = {}
        for name, value in snapshot.get("modules", {}).items():
            if name == "context":
                continue
            binding = self._participants.get(name)
            component = binding.resolve() if binding else None
            importer = getattr(component, "restore_state", None)
            if component is None or (value.get("state") is not None and not callable(importer)):
                if value.get("required"):
                    raise SessionSerializationError(f"Required session component is missing: {name}")
                unknown[name] = deepcopy(value)
                report.add_issue(component=name, code="component_missing", message=f"Supply {name} explicitly to restore its state")
                continue
            if value.get("version", 1) != binding.version:
                raise SessionSerializationError(f"Unsupported state version for {name}")
            if value.get("state") is not None:
                validator = getattr(component, "validate_state", None)
                if callable(validator):
                    validator(deepcopy(value["state"]))
                operations.append((name, component, deepcopy(value["state"])))
        for name, binding in self._participants.items():
            if name == "context":
                continue
            if snapshot.get("modules", {}).get(name, {}).get("state") is None:
                component = binding.resolve()
                if callable(getattr(component, "reset_session_state", None)):
                    operations.append((name, component, None))
        saved = self.agent.context_manager.export_state()
        backups = []
        try:
            self.agent.context_manager.restore_state(snapshot.get("history"))
            for name, component, value in operations:
                exporter = getattr(component, "export_state", None)
                backups.append((component, deepcopy(exporter()) if callable(exporter) else None))
                outcome = component.reset_session_state() if value is None else component.restore_state(value)
                if isinstance(outcome, dict):
                    report.extend_component(name, outcome)
        except BaseException:
            self.agent.context_manager.restore_state(saved)
            for component, backup in reversed(backups):
                component.restore_state(backup)
            raise
        self._unknown = unknown
        self.agent.session_id = session_id
        self.agent.reasoning = snapshot.get("reasoning")
        self.agent.current_task_id = snapshot.get("currentTaskId")
        self._revisions[(id(store), session_id)] = record["revision"]
        self._metadata[(id(store), session_id)] = deepcopy(record["metadata"])
        missing = [name for name in snapshot.get("tools", []) if self.agent.tool_registry is None or not self.agent.tool_registry.has_tool(name)]
        report.note_missing_tools(missing)
        reconcile = getattr(self.agent.metamessage_manager, "reconcile_history", None)
        if callable(reconcile):
            reconcile()
        self.agent._refresh_execution_context()
        self.agent.last_restore_report = report
        if notify and self.agent.hook_manager is not None:
            hook = self.agent.hook_manager.after_session_restore({"session_id": session_id, "restore_report": report, "snapshot": snapshot})
            self._apply_report(hook, report)
        return self.agent
