"""Session lifecycle orchestration over explicitly injected runtime dependencies."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
import inspect
import json
from threading import RLock
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid4

from context.base import BaseContextManager
from runtime.agent import AgentRuntime
from core.Exception import SessionNotFoundError, SessionSerializationError

from .report import ComponentRestoreReport, SessionRestoreReport
from .state import (
    SessionState, SessionStateData, ValidatableSessionState,
    ResettableSessionState, ForkableSessionState,
)
from .stores.base import BaseSessionStore


def _state_json(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Session state must contain data, not {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class SessionStateParticipant:
    """A component that contributes serializable state to a session."""

    name: str
    component: SessionState | Callable[[], SessionState | None]
    version: int = 1
    required: bool = False
    forkable: bool = True

    def resolve(self) -> SessionState | None:
        if inspect.isfunction(self.component) or inspect.ismethod(self.component):
            component = self.component()
        else:
            component = self.component
        if component is None or isinstance(component, SessionState):
            return component
        if self.required:
            raise TypeError(f"Required participant {self.name!r} must implement SessionState")
        return None


@dataclass(frozen=True, slots=True)
class SessionBindings:
    """Explicit module dependencies; shared execution state lives in AgentRuntime."""

    agent_type: str
    agent_name: str
    context: Callable[[], BaseContextManager]
    provider_name: Callable[[], str | None]
    tool_names: Callable[[], list[str]]
    has_tool: Callable[[str], bool]



class SessionManager:
    """Capture, persist, fork, and restore one explicitly assembled runtime."""

    def __init__(self, runtime: AgentRuntime, bindings: SessionBindings) -> None:
        if not isinstance(runtime, AgentRuntime):
            raise TypeError("runtime must be AgentRuntime")
        self.runtime = runtime
        self.bindings = bindings
        self.store: BaseSessionStore | None = None
        self._participants: dict[str, SessionStateParticipant] = {}
        self._revisions: dict[tuple[int, str], int] = {}
        self._unknown: dict[str, Any] = {}
        self._lock = RLock()
        self.busy = False
        self._metadata: dict[tuple[int, str], dict[str, Any]] = {}
        self._stores: dict[int, BaseSessionStore] = {}
        self.last_restored_snapshot: dict[str, Any] | None = None

    def register(self, participant: SessionStateParticipant) -> None:
        if not isinstance(participant, SessionStateParticipant) or not participant.name:
            raise TypeError("Expected a named SessionStateParticipant")
        if participant.version < 1:
            raise ValueError("Session state versions must be positive")
        self._participants[participant.name] = participant

    def resolve_store(self, store: BaseSessionStore | str | None = None) -> BaseSessionStore:
        resolved: BaseSessionStore | str | None = store if store is not None else self.store
        if isinstance(resolved, str):
            from .stores.sqlite import SQLiteSessionStore

            if self.store is None or getattr(self.store, "db_path", None) != resolved:
                self.store = SQLiteSessionStore(resolved)
            resolved = self.store
        if resolved is None:
            from .stores.sqlite import SQLiteSessionStore

            self.store = resolved = SQLiteSessionStore()
        if not isinstance(resolved, BaseSessionStore):
            raise TypeError("Session store must implement BaseSessionStore")
        self._stores[id(resolved)] = resolved
        return resolved

    def _ensure_idle(self) -> None:
        if self.busy:
            raise RuntimeError("A session operation is already in progress")
        self.runtime.ensure_idle()

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._lock:
            self._ensure_idle()
            with self.runtime.operation():
                self.busy = True
                try:
                    yield
                finally:
                    self.busy = False

    def snapshot(self, *, fork: bool = False) -> dict[str, Any]:
        modules = {} if fork else deepcopy(self._unknown)
        for name, binding in self._participants.items():
            if name == "context":
                continue
            component = binding.resolve()
            if component is None or (fork and not binding.forkable):
                continue
            value = component.export_state()
            if fork and isinstance(component, ForkableSessionState):
                value = component.fork_state(deepcopy(value))
            modules[name] = {
                "version": binding.version,
                "required": binding.required,
                "implementation": type(component).__name__,
                "state": value,
            }
        snapshot = {
            "schemaVersion": 4,
            "agentType": self.bindings.agent_type,
            "name": self.bindings.agent_name,
            "history": self.bindings.context().export_state(),
            "modules": modules,
            "tools": self.bindings.tool_names(),
            "reasoning": deepcopy(self.runtime.reasoning),
            "currentTaskId": None if fork else self.runtime.current_task_id,
        }
        return json.loads(
            json.dumps(snapshot, ensure_ascii=False, allow_nan=False, default=_state_json)
        )

    def save(
        self,
        session_id: str | None = None,
        *,
        store: BaseSessionStore | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        with self._operation():
            resolved = self.resolve_store(store)
            resolved_id = session_id or self.runtime.session_id
            key = (id(resolved), resolved_id)
            record = resolved.create_or_update_session(
                session_id=resolved_id,
                agent_type=self.bindings.agent_type,
                agent_name=self.bindings.agent_name,
                snapshot=self.snapshot(),
                metadata=self._metadata.get(key, {}) if metadata is None else dict(metadata),
                expires_at=expires_at,
                expected_revision=self._revisions.get(key),
            )
            self._remember_record(resolved, resolved_id, record)
            self.runtime.session_id = resolved_id
            return resolved_id

    def fork(
        self,
        session_id: str | None = None,
        *,
        store: BaseSessionStore | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self._operation():
            self._ensure_forkable()
            resolved = self.resolve_store(store)
            source_id = self.runtime.session_id
            fork_id = session_id or f"session_{uuid4().hex}"
            lineage = {
                **dict(metadata or {}),
                "forked_from_session_id": source_id,
                "forked_from_revision": self._revisions.get((id(resolved), source_id)),
            }
            resolved.create_or_update_session(
                session_id=fork_id,
                agent_type=self.bindings.agent_type,
                agent_name=self.bindings.agent_name,
                snapshot=self.snapshot(fork=True),
                metadata=lineage,
            )
            return fork_id

    def restore(
        self,
        session_id: str,
        *,
        store: BaseSessionStore | str | None = None,
    ) -> SessionRestoreReport:
        with self._operation():
            resolved = self.resolve_store(store)
            return self._restore_record(session_id, resolved, resolved.get_session(session_id))

    async def asave(
        self,
        session_id: str | None = None,
        *,
        store: BaseSessionStore | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        with self._operation():
            resolved = self.resolve_store(store)
            resolved_id = session_id or self.runtime.session_id
            key = (id(resolved), resolved_id)
            record = await resolved.asave_session(
                session_id=resolved_id,
                agent_type=self.bindings.agent_type,
                agent_name=self.bindings.agent_name,
                snapshot=self.snapshot(),
                metadata=self._metadata.get(key, {}) if metadata is None else dict(metadata),
                expires_at=expires_at,
                expected_revision=self._revisions.get(key),
            )
            self._remember_record(resolved, resolved_id, record)
            self.runtime.session_id = resolved_id
            return resolved_id

    async def arestore(
        self,
        session_id: str,
        *,
        store: BaseSessionStore | str | None = None,
    ) -> SessionRestoreReport:
        with self._operation():
            resolved = self.resolve_store(store)
            return self._restore_record(
                session_id,
                resolved,
                await resolved.aget_session(session_id),
            )

    async def afork(
        self,
        session_id: str | None = None,
        *,
        store: BaseSessionStore | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self._operation():
            self._ensure_forkable()
            resolved = self.resolve_store(store)
            source_id = self.runtime.session_id
            fork_id = session_id or f"session_{uuid4().hex}"
            lineage = {
                **dict(metadata or {}),
                "forked_from_session_id": source_id,
                "forked_from_revision": self._revisions.get((id(resolved), source_id)),
            }
            await resolved.asave_session(
                session_id=fork_id,
                agent_type=self.bindings.agent_type,
                agent_name=self.bindings.agent_name,
                snapshot=self.snapshot(fork=True),
                metadata=lineage,
            )
            return fork_id

    def _ensure_forkable(self) -> None:
        if self.runtime.interruptions.get_last_interrupt() is not None:
            raise SessionSerializationError("Resolve the pending interaction before forking")

    def _remember_record(
        self,
        store: BaseSessionStore,
        session_id: str,
        record: Mapping[str, Any],
    ) -> None:
        key = (id(store), session_id)
        self._revisions[key] = int(record["revision"])
        self._metadata[key] = deepcopy(dict(record["metadata"]))

    def _restore_record(
        self,
        session_id: str,
        store: BaseSessionStore,
        record: Mapping[str, Any] | None,
    ) -> SessionRestoreReport:
        if record is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        snapshot = dict(record["snapshot"])
        if snapshot.get("schemaVersion") != 4:
            raise SessionSerializationError("Explicitly migrate the legacy session to V4 first")

        report = SessionRestoreReport(session_id=session_id, agent_type=self.bindings.agent_type)
        history = snapshot.get("history")
        try:
            self.bindings.context().validate_state(history)
        except Exception as exc:
            raise SessionSerializationError("Invalid Context session state") from exc
        provider = history.get("providerName") if isinstance(history, dict) else None
        current_provider = self.bindings.provider_name()
        if provider and provider != current_provider:
            report.add_issue(
                component="history",
                code="provider_drift",
                message="Replay was rebuilt for a different provider",
                metadata={"savedProvider": provider, "currentProvider": current_provider},
            )

        operations: list[tuple[str, SessionState, SessionStateData]] = []
        unknown: dict[str, Any] = {}
        module_states = dict(snapshot.get("modules") or {})
        for name, raw_value in module_states.items():
            if name == "context":
                continue
            value = dict(raw_value)
            binding = self._participants.get(name)
            component = binding.resolve() if binding else None
            if component is None:
                if value.get("required"):
                    raise SessionSerializationError(f"Required session component is missing: {name}")
                unknown[name] = deepcopy(value)
                report.add_issue(
                    component=name,
                    code="component_missing",
                    message=f"Supply {name} explicitly to restore its state",
                )
                continue
            if value.get("version", 1) != binding.version:
                raise SessionSerializationError(f"Unsupported state version for {name}")
            if value.get("state") is not None:
                if isinstance(component, ValidatableSessionState):
                    component.validate_state(deepcopy(value["state"]))
                operations.append((name, component, deepcopy(value["state"])))

        for name, binding in self._participants.items():
            if name == "context" or module_states.get(name, {}).get("state") is not None:
                continue
            component = binding.resolve()
            if isinstance(component, ResettableSessionState):
                operations.append((name, component, None))

        context = self.bindings.context()
        saved_history = context.export_state()
        backups: list[tuple[SessionState, SessionStateData]] = []
        try:
            context.restore_state(history)
            for name, component, value in operations:
                backups.append((component, deepcopy(component.export_state())))
                outcome = (
                    component.reset_session_state()
                    if value is None
                    else component.restore_state(value)
                )
                if isinstance(outcome, ComponentRestoreReport):
                    report.extend_component(name, outcome.to_dict())
                elif isinstance(outcome, dict):
                    report.extend_component(name, outcome)
                elif outcome is not None:
                    raise TypeError(f"Invalid restore result from session participant {name!r}")
        except BaseException as error:
            rollback_errors = []
            try:
                context.restore_state(saved_history)
            except Exception as rollback_error:
                rollback_errors.append(f"context: {rollback_error}")
            for component, backup in reversed(backups):
                try:
                    component.restore_state(backup)
                except Exception as rollback_error:
                    rollback_errors.append(f"{type(component).__name__}: {rollback_error}")
            if rollback_errors:
                raise SessionSerializationError(
                    "Session restore failed; rollback was incomplete: " + "; ".join(rollback_errors)
                ) from error
            raise

        self._unknown = unknown
        self.runtime.session_id = session_id
        self.runtime.reasoning = snapshot.get("reasoning")
        self.runtime.current_task_id = snapshot.get("currentTaskId")
        self._remember_record(store, session_id, record)
        report.note_missing_tools(
            [name for name in snapshot.get("tools", []) if not self.bindings.has_tool(name)]
        )
        self.runtime.metamessages.reconcile_history()
        self.last_restored_snapshot = snapshot
        return report
