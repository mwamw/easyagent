"""Foreground and background process helpers for local command tools."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
from uuid import uuid4


CommandType = str | Sequence[str]


@dataclass(slots=True)
class ProcessExecutionResult:
    command: CommandType
    cwd: str
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(slots=True)
class BackgroundTaskSnapshot:
    task_id: str
    command: CommandType
    cwd: str
    status: str
    return_code: Optional[int]
    stdout: str
    stderr: str
    started_at: float
    finished_at: Optional[float] = None


@dataclass
class _ManagedBackgroundTask:
    task_id: str
    command: CommandType
    cwd: str
    process: subprocess.Popen[str]
    started_at: float
    stdout_chunks: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished_at: Optional[float] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    stop_requested: bool = False

    def append_stdout(self, chunk: str) -> None:
        with self.lock:
            self.stdout_chunks.append(chunk)

    def append_stderr(self, chunk: str) -> None:
        with self.lock:
            self.stderr_chunks.append(chunk)

    def snapshot(self) -> BackgroundTaskSnapshot:
        return_code = self.process.poll()
        if return_code is not None and self.finished_at is None:
            self.finished_at = time.time()
        if return_code is None:
            status = "running"
        elif self.stop_requested:
            status = "terminated"
        else:
            status = "completed"
        return BackgroundTaskSnapshot(
            task_id=self.task_id,
            command=self.command,
            cwd=self.cwd,
            status=status,
            return_code=return_code,
            stdout="".join(self.stdout_chunks),
            stderr="".join(self.stderr_chunks),
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class ProcessManager:
    def __init__(self, *, shell: str = "bash", max_background_tasks: int = 8):
        self.shell = shell
        self.max_background_tasks = max_background_tasks
        self._tasks: Dict[str, _ManagedBackgroundTask] = {}
        self._lock = threading.Lock()

    def run(
        self,
        command: CommandType,
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout_ms: Optional[int] = None,
        use_shell: Optional[bool] = None,
    ) -> ProcessExecutionResult:
        resolved_cwd = os.path.abspath(cwd or os.getcwd())
        use_shell = isinstance(command, str) if use_shell is None else use_shell
        timeout_s = None if timeout_ms is None else timeout_ms / 1000
        popen_kwargs = self._build_popen_kwargs(command, resolved_cwd, env, use_shell)
        try:
            completed = subprocess.run(
                **popen_kwargs,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return ProcessExecutionResult(
                command=command,
                cwd=resolved_cwd,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ProcessExecutionResult(
                command=command,
                cwd=resolved_cwd,
                return_code=-1,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )

    def start_background(
        self,
        command: CommandType,
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        use_shell: Optional[bool] = None,
    ) -> BackgroundTaskSnapshot:
        resolved_cwd = os.path.abspath(cwd or os.getcwd())
        use_shell = isinstance(command, str) if use_shell is None else use_shell
        with self._lock:
            active_count = sum(1 for task in self._tasks.values() if task.process.poll() is None)
            if active_count >= self.max_background_tasks:
                raise RuntimeError("后台任务数量已达到上限。")

        popen_kwargs = self._build_popen_kwargs(command, resolved_cwd, env, use_shell)
        process = subprocess.Popen(
            **popen_kwargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        task = _ManagedBackgroundTask(
            task_id=f"task_{uuid4().hex[:12]}",
            command=command,
            cwd=resolved_cwd,
            process=process,
            started_at=time.time(),
        )
        task.stdout_thread = self._start_reader_thread(task, stream_name="stdout")
        task.stderr_thread = self._start_reader_thread(task, stream_name="stderr")
        with self._lock:
            self._tasks[task.task_id] = task
        return task.snapshot()

    def get_task(
        self,
        task_id: str,
        *,
        block: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> BackgroundTaskSnapshot:
        task = self._get_managed_task(task_id)
        if block:
            self._wait_for_task(task, timeout_ms=timeout_ms)
        return task.snapshot()

    def get_output(
        self,
        task_id: str,
        *,
        block: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> ProcessExecutionResult:
        snapshot = self.get_task(task_id, block=block, timeout_ms=timeout_ms)
        return ProcessExecutionResult(
            command=snapshot.command,
            cwd=snapshot.cwd,
            return_code=snapshot.return_code if snapshot.return_code is not None else 0,
            stdout=snapshot.stdout,
            stderr=snapshot.stderr,
            timed_out=False,
        )

    def stop(self, task_id: str, *, kill: bool = False) -> BackgroundTaskSnapshot:
        task = self._get_managed_task(task_id)
        if task.process.poll() is None:
            task.stop_requested = True
            if kill:
                task.process.kill()
            else:
                task.process.terminate()
            try:
                task.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                task.process.kill()
                task.process.wait(timeout=5)
        self._finalize_task(task)
        return task.snapshot()

    def list_tasks(self) -> list[BackgroundTaskSnapshot]:
        with self._lock:
            return [task.snapshot() for task in self._tasks.values()]

    def close(self) -> None:
        for snapshot in self.list_tasks():
            if snapshot.status == "running":
                self.stop(snapshot.task_id, kill=True)

    def _get_managed_task(self, task_id: str) -> _ManagedBackgroundTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"后台任务不存在: {task_id}")
        return task

    def _wait_for_task(
        self,
        task: _ManagedBackgroundTask,
        *,
        timeout_ms: Optional[int] = None,
    ) -> None:
        if task.process.poll() is not None:
            self._finalize_task(task)
            return

        timeout_s = None if timeout_ms is None or timeout_ms <= 0 else timeout_ms / 1000
        try:
            task.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return
        self._finalize_task(task)

    def _finalize_task(self, task: _ManagedBackgroundTask) -> None:
        task.finished_at = task.finished_at or time.time()
        self._join_reader_thread(task.stdout_thread)
        self._join_reader_thread(task.stderr_thread)

    @staticmethod
    def _join_reader_thread(thread: Optional[threading.Thread]) -> None:
        if thread is None:
            return
        thread.join(timeout=0.2)

    def _build_popen_kwargs(
        self,
        command: CommandType,
        cwd: str,
        env: Optional[dict[str, str]],
        use_shell: bool,
    ) -> dict[str, Any]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": merged_env,
            "shell": use_shell,
        }
        if use_shell:
            kwargs["args"] = command if isinstance(command, str) else " ".join(command)
            kwargs["executable"] = self.shell
        else:
            if isinstance(command, str):
                kwargs["args"] = command.split()
            else:
                kwargs["args"] = list(command)
        return kwargs

    def _start_reader_thread(self, task: _ManagedBackgroundTask, *, stream_name: str) -> threading.Thread:
        stream = getattr(task.process, stream_name)
        assert stream is not None

        def _reader() -> None:
            for chunk in stream:
                if stream_name == "stdout":
                    task.append_stdout(chunk)
                else:
                    task.append_stderr(chunk)
            stream.close()

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        return thread
