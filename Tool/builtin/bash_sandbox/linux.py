"""Linux Bubblewrap command construction for BashTool."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

from .config import (
    BashSandboxConfig,
    SandboxedCommand,
    SandboxDecision,
    SandboxNetworkMode,
    SandboxStatus,
)
from .errors import (
    SandboxBypassDeniedError,
    SandboxConfigurationError,
    SandboxPathError,
    SandboxUnavailableError,
)


class LinuxBubblewrapSandbox:
    """Bash execution backend; this is not an Agent module."""

    def __init__(
        self,
        config: BashSandboxConfig,
        *,
        workspace_root: str,
        allowed_roots: tuple[str, ...] = (),
        shell: str = "bash",
    ) -> None:
        self.config = config
        self.workspace_root = self._existing_realpath(workspace_root, "workspace_root")
        self.allowed_roots = tuple(
            self._existing_realpath(path, "allowed_roots") for path in allowed_roots
        )
        self.shell = shell
        self._temp_dir: str | None = None
        self._status: SandboxStatus | None = None

    def status(self) -> SandboxStatus:
        if self._status is not None:
            return self._status
        errors: list[str] = []
        warnings: list[str] = []
        executable = self.config.bubblewrap_path or shutil.which("bwrap")
        if sys.platform != "linux":
            errors.append("Linux Bubblewrap sandbox requires Linux")
        if executable is None or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
            errors.append("bubblewrap executable was not found")
            executable = None
        userns = Path("/proc/sys/kernel/unprivileged_userns_clone")
        if userns.exists():
            try:
                if userns.read_text(encoding="utf-8").strip() == "0":
                    errors.append("unprivileged user namespaces are disabled")
            except OSError as exc:
                warnings.append(f"could not inspect user namespace setting: {exc}")
        if self.config.enabled and not errors and executable is not None:
            probe = [
                executable, "--new-session", "--die-with-parent",
                "--ro-bind", "/", "/", "--dev", "/dev",
                "--unshare-pid", "--proc", "/proc",
            ]
            if self.config.network == SandboxNetworkMode.DENY:
                probe.append("--unshare-net")
            probe.extend(["--", "/bin/true"])
            try:
                completed = subprocess.run(
                    probe, capture_output=True, text=True, timeout=3, check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    errors.append(f"bubblewrap probe failed: {detail or completed.returncode}")
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"bubblewrap probe failed: {exc}")
        self._status = SandboxStatus(
            configured=self.config.enabled,
            available=not errors,
            platform=sys.platform,
            executable=executable,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )
        return self._status

    def decide(self, command: str, *, dangerously_disable: bool = False) -> SandboxDecision:
        if not self.config.enabled:
            return SandboxDecision(False, False, self.config.mode, reason="sandbox disabled")
        if dangerously_disable:
            if not self.config.allow_unsandboxed_commands:
                raise SandboxBypassDeniedError("Sandbox bypass is disabled by Bash sandbox policy")
            return SandboxDecision(
                True, False, self.config.mode, bypassed=True,
                reason="dangerouslyDisableSandbox requested",
            )
        if self._is_excluded(command):
            if not self.config.allow_unsandboxed_commands:
                raise SandboxBypassDeniedError(
                    "Command is excluded from sandboxing and unsandboxed execution is disabled"
                )
            return SandboxDecision(
                True, False, self.config.mode, fallback=True,
                reason="command matched excluded_commands",
            )
        status = self.status()
        if not status.available:
            reason = "; ".join(status.errors) or "sandbox unavailable"
            if self.config.fail_if_unavailable or not self.config.allow_unsandboxed_commands:
                raise SandboxUnavailableError(reason)
            return SandboxDecision(
                True, False, self.config.mode, fallback=True, reason=reason,
            )
        return SandboxDecision(True, True, self.config.mode)

    def wrap(self, command: str, *, cwd: str) -> SandboxedCommand:
        status = self.status()
        if not status.available or status.executable is None:
            raise SandboxUnavailableError("; ".join(status.errors) or "sandbox unavailable")
        cwd = self._existing_realpath(cwd, "cwd")
        write_paths = self._write_paths()
        if not self._within(cwd, write_paths):
            raise SandboxPathError("Bash cwd must be within a sandbox writable root")

        args: list[str] = [status.executable, "--new-session", "--die-with-parent"]
        args.extend(["--ro-bind", "/", "/"])
        for path in write_paths:
            args.extend(["--bind", path, path])

        deny_write = list(self._resolved_config_paths(self.config.deny_write, "deny_write"))
        if self.config.protect_git:
            for root in write_paths:
                for relative in (".git/hooks", ".git/config"):
                    candidate = os.path.join(root, relative)
                    if os.path.exists(candidate):
                        deny_write.append(os.path.realpath(candidate))
        for path in self._dedupe_nested(deny_write):
            if self._within(path, write_paths):
                args.extend(["--ro-bind", path, path])

        deny_read = self._resolved_config_paths(self.config.deny_read, "deny_read")
        allow_read = self._resolved_config_paths(self.config.allow_read, "allow_read")
        for path in deny_read:
            if os.path.isdir(path):
                args.extend(["--tmpfs", path])
                for allowed in allow_read:
                    if self._is_within_path(allowed, path):
                        args.extend(["--ro-bind", allowed, allowed])
            else:
                args.extend(["--ro-bind", "/dev/null", path])

        args.extend(["--dev", "/dev", "--unshare-pid", "--proc", "/proc"])
        if self.config.network == SandboxNetworkMode.DENY:
            args.append("--unshare-net")

        temp_dir = self._sandbox_temp_dir()
        args.extend(["--bind", temp_dir, temp_dir])
        args.extend(["--setenv", "SANDBOX_RUNTIME", "1"])
        args.extend(["--setenv", "TMPDIR", temp_dir])
        args.extend(["--chdir", cwd])

        shell = shutil.which(self.shell)
        if shell is None:
            raise SandboxConfigurationError(f"Shell executable was not found: {self.shell}")
        args.extend(["--", shell, "-c", command])
        return SandboxedCommand(tuple(args), cwd, {"SANDBOX_RUNTIME": "1", "TMPDIR": temp_dir})

    def close(self) -> None:
        if self._temp_dir is None:
            return
        try:
            shutil.rmtree(self._temp_dir)
        finally:
            self._temp_dir = None

    def _sandbox_temp_dir(self) -> str:
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix=f"easyagent-sandbox-{os.getpid()}-{uuid4().hex[:6]}-")
            os.chmod(self._temp_dir, 0o700)
        return self._temp_dir

    def _write_paths(self) -> tuple[str, ...]:
        paths = [self.workspace_root, *self.allowed_roots]
        paths.extend(self._resolved_config_paths(self.config.allow_write, "allow_write"))
        return tuple(self._dedupe_nested(paths))

    def _resolved_config_paths(self, paths: tuple[str, ...], field: str) -> tuple[str, ...]:
        resolved = []
        for value in paths:
            expanded = os.path.expanduser(value)
            if not os.path.isabs(expanded):
                expanded = os.path.join(self.workspace_root, expanded)
            expanded = os.path.abspath(expanded)
            if any(char in expanded for char in "*?["):
                raise SandboxPathError(f"Linux sandbox does not accept glob paths in {field}: {value}")
            real = self._existing_realpath(expanded, field)
            if real != expanded:
                raise SandboxPathError(
                    f"Sandbox {field} path contains a symbolic-link component: {expanded} -> {real}"
                )
            resolved.append(real)
        return tuple(self._dedupe_nested(resolved))

    def _is_excluded(self, command: str) -> bool:
        stripped = command.strip()
        return any(
            fnmatch.fnmatchcase(stripped, pattern)
            or stripped == pattern
            or stripped.startswith(pattern + " ")
            for pattern in self.config.excluded_commands
        )

    @staticmethod
    def _existing_realpath(path: str, field: str) -> str:
        expanded = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.exists(expanded):
            raise SandboxPathError(f"Sandbox {field} path does not exist: {expanded}")
        return os.path.realpath(expanded)

    @staticmethod
    def _is_within_path(path: str, root: str) -> bool:
        try:
            return os.path.commonpath((path, root)) == root
        except ValueError:
            return False

    @classmethod
    def _within(cls, path: str, roots: tuple[str, ...] | list[str]) -> bool:
        return any(cls._is_within_path(path, root) for root in roots)

    @classmethod
    def _dedupe_nested(cls, paths: list[str] | tuple[str, ...]) -> list[str]:
        unique = sorted(set(paths), key=lambda value: (len(value), value))
        kept: list[str] = []
        for path in unique:
            if not cls._within(path, kept):
                kept.append(path)
        return kept
