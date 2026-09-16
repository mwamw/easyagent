"""Typed configuration and execution facts for the Bash Linux sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BashSandboxMode(str, Enum):
    REGULAR = "regular"
    AUTO_ALLOW = "auto_allow"


class SandboxNetworkMode(str, Enum):
    INHERIT = "inherit"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class BashSandboxConfig:
    enabled: bool = False
    mode: BashSandboxMode = BashSandboxMode.REGULAR
    allow_unsandboxed_commands: bool = True
    fail_if_unavailable: bool = False
    network: SandboxNetworkMode = SandboxNetworkMode.DENY
    allow_write: tuple[str, ...] = ()
    deny_write: tuple[str, ...] = ()
    deny_read: tuple[str, ...] = ()
    allow_read: tuple[str, ...] = ()
    excluded_commands: tuple[str, ...] = ()
    protect_git: bool = True
    bubblewrap_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", BashSandboxMode(self.mode))
        object.__setattr__(self, "network", SandboxNetworkMode(self.network))
        for name in (
            "allow_write", "deny_write", "deny_read", "allow_read",
            "excluded_commands",
        ):
            values = tuple(str(item).strip() for item in getattr(self, name) if str(item).strip())
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class SandboxStatus:
    configured: bool
    available: bool
    platform: str
    backend: str = "bubblewrap"
    executable: str | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "available": self.available,
            "platform": self.platform,
            "backend": self.backend,
            "executable": self.executable,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SandboxDecision:
    configured: bool
    applied: bool
    mode: BashSandboxMode
    backend: str = "bubblewrap"
    bypassed: bool = False
    fallback: bool = False
    reason: str = ""

    @property
    def auto_allow(self) -> bool:
        return self.applied and self.mode == BashSandboxMode.AUTO_ALLOW

    def to_dict(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "applied": self.applied,
            "mode": self.mode.value,
            "backend": self.backend,
            "bypassed": self.bypassed,
            "fallback": self.fallback,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SandboxedCommand:
    argv: tuple[str, ...]
    cwd: str
    environment: dict[str, str] = field(default_factory=dict)

