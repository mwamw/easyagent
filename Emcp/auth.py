"""Authentication helpers for MCP transports."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


MCPAuthKind = Literal["none", "bearer", "basic", "header"]


@dataclass(slots=True)
class MCPAuthConfig:
    kind: MCPAuthKind = "none"
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    header_name: Optional[str] = None
    header_value: Optional[str] = None
    persist_credentials: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "MCPAuthConfig":
        if isinstance(value, MCPAuthConfig):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return cls()
            return cls(kind="bearer", token=stripped)
        if isinstance(value, tuple) and len(value) == 2:
            username, password = value
            return cls(
                kind="basic",
                username=str(username or ""),
                password=str(password or ""),
            )
        if not isinstance(value, dict):
            return cls(metadata={"raw": value})

        kind = str(value.get("kind") or value.get("type") or "none").strip().lower() or "none"
        metadata = dict(value.get("metadata") or {})
        return cls(
            kind=kind if kind in {"none", "bearer", "basic", "header"} else "none",
            token=value.get("token"),
            username=value.get("username"),
            password=value.get("password"),
            header_name=value.get("header_name") or value.get("headerName"),
            header_value=value.get("header_value") or value.get("headerValue"),
            persist_credentials=bool(value.get("persist_credentials", False)),
            metadata=metadata,
        )

    def is_configured(self) -> bool:
        if self.kind == "bearer":
            return bool(str(self.token or "").strip())
        if self.kind == "basic":
            return bool(str(self.username or "").strip())
        if self.kind == "header":
            return bool(str(self.header_name or "").strip())
        return False

    def build_headers(self) -> dict[str, str]:
        if self.kind == "none":
            return {}
        if self.kind == "bearer" and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        if self.kind == "basic":
            raw = f"{self.username or ''}:{self.password or ''}".encode("utf-8")
            encoded = base64.b64encode(raw).decode("ascii")
            return {"Authorization": f"Basic {encoded}"}
        if self.kind == "header" and self.header_name:
            return {str(self.header_name): str(self.header_value or "")}
        return {}

    def merge_into_transport_kwargs(self, transport_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(transport_kwargs or {})
        if not self.is_configured():
            return merged

        if self.kind == "basic" and "auth" not in merged:
            merged["auth"] = (str(self.username or ""), str(self.password or ""))

        headers = dict(merged.get("headers") or {})
        headers.update(self.build_headers())
        if headers:
            merged["headers"] = headers
        return merged

    def export_state(self, *, include_secrets: bool = False) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "persistCredentials": self.persist_credentials,
            "metadata": dict(self.metadata),
        }
        if include_secrets or self.persist_credentials:
            payload.update(
                {
                    "token": self.token,
                    "username": self.username,
                    "password": self.password,
                    "headerName": self.header_name,
                    "headerValue": self.header_value,
                }
            )
            return payload

        if self.username:
            payload["username"] = self.username
        if self.header_name:
            payload["headerName"] = self.header_name
        if self.token:
            payload["tokenPresent"] = True
        if self.password:
            payload["passwordPresent"] = True
        if self.header_value:
            payload["headerValuePresent"] = True
        return payload

    @classmethod
    def from_state(cls, payload: dict[str, Any] | None) -> "MCPAuthConfig":
        state = dict(payload or {})
        return cls(
            kind=str(state.get("kind") or "none").strip().lower() or "none",
            token=state.get("token"),
            username=state.get("username"),
            password=state.get("password"),
            header_name=state.get("headerName"),
            header_value=state.get("headerValue"),
            persist_credentials=bool(state.get("persistCredentials", False)),
            metadata=dict(state.get("metadata") or {}),
        )
