"""Connection lifecycle management for MCP runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .policy import MCPPolicyContext


class MCPPolicyError(RuntimeError):
    """Raised when MCP policy blocks an operation."""


@dataclass(slots=True)
class MCPConnectionState:
    server_name: str
    status: str = "disconnected"
    retry_count: int = 0
    last_operation: Optional[str] = None
    last_error: Optional[str] = None
    last_error_type: Optional[str] = None
    last_connected_at: Optional[float] = None
    last_disconnected_at: Optional[float] = None
    transport: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "serverName": self.server_name,
            "status": self.status,
            "retryCount": self.retry_count,
            "lastOperation": self.last_operation,
            "lastError": self.last_error,
            "lastErrorType": self.last_error_type,
            "lastConnectedAt": self.last_connected_at,
            "lastDisconnectedAt": self.last_disconnected_at,
            "transport": dict(self.transport),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MCPConnectionState":
        data = dict(payload or {})
        return cls(
            server_name=str(data.get("serverName") or "").strip() or "mcp",
            status=str(data.get("status") or "disconnected"),
            retry_count=int(data.get("retryCount", 0) or 0),
            last_operation=data.get("lastOperation"),
            last_error=data.get("lastError"),
            last_error_type=data.get("lastErrorType"),
            last_connected_at=data.get("lastConnectedAt"),
            last_disconnected_at=data.get("lastDisconnectedAt"),
            transport=dict(data.get("transport") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


class MCPConnectionManager:
    """Wrap one MCP client with lifecycle state, retry, and policy checks."""

    def __init__(
        self,
        *,
        client: Any,
        server_name: str,
        auto_connect: bool = True,
        max_retries: int = 1,
        persist_connection: bool = False,
        policy_context: Optional[MCPPolicyContext] = None,
    ):
        self.client = client
        self.auto_connect = auto_connect
        self.max_retries = max(int(max_retries or 0), 0)
        self.persist_connection = persist_connection
        self.policy_context = policy_context or MCPPolicyContext()
        self.state = MCPConnectionState(server_name=str(server_name or "mcp"))

    def classify_exception(self, exc: Exception) -> str:
        if isinstance(exc, MCPPolicyError):
            return "mcp_policy_denied"
        name = exc.__class__.__name__.lower()
        detail = str(exc).lower()
        if "auth" in name or "auth" in detail or "unauthorized" in detail or "forbidden" in detail:
            return "mcp_auth_error"
        if "timeout" in name or "timeout" in detail:
            return "mcp_timeout"
        if "connect" in name or "connect" in detail or "transport" in detail or "not connected" in detail:
            return "mcp_connection_error"
        return "mcp_operation_error"

    def describe_error(self, exc: Exception) -> dict[str, Any]:
        return {
            "errorType": self.classify_exception(exc),
            "message": str(exc),
            "serverName": self.state.server_name,
            "connectionState": self.state.to_dict(),
        }

    def describe_state(self) -> dict[str, Any]:
        return self.state.to_dict()

    async def connect(self) -> None:
        if self.client.is_connected():
            self.state.status = "connected"
            self._refresh_transport_info()
            return
        self.state.status = "connecting"
        await self.client.connect()
        self.state.status = "connected"
        self.state.last_error = None
        self.state.last_error_type = None
        self.state.last_connected_at = time.time()
        self._refresh_transport_info()

    async def disconnect(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        if self.client.is_connected():
            await self.client.disconnect(exc_type, exc_val, exc_tb)
        self.state.status = "disconnected"
        self.state.last_disconnected_at = time.time()
        self._refresh_transport_info()

    def _refresh_transport_info(self) -> None:
        info_getter = getattr(self.client, "get_transport_info", None)
        if callable(info_getter):
            try:
                self.state.transport = dict(info_getter() or {})
            except Exception:
                self.state.transport = {}

    async def run_operation(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        operation_name: str,
        capability_kind: str,
        capability_name: Optional[str] = None,
        auto_disconnect: Optional[bool] = None,
    ) -> Any:
        decision = self.policy_context.authorize(
            server_name=self.state.server_name,
            capability_kind=capability_kind,
            capability_name=capability_name,
        )
        self.state.last_operation = operation_name
        if not decision.allowed:
            self.state.status = "denied"
            self.state.last_error = decision.reason
            self.state.last_error_type = "mcp_policy_denied"
            raise MCPPolicyError(decision.reason)

        should_auto_connect = self.auto_connect and self.policy_context.allow_auto_connect
        transient_disconnect = (
            not self.persist_connection
            if auto_disconnect is None
            else bool(auto_disconnect)
        )

        attempts = self.max_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            self.state.retry_count = attempt - 1
            was_connected = self.client.is_connected()
            try:
                if not was_connected:
                    if not should_auto_connect:
                        raise RuntimeError("MCP client is not connected and current policy forbids auto-connect.")
                    await self.connect()
                result = await operation()
                self.state.status = "connected" if self.client.is_connected() else "disconnected"
                self.state.last_error = None
                self.state.last_error_type = None
                return result
            except Exception as exc:
                last_exc = exc
                error_type = self.classify_exception(exc)
                self.state.status = "error"
                self.state.last_error = str(exc)
                self.state.last_error_type = error_type
                if attempt >= attempts or error_type in {"mcp_policy_denied", "mcp_auth_error"}:
                    raise
                try:
                    if self.client.is_connected():
                        await self.disconnect()
                except Exception:
                    pass
            finally:
                if transient_disconnect and not was_connected and self.client.is_connected():
                    await self.disconnect()

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("MCP operation failed without a captured exception.")

    def export_state(self) -> dict[str, Any]:
        return {
            "autoConnect": self.auto_connect,
            "maxRetries": self.max_retries,
            "persistConnection": self.persist_connection,
            "state": self.state.to_dict(),
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        state = dict(payload or {})
        self.auto_connect = bool(state.get("autoConnect", self.auto_connect))
        self.max_retries = max(int(state.get("maxRetries", self.max_retries) or 0), 0)
        self.persist_connection = bool(state.get("persistConnection", self.persist_connection))
        restored_state = MCPConnectionState.from_dict(state.get("state"))
        if restored_state.server_name:
            self.state = restored_state
