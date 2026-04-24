"""MCP runtime manager for EasyAgent."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional, Protocol

from Emcp import MCPClient

from .auth import MCPAuthConfig
from .cache import MCPCapabilitySnapshot, MCPServerCache
from .connection_manager import MCPConnectionManager
from .policy import MCPPolicyContext


class MCPClientProtocol(Protocol):
    """Protocol implemented by MCP client adapters."""

    def is_connected(self) -> bool:
        ...

    async def connect(self) -> None:
        ...

    async def disconnect(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        ...

    async def list_tools(self) -> List[Dict[str, Any]]:
        ...

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        ...

    async def list_resources(self) -> List[Dict[str, Any]]:
        ...

    async def read_resource(self, uri: str) -> Any:
        ...

    async def list_prompts(self) -> List[Dict[str, Any]]:
        ...

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        ...


def run_coroutine_in_thread(coro: Any) -> Any:
    """Fallback helper for sync calls made from inside a running event loop."""

    result_holder: Dict[str, Any] = {"value": None, "error": None}

    def _runner() -> None:
        try:
            result_holder["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - passthrough wrapper
            result_holder["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if result_holder["error"] is not None:
        raise result_holder["error"]
    return result_holder["value"]


MCPServerSnapshot = MCPCapabilitySnapshot


def _export_server_source(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, list, dict)):
        return {
            "value": value,
            "restorable": True,
            "kind": type(value).__name__,
        }
    return {
        "value": str(value),
        "restorable": False,
        "kind": type(value).__name__,
    }


class MCPRuntimeManager:
    """Shared MCP runtime manager.

    It owns one MCP client lifecycle, policy, cache, and capability snapshot.
    """

    def __init__(
        self,
        server_source: Any,
        server_args: Optional[List[str]] = None,
        transport_type: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        tool_prefix: str = "",
        auto_connect: bool = True,
        client: Optional[MCPClientProtocol] = None,
        auth_config: Optional[Any] = None,
        policy_context: Optional[MCPPolicyContext] = None,
        cache_store: Optional[MCPServerCache] = None,
        connection_manager: Optional[MCPConnectionManager] = None,
        max_retries: Optional[int] = None,
        persist_connection: Optional[bool] = None,
        **transport_kwargs: Any,
    ):
        self.server_source = server_source
        self.server_args = list(server_args or [])
        self.transport_type = transport_type
        self.env = dict(env or {})
        self.tool_prefix = tool_prefix
        self.auto_connect = auto_connect
        self.transport_kwargs = dict(transport_kwargs)
        self.auth_config = MCPAuthConfig.from_value(auth_config)
        self.policy_context = MCPPolicyContext.from_value(policy_context)
        self.cache = cache_store or MCPServerCache(
            capability_ttl_seconds=self.policy_context.capability_cache_ttl_seconds,
            resource_ttl_seconds=self.policy_context.resource_cache_ttl_seconds,
            prompt_ttl_seconds=self.policy_context.prompt_cache_ttl_seconds,
        )
        self._sync_loop: Optional[asyncio.AbstractEventLoop] = None

        client_kwargs = self.auth_config.merge_into_transport_kwargs(self.transport_kwargs)
        self.client: MCPClientProtocol = client or MCPClient(
            server_source=server_source,
            server_args=self.server_args,
            transport_type=transport_type,
            env=self.env,
            **client_kwargs,
        )
        self.connection_manager = connection_manager or MCPConnectionManager(
            client=self.client,
            server_name=self.server_label,
            auto_connect=self.auto_connect,
            max_retries=(
                self.policy_context.max_retries
                if max_retries is None
                else max_retries
            ),
            persist_connection=(
                self.policy_context.persist_connection
                if persist_connection is None
                else bool(persist_connection)
            ),
            policy_context=self.policy_context,
        )

    @property
    def server_label(self) -> str:
        value = str(self.server_source)
        value = value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        value = value.rsplit(".", 1)[0]
        normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")
        return normalized or "mcp"

    @property
    def source_identifier(self) -> str:
        return f"mcp://{self.server_label}"

    def connect(self) -> None:
        self._run_sync(self.connection_manager.connect())

    def close(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        status = "closed"
        try:
            self._run_sync(self.connection_manager.disconnect())
        except Exception as exc:
            status = "degraded"
            issues.append(
                {
                    "component": "mcp_runtime",
                    "code": "disconnect_failed",
                    "message": f"关闭 MCP 连接失败: {exc}",
                    "severity": "warning",
                }
            )
        self._close_sync_loop()
        return {
            "status": status,
            "metadata": {
                "serverName": self.server_label,
                "sourceIdentifier": self.source_identifier,
                "connectionState": self.connection_manager.describe_state(),
            },
            "issues": issues,
        }

    def _run_sync(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = self._ensure_sync_loop()
            return loop.run_until_complete(coro)
        return run_coroutine_in_thread(coro)

    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        if self._sync_loop is None or self._sync_loop.is_closed():
            self._sync_loop = asyncio.new_event_loop()
        return self._sync_loop

    def _close_sync_loop(self) -> None:
        if self._sync_loop is None:
            return
        self._sync_loop.close()
        self._sync_loop = None

    def ensure_connected(self) -> None:
        if self.client.is_connected():
            return
        if not self.auto_connect or not self.policy_context.allow_auto_connect:
            raise RuntimeError("MCP client is not connected.")
        self.connect()

    async def _run_with_connection(
        self,
        operation,
        *,
        operation_name: str,
        capability_kind: str,
        capability_name: Optional[str] = None,
        auto_disconnect: Optional[bool] = None,
    ):
        return await self.connection_manager.run_operation(
            operation,
            operation_name=operation_name,
            capability_kind=capability_kind,
            capability_name=capability_name,
            auto_disconnect=auto_disconnect,
        )

    def list_remote_tools(self) -> List[Dict[str, Any]]:
        tools = self._run_sync(self.alist_remote_tools())
        self.cache.update_capabilities(
            server_name=self.server_label,
            source_identifier=self.source_identifier,
            tools=list(tools),
        )
        return tools

    async def alist_remote_tools(self) -> List[Dict[str, Any]]:
        return await self._run_with_connection(
            self.client.list_tools,
            operation_name="list_tools",
            capability_kind="tool_list",
        )

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return self._run_sync(self.aexecute_tool(tool_name, arguments))

    async def aexecute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return await self._run_with_connection(
            lambda: self.client.call_tool(tool_name, arguments),
            operation_name=f"call_tool:{tool_name}",
            capability_kind="tool",
            capability_name=tool_name,
        )

    def list_remote_resources(self) -> List[Dict[str, Any]]:
        resources = self._run_sync(self.alist_remote_resources())
        self.cache.update_capabilities(
            server_name=self.server_label,
            source_identifier=self.source_identifier,
            resources=list(resources),
        )
        return resources

    async def alist_remote_resources(self) -> List[Dict[str, Any]]:
        return await self._run_with_connection(
            self.client.list_resources,
            operation_name="list_resources",
            capability_kind="resource_list",
        )

    def read_remote_resource(self, uri: str, *, refresh: bool = False) -> Any:
        return self._run_sync(self.aread_remote_resource(uri, refresh=refresh))

    async def aread_remote_resource(self, uri: str, *, refresh: bool = False) -> Any:
        if not refresh:
            cached = self.cache.get_resource(uri)
            if cached is not None:
                return cached
        result = await self._run_with_connection(
            lambda: self.client.read_resource(uri),
            operation_name=f"read_resource:{uri}",
            capability_kind="resource_read",
            capability_name=uri,
        )
        self.cache.set_resource(uri, result)
        return result

    def list_remote_prompts(self) -> List[Dict[str, Any]]:
        prompts = self._run_sync(self.alist_remote_prompts())
        self.cache.update_capabilities(
            server_name=self.server_label,
            source_identifier=self.source_identifier,
            prompts=list(prompts),
        )
        return prompts

    async def alist_remote_prompts(self) -> List[Dict[str, Any]]:
        return await self._run_with_connection(
            self.client.list_prompts,
            operation_name="list_prompts",
            capability_kind="prompt_list",
        )

    def get_remote_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
        *,
        refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._run_sync(self.aget_remote_prompt(prompt_name, arguments, refresh=refresh))

    async def aget_remote_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
        *,
        refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        if not refresh:
            cached = self.cache.get_prompt(prompt_name, arguments)
            if cached is not None:
                return cached
        result = await self._run_with_connection(
            lambda: self.client.get_prompt(prompt_name, arguments or {}),
            operation_name=f"get_prompt:{prompt_name}",
            capability_kind="prompt_get",
            capability_name=prompt_name,
        )
        self.cache.set_prompt(prompt_name, arguments, result)
        return result

    def snapshot(self, *, refresh: bool = False) -> MCPServerSnapshot:
        snapshot = self.cache.get_capability_snapshot()
        if snapshot is None:
            snapshot = self.cache.update_capabilities(
                server_name=self.server_label,
                source_identifier=self.source_identifier,
            )
        if refresh or not snapshot.tools:
            try:
                snapshot = self.cache.update_capabilities(
                    server_name=self.server_label,
                    source_identifier=self.source_identifier,
                    tools=list(self.list_remote_tools()),
                )
            except Exception:
                pass
        if refresh or not snapshot.resources:
            try:
                snapshot = self.cache.update_capabilities(
                    server_name=self.server_label,
                    source_identifier=self.source_identifier,
                    resources=list(self.list_remote_resources()),
                )
            except Exception:
                pass
        if refresh or not snapshot.prompts:
            try:
                snapshot = self.cache.update_capabilities(
                    server_name=self.server_label,
                    source_identifier=self.source_identifier,
                    prompts=list(self.list_remote_prompts()),
                )
            except Exception:
                pass
        return snapshot

    def connection_state(self) -> dict[str, Any]:
        return self.connection_manager.describe_state()

    def describe_error(self, exc: Exception) -> dict[str, Any]:
        return self.connection_manager.describe_error(exc)

    def export_state(self) -> dict[str, Any]:
        snapshot = self.snapshot(refresh=False)
        return {
            "schemaVersion": 1,
            "serverName": self.server_label,
            "sourceIdentifier": self.source_identifier,
            "serverSource": _export_server_source(self.server_source),
            "serverArgs": list(self.server_args),
            "transportType": self.transport_type,
            "env": dict(self.env),
            "transportKwargs": dict(self.transport_kwargs),
            "toolPrefix": self.tool_prefix,
            "autoConnect": self.auto_connect,
            "auth": self.auth_config.export_state(),
            "policy": self.policy_context.export_state(),
            "cache": self.cache.export_state(),
            "capabilitySnapshot": snapshot.to_dict(),
            "connection": self.connection_manager.export_state(),
        }

    def restore_state(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(payload or {})
        restored_items: list[str] = []
        degraded_items: list[str] = []
        issues: list[dict[str, Any]] = []

        policy_state = state.get("policy")
        if policy_state:
            self.policy_context.restore_state(policy_state)
            restored_items.append("policy")

        cache_state = state.get("cache")
        if cache_state:
            self.cache.restore_state(cache_state)
            restored_items.append("cache")

        connection_state = state.get("connection")
        if connection_state:
            self.connection_manager.restore_state(connection_state)
            restored_items.append("connection")

        capability_snapshot = state.get("capabilitySnapshot")
        if capability_snapshot and self.cache.capability_snapshot is None:
            restored_snapshot = MCPCapabilitySnapshot.from_dict(capability_snapshot)
            if restored_snapshot is not None:
                self.cache.capability_snapshot = restored_snapshot
                restored_items.append("capability_snapshot")
            else:
                degraded_items.append("capability_snapshot")
                issues.append(
                    {
                        "code": "capability_snapshot_restore_failed",
                        "message": "MCP capability snapshot 快照存在，但解析失败。",
                        "severity": "warning",
                    }
                )

        return {
            "status": "restored" if not degraded_items else "degraded",
            "restoredItems": restored_items,
            "degradedItems": degraded_items,
            "metadata": {
                "serverName": self.server_label,
                "sourceIdentifier": self.source_identifier,
            },
            "issues": issues,
        }


class MCPHub:
    """Registry for multiple MCP runtime managers keyed by logical server name."""

    def __init__(self):
        self._managers: Dict[str, MCPRuntimeManager] = {}
        self._lock = threading.RLock()

    @staticmethod
    def normalize_server_name(value: str) -> str:
        normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")
        return normalized or "mcp"

    def register_manager(
        self,
        manager: MCPRuntimeManager,
        *,
        server_name: Optional[str] = None,
        replace: bool = False,
    ) -> str:
        resolved_name = self.normalize_server_name(server_name or manager.server_label)
        with self._lock:
            existing = self._managers.get(resolved_name)
            if existing is not None and existing is not manager and not replace:
                raise ValueError(f"MCP server '{resolved_name}' 已存在。")
            self._managers[resolved_name] = manager
        return resolved_name

    def unregister_manager(self, server_name: str) -> MCPRuntimeManager:
        resolved_name = self.normalize_server_name(server_name)
        with self._lock:
            try:
                return self._managers.pop(resolved_name)
            except KeyError as exc:
                raise KeyError(f"未知 MCP server: {resolved_name}") from exc

    def get_manager(self, server_name: str) -> MCPRuntimeManager:
        resolved_name = self.normalize_server_name(server_name)
        with self._lock:
            try:
                return self._managers[resolved_name]
            except KeyError as exc:
                available = ", ".join(sorted(self._managers.keys()))
                detail = f"未知 MCP server: {resolved_name}"
                if available:
                    detail += f"。可用 server: {available}"
                raise KeyError(detail) from exc

    def list_servers(self) -> List[str]:
        with self._lock:
            return sorted(self._managers.keys())

    def list_resources(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if server_name:
            manager = self.get_manager(server_name)
            resolved_name = self.normalize_server_name(server_name)
            return [
                {
                    "server": resolved_name,
                    **resource,
                }
                for resource in manager.list_remote_resources()
            ]

        aggregated: List[Dict[str, Any]] = []
        for current_name in self.list_servers():
            manager = self.get_manager(current_name)
            for resource in manager.list_remote_resources():
                aggregated.append(
                    {
                        "server": current_name,
                        **resource,
                    }
                )
        return aggregated

    def read_resource(self, server_name: str, uri: str) -> Any:
        manager = self.get_manager(server_name)
        return manager.read_remote_resource(uri)

    def export_state(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "servers": [
                {
                    "serverName": server_name,
                    "manager": manager.export_state(),
                }
                for server_name, manager in sorted(self._managers.items())
            ],
        }

    def restore_state(
        self,
        payload: dict[str, Any] | None,
        *,
        manager_factory,
        client_overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        restored_items: list[str] = []
        degraded_items: list[str] = []
        overrides = dict(client_overrides or {})
        for item in list(dict(payload or {}).get("servers") or []):
            server_name = str(item.get("serverName") or "").strip()
            manager_payload = dict(item.get("manager") or {})
            client = overrides.get(server_name)
            try:
                manager = manager_factory(manager_payload, client=client)
                self.register_manager(manager, server_name=server_name, replace=True)
                restored_items.append(server_name)
            except Exception:
                degraded_items.append(server_name)
        return {
            "status": "restored" if not degraded_items else "degraded",
            "restoredItems": restored_items,
            "degradedItems": degraded_items,
            "metadata": {"serverCount": len(restored_items)},
        }

    @classmethod
    def from_state(
        cls,
        payload: dict[str, Any] | None,
        *,
        manager_factory,
        client_overrides: Optional[dict[str, Any]] = None,
    ) -> "MCPHub":
        hub = cls()
        hub.restore_state(
            payload,
            manager_factory=manager_factory,
            client_overrides=client_overrides,
        )
        return hub

    def close(self) -> dict[str, Any]:
        report = {
            "status": "closed",
            "metadata": {"serverCount": len(self.list_servers())},
            "issues": [],
        }
        for server_name in self.list_servers():
            manager = self.get_manager(server_name)
            try:
                close_report = manager.close()
                if close_report.get("status") == "degraded" and report["status"] == "closed":
                    report["status"] = "degraded"
            except Exception as exc:
                report["status"] = "degraded"
                report["issues"].append(
                    {
                        "component": "mcp_runtime",
                        "code": "manager_close_failed",
                        "message": f"关闭 MCP server `{server_name}` 失败: {exc}",
                        "severity": "warning",
                    }
                )
        return report
