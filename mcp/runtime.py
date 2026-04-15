"""MCP runtime manager for EasyAgent.

This module centralizes MCP client lifecycle and capability discovery so
tools, resources, and prompts can share the same runtime surface.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from mcp import MCPClient


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
    """Fallback helper for sync calls made from inside a running event loop.

    This keeps compatibility for sync wrappers, but the preferred persistent
    path is `asyncio.Runner` in threads without an active loop.
    """

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


@dataclass(slots=True)
class MCPServerSnapshot:
    """Cached MCP capability snapshot."""

    tools: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)


class MCPRuntimeManager:
    """Shared MCP runtime manager.

    It owns the client lifecycle and exposes a uniform API for tools,
    resources, and prompts.
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
        **transport_kwargs: Any,
    ):
        self.server_source = server_source
        self.server_args = server_args
        self.transport_type = transport_type
        self.env = env
        self.tool_prefix = tool_prefix
        self.auto_connect = auto_connect
        self._snapshot = MCPServerSnapshot()
        self._sync_loop: Optional[asyncio.AbstractEventLoop] = None

        self.client: MCPClientProtocol = client or MCPClient(
            server_source=server_source,
            server_args=server_args,
            transport_type=transport_type,
            env=env,
            **transport_kwargs,
        )

    @property
    def server_label(self) -> str:
        value = str(self.server_source)
        value = value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        value = value.rsplit(".", 1)[0]
        normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")
        return normalized or "mcp"

    def connect(self) -> None:
        if self.client.is_connected():
            return
        self._run_sync(self.client.connect())

    def close(self) -> None:
        if self.client.is_connected():
            self._run_sync(self.client.disconnect())
        self._close_sync_loop()

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
        if not self.auto_connect:
            raise RuntimeError("MCP client is not connected.")
        self.connect()

    async def _run_with_connection(self, operation):
        if not self.auto_connect and not self.client.is_connected():
            raise RuntimeError("MCP client is not connected.")

        if self.auto_connect:
            await self.client.connect()
            try:
                return await operation()
            finally:
                await self.client.disconnect()

        return await operation()

    def list_remote_tools(self) -> List[Dict[str, Any]]:
        tools = self._run_sync(self.alist_remote_tools())
        self._snapshot.tools = list(tools)
        return tools

    async def alist_remote_tools(self) -> List[Dict[str, Any]]:
        return await self._run_with_connection(self.client.list_tools)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return self._run_sync(self.aexecute_tool(tool_name, arguments))

    async def aexecute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return await self._run_with_connection(
            lambda: self.client.call_tool(tool_name, arguments)
        )

    def list_remote_resources(self) -> List[Dict[str, Any]]:
        resources = self._run_sync(self.alist_remote_resources())
        self._snapshot.resources = list(resources)
        return resources

    async def alist_remote_resources(self) -> List[Dict[str, Any]]:
        return await self._run_with_connection(self.client.list_resources)

    def read_remote_resource(self, uri: str) -> Any:
        return self._run_sync(self.aread_remote_resource(uri))

    async def aread_remote_resource(self, uri: str) -> Any:
        return await self._run_with_connection(lambda: self.client.read_resource(uri))

    def list_remote_prompts(self) -> List[Dict[str, Any]]:
        prompts = self._run_sync(self.alist_remote_prompts())
        self._snapshot.prompts = list(prompts)
        return prompts

    async def alist_remote_prompts(self) -> List[Dict[str, Any]]:
        return await self._run_with_connection(self.client.list_prompts)

    def get_remote_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        return self._run_sync(self.aget_remote_prompt(prompt_name, arguments))

    async def aget_remote_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        return await self._run_with_connection(
            lambda: self.client.get_prompt(prompt_name, arguments or {})
        )

    def snapshot(self, *, refresh: bool = False) -> MCPServerSnapshot:
        if refresh or not self._snapshot.tools:
            try:
                self._snapshot.tools = list(self.list_remote_tools())
            except Exception:
                pass
        if refresh or not self._snapshot.resources:
            try:
                self._snapshot.resources = list(self.list_remote_resources())
            except Exception:
                pass
        if refresh or not self._snapshot.prompts:
            try:
                self._snapshot.prompts = list(self.list_remote_prompts())
            except Exception:
                pass
        return MCPServerSnapshot(
            tools=list(self._snapshot.tools),
            resources=list(self._snapshot.resources),
            prompts=list(self._snapshot.prompts),
        )


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
