"""
MCP client adapter for EasyAgent.

This module wraps fastmcp client usage and normalizes transport handling.
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import Any, Dict, List, Optional, Tuple
import inspect

FASTMCP_AVAILABLE = importlib.util.find_spec("fastmcp") is not None

from logging import getLogger
logger = getLogger(__name__)
def _load_fastmcp_classes() -> Tuple[type, type, type, type, type]:
    from fastmcp import Client, FastMCP  # type: ignore[import-not-found]
    from fastmcp.client.transports import (  # type: ignore[import-not-found]
        PythonStdioTransport,
        SSETransport,
        StreamableHttpTransport,
    )

    return Client, FastMCP, PythonStdioTransport, SSETransport, StreamableHttpTransport


class MCPClient:
    """MCP client with multi-transport support."""

    def __init__(
        self,
        server_source: Any,
        server_args: Optional[List[str]] = None,
        transport_type: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **transport_kwargs: Any,
    ):
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "MCP integration requires 'fastmcp'. Install with: pip install fastmcp>=2.0.0"
            )

        self.server_args = server_args or []
        self.transport_type = transport_type
        self.env = env or {}
        self.transport_kwargs = transport_kwargs
        self.server_source = self._prepare_server_source(server_source)
        self.client: Optional[Any] = None
        self._context_manager = None

    def _prepare_server_source(self, server_source: Any) -> Any:
        _, fastmcp_cls, py_stdio_cls, sse_cls, http_cls = _load_fastmcp_classes()

        if isinstance(server_source, fastmcp_cls):
            return server_source

        if isinstance(server_source, dict):
            return self._create_transport_from_config(server_source)

        if isinstance(server_source, str) and (
            server_source.startswith("http://") or server_source.startswith("https://")
        ):
            selected_transport = self.transport_type or "http"
            if selected_transport == "sse":
                return sse_cls(url=server_source, **self.transport_kwargs)
            return http_cls(url=server_source, **self.transport_kwargs)

        if isinstance(server_source, str) and server_source.endswith(".py"):
            return py_stdio_cls(
                script_path=server_source,
                args=self.server_args,
                env=self.env if self.env else None,
                **self.transport_kwargs,
            )

        if isinstance(server_source, list) and len(server_source) >= 1:
            if (
                server_source[0] == "python"
                and len(server_source) > 1
                and server_source[1].endswith(".py")
            ):
                return py_stdio_cls(
                    script_path=server_source[1],
                    args=server_source[2:] + self.server_args,
                    env=self.env if self.env else None,
                    **self.transport_kwargs,
                )

            from fastmcp.client.transports import StdioTransport  # type: ignore[import-not-found]

            return StdioTransport(
                command=server_source[0],
                args=server_source[1:] + self.server_args,
                env=self.env if self.env else None,
                **self.transport_kwargs,
            )

        return server_source

    def _create_transport_from_config(self, config: Dict[str, Any]) -> Any:
        _, _, py_stdio_cls, sse_cls, http_cls = _load_fastmcp_classes()
        transport = config.get("transport", "stdio")

        if transport == "stdio":
            args = config.get("args", [])
            if args and isinstance(args[0], str) and args[0].endswith(".py"):
                return py_stdio_cls(
                    script_path=args[0],
                    args=args[1:] + self.server_args,
                    env=config.get("env"),
                    cwd=config.get("cwd"),
                    **self.transport_kwargs,
                )

            from fastmcp.client.transports import StdioTransport  # type: ignore[import-not-found]

            return StdioTransport(
                command=config.get("command", "python"),
                args=args + self.server_args,
                env=config.get("env"),
                cwd=config.get("cwd"),
                **self.transport_kwargs,
            )

        if transport == "sse":
            return sse_cls(
                url=config["url"],
                headers=config.get("headers"),
                auth=config.get("auth"),
                **self.transport_kwargs,
            )

        if transport == "http":
            return http_cls(
                url=config["url"],
                headers=config.get("headers"),
                auth=config.get("auth"),
                **self.transport_kwargs,
            )

        raise ValueError(f"Unsupported transport type: {transport}")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect(exc_type, exc_val, exc_tb)
        logger.info("MCP client context exited, transport closed.")
    async def connect(self) -> None:
        if self.client is not None:
            return
        client_cls, _, _, _, _ = _load_fastmcp_classes()
        raw_client = client_cls(self.server_source)
        self._context_manager = raw_client
        if self._context_manager is None:
            raise RuntimeError("Failed to initialize MCP client context manager")
        entered_client = await self._context_manager.__aenter__()
        self.client = entered_client or raw_client

    async def disconnect(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        client_ref = self.client

        if self._context_manager is not None:
            try:
                await self._context_manager.__aexit__(exc_type, exc_val, exc_tb)
            except asyncio.CancelledError:
                # fastmcp may cancel internal background tasks during normal shutdown.
                pass

        # Best-effort fallback: explicitly close transport resources if exposed.
        await self._close_transport_fallback(client_ref)

        # Let pending callbacks run before outer event loop gets closed.
        await asyncio.sleep(0)
        self.client = None
        self._context_manager = None

    async def _close_transport_fallback(self, client_ref: Optional[Any]) -> None:
        """Try to close low-level transport resources to reduce shutdown warnings."""
        if client_ref is None:
            return

        transport = getattr(client_ref, "transport", None)
        if transport is None:
            return

        # Prefer async close when available, then fallback to sync close.
        for method_name in ("aclose", "close"):
            method = getattr(transport, method_name, None)
            if not callable(method):
                continue

            try:
                result = method()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # Ignore transport-close errors at shutdown stage.
                pass
            break

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        result = await self.client.list_tools()

        if hasattr(result, "tools"):
            tools = result.tools
        elif isinstance(result, list):
            tools = result
        else:
            tools = []

        return [
            {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": getattr(tool, "inputSchema", {}) or {},
            }
            for tool in tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        result = await self.client.call_tool(tool_name, arguments)

        if hasattr(result, "content") and result.content:
            if len(result.content) == 1:
                content = result.content[0]
                if hasattr(content, "text"):
                    return content.text
                if hasattr(content, "data"):
                    return content.data
            return [
                getattr(item, "text", getattr(item, "data", str(item)))
                for item in result.content
            ]
        return None

    async def ping(self) -> bool:
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    def is_connected(self) -> bool:
        return self.client is not None

    async def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有可用的资源"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_resources()
        return [
            {
                "uri": resource.uri,
                "name": resource.name or "",
                "description": resource.description or "",
                "mime_type": getattr(resource, 'mimeType', None)
            }
            for resource in result.resources
        ]

    async def read_resource(self, uri: str) -> Any:
        """读取资源内容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.read_resource(uri)

        # 解析资源内容
        if hasattr(result, 'contents') and result.contents:
            if len(result.contents) == 1:
                content = result.contents[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'blob'):
                    return content.blob
            return [
                getattr(c, 'text', getattr(c, 'blob', str(c)))
                for c in result.contents
            ]
        return None

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有可用的提示词模板"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_prompts()
        return [
            {
                "name": prompt.name,
                "description": prompt.description or "",
                "arguments": getattr(prompt, 'arguments', [])
            }
            for prompt in result.prompts
        ]

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """获取提示词内容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.get_prompt(prompt_name, arguments or {})

        # 解析提示词消息
        if hasattr(result, 'messages') and result.messages:
            return [
                {
                    "role": msg.role,
                    "content": getattr(msg.content, 'text', str(msg.content)) if hasattr(msg.content, 'text') else str(msg.content)
                }
                for msg in result.messages
            ]
        return []

    def get_transport_info(self) -> Dict[str, Any]:
        """获取传输信息"""
        if not self.client:
            return {"status": "not_connected"}
        
        transport = getattr(self.client, 'transport', None)
        if transport:
            return {
                "status": "connected",
                "transport_type": type(transport).__name__,
                "transport_info": str(transport)
            }
        return {"status": "unknown"}
