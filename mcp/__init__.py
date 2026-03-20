"""EasyAgent MCP integration package.

This package name collides with the upstream MCP SDK package name (also `mcp`).
To keep compatibility while preserving the current project structure, we:
1) expose `MCPClient` eagerly (safe, no fastmcp import at module import time),
2) load `MCPServer` lazily,
3) extend this package search path with site-packages MCP path when available,
   so imports like `mcp.types` (required by `fastmcp`) can resolve correctly.
"""

from __future__ import annotations

import importlib.util
import importlib
import os
import sys
from typing import Any

from .mcp_client import MCPClient

# Lazy-exported symbol placeholder for static analyzers.
MCPServer: Any


def _extend_package_path_for_sdk() -> None:
	"""Allow `mcp.types` to resolve from installed MCP SDK package path."""
	current_dir = os.path.dirname(__file__)
	package_paths = globals().get("__path__", [])

	for base in sys.path:
		if not isinstance(base, str):
			continue
		candidate = os.path.join(base, "mcp")
		if os.path.abspath(candidate) == os.path.abspath(current_dir):
			continue
		if os.path.isfile(os.path.join(candidate, "types.py")) and candidate not in package_paths:
			package_paths.append(candidate)


def _resolve_upstream_attr(name: str) -> Any:
	"""Resolve symbols from installed MCP SDK submodules.

	We cannot import upstream `mcp.__init__` directly because this local package
	shadows it by name. Instead, we import common upstream submodules via this
	package's extended __path__ and pull symbols lazily.
	"""
	candidate_modules = [
		"types",
		"client.session",
		"client.session_group",
		"client.stdio",
		"server.session",
		"server.stdio",
		"shared.exceptions",
	]

	for mod in candidate_modules:
		try:
			module = importlib.import_module(f".{mod}", __name__)
		except Exception:
			continue
		if hasattr(module, name):
			return getattr(module, name)

	raise AttributeError(name)


_extend_package_path_for_sdk()


def __getattr__(name: str) -> Any:
	if name == "MCPServer":
		from .mcp_server import MCPServer

		return MCPServer

	try:
		return _resolve_upstream_attr(name)
	except AttributeError:
		pass

	raise AttributeError(f"module 'mcp' has no attribute {name!r}")


__all__ = ["MCPClient", "MCPServer"]
