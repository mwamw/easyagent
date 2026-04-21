"""Caching primitives for MCP capabilities, resources, and prompts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


def _stable_prompt_key(prompt_name: str, arguments: Optional[dict[str, str]] = None) -> str:
    payload = {
        "promptName": str(prompt_name or ""),
        "arguments": dict(arguments or {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


@dataclass(slots=True)
class MCPCapabilitySnapshot:
    server_name: str
    source_identifier: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    refreshed_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_stale(self, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return False
        if self.refreshed_at is None:
            return True
        return (time.time() - float(self.refreshed_at)) > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "serverName": self.server_name,
            "sourceIdentifier": self.source_identifier,
            "tools": list(self.tools),
            "resources": list(self.resources),
            "prompts": list(self.prompts),
            "refreshedAt": self.refreshed_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MCPCapabilitySnapshot" | None:
        data = dict(payload or {})
        server_name = str(data.get("serverName") or "").strip()
        source_identifier = str(data.get("sourceIdentifier") or "").strip()
        if not server_name or not source_identifier:
            return None
        return cls(
            server_name=server_name,
            source_identifier=source_identifier,
            tools=list(data.get("tools") or []),
            resources=list(data.get("resources") or []),
            prompts=list(data.get("prompts") or []),
            refreshed_at=data.get("refreshedAt"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class MCPCacheEntry:
    value: Any
    cached_at: float

    def is_stale(self, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.cached_at) > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "cachedAt": self.cached_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MCPCacheEntry" | None:
        data = dict(payload or {})
        if "cachedAt" not in data:
            return None
        return cls(value=data.get("value"), cached_at=float(data.get("cachedAt") or 0))


@dataclass(slots=True)
class MCPServerCache:
    capability_ttl_seconds: int = 300
    resource_ttl_seconds: int = 0
    prompt_ttl_seconds: int = 0
    capability_snapshot: Optional[MCPCapabilitySnapshot] = None
    resource_entries: dict[str, MCPCacheEntry] = field(default_factory=dict)
    prompt_entries: dict[str, MCPCacheEntry] = field(default_factory=dict)

    def update_capabilities(
        self,
        *,
        server_name: str,
        source_identifier: str,
        tools: Optional[list[dict[str, Any]]] = None,
        resources: Optional[list[dict[str, Any]]] = None,
        prompts: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MCPCapabilitySnapshot:
        current = self.capability_snapshot or MCPCapabilitySnapshot(
            server_name=server_name,
            source_identifier=source_identifier,
        )
        current.server_name = server_name
        current.source_identifier = source_identifier
        if tools is not None:
            current.tools = list(tools)
        if resources is not None:
            current.resources = list(resources)
        if prompts is not None:
            current.prompts = list(prompts)
        current.refreshed_at = time.time()
        if metadata:
            current.metadata.update(dict(metadata))
        self.capability_snapshot = current
        return current

    def get_capability_snapshot(self) -> Optional[MCPCapabilitySnapshot]:
        return self.capability_snapshot

    def get_resource(self, uri: str) -> Any:
        entry = self.resource_entries.get(str(uri))
        if entry is None:
            return None
        if entry.is_stale(self.resource_ttl_seconds):
            self.resource_entries.pop(str(uri), None)
            return None
        return entry.value

    def set_resource(self, uri: str, value: Any) -> None:
        self.resource_entries[str(uri)] = MCPCacheEntry(value=value, cached_at=time.time())

    def invalidate_resource(self, uri: str | None = None) -> None:
        if uri is None:
            self.resource_entries.clear()
            return
        self.resource_entries.pop(str(uri), None)

    def get_prompt(self, prompt_name: str, arguments: Optional[dict[str, str]] = None) -> Any:
        key = _stable_prompt_key(prompt_name, arguments)
        entry = self.prompt_entries.get(key)
        if entry is None:
            return None
        if entry.is_stale(self.prompt_ttl_seconds):
            self.prompt_entries.pop(key, None)
            return None
        return entry.value

    def set_prompt(self, prompt_name: str, arguments: Optional[dict[str, str]], value: Any) -> None:
        key = _stable_prompt_key(prompt_name, arguments)
        self.prompt_entries[key] = MCPCacheEntry(value=value, cached_at=time.time())

    def invalidate_prompt(self, prompt_name: str | None = None, arguments: Optional[dict[str, str]] = None) -> None:
        if prompt_name is None:
            self.prompt_entries.clear()
            return
        key = _stable_prompt_key(prompt_name, arguments)
        self.prompt_entries.pop(key, None)

    def export_state(self) -> dict[str, Any]:
        return {
            "capabilityTtlSeconds": self.capability_ttl_seconds,
            "resourceTtlSeconds": self.resource_ttl_seconds,
            "promptTtlSeconds": self.prompt_ttl_seconds,
            "capabilitySnapshot": self.capability_snapshot.to_dict() if self.capability_snapshot is not None else None,
            "resourceEntries": {
                uri: entry.to_dict()
                for uri, entry in self.resource_entries.items()
            },
            "promptEntries": {
                key: entry.to_dict()
                for key, entry in self.prompt_entries.items()
            },
        }

    def restore_state(self, payload: dict[str, Any] | None) -> None:
        state = dict(payload or {})
        self.capability_ttl_seconds = int(state.get("capabilityTtlSeconds", self.capability_ttl_seconds) or 0)
        self.resource_ttl_seconds = int(state.get("resourceTtlSeconds", self.resource_ttl_seconds) or 0)
        self.prompt_ttl_seconds = int(state.get("promptTtlSeconds", self.prompt_ttl_seconds) or 0)
        self.capability_snapshot = MCPCapabilitySnapshot.from_dict(state.get("capabilitySnapshot"))
        self.resource_entries = {}
        for uri, entry_payload in dict(state.get("resourceEntries") or {}).items():
            entry = MCPCacheEntry.from_dict(entry_payload)
            if entry is not None:
                self.resource_entries[str(uri)] = entry
        self.prompt_entries = {}
        for key, entry_payload in dict(state.get("promptEntries") or {}).items():
            entry = MCPCacheEntry.from_dict(entry_payload)
            if entry is not None:
                self.prompt_entries[str(key)] = entry
