"""MCP tool integration for EasyAgent.

This module bridges MCP tools/resources into EasyAgent's ToolRegistry.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeListMcpResourcesInput, ClaudeReadMcpResourceInput
from Emcp.runtime import MCPClientProtocol, MCPHub, MCPRuntimeManager

logger = logging.getLogger(__name__)


def _json_type_to_python_type(type_name: Optional[str]) -> Any:
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(type_name or "", Any)


def _sanitize_model_name(value: str) -> str:
    normalized = []
    for ch in value:
        if ch.isalnum() or ch == "_":
            normalized.append(ch)
        else:
            normalized.append("_")
    safe = "".join(normalized).strip("_")
    return safe or "mcp_item"


def _prefixed_name(prefix: str, base_name: str) -> str:
    return f"{prefix}{base_name}" if prefix else base_name


def _build_pydantic_model_from_schema(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

    if not properties:
        class EmptyParams(BaseModel):
            pass

        return EmptyParams

    fields: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            field_schema = {}

        py_type = _json_type_to_python_type(field_schema.get("type"))
        description = field_schema.get("description", "")

        if field_name in required:
            fields[field_name] = (py_type, Field(description=description))
        else:
            default = field_schema.get("default", None)
            fields[field_name] = (
                Optional[py_type],
                Field(default=default, description=description),
            )

    model_name = f"MCP_{_sanitize_model_name(tool_name)}_Params"
    return create_model(model_name, **fields)


def _normalize_annotations(annotations: Any) -> Dict[str, Any]:
    if annotations is None:
        return {}
    if isinstance(annotations, dict):
        return dict(annotations)
    if hasattr(annotations, "model_dump"):
        return annotations.model_dump(mode="json")
    if hasattr(annotations, "__dict__"):
        return {
            key: value
            for key, value in vars(annotations).items()
            if not key.startswith("_")
        }
    return {}


def _annotation_flag(annotations: Dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = annotations.get(key)
        if isinstance(value, bool):
            return value
    return False


def _extract_mcp_hint_state(annotations: Dict[str, Any]) -> Dict[str, bool]:
    read_only = _annotation_flag(annotations, "readOnlyHint", "read_only")
    destructive = _annotation_flag(annotations, "destructiveHint", "destructive")
    open_world = _annotation_flag(annotations, "openWorldHint", "open_world")
    idempotent_present = any(
        isinstance(annotations.get(key), bool)
        for key in ("idempotentHint", "idempotent")
    )
    idempotent = _annotation_flag(annotations, "idempotentHint", "idempotent")
    return {
        "read_only": read_only,
        "destructive": destructive,
        "open_world": open_world,
        "idempotent_present": idempotent_present,
        "idempotent": idempotent,
    }


def _build_mcp_guidance(
    base_guidance: str,
    annotations: Dict[str, Any],
    *,
    capability_kind: str,
) -> str:
    hints = _extract_mcp_hint_state(annotations)
    lines: List[str] = []
    if base_guidance.strip():
        lines.append(base_guidance.strip())

    lines.extend(
        [
            "Use remote MCP capabilities when local built-in tools, skills, and existing context are insufficient.",
            "Follow the parameter schema exactly. Do not invent fields or substitute prose for structured arguments.",
            "For costly or high-impact operations, narrow the target and make the minimum necessary call.",
        ]
    )

    if hints["read_only"]:
        lines.append("This MCP capability declares itself read-only and is not expected to modify remote state.")
    if hints["destructive"]:
        lines.append("This MCP capability declares possible remote side effects. Verify the target, arguments, and necessity before calling.")
    if hints["open_world"]:
        lines.append("This MCP capability depends on external world or system state. Results may change; revalidate when needed.")
    if hints["idempotent_present"]:
        if hints["idempotent"]:
            lines.append("This MCP capability declares itself idempotent; retry only when necessary and still authorized.")
        else:
            lines.append("This MCP capability is not declared idempotent. Do not retry blindly or issue duplicate parallel calls.")

    if capability_kind == "resource_read":
        lines.append("Use resource annotations, when present, to assess relevance to the current task.")
        lines.append("Prefer listing resources before reading. Do not guess resource URIs.")
    elif capability_kind == "resource_list":
        lines.append("Returns a resource directory, not resource contents. Use a listed URI to read relevant content.")
    elif capability_kind == "tool":
        lines.append("If remote results conflict with local assumptions, verify the discrepancy and explain it when relevant.")
    elif capability_kind in {"prompt_list", "prompt_get"}:
        lines.append("An MCP prompt is a remote template, not a final answer. Interpret it in the context of the current task.")

    return "\n".join(line for line in lines if line)


class MCPListResourcesParams(BaseModel):
    pass


class MCPReadResourceParams(BaseModel):
    uri: str = Field(description="URI of the MCP resource to read")


def _effective_server_name(manager: "MCPToolManager") -> str:
    return manager.registry_server_name


def _mcp_source_identifier(server_name: str, capability_kind: str, capability_name: Optional[str] = None) -> str:
    normalized_server = str(server_name or "mcp").strip() or "mcp"
    normalized_kind = str(capability_kind or "capabilities").strip() or "capabilities"
    if capability_name:
        return f"mcp://{normalized_server}/{normalized_kind}/{capability_name}"
    return f"mcp://{normalized_server}/{normalized_kind}"


def _serialize_for_display(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _build_mcp_display_text(
    title: str,
    payload: Any,
    *,
    summary: Optional[str] = None,
) -> str:
    lines = [title.strip()]
    if summary:
        lines.append(summary.strip())
    if payload is not None:
        lines.append("结构化详情:")
        lines.append(_serialize_for_display(payload))
    return "\n\n".join(line for line in lines if line)


def _mcp_error_to_tool_result(
    exc: Exception,
    *,
    manager: "MCPToolManager",
    capability_kind: str,
    capability_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ToolResult:
    detail = manager.describe_error(exc)
    server_name = _effective_server_name(manager)
    error_type = detail.get("errorType") or "mcp_operation_error"
    message = str(detail.get("message") or str(exc))
    title = f"MCP server `{server_name}` 的 `{capability_kind}` 调用失败。"
    summary_lines = [
        f"错误类型: {error_type}",
        f"原因: {message}",
    ]
    if capability_name:
        summary_lines.insert(0, f"目标能力: {capability_name}")
    display_text = _build_mcp_display_text(
        title,
        detail,
        summary="\n".join(summary_lines),
    )
    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            "mcp_server": server_name,
            "mcp_error_type": error_type,
            "mcp_capability_kind": capability_kind,
        }
    )
    if capability_name:
        merged_metadata["mcp_capability_name"] = capability_name
    return ToolResult.error(
        message,
        error_type=error_type,
        display_text=display_text,
        metadata=merged_metadata,
    )


def _format_resource_listing(resources: List[Dict[str, Any]]) -> str:
    if not resources:
        return "未发现 MCP 资源。"

    lines: List[str] = []
    for index, resource in enumerate(resources, start=1):
        uri = str(resource.get("uri", ""))
        server = str(resource.get("server", ""))
        name = str(resource.get("name", "") or "")
        description = str(resource.get("description", "") or "")
        prefix = f"{index}. "
        header = f"{prefix}{uri}"
        if server:
            header += f"  [server={server}]"
        lines.append(header)
        if name:
            lines.append(f"   名称: {name}")
        if description:
            lines.append(f"   描述: {description}")
    return "\n".join(lines)


def _resource_read_result_to_tool_result(
    result: Any,
    *,
    server_name: str,
    uri: str,
) -> ToolResult:
    metadata = {
        "mcp_server": server_name,
        "uri": uri,
        "source_identifier": _mcp_source_identifier(server_name, "resources", uri),
    }
    if isinstance(result, bytes):
        payload = {
            "server": server_name,
            "uri": uri,
            "encoding": "base64",
            "content": base64.b64encode(result).decode("ascii"),
        }
        return ToolResult.success(
            structured_data=payload,
            metadata=metadata,
        )
    if isinstance(result, list):
        normalized: List[Any] = []
        for item in result:
            if isinstance(item, bytes):
                normalized.append(
                    {
                        "encoding": "base64",
                        "content": base64.b64encode(item).decode("ascii"),
                    }
                )
            else:
                normalized.append(item)
        payload = {"server": server_name, "uri": uri, "content": normalized}
        return ToolResult.success(
            structured_data=payload,
            metadata=metadata,
        )
    payload = {"server": server_name, "uri": uri, "content": str(result or "")}
    return ToolResult.success(
        content=str(result or ""),
        structured_data=payload,
        metadata=metadata,
    )


class MCPWrappedTool(Tool):
    """Wrap a remote MCP tool as an EasyAgent Tool."""

    def __init__(
        self,
        manager: "MCPToolManager",
        tool_info: Dict[str, Any],
        prefix: str = "",
    ):
        self.manager = manager
        self.tool_info = tool_info
        self.mcp_tool_name = tool_info.get("name", "unknown")

        tool_name = _prefixed_name(prefix, self.mcp_tool_name)
        description = tool_info.get("description") or f"MCP tool: {self.mcp_tool_name}"
        input_schema = tool_info.get("input_schema") or {}
        annotations = _normalize_annotations(tool_info.get("annotations"))
        hint_state = _extract_mcp_hint_state(annotations)

        parameters = _build_pydantic_model_from_schema(self.mcp_tool_name, input_schema)
        base_guidance = (
            f"This is the remote MCP tool `{tool_name}`.\n"
            f"It maps to `{self.mcp_tool_name}` on MCP server `{_effective_server_name(manager)}`.\n"
            "Check that the task needs this remote capability, then use the minimum necessary arguments.\n"
            "When reusing results, reference their structured fields rather than rewriting key facts."
        )
        server_name = _effective_server_name(manager)
        source_identifier = _mcp_source_identifier(server_name, "tools", self.mcp_tool_name)

        super().__init__(
            name=tool_name,
            description=description,
            parameters=parameters,
            guidance=_build_mcp_guidance(base_guidance, annotations, capability_kind="tool"),
            source="mcp",
            read_only=hint_state["read_only"],
            destructive=hint_state["destructive"],
            requires_confirmation=hint_state["destructive"] or hint_state["open_world"],
            risk_categories=["mcp"],
            side_effect_level=(
                "none"
                if hint_state["read_only"] and not hint_state["open_world"]
                else ("high" if hint_state["destructive"] else "medium")
            ),
            resource_scope=[source_identifier],
            supports_parallel=(
                hint_state["idempotent"]
                if hint_state["idempotent_present"]
                else (hint_state["read_only"] and not hint_state["open_world"] and not hint_state["destructive"])
            ),
            tags=["mcp", "remote"],
            metadata={
                "mcp_tool_name": self.mcp_tool_name,
                "mcp_server": _effective_server_name(manager),
                "mcp_annotations": annotations,
                "mcp_read_only": hint_state["read_only"],
                "mcp_destructive": hint_state["destructive"],
                "mcp_open_world": hint_state["open_world"],
                "mcp_idempotent": hint_state["idempotent"] if hint_state["idempotent_present"] else None,
                "mcp_capability_kind": "tool",
                "source_identifier": source_identifier,
            },
        )

    def run(self, parameters: dict):
        try:
            result = self.manager.execute_tool(self.mcp_tool_name, parameters)
        except Exception as exc:
            return _mcp_error_to_tool_result(
                exc,
                manager=self.manager,
                capability_kind="tool",
                capability_name=self.mcp_tool_name,
                metadata={"mcp_tool_name": self.mcp_tool_name},
            )
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, (dict, list)):
            payload = {
                "server": _effective_server_name(self.manager),
                "tool": self.mcp_tool_name,
                "result": result,
            }
            source_identifier = _mcp_source_identifier(_effective_server_name(self.manager), "tools", self.mcp_tool_name)
            return ToolResult.success(
                structured_data=payload,
                metadata={
                    "mcp_tool_name": self.mcp_tool_name,
                    "mcp_server": _effective_server_name(self.manager),
                    "source_identifier": source_identifier,
                },
            )
        if result is None:
            payload = {
                "server": _effective_server_name(self.manager),
                "tool": self.mcp_tool_name,
                "result": None,
            }
            source_identifier = _mcp_source_identifier(_effective_server_name(self.manager), "tools", self.mcp_tool_name)
            return ToolResult.success(
                "",
                structured_data=payload,
                metadata={
                    "mcp_tool_name": self.mcp_tool_name,
                    "mcp_server": _effective_server_name(self.manager),
                    "source_identifier": source_identifier,
                },
            )
        payload = {
            "server": _effective_server_name(self.manager),
            "tool": self.mcp_tool_name,
            "result": str(result),
        }
        source_identifier = _mcp_source_identifier(_effective_server_name(self.manager), "tools", self.mcp_tool_name)
        return ToolResult.success(
            str(result),
            structured_data=payload,
            metadata={
                "mcp_tool_name": self.mcp_tool_name,
                "mcp_server": _effective_server_name(self.manager),
                "source_identifier": source_identifier,
            },
        )


class MCPListResourcesTool(Tool):
    """List MCP resources exposed by one server."""

    def __init__(self, manager: "MCPToolManager", prefix: str = ""):
        self.manager = manager
        source_identifier = _mcp_source_identifier(_effective_server_name(manager), "resources")
        super().__init__(
            name=_prefixed_name(prefix, "list_mcp_resources"),
            description=f"List resources currently exposed by MCP server `{_effective_server_name(manager)}`.",
            parameters=MCPListResourcesParams,
            guidance=_build_mcp_guidance((
                "Use to browse a remote MCP resource directory and discover readable URIs.\n"
                "Inspect the directory before reading specific resources. This tool does not read resource contents."
            ), {}, capability_kind="resource_list"),
            read_only=True,
            source="mcp",
            tags=["mcp", "resource", "read"],
            supports_parallel=True,
            risk_categories=["mcp"],
            side_effect_level="none",
            resource_scope=[source_identifier],
            metadata={
                "mcp_server": _effective_server_name(manager),
                "mcp_resource_tool": True,
                "mcp_capability_kind": "resource_list",
                "source_identifier": source_identifier,
            },
        )

    def run(self, parameters: dict) -> ToolResult:
        server_name = _effective_server_name(self.manager)
        try:
            resources = [
                {
                    "server": server_name,
                    **resource,
                }
                for resource in self.manager.list_remote_resources()
            ]
        except Exception as exc:
            return _mcp_error_to_tool_result(
                exc,
                manager=self.manager,
                capability_kind="resource_list",
                metadata={"mcp_server": server_name},
            )
        payload = {
            "server": server_name,
            "resources": resources,
        }
        return ToolResult.success(
            _format_resource_listing(resources),
            structured_data=payload,
            metadata={
                "mcp_server": server_name,
                "source_identifier": _mcp_source_identifier(server_name, "resources"),
            },
        )


class MCPReadResourceTool(Tool):
    """Read one MCP resource."""

    def __init__(self, manager: "MCPToolManager", prefix: str = ""):
        self.manager = manager
        source_identifier = _mcp_source_identifier(_effective_server_name(manager), "resources", "{uri}")
        super().__init__(
            name=_prefixed_name(prefix, "read_mcp_resource"),
            description=f"Read the resource at a specified URI on MCP server `{_effective_server_name(manager)}`.",
            parameters=MCPReadResourceParams,
            guidance=_build_mcp_guidance((
                "Use `list_mcp_resources` to discover available URIs, then read a specific resource.\n"
                "Do not read an unverified or guessed URI.\n"
                "Binary resource content is returned as base64."
            ), {}, capability_kind="resource_read"),
            read_only=True,
            source="mcp",
            tags=["mcp", "resource", "read"],
            supports_parallel=True,
            risk_categories=["mcp"],
            side_effect_level="none",
            resource_scope=[source_identifier],
            metadata={
                "mcp_server": _effective_server_name(manager),
                "mcp_resource_tool": True,
                "mcp_capability_kind": "resource_read",
                "source_identifier": source_identifier,
            },
        )

    def run(self, parameters: dict) -> ToolResult:
        uri = str(parameters.get("uri", "")).strip()
        if not uri:
            return ToolResult.error("错误：必须提供资源 URI。", error_type="invalid_parameters")

        server_name = _effective_server_name(self.manager)
        try:
            result = self.manager.read_remote_resource(uri)
        except Exception as exc:
            return _mcp_error_to_tool_result(
                exc,
                manager=self.manager,
                capability_kind="resource_read",
                capability_name=uri,
                metadata={"mcp_server": server_name, "uri": uri},
            )
        return _resource_read_result_to_tool_result(result, server_name=server_name, uri=uri)


class MCPHubListResourcesTool(Tool):
    """List MCP resources across all registered servers or one selected server."""

    def __init__(self, hub: MCPHub):
        self.hub = hub
        super().__init__(
            name="ListMcpResources",
            description="Lists resources exposed by registered MCP servers.",
            parameters=ClaudeListMcpResourcesInput,
            guidance=(
                "Use to discover resource URIs. Specify server when known; otherwise resources from registered servers are aggregated. Use ReadMcpResource for resource content."
            ),
            read_only=True,
            source="mcp",
            tags=["mcp", "resource", "read", "claude_code"],
            supports_parallel=True,
            risk_categories=["mcp"],
            side_effect_level="none",
            resource_scope=["mcp://hub/resources"],
            metadata={
                "mcp_hub": True,
                "mcp_capability_kind": "resource_list",
                "source_identifier": "mcp://hub/resources",
            },
        )

    def run(self, parameters: dict) -> ToolResult:
        server = str(parameters.get("server") or "").strip() or None
        try:
            resources = self.hub.list_resources(server)
        except KeyError as exc:
            return ToolResult.error(
                str(exc),
                error_type="mcp_server_not_found",
                display_text=_build_mcp_display_text(
                    "读取 MCP 资源目录失败。",
                    {"server": server, "availableServers": self.hub.list_servers()},
                    summary=str(exc),
                ),
                metadata={"server": server},
            )
        payload = {
            "server": server,
            "resources": resources,
            "servers": self.hub.list_servers(),
        }
        return ToolResult.success(
            _format_resource_listing(resources),
            structured_data=payload,
            metadata={
                "server": server,
                "mcp_servers": self.hub.list_servers(),
                "source_identifier": "mcp://hub/resources",
            },
        )


class MCPHubReadResourceTool(Tool):
    """Read an MCP resource from a selected server."""

    def __init__(self, hub: MCPHub):
        self.hub = hub
        super().__init__(
            name="ReadMcpResource",
            description="Reads resource content from a specified MCP server and URI.",
            parameters=ClaudeReadMcpResourceInput,
            guidance=(
                "Use a known server and URI from ListMcpResources. Resource content comes from the remote server."
            ),
            read_only=True,
            source="mcp",
            tags=["mcp", "resource", "read", "claude_code"],
            supports_parallel=True,
            risk_categories=["mcp"],
            side_effect_level="none",
            resource_scope=["mcp://hub/resources/read"],
            metadata={
                "mcp_hub": True,
                "mcp_capability_kind": "resource_read",
                "source_identifier": "mcp://hub/resources/read",
            },
        )

    def run(self, parameters: dict) -> ToolResult:
        server = str(parameters.get("server") or "").strip()
        uri = str(parameters.get("uri") or "").strip()
        if not server:
            return ToolResult.error("错误：必须提供 MCP server 名称。", error_type="invalid_parameters")
        if not uri:
            return ToolResult.error("错误：必须提供资源 URI。", error_type="invalid_parameters")

        try:
            result = self.hub.read_resource(server, uri)
            normalized_server = self.hub.normalize_server_name(server)
        except KeyError as exc:
            return ToolResult.error(
                str(exc),
                error_type="mcp_server_not_found",
                display_text=_build_mcp_display_text(
                    "读取 MCP 资源失败。",
                    {"server": server, "uri": uri, "availableServers": self.hub.list_servers()},
                    summary=str(exc),
                ),
                metadata={"server": server, "uri": uri},
            )
        return _resource_read_result_to_tool_result(result, server_name=normalized_server, uri=uri)


class MCPToolManager(MCPRuntimeManager):
    """Manage MCP runtime and register remote capabilities into EasyAgent."""

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
        policy_context: Optional[Any] = None,
        cache_store: Optional[Any] = None,
        max_retries: Optional[int] = None,
        persist_connection: Optional[bool] = None,
        include_resources: bool = False,
        resource_tool_prefix: Optional[str] = None,
        **transport_kwargs: Any,
    ):
        super().__init__(
            server_source=server_source,
            server_args=server_args,
            transport_type=transport_type,
            env=env,
            tool_prefix=tool_prefix,
            auto_connect=auto_connect,
            client=client,
            auth_config=auth_config,
            policy_context=policy_context,
            cache_store=cache_store,
            max_retries=max_retries,
            persist_connection=persist_connection,
            **transport_kwargs,
        )
        self.include_resources = include_resources
        self.resource_tool_prefix = resource_tool_prefix
        self._wrapped_tools: List[MCPWrappedTool] = []
        self._resource_tools: List[Tool] = []
        self._registered_server_name: Optional[str] = None

    @property
    def registry_server_name(self) -> str:
        return self._registered_server_name or self.server_label

    def build_resource_tools(self, resource_tool_prefix: Optional[str] = None) -> List[Tool]:
        prefix = resource_tool_prefix
        if prefix is None:
            prefix = self.tool_prefix or f"{self.registry_server_name}_"
        return [
            MCPListResourcesTool(self, prefix=prefix),
            MCPReadResourceTool(self, prefix=prefix),
        ]

    def register_to_registry(
        self,
        registry: ToolRegistry,
        *,
        include_resources: Optional[bool] = None,
        resource_tool_prefix: Optional[str] = None,
        hub: Optional[MCPHub] = None,
        server_name: Optional[str] = None,
        legacy_resource_tools: Optional[bool] = None,
    ) -> List[Tool]:
        if hub is not None:
            self._registered_server_name = hub.register_manager(self, server_name=server_name)
            register_surface = getattr(registry, "register_runtime_surface", None)
            if callable(register_surface):
                register_surface("mcp_hub", "default", hub)
        else:
            self._registered_server_name = server_name or self.server_label
        if getattr(self, "connection_manager", None) is not None:
            try:
                self.connection_manager.state.server_name = self.registry_server_name
            except Exception:
                pass
        register_surface = getattr(registry, "register_runtime_surface", None)
        if callable(register_surface):
            register_surface("mcp_manager", self.registry_server_name, self)

        remote_tools = self.list_remote_tools()
        wrapped: List[Tool] = []

        for tool_info in remote_tools:
            tool = MCPWrappedTool(
                manager=self,
                tool_info=tool_info,
                prefix=self.tool_prefix,
            )
            registry.register_tool(tool)
            wrapped.append(tool)

        self._wrapped_tools = [tool for tool in wrapped if isinstance(tool, MCPWrappedTool)]

        should_include_resources = self.include_resources if include_resources is None else include_resources
        resource_prefix = self.resource_tool_prefix if resource_tool_prefix is None else resource_tool_prefix
        use_legacy_resource_tools = legacy_resource_tools if legacy_resource_tools is not None else hub is None

        if should_include_resources:
            if hub is not None and not use_legacy_resource_tools:
                resource_tools = list(register_mcp_resource_hub_tools(registry, hub))
            else:
                resource_tools = self.build_resource_tools(resource_prefix)
                for tool in resource_tools:
                    registry.register_tool(tool)
            self._resource_tools = list(resource_tools)
            wrapped.extend(resource_tools)

        return wrapped

    def get_wrapped_tools(self) -> List[MCPWrappedTool]:
        return list(self._wrapped_tools)

    def get_resource_tools(self) -> List[Tool]:
        return list(self._resource_tools)

    def export_state(self) -> dict[str, Any]:
        payload = super().export_state()
        payload.update(
            {
                "includeResources": self.include_resources,
                "resourceToolPrefix": self.resource_tool_prefix,
                "registryServerName": self._registered_server_name,
                "wrappedToolNames": [tool.name for tool in self._wrapped_tools],
                "resourceToolNames": [tool.name for tool in self._resource_tools],
            }
        )
        return payload

    def restore_state(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        report = super().restore_state(payload)
        state = dict(payload or {})
        self.include_resources = bool(state.get("includeResources", self.include_resources))
        self.resource_tool_prefix = state.get("resourceToolPrefix", self.resource_tool_prefix)
        self._registered_server_name = state.get("registryServerName", self._registered_server_name)
        return report

    @classmethod
    def from_state(
        cls,
        payload: dict[str, Any],
        *,
        client: Optional[MCPClientProtocol] = None,
    ) -> "MCPToolManager":
        state = dict(payload or {})
        server_source_info = dict(state.get("serverSource") or {})
        if not bool(server_source_info.get("restorable", False)) and client is None:
            raise RuntimeError("MCP server_source 不可恢复；请在恢复时显式提供 client。")

        manager = cls(
            server_source=server_source_info.get("value"),
            server_args=list(state.get("serverArgs") or []),
            transport_type=state.get("transportType"),
            env=dict(state.get("env") or {}),
            tool_prefix=str(state.get("toolPrefix") or ""),
            auto_connect=bool(state.get("autoConnect", True)),
            client=client,
            auth_config=state.get("auth"),
            policy_context=state.get("policy"),
            include_resources=bool(state.get("includeResources", False)),
            resource_tool_prefix=state.get("resourceToolPrefix"),
            **dict(state.get("transportKwargs") or {}),
        )
        manager.restore_state(state)
        return manager


def build_mcp_hub_resource_tools(hub: MCPHub) -> List[Tool]:
    return [
        MCPHubListResourcesTool(hub),
        MCPHubReadResourceTool(hub),
    ]


def register_mcp_resource_hub_tools(
    registry: ToolRegistry,
    hub: MCPHub,
) -> tuple[Tool, Tool]:
    register_surface = getattr(registry, "register_runtime_surface", None)
    if callable(register_surface):
        register_surface("mcp_hub", "default", hub)
    registered: List[Tool] = []
    for tool in build_mcp_hub_resource_tools(hub):
        existing = registry.get_tool(tool.name)
        if existing is None:
            registry.register_tool(tool)
            registered.append(tool)
            continue
        registered.append(existing)
    return registered[0], registered[1]


def register_mcp_tools(
    registry: ToolRegistry,
    server_source: Any,
    server_args: Optional[List[str]] = None,
    transport_type: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tool_prefix: str = "",
    auto_connect: bool = True,
    client: Optional[MCPClientProtocol] = None,
    auth_config: Optional[Any] = None,
    policy_context: Optional[Any] = None,
    cache_store: Optional[Any] = None,
    max_retries: Optional[int] = None,
    persist_connection: Optional[bool] = None,
    *,
    include_resources: bool = False,
    resource_tool_prefix: Optional[str] = None,
    hub: Optional[MCPHub] = None,
    server_name: Optional[str] = None,
    legacy_resource_tools: Optional[bool] = None,
    **transport_kwargs: Any,
) -> MCPToolManager:
    manager = MCPToolManager(
        server_source=server_source,
        server_args=server_args,
        transport_type=transport_type,
        env=env,
        tool_prefix=tool_prefix,
        auto_connect=auto_connect,
        client=client,
        auth_config=auth_config,
        policy_context=policy_context,
        cache_store=cache_store,
        max_retries=max_retries,
        persist_connection=persist_connection,
        include_resources=include_resources,
        resource_tool_prefix=resource_tool_prefix,
        **transport_kwargs,
    )
    manager.register_to_registry(
        registry,
        hub=hub,
        server_name=server_name,
        legacy_resource_tools=legacy_resource_tools,
    )
    return manager


def mcptool(
    server_source: Any,
    server_args: Optional[List[str]] = None,
    transport_type: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tool_prefix: str = "",
    auto_connect: bool = True,
    client: Optional[MCPClientProtocol] = None,
    auth_config: Optional[Any] = None,
    policy_context: Optional[Any] = None,
    cache_store: Optional[Any] = None,
    max_retries: Optional[int] = None,
    persist_connection: Optional[bool] = None,
    include_resources: bool = False,
    resource_tool_prefix: Optional[str] = None,
    **transport_kwargs: Any,
) -> MCPToolManager:
    return MCPToolManager(
        server_source=server_source,
        server_args=server_args,
        transport_type=transport_type,
        env=env,
        tool_prefix=tool_prefix,
        auto_connect=auto_connect,
        client=client,
        auth_config=auth_config,
        policy_context=policy_context,
        cache_store=cache_store,
        max_retries=max_retries,
        persist_connection=persist_connection,
        include_resources=include_resources,
        resource_tool_prefix=resource_tool_prefix,
        **transport_kwargs,
    )
