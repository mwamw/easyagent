"""MCP tool integration for EasyAgent.

This module bridges MCP tools/resources into EasyAgent's ToolRegistry and
maps MCP prompts into SkillRegistry on-demand skills.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model

from skill import SkillRegistry

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


def _normalize_registry_name(value: str) -> str:
    return _sanitize_model_name(value).lower()


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
            "调用前先确认当前任务是否真的需要远程 MCP 能力，而不是本地 builtin tool、Skill 或已有上下文就能完成。",
            "参数必须完全遵守 schema；不要臆造字段，也不要把自然语言说明塞进结构化参数。",
            "如果这是高成本或高副作用操作，先缩小目标范围，再执行最小必要调用。",
        ]
    )

    if hints["read_only"]:
        lines.append("该 MCP 能力声明为只读，预期不会修改远程状态。")
    if hints["destructive"]:
        lines.append("该 MCP 能力声明可能产生远程副作用；调用前先确认目标、参数和必要性。")
    if hints["open_world"]:
        lines.append("该 MCP 能力声明依赖外部世界或外部系统状态；结果可能随时间变化，必要时重新验证。")
    if hints["idempotent_present"]:
        if hints["idempotent"]:
            lines.append("该 MCP 能力声明为幂等；在必要时可以安全重试。")
        else:
            lines.append("该 MCP 能力未声明为幂等；不要盲目重试或并行重复调用。")

    if capability_kind == "resource_read":
        lines.append("如资源列表项带有 annotations，应结合这些 annotations 判断资源是否适合当前任务。")
        lines.append("优先先列资源再读资源；不要在不知道 URI 是否存在时盲目读取。")
    elif capability_kind == "resource_list":
        lines.append("返回的是资源目录，不是资源正文；拿到 URI 后再决定是否继续读取。")
    elif capability_kind == "tool":
        lines.append("如果服务返回内容与本地推断冲突，应优先以远程返回为准，并在必要时解释冲突来源。")
    elif capability_kind in {"prompt_list", "prompt_get"}:
        lines.append("MCP prompt 是远程模板，不等于最终答案；应把它作为执行上下文的一部分来理解。")

    return "\n".join(line for line in lines if line)


def _normalize_prompt_arguments(arguments: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(arguments, list):
        return normalized

    for item in arguments:
        if isinstance(item, dict):
            normalized.append(
                {
                    "name": str(item.get("name", "")),
                    "description": str(item.get("description", "")),
                    "required": bool(item.get("required", False)),
                }
            )
            continue

        normalized.append(
            {
                "name": str(getattr(item, "name", "")),
                "description": str(getattr(item, "description", "")),
                "required": bool(getattr(item, "required", False)),
            }
        )
    return normalized


def _build_prompt_listing_description(prompt_name: str, prompt_info: Dict[str, Any]) -> str:
    description = str(prompt_info.get("description", "")).strip()
    args = _normalize_prompt_arguments(prompt_info.get("arguments"))
    if not args:
        return description or f"MCP prompt `{prompt_name}`"

    required = [item["name"] for item in args if item.get("required")]
    arg_note = f"参数: {', '.join(item['name'] for item in args if item['name'])}"
    if required:
        arg_note = f"必填参数: {', '.join(required)}"
    if description:
        return f"{description}；{arg_note}"
    return f"MCP prompt `{prompt_name}`；{arg_note}"


class MCPListResourcesParams(BaseModel):
    pass


class MCPReadResourceParams(BaseModel):
    uri: str = Field(description="要读取的 MCP 资源 URI")


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
            f"这是一个远程 MCP 工具 `{tool_name}`。\n"
            f"它映射到 MCP server `{_effective_server_name(manager)}` 上的原始能力 `{self.mcp_tool_name}`。\n"
            "调用它时要先判断当前问题是否确实需要远程能力，再按最小必要参数执行。\n"
            "如果你需要复用结果，优先引用返回里的结构化字段，而不是自行改写关键事实。"
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
            description=f"列出 MCP 服务 `{_effective_server_name(manager)}` 当前暴露的资源。",
            parameters=MCPListResourcesParams,
            guidance=_build_mcp_guidance((
                "当你需要浏览远程 MCP 资源目录、确认有哪些 URI 可读取时使用。\n"
                "先看资源目录，再决定是否继续读取具体资源；不要把它当成资源正文读取工具。"
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
            description=f"读取 MCP 服务 `{_effective_server_name(manager)}` 上指定 URI 的资源内容。",
            parameters=MCPReadResourceParams,
            guidance=_build_mcp_guidance((
                "先通过 `list_mcp_resources` 确认可用 URI，再读取具体资源。\n"
                "如果你还没确认 URI 是否存在，先不要直接读取。\n"
                "如果返回的是二进制内容，结果会以 base64 形式返回。"
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
            description="列出已注册 MCP server 暴露的资源；可按 server 过滤。",
            parameters=ClaudeListMcpResourcesInput,
            guidance=(
                "当需要浏览 MCP 资源目录时使用。\n"
                "若已知目标 server，优先显式提供 server；否则会聚合所有已注册 server 的资源。\n"
                "它返回的是资源目录，不是资源正文；拿到 URI 后再用 `ReadMcpResource` 读取。"
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
            description="读取指定 MCP server 上某个 URI 的资源内容。",
            parameters=ClaudeReadMcpResourceInput,
            guidance=(
                "先通过 `ListMcpResources` 确认可用 server 和 URI，再读取资源正文。\n"
                "如果你还不能确认 server/URI 是否正确，先不要盲读。\n"
                "读取结果来自远程 MCP server，应优先以远程返回为准。"
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
        self._registered_prompt_skills: List[str] = []
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

    def register_prompt_skills(
        self,
        registry: "SkillRegistry",
        *,
        skill_prefix: Optional[str] = None,
        replace_existing: bool = True,
    ) -> List[str]:
        from skill.builtin.mcp_skill import MCPPromptSkill

        try:
            prompt_infos = self.list_remote_prompts()
        except Exception as e:
            logger.debug(f"Failed to list remote prompts for {self.server_label}: {e}")
            prompt_infos = []
        prefix = skill_prefix if skill_prefix is not None else f"{self.server_label}_"

        registered: List[str] = []
        for prompt_info in prompt_infos:
            prompt_name = str(prompt_info.get("name", "")).strip()
            if not prompt_name:
                continue

            skill_name = _prefixed_name(prefix, _normalize_registry_name(prompt_name))
            if registry.has(skill_name):
                manifest = registry.get_manifest(skill_name)
                if manifest.source_type != "mcp_prompt":
                    logger.warning("Skill '%s' 已存在且不是 mcp_prompt，跳过注册", skill_name)
                    continue
                if replace_existing:
                    registry.unregister(skill_name)
                else:
                    registered.append(skill_name)
                    continue

            normalized_prompt_info = {
                "name": prompt_name,
                "description": str(prompt_info.get("description", "") or ""),
                "arguments": _normalize_prompt_arguments(prompt_info.get("arguments")),
            }

            def _factory(
                _skill_name: str = skill_name,
                _prompt_name: str = prompt_name,
                _prompt_info: Dict[str, Any] = normalized_prompt_info,
                **kwargs: Any,
            ):
                return MCPPromptSkill(
                    manager=self,
                    prompt_name=_prompt_name,
                    prompt_info=_prompt_info,
                    skill_name=_skill_name,
                    prompt_arguments=kwargs.pop("prompt_arguments", None),
                )

            registry.register_factory(skill_name, _factory)
            registry.update_metadata(
                skill_name,
                description=normalized_prompt_info["description"] or f"MCP prompt: {prompt_name}",
                tags=["mcp", "prompt", self.server_label],
                listing_description=_build_prompt_listing_description(prompt_name, normalized_prompt_info),
                when_to_use=(
                    normalized_prompt_info["description"]
                    or f"当需要使用 MCP prompt `{prompt_name}` 的远程指令模板时"
                ),
                exposure_mode="on_demand",
                execution_mode="inline",
                source_type="mcp_prompt",
                source_path=f"mcp://{self.server_label}/prompts/{prompt_name}",
                tool_names=[],
                mcp_server=self.server_label,
                mcp_prompt_name=prompt_name,
                mcp_prompt_arguments=normalized_prompt_info["arguments"],
                source_identifier=f"mcp://{self.server_label}/prompts/{prompt_name}",
            )
            registered.append(skill_name)

        self._registered_prompt_skills = list(registered)
        return registered

    def get_wrapped_tools(self) -> List[MCPWrappedTool]:
        return list(self._wrapped_tools)

    def get_resource_tools(self) -> List[Tool]:
        return list(self._resource_tools)

    def get_registered_prompt_skills(self) -> List[str]:
        return list(self._registered_prompt_skills)

    def export_state(self) -> dict[str, Any]:
        payload = super().export_state()
        payload.update(
            {
                "includeResources": self.include_resources,
                "resourceToolPrefix": self.resource_tool_prefix,
                "registryServerName": self._registered_server_name,
                "registeredPromptSkills": list(self._registered_prompt_skills),
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
        self._registered_prompt_skills = [
            str(item).strip()
            for item in list(state.get("registeredPromptSkills") or [])
            if str(item).strip()
        ]
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
