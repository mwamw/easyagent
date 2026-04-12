"""MCP tool integration for EasyAgent.

This module bridges MCP tools/resources into EasyAgent's ToolRegistry and
maps MCP prompts into SkillRegistry on-demand skills.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model

from skill import SkillRegistry

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from mcp.runtime import MCPClientProtocol, MCPRuntimeManager

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
            "严格按照参数 schema 提供输入，不要臆造额外字段。\n"
            "结果来自外部服务或远程进程；若结果与本地上下文冲突，以最新返回结果为准。"
        )

        super().__init__(
            name=tool_name,
            description=description,
            parameters=parameters,
            guidance=_build_mcp_guidance(base_guidance, annotations, capability_kind="tool"),
            source="mcp",
            read_only=hint_state["read_only"],
            destructive=hint_state["destructive"],
            requires_confirmation=hint_state["destructive"] or hint_state["open_world"],
            supports_parallel=(
                hint_state["idempotent"]
                if hint_state["idempotent_present"]
                else (hint_state["read_only"] and not hint_state["open_world"] and not hint_state["destructive"])
            ),
            tags=["mcp", "remote"],
            metadata={
                "mcp_tool_name": self.mcp_tool_name,
                "mcp_server": manager.server_label,
                "mcp_annotations": annotations,
                "mcp_read_only": hint_state["read_only"],
                "mcp_destructive": hint_state["destructive"],
                "mcp_open_world": hint_state["open_world"],
                "mcp_idempotent": hint_state["idempotent"] if hint_state["idempotent_present"] else None,
            },
        )

    def run(self, parameters: dict):
        result = self.manager.execute_tool(self.mcp_tool_name, parameters)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, (dict, list)):
            return ToolResult.success(
                structured_data=result,
                metadata={"mcp_tool_name": self.mcp_tool_name},
            )
        if result is None:
            return ToolResult.success("", metadata={"mcp_tool_name": self.mcp_tool_name})
        return ToolResult.success(
            str(result),
            metadata={"mcp_tool_name": self.mcp_tool_name},
        )


class MCPListResourcesTool(Tool):
    """List MCP resources exposed by one server."""

    def __init__(self, manager: "MCPToolManager", prefix: str = ""):
        self.manager = manager
        super().__init__(
            name=_prefixed_name(prefix, "list_mcp_resources"),
            description=f"列出 MCP 服务 `{manager.server_label}` 当前暴露的资源。",
            parameters=MCPListResourcesParams,
            guidance=_build_mcp_guidance((
                "当你需要浏览远程 MCP 资源目录、先确认有哪些资源可读时使用。"
                "它只返回资源清单，不返回资源正文。"
            ), {}, capability_kind="resource_list"),
            read_only=True,
            source="mcp",
            tags=["mcp", "resource", "read"],
            supports_parallel=True,
            metadata={"mcp_server": manager.server_label, "mcp_resource_tool": True},
        )

    def run(self, parameters: dict) -> ToolResult:
        resources = self.manager.list_remote_resources()
        return ToolResult.success(
            structured_data=resources,
            metadata={"mcp_server": self.manager.server_label},
        )


class MCPReadResourceTool(Tool):
    """Read one MCP resource."""

    def __init__(self, manager: "MCPToolManager", prefix: str = ""):
        self.manager = manager
        super().__init__(
            name=_prefixed_name(prefix, "read_mcp_resource"),
            description=f"读取 MCP 服务 `{manager.server_label}` 上指定 URI 的资源内容。",
            parameters=MCPReadResourceParams,
            guidance=_build_mcp_guidance((
                "先通过 `list_mcp_resources` 确认可用 URI，再读取具体资源。"
                "如果返回的是二进制内容，会以 base64 文本形式返回。"
            ), {}, capability_kind="resource_read"),
            read_only=True,
            source="mcp",
            tags=["mcp", "resource", "read"],
            supports_parallel=True,
            metadata={"mcp_server": manager.server_label, "mcp_resource_tool": True},
        )

    def run(self, parameters: dict) -> ToolResult:
        uri = str(parameters.get("uri", "")).strip()
        if not uri:
            return ToolResult.error("错误：必须提供资源 URI。", error_type="invalid_parameters")

        result = self.manager.read_remote_resource(uri)
        if isinstance(result, bytes):
            encoded = base64.b64encode(result).decode("ascii")
            return ToolResult.success(
                structured_data={
                    "uri": uri,
                    "encoding": "base64",
                    "content": encoded,
                },
                metadata={"mcp_server": self.manager.server_label, "uri": uri},
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
            return ToolResult.success(
                structured_data={"uri": uri, "content": normalized},
                metadata={"mcp_server": self.manager.server_label, "uri": uri},
            )
        return ToolResult.success(
            content=str(result or ""),
            metadata={"mcp_server": self.manager.server_label, "uri": uri},
        )


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
            **transport_kwargs,
        )
        self.include_resources = include_resources
        self.resource_tool_prefix = resource_tool_prefix
        self._wrapped_tools: List[MCPWrappedTool] = []
        self._resource_tools: List[Tool] = []
        self._registered_prompt_skills: List[str] = []

    def build_resource_tools(self, resource_tool_prefix: Optional[str] = None) -> List[Tool]:
        prefix = resource_tool_prefix
        if prefix is None:
            prefix = self.tool_prefix or f"{self.server_label}_"
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
    ) -> List[Tool]:
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

        if should_include_resources:
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


def register_mcp_tools(
    registry: ToolRegistry,
    server_source: Any,
    server_args: Optional[List[str]] = None,
    transport_type: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tool_prefix: str = "",
    auto_connect: bool = True,
    *,
    include_resources: bool = False,
    resource_tool_prefix: Optional[str] = None,
    **transport_kwargs: Any,
) -> MCPToolManager:
    manager = MCPToolManager(
        server_source=server_source,
        server_args=server_args,
        transport_type=transport_type,
        env=env,
        tool_prefix=tool_prefix,
        auto_connect=auto_connect,
        include_resources=include_resources,
        resource_tool_prefix=resource_tool_prefix,
        **transport_kwargs,
    )
    manager.register_to_registry(registry)
    return manager


def mcptool(
    server_source: Any,
    server_args: Optional[List[str]] = None,
    transport_type: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tool_prefix: str = "",
    auto_connect: bool = True,
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
        include_resources=include_resources,
        resource_tool_prefix=resource_tool_prefix,
        **transport_kwargs,
    )
