from __future__ import annotations

from functools import wraps
from typing import Any, Literal, Type

from pydantic import BaseModel

from .BaseTool import Tool, ToolResult, ToolSpec
from core.permissions.base import BasePermissionEngine
from core.permissions.models import AuthorizationContext, PermissionRequest, ToolPermissionDescriptor


ToolVisibility = Literal["resident", "runtime", "turn"]
ToolConflictPolicy = Literal["replace", "error", "keep_existing"]
from core.Exception import ToolNotFoundError, ToolRegistryError
class ToolRegistry:
    def __init__(self, *, conflict_policy: ToolConflictPolicy = "replace"):
        self.tools: dict[str, Tool] = {}
        self._permission_descriptors: dict[str, ToolPermissionDescriptor] = {}
        self._tool_visibility: dict[str, ToolVisibility] = {}
        self._runtime_surfaces: dict[str, dict[str, Any]] = {}
        self._deferred_expanded_tool_names: set[str] = set()
        self.conflict_policy: ToolConflictPolicy = conflict_policy

    def register_tool(
        self,
        tool: Tool,
        *,
        visibility: ToolVisibility = "resident",
        expose_in_deferred: bool | None = None,
        permission_descriptor: ToolPermissionDescriptor | None = None,
        conflict_policy: ToolConflictPolicy | None = None,
    ):
        policy = conflict_policy or self.conflict_policy
        existing = self.tools.get(tool.name)
        if existing is not None:
            if policy == "keep_existing":
                return existing
            if policy == "error":
                raise ToolRegistryError(f"Tool '{tool.name}' 已存在，当前冲突策略禁止覆盖。")
        self.tools[tool.name] = tool
        if permission_descriptor is not None:
            if permission_descriptor.tool_name != tool.name:
                raise ToolRegistryError(
                    f"Permission descriptor name '{permission_descriptor.tool_name}' does not match tool '{tool.name}'."
                )
            self._permission_descriptors[tool.name] = permission_descriptor
        self._tool_visibility[tool.name] = visibility
        tool.spec.visibility_scope = visibility
        if expose_in_deferred is not None:
            tool.spec.expose_in_deferred = bool(expose_in_deferred)
        return tool

    def mount_runtime_tool(self, tool: Tool, *, expose_in_deferred: bool | None = None):
        return self.register_tool(tool, visibility="runtime", expose_in_deferred=expose_in_deferred)

    def mount_turn_tool(self, tool: Tool, *, expose_in_deferred: bool | None = None):
        return self.register_tool(tool, visibility="turn", expose_in_deferred=expose_in_deferred)

    def clear_runtime_tools(self) -> None:
        names = [
            name for name, visibility in self._tool_visibility.items()
            if visibility in {"runtime", "turn"}
        ]
        for name in names:
            self.unregister_tool(name)

    def tool(self, name: str, description: str, parameters: Type[BaseModel]):
        """装饰器：注册函数为工具。"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            class FunctionTool(Tool):
                def run(self, parameters: dict):
                    return func(**parameters)

            tool_instance = FunctionTool(name, description, parameters)
            self.register_tool(tool_instance)
            return wrapper

        return decorator

    def _stable_tool_sort_key(self, name: str) -> tuple[int, int, str]:
        visibility_rank = {"resident": 0, "runtime": 1, "turn": 2}
        tool = self.tools.get(name)
        source_rank = 1
        if tool is not None:
            try:
                source = tool.get_spec().source
                if source == "builtin":
                    source_rank = 0
            except Exception:
                source_rank = 1
        return (
            visibility_rank.get(self._tool_visibility.get(name, "resident"), 99),
            source_rank,
            name,
        )

    def _visible_tool_names(self, scope: str = "all", *, stable: bool = False) -> list[str]:
        if scope == "all":
            names = list(self.tools.keys())
        elif scope == "resident":
            names = [
                name for name, visibility in self._tool_visibility.items()
                if visibility == "resident"
            ]
        elif scope == "runtime":
            names = [
                name for name, visibility in self._tool_visibility.items()
                if visibility == "runtime"
            ]
        elif scope == "turn":
            names = [
                name for name, visibility in self._tool_visibility.items()
                if visibility == "turn"
            ]
        else:
            raise ValueError(f"未知工具可见性 scope: {scope}")
        if stable:
            names.sort(key=self._stable_tool_sort_key)
        return names

    def get_visible_tools(self, scope: str = "all", *, stable: bool = False) -> list[Tool]:
        names = self._visible_tool_names(scope=scope, stable=stable)
        return [self.tools[name] for name in names if name in self.tools]

    def list_tool_specs(self, scope: str = "all", *, stable: bool = False) -> list[ToolSpec]:
        return [tool.get_spec() for tool in self.get_visible_tools(scope=scope, stable=stable)]

    def get_tool_spec(self, name: str) -> ToolSpec | None:
        tool = self.get_tool(name)
        return tool.get_spec() if tool else None

    def get_permission_descriptor(self, name: str) -> ToolPermissionDescriptor | None:
        tool = self.get_tool(name)
        if tool is None:
            return None
        return self._permission_descriptors.get(name) or tool.get_permission_descriptor()

    def resolve_tool_call(self, name: str, parameters: dict[str, Any]):
        """Resolve and validate a model call without authorizing or executing it."""
        tool, validated = self.validate_tool_call(name, parameters)
        descriptor = self.get_permission_descriptor(tool.name)
        if descriptor is None:
            raise ToolRegistryError(f"Tool '{tool.name}' has no permission descriptor")
        return tool, validated, descriptor

    def build_permission_request(
        self,
        name: str,
        parameters: dict[str, Any],
        *,
        context: AuthorizationContext | None = None,
    ) -> PermissionRequest:
        tool, validated, descriptor = self.resolve_tool_call(name, parameters)
        return PermissionRequest(
            tool_name=tool.name,
            arguments=validated,
            descriptor=descriptor,
            context=context or AuthorizationContext(),
        )

    def authorize_request(
        self,
        request: PermissionRequest,
        *,
        permission_engine: BasePermissionEngine,
    ):
        """Authorize a normalized request through the new framework contract."""
        return permission_engine.authorize(request)

    def get_tools_description(self) -> list[dict[str, Any]]:
        return self.list_tool_descriptors(stable=True, include_parameters=True)

    def list_tool_descriptors(
        self,
        scope: str = "all",
        *,
        stable: bool = False,
        include_parameters: bool = False,
        include_internal: bool = False,
    ) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        for tool in self.get_visible_tools(scope=scope, stable=stable):
            spec = tool.get_spec()
            if not include_internal and spec.metadata.get("internal_tool"):
                continue
            descriptor = spec.to_description_payload()
            permission = tool.get_permission_descriptor()
            descriptor.update({
                "requires_confirmation": permission.requires_confirmation,
                "risk_categories": list(permission.risk_categories),
                "side_effect_level": permission.side_effect_level,
                "resource_scope": list(permission.resources),
            })
            if not include_parameters:
                descriptor.pop("parameters", None)
            descriptors.append(descriptor)
        return descriptors

    def expand_deferred_tools(self, names: list[str]) -> list[ToolSpec]:
        expanded: list[ToolSpec] = []
        for name in names:
            normalized = str(name or "").strip()
            if not normalized:
                continue
            tool = self.get_tool(normalized)
            if tool is None:
                raise ToolNotFoundError(f"Tool {normalized} not found")
            self._deferred_expanded_tool_names.add(normalized)
            expanded.append(tool.get_spec())
        return expanded

    def clear_deferred_tool_expansions(self) -> None:
        self._deferred_expanded_tool_names.clear()

    def get_deferred_expanded_tool_names(self) -> list[str]:
        return sorted(self._deferred_expanded_tool_names, key=self._stable_tool_sort_key)

    def validate_tool_call(self, name: str, parameters: dict[str, Any]) -> tuple[Tool, dict[str, Any]]:
        tool = self.get_tool(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool {name} not found")
        if not isinstance(parameters, dict):
            raise ValueError(f"工具参数必须是 dict，收到: {type(parameters).__name__}")
        validated = tool.validate_parameters(parameters)
        return tool, validated

    def normalize_tool_result(self, name: str, raw_result: Any) -> ToolResult:
        if isinstance(raw_result, ToolResult):
            return raw_result
        if raw_result is None:
            return ToolResult.success("工具执行完成，无返回结果")
        if isinstance(raw_result, (dict, list)):
            return ToolResult.success(
                structured_data=raw_result,
                metadata={"tool_name": name},
            )
        return ToolResult.success(str(raw_result), metadata={"tool_name": name})

    def execute_tool_result(
        self,
        name: str,
        parameters: dict[str, Any],
    ) -> ToolResult:
        try:
            tool, validated = self.validate_tool_call(name, parameters)
            raw_result = tool.run(validated)
            result = self.normalize_tool_result(name, raw_result)
            self._enrich_tool_result(tool, result)
        except ToolNotFoundError as e:
            result = ToolResult.error(str(e), error_type="tool_not_found")
            raise e
        return result

    def _enrich_tool_result(self, tool: Tool, result: ToolResult) -> None:
        spec = tool.get_spec()
        metadata = dict(result.metadata)
        metadata.setdefault("tool_name", tool.name)
        metadata.setdefault("tool_visibility", self.get_tool_visibility(tool.name))
        metadata.setdefault("tool_source", spec.source)
        permission = tool.get_permission_descriptor()
        metadata.setdefault("side_effect_level", permission.side_effect_level)
        metadata.setdefault("resource_scope", list(permission.resources))
        metadata.setdefault("visibility_scope", spec.visibility_scope)
        source_identifier = spec.metadata.get("source_identifier")
        if source_identifier:
            metadata.setdefault("source_identifier", source_identifier)
        if permission.risk_categories:
            metadata.setdefault("risk_categories", list(permission.risk_categories))
        result.metadata = metadata

    def execute_tool(self, name: str, parameters: dict):
        result = self.execute_tool_result(name, parameters)
        return result.to_display_string()

    def register_runtime_surface(self, kind: str, name: str, surface: Any) -> Any:
        normalized_kind = str(kind or "").strip()
        normalized_name = str(name or "").strip()
        if not normalized_kind:
            raise ToolRegistryError("runtime surface kind 不能为空。")
        if not normalized_name:
            raise ToolRegistryError("runtime surface name 不能为空。")
        bucket = self._runtime_surfaces.setdefault(normalized_kind, {})
        bucket[normalized_name] = surface
        return surface

    def get_runtime_surface(self, kind: str, name: str) -> Any:
        bucket = self._runtime_surfaces.get(str(kind or "").strip(), {})
        return bucket.get(str(name or "").strip())

    def list_runtime_surfaces(self, kind: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
        if kind is None:
            return {
                surface_kind: dict(items)
                for surface_kind, items in self._runtime_surfaces.items()
            }
        return dict(self._runtime_surfaces.get(str(kind or "").strip(), {}))

    def unregister_runtime_surface(self, kind: str, name: str) -> None:
        bucket = self._runtime_surfaces.get(str(kind or "").strip())
        if not bucket:
            return
        bucket.pop(str(name or "").strip(), None)
        if not bucket:
            self._runtime_surfaces.pop(str(kind or "").strip(), None)

    def export_tools(
        self,
        provider_name: str = "openai",
        *,
        scope: str = "all",
        mode: str = "full",
        selected_names: list[str] | None = None,
    ) -> Any:
        from core.providers.tool_schema import create_tool_schema_adapter

        adapter = create_tool_schema_adapter(provider_name)
        visible_tools = self.get_visible_tools(scope=scope, stable=True)
        if mode == "deferred":
            explicit = {str(name).strip() for name in list(selected_names or []) if str(name or "").strip()}
            expanded = set(self.get_deferred_expanded_tool_names())
            allowed_names = expanded | explicit
            filtered_tools: list[Tool] = []
            for tool in visible_tools:
                spec = tool.get_spec()
                if spec.expose_in_deferred:
                    filtered_tools.append(tool)
                    continue
                if tool.name in allowed_names:
                    filtered_tools.append(tool)
            return adapter.export_tools(filtered_tools)
        return adapter.export_tools(visible_tools)

    def get_tools_for_provider(self, provider_name: str, *, scope: str = "all") -> Any:
        return self.export_tools(provider_name, scope=scope)

    def get_openai_tools(self) -> list[dict]:
        return self.export_tools("openai")

    def unregister_tool(self, name: str):
        if name in self.tools:
            del self.tools[name]
            self._tool_visibility.pop(name, None)
            self._permission_descriptors.pop(name, None)
        else:
            print(f"Tool {name} not found")

    def register_tools(
        self,
        tools: list,
        *,
        visibility: ToolVisibility = "resident",
        expose_in_deferred: bool | None = None,
        conflict_policy: ToolConflictPolicy | None = None,
    ) -> None:
        """批量注册多个工具。"""
        for tool in tools:
            self.register_tool(
                tool,
                visibility=visibility,
                expose_in_deferred=expose_in_deferred,
                conflict_policy=conflict_policy,
            )

    def unregister_tools(self, names: list) -> None:
        """批量移除多个工具。"""
        for name in names:
            self.unregister_tool(name)

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self.tools

    def get_tool_names(self) -> list[str]:
        """获取所有已注册工具名称。"""
        return list(self.tools.keys())

    def get_tool(self, name: str):
        return self.tools.get(name)

    def get_tool_visibility(self, name: str) -> ToolVisibility | None:
        return self._tool_visibility.get(name)
