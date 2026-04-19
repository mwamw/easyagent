from __future__ import annotations

from functools import wraps
from typing import Any, Literal, Type

from pydantic import BaseModel

from .BaseTool import Tool, ToolResult, ToolSpec
from core.permissions import PermissionBehavior, PermissionContext, PermissionEngine


ToolVisibility = Literal["resident", "runtime", "turn"]
from core.Exception import ToolNotFoundError
class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._tool_visibility: dict[str, ToolVisibility] = {}

    def register_tool(self, tool: Tool, *, visibility: ToolVisibility = "resident"):
        self.tools[tool.name] = tool
        self._tool_visibility[tool.name] = visibility

    def mount_runtime_tool(self, tool: Tool):
        self.register_tool(tool, visibility="runtime")
        return tool

    def mount_turn_tool(self, tool: Tool):
        self.register_tool(tool, visibility="turn")
        return tool

    def clear_runtime_tools(self) -> None:
        names = [
            name for name, visibility in self._tool_visibility.items()
            if visibility in {"runtime", "turn"}
        ]
        for name in names:
            self.unregister_tool(name)

    def registry(self, item):
        """兼容注册入口：支持 Tool 实例或带 register_to_registry 的对象。"""
        if isinstance(item, Tool):
            self.register_tool(item)
            return item

        register_fn = getattr(item, "register_to_registry", None)
        if callable(register_fn):
            return register_fn(self)

        raise ValueError("registry(...) 仅支持 Tool 实例或可注册对象")

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

    def get_visible_tools(self, scope: str = "all") -> list[Tool]:
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
        return [self.tools[name] for name in names if name in self.tools]

    def list_tool_specs(self, scope: str = "all") -> list[ToolSpec]:
        return [tool.get_spec() for tool in self.get_visible_tools(scope=scope)]

    def get_tool_spec(self, name: str) -> ToolSpec | None:
        tool = self.get_tool(name)
        return tool.get_spec() if tool else None

    def get_tools_description(self) -> list[dict[str, Any]]:
        return [spec.to_description_payload() for spec in self.list_tool_specs()]

    def validate_tool_call(self, name: str, parameters: dict[str, Any]) -> tuple[Tool, dict[str, Any]]:
        tool = self.get_tool(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool {name} not found")
        if not isinstance(parameters, dict):
            raise ValueError(f"工具参数必须是 dict，收到: {type(parameters).__name__}")
        validated = tool.validate_parameters(parameters)
        return tool, validated

    def authorize_tool_call(
        self,
        tool: Tool,
        parameters: dict[str, Any],
        *,
        permission_context: PermissionContext | None = None,
        permission_engine: PermissionEngine | None = None,
    ) -> ToolResult | None:
        spec = tool.get_spec()
        effective_engine = permission_engine or (PermissionEngine() if permission_context is not None else None)
        if effective_engine is not None:
            decision = effective_engine.authorize(tool, parameters, permission_context)
            if decision.behavior == PermissionBehavior.DENY:
                return ToolResult.error(
                    decision.reason,
                    error_type="permission_denied",
                    metadata={
                        "tool_name": tool.name,
                        "parameters": parameters,
                        "permission_behavior": decision.behavior.value,
                        "permission_reason": decision.reason,
                        "matched_rule_source": decision.matched_rule_source,
                        "risk_categories": list(decision.risk_categories),
                    },
                )
            if decision.behavior == PermissionBehavior.ASK:
                return ToolResult.needs_confirmation(
                    decision.reason,
                    metadata={
                        "tool_name": tool.name,
                        "parameters": parameters,
                        "permission_behavior": decision.behavior.value,
                        "permission_reason": decision.reason,
                        "matched_rule_source": decision.matched_rule_source,
                        "risk_categories": list(decision.risk_categories),
                        "requires_confirmation": True,
                    },
                )
            return None
        if spec.requires_confirmation:
            return ToolResult.needs_confirmation(
                f"工具 '{tool.name}' 需要用户确认后才能执行。",
                metadata={
                    "tool_name": tool.name,
                    "parameters": parameters,
                    "requires_confirmation": True,
                },
            )
        return None

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
        *,
        permission_context: PermissionContext | None = None,
        permission_engine: PermissionEngine | None = None,
    ) -> ToolResult:
        try:
            tool, validated = self.validate_tool_call(name, parameters)
            auth_result = self.authorize_tool_call(
                tool,
                validated,
                permission_context=permission_context,
                permission_engine=permission_engine,
            )
            if auth_result is not None:
                self._enrich_tool_result(tool, auth_result)
                return auth_result
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
        if spec.risk_categories:
            metadata.setdefault("risk_categories", list(spec.risk_categories))
        if spec.demand_skill_tool:
            metadata.setdefault("demand_skill_tool", True)
            if spec.demand_skill_name:
                metadata.setdefault("demand_skill_name", spec.demand_skill_name)
            note = "注意：该工具来自按需 Skill 的临时挂载，只在当前请求实际提供的 tools 集合中可用；后续再次调用前必须确认它当前仍然存在。"
            base_text = result.to_display_string().strip()
            result.display_text = f"{note}\n\n执行结果:\n{base_text}" if base_text else note
        result.metadata = metadata

    def execute_tool(self, name: str, parameters: dict):
        result = self.execute_tool_result(name, parameters)
        return result.to_display_string()

    def export_tools(self, provider_name: str = "openai", *, scope: str = "all") -> Any:
        from core.providers.tool_schema import create_tool_schema_adapter

        adapter = create_tool_schema_adapter(provider_name)
        return adapter.export_tools(self.get_visible_tools(scope=scope))

    def get_tools_for_provider(self, provider_name: str, *, scope: str = "all") -> Any:
        return self.export_tools(provider_name, scope=scope)

    def get_openai_tools(self) -> list[dict]:
        return self.export_tools("openai")

    def unregister_tool(self, name: str):
        if name in self.tools:
            del self.tools[name]
            self._tool_visibility.pop(name, None)
        else:
            print(f"Tool {name} not found")

    def register_tools(self, tools: list) -> None:
        """批量注册多个工具。"""
        for tool in tools:
            self.register_tool(tool)

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

    # ==================== 向后兼容别名 ====================

    def registerTool(self, tool: Tool):
        """向后兼容：请改用 register_tool。"""
        return self.register_tool(tool)

    def executeTool(self, name: str, parameters: dict):
        """向后兼容：请改用 execute_tool。"""
        return self.execute_tool(name, parameters)

    def get_Tool(self, name: str):
        """向后兼容：请改用 get_tool。"""
        return self.get_tool(name)

    def disregister_tool(self, name: str):
        """向后兼容：请改用 unregister_tool。"""
        return self.unregister_tool(name)
