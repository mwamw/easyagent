from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool


class ToolSchemaToolParams(BaseModel):
    tool_names: list[str] = Field(
        default_factory=list,
        description="要为当前轮展开完整 schema 的工具名列表。",
    )


class ToolSchemaTool(Tool):
    def __init__(self, registry: Any):
        super().__init__(
            name="tool_schema_tool",
            description=(
                "按需展开一个或多个 deferred tools 的完整 schema。"
                "当工具目录里已经出现候选工具，但当前 tools 集合里还没有它们的可调用 schema 时，"
                "先调用此工具展开所需工具，再在后续回合真正调用这些工具。"
            ),
            parameters=ToolSchemaToolParams,
            read_only=True,
            source="builtin",
            expose_in_deferred=True,
            metadata={
                "internal_tool": True,
            },
        )
        self._registry = registry

    def run(self, parameters: dict) -> dict[str, Any]:
        tool_names = [str(name or "").strip() for name in list(parameters.get("tool_names") or [])]
        tool_names = [name for name in tool_names if name and name != self.name]
        expanded_specs = self._registry.expand_deferred_tools(tool_names)
        return {
            "expandedToolNames": [spec.name for spec in expanded_specs],
            "toolDescriptors": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "tags": list(spec.tags),
                    "read_only": spec.read_only,
                    "requires_confirmation": spec.requires_confirmation,
                    "visibility_scope": spec.visibility_scope,
                }
                for spec in expanded_specs
            ],
            "message": (
                "已为当前 invoke 展开工具 schema。后续同一轮推理可直接调用这些工具；"
                "新的 invoke 如需继续使用，应再次展开。"
            ),
        }


def register_tool_schema_tool(registry: Any) -> ToolSchemaTool:
    tool = ToolSchemaTool(registry)
    registry.register_tool(tool, visibility="resident")
    return tool
