from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool


class ToolSchemaToolParams(BaseModel):
    tool_names: list[str] = Field(
        default_factory=list,
        description="Names of tools whose full schemas should be loaded for this invocation",
    )


class ToolSchemaTool(Tool):
    def __init__(self, registry: Any):
        super().__init__(
            name="tool_schema_tool",
            description=(
                "Loads full schemas for one or more deferred tools. If a tool is listed but its callable schema is not present, load it here before invoking it in a subsequent model response."
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
                "Tool schemas have been loaded for the current invocation. You may call these tools "
                "in subsequent model responses in this invocation; load them again in a new invocation if needed."
            ),
        }


def register_tool_schema_tool(registry: Any) -> ToolSchemaTool:
    tool = ToolSchemaTool(registry)
    registry.register_tool(tool, visibility="resident")
    return tool
