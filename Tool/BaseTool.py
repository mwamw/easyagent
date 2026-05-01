from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel

ToolResultStatus = Literal["success", "error", "needs_confirmation"]
ToolOutputMode = Literal["text", "json", "markdown"]
ToolSideEffectLevel = Literal["none", "low", "medium", "high"]
ToolVisibilityScope = Literal["resident", "runtime", "turn"]
# 兼容字段：早期版本曾用它控制 tool prompt 是否注入 system/runtime prompt。
# 当前框架主路径不再读取该字段；tool prompt 会统一折叠进 tool schema 的 description。
ToolPromptVisibility = Literal["none", "resident", "runtime"]


@dataclass(slots=True)
class ToolSpec:
    """框架内部统一使用的工具元数据。"""

    name: str
    description: str
    parameters_model: Type[BaseModel]
    guidance: str = ""
    read_only: bool = False
    destructive: bool = False
    requires_confirmation: bool = False
    supports_parallel: bool = True
    output_mode: ToolOutputMode = "text"
    source: str = "custom"
    ephemeral: bool = False
    prompt: str = ""
    prompt_visibility: ToolPromptVisibility = "none"
    demand_skill_tool: bool = False
    demand_skill_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    risk_categories: list[str] = field(default_factory=list)
    side_effect_level: ToolSideEffectLevel = "none"
    resource_scope: list[str] = field(default_factory=list)
    visibility_scope: ToolVisibilityScope = "resident"
    expose_in_deferred: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def parameter_schema(self) -> dict[str, Any]:
        """返回清洗后的参数 schema。"""
        schema = self.parameters_model.model_json_schema()
        defs = schema.pop("$defs", {})

        def resolve_schema(schema_node: Any) -> Any:
            if isinstance(schema_node, dict):
                if "$ref" in schema_node:
                    ref_key = schema_node["$ref"].split("/")[-1]
                    resolved = defs.get(ref_key, {}).copy()
                    for key, value in schema_node.items():
                        if key != "$ref" and key not in resolved:
                            resolved[key] = value
                    return resolve_schema(resolved)

                if "anyOf" in schema_node:
                    non_null_schemas = [
                        item for item in schema_node["anyOf"]
                        if item.get("type") != "null"
                    ]
                    if len(non_null_schemas) == 1:
                        merged = {
                            key: value for key, value in schema_node.items()
                            if key not in {"anyOf", "default"}
                        }
                        resolved_child = resolve_schema(non_null_schemas[0])
                        for key, value in resolved_child.items():
                            merged[key] = value
                        return merged
                    schema_node["anyOf"] = [resolve_schema(item) for item in schema_node["anyOf"]]

                if isinstance(schema_node.get("title"), str):
                    schema_node.pop("title", None)
                for key, value in list(schema_node.items()):
                    schema_node[key] = resolve_schema(value)
                return schema_node

            if isinstance(schema_node, list):
                return [resolve_schema(item) for item in schema_node]

            return schema_node

        return resolve_schema(schema)

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI 风格 function calling schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.build_schema_description(),
                "parameters": self.parameter_schema(),
            },
        }

    def build_schema_description(self) -> str:
        """构建发送给模型的工具 description。"""
        parts: list[str] = []
        description = self.description.strip()
        guidance = self.guidance.strip()
        prompt = self.prompt.strip()

        if description:
            parts.append(description)
        if guidance:
            parts.append(guidance)
        if prompt:
            parts.append(prompt)
        if self.demand_skill_tool:
            note = "该工具来自按需 Skill 的临时挂载，不会常驻在 tools 集合中；只有它当前实际出现在本次请求提供的 tools 列表里时，才可以调用。"
            if self.demand_skill_name:
                note += f" 若后续还需要它，应先重新调用 `skill_tool(skill_name=\"{self.demand_skill_name}\")`。"
            parts.append(note)

        body = "\n\n".join(part for part in parts if part)
        return body or self.name

    def to_description_payload(self) -> dict[str, Any]:
        """转成易于 prompt/listing 使用的结构化描述。"""
        return {
            "type": "tool",
            "name": self.name,
            "description": self.description,
            "guidance": self.guidance,
            "read_only": self.read_only,
            "destructive": self.destructive,
            "requires_confirmation": self.requires_confirmation,
            "supports_parallel": self.supports_parallel,
            "output_mode": self.output_mode,
            "source": self.source,
            "ephemeral": self.ephemeral,
            "has_prompt": bool(self.prompt.strip()),
            "prompt_visibility": self.prompt_visibility,
            "demand_skill_tool": self.demand_skill_tool,
            "demand_skill_name": self.demand_skill_name,
            "tags": list(self.tags),
            "risk_categories": list(self.risk_categories),
            "side_effect_level": self.side_effect_level,
            "resource_scope": list(self.resource_scope),
            "visibility_scope": self.visibility_scope,
            "expose_in_deferred": self.expose_in_deferred,
            "parameters": self.parameter_schema(),
            "metadata": dict(self.metadata),
        }

    def to_intermediate_schema(self) -> dict[str, Any]:
        """返回 provider 适配层消费的统一工具描述。"""
        return {
            "name": self.name,
            "description": self.build_schema_description(),
            "parameters": self.parameter_schema(),
            "output_mode": self.output_mode,
            "source": self.source,
            "ephemeral": self.ephemeral,
            "tags": list(self.tags),
            "risk_categories": list(self.risk_categories),
            "side_effect_level": self.side_effect_level,
            "resource_scope": list(self.resource_scope),
            "visibility_scope": self.visibility_scope,
            "expose_in_deferred": self.expose_in_deferred,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ToolResult:
    """统一的工具执行结果协议。"""

    status: ToolResultStatus = "success"
    content: str = ""
    display_text: Optional[str] = None
    structured_data: Any = None
    ephemeral_context: Any = None
    error_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_display_string(self) -> str:
        """获取默认展示文本。"""
        if self.display_text is not None:
            return self.display_text
        if self.content:
            return self.content
        if self.structured_data is not None:
            return json.dumps(self.structured_data, ensure_ascii=False, indent=2)
        if self.status == "needs_confirmation":
            return "工具执行需要用户确认。"
        return ""

    @classmethod
    def success(
        cls,
        content: str = "",
        *,
        display_text: Optional[str] = None,
        structured_data: Any = None,
        ephemeral_context: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            status="success",
            content=content,
            display_text=display_text,
            structured_data=structured_data,
            ephemeral_context=ephemeral_context,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def error(
        cls,
        content: str,
        *,
        error_type: str = "tool_error",
        display_text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            status="error",
            content=content,
            display_text=display_text,
            error_type=error_type,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def needs_confirmation(
        cls,
        content: str,
        *,
        display_text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            status="needs_confirmation",
            content=content,
            display_text=display_text,
            metadata=dict(metadata or {}),
        )


class Tool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Type[BaseModel],
        *,
        guidance: str = "",
        read_only: bool = False,
        destructive: bool = False,
        requires_confirmation: bool = False,
        supports_parallel: bool = True,
        output_mode: ToolOutputMode = "text",
        source: str = "custom",
        ephemeral: bool = False,
        prompt: str = "",
        prompt_visibility: ToolPromptVisibility = "none",
        demand_skill_tool: bool = False,
        demand_skill_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        risk_categories: Optional[list[str]] = None,
        side_effect_level: Optional[ToolSideEffectLevel] = None,
        resource_scope: Optional[list[str]] = None,
        visibility_scope: ToolVisibilityScope = "resident",
        expose_in_deferred: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ):
        resolved_side_effect_level: ToolSideEffectLevel
        if side_effect_level is not None:
            resolved_side_effect_level = side_effect_level
        elif destructive:
            resolved_side_effect_level = "high"
        elif read_only:
            resolved_side_effect_level = "none"
        else:
            resolved_side_effect_level = "medium"
        self.spec = ToolSpec(
            name=name,
            description=description,
            parameters_model=parameters,
            guidance=guidance,
            read_only=read_only,
            destructive=destructive,
            requires_confirmation=requires_confirmation,
            supports_parallel=supports_parallel,
            output_mode=output_mode,
            source=source,
            ephemeral=ephemeral,
            prompt=prompt,
            prompt_visibility=prompt_visibility,
            demand_skill_tool=demand_skill_tool,
            demand_skill_name=demand_skill_name,
            tags=list(tags or []),
            risk_categories=list(risk_categories or []),
            side_effect_level=resolved_side_effect_level,
            resource_scope=list(resource_scope or []),
            visibility_scope=visibility_scope,
            expose_in_deferred=expose_in_deferred,
            metadata=dict(metadata or {}),
        )
        self.name = self.spec.name
        self.description = self.spec.description
        self.parameters = self.spec.parameters_model

    @abstractmethod
    def run(self, parameters: dict) -> Any:
        pass

    async def arun(self, parameters: dict) -> Any:
        """异步执行工具。

        默认实现直接调用同步 ``run()``。子类（如 ``MCPWrappedTool``）
        可重写此方法以提供原生异步执行，避免线程桥接开销。
        """
        return self.run(parameters)

    def get_guidance(self) -> str:
        return self.spec.guidance

    def build_prompt(self) -> str:
        """兼容接口：返回工具附加说明文本。"""
        return self.spec.prompt

    def get_prompt_visibility(self) -> ToolPromptVisibility:
        """兼容接口：当前框架主路径不再使用该字段控制注入方式。"""
        return self.spec.prompt_visibility

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.spec.name,
            description=self.spec.description,
            parameters_model=self.spec.parameters_model,
            guidance=self.spec.guidance,
            read_only=self.spec.read_only,
            destructive=self.spec.destructive,
            requires_confirmation=self.spec.requires_confirmation,
            supports_parallel=self.spec.supports_parallel,
            output_mode=self.spec.output_mode,
            source=self.spec.source,
            ephemeral=self.spec.ephemeral,
            prompt=self.spec.prompt,
            prompt_visibility=self.spec.prompt_visibility,
            demand_skill_tool=self.spec.demand_skill_tool,
            demand_skill_name=self.spec.demand_skill_name,
            tags=list(self.spec.tags),
            risk_categories=list(self.spec.risk_categories),
            side_effect_level=self.spec.side_effect_level,
            resource_scope=list(self.spec.resource_scope),
            visibility_scope=self.spec.visibility_scope,
            expose_in_deferred=self.spec.expose_in_deferred,
            metadata=dict(self.spec.metadata),
        )

    def mark_as_demand_skill_tool(self, skill_name: str) -> None:
        self.spec.demand_skill_tool = True
        self.spec.demand_skill_name = skill_name
        self.spec.ephemeral = True

    def clear_demand_skill_tool(self) -> None:
        self.spec.demand_skill_tool = False
        self.spec.demand_skill_name = None

    def validate_parameters(self, parameters: dict) -> dict[str, Any]:
        validated = self.parameters.model_validate(parameters)
        if hasattr(validated, "model_dump"):
            return validated.model_dump(mode="python")
        return dict(parameters)

    def get_openai_schema(self) -> dict[str, Any]:
        return self.to_provider_schema("openai")

    def to_provider_schema(self, provider: str = "openai") -> Any:
        from core.providers.tool_schema import create_tool_schema_adapter

        return create_tool_schema_adapter(provider).export_tool(self)

    def __call__(self, parameters: dict):
        validated = self.validate_parameters(parameters)
        return self.run(validated)
