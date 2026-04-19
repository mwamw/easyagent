"""Provider-owned tool schema adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Iterable, Optional

from Tool.BaseTool import Tool


_GOOGLE_ALLOWED_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "$id",
    "$anchor",
    "type",
    "format",
    "title",
    "description",
    "enum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "oneOf",
    "properties",
    "additionalProperties",
    "required",
}

_GOOGLE_TYPE_MAP = {
    "object": "OBJECT",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
}


def _normalize_schema_node(
    node: Any,
    *,
    allowed_keys: Optional[set[str]] = None,
    drop_keys: Optional[set[str]] = None,
    upper_case_types: bool = False,
    const_to_enum: bool = False,
) -> Any:
    if isinstance(node, list):
        normalized_items = [
            _normalize_schema_node(
                item,
                allowed_keys=allowed_keys,
                drop_keys=drop_keys,
                upper_case_types=upper_case_types,
                const_to_enum=const_to_enum,
            )
            for item in node
        ]
        return [item for item in normalized_items if item not in (None, {}, [])]

    if not isinstance(node, dict):
        return node

    source = deepcopy(node)
    normalized: dict[str, Any] = {}
    if const_to_enum and "const" in source and "enum" not in source:
        const_value = source.pop("const")
        if const_value is not None:
            source["enum"] = [const_value]

    for key, value in source.items():
        if drop_keys and key in drop_keys:
            continue
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if key == "type":
            if isinstance(value, str):
                if upper_case_types:
                    value = _GOOGLE_TYPE_MAP.get(value.lower(), value)
                normalized[key] = value
                continue
            if isinstance(value, list):
                values = [item for item in value if item != "null"]
                if len(values) == 1:
                    item = values[0]
                    if upper_case_types and isinstance(item, str):
                        item = _GOOGLE_TYPE_MAP.get(item.lower(), item)
                    normalized[key] = item
                elif values:
                    normalized[key] = [
                        _GOOGLE_TYPE_MAP.get(item.lower(), item) if upper_case_types and isinstance(item, str) else item
                        for item in values
                    ]
                continue
        if key == "properties" and isinstance(value, dict):
            properties: dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                normalized_prop = _normalize_schema_node(
                    prop_schema,
                    allowed_keys=allowed_keys,
                    drop_keys=drop_keys,
                    upper_case_types=upper_case_types,
                    const_to_enum=const_to_enum,
                )
                if normalized_prop not in (None, {}, []):
                    properties[prop_name] = normalized_prop
            normalized[key] = properties
            continue
        if key in {"items", "additionalProperties"}:
            normalized_value = _normalize_schema_node(
                value,
                allowed_keys=allowed_keys,
                drop_keys=drop_keys,
                upper_case_types=upper_case_types,
                const_to_enum=const_to_enum,
            )
            if normalized_value not in (None, {}, []):
                normalized[key] = normalized_value
            elif isinstance(value, bool):
                normalized[key] = value
            continue
        if key in {"anyOf", "oneOf", "prefixItems"}:
            normalized_value = _normalize_schema_node(
                value,
                allowed_keys=allowed_keys,
                drop_keys=drop_keys,
                upper_case_types=upper_case_types,
                const_to_enum=const_to_enum,
            )
            if normalized_value:
                normalized[key] = normalized_value
            continue

        normalized_value = _normalize_schema_node(
            value,
            allowed_keys=allowed_keys,
            drop_keys=drop_keys,
            upper_case_types=upper_case_types,
            const_to_enum=const_to_enum,
        )
        if normalized_value in (None, {}, []):
            continue
        normalized[key] = normalized_value

    if "properties" in normalized and "required" in normalized:
        properties = normalized.get("properties", {})
        required = [key for key in normalized["required"] if key in properties]
        if required:
            normalized["required"] = required
        else:
            normalized.pop("required", None)

    return normalized


def _sanitize_google_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_schema_node(
        schema,
        allowed_keys=_GOOGLE_ALLOWED_SCHEMA_KEYS,
        drop_keys={"default"},
        upper_case_types=True,
        const_to_enum=True,
    )
    if "type" not in normalized:
        normalized["type"] = "OBJECT"
    return normalized


def _sanitize_anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _normalize_schema_node(
        schema,
        drop_keys={"default"},
        const_to_enum=True,
    )


class ToolSchemaAdapter(ABC):
    provider_name: str

    @abstractmethod
    def export_tool(self, tool: Tool) -> Any:
        raise NotImplementedError

    def export_tools(self, tools: Iterable[Tool]) -> Any:
        return [self.export_tool(tool) for tool in tools]

    @staticmethod
    def _descriptor(tool: Tool) -> dict[str, Any]:
        return tool.get_spec().to_intermediate_schema()


class OpenAIToolSchemaAdapter(ToolSchemaAdapter):
    provider_name = "openai"

    def export_tool(self, tool: Tool) -> dict[str, Any]:
        descriptor = self._descriptor(tool)
        return {
            "type": "function",
            "function": {
                "name": descriptor["name"],
                "description": descriptor["description"],
                "parameters": descriptor["parameters"],
            },
        }


class OpenAIResponsesToolSchemaAdapter(ToolSchemaAdapter):
    provider_name = "openai_responses"

    def export_tool(self, tool: Tool) -> dict[str, Any]:
        descriptor = self._descriptor(tool)
        return {
            "type": "function",
            "name": descriptor["name"],
            "description": descriptor["description"],
            "parameters": descriptor["parameters"],
        }


class AnthropicToolSchemaAdapter(ToolSchemaAdapter):
    provider_name = "anthropic_native"

    def export_tool(self, tool: Tool) -> dict[str, Any]:
        descriptor = self._descriptor(tool)
        return {
            "name": descriptor["name"],
            "description": descriptor["description"],
            "input_schema": _sanitize_anthropic_schema(descriptor["parameters"]),
        }


class GoogleNativeToolSchemaAdapter(ToolSchemaAdapter):
    provider_name = "google_native"

    def export_tool(self, tool: Tool) -> dict[str, Any]:
        descriptor = self._descriptor(tool)
        return {
            "name": descriptor["name"],
            "description": descriptor["description"],
            "parameters": _sanitize_google_schema(descriptor["parameters"]),
        }

    def export_tools(self, tools: Iterable[Tool]) -> list[dict[str, Any]]:
        declarations = [self.export_tool(tool) for tool in tools]
        if not declarations:
            return []
        return [{"function_declarations": declarations}]


def create_tool_schema_adapter(provider_name: Optional[str]) -> ToolSchemaAdapter:
    normalized = (provider_name or "openai").lower()
    if normalized == "openai_responses":
        return OpenAIResponsesToolSchemaAdapter()
    if normalized in {"google_native", "gemini_native"}:
        return GoogleNativeToolSchemaAdapter()
    if normalized in {"anthropic_native", "claude_native"}:
        return AnthropicToolSchemaAdapter()
    return OpenAIToolSchemaAdapter()


__all__ = [
    "ToolSchemaAdapter",
    "OpenAIToolSchemaAdapter",
    "OpenAIResponsesToolSchemaAdapter",
    "AnthropicToolSchemaAdapter",
    "GoogleNativeToolSchemaAdapter",
    "create_tool_schema_adapter",
]
