"""Static recipe/schema checks. Never invoke factories, prepare hooks, or models."""
from __future__ import annotations

import json
from typing import Mapping

from pydantic import BaseModel, ValidationError

from configuration import ConfigurationRegistry
from configuration.errors import ConfigurationError
from configuration.models import AgentOptions, MCPOptions
from core.Config import Config
from ui.configuration import UIRegistry

from .syntax import LanguageError
from .values import ParameterHole, Recipe, Value, at_location, contains_symbol, plain


COMPONENT_FIELDS = {
    "model", "prompt", "context", "executor", "permission", "memory", "plan", "tasks",
    "session", "observability", "hooks", "callbacks", "codeintel", "worktree", "multi_agent",
}
REQUIRED = {"model", "prompt", "context", "executor"}


def _schema_fields(value: Value, schema: dict, definitions: dict) -> None:
    """Reject typos even in legacy dataclass schemas that silently ignore extras.

    Pydantic still owns actual validation; this only checks declared object keys.
    Open mappings (e.g. provider_options) retain their own arbitrary keys.
    """
    if "$ref" in schema:
        schema = definitions.get(schema["$ref"].rsplit("/", 1)[-1], {})
    alternatives = schema.get("anyOf", schema.get("oneOf", []))
    if alternatives:
        shape = "object" if isinstance(value.data, dict) else "array" if isinstance(value.data, tuple) else None
        errors = []
        for candidate in alternatives:
            resolved = definitions.get(candidate.get("$ref", "").rsplit("/", 1)[-1], candidate)
            if shape and resolved.get("type") not in (None, shape):
                continue
            try:
                _schema_fields(value, resolved, definitions)
            except LanguageError as exc:
                errors.append(exc)
            else:
                return
        if errors:
            raise errors[0]
    if isinstance(value.data, dict):
        properties = schema.get("properties")
        additional = schema.get("additionalProperties")
        for key, child in value.data.items():
            if properties is not None and key not in properties and additional is not True and not isinstance(additional, dict):
                raise LanguageError(child.span, f"Unknown configuration field {key}", code="EA203")
            nested = (properties or {}).get(key, additional if isinstance(additional, dict) else {})
            _schema_fields(child, nested, definitions)
    elif isinstance(value.data, tuple):
        for child in value.data:
            _schema_fields(child, schema.get("items", {}), definitions)


def schema_check(
    model: type[BaseModel], value: Value, *, runtime: bool = False,
    environ: Mapping[str, str] | None = None,
) -> None:
    data = plain(value, environ=environ)
    if isinstance(data, dict):
        allowed = set(model.model_fields)
        allowed.update(f.alias for f in model.model_fields.values() if isinstance(f.alias, str))
        unknown = set(data) - allowed
        if unknown:
            key = sorted(unknown)[0]
            raise LanguageError(at_location(value, (key,)).span, f"Unknown configuration field {key}", code="EA203")
    schema = model.model_json_schema()
    # Schema aliases and Python field names may both be accepted by a registered
    # model. Keep both for typo detection; Pydantic enforces its alias policy.
    named_schema = model.model_json_schema(by_alias=False)
    pairs = [(schema, named_schema)]
    pairs.extend((definition, named_schema.get("$defs", {}).get(name, {}))
                 for name, definition in schema.get("$defs", {}).items())
    for aliased, named in pairs:
        if "properties" in aliased:
            for key, field_schema in named.get("properties", {}).items():
                aliased["properties"].setdefault(key, field_schema)
    _schema_fields(value, schema, schema.get("$defs", {}))
    try:
        # JSON strict mode permits JSON representations of Path/Enum while
        # refusing YAML-style coercions such as "32" -> integer or true -> 1.
        model.model_validate_json(json.dumps(data, ensure_ascii=False), strict=True)
    except ValidationError as exc:
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            selected = at_location(value, error["loc"])
            symbolic = contains_symbol(selected)
            if not runtime and symbolic and error["type"] not in {
                "int_type", "float_type", "bool_type", "string_type", "list_type", "dict_type", "model_type",
                "missing", "extra_forbidden", "union_tag_not_found",
            }:
                # Constraints on env values/parameters are checked on binding.
                continue
            location = ".".join(map(str, error["loc"])) or "<configuration>"
            messages = {
                "missing": "required field is missing", "extra_forbidden": "unknown field",
                "int_type": "expected Integer", "float_type": "expected Number",
                "bool_type": "expected Boolean", "string_type": "expected String",
                "list_type": "expected List", "dict_type": "expected Object",
                "literal_error": "value is not one of the supported choices",
                "enum": "value is not one of the supported choices",
                "value_error": "invalid value or conflicting fields",
            }
            message = messages.get(error["type"], "invalid value (" + error["type"] + ")")
            raise LanguageError(selected.span, f"{location}: {message}", code="EA203") from None
    except (TypeError, RuntimeError) as exc:
        raise LanguageError(value.span, f"Configuration validator failed ({type(exc).__name__})", code="EA203") from None


class Checker:
    def __init__(self, registry: ConfigurationRegistry, ui_registry: UIRegistry) -> None:
        self.registry, self.ui_registry = registry, ui_registry

    def check(self, value: Value, *, runtime: bool = False, environ: Mapping[str, str] | None = None) -> None:
        if isinstance(value.data, ParameterHole):
            return
        if not isinstance(value.data, Recipe):
            plain(value, environ=environ)
            return
        recipe = value.data
        if recipe.kind == "agent":
            self.agent(recipe, runtime=runtime, environ=environ)
            return
        if not recipe.enabled:
            return
        fields = recipe.fields
        kind = {"permission": "permissions", "tool": "tools"}.get(recipe.kind, recipe.kind)
        try:
            if kind == "model":
                if recipe.implementation not in self.registry.models:
                    raise ConfigurationError("model", "Unknown model implementation")
                model = self.registry.models[recipe.implementation][0]
            else:
                model = self.registry.get(kind, recipe.implementation).options
        except ConfigurationError:
            raise LanguageError(recipe.span, f"Unknown {recipe.kind} implementation {recipe.implementation}", code="EA203") from None
        if kind == "prompt":
            if "text" in fields and "file" in fields:
                raise LanguageError(recipe.span, "Prompt accepts text or file, not both", code="EA203")
            if "file" in fields and fields["file"].kind != "Path":
                raise LanguageError(fields["file"].span, "Prompt file requires path(...)", code="EA203")
            if "text" in fields and fields["text"].kind != "String":
                raise LanguageError(fields["text"].span, "Prompt text requires String", code="EA203")
            fields = {k: v for k, v in fields.items() if k not in ("text", "file")}
        schema_check(model, Value("Object", fields, recipe.span), runtime=runtime, environ=environ)

    def agent(self, recipe: Recipe, *, runtime: bool, environ: Mapping[str, str] | None) -> None:
        fields = recipe.fields
        allowed = COMPONENT_FIELDS | {"name", "description", "workspace", "allowed_roots", "config", "tools", "skills", "mcp"}
        for name, value in fields.items():
            if name not in allowed:
                raise LanguageError(value.span, f"Unknown Agent field {name}", code="EA203")
            if name in COMPONENT_FIELDS:
                if value.kind == "None" and name not in REQUIRED:
                    continue
                if value.kind != name:
                    raise LanguageError(value.span, f"{name} requires {name}, received {value.kind}", code="EA202")
                self.check(value, runtime=runtime, environ=environ)
        if "model" not in fields:
            raise LanguageError(recipe.span, "Agent requires a model", code="EA203")
        if "workspace" in fields and fields["workspace"].kind != "Path":
            raise LanguageError(fields["workspace"].span, "workspace requires path(...) or a Path parameter", code="EA202")
        agent_fields = {k: v for k, v in fields.items() if k in {"name", "description", "workspace", "allowed_roots", "config"}}
        schema_check(AgentOptions, Value("Object", agent_fields, recipe.span), runtime=runtime, environ=environ)
        if "config" in fields:
            config = fields["config"]
            if isinstance(config.data, dict) and set(config.data) & {"workspace_root", "allowed_roots"}:
                raise LanguageError(config.span, "Use Agent workspace / allowed_roots fields", code="EA203")
            schema_check(Config, config, runtime=runtime, environ=environ)
        if "tools" in fields and fields["tools"].kind != "None":
            value = fields["tools"]
            if value.kind != "tools" or not isinstance(value.data, tuple):
                raise LanguageError(value.span, "Use tools { name(...) }", code="EA202")
            seen, names = set(), set()
            for tool in value.data:
                self.check(tool, runtime=runtime, environ=environ)
                item = tool.data
                assert isinstance(item, Recipe)
                if not item.enabled:
                    continue
                declared = set(self.registry.get("tools", item.implementation).tool_names)
                if item.implementation in seen or names & declared:
                    raise LanguageError(item.span, "Duplicate/conflicting tool selection", code="EA203")
                seen.add(item.implementation)
                names.update(declared)
        if "skills" in fields and fields["skills"].kind != "None":
            value = fields["skills"]
            if value.kind != "skills" or not value.data:
                raise LanguageError(value.span, "Use skills { directory(path(...)) } with at least one directory", code="EA202")
        if "mcp" in fields and fields["mcp"].kind != "None":
            schema_check(MCPOptions, fields["mcp"], runtime=runtime, environ=environ)

    def ui(self, name: str, options: Value) -> None:
        try:
            model = self.ui_registry.config_model(name)
        except ConfigurationError:
            raise LanguageError(options.span, f"Unknown UI implementation {name}", code="EA203") from None
        schema_check(model, options)
