"""Safe file parsing and explicit environment references, without creating Agents."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .errors import ConfigurationError
from .models import AgentConfiguration

T = TypeVar("T", bound=BaseModel)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ConfigurationError(
                f"line {key_node.start_mark.line + 1}",
                "Mapping keys must be unique strings",
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def validate_options(model: type[T], value: object, location: str) -> T:
    """Never include input values or validator exceptions in user-facing errors."""
    if isinstance(value, dict):
        allowed = set(model.model_fields)
        allowed.update(f.alias for f in model.model_fields.values() if isinstance(f.alias, str))
        if set(value) - allowed:
            raise ConfigurationError(location, "Unknown configuration field(s): " +
                                     ", ".join(sorted(set(value) - allowed)))
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=True, include_url=False)
        def explanation(error):
            kind = error["type"]
            messages = {
                "missing": "Required field is missing", "extra_forbidden": "Unknown field",
                "int_parsing": "Expected an integer", "int_type": "Expected an integer",
                "float_parsing": "Expected a number", "bool_parsing": "Expected true or false",
                "string_type": "Expected text", "list_type": "Expected a list",
                "dict_type": "Expected a mapping", "model_type": "Expected a mapping",
                "value_error": "Invalid value or conflicting fields",
            }
            if kind in ("literal_error", "enum"):
                return "Allowed values: " + str(error.get("ctx", {}).get("expected", "see schema"))
            if kind in ("greater_than", "greater_than_equal", "less_than", "less_than_equal"):
                return "Number is outside the permitted range"
            return messages.get(kind, "Invalid value (" + kind + ")")
        details = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or '<root>'}: {explanation(e)}" for e in errors
        )
        raise ConfigurationError(location, details) from None


def _resolve_env(
    value: Any, location: str, ancestors: frozenset[int] = frozenset(), active: bool = True,
    *, resolve: bool = True, environ: Mapping[str, str] | None = None,
) -> Any:
    if not isinstance(value, (list, dict)):
        return value
    if id(value) in ancestors:
        raise ConfigurationError(location, "Recursive YAML aliases are not supported")
    ancestors = ancestors | {id(value)}
    if isinstance(value, list):
        return [_resolve_env(item, f"{location}[{i}]", ancestors, active, resolve=resolve, environ=environ) for i, item in enumerate(value)]
    # Disabled sections are structurally checked, but do not require credentials.
    active = active and value.get("enabled") is not False
    if set(value) == {"env"}:
        name = value["env"]
        if not isinstance(name, str) or not name:
            raise ConfigurationError(location, "Environment reference requires a nonempty name")
        if not resolve:
            return {"env": name}
        if not active:
            return ""
        environment = os.environ if environ is None else environ
        if name not in environment:
            raise ConfigurationError(location, f"Required environment variable {name} is missing")
        return environment[name]
    return {key: _resolve_env(item, f"{location}.{key}", ancestors, active, resolve=resolve, environ=environ) for key, item in value.items()}


def read_configuration_source(path: str | Path) -> dict[str, Any]:
    """Read safe YAML without binding env references, for source packaging tools."""
    source = Path(path).expanduser().resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=_UniqueLoader)
    except ConfigurationError:
        raise
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"{source}:line {mark.line + 1}" if mark else str(source)
        raise ConfigurationError(where, "Invalid YAML (safe mappings only)") from None
    except (OSError, UnicodeError):
        raise ConfigurationError(str(source), "Cannot read configuration file") from None
    if not isinstance(raw, dict):
        raise ConfigurationError(str(source), "Configuration must be a mapping")
    return _resolve_env(raw, str(source), resolve=False)


def read_configuration(path: str | Path, *, environ: Mapping[str, str] | None = None) -> AgentConfiguration:
    source = Path(path).expanduser().resolve()
    raw = read_configuration_source(source)
    return validate_options(AgentConfiguration, _resolve_env(raw, str(source), environ=environ), str(source))
