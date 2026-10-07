"""Source-only validation and credential placeholders for generated applications."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ValidationError

from configuration import AgentConfiguration, ConfigurationError, default_registry
from configuration.models import ComponentSpec
from core.Config import Config
from ui.configuration import default_ui_registry


def is_env(value: object) -> bool:
    return isinstance(value, dict) and set(value) == {"env"}


def environment_names(value: Any, *, include_disabled: bool = False) -> set[str]:
    if is_env(value):
        name = value["env"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ConfigurationError("env", "Use portable environment variable names")
        return {name}
    if isinstance(value, dict):
        if value.get("enabled") is False and not include_disabled:
            return set()
        return set().union(*(environment_names(v, include_disabled=include_disabled) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(environment_names(v, include_disabled=include_disabled) for v in value))
    return set()


def _sensitive_name(name: str) -> bool:
    key = name.lower().replace("-", "_")
    return key in {"api_key", "apikey", "token", "secret", "password", "passwd", "authorization",
                   "proxy_authorization", "cookie", "set_cookie", "credential", "credentials"} or key.endswith(
                       ("_api_key", "_token", "_secret", "_password"))


def protect_credentials(raw: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Never read the environment or write literal values into .env.example.

    This is field-based protection, not a secret scanner for prompts/asset contents.
    """
    reserved = environment_names(raw, include_disabled=True)
    converted = []

    def visit(value: Any, location: tuple[str, ...] = ()) -> Any:
        if is_env(value):
            return dict(value)
        secret_url = False
        if isinstance(value, str) and "://" in value:
            try:
                url = urlsplit(value)
                secret_url = bool(url.username or url.password or any(
                    _sensitive_name(key) or key.lower() == "key" for key, _ in parse_qsl(url.query)))
            except ValueError:
                pass
        if isinstance(value, str) and value and location and (_sensitive_name(location[-1]) or secret_url):
            base = "EA_APP_" + re.sub(r"[^A-Z0-9_]", "_", "_".join(location).upper())[:100]
            name, suffix = base, 1
            while name in reserved:
                suffix += 1
                name = f"{base}_{suffix}"
            reserved.add(name)
            converted.append(".".join(location))
            return {"env": name}
        if isinstance(value, dict):
            return {key: visit(item, (*location, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, (*location, str(index))) for index, item in enumerate(value)]
        return value

    return visit(deepcopy(raw)), tuple(converted)


def unbound_options(model: type[BaseModel], raw: object, location: str) -> BaseModel:
    """Validate concrete fields while deferring values supplied by env at startup."""
    references: set[tuple[str | int, ...]] = set()

    def placeholder(value: Any, path: tuple[str | int, ...] = ()) -> Any:
        if is_env(value):
            references.add(path)
            return "__ENV_AT_STARTUP__"
        if isinstance(value, dict):
            return {key: placeholder(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [placeholder(item, (*path, i)) for i, item in enumerate(value)]
        return value

    data = placeholder(raw)
    if isinstance(data, dict):
        allowed = set(model.model_fields)
        allowed.update(f.alias for f in model.model_fields.values() if isinstance(f.alias, str))
        unknown = set(data) - allowed
        if unknown:
            raise ConfigurationError(location, "Unknown fields: " + ", ".join(sorted(unknown)))
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            where = tuple(error["loc"])
            dynamic = any(where[:len(ref)] == ref for ref in references)
            if error["type"] not in {"missing", "extra_forbidden", "list_type", "dict_type", "model_type"} and (
                dynamic or (not where and references and error["type"] == "value_error")
            ):
                continue
            detail = ".".join(map(str, where)) or "<root>"
            raise ConfigurationError(location + "." + detail, "Invalid configuration (" + error["type"] + ")") from None
        # Structural AgentConfiguration fields must remain concrete so builders
        # can determine which capabilities to package. Config values stay raw.
        if model is AgentConfiguration:
            raise ConfigurationError(location, "Agent structure must be concrete; env belongs in component configuration") from None
        return model.model_construct()


def validate_source(raw: dict[str, Any]) -> AgentConfiguration:
    registry = default_registry()
    document = unbound_options(AgentConfiguration, raw, "config")
    assert isinstance(document, AgentConfiguration)
    if is_env(raw.get("agent", {}).get("name")):
        raise ConfigurationError("agent.name", "Application name must be concrete")
    model_name = document.model.implementation
    if model_name not in registry.models:
        raise ConfigurationError("model", "Unknown implementation; custom Python factories require a custom launcher")
    raw_model = raw["model"]
    model_config = raw_model.get("config", {k: v for k, v in raw_model.items() if k not in {"implementation", "enabled"}})
    unbound_options(registry.models[model_name][0], model_config, "model.config")
    for kind in AgentConfiguration.model_fields:
        if kind in {"model", "ui"}:
            continue
        spec = getattr(document, kind)
        if isinstance(spec, ComponentSpec) and spec.enabled:
            entry = registry.get(kind, spec.implementation)
            unbound_options(entry.options, raw[kind].get("config", {}), kind + ".config")
    seen, tool_names = set(), set()
    for index, tool in enumerate(document.tools):
        if not tool.enabled:
            continue
        entry = registry.get("tools", tool.name)
        if tool.name in seen or tool_names.intersection(entry.tool_names):
            raise ConfigurationError(f"tools[{index}]", "Duplicate or conflicting tool selection")
        seen.add(tool.name)
        tool_names.update(entry.tool_names)
        unbound_options(entry.options, raw["tools"][index].get("config", {}), f"tools[{index}].config")
    if {"workspace_root", "allowed_roots"} & document.agent.config.keys():
        raise ConfigurationError("agent.config", "Use agent.workspace / agent.allowed_roots")
    unbound_options(Config, raw.get("agent", {}).get("config", {}), "agent.config")
    if document.ui is not None and document.ui.enabled:
        unbound_options(default_ui_registry().config_model(document.ui.implementation), raw["ui"].get("config", {}), "ui.config")
    return document
