"""UI selection is separate from Agent construction and execution."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, Field

from configuration import ConfigurationError, Options, read_configuration
from configuration.loader import validate_options
from .base import BaseAgentUI

O = TypeVar("O", bound=BaseModel)


class TerminalOptions(Options):
    show_tool_results: bool = False
    streaming: bool = True
    max_iter: int = Field(default=20, gt=0)


class UIRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[type[BaseModel], Callable[[BaseModel], BaseAgentUI]]] = {}

    def register(self, name: str, *, config_model: type[O], factory: Callable[[O], BaseAgentUI]) -> None:
        if name in self._entries:
            raise ValueError("UI implementation already registered")
        self._entries[name] = (config_model, factory)

    def create(self, name: str, config: dict) -> BaseAgentUI:
        self.config_model(name)
        model, factory = self._entries[name]
        options = validate_options(model, config, "ui.config")
        instance = factory(options)
        if not isinstance(instance, BaseAgentUI):
            raise ConfigurationError("ui", "UI factory must return BaseAgentUI")
        return instance

    def config_model(self, name: str) -> type[BaseModel]:
        """Read-only schema access; does not import or instantiate a UI."""
        if name not in self._entries:
            raise ConfigurationError("ui.implementation", f"Unknown UI {name!r}")
        return self._entries[name][0]

    def schemas(self) -> dict[str, dict]:
        return {name: model.model_json_schema() for name, (model, _) in self._entries.items()}


def default_ui_registry() -> UIRegistry:
    registry = UIRegistry()

    def terminal(options: TerminalOptions) -> BaseAgentUI:
        try:
            from .terminal import TerminalUI
        except ImportError:
            raise ConfigurationError("ui", 'Install terminal dependencies: pip install -e ".[ui]"') from None
        return TerminalUI(**options.model_dump())

    registry.register("terminal", config_model=TerminalOptions, factory=terminal)
    return registry


def load_ui(path: str | Path, *, registry: UIRegistry | None = None) -> BaseAgentUI:
    document = read_configuration(path)
    spec = document.ui
    if spec is not None and not spec.enabled:
        raise ConfigurationError("ui.enabled", "UI is disabled; use easyagent-run or the Python API")
    name = spec.implementation if spec else "terminal"
    options = spec.config if spec else {}
    return (registry or default_ui_registry()).create(name, options)
