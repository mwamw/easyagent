"""Explicit configuration factories. No import strings, eval, or new module base."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

from .errors import ConfigurationError
from .loader import validate_options
from .models import ConfigurationPaths

if TYPE_CHECKING:
    from agent import BasicAgent
    from core.llm import EasyLLM
    from permissions import BasePermissionEngine
    from memory import BaseMemory
    from context.base import BaseContextManager
    from executor.base import BaseAgentExecutor
    from prompt.base import BaseSystemPromptComposer
    from multiagent import BaseMultiAgent
    from Tool import ToolRegistry

O = TypeVar("O", bound=BaseModel)
C = TypeVar("C")


@dataclass(frozen=True)
class BuildContext:
    paths: ConfigurationPaths
    llm: EasyLLM
    _rollback: list[Callable[[], None]] = field(default_factory=list, repr=False, compare=False)

    def on_failure(self, cleanup: Callable[[], None]) -> None:
        """Register cleanup for resources not yet owned by an installed module."""
        self._rollback.append(cleanup)

    def rollback(self) -> bool:
        complete = True
        while self._rollback:
            try:
                self._rollback.pop()()
            except Exception:
                complete = False
        return complete


@dataclass(frozen=True)
class FactoryEntry(Generic[O]):
    options: type[O]
    install: Callable[[BasicAgent, O, BuildContext], None]
    prepare: Callable[[O, ConfigurationPaths], O] | None = None
    tool_names: tuple[str, ...] = ()

    def validate(self, raw: object, location: str, paths: ConfigurationPaths) -> O:
        value = validate_options(self.options, raw, location)
        prepared = self.prepare(value, paths) if self.prepare else value
        if not isinstance(prepared, self.options):
            raise ConfigurationError(location, "prepare must return the registered configuration type")
        return prepared


def close_resource(resource: object) -> None:
    """Only used for a newly created resource that failed to install."""
    close = getattr(resource, "close", None)
    if callable(close):
        close()


class ConfigurationRegistry:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], FactoryEntry] = {}
        self.models: dict[str, tuple[type[BaseModel], Callable[[BaseModel, ConfigurationPaths], EasyLLM]]] = {}

    def register_model(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, ConfigurationPaths], EasyLLM],
    ) -> None:
        if name in self.models:
            raise ValueError(f"Model factory {name!r} is already registered")
        self.models[name] = (config_model, factory)

    def register_installer(
        self, kind: str, name: str, *, config_model: type[O],
        install: Callable[[BasicAgent, O, BuildContext], None],
        prepare: Callable[[O, ConfigurationPaths], O] | None = None,
    ) -> None:
        """Advanced assembly callback; modules themselves never receive Agent."""
        key = (kind, name)
        if key in self.entries:
            raise ValueError(f"Configuration factory {kind}/{name} is already registered")
        self.entries[key] = FactoryEntry(config_model, install, prepare)

    def get(self, kind: str, name: str) -> FactoryEntry:
        try:
            return self.entries[(kind, name)]
        except KeyError:
            choices = ", ".join(sorted(n for k, n in self.entries if k == kind)) or "(none)"
            raise ConfigurationError(kind, f"Unknown implementation {name!r}; available: {choices}") from None

    def _component(
        self, kind: str, name: str, config_model: type[O],
        factory: Callable[[O, BuildContext], C],
        install: Callable[[BasicAgent, C], object],
        prepare: Callable[[O, ConfigurationPaths], O] | None = None,
    ) -> None:
        def assemble(agent: BasicAgent, options: O, context: BuildContext) -> None:
            component = factory(options, context)
            closed = False

            def close_once():
                nonlocal closed
                if not closed:
                    close_resource(component)
                    closed = True

            # These contracts have no Agent-owned close lifecycle. Roll back
            # their resources if a later installation fails.
            if kind in ("permissions", "executor", "prompt"):
                context.on_failure(close_once)
            try:
                from context.base import BaseContextManager
                from executor.base import BaseAgentExecutor
                from memory.base import BaseMemory
                from multiagent.base import BaseMultiAgent
                from permissions.base import BasePermissionEngine
                from prompt.base import BaseSystemPromptComposer
                contracts = {
                    "context": BaseContextManager, "executor": BaseAgentExecutor,
                    "memory": BaseMemory, "multi_agent": BaseMultiAgent,
                    "permissions": BasePermissionEngine, "prompt": BaseSystemPromptComposer,
                }
                if not isinstance(component, contracts[kind]):
                    raise TypeError(f"{kind} factory returned an invalid component")
                install(agent, component)
            except BaseException:
                try:
                    close_once()
                except Exception:
                    context.on_failure(close_once)
                raise
        self.register_installer(kind, name, config_model=config_model, install=assemble, prepare=prepare)

    def register_permission(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, BuildContext], BasePermissionEngine],
        prepare: Callable[[O, ConfigurationPaths], O] | None = None,
    ) -> None:
        self._component("permissions", name, config_model, factory,
                        lambda a, c: a.with_permissions(c), prepare)

    def register_memory(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, BuildContext], BaseMemory],
        prepare: Callable[[O, ConfigurationPaths], O] | None = None,
    ) -> None:
        self._component("memory", name, config_model, factory,
                        lambda a, c: a.with_memory(c, owned=True), prepare)

    def register_context(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, BuildContext], BaseContextManager],
        prepare: Callable[[O, ConfigurationPaths], O] | None = None,
    ) -> None:
        self._component("context", name, config_model, factory, lambda a, c: a.with_context(c), prepare)

    def register_executor(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, BuildContext], BaseAgentExecutor],
    ) -> None:
        self._component("executor", name, config_model, factory, lambda a, c: a.with_executor(c))

    def register_prompt(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, BuildContext], BaseSystemPromptComposer],
    ) -> None:
        self._component("prompt", name, config_model, factory, lambda a, c: a.with_prompt(c))

    def register_multi_agent(
        self, name: str, *, config_model: type[O],
        factory: Callable[[O, BuildContext], BaseMultiAgent],
    ) -> None:
        self._component("multi_agent", name, config_model, factory, lambda a, c: a.with_multi_agent(c))

    def register_tools(
        self, name: str, *, config_model: type[O],
        register: Callable[[ToolRegistry, O, BuildContext], None],
        prepare: Callable[[O, ConfigurationPaths], O] | None = None,
        tool_names: tuple[str, ...] = (),
    ) -> None:
        def install(agent: BasicAgent, options: O, context: BuildContext) -> None:
            # A temporary registry catches duplicates within a group as well as
            # conflicts with previously installed groups. It is not a new runtime.
            from Tool import ToolRegistry
            group = ToolRegistry(conflict_policy="error")
            register(group, options, context)
            if agent.tool_registry is None:
                agent.with_tool(ToolRegistry(conflict_policy="error"))
            for tool in group.get_visible_tools():
                spec = group.get_tool_spec(tool.name)
                agent.tool_registry.register_tool(
                    tool, visibility=group.get_tool_visibility(tool.name),
                    expose_in_deferred=spec.expose_in_deferred,
                    permission_descriptor=group.get_permission_descriptor(tool.name),
                    conflict_policy="error",
                )
        self.register_installer("tools", name, config_model=config_model, install=install, prepare=prepare)
        entry = self.entries[("tools", name)]
        self.entries[("tools", name)] = FactoryEntry(entry.options, entry.install, entry.prepare, tool_names)
