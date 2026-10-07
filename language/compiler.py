"""Language front end. Output is recipes for the ordinary Agent builder."""
from __future__ import annotations

from pathlib import Path

from configuration import ConfigurationRegistry, default_registry
from ui.configuration import UIRegistry, default_ui_registry

from .checking import Checker
from .evaluation import Evaluator
from .program import CompiledProgram, Launch
from .syntax import Declaration, Definition, LanguageError, Span
from .values import Value, resolve_path


def compile_file(
    path: str | Path, *, registry: ConfigurationRegistry | None = None,
    ui_registry: UIRegistry | None = None,
) -> CompiledProgram:
    """Check all declarations, including unused templates; do not instantiate anything."""
    source = Path(resolve_path(str(path), Path.cwd(), Span(Path(path), 1, 1)))
    registry = registry if registry is not None else default_registry()
    ui_registry = ui_registry if ui_registry is not None else default_ui_registry()
    evaluator = Evaluator()
    checker = Checker(registry, ui_registry)
    try:
        root = evaluator.load(source)
        for file, module in evaluator.modules.items():
            for declaration in module.declarations:
                if isinstance(declaration, Definition):
                    value = evaluator.expand(file, declaration, {}, declaration.span, symbolic=True)
                else:
                    value = evaluator.resolve(file, declaration.name, declaration.span)
                checker.check(value)
            if module.run:
                run = module.run
                value = evaluator.resolve(file, run.agent, run.span)
                evaluator.require(value, "agent", run.span)
                options = Value("Object", evaluator.fields(run.options, file, {}), run.options.span)
                if run.ui:
                    checker.ui(run.ui, options)
                elif options.data:
                    raise LanguageError(run.span, "Run options require 'with <ui>'", code="EA203")
        agents = {}
        for name, (_, declaration) in evaluator.symbols[source].items():
            if isinstance(declaration, Declaration) and declaration.kind == "agent":
                value = evaluator.resolve(source, name, declaration.span)
                checker.check(value)
                agents[name] = value
        launch = None
        if root.run:
            run = root.run
            launch = Launch(run.agent, run.ui, Value("Object", evaluator.fields(run.options, source, {}), run.options.span), run.span)
        return CompiledProgram(source, agents, registry, ui_registry, launch, tuple(evaluator.modules))
    except RecursionError:
        raise LanguageError(Span(source, 1, 1), "Declaration nesting exceeds compiler limits", code="EA202") from None
