"""Shared generated-app launcher. Reuses AgentBuilder and public Agent/UI APIs."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys

from agent import BasicAgent
from configuration import AgentBuilder, AgentConfiguration, ConfigurationError, ConfigurationRegistry, read_configuration
from configuration.loader import validate_options
from ui import BaseAgentUI
from ui.configuration import UIRegistry, default_ui_registry


def application_configuration(directory: str | Path, workspace: str | Path | None = None) -> tuple[Path, AgentConfiguration]:
    root = Path(directory).expanduser().resolve()
    # Explicit file only; do not search cwd/parents or execute shell expressions.
    from dotenv import dotenv_values
    environment = {}
    if (root / ".env").is_file():
        for key, value in dotenv_values(root / ".env", interpolate=False).items():
            if value is not None:
                environment[key] = value
    environment.update(os.environ)
    source = root / "agent.yaml"
    document = read_configuration(source, environ=environment)
    if workspace is not None:
        document.agent.workspace = str(Path(workspace).expanduser().resolve())
    return source, document


def create_application_agent(
    directory: str | Path, *, workspace: str | Path | None = None,
    registry: ConfigurationRegistry | None = None,
) -> BasicAgent:
    source, document = application_configuration(directory, workspace)
    return AgentBuilder(registry).build_document(document, source=source)


async def _run(
    source: Path, document: AgentConfiguration, query: str | None,
    registry: ConfigurationRegistry | None, ui_registry: UIRegistry,
) -> None:
    agent: BasicAgent | None = None
    ui: BaseAgentUI | None = None
    try:
        if query is None:
            spec = document.ui
            if spec is not None and not spec.enabled:
                raise ConfigurationError("ui", "UI is disabled; pass --query")
            ui = ui_registry.create(spec.implementation if spec else "terminal", spec.config if spec else {})
        agent = AgentBuilder(registry).build_document(document, source=source)
        if query is not None:
            print(await agent.ainvoke(query))
        else:
            assert ui is not None
            await ui.run(agent)
    finally:
        try:
            if ui is not None:
                await ui.aclose()
        finally:
            if agent is not None:
                await agent.aclose()


def run_application(
    directory: str | Path, argv: list[str] | None = None, *,
    registry: ConfigurationRegistry | None = None, ui_registry: UIRegistry | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run the Agent defined by this application's agent.yaml")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="Validate configuration without model calls")
    actions.add_argument("--query", help="Single request instead of interactive UI")
    parser.add_argument("--workspace", help="Override the default Agent workspace, not explicit tool paths")
    args = parser.parse_args(argv)
    try:
        source, document = application_configuration(directory, args.workspace)
        uis = ui_registry if ui_registry is not None else default_ui_registry()
        if args.check:
            AgentBuilder(registry).validate_document(document, source=source)
            if document.ui is not None and document.ui.enabled:
                validate_options(uis.config_model(document.ui.implementation), document.ui.config, "ui.config")
            print("Application configuration valid; services, credentials and OS capabilities were not tested.")
            return 0
        if args.query is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise ConfigurationError("terminal", "Interactive terminal required; use --query")
        asyncio.run(_run(source, document, args.query, registry, uis))
        return 0
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from core.Exception import ToolInterruption
        print("User interaction required; run without --query to handle approval." if isinstance(exc, ToolInterruption)
              else f"Application failed ({type(exc).__name__})", file=sys.stderr)
        return 3
