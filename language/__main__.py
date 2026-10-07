"""Check, inspect and run .ea files without a second Agent execution loop."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from configuration import ConfigurationError, default_registry
from ui.configuration import default_ui_registry

from .compiler import compile_file
from .program import CompiledProgram
from .syntax import LanguageError


async def run_program(program: CompiledProgram, name: str | None, query: str | None) -> None:
    agent = None
    ui = None
    environ = dict(os.environ)
    try:
        if query is None:
            ui = program.create_ui(environ=environ)
        agent = program.build(name, environ=environ)
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


def schemas() -> dict[str, object]:
    registry = default_registry()
    return {
        "language_version": 1,
        "models": {name: model.model_json_schema() for name, (model, _) in registry.models.items()},
        "components": {f"{kind}/{name}": entry.options.model_json_schema()
                       for (kind, name), entry in registry.entries.items()},
        "ui": default_ui_registry().schemas(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EasyAgent Language — define your Agent Harness")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Static checking; no secrets or clients required")
    check.add_argument("source")
    check.add_argument("--environment", action="store_true", help="Also resolve environment and inspect local resources")
    check.add_argument("--agent")
    explain = commands.add_parser("explain", help="Show value-free assembly preview")
    explain.add_argument("source")
    run = commands.add_parser("run", help="Run with the ordinary Agent API")
    run.add_argument("source")
    run.add_argument("--agent")
    run.add_argument("--query", help="Single request instead of interactive UI")
    commands.add_parser("schema", help="Print registered component JSON Schemas")
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            print(json.dumps(schemas(), ensure_ascii=False, indent=2))
            return 0
        program = compile_file(args.source)
        if args.command == "check":
            if args.environment:
                selected = program.validate_environment(args.agent)
                print("Source and local configuration valid: " + ", ".join(selected))
            else:
                if args.agent:
                    program.select(args.agent)
                print(f"Source valid: {len(program.sources)} file(s), {len(program.agents)} Agent declaration(s). "
                      "Credentials, services and OS capabilities have not been checked.")
        elif args.command == "explain":
            print(json.dumps(program.explain(), ensure_ascii=False, indent=2))
        else:
            program.select(args.agent)
            if args.query is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
                print("Interactive terminal required; pass --query for non-interactive execution.", file=sys.stderr)
                return 2
            asyncio.run(run_program(program, args.agent, args.query))
        return 0
    except (LanguageError, ConfigurationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from core.Exception import ToolInterruption
        message = "User interaction required; use the interactive UI or Python API." if isinstance(exc, ToolInterruption) else f"Run failed ({type(exc).__name__})"
        print(message, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
