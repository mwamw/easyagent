"""Thin terminal launcher. Owns and closes the Agent; UIs only borrow it."""
import argparse
import asyncio
import sys

from configuration import ConfigurationError, load_agent
from .configuration import load_ui


async def run_file(path: str) -> None:
    ui = load_ui(path)
    agent = None
    try:
        agent = load_agent(path)
        await ui.run(agent)
    finally:
        try:
            await ui.aclose()
        finally:
            if agent is not None:
                await agent.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive EasyAgent terminal UI")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Interactive terminal required; use easyagent-run --query for non-interactive use.", file=sys.stderr)
        return 2
    try:
        asyncio.run(run_file(args.config))
        return 0
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"UI failed ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
