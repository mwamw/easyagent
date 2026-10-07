"""Thin single-request runner: python -m configuration agent.yaml --check."""
import argparse
import sys

from . import ConfigurationError, load_agent, validate_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and run an EasyAgent configuration")
    parser.add_argument("config")
    parser.add_argument("--check", action="store_true", help="Validate without constructing clients or running tools")
    parser.add_argument("--query", help="Run one request; an ASK interruption is reported, never auto-approved")
    args = parser.parse_args(argv)
    if not args.check and args.query is None:
        parser.error("Use --check or --query")
    agent = None
    try:
        if args.check:
            print("Configuration valid: " + ", ".join(validate_agent(args.config)))
            return 0
        agent = load_agent(args.config)
        print(agent.invoke(args.query))
        return 0
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        # Runtime exceptions may contain secrets or model input; do not echo them.
        from core.Exception import ToolInterruption
        message = "User interaction required; use a Python/UI caller to handle the interruption." if isinstance(exc, ToolInterruption) else f"Run failed ({type(exc).__name__})"
        print(message, file=sys.stderr)
        return 3
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
