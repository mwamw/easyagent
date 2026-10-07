"""buildagent --file agent.yaml --output ./my-agent"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from shlex import join
import sys

from configuration import ConfigurationError
from .builder import plan_application, write_application


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate runnable Agent applications from YAML")
    parser.add_argument("--file", nargs="+", required=True, help="YAML files or quoted glob patterns")
    parser.add_argument("--output", type=Path, help="New app directory, or new parent directory for multiple files")
    parser.add_argument("--check", action="store_true", help="Validate packaging without creating output or reading credentials")
    args = parser.parse_args(argv)
    try:
        sources = []
        for pattern in args.file:
            matches = sorted(glob(str(Path(pattern).expanduser())))
            if not matches:
                raise ConfigurationError("--file", "A file or glob pattern did not match any files")
            for match in matches:
                source = Path(match).resolve()
                if source not in sources:
                    sources.append(source)
        stems = [source.stem for source in sources]
        if len(stems) != len(set(stems)):
            raise ConfigurationError("--file", "Batch inputs must have distinct filename stems")
        if len(sources) > 1 and args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise ConfigurationError("--output", "Batch output directory already exists")
        # Preflight every source before writing any application.
        plans = [plan_application(source) for source in sources]
        targets = [args.output if len(plans) == 1 and args.output is not None
                   else (args.output or Path.cwd()) / (source.stem + "-app") for source in sources]
        for target in targets:
            if target.exists() or target.is_symlink():
                raise ConfigurationError(str(target), "Output already exists; choose a new directory")
        if args.check:
            print(f"Packaging valid: {len(plans)} application(s); no output or model clients created.")
            return 0
        for plan, target in zip(plans, targets):
            written = write_application(plan, target)
            print(f"Generated: {written}")
            print("Run: " + join(["python", str(written / "app.py")]))
            if plan.environment:
                print("Required environment variables: " + ", ".join(plan.environment))
        print("Workspaces remain external; review README.md and build.json before moving or sharing the application.")
        return 0
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Build failed ({type(exc).__name__}); inspect output before retrying.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
