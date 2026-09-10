"""Copy legacy data to a new database; default execution is a dry run."""

import argparse
import json
from core.session.migration import migrate_session_v3
from memory.migration import migrate_memory_sqlite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["session", "memory-sqlite"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--apply", action="store_true", help="write a new target; never modifies the source")
    args = parser.parse_args()
    migrate = migrate_session_v3 if args.kind == "session" else migrate_memory_sqlite
    print(json.dumps(migrate(args.source, args.target, dry_run=not args.apply), indent=2))


if __name__ == "__main__":
    main()
