"""One-shot importer for `config/spells.yaml`.

The runtime never modifies the spell database; new or updated entries are
produced by re-running this script. The intent is a script that pulls spell
data from an authoritative external source (e.g. a dungeon-journal scrape or
a community-maintained JSON) and writes it to `config/spells.yaml` in the
schema documented in `wow_alert.events.Spell`.

This script is currently a stub. To add or change spells, edit
`config/spells.yaml` by hand using the documented schema.
"""
from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate config/spells.yaml from an external source.",
    )
    p.add_argument("--dungeon", required=False)
    p.add_argument("--source", default=None, help="Source identifier (TBD).")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    print(
        "import_spells is a stub. Hand-edit config/spells.yaml using the schema\n"
        "documented in that file and in wow_alert/events.py (the Spell model).",
        file=sys.stderr,
    )
    if args.dungeon:
        print(f"  requested dungeon: {args.dungeon}", file=sys.stderr)
    if args.source:
        print(f"  requested source:  {args.source}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
