from __future__ import annotations

import argparse

from rag_core.cli.deprecations import DeprecatedStoreAction


def add_events_jsonl_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trace-jsonl",
        dest="events_jsonl",
        help="Append pipeline events as JSON lines to this path.",
    )
    parser.add_argument(
        "--events-jsonl",
        dest="events_jsonl",
        action=DeprecatedStoreAction,
        replacement="--trace-jsonl",
        help=argparse.SUPPRESS,
    )


__all__ = ["add_events_jsonl_flag"]
