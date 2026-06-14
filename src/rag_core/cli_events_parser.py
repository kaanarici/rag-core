from __future__ import annotations

import argparse


def add_events_jsonl_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--events-jsonl",
        help="Append pipeline events as JSON lines to this path.",
    )


__all__ = ["add_events_jsonl_flag"]
