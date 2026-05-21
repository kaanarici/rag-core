from __future__ import annotations

import argparse


def add_trace_summary_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    trace_summary = subparsers.add_parser(
        "trace-summary",
        help="Summarize retrieval and embedding behavior from an events JSONL trace file.",
    )
    trace_summary.add_argument("path", help="Path written by --events-jsonl.")
    trace_summary.add_argument("--json", action="store_true", help="Emit JSON output.")


__all__ = ["add_trace_summary_command"]
