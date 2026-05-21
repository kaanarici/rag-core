from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_help_examples import apply_command_examples


def add_ingest_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest supported files into the configured vector store.",
    )
    ingest.description = (
        "Ingest supported local files into the configured vector store."
    )
    add_config_flags(ingest)
    ingest.add_argument(
        "path",
        help=(
            "Supported file, directory, or shell glob (e.g. './docs/*.md'). "
            "Glob patterns are expanded recursively."
        ),
    )
    ingest.add_argument("--namespace", required=True)
    ingest.add_argument(
        "--corpus-id",
        required=True,
        help=(
            "Corpus partition for ingested documents. ingest writes under one corpus "
            "per invocation; rerun the command for additional corpora."
        ),
    )
    ingest.add_argument(
        "--force-reindex",
        action="store_true",
        help="Reindex matching files even when content is unchanged.",
    )
    ingest.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum files to ingest concurrently. Default: 1.",
    )
    ingest.add_argument(
        "--manifest-dir",
        default=manifest_dir_default,
        help=(
            "Directory under which JSONL manifests are written. "
            f"Default: {manifest_dir_help_default}"
        ),
    )
    ingest.add_argument(
        "--plan-json",
        action="store_true",
        help=(
            "Print the fingerprinted source-item plan and manifest reconciliation, "
            "then exit without constructing the runtime."
        ),
    )
    ingest.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable metadata field applied to every ingested document.",
    )
    add_events_jsonl_flag(ingest)
    ingest.add_argument(
        "--json", action="store_true", help="Emit one JSON object per file."
    )
    apply_command_examples(ingest, "ingest")


__all__ = ["add_ingest_command"]
