from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_help_examples import apply_command_examples
from rag_core.cli_source_flags import (
    add_force_reindex_flag,
    add_json_flag,
    add_manifest_dir_flag,
    add_max_concurrency_flag,
    add_metadata_flag,
    add_plan_json_flag,
    add_scope_flags,
)


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
    add_scope_flags(
        ingest,
        corpus_id_help=(
            "Corpus partition for ingested documents. ingest writes under one corpus "
            "per invocation; rerun the command for additional corpora."
        ),
    )
    add_force_reindex_flag(
        ingest,
        help="Reindex matching files even when content is unchanged.",
    )
    add_max_concurrency_flag(
        ingest,
        help="Maximum files to ingest concurrently.",
    )
    add_manifest_dir_flag(
        ingest,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    add_plan_json_flag(
        ingest,
        help=(
            "Print the fingerprinted source-item plan and manifest reconciliation, "
            "then exit without assembling RAGCore."
        ),
    )
    add_metadata_flag(
        ingest,
        help="Repeatable metadata field applied to every ingested document.",
    )
    add_events_jsonl_flag(ingest)
    add_json_flag(ingest, help="Emit one JSON object per file.")
    apply_command_examples(ingest, "ingest")


__all__ = ["add_ingest_command"]
