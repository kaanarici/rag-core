from __future__ import annotations

import argparse

from rag_core.archive_sources import ArchiveLimits
from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_fetch_parser import add_fetch_flags
from rag_core.cli_help_examples import apply_command_examples
from rag_core.cli_source_flags import (
    add_force_reindex_flag,
    add_json_flag,
    add_manifest_dir_flag,
    add_metadata_flag,
    add_plan_json_flag,
    add_scope_flags,
)
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY


def add_ingest_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    limits = ArchiveLimits()
    ingest = subparsers.add_parser(
        "ingest",
        help="Ingest a file, folder, .zip archive, HTTP(S) URL, or explicit URL list.",
    )
    ingest.description = (
        "Ingest one unambiguous source into the configured vector store. A local "
        "file, directory, or glob uses local file ingest. A path ending in .zip "
        "uses ZIP archive ingest. An http(s):// source uses single-URL ingest. "
        "Use --url-list for a text file of URLs; .txt is never auto-detected as "
        "a URL list."
    )
    add_config_flags(ingest)
    ingest.add_argument(
        "path",
        nargs="?",
        metavar="source",
        help=(
            "Supported file, directory, shell glob, .zip archive, or HTTP(S) URL."
        ),
    )
    ingest.add_argument(
        "--url-list",
        dest="url_file",
        metavar="file.txt",
        help="Text file with one HTTP(S) URL per line. Required for bulk URL ingest.",
    )
    add_scope_flags(
        ingest,
        corpus_id_help=(
            "Corpus partition for ingested documents. ingest writes under one "
            "corpus per invocation; rerun the command for additional corpora."
        ),
    )
    ingest.add_argument("--document-id", help="Document id for a single URL source.")
    add_force_reindex_flag(
        ingest,
        help="Reindex matching sources even when content is unchanged.",
    )
    ingest.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=(
            "Maximum files, archive members, or URL-list entries to ingest "
            f"concurrently. Default: {DEFAULT_INGEST_MAX_CONCURRENCY}."
        ),
    )
    ingest.add_argument(
        "--archive-max-entries",
        type=int,
        default=None,
        help=f"Maximum non-directory ZIP members to inspect. Default: {limits.max_entries}.",
    )
    ingest.add_argument(
        "--archive-max-entry-bytes",
        type=int,
        default=None,
        help=(
            "Maximum uncompressed bytes per supported ZIP member. "
            f"Default: {limits.max_entry_bytes}."
        ),
    )
    ingest.add_argument(
        "--archive-max-total-bytes",
        type=int,
        default=None,
        help=(
            "Maximum total uncompressed bytes across supported ZIP members. "
            f"Default: {limits.max_total_bytes}."
        ),
    )
    add_manifest_dir_flag(
        ingest,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    add_plan_json_flag(
        ingest,
        help=(
            "Print the source-item plan and manifest reconciliation, then exit "
            "without assembling RAGCore. Available for local, archive, and "
            "URL-list sources."
        ),
    )
    add_metadata_flag(
        ingest,
        help="Repeatable metadata field applied to every ingested document.",
    )
    add_events_jsonl_flag(ingest)
    add_fetch_flags(ingest)
    add_json_flag(ingest, help="Emit JSON output for the selected source type.")
    apply_command_examples(ingest, "ingest")


__all__ = ["add_ingest_command"]
