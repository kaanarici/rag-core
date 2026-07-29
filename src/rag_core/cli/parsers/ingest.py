from __future__ import annotations

import argparse
from typing import Literal

from rag_core.ingest.sources.archive import ArchiveLimits
from rag_core.cli.parsers.config import add_config_flags
from rag_core.cli.parsers.events import add_events_jsonl_flag
from rag_core.cli.parsers.fetch import add_fetch_flags
from rag_core.cli.help_examples import apply_command_examples
from rag_core.cli.parsers.sources import (
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
    command_name: Literal["add", "ingest"] = "add",
    deprecated_alias_for: str | None = None,
) -> None:
    limits = ArchiveLimits()
    ingest = subparsers.add_parser(
        command_name,
        help=(
            argparse.SUPPRESS
            if deprecated_alias_for is not None
            else "Add a file, folder, .zip archive, HTTP(S) URL, or explicit URL list."
        ),
    )
    if deprecated_alias_for is not None:
        ingest.set_defaults(
            deprecated_command=command_name,
            canonical_command=deprecated_alias_for,
        )
    ingest.description = (
        "Add one unambiguous source to the configured vector store. A local "
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
        collection_help=(
            "Collection for added documents. add writes under one collection "
            "per invocation; rerun the command for additional collections."
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
            "without assembling Engine. Available for local, archive, and "
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
    apply_command_examples(ingest, command_name)


__all__ = ["add_ingest_command"]
