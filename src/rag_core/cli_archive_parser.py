from __future__ import annotations

import argparse

from rag_core.archive_sources import ArchiveLimits
from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_source_flags import (
    add_force_reindex_flag,
    add_json_flag,
    add_manifest_dir_flag,
    add_max_concurrency_flag,
    add_metadata_flag,
    add_plan_json_flag,
    add_scope_flags,
)


def add_ingest_archive_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    limits = ArchiveLimits()
    ingest_archive = subparsers.add_parser(
        "ingest-archive",
        help="Ingest supported files inside a ZIP archive.",
    )
    ingest_archive.description = (
        "Ingest supported files inside one ZIP archive. Archive members are read "
        "directly with path and byte-count limits; this command does not extract "
        "files to disk or recurse into nested archives."
    )
    add_config_flags(ingest_archive)
    ingest_archive.add_argument("archive_path", help="ZIP archive to ingest.")
    add_scope_flags(ingest_archive)
    add_force_reindex_flag(
        ingest_archive,
        help="Reindex archive members even when content is unchanged.",
    )
    add_max_concurrency_flag(
        ingest_archive,
        help="Maximum archive members to ingest concurrently.",
    )
    ingest_archive.add_argument(
        "--archive-max-entries",
        type=int,
        default=limits.max_entries,
        help=f"Maximum non-directory ZIP members to inspect. Default: {limits.max_entries}.",
    )
    ingest_archive.add_argument(
        "--archive-max-entry-bytes",
        type=int,
        default=limits.max_entry_bytes,
        help=(
            "Maximum uncompressed bytes per supported ZIP member. "
            f"Default: {limits.max_entry_bytes}."
        ),
    )
    ingest_archive.add_argument(
        "--archive-max-total-bytes",
        type=int,
        default=limits.max_total_bytes,
        help=(
            "Maximum total uncompressed bytes across supported ZIP members. "
            f"Default: {limits.max_total_bytes}."
        ),
    )
    add_manifest_dir_flag(
        ingest_archive,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    add_plan_json_flag(
        ingest_archive,
        help=(
            "Print the fingerprinted archive source-item plan and manifest "
            "reconciliation, then exit without assembling RAGCore."
        ),
    )
    add_metadata_flag(
        ingest_archive,
        help="Repeatable metadata field applied to every ingested archive member.",
    )
    add_events_jsonl_flag(ingest_archive)
    add_json_flag(ingest_archive, help="Emit one JSON object per archive member.")


__all__ = ["add_ingest_archive_command"]
