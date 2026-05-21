from __future__ import annotations

import argparse

from rag_core.archive_sources import ArchiveLimits
from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag


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
    ingest_archive.add_argument("--namespace", required=True)
    ingest_archive.add_argument("--corpus-id", required=True)
    ingest_archive.add_argument(
        "--force-reindex",
        action="store_true",
        help="Reindex archive members even when content is unchanged.",
    )
    ingest_archive.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum archive members to ingest concurrently. Default: 1.",
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
    ingest_archive.add_argument(
        "--manifest-dir",
        default=manifest_dir_default,
        help=(
            "Directory under which JSONL manifests are written. "
            f"Default: {manifest_dir_help_default}"
        ),
    )
    ingest_archive.add_argument(
        "--plan-json",
        action="store_true",
        help=(
            "Print the fingerprinted archive source-item plan and manifest "
            "reconciliation, then exit without constructing the runtime."
        ),
    )
    ingest_archive.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable metadata field applied to every ingested archive member.",
    )
    add_events_jsonl_flag(ingest_archive)
    ingest_archive.add_argument(
        "--json", action="store_true", help="Emit one JSON object per archive member."
    )


__all__ = ["add_ingest_archive_command"]
