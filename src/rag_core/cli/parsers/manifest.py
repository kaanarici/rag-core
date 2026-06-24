from __future__ import annotations

import argparse

from rag_core.cli.help_examples import apply_command_examples
from rag_core.cli.parsers.sources import (
    add_json_flag,
    add_manifest_dir_flag,
    add_metadata_flag,
    add_scope_flags,
)


def add_manifest_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    manifest = subparsers.add_parser(
        "manifest",
        help="Preview one file manifest entry or compact a corpus manifest.",
    )
    manifest.description = (
        "Preview the manifest entry for one local file without indexing it. "
        "Use --compact to rewrite one JSONL corpus manifest to the latest entry "
        "per document."
    )
    manifest.add_argument("path", nargs="?", help="Path to the local file.")
    manifest.add_argument(
        "--compact",
        action="store_true",
        help="Compact the selected JSONL corpus manifest instead of previewing a file.",
    )
    add_scope_flags(
        manifest,
        collection_help=(
            "Collection for the previewed or compacted manifest entries."
        ),
    )
    manifest.add_argument("--document-id")
    manifest.add_argument("--document-key")
    add_metadata_flag(
        manifest,
        help="Repeatable metadata field.",
    )
    add_json_flag(manifest, help="Emit JSON output.")
    add_manifest_dir_flag(
        manifest,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    apply_command_examples(manifest, "manifest")


__all__ = ["add_manifest_command"]
