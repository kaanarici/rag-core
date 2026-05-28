from __future__ import annotations

import argparse

from rag_core.cli_help_examples import apply_command_examples
from rag_core.cli_source_flags import (
    add_json_flag,
    add_manifest_dir_flag,
    add_metadata_flag,
    add_scope_flags,
)


def add_manifest_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    manifest = subparsers.add_parser(
        "manifest",
        help="Preview the manifest entry for one file without indexing it.",
    )
    manifest.add_argument("path", help="Path to the local file.")
    add_scope_flags(
        manifest,
        corpus_id_help=(
            "Corpus partition for the previewed entry. "
            "Manifest previews one file per call."
        ),
    )
    manifest.add_argument("--document-id")
    manifest.add_argument("--document-key")
    add_metadata_flag(
        manifest,
        help="Repeatable metadata field.",
    )
    add_json_flag(manifest, help="Emit JSON output.")
    apply_command_examples(manifest, "manifest")

    manifest_compact = subparsers.add_parser(
        "manifest-compact",
        help="Compact one JSONL corpus manifest to the latest entry per document.",
    )
    add_manifest_dir_flag(
        manifest_compact,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    add_scope_flags(manifest_compact)
    add_json_flag(manifest_compact, help="Emit JSON output.")


__all__ = ["add_manifest_commands"]
