from __future__ import annotations

import argparse


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
    manifest.add_argument("--namespace", required=True)
    manifest.add_argument(
        "--corpus-id",
        required=True,
        help="Corpus partition for the previewed entry. Manifest previews one file per call.",
    )
    manifest.add_argument("--document-id")
    manifest.add_argument("--document-key")
    manifest.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable metadata field.",
    )
    manifest.add_argument("--json", action="store_true", help="Emit JSON output.")

    manifest_compact = subparsers.add_parser(
        "manifest-compact",
        help="Compact one JSONL corpus manifest to the latest entry per document.",
    )
    manifest_compact.add_argument(
        "--manifest-dir",
        default=manifest_dir_default,
        help=f"Directory under which JSONL manifests are written. Default: {manifest_dir_help_default}",
    )
    manifest_compact.add_argument("--namespace", required=True)
    manifest_compact.add_argument("--corpus-id", required=True)
    manifest_compact.add_argument(
        "--json", action="store_true", help="Emit JSON output."
    )


__all__ = ["add_manifest_commands"]
