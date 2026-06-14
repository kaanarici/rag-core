from __future__ import annotations

import argparse

from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY


def add_scope_flags(
    parser: argparse.ArgumentParser,
    *,
    corpus_id_help: str | None = None,
) -> None:
    parser.add_argument("--namespace", required=True)
    if corpus_id_help is None:
        parser.add_argument("--corpus-id", required=True)
    else:
        parser.add_argument("--corpus-id", required=True, help=corpus_id_help)


def add_force_reindex_flag(parser: argparse.ArgumentParser, *, help: str) -> None:
    parser.add_argument("--force-reindex", action="store_true", help=help)


def add_max_concurrency_flag(parser: argparse.ArgumentParser, *, help: str) -> None:
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_INGEST_MAX_CONCURRENCY,
        help=f"{help} Default: {DEFAULT_INGEST_MAX_CONCURRENCY}.",
    )


def add_plan_json_flag(parser: argparse.ArgumentParser, *, help: str) -> None:
    parser.add_argument("--plan-json", action="store_true", help=help)


def add_json_flag(parser: argparse.ArgumentParser, *, help: str) -> None:
    parser.add_argument("--json", action="store_true", help=help)


def add_manifest_dir_flag(
    parser: argparse.ArgumentParser,
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    parser.add_argument(
        "--manifest-dir",
        default=manifest_dir_default,
        help=(
            "Directory under which JSONL manifests are written. "
            f"Default: {manifest_dir_help_default}"
        ),
    )


def add_metadata_flag(
    parser: argparse.ArgumentParser,
    *,
    help: str,
) -> None:
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=help,
    )
