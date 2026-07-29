from __future__ import annotations

import argparse

from rag_core.cli.deprecations import (
    DeprecatedAppendAction,
    DeprecatedStoreAction,
    DeprecatedStoreTrueAction,
)
from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.scope import DEFAULT_NAMESPACE


class _AppendCsvValues(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        current = getattr(namespace, self.dest, None)
        items = [] if current is None else list(current)
        if not isinstance(values, str):
            raise TypeError("collections must be a comma-separated string")
        items.extend(value.strip() for value in values.split(",") if value.strip())
        setattr(namespace, self.dest, items)


def add_scope_flags(
    parser: argparse.ArgumentParser,
    *,
    collection_help: str | None = None,
) -> None:
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--collection",
        dest="collection",
        help=collection_help,
    )
    parser.add_argument(
        "--corpus-id",
        dest="collection",
        action=DeprecatedStoreAction,
        replacement="--collection",
        help=argparse.SUPPRESS,
    )


def add_collection_filters(parser: argparse.ArgumentParser, *, help: str) -> None:
    parser.add_argument(
        "--collection",
        dest="collection",
        action="append",
        default=[],
        help=help,
    )
    parser.add_argument(
        "--collections",
        dest="collection",
        action=_AppendCsvValues,
        metavar="a,b",
        help="Comma-separated collections to search.",
    )
    parser.add_argument(
        "--corpus-id",
        dest="collection",
        action=DeprecatedAppendAction,
        replacement="--collection",
        help=argparse.SUPPRESS,
    )


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
    parser.add_argument("--dry-run", dest="plan_json", action="store_true", help=help)
    parser.add_argument(
        "--plan-json",
        dest="plan_json",
        action=DeprecatedStoreTrueAction,
        replacement="--dry-run --json",
        help=argparse.SUPPRESS,
    )


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
