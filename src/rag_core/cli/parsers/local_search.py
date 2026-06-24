from __future__ import annotations

import argparse

from rag_core.cli.deprecations import DeprecatedStoreAction
from rag_core.cli.parsers.events import add_events_jsonl_flag
from rag_core.cli.help_examples import apply_command_examples
from rag_core.local_search.models import (
    DEFAULT_LOCAL_MAX_FILES,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_LOCAL_SEARCH_NAMESPACE,
)


def add_local_search_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command_name: str = "local-search",
    deprecated_alias_for: str | None = None,
) -> argparse.ArgumentParser:
    local_search = subparsers.add_parser(
        command_name,
        help=(
            argparse.SUPPRESS
            if deprecated_alias_for is not None
            else "Index a local file/folder and search it without API keys or external Qdrant."
        ),
    )
    if deprecated_alias_for is not None:
        local_search.set_defaults(
            deprecated_command=command_name,
            canonical_command=deprecated_alias_for,
        )
    local_search.add_argument("path", help="Local file or folder to index.")
    local_search.add_argument("query", help="Search query to run after indexing.")
    local_search.add_argument("--namespace", default=DEFAULT_LOCAL_SEARCH_NAMESPACE)
    local_search.add_argument(
        "--collection",
        dest="collection",
        help="Collection for indexed documents. Local search uses one collection per invocation.",
    )
    local_search.add_argument(
        "--corpus-id",
        dest="collection",
        action=DeprecatedStoreAction,
        replacement="--collection",
        help=argparse.SUPPRESS,
    )
    local_search.add_argument("--limit", type=int, default=DEFAULT_LOCAL_SEARCH_LIMIT)
    local_search.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_LOCAL_MAX_FILES,
        help="Maximum supported files to index from a folder.",
    )
    local_search.add_argument(
        "--demo",
        action="store_true",
        help="Use deterministic demo embeddings for no-download smoke tests.",
    )
    add_events_jsonl_flag(local_search)
    local_search.add_argument("--json", action="store_true", help="Emit JSON output.")
    apply_command_examples(local_search, "local-search")
    return local_search


__all__ = ["add_local_search_command"]
