from __future__ import annotations

import argparse

from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_help_examples import apply_command_examples
from rag_core.local_search_models import (
    DEFAULT_LOCAL_MAX_FILES,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_LOCAL_SEARCH_NAMESPACE,
)


def add_local_search_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    local_search = subparsers.add_parser(
        "local-search",
        help="Index a local file/folder and search it without API keys or external Qdrant.",
    )
    local_search.add_argument("path", help="Local file or folder to index.")
    local_search.add_argument("query", help="Search query to run after indexing.")
    local_search.add_argument("--namespace", default=DEFAULT_LOCAL_SEARCH_NAMESPACE)
    local_search.add_argument(
        "--corpus-id",
        help="Corpus partition for indexed documents. local-search uses one corpus per invocation.",
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
