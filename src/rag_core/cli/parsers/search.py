from __future__ import annotations

import argparse

from rag_core.cli.parsers.config import add_config_flags
from rag_core.cli.deprecations import DeprecatedStoreAction, DeprecatedStoreTrueAction
from rag_core.cli.parsers.events import add_events_jsonl_flag
from rag_core.cli.help_examples import apply_command_examples
from rag_core.cli.profile_help import query_plan_preset_help, search_profile_help
from rag_core.cli.parsers.sources import add_collection_filters
from rag_core.retrieval_defaults import DEFAULT_CONTEXT_LIMIT, DEFAULT_SEARCH_LIMIT
from rag_core.local_search.models import (
    DEFAULT_LOCAL_MAX_FILES,
)
from rag_core.search.context_pack import CONTEXT_ORDER_VALUES
from rag_core.search.planning import QUERY_PLAN_PRESETS, SEARCH_PROFILES


def add_search_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    search = subparsers.add_parser(
        "search",
        help="Search a local path or the configured vector store.",
    )
    search.description = (
        "Search a local path when one is supplied, otherwise search the configured "
        "vector store."
    )
    _add_search_or_context_flags(search, context_command=False)
    apply_command_examples(search, "search")


def add_context_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    context = subparsers.add_parser(
        "context",
        help=(
            "Emit context-pack JSON with prompt-safe context_text from a local "
            "path or configured vector store."
        ),
    )
    context.description = (
        "Build prompt-safe context from a local path when one is supplied, "
        "otherwise from the configured vector store."
    )
    _add_search_or_context_flags(context, context_command=True)
    apply_command_examples(context, "context")


def _add_search_or_context_flags(
    parser: argparse.ArgumentParser,
    *,
    context_command: bool,
) -> None:
    add_config_flags(parser)
    parser.add_argument("text", help="Search query text.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional local file or folder to index before running the query.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Logical namespace. Defaults to local for path searches and default for configured searches.",
    )
    add_collection_filters(
        parser,
        help="Repeatable. At least one collection must be specified for configured search.",
    )
    parser.add_argument(
        "--content-type",
        action="append",
        default=[],
        help="Repeatable. Narrow results to a content type such as document or code.",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Repeatable. Narrow results to a document id inside the corpus scope.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            f"Maximum results. Default: {DEFAULT_SEARCH_LIMIT}; "
            f"{DEFAULT_CONTEXT_LIMIT} for context output."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply the configured reranker to the result set.",
    )
    query_plan_group = parser.add_mutually_exclusive_group()
    query_plan_group.add_argument(
        "--search-profile",
        choices=SEARCH_PROFILES,
        default=None,
        help=search_profile_help(
            prefix="Common search profile.",
            suffix="Mutually exclusive with --plan.",
        ),
    )
    query_plan_group.add_argument(
        "--plan",
        dest="query_plan_preset",
        choices=QUERY_PLAN_PRESETS,
        default=None,
        help=query_plan_preset_help(
            prefix="Built-in QueryPlan recipe.",
            suffix="Mutually exclusive with --search-profile.",
        ),
    )
    query_plan_group.add_argument(
        "--query-plan-preset",
        dest="query_plan_preset",
        choices=QUERY_PLAN_PRESETS,
        action=DeprecatedStoreAction,
        replacement="--plan",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--metadata-filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable exact metadata filter applied to the query.",
    )
    if context_command:
        parser.set_defaults(context=True)
    else:
        parser.set_defaults(context=False)
        parser.add_argument(
            "--context",
            dest="context",
            action=DeprecatedStoreTrueAction,
            replacement="rag-core context <query> [path]",
            help=argparse.SUPPRESS,
        )
    context_help = None if context_command else argparse.SUPPRESS
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help=context_help
        or "Approximate character budget for returned snippets.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help=context_help
        or "Approximate token budget for returned snippets.",
    )
    parser.add_argument(
        "--context-order",
        choices=CONTEXT_ORDER_VALUES,
        default="rank",
        help=context_help
        or "Prompt context render order. Default: rank.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_LOCAL_MAX_FILES,
        help="Maximum supported files to index when path is supplied.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use deterministic demo embeddings for local path smoke tests.",
    )
    add_events_jsonl_flag(parser)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")


__all__ = ["add_context_command", "add_search_command"]
