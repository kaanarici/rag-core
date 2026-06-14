from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_help_examples import apply_command_examples
from rag_core.cli_profile_help import query_plan_preset_help, search_profile_help
from rag_core.retrieval_defaults import DEFAULT_CONTEXT_LIMIT, DEFAULT_SEARCH_LIMIT
from rag_core.search.context_pack import CONTEXT_ORDER_VALUES
from rag_core.search.planning import QUERY_PLAN_PRESETS, SEARCH_PROFILES


def add_search_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    search = subparsers.add_parser(
        "search",
        help="Search the configured vector store.",
    )
    search.description = (
        "Search the configured vector store. Add --context to emit prompt-safe "
        "context-pack JSON instead of raw search hits."
    )
    add_config_flags(search)
    search.add_argument("text", help="Search query text.")
    search.add_argument("--namespace", required=True)
    search.add_argument(
        "--corpus-id",
        action="append",
        default=[],
        help="Repeatable. At least one corpus must be specified.",
    )
    search.add_argument(
        "--content-type",
        action="append",
        default=[],
        help="Repeatable. Narrow results to a content type such as document or code.",
    )
    search.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Repeatable. Narrow results to a document id inside the corpus scope.",
    )
    search.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            f"Maximum results. Default: {DEFAULT_SEARCH_LIMIT}; "
            f"{DEFAULT_CONTEXT_LIMIT} with --context."
        ),
    )
    search.add_argument(
        "--rerank",
        action="store_true",
        help="Apply the configured reranker to the result set.",
    )
    query_plan_group = search.add_mutually_exclusive_group()
    query_plan_group.add_argument(
        "--search-profile",
        choices=SEARCH_PROFILES,
        default=None,
        help=search_profile_help(
            prefix="Common search profile.",
            suffix="Mutually exclusive with --query-plan-preset.",
        ),
    )
    query_plan_group.add_argument(
        "--query-plan-preset",
        choices=QUERY_PLAN_PRESETS,
        default=None,
        help=query_plan_preset_help(
            prefix="Built-in QueryPlan recipe.",
            suffix="Mutually exclusive with --search-profile.",
        ),
    )
    search.add_argument(
        "--metadata-filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable exact metadata filter applied to the query.",
    )
    search.add_argument(
        "--context",
        action="store_true",
        help="Emit context-pack JSON with prompt-safe context_text instead of raw hits.",
    )
    search.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Approximate character budget for returned snippets. Requires --context.",
    )
    search.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Approximate token budget for returned snippets. Requires --context.",
    )
    search.add_argument(
        "--context-order",
        choices=CONTEXT_ORDER_VALUES,
        default="rank",
        help="Prompt context render order for --context. Default: rank.",
    )
    add_events_jsonl_flag(search)
    search.add_argument("--json", action="store_true", help="Emit JSON output.")
    apply_command_examples(search, "search")


__all__ = ["add_search_command"]
