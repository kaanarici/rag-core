from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_profile_help import query_plan_preset_help, search_profile_help
from rag_core.search.planning import QUERY_PLAN_PRESETS, SEARCH_PROFILES


def add_query_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    _add_query_like_command(
        subparsers,
        "search",
        help="Search the configured vector store.",
        context_json_default=False,
        include_context_toggle=False,
        include_context_budget=False,
    )

    _add_query_like_command(
        subparsers,
        "retrieve-context",
        help="Search and emit a model-ready context pack JSON payload.",
        context_json_default=True,
        include_context_toggle=False,
    )


def _add_query_like_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help: str,
    context_json_default: bool,
    include_context_toggle: bool = True,
    include_context_budget: bool = True,
) -> argparse.ArgumentParser:
    query = subparsers.add_parser(
        name,
        help=help,
    )
    add_config_flags(query)
    query.add_argument("text", help="Search query text.")
    query.add_argument("--namespace", required=True)
    query.add_argument(
        "--corpus-id",
        action="append",
        default=[],
        help="Repeatable. At least one corpus must be specified.",
    )
    query.add_argument("--limit", type=int, default=10)
    query.add_argument(
        "--rerank",
        action="store_true",
        help="Apply the configured reranker to the result set.",
    )
    query_plan_group = query.add_mutually_exclusive_group()
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
    query.add_argument(
        "--metadata-filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable exact metadata filter applied to the query.",
    )
    query.set_defaults(context_json=context_json_default)
    if include_context_toggle:
        query.add_argument(
            "--context-json",
            action="store_true",
            help="Emit a model-ready context pack JSON payload instead of raw search hits.",
        )
    query.set_defaults(max_context_chars=None, max_context_tokens=None)
    if include_context_budget:
        query.add_argument(
            "--max-context-chars",
            type=int,
            default=None,
            help="Approximate character budget for returned snippets.",
        )
        query.add_argument(
            "--max-context-tokens",
            type=int,
            default=None,
            help="Approximate token budget for returned snippets.",
        )
    add_events_jsonl_flag(query)
    if context_json_default:
        query.set_defaults(json=False)
    else:
        query.add_argument("--json", action="store_true", help="Emit JSON output.")
    return query


__all__ = ["add_query_command"]
