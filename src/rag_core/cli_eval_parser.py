from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_profile_help import query_plan_preset_help, search_profile_help
from rag_core.search.planning import QUERY_PLAN_PRESETS, SEARCH_PROFILES


def add_eval_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run retrieval-quality eval cases against an indexed corpus.",
    )
    eval_parser.description = (
        "Run JSONL eval cases through RAGCore.search. The corpus must already "
        "be indexed in the configured vector store."
    )
    add_config_flags(eval_parser)
    eval_parser.add_argument(
        "--cases",
        required=True,
        help=(
            "JSONL file of eval cases. Each row must include query, namespace, "
            "corpus_ids, and expected_chunk_ids."
        ),
    )
    eval_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Evaluate with the configured reranker enabled.",
    )
    eval_parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="Run baseline and reranked evals and emit metric deltas.",
    )
    eval_query_plan_group = eval_parser.add_mutually_exclusive_group()
    eval_query_plan_group.add_argument(
        "--search-profile",
        choices=SEARCH_PROFILES,
        default=None,
        help=search_profile_help(
            prefix="Common search profile to use for each eval query.",
            suffix="Mutually exclusive with --query-plan-preset.",
        ),
    )
    eval_query_plan_group.add_argument(
        "--query-plan-preset",
        choices=QUERY_PLAN_PRESETS,
        default=None,
        help=query_plan_preset_help(
            prefix="Built-in QueryPlan recipe to use for each eval query.",
            suffix="Mutually exclusive with --search-profile.",
        ),
    )
    eval_query_plan_group.add_argument(
        "--compare-search-profiles",
        choices=SEARCH_PROFILES,
        nargs="+",
        default=None,
        metavar="PROFILE",
        help=search_profile_help(
            prefix="Run multiple common search profiles on the same eval cases.",
            suffix=(
                "Mutually exclusive with --search-profile and --query-plan-preset."
            ),
        ),
    )
    eval_parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=None,
        help="Maximum retrieved candidates sent to the reranker.",
    )
    eval_parser.add_argument(
        "--rerank-max-output",
        type=int,
        default=None,
        help="Maximum reranked rows accepted from the provider.",
    )
    eval_parser.add_argument(
        "--rerank-timeout",
        type=float,
        default=None,
        help="Reranker timeout in seconds for each eval query.",
    )
    eval_parser.add_argument(
        "--rerank-fail-fast",
        action="store_true",
        help="Fail the eval instead of falling back when reranking errors.",
    )
    eval_parser.add_argument(
        "--min-recall-at-5",
        type=float,
        default=None,
        help="Fail with exit code 1 when mean recall@5 is below this floor.",
    )
    eval_parser.add_argument(
        "--min-recall-at-10",
        type=float,
        default=None,
        help="Fail with exit code 1 when mean recall@10 is below this floor.",
    )
    eval_parser.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Fail with exit code 1 when mean MRR is below this floor.",
    )
    eval_parser.add_argument(
        "--min-ndcg-at-10",
        type=float,
        default=None,
        help="Fail with exit code 1 when mean nDCG@10 is below this floor.",
    )
    eval_parser.add_argument(
        "--max-mean-latency-ms",
        type=float,
        default=None,
        help="Fail with exit code 1 when mean eval latency exceeds this ceiling.",
    )
    eval_parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=None,
        help="Fail with exit code 1 when p95 eval latency exceeds this ceiling.",
    )
    eval_parser.add_argument(
        "--min-throughput-qps",
        type=float,
        default=None,
        help="Fail with exit code 1 when eval throughput falls below this floor.",
    )
    add_events_jsonl_flag(eval_parser)
    eval_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    eval_parser.add_argument(
        "--json-raw",
        action="store_true",
        help="Emit raw case/query identifiers with --json.",
    )


__all__ = ["add_eval_command"]
