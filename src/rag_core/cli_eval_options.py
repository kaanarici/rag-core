from __future__ import annotations

import argparse
import math

from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.types import RerankBudget


def rerank_budget_from_args(args: argparse.Namespace) -> RerankBudget | None:
    if not any(
        (
            args.rerank_candidates is not None,
            args.rerank_max_output is not None,
            args.rerank_timeout is not None,
            args.rerank_fail_fast,
        )
    ):
        return None
    return RerankBudget(
        candidate_count=args.rerank_candidates,
        max_output=args.rerank_max_output,
        timeout_seconds=args.rerank_timeout,
        fallback_on_error=not args.rerank_fail_fast,
    )


def compare_search_profiles_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    profiles = tuple(args.compare_search_profiles or ())
    if not profiles:
        return ()
    if len(profiles) < 2:
        raise ValueError("--compare-search-profiles requires at least two profiles")
    duplicates = sorted({profile for profile in profiles if profiles.count(profile) > 1})
    if duplicates:
        names = ", ".join(duplicates)
        raise ValueError(f"--compare-search-profiles entries must be unique: {names}")
    return tuple(sorted(profiles))


def eval_quality_gate_thresholds_from_args(
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    metric_floors = (
        ("min_recall_at_5", "--min-recall-at-5", "recall_at_5"),
        ("min_recall_at_10", "--min-recall-at-10", "recall_at_10"),
        ("min_mrr", "--min-mrr", "mrr"),
        ("min_ndcg_at_10", "--min-ndcg-at-10", "ndcg_at_10"),
    )
    for attr, flag, metric in metric_floors:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{flag} must be a finite number between 0 and 1")
        thresholds[metric] = {"minimum": float(value)}
    latency_ceiling = getattr(args, "max_mean_latency_ms", None)
    if latency_ceiling is not None:
        if not math.isfinite(latency_ceiling) or latency_ceiling < 0.0:
            raise ValueError("--max-mean-latency-ms must be finite and non-negative")
        thresholds["latency_ms"] = {"maximum": float(latency_ceiling)}
    p95_latency_ceiling = getattr(args, "max_p95_latency_ms", None)
    if p95_latency_ceiling is not None:
        if not math.isfinite(p95_latency_ceiling) or p95_latency_ceiling < 0.0:
            raise ValueError("--max-p95-latency-ms must be finite and non-negative")
        thresholds["latency_p95_ms"] = {"maximum": float(p95_latency_ceiling)}
    throughput_floor = getattr(args, "min_throughput_qps", None)
    if throughput_floor is not None:
        if not math.isfinite(throughput_floor) or throughput_floor < 0.0:
            raise ValueError("--min-throughput-qps must be finite and non-negative")
        thresholds["throughput_qps"] = {"minimum": float(throughput_floor)}
    return thresholds


def eval_run_payload(
    args: argparse.Namespace,
    *,
    mode: str,
    rerank: bool,
    rerank_budget: RerankBudget | None,
    search_profile: str | None = None,
    query_plan_preset: str | None = None,
) -> dict[str, object]:
    runtime = describe_runtime_metadata()
    payload: dict[str, object] = {
        "mode": mode,
        "rag_core_version": runtime.get("package_version"),
        "python_version": runtime.get("python_version"),
        "vector_store": args.vector_store,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_dimensions": args.embedding_dimensions,
        "search_profile": search_profile,
        "query_plan_preset": query_plan_preset,
        "rerank": rerank,
        "reranker_provider": args.reranker_provider,
        "reranker_model": args.reranker_model,
        "rerank_budget": _rerank_budget_payload(rerank_budget),
    }
    if args.vector_store == "qdrant":
        payload["qdrant_collection"] = args.qdrant_collection
    if args.vector_store == "turbopuffer":
        payload["turbopuffer_namespace"] = args.turbopuffer_namespace
    return {key: value for key, value in payload.items() if value is not None}


def _rerank_budget_payload(budget: RerankBudget | None) -> dict[str, object] | None:
    if budget is None:
        return None
    payload: dict[str, object] = {
        "candidate_count": budget.candidate_count,
        "max_output": budget.max_output,
        "timeout_seconds": budget.timeout_seconds,
        "fallback_on_error": budget.fallback_on_error,
    }
    return {key: value for key, value in payload.items() if value is not None}
