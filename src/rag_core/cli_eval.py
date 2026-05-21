from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli_core_runtime import run_with_ready_core
from rag_core.cli_eval_options import (
    compare_search_profiles_from_args,
    eval_quality_gate_thresholds_from_args,
    eval_run_payload,
    rerank_budget_from_args,
)
from rag_core.cli_search_options import query_plan_from_args
from rag_core.core_models import RAGCoreConfig
from rag_core.evals import (
    add_quality_gate,
    eval_comparison_report,
    eval_exit_code,
    eval_profile_comparison_report,
    eval_report,
    format_eval_comparison_report,
    format_eval_profile_comparison_report,
    format_eval_report,
    load_cases,
    redact_eval_report,
    run_eval,
)
from rag_core.search.planning import search_profile

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.evals import EvalCase, EvalResult
    from rag_core.search.query_plan import QueryPlan
    from rag_core.search.types import RerankBudget


async def run_eval_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[[RAGCoreConfig], "RAGCore"],
) -> int:
    if args.json_raw and not args.json:
        raise ValueError("--json-raw requires --json")
    if args.rerank and args.compare_rerank:
        raise ValueError("--rerank and --compare-rerank cannot be used together")
    compare_search_profiles = compare_search_profiles_from_args(args)
    if compare_search_profiles and args.compare_rerank:
        raise ValueError("--compare-search-profiles and --compare-rerank cannot be used together")
    rerank_budget = rerank_budget_from_args(args)
    if rerank_budget is not None and not (args.rerank or args.compare_rerank):
        raise ValueError("rerank budget flags require --rerank or --compare-rerank")
    gate_thresholds = eval_quality_gate_thresholds_from_args(args)
    query_plan = query_plan_from_args(args, limit=10)

    cases = load_cases(Path(args.cases))
    config = RAGCoreConfig.from_cli(args)
    profile_runs: dict[str, dict[str, object]] = {}
    if compare_search_profiles:
        profile_results: dict[str, Sequence[EvalResult]] = {}
        for profile in compare_search_profiles:
            profile_results[profile] = await _run_eval_with_core(
                config,
                core_factory=core_factory,
                cases=cases,
                rerank=args.rerank,
                rerank_budget=rerank_budget,
                query_plan=search_profile(profile, limit=10),
            )
            profile_runs[profile] = eval_run_payload(
                args,
                mode="search_profile",
                rerank=args.rerank,
                rerank_budget=rerank_budget,
                search_profile=profile,
            )
    elif args.compare_rerank:
        baseline_results = await _run_eval_with_core(
            config,
            core_factory=core_factory,
            cases=cases,
            rerank=False,
            rerank_budget=None,
            query_plan=query_plan,
        )
        reranked_results = await _run_eval_with_core(
            config,
            core_factory=core_factory,
            cases=cases,
            rerank=True,
            rerank_budget=rerank_budget,
            query_plan=query_plan,
        )
    else:
        results = await _run_eval_with_core(
            config,
            core_factory=core_factory,
            cases=cases,
            rerank=args.rerank,
            rerank_budget=rerank_budget,
            query_plan=query_plan,
        )
    if compare_search_profiles:
        payload = eval_profile_comparison_report(
            profile_results,
            profile_runs=profile_runs,
        )
        add_quality_gate(
            payload,
            {
                name: _require_mapping(profile)
                for name, profile in _require_mapping(payload.get("profiles")).items()
            },
            gate_thresholds,
        )
        _attach_overall_status(payload)
        if args.json:
            _emit_json(payload, raw=args.json_raw)
            return eval_exit_code(payload)
        print("\n".join(format_eval_profile_comparison_report(payload)))
        return eval_exit_code(payload)
    if args.compare_rerank:
        payload = eval_comparison_report(
            baseline_results,
            reranked_results,
            baseline_run=eval_run_payload(
                args,
                mode="baseline",
                rerank=False,
                rerank_budget=None,
                search_profile=args.search_profile,
                query_plan_preset=args.query_plan_preset,
            ),
            reranked_run=eval_run_payload(
                args,
                mode="reranked",
                rerank=True,
                rerank_budget=rerank_budget,
                search_profile=args.search_profile,
                query_plan_preset=args.query_plan_preset,
            ),
        )
        add_quality_gate(
            payload,
            {
                "reranked": _require_mapping(payload.get("reranked")),
            },
            gate_thresholds,
        )
        _attach_overall_status(payload)
        if args.json:
            _emit_json(payload, raw=args.json_raw)
            return eval_exit_code(payload)
        print("\n".join(format_eval_comparison_report(payload)))
        return eval_exit_code(payload)
    payload = eval_report(
        results,
        run=eval_run_payload(
            args,
            mode="single",
            rerank=args.rerank,
            rerank_budget=rerank_budget,
            search_profile=args.search_profile,
            query_plan_preset=args.query_plan_preset,
        ),
    )
    add_quality_gate(payload, {"eval": payload}, gate_thresholds)
    _attach_overall_status(payload)
    if args.json:
        _emit_json(payload, raw=args.json_raw)
        return eval_exit_code(payload)
    print("\n".join(format_eval_report(payload)))
    return eval_exit_code(payload)


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _emit_json(payload: dict[str, object], *, raw: bool) -> None:
    output = payload if raw else redact_eval_report(payload)
    print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))


def _attach_overall_status(payload: dict[str, object]) -> None:
    exit_code = eval_exit_code(payload)
    payload["overall"] = {
        "passed": exit_code == 0,
        "exit_code": exit_code,
    }


async def _run_eval_with_core(
    config: RAGCoreConfig,
    *,
    core_factory: Callable[[RAGCoreConfig], "RAGCore"],
    cases: Sequence["EvalCase"],
    rerank: bool,
    rerank_budget: "RerankBudget | None",
    query_plan: "QueryPlan | None",
) -> list[EvalResult]:
    async def run(core: RAGCore) -> list[EvalResult]:
        results = await run_eval(
            core,
            cases,
            rerank=rerank,
            rerank_budget=rerank_budget,
            query_plan=query_plan,
        )
        return results

    return await run_with_ready_core(
        core_factory=lambda: core_factory(config),
        action="eval",
        run=run,
    )
