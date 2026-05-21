from __future__ import annotations

import argparse

from rag_core.search.planning import query_plan_preset, search_profile
from rag_core.search.query_plan import QueryPlan


def query_plan_from_args(args: argparse.Namespace, *, limit: int) -> QueryPlan | None:
    if args.search_profile and args.query_plan_preset:
        raise ValueError("--search-profile and --query-plan-preset cannot be used together")
    if args.search_profile:
        return search_profile(args.search_profile, limit=limit)
    if args.query_plan_preset:
        return query_plan_preset(args.query_plan_preset, limit=limit)
    return None
