from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from rag_core.cli.core_runtime import run_with_ready_core
from rag_core.cli.inputs import parse_metadata_fields, parse_non_empty_values
from rag_core.cli.commands.local_search import (
    run_local_context_command,
    run_local_search_command,
)
from rag_core.cli.output import float_value, search_hit_payload
from rag_core.cli.search_options import query_plan_from_args
from rag_core.core_models import Config
from rag_core.retrieval_defaults import DEFAULT_CONTEXT_LIMIT, DEFAULT_SEARCH_LIMIT
from rag_core.search import And, Filter, Term
from rag_core.search.context_pack import (
    context_pack_response_payload,
    validate_context_order,
)

if TYPE_CHECKING:
    from rag_core.core import Engine
    from rag_core.events.sink import EventSink
    from rag_core.search import Context, SearchResult


async def run_search_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., "Engine"],
    event_sink: "EventSink | None",
) -> int:
    if getattr(args, "path", None) is not None:
        if args.context:
            return await run_local_context_command(args, event_sink=event_sink)
        return await run_local_search_command(args, event_sink=event_sink)
    if not args.collection:
        raise ValueError(
            "--collection is required (repeat the flag or use --collections a,b)"
        )
    if not args.context and (
        args.max_context_chars is not None or args.max_context_tokens is not None
    ):
        raise ValueError("--max-context-* requires the context command")
    if args.max_context_chars is not None and args.max_context_chars <= 0:
        raise ValueError("--max-context-chars must be positive")
    if args.max_context_tokens is not None and args.max_context_tokens <= 0:
        raise ValueError("--max-context-tokens must be positive")
    limit = _limit_from_args(args)
    if limit <= 0:
        raise ValueError("--limit must be positive")
    plan = query_plan_from_args(args, limit=limit)
    metadata_filter = _metadata_filter_from_fields(args.metadata_filter)
    content_types = parse_non_empty_values(args.content_type, field="--content-type")
    document_ids = parse_non_empty_values(args.document_id, field="--document-id")
    config = Config.from_cli(args)
    if args.context:
        action = "context"
        context_order = validate_context_order(args.context_order)

        async def run_context(core: Engine) -> "Context":
            return await core.context(
                query=args.text,
                namespace=args.namespace,
                collections=list(args.collection),
                limit=limit,
                rerank=args.rerank,
                query_plan=plan,
                content_types=content_types,
                document_ids=document_ids,
                metadata_filter=metadata_filter,
                max_chars=args.max_context_chars,
                max_tokens=args.max_context_tokens,
            )

        pack = await run_with_ready_core(
            core_factory=lambda: core_factory(config, event_sink=event_sink),
            action=action,
            run=run_context,
        )
    else:
        action = "search"

        async def run_search(core: Engine) -> list["SearchResult"]:
            return await core.search(
                query=args.text,
                namespace=args.namespace,
                collections=list(args.collection),
                limit=limit,
                rerank=args.rerank,
                query_plan=plan,
                content_types=content_types,
                document_ids=document_ids,
                metadata_filter=metadata_filter,
            )

        hits = await run_with_ready_core(
            core_factory=lambda: core_factory(config, event_sink=event_sink),
            action=action,
            run=run_search,
        )

    if args.context:
        print(
            json.dumps(
                context_pack_response_payload(pack, context_order=context_order),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    payload = [search_hit_payload(hit) for hit in hits]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not payload:
        print("- no hits")
        return 0
    for entry in payload:
        title = entry.get("title") or entry.get("document_id") or "unknown"
        text = str(entry.get("text") or "").replace("\n", " ")
        print(f"- {float_value(entry.get('score')):.3f} {title}: {text[:120]}")
    return 0


def _limit_from_args(args: argparse.Namespace) -> int:
    limit = args.limit
    if isinstance(limit, int):
        return limit
    return DEFAULT_CONTEXT_LIMIT if args.context else DEFAULT_SEARCH_LIMIT


def _metadata_filter_from_fields(values: Sequence[str]) -> Filter | None:
    fields = parse_metadata_fields(values)
    terms = tuple(Term(field=key, value=value) for key, value in fields.items())
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return And(filters=terms)
