from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli.output import float_value, require_mapping
from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.ingest.local import (
    LocalContextCoreFactory,
    run_local_context,
    run_local_search,
)
from rag_core.local_search.models import (
    DEFAULT_LOCAL_SEARCH_COLLECTION,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_LOCAL_SEARCH_NAMESPACE,
    LocalSearchRequest,
)
from rag_core.retrieval_defaults import DEFAULT_CONTEXT_LIMIT
from rag_core.search.context_pack import (
    context_pack_response_payload,
    validate_context_order,
)

if TYPE_CHECKING:
    from rag_core.core import Engine
    from rag_core.events.sink import EventSink


async def run_local_search_command(
    args: argparse.Namespace,
    *,
    event_sink: "EventSink | None",
) -> int:
    core_factory = _core_factory_from_args(args, event_sink=event_sink)
    result = await run_local_search(
        _local_search_request_from_args(args, default_limit=DEFAULT_LOCAL_SEARCH_LIMIT),
        core_factory=core_factory,
    )
    _emit_local_search(result.to_payload(), as_json=args.json)
    return 0


async def run_local_context_command(
    args: argparse.Namespace,
    *,
    event_sink: "EventSink | None",
) -> int:
    core_factory = _core_factory_from_args(args, event_sink=event_sink)
    pack = await run_local_context(
        _local_search_request_from_args(args, default_limit=DEFAULT_CONTEXT_LIMIT),
        max_chars=args.max_context_chars,
        max_tokens=args.max_context_tokens,
        core_factory=core_factory,
    )
    context_order = validate_context_order(args.context_order)
    print(
        json.dumps(
            context_pack_response_payload(pack, context_order=context_order),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _local_search_request_from_args(
    args: argparse.Namespace,
    *,
    default_limit: int,
) -> LocalSearchRequest:
    return LocalSearchRequest(
        path=Path(args.path),
        query=_query_from_args(args),
        namespace=args.namespace or DEFAULT_LOCAL_SEARCH_NAMESPACE,
        collection=_local_collection_from_args(args),
        limit=args.limit if isinstance(args.limit, int) else default_limit,
        max_files=args.max_files,
    )


def _query_from_args(args: argparse.Namespace) -> str:
    value = getattr(args, "query", None)
    if isinstance(value, str):
        return value
    return str(args.text)


def _local_collection_from_args(args: argparse.Namespace) -> str | None:
    value = getattr(args, "collection", None)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if len(value) > 1:
            raise ValueError("--collection can be supplied only once with a local path")
        if value:
            item = value[0]
            if isinstance(item, str):
                return item
    return None


def _core_factory_from_args(
    args: argparse.Namespace,
    *,
    event_sink: "EventSink | None",
) -> LocalContextCoreFactory:
    if args.demo:
        from rag_core.demo import build_demo_core

        def demo_core_factory() -> Engine:
            return build_demo_core(
                store_collection=DEFAULT_LOCAL_SEARCH_COLLECTION, event_sink=event_sink
            )

        return demo_core_factory

    def local_core_factory() -> Engine:
        return Engine(Config.local(), event_sink=event_sink)

    return local_core_factory


def _emit_local_search(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"Indexed: {_payload_int(payload, 'indexed_count')} files")
    print(
        "Skipped: "
        f"{_payload_int(payload, 'skipped_count')} files "
        f"(unsupported={_payload_int(payload, 'skipped_unsupported_count')}, "
        f"empty={_payload_int(payload, 'skipped_empty_count')}, "
        f"failed={len(_payload_list(payload, 'skipped_failed'))})"
    )
    if payload.get("truncated") is True:
        print("Truncated: yes; rerun with --max-files to include more supported files")
    _emit_skipped_failures(payload)
    print(f"Corpus: {payload.get('namespace')}/{payload.get('collection')}")
    print(f"Query: {payload.get('query')}")
    print("Top hits:")
    hits = payload.get("hits")
    if not isinstance(hits, list) or not hits:
        print("- none")
        return
    for raw_hit in hits:
        hit = require_mapping(raw_hit)
        title = (
            hit.get("title")
            or hit.get("document_key")
            or hit.get("document_id")
            or "unknown"
        )
        text = str(hit.get("text") or "").replace("\n", " ")
        print(f"- {float_value(hit.get('score')):.3f} {title}: {text[:120]}")


def _emit_skipped_failures(payload: dict[str, object]) -> None:
    skipped_failed = _payload_list(payload, "skipped_failed")
    if not skipped_failed:
        return
    print("Failed files:")
    for raw_failure in skipped_failed[:3]:
        failure = require_mapping(raw_failure)
        print(f"- {failure.get('path')}: {failure.get('error')}")
    if len(skipped_failed) > 3:
        print(f"- ... {len(skipped_failed) - 3} more")


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _payload_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    return []
