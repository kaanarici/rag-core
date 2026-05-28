from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli_output import float_value, require_mapping
from rag_core.local_ingest import LocalSearchCoreFactory, run_local_search
from rag_core.local_search_models import (
    DEFAULT_LOCAL_SEARCH_COLLECTION,
    LocalSearchRequest,
)

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.events.sink import EventSink


async def run_local_search_command(
    args: argparse.Namespace,
    *,
    event_sink: "EventSink | None",
) -> int:
    core_factory: LocalSearchCoreFactory | None = None
    if event_sink is not None:
        from rag_core.demo import build_demo_core

        def core_factory() -> RAGCore:
            return build_demo_core(
                collection=DEFAULT_LOCAL_SEARCH_COLLECTION, event_sink=event_sink
            )

    result = await run_local_search(
        LocalSearchRequest(
            path=Path(args.path),
            query=args.query,
            namespace=args.namespace,
            corpus_id=args.corpus_id,
            limit=args.limit,
            max_files=args.max_files,
        ),
        core_factory=core_factory,
    )
    _emit_local_search(result.to_payload(), as_json=args.json)
    return 0


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
    print(f"Corpus: {payload.get('namespace')}/{payload.get('corpus_id')}")
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
