from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rag_core.cli.config import config_from_args
from rag_core.cli.inputs import parse_metadata_fields, parse_non_empty_values
from dataclasses import asdict
from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.demo import build_demo_core
from rag_core.ingest.local import LocalIngestResult
from rag_core.ingest.sources.local import reject_local_symlink_path
from rag_core.local import DEFAULT_LOCAL_TENANT_ID, local_collection
from rag_core.retrieval_defaults import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_LOCAL_SEARCH_LIMIT,
    DEFAULT_SEARCH_LIMIT,
)
from rag_core.search import And, Context, Filter, SearchResult, Term
from rag_core.search.context_pack import (
    context_pack_response_payload,
    validate_context_order,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


async def run_search_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., Engine],
    event_sink: EventSink | None,
) -> int:
    config: Config | None = None
    path = Path(args.path) if args.path is not None else None
    if path is not None:
        reject_local_symlink_path(path)
        path = path.resolve(strict=True)
        collections = list(args.collection or [local_collection(path)])
        if len(collections) != 1:
            raise ValueError("--collection can be supplied only once with a local path")
        tenant_id = args.tenant_id or DEFAULT_LOCAL_TENANT_ID
        if args.demo:

            def factory() -> Engine:
                return build_demo_core(
                    store_collection="local_search",
                    event_sink=event_sink,
                )
        else:
            config = config_from_args(args, path=path)

            def factory() -> Engine:
                return core_factory(config, event_sink=event_sink)
    else:
        if not args.collection:
            raise ValueError(
                "--collection is required (repeat the flag for multiple collections)"
            )
        if args.persist_dir is not None:
            raise ValueError("--persist-dir requires a local path")
        collections = list(args.collection)
        tenant_id = args.tenant_id
        config = config_from_args(args)

        def configured_core_factory() -> Engine:
            return core_factory(config, event_sink=event_sink)

        factory = configured_core_factory
    if not args.context and (
        args.max_context_chars is not None or args.max_context_tokens is not None
    ):
        raise ValueError("--max-context-* requires the context command")
    if args.max_context_chars is not None and args.max_context_chars <= 0:
        raise ValueError("--max-context-chars must be positive")
    if args.max_context_tokens is not None and args.max_context_tokens <= 0:
        raise ValueError("--max-context-tokens must be positive")
    limit = args.limit
    if limit is None:
        limit = (
            DEFAULT_CONTEXT_LIMIT
            if args.context
            else DEFAULT_LOCAL_SEARCH_LIMIT
            if path is not None
            else DEFAULT_SEARCH_LIMIT
        )
    if limit <= 0:
        raise ValueError("--limit must be positive")
    metadata_filter = _metadata_filter_from_fields(args.metadata_filter)
    content_types = parse_non_empty_values(args.content_type, field="--content-type")
    document_ids = parse_non_empty_values(args.document_id, field="--document-id")
    context_order = validate_context_order(args.context_order)

    async def run(
        core: Engine,
    ) -> tuple[LocalIngestResult | None, Context | list[SearchResult]]:
        ingest = None
        if path is not None:
            ingest = await core.add(
                path,
                tenant_id=tenant_id,
                collection=collections[0],
                max_files=args.max_files,
                prune_missing=path.is_dir()
                and not args.demo
                and config is not None
                and config.ingest.manifest_directory is not None,
            )
            if ingest.succeeded_count == 0:
                raise ValueError(f"no files could be indexed under {path}")
        if args.context:
            return ingest, await core.context(
                query=args.text,
                tenant_id=tenant_id,
                collections=collections,
                limit=limit,
                rerank=False,
                mode=args.mode,
                content_types=content_types,
                document_ids=document_ids,
                metadata_filter=metadata_filter,
                max_chars=args.max_context_chars,
                max_tokens=args.max_context_tokens,
            )
        return ingest, await core.search(
            query=args.text,
            tenant_id=tenant_id,
            collections=collections,
            limit=limit,
            rerank=False,
            mode=args.mode,
            content_types=content_types,
            document_ids=document_ids,
            metadata_filter=metadata_filter,
        )

    async with factory() as core:
        ingest, result = await run(core)
    if args.context:
        if ingest is not None and (ingest.failed_count or ingest.truncated):
            print(
                f"Index incomplete: {ingest.failed_count} failed file(s); truncated={ingest.truncated}",
                file=sys.stderr,
            )
        print(
            json.dumps(
                context_pack_response_payload(
                    cast(Context, result), context_order=context_order
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 1 if ingest is not None and ingest.failed_count else 0
    result = cast(list[SearchResult], result)
    if args.json:
        hits = [
            {key: value for key, value in asdict(hit).items() if value is not None}
            for hit in result
        ]
        payload: object = hits
        if ingest is not None:
            payload = {
                "query": args.text,
                "tenant_id": tenant_id,
                "collection": collections[0],
                "indexed_count": ingest.succeeded_count,
                "indexed": [
                    {
                        "document_id": r.document_id,
                        "document_key": r.document_key,
                        "filename": r.filename,
                        "chunk_count": r.chunk_count,
                        "ingest_state": r.ingest_state,
                    }
                    for r in ingest.succeeded
                ],
                "skipped_count": ingest.failed_count,
                "skipped_failed": [
                    {"path": r.path, "error": r.error} for r in ingest.failed
                ],
                "truncated": ingest.truncated,
                "hits": hits,
            }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        if ingest is not None:
            print(f"Indexed: {ingest.succeeded_count} files")
            print(f"Failed: {ingest.failed_count} files")
            if ingest.truncated:
                print(
                    "Truncated: yes; rerun with --max-files to include more supported files"
                )
            if ingest.failed:
                print("Failed files:")
                for failure in ingest.failed[:3]:
                    print(f"- {failure.path}: {failure.error}")
                if ingest.failed_count > 3:
                    print(f"- ... {ingest.failed_count - 3} more")
            print(f"Corpus: {tenant_id}/{collections[0]}")
            print(f"Query: {args.text}")
            print("Top hits:")
        for hit in result:
            title = hit.title or hit.document_key or hit.document_id or "unknown"
            text = hit.text.replace("\n", " ")
            print(f"- {hit.score:.3f} {title}: {text[:120]}")
        if not result:
            print("- none" if ingest is not None else "- no hits")
    return 1 if ingest is not None and ingest.failed_count else 0


def _metadata_filter_from_fields(values: Sequence[str]) -> Filter | None:
    terms = tuple(
        Term(field=k, value=v) for k, v in parse_metadata_fields(values).items()
    )
    return terms[0] if len(terms) == 1 else And(filters=terms) if terms else None
