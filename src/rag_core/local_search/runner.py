from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Protocol

from rag_core.local_search.planning import (
    LocalSearchRunSpec,
    build_local_search_run_spec,
    default_collection,
    discover_local_files,
)
from rag_core.local_search.models import (
    LocalSearchRequest,
    LocalSearchResult,
    LocalSearchSkippedFailure,
)
from rag_core.safe_messages import error_message
from rag_core.search import Context, SearchResult


class LocalSearchCore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def add_file(
        self,
        path: Path,
        *,
        namespace: str,
        collection: str,
        document_key: str,
    ) -> object: ...

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int,
        rerank: bool,
    ) -> list[SearchResult]: ...

    async def close(self) -> None: ...


class LocalContextCore(LocalSearchCore, Protocol):
    async def context(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int,
        rerank: bool,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> Context: ...


LocalSearchCoreFactory = Callable[[], LocalSearchCore]
LocalContextCoreFactory = Callable[[], LocalContextCore]


async def run_local_search(
    request: LocalSearchRequest,
    *,
    core_factory: LocalSearchCoreFactory | None = None,
) -> LocalSearchResult:
    run_spec = build_local_search_run_spec(request)
    factory = core_factory or _default_local_search_core_factory
    core = factory()
    try:
        await core.ensure_ready()
        indexed, skipped_failed = await _index_local_documents(core, run_spec)

        if not indexed:
            raise ValueError("no files could be indexed under %s" % run_spec.root)

        hits = await _search_local_documents(core, run_spec)
        return _local_search_result(
            run_spec, indexed=indexed, skipped_failed=skipped_failed, hits=hits
        )
    finally:
        await core.close()


async def run_local_context(
    request: LocalSearchRequest,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    core_factory: LocalContextCoreFactory | None = None,
) -> Context:
    run_spec = build_local_search_run_spec(request)
    factory = core_factory or _default_local_search_core_factory
    core = factory()
    try:
        await core.ensure_ready()
        indexed, _skipped_failed = await _index_local_documents(core, run_spec)

        if not indexed:
            raise ValueError("no files could be indexed under %s" % run_spec.root)

        return await _context_local_documents(
            core,
            run_spec,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
    finally:
        await core.close()


def local_search_hit_payload(hit: SearchResult) -> dict[str, object]:
    payload = _dataclass_payload(hit)
    return {
        "score": payload.get("score"),
        "title": payload.get("title"),
        "document_id": payload.get("document_id"),
        "document_key": payload.get("document_key"),
        "document_path": payload.get("document_path"),
        "chunk_index": payload.get("chunk_index"),
        "text": payload.get("text"),
    }


async def _index_local_documents(
    core: LocalSearchCore,
    run_spec: LocalSearchRunSpec,
) -> tuple[list[dict[str, object]], list[LocalSearchSkippedFailure]]:
    indexed: list[dict[str, object]] = []
    skipped_failed: list[LocalSearchSkippedFailure] = []
    for document in run_spec.documents:
        try:
            ingested = await core.add_file(
                document.path,
                namespace=run_spec.namespace,
                collection=run_spec.collection,
                document_key=document.document_key,
            )
        except Exception as exc:  # noqa: BLE001 - keep indexing remaining files.
            skipped_failed.append(
                LocalSearchSkippedFailure(
                    path=str(document.path), error=error_message(exc)
                )
            )
            continue
        indexed.append(_dataclass_payload(ingested))
    return indexed, skipped_failed


async def _search_local_documents(
    core: LocalSearchCore, run_spec: LocalSearchRunSpec
) -> list[SearchResult]:
    return await core.search(
        query=run_spec.query,
        namespace=run_spec.namespace,
        collections=[run_spec.collection],
        limit=run_spec.limit,
        rerank=False,
    )


async def _context_local_documents(
    core: LocalContextCore,
    run_spec: LocalSearchRunSpec,
    *,
    max_chars: int | None,
    max_tokens: int | None,
) -> Context:
    return await core.context(
        query=run_spec.query,
        namespace=run_spec.namespace,
        collections=[run_spec.collection],
        limit=run_spec.limit,
        rerank=False,
        max_chars=max_chars,
        max_tokens=max_tokens,
    )


def _local_search_result(
    run_spec: LocalSearchRunSpec,
    *,
    indexed: list[dict[str, object]],
    skipped_failed: list[LocalSearchSkippedFailure],
    hits: list[SearchResult],
) -> LocalSearchResult:
    return LocalSearchResult(
        query=run_spec.query,
        namespace=run_spec.namespace,
        collection=run_spec.collection,
        indexed=indexed,
        skipped_unsupported_count=run_spec.skipped_unsupported_count,
        skipped_empty_count=run_spec.skipped_empty_count,
        skipped_failed=skipped_failed,
        truncated=run_spec.truncated,
        hits=[local_search_hit_payload(hit) for hit in hits],
    )


def _default_local_search_core_factory() -> LocalContextCore:
    from rag_core.core import Engine
    from rag_core.core_models import Config

    return Engine(Config.local())


def _dataclass_payload(value: object) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        return {str(key): item for key, item in payload.items()}
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items()}
    raise TypeError(f"value cannot be converted to payload: {type(value)!r}")


__all__ = [
    "LocalContextCore",
    "LocalContextCoreFactory",
    "LocalSearchCore",
    "LocalSearchCoreFactory",
    "default_collection",
    "discover_local_files",
    "local_search_hit_payload",
    "run_local_context",
    "run_local_search",
]
