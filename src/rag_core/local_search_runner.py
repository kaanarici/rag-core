from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Protocol

from rag_core.cli_inputs import cli_error_message
from rag_core.local_search_planning import (
    LocalSearchPlan,
    build_local_search_plan,
    default_corpus_id,
    discover_local_files,
)
from rag_core.local_search_models import (
    LocalSearchRequest,
    LocalSearchResult,
    LocalSearchSkippedFailure,
)
from rag_core.search.types import SearchResult


class LocalSearchCore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
    ) -> object: ...

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int,
        rerank: bool,
    ) -> list[SearchResult]: ...

    async def close(self) -> None: ...


LocalSearchCoreFactory = Callable[[], LocalSearchCore]


async def run_local_search(
    request: LocalSearchRequest,
    *,
    core_factory: LocalSearchCoreFactory | None = None,
) -> LocalSearchResult:
    plan = build_local_search_plan(request)
    factory = core_factory or _default_local_search_core_factory
    core = factory()
    try:
        await core.ensure_ready()
        indexed, skipped_failed = await _index_local_documents(core, plan)

        if not indexed:
            raise ValueError("no files could be indexed under %s" % plan.root)

        hits = await _search_local_documents(core, plan)
        return _local_search_result(
            plan, indexed=indexed, skipped_failed=skipped_failed, hits=hits
        )
    finally:
        await core.close()


def search_hit_payload(hit: SearchResult) -> dict[str, object]:
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
    plan: LocalSearchPlan,
) -> tuple[list[dict[str, object]], list[LocalSearchSkippedFailure]]:
    indexed: list[dict[str, object]] = []
    skipped_failed: list[LocalSearchSkippedFailure] = []
    for document in plan.documents:
        try:
            ingested = await core.ingest_file(
                document.path,
                namespace=plan.namespace,
                corpus_id=plan.corpus_id,
                document_key=document.document_key,
            )
        except Exception as exc:  # noqa: BLE001 - keep indexing remaining files.
            skipped_failed.append(
                LocalSearchSkippedFailure(
                    path=str(document.path), error=cli_error_message(exc)
                )
            )
            continue
        indexed.append(_dataclass_payload(ingested))
    return indexed, skipped_failed


async def _search_local_documents(
    core: LocalSearchCore, plan: LocalSearchPlan
) -> list[SearchResult]:
    return await core.search(
        query=plan.query,
        namespace=plan.namespace,
        corpus_ids=[plan.corpus_id],
        limit=plan.limit,
        rerank=False,
    )


def _local_search_result(
    plan: LocalSearchPlan,
    *,
    indexed: list[dict[str, object]],
    skipped_failed: list[LocalSearchSkippedFailure],
    hits: list[SearchResult],
) -> LocalSearchResult:
    return LocalSearchResult(
        query=plan.query,
        namespace=plan.namespace,
        corpus_id=plan.corpus_id,
        indexed=indexed,
        skipped_unsupported_count=plan.skipped_unsupported_count,
        skipped_empty_count=plan.skipped_empty_count,
        skipped_failed=skipped_failed,
        truncated=plan.truncated,
        hits=[search_hit_payload(hit) for hit in hits],
    )


def _default_local_search_core_factory() -> LocalSearchCore:
    from rag_core.demo import build_demo_core

    return build_demo_core(collection="local_search")


def _dataclass_payload(value: object) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        return {str(key): item for key, item in payload.items()}
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items()}
    raise TypeError(f"value cannot be converted to payload: {type(value)!r}")


__all__ = [
    "LocalSearchCore",
    "LocalSearchCoreFactory",
    "default_corpus_id",
    "discover_local_files",
    "run_local_search",
    "search_hit_payload",
]
