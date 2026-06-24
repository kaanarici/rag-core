from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

from rag_core.evals import (
    EvalCase,
    EvalResult,
    eval_report,
    load_cases,
    run_eval,
)
from rag_core.evals.report_models import EvalReport
from rag_core.local_search.models import (
    DEFAULT_LOCAL_MAX_FILES,
    DEFAULT_LOCAL_SEARCH_COLLECTION,
    LocalSearchRequest,
    LocalSearchSkippedFailure,
)
from rag_core.local_search.planning import (
    LocalSearchRunSpec,
    build_local_search_run_spec,
)
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.safe_messages import error_message
from rag_core.search.planning import DEFAULT_SEARCH_PROFILE, search_profile

if TYPE_CHECKING:
    from rag_core import Engine, SearchResult
    from rag_core.search import QueryPlan, RerankBudget


@dataclass(frozen=True)
class LocalEvalScope:
    namespace: str
    collection: str


@dataclass(frozen=True)
class LocalEvalResult:
    report: EvalReport
    indexed: list[dict[str, object]]
    skipped_unsupported_count: int
    skipped_empty_count: int
    skipped_failed: list[LocalSearchSkippedFailure]
    truncated: bool

    @property
    def indexed_count(self) -> int:
        return len(self.indexed)

    @property
    def skipped_count(self) -> int:
        return (
            self.skipped_unsupported_count
            + self.skipped_empty_count
            + len(self.skipped_failed)
        )


class LocalEvalCore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def add_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> object: ...

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int,
        rerank: bool,
        rerank_budget: "RerankBudget | None" = None,
        query_plan: "QueryPlan | None" = None,
    ) -> list["SearchResult"]: ...

    async def close(self) -> None: ...


LocalEvalCoreFactory = Callable[[], LocalEvalCore]


async def run_local_eval(
    *,
    path: Path,
    cases_path: Path,
    max_files: int = DEFAULT_LOCAL_MAX_FILES,
    max_concurrency: int = 1,
    search_profile_name: str = DEFAULT_SEARCH_PROFILE,
    core_factory: LocalEvalCoreFactory | None = None,
) -> LocalEvalResult:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    cases = load_cases(cases_path)
    scope = infer_local_eval_scope(cases)
    run_spec = build_local_search_run_spec(
        LocalSearchRequest(
            path=path,
            query=cases[0].query,
            namespace=scope.namespace,
            collection=scope.collection,
            max_files=max_files,
        )
    )
    eval_cases = _resolve_local_eval_case_ids(cases, run_spec)
    factory = core_factory or _default_local_eval_core_factory
    core = factory()
    wall_clock_seconds = 0.0
    results: list[EvalResult] | None = None
    try:
        await core.ensure_ready()
        indexed, skipped_failed = await _index_local_eval_documents(core, run_spec)
        if not indexed:
            raise ValueError("no files could be indexed under %s" % run_spec.root)
        eval_started = perf_counter()
        plan = search_profile(search_profile_name, limit=DEFAULT_SEARCH_LIMIT)
        results = await run_eval(
            cast("Engine", core),
            eval_cases,
            query_plan=plan,
            max_concurrency=max_concurrency,
        )
        wall_clock_seconds = perf_counter() - eval_started
    finally:
        await core.close()

    run = _local_eval_run_metadata(
        search_profile_name=search_profile_name,
        max_concurrency=max_concurrency,
        wall_clock_seconds=wall_clock_seconds,
        run_spec=run_spec,
        indexed_count=len(indexed),
        skipped_failed_count=len(skipped_failed),
    )
    assert results is not None
    report = eval_report(results, run=run)
    return LocalEvalResult(
        report=report,
        indexed=indexed,
        skipped_unsupported_count=run_spec.skipped_unsupported_count,
        skipped_empty_count=run_spec.skipped_empty_count,
        skipped_failed=skipped_failed,
        truncated=run_spec.truncated,
    )


def infer_local_eval_scope(cases: Sequence[EvalCase]) -> LocalEvalScope:
    namespaces = {case.namespace for case in cases}
    if len(namespaces) != 1:
        raise ValueError("eval cases must share one namespace")
    collections = {case.collections for case in cases}
    if len(collections) != 1:
        raise ValueError("eval cases must share one collections value")
    only_collections = next(iter(collections))
    if len(only_collections) != 1:
        raise ValueError("eval cases must target exactly one collection")
    return LocalEvalScope(
        namespace=next(iter(namespaces)),
        collection=only_collections[0],
    )


def _resolve_local_eval_case_ids(
    cases: Sequence[EvalCase],
    run_spec: LocalSearchRunSpec,
) -> list[EvalCase]:
    document_key_by_alias = _document_key_aliases(run_spec)
    return [
        EvalCase(
            query=case.query,
            namespace=case.namespace,
            collections=case.collections,
            expected_ids=tuple(
                document_key_by_alias.get(expected_id, expected_id)
                for expected_id in case.expected_ids
            ),
            expected_grades=_resolved_expected_grades(
                case.expected_grades,
                document_key_by_alias,
            ),
            case_id=case.case_id,
            expected_context_contains=case.expected_context_contains,
            forbidden_context_contains=case.forbidden_context_contains,
            forbidden_private_identifiers=case.forbidden_private_identifiers,
            expected_citation_count_min=case.expected_citation_count_min,
            expected_source_count_min=case.expected_source_count_min,
            max_context_chars=case.max_context_chars,
            max_context_tokens=case.max_context_tokens,
        )
        for case in cases
    ]


def _document_key_aliases(run_spec: LocalSearchRunSpec) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for document in run_spec.documents:
        aliases[document.document_key] = document.document_key
        local_alias = _local_document_key_alias(document.document_key)
        if local_alias is not None:
            aliases.setdefault(local_alias, document.document_key)
    return aliases


def _local_document_key_alias(document_key: str) -> str | None:
    if not document_key.startswith("local:"):
        return None
    public_key, separator, _source = document_key.removeprefix("local:").partition(
        "#source:"
    )
    if not separator or not public_key:
        return None
    return public_key


def _resolved_expected_grades(
    grades: Mapping[str, int] | None,
    document_key_by_alias: Mapping[str, str],
) -> Mapping[str, int] | None:
    if grades is None:
        return None
    return {
        document_key_by_alias.get(expected_id, expected_id): grade
        for expected_id, grade in grades.items()
    }


async def _index_local_eval_documents(
    core: LocalEvalCore,
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
                document_id=document.document_key,
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


def _local_eval_run_metadata(
    *,
    search_profile_name: str,
    max_concurrency: int,
    wall_clock_seconds: float,
    run_spec: LocalSearchRunSpec,
    indexed_count: int,
    skipped_failed_count: int,
) -> EvalReport:
    payload: EvalReport = {
        "mode": "local_eval",
        "vector_store": "embedded_qdrant",
        "embedding_model": "demo-dense-v1",
        "embedding_note": "deterministic demo embeddings check wiring, not semantic retrieval quality",
        "rerank": False,
        "max_concurrency": max_concurrency,
        "wall_clock_seconds": wall_clock_seconds,
        "namespace": run_spec.namespace,
        "collection": run_spec.collection,
        "indexed_count": indexed_count,
        "skipped_count": (
            run_spec.skipped_unsupported_count
            + run_spec.skipped_empty_count
            + skipped_failed_count
        ),
        "skipped_unsupported_count": run_spec.skipped_unsupported_count,
        "skipped_empty_count": run_spec.skipped_empty_count,
        "skipped_failed_count": skipped_failed_count,
        "truncated": run_spec.truncated,
        "search_profile": search_profile_name,
    }
    return payload


def _default_local_eval_core_factory() -> LocalEvalCore:
    from rag_core.demo import build_demo_core

    return build_demo_core(store_collection=DEFAULT_LOCAL_SEARCH_COLLECTION)


def _dataclass_payload(value: object) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        return {str(key): item for key, item in payload.items()}
    if hasattr(value, "__dict__"):
        return {str(key): item for key, item in vars(value).items()}
    raise TypeError(f"value cannot be converted to payload: {type(value)!r}")


__all__ = [
    "LocalEvalCore",
    "LocalEvalCoreFactory",
    "LocalEvalResult",
    "LocalEvalScope",
    "infer_local_eval_scope",
    "run_local_eval",
]
