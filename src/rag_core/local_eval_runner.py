from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from rag_core.cli_inputs import cli_error_message
from rag_core.evals import (
    EvalCase,
    EvalThresholds,
    add_quality_gate,
    eval_report,
    load_cases,
    run_eval,
)
from rag_core.evals.report_models import EvalReport
from rag_core.local_search_models import (
    DEFAULT_LOCAL_MAX_FILES,
    DEFAULT_LOCAL_SEARCH_COLLECTION,
    LocalSearchRequest,
    LocalSearchSkippedFailure,
)
from rag_core.local_search_planning import (
    LocalSearchRunSpec,
    build_local_search_run_spec,
)
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.planning import DEFAULT_SEARCH_PROFILE, search_profile

if TYPE_CHECKING:
    from rag_core import RAGCore, SearchResult
    from rag_core.search import QueryPlan, RerankBudget


@dataclass(frozen=True)
class LocalEvalScope:
    namespace: str
    corpus_id: str


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

    async def ingest_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> object: ...

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
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
    search_profile_name: str = DEFAULT_SEARCH_PROFILE,
    thresholds: EvalThresholds | None = None,
    core_factory: LocalEvalCoreFactory | None = None,
) -> LocalEvalResult:
    cases = load_cases(cases_path)
    scope = infer_local_eval_scope(cases)
    run_spec = build_local_search_run_spec(
        LocalSearchRequest(
            path=path,
            query=cases[0].query,
            namespace=scope.namespace,
            corpus_id=scope.corpus_id,
            max_files=max_files,
        )
    )
    eval_cases = _resolve_local_eval_case_ids(cases, run_spec)
    factory = core_factory or _default_local_eval_core_factory
    core = factory()
    try:
        await core.ensure_ready()
        indexed, skipped_failed = await _index_local_eval_documents(core, run_spec)
        if not indexed:
            raise ValueError("no files could be indexed under %s" % run_spec.root)
        plan = search_profile(search_profile_name, limit=DEFAULT_SEARCH_LIMIT)
        results = await run_eval(cast("RAGCore", core), eval_cases, query_plan=plan)
    finally:
        await core.close()

    report = eval_report(
        results,
        run={
            "mode": "local_eval",
            "vector_store": "embedded_qdrant",
            "embedding_model": "demo-dense-v1",
            "search_profile": search_profile_name,
            "rerank": False,
            "namespace": run_spec.namespace,
            "corpus_id": run_spec.corpus_id,
            "indexed_count": len(indexed),
            "skipped_count": (
                run_spec.skipped_unsupported_count
                + run_spec.skipped_empty_count
                + len(skipped_failed)
            ),
            "skipped_unsupported_count": run_spec.skipped_unsupported_count,
            "skipped_empty_count": run_spec.skipped_empty_count,
            "skipped_failed_count": len(skipped_failed),
            "truncated": run_spec.truncated,
        },
    )
    add_quality_gate(report, {"local_eval": report}, thresholds or {})
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
        raise ValueError("local-eval cases must share one namespace")
    corpus_ids = {case.corpus_ids for case in cases}
    if len(corpus_ids) != 1:
        raise ValueError("local-eval cases must share one corpus_ids value")
    only_corpus_ids = next(iter(corpus_ids))
    if len(only_corpus_ids) != 1:
        raise ValueError("local-eval cases must target exactly one corpus_id")
    return LocalEvalScope(
        namespace=next(iter(namespaces)),
        corpus_id=only_corpus_ids[0],
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
            corpus_ids=case.corpus_ids,
            expected_ids=tuple(
                document_key_by_alias.get(expected_id, expected_id)
                for expected_id in case.expected_ids
            ),
            expected_grades=_resolved_expected_grades(
                case.expected_grades,
                document_key_by_alias,
            ),
            case_id=case.case_id,
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
            ingested = await core.ingest_file(
                document.path,
                namespace=run_spec.namespace,
                corpus_id=run_spec.corpus_id,
                document_id=document.document_key,
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


def _default_local_eval_core_factory() -> LocalEvalCore:
    from rag_core.demo import build_demo_core

    return build_demo_core(collection=DEFAULT_LOCAL_SEARCH_COLLECTION)


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
