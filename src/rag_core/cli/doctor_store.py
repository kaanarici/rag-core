from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from rag_core.cli.inputs import cli_safe_error_message
from rag_core.core_models import Config
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions
from rag_core.search.vector_models import SparseVector

if TYPE_CHECKING:
    from rag_core.core import Engine


_DIMENSION_MISMATCH_RE = re.compile(
    r"uses (\d+) dimensions, but the current embedding provider uses (\d+)"
)


class DoctorStoreOutcome:
    __slots__ = ("health", "fix_summary")

    def __init__(self, health: dict[str, object], fix_summary: dict[str, object]) -> None:
        self.health = health
        self.fix_summary = fix_summary


class _DoctorEmbeddingStub:
    def __init__(self, *, model: str, dimensions: int) -> None:
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("doctor must not embed; use ingest/query for real workloads")

    async def embed_query(self, query: str) -> list[float]:
        raise RuntimeError("doctor must not embed; use ingest/query for real workloads")


class _DoctorSparseEmbedderStub:
    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        raise RuntimeError("doctor must not sparse-embed; use ingest/query for real workloads")

    def embed_query(self, query: str) -> SparseVector:
        raise RuntimeError("doctor must not sparse-embed; use ingest/query for real workloads")

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        raise RuntimeError("doctor must not sparse-embed; use ingest/query for real workloads")


async def exercise_doctor_store(
    config: Config,
    *,
    core_factory: Callable[..., "Engine"],
    capture_dimension_mismatch: bool,
) -> DoctorStoreOutcome:
    dimensions = resolve_embedding_dimensions(
        provider=config.embedding.provider,
        model=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )
    core = core_factory(
        config,
        embedding_provider=_DoctorEmbeddingStub(
            model=config.embedding.model,
            dimensions=dimensions,
        ),
        sparse_embedder=_DoctorSparseEmbedderStub(),
    )
    try:
        try:
            await core.ensure_ready()
        except ValueError as exc:
            mismatch = _parse_dimension_mismatch(str(exc))
            if mismatch is None or not capture_dimension_mismatch:
                raise
            actual, declared = mismatch
            return DoctorStoreOutcome(
                health={"healthy": False, "error": str(exc)},
                fix_summary={
                    "status": "dimension_mismatch",
                    "expected": declared,
                    "actual": actual,
                    "message": (
                        "Existing collection dimensions differ from the configured "
                        "embedding model. Doctor refuses to mutate; run "
                        "`rag-core add --force-reindex` against a fresh collection "
                        "or recreate the collection manually."
                    ),
                },
            )
        except Exception as exc:
            return DoctorStoreOutcome(
                health={"healthy": False, "error": cli_safe_error_message(exc, action="doctor")},
                fix_summary={
                    "status": "store_unavailable",
                    "message": "Vector store is not reachable with the current configuration.",
                },
            )
        health = await core.check_health()
        return DoctorStoreOutcome(
            health=health,
            fix_summary={
                "status": "ok",
                "expected": dimensions,
                "message": "Collection exists and matches the configured embedding shape.",
            },
        )
    finally:
        await core.close()


def _parse_dimension_mismatch(message: str) -> tuple[int, int] | None:
    match = _DIMENSION_MISMATCH_RE.search(message)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
