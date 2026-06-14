from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, Protocol, cast

import pytest

import rag_core.cli_doctor_providers as doctor_providers
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.demo import DemoEmbeddingProvider
from rag_core.search.providers.cohere import CohereEmbeddingProvider, CohereReranker
from rag_core.search.providers.local_embedding import LocalEmbeddingProvider
from rag_core.search.providers.reranker import NoOpReranker
from rag_core.search.providers.embedding import OpenAIEmbeddingProvider
from rag_core.search.providers.voyage import VoyageEmbeddingProvider, VoyageReranker
from rag_core.search.providers.zeroentropy import (
    ZeroEntropyEmbeddingProvider,
    ZeroEntropyReranker,
)
from tests.support import TEST_API_SECRET

pytestmark = [pytest.mark.plumbing]

SECRET = TEST_API_SECRET


class _HealthProvider(Protocol):
    async def check_health(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class _HealthCase:
    id: str
    build: Callable[[list[object]], tuple[_HealthProvider, Callable[[], int]]]
    success: object
    provider: str
    kind: str


class AuthenticationFailure(Exception):
    pass


class NetworkConnectionError(Exception):
    pass


class _Sequence:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def next(self) -> object:
        self.calls += 1
        if not self._outcomes:
            raise AssertionError("test provider outcomes exhausted")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _OpenAIEmbeddings:
    def __init__(self, outcomes: list[object]) -> None:
        self.sequence = _Sequence(outcomes)

    async def create(self, **_kwargs: object) -> object:
        return self.sequence.next()


class _VoyageEmbedClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.sequence = _Sequence(outcomes)

    def embed(self, _texts: list[str], **_kwargs: object) -> object:
        return self.sequence.next()


class _VoyageRerankClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.sequence = _Sequence(outcomes)

    def rerank(
        self,
        _query: str,
        _documents: list[str],
        _model: str,
        _top_k: int,
    ) -> object:
        return self.sequence.next()


class _ZeroEntropyModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.sequence = _Sequence(outcomes)

    def embed(self, **_kwargs: object) -> object:
        return self.sequence.next()

    def rerank(self, **_kwargs: object) -> object:
        return self.sequence.next()


class _CohereClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.sequence = _Sequence(outcomes)

    async def rerank(self, **_kwargs: object) -> object:
        return self.sequence.next()

    async def embed(self, **_kwargs: object) -> object:
        return self.sequence.next()


def _openai_case(outcomes: list[object]) -> tuple[_HealthProvider, Callable[[], int]]:
    api = _OpenAIEmbeddings(outcomes)
    provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
    provider._client = SimpleNamespace(embeddings=api)
    provider._model = "text-embedding-3-small"
    provider._dimensions = 2
    provider._send_dimensions = True
    return cast(_HealthProvider, provider), lambda: api.sequence.calls


def _voyage_embedding_case(
    outcomes: list[object],
) -> tuple[_HealthProvider, Callable[[], int]]:
    client = _VoyageEmbedClient(outcomes)
    provider = VoyageEmbeddingProvider.__new__(VoyageEmbeddingProvider)
    provider._client = client
    provider._model = "voyage-4"
    provider._dimensions = 2
    provider._send_dimensions = True
    return cast(_HealthProvider, provider), lambda: client.sequence.calls


def _voyage_reranker_case(
    outcomes: list[object],
) -> tuple[_HealthProvider, Callable[[], int]]:
    client = _VoyageRerankClient(outcomes)
    provider = VoyageReranker.__new__(VoyageReranker)
    provider._client = client
    provider._model = "rerank-2.5-lite"
    return cast(_HealthProvider, provider), lambda: client.sequence.calls


def _zeroentropy_embedding_case(
    outcomes: list[object],
) -> tuple[_HealthProvider, Callable[[], int]]:
    models = _ZeroEntropyModels(outcomes)
    provider = ZeroEntropyEmbeddingProvider.__new__(ZeroEntropyEmbeddingProvider)
    provider._client = SimpleNamespace(models=models)
    provider._model = "zembed-1"
    provider._dimensions = 2
    return cast(_HealthProvider, provider), lambda: models.sequence.calls


def _zeroentropy_reranker_case(
    outcomes: list[object],
) -> tuple[_HealthProvider, Callable[[], int]]:
    models = _ZeroEntropyModels(outcomes)
    provider = ZeroEntropyReranker.__new__(ZeroEntropyReranker)
    provider._client = SimpleNamespace(models=models)
    provider._model = "zerank-2"
    return cast(_HealthProvider, provider), lambda: models.sequence.calls


def _cohere_case(outcomes: list[object]) -> tuple[_HealthProvider, Callable[[], int]]:
    client = _CohereClient(outcomes)
    provider = CohereReranker.__new__(CohereReranker)
    provider._client = client
    provider._model = "rerank-v3.5"
    return cast(_HealthProvider, provider), lambda: client.sequence.calls


def _cohere_embedding_case(
    outcomes: list[object],
) -> tuple[_HealthProvider, Callable[[], int]]:
    client = _CohereClient(outcomes)
    provider = CohereEmbeddingProvider.__new__(CohereEmbeddingProvider)
    provider._client = client
    provider._model = "embed-v4.0"
    provider._dimensions = 2
    return cast(_HealthProvider, provider), lambda: client.sequence.calls


HEALTH_CASES = (
    _HealthCase(
        id="openai_embedding",
        build=_openai_case,
        success=SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]),
        provider="openai",
        kind="embedding",
    ),
    _HealthCase(
        id="voyage_embedding",
        build=_voyage_embedding_case,
        success=SimpleNamespace(embeddings=[[0.1, 0.2]]),
        provider="voyage",
        kind="embedding",
    ),
    _HealthCase(
        id="voyage_reranker",
        build=_voyage_reranker_case,
        success=SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=0.9)]
        ),
        provider="voyage",
        kind="reranker",
    ),
    _HealthCase(
        id="zeroentropy_embedding",
        build=_zeroentropy_embedding_case,
        success=SimpleNamespace(
            results=[SimpleNamespace(embedding=[0.1, 0.2])]
        ),
        provider="zeroentropy",
        kind="embedding",
    ),
    _HealthCase(
        id="zeroentropy_reranker",
        build=_zeroentropy_reranker_case,
        success=SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=0.9)]
        ),
        provider="zeroentropy",
        kind="reranker",
    ),
    _HealthCase(
        id="cohere_reranker",
        build=_cohere_case,
        success=SimpleNamespace(
            results=[SimpleNamespace(index=0, relevance_score=0.9)]
        ),
        provider="cohere",
        kind="reranker",
    ),
    _HealthCase(
        id="cohere_embedding",
        build=_cohere_embedding_case,
        success=SimpleNamespace(
            embeddings=SimpleNamespace(float_=[[0.1, 0.2]])
        ),
        provider="cohere",
        kind="embedding",
    ),
)


@pytest.mark.parametrize("case", HEALTH_CASES, ids=lambda case: case.id)
def test_provider_health_check_uses_one_successful_probe(case: _HealthCase) -> None:
    provider, calls = case.build([case.success])

    health = asyncio.run(provider.check_health())

    assert health["healthy"] is True
    assert health["adapter"] == case.provider
    assert health["kind"] == case.kind
    assert calls() == 1


@pytest.mark.parametrize("case", HEALTH_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize(
    ("exc", "category", "message_term"),
    (
        (AuthenticationFailure(f"raw auth failure {SECRET}"), "auth", "verify"),
        (
            NetworkConnectionError(f"raw network failure {SECRET}"),
            "network",
            "network access",
        ),
    ),
)
def test_provider_health_check_failures_are_sanitized_and_not_retried(
    case: _HealthCase,
    exc: Exception,
    category: str,
    message_term: str,
) -> None:
    provider, calls = case.build([exc, case.success])

    health = asyncio.run(provider.check_health())

    assert health["healthy"] is False
    assert health["adapter"] == case.provider
    assert health["kind"] == case.kind
    assert health["error"] == type(exc).__name__
    assert health["error_category"] == category
    assert message_term in str(health["message"])
    assert calls() == 1
    assert SECRET not in repr(health)
    assert "raw " not in repr(health)


def test_demo_local_and_noop_health_checks() -> None:
    class _FakeTextEmbedding:
        constructed: ClassVar[int] = 0

        def __init__(self, **_kwargs: object) -> None:
            type(self).constructed += 1

    demo = asyncio.run(DemoEmbeddingProvider(dimensions=7).check_health())
    local = LocalEmbeddingProvider(
        model="BAAI/bge-small-en-v1.5",
        text_embedding_loader=lambda: _FakeTextEmbedding,
    )
    local_health = asyncio.run(local.check_health())
    noop = asyncio.run(NoOpReranker().check_health())

    assert demo["healthy"] is True
    assert demo["adapter"] == "demo"
    assert local_health["healthy"] is True
    assert local_health["adapter"] == "local"
    assert _FakeTextEmbedding.constructed == 1
    assert noop["healthy"] is True
    assert noop["adapter"] == "none"


def test_local_health_failure_omits_loader_exception_detail() -> None:
    def fail_loader() -> type[Any]:
        raise RuntimeError(f"raw local loader failure {SECRET}")

    provider = LocalEmbeddingProvider(
        model="BAAI/bge-small-en-v1.5",
        text_embedding_loader=fail_loader,
    )

    health = asyncio.run(provider.check_health())

    assert health["healthy"] is False
    assert health["error"] == "RuntimeError"
    assert health["error_category"] == "provider"
    assert SECRET not in repr(health)
    assert "raw local loader failure" not in repr(health)


def test_doctor_provider_health_skips_providers_without_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoHealthEmbedding:
        @property
        def dimensions(self) -> int:
            return 64

        @property
        def model_name(self) -> str:
            return "demo-dense-v1"

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        async def embed_query(self, query: str) -> list[float]:
            return [0.0]

    monkeypatch.setattr(
        doctor_providers,
        "create_embedding_provider",
        lambda **_kwargs: _NoHealthEmbedding(),
    )
    monkeypatch.setattr(
        doctor_providers,
        "create_reranker",
        lambda **_kwargs: object(),
    )
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=64,
        ),
    )

    health = asyncio.run(doctor_providers.exercise_doctor_model_providers(config))

    assert health == {}
