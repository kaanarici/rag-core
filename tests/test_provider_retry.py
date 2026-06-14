from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest

from rag_core.search.providers import provider_retry
from rag_core.search.providers.cohere import CohereEmbeddingProvider, CohereReranker
from rag_core.search.providers.openai_embedding import embed_openai_texts
from rag_core.search.providers.provider_retry import retry_provider_call
from rag_core.search.providers.voyage import VoyageEmbeddingProvider, VoyageReranker
from rag_core.search.providers.zeroentropy import (
    ZeroEntropyEmbeddingProvider,
    ZeroEntropyReranker,
)
from tests.support import TEST_API_SECRET

LOGGER_NAME = "rag_core.search.providers.provider_retry"
SECRET = TEST_API_SECRET


@dataclass(frozen=True)
class _ProviderErrors:
    transient: Callable[[], Exception]
    permanent: Callable[[], Exception]


@dataclass(frozen=True)
class _CallSiteRun:
    run: Callable[[], Coroutine[Any, Any, object]]
    call_count: Callable[[], int]


@dataclass(frozen=True)
class _ProviderCase:
    id: str
    install_errors: Callable[[pytest.MonkeyPatch], _ProviderErrors]
    build_run: Callable[[list[object]], _CallSiteRun]
    success: object


class _SensitiveTransientError(Exception):
    pass


def test_retry_provider_call_logs_sanitized_retry_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> object:
        outcomes: list[object] = [
            _SensitiveTransientError(f"private prompt text {SECRET}"),
            "ok",
        ]
        sleeps: list[float] = []

        async def call_provider() -> object:
            return _pop_outcome(outcomes)

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        result = await retry_provider_call(
            call_provider,
            classify=lambda exc: isinstance(exc, _SensitiveTransientError),
            provider_name="openai",
            sleep=fake_sleep,
            rand=lambda: 0.5,
        )
        assert sleeps == [0.25]
        return result

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = asyncio.run(run())

    assert result == "ok"
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "provider=openai" in message
    assert "attempt=1" in message
    assert "max_attempts=3" in message
    assert "delay_seconds=0.250" in message
    assert "error_type=_SensitiveTransientError" in message
    assert "private prompt text" not in message
    assert SECRET not in message
    assert all(record.exc_info is None for record in caplog.records)


def test_retry_provider_call_re_raises_first_transient_after_budget() -> None:
    async def run() -> None:
        await retry_provider_call(
            call_provider,
            classify=lambda exc: isinstance(exc, _SensitiveTransientError),
            provider_name="voyage",
            sleep=fake_sleep,
            rand=lambda: 2.0,
            max_delay=0.75,
        )

    first = _SensitiveTransientError("first")
    outcomes: list[object] = [
        first,
        _SensitiveTransientError("second"),
        _SensitiveTransientError("third"),
    ]
    sleeps: list[float] = []

    async def call_provider() -> object:
        return _pop_outcome(outcomes)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(_SensitiveTransientError) as exc_info:
        asyncio.run(run())

    assert exc_info.value is first
    assert sleeps == [0.5, 0.75]


def test_provider_call_sites_retry_transient_errors_then_success(
    monkeypatch: pytest.MonkeyPatch,
    provider_case: _ProviderCase,
) -> None:
    sleeps = _patch_retry_clock(monkeypatch)
    errors = provider_case.install_errors(monkeypatch)
    run = provider_case.build_run(
        [errors.transient(), errors.transient(), provider_case.success]
    )

    result = asyncio.run(run.run())

    assert result
    assert run.call_count() == 3
    assert sleeps == [0.5, 1.0]


def test_provider_call_sites_do_not_retry_permanent_errors(
    monkeypatch: pytest.MonkeyPatch,
    provider_case: _ProviderCase,
) -> None:
    sleeps = _patch_retry_clock(monkeypatch)
    errors = provider_case.install_errors(monkeypatch)
    permanent = errors.permanent()
    run = provider_case.build_run([permanent, provider_case.success])

    with pytest.raises(type(permanent)) as exc_info:
        asyncio.run(run.run())

    assert exc_info.value is permanent
    assert run.call_count() == 1
    assert sleeps == []


def test_provider_call_sites_exhaust_budget_with_original_error(
    monkeypatch: pytest.MonkeyPatch,
    provider_case: _ProviderCase,
) -> None:
    sleeps = _patch_retry_clock(monkeypatch)
    errors = provider_case.install_errors(monkeypatch)
    first = errors.transient()
    run = provider_case.build_run(
        [first, errors.transient(), errors.transient(), provider_case.success]
    )

    with pytest.raises(type(first)) as exc_info:
        asyncio.run(run.run())

    assert exc_info.value is first
    assert run.call_count() == 3
    assert sleeps == [0.5, 1.0]


def _patch_retry_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(provider_retry.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(provider_retry.random, "random", lambda: 1.0)
    return sleeps


def _openai_errors() -> _ProviderErrors:
    def transient() -> Exception:
        return _openai_status_error(openai.RateLimitError, 429)

    def permanent() -> Exception:
        return _openai_status_error(openai.BadRequestError, 400)

    return _ProviderErrors(transient=transient, permanent=permanent)


def _openai_status_error(error_type: Any, status_code: int) -> Exception:
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.example.test/v1"),
    )
    return cast(
        Exception,
        error_type(
            f"provider failure contained private prompt {SECRET}",
            response=response,
            body={"error": SECRET},
        ),
    )


def _install_voyage_errors(monkeypatch: pytest.MonkeyPatch) -> _ProviderErrors:
    parent = ModuleType("voyageai")
    errors = ModuleType("voyageai.error")

    class APIError(Exception):
        def __init__(self, http_status: int) -> None:
            super().__init__(f"provider failure contained private prompt {SECRET}")
            self.http_status = http_status

    class APIConnectionError(APIError):
        pass

    class Timeout(APIError):
        pass

    class RateLimitError(APIError):
        pass

    class InvalidRequestError(APIError):
        pass

    setattr(errors, "APIError", APIError)
    setattr(errors, "APIConnectionError", APIConnectionError)
    setattr(errors, "Timeout", Timeout)
    setattr(errors, "RateLimitError", RateLimitError)
    setattr(errors, "InvalidRequestError", InvalidRequestError)
    setattr(parent, "error", errors)
    monkeypatch.setitem(sys.modules, "voyageai", parent)
    monkeypatch.setitem(sys.modules, "voyageai.error", errors)
    return _ProviderErrors(
        transient=lambda: RateLimitError(429),
        permanent=lambda: InvalidRequestError(400),
    )


def _install_zeroentropy_errors(monkeypatch: pytest.MonkeyPatch) -> _ProviderErrors:
    module = ModuleType("zeroentropy")

    class APIStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"provider failure contained private prompt {SECRET}")
            self.status_code = status_code

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class RateLimitError(APIStatusError):
        pass

    class BadRequestError(APIStatusError):
        pass

    setattr(module, "APIStatusError", APIStatusError)
    setattr(module, "APIConnectionError", APIConnectionError)
    setattr(module, "APITimeoutError", APITimeoutError)
    setattr(module, "RateLimitError", RateLimitError)
    setattr(module, "BadRequestError", BadRequestError)
    monkeypatch.setitem(sys.modules, "zeroentropy", module)
    return _ProviderErrors(
        transient=lambda: RateLimitError(429),
        permanent=lambda: BadRequestError(400),
    )


def _install_cohere_errors(monkeypatch: pytest.MonkeyPatch) -> _ProviderErrors:
    parent = ModuleType("cohere")
    errors = ModuleType("cohere.errors")

    class ApiError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"provider failure contained private prompt {SECRET}")
            self.status_code = status_code

    class TooManyRequestsError(ApiError):
        pass

    class InternalServerError(ApiError):
        pass

    class ServiceUnavailableError(ApiError):
        pass

    class GatewayTimeoutError(ApiError):
        pass

    class BadRequestError(ApiError):
        pass

    setattr(errors, "ApiError", ApiError)
    setattr(errors, "TooManyRequestsError", TooManyRequestsError)
    setattr(errors, "InternalServerError", InternalServerError)
    setattr(errors, "ServiceUnavailableError", ServiceUnavailableError)
    setattr(errors, "GatewayTimeoutError", GatewayTimeoutError)
    setattr(errors, "BadRequestError", BadRequestError)
    setattr(parent, "errors", errors)
    monkeypatch.setitem(sys.modules, "cohere", parent)
    monkeypatch.setitem(sys.modules, "cohere.errors", errors)
    return _ProviderErrors(
        transient=lambda: TooManyRequestsError(429),
        permanent=lambda: BadRequestError(400),
    )


@pytest.fixture(
    params=(
        _ProviderCase(
            id="openai_embedding",
            install_errors=lambda _monkeypatch: _openai_errors(),
            build_run=lambda outcomes: _build_openai_embedding_run(outcomes),
            success=SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
            ),
        ),
        _ProviderCase(
            id="voyage_embedding",
            install_errors=_install_voyage_errors,
            build_run=lambda outcomes: _build_voyage_embedding_run(outcomes),
            success=SimpleNamespace(embeddings=[[0.1, 0.2]]),
        ),
        _ProviderCase(
            id="voyage_rerank",
            install_errors=_install_voyage_errors,
            build_run=lambda outcomes: _build_voyage_rerank_run(outcomes),
            success=SimpleNamespace(
                results=[SimpleNamespace(index=0, relevance_score=0.9)]
            ),
        ),
        _ProviderCase(
            id="zeroentropy_embedding",
            install_errors=_install_zeroentropy_errors,
            build_run=lambda outcomes: _build_zeroentropy_embedding_run(outcomes),
            success=SimpleNamespace(
                results=[SimpleNamespace(embedding=[0.1, 0.2])]
            ),
        ),
        _ProviderCase(
            id="zeroentropy_rerank",
            install_errors=_install_zeroentropy_errors,
            build_run=lambda outcomes: _build_zeroentropy_rerank_run(outcomes),
            success=SimpleNamespace(
                results=[SimpleNamespace(index=0, relevance_score=0.9)]
            ),
        ),
        _ProviderCase(
            id="cohere_rerank",
            install_errors=_install_cohere_errors,
            build_run=lambda outcomes: _build_cohere_rerank_run(outcomes),
            success=SimpleNamespace(
                results=[SimpleNamespace(index=0, relevance_score=0.9)]
            ),
        ),
        _ProviderCase(
            id="cohere_embedding",
            install_errors=_install_cohere_errors,
            build_run=lambda outcomes: _build_cohere_embedding_run(outcomes),
            success=SimpleNamespace(
                embeddings=SimpleNamespace(float_=[[0.1, 0.2]])
            ),
        ),
    ),
    ids=lambda case: case.id,
)
def provider_case(request: pytest.FixtureRequest) -> _ProviderCase:
    return cast(_ProviderCase, request.param)


class _SequenceAsyncEmbeddings:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    async def create(self, **_kwargs: object) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)


class _SequenceVoyageEmbedClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def embed(self, _texts: list[str], **_kwargs: object) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)


class _SequenceVoyageRerankClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def rerank(
        self,
        _query: str,
        _documents: list[str],
        _model: str,
        _top_k: int,
    ) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)


class _SequenceZeroEntropyModels:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def embed(self, **_kwargs: object) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)

    def rerank(self, **_kwargs: object) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)


class _SequenceCohereClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    async def rerank(self, **_kwargs: object) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)

    async def embed(self, **_kwargs: object) -> object:
        self.calls += 1
        return _pop_outcome(self._outcomes)


def _build_openai_embedding_run(outcomes: list[object]) -> _CallSiteRun:
    api = _SequenceAsyncEmbeddings(outcomes)
    client = SimpleNamespace(embeddings=api)

    async def run() -> object:
        return await embed_openai_texts(
            client,
            model="text-embedding-3-small",
            dimensions=2,
            send_dimensions=True,
            texts=["private prompt text"],
        )

    return _CallSiteRun(run=run, call_count=lambda: api.calls)


def _build_voyage_embedding_run(outcomes: list[object]) -> _CallSiteRun:
    client = _SequenceVoyageEmbedClient(outcomes)
    provider = VoyageEmbeddingProvider.__new__(VoyageEmbeddingProvider)
    provider._client = client
    provider._model = "voyage-4"
    provider._dimensions = 2
    provider._send_dimensions = True

    async def run() -> object:
        return await provider.embed_texts(["private prompt text"])

    return _CallSiteRun(run=run, call_count=lambda: client.calls)


def _build_voyage_rerank_run(outcomes: list[object]) -> _CallSiteRun:
    client = _SequenceVoyageRerankClient(outcomes)
    reranker = VoyageReranker.__new__(VoyageReranker)
    reranker._client = client
    reranker._model = "rerank-2.5-lite"

    async def run() -> object:
        return await reranker.rerank(
            "private query",
            ["private document"],
            top_k=1,
        )

    return _CallSiteRun(run=run, call_count=lambda: client.calls)


def _build_zeroentropy_embedding_run(outcomes: list[object]) -> _CallSiteRun:
    models = _SequenceZeroEntropyModels(outcomes)
    provider = ZeroEntropyEmbeddingProvider.__new__(ZeroEntropyEmbeddingProvider)
    provider._client = SimpleNamespace(models=models)
    provider._model = "zembed-1"
    provider._dimensions = 2

    async def run() -> object:
        return await provider.embed_texts(["private prompt text"])

    return _CallSiteRun(run=run, call_count=lambda: models.calls)


def _build_zeroentropy_rerank_run(outcomes: list[object]) -> _CallSiteRun:
    models = _SequenceZeroEntropyModels(outcomes)
    reranker = ZeroEntropyReranker.__new__(ZeroEntropyReranker)
    reranker._client = SimpleNamespace(models=models)
    reranker._model = "zerank-2"

    async def run() -> object:
        return await reranker.rerank(
            "private query",
            ["private document"],
            top_k=1,
        )

    return _CallSiteRun(run=run, call_count=lambda: models.calls)


def _build_cohere_rerank_run(outcomes: list[object]) -> _CallSiteRun:
    client = _SequenceCohereClient(outcomes)
    reranker = CohereReranker.__new__(CohereReranker)
    reranker._client = client
    reranker._model = "rerank-v3.5"

    async def run() -> object:
        return await reranker.rerank(
            "private query",
            ["private document"],
            top_k=1,
        )

    return _CallSiteRun(run=run, call_count=lambda: client.calls)


def _build_cohere_embedding_run(outcomes: list[object]) -> _CallSiteRun:
    client = _SequenceCohereClient(outcomes)
    provider = CohereEmbeddingProvider.__new__(CohereEmbeddingProvider)
    provider._client = client
    provider._model = "embed-v4.0"
    provider._dimensions = 2

    async def run() -> object:
        return await provider.embed_texts(["private prompt text"])

    return _CallSiteRun(run=run, call_count=lambda: client.calls)


def _pop_outcome(outcomes: list[object]) -> object:
    if not outcomes:
        raise AssertionError("test provider outcomes exhausted")
    outcome = outcomes.pop(0)
    if isinstance(outcome, Exception):
        raise outcome
    return outcome
