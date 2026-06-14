from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

import rag_core.search.providers.voyage as voyage_provider_module
import rag_core.search.providers.zeroentropy as zeroentropy_provider_module
from rag_core.search.providers.voyage import VoyageEmbeddingProvider
from rag_core.search.providers.zeroentropy import ZeroEntropyEmbeddingProvider

SECRET = "sk-test-secret"


class _DangerousVector:
    def __repr__(self) -> str:
        return f"repr leaked {SECRET}\nTraceback (most recent call last):"


DangerousTypeName = type(
    f"EmbeddingTypeNameLeak_{SECRET}_Traceback",
    (),
    {},
)


class _FakeVoyageClient:
    def __init__(self, embeddings: list[object]) -> None:
        self._embeddings = embeddings
        self.calls: list[dict[str, object]] = []

    def embed(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
        self.calls.append({"texts": list(texts), **kwargs})
        return SimpleNamespace(embeddings=self._embeddings)


class _FakeVoyageModule:
    def __init__(self, embeddings: list[object]) -> None:
        self._embeddings = embeddings
        self.clients: list[_FakeVoyageClient] = []

    def Client(self, api_key: str | None = None) -> _FakeVoyageClient:
        client = _FakeVoyageClient(self._embeddings)
        self.clients.append(client)
        return client


class _FakeZeroEntropyModels:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def embed(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(results=self._rows)


class _FakeZeroEntropyClient:
    def __init__(self, rows: list[object]) -> None:
        self.models = _FakeZeroEntropyModels(rows)


def _build_voyage_provider(
    rows: list[object], dimensions: int
) -> VoyageEmbeddingProvider:
    provider = VoyageEmbeddingProvider.__new__(VoyageEmbeddingProvider)
    provider._client = _FakeVoyageClient(rows)
    provider._model = "voyage-4"
    provider._dimensions = dimensions
    provider._send_dimensions = True
    return provider


def _build_initialized_voyage_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[object],
    dimensions: int,
    model: str,
) -> tuple[VoyageEmbeddingProvider, _FakeVoyageClient]:
    fake_module = _FakeVoyageModule(rows)
    monkeypatch.setattr(voyage_provider_module, "_import_voyageai", lambda: fake_module)
    provider = VoyageEmbeddingProvider(model=model, dimensions=dimensions)
    return provider, fake_module.clients[0]


def _build_zeroentropy_provider(
    rows: list[object],
    dimensions: int,
) -> ZeroEntropyEmbeddingProvider:
    provider = ZeroEntropyEmbeddingProvider.__new__(ZeroEntropyEmbeddingProvider)
    provider._client = _FakeZeroEntropyClient(rows)
    provider._model = "zembed-1"
    provider._dimensions = dimensions
    return provider


def _sanitize_assertions(message: str) -> None:
    assert SECRET not in message
    assert "Traceback" not in message
    assert "repr leaked" not in message
    assert "EmbeddingTypeNameLeak" not in message
    assert "_DangerousVector" not in message


@pytest.mark.parametrize(
    ("build_provider", "rows"),
    [
        (
            _build_voyage_provider,
            [[1, 2.5], [3.0, 4]],
        ),
        (
            _build_zeroentropy_provider,
            [
                SimpleNamespace(embedding=[1, 2.5]),
                SimpleNamespace(embedding=[3.0, 4]),
            ],
        ),
    ],
)
def test_embed_sync_preserves_order_and_casts_values_to_float(
    build_provider: Any,
    rows: list[object],
) -> None:
    provider = build_provider(rows, 2)

    vectors = provider._embed_sync(["first", "second"], "document")

    assert vectors == [[1.0, 2.5], [3.0, 4.0]]


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        ("voyage-finance-2", 1024),
        ("voyage-law-2", 1024),
        ("voyage-code-2", 1536),
    ],
)
def test_voyage_fixed_dimension_models_omit_output_dimension(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    dimensions: int,
) -> None:
    provider, client = _build_initialized_voyage_provider(
        monkeypatch,
        rows=[[0.0] * dimensions],
        dimensions=dimensions,
        model=model,
    )

    provider._embed_sync(["contract"], "document")

    assert client.calls == [
        {
            "texts": ["contract"],
            "model": model,
            "input_type": "document",
        }
    ]


@pytest.mark.parametrize(
    "model",
    [
        "voyage-4",
        "voyage-code-3",
        "voyage-3-large",
        "voyage-3.5",
        "voyage-3.5-lite",
    ],
)
def test_voyage_flexible_dimension_models_send_output_dimension(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    provider, client = _build_initialized_voyage_provider(
        monkeypatch,
        rows=[[0.0] * 512],
        dimensions=512,
        model=model,
    )

    provider._embed_sync(["contract"], "query")

    assert client.calls == [
        {
            "texts": ["contract"],
            "model": model,
            "input_type": "query",
            "output_dimension": 512,
        }
    ]


def test_voyage_unknown_models_omit_output_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, client = _build_initialized_voyage_provider(
        monkeypatch,
        rows=[[0.0] * 1024],
        dimensions=1024,
        model="custom-voyage-model",
    )

    provider._embed_sync(["contract"], "document")

    assert client.calls == [
        {
            "texts": ["contract"],
            "model": "custom-voyage-model",
            "input_type": "document",
        }
    ]


@pytest.mark.parametrize(
    ("provider_name", "build_provider", "rows", "texts", "dimensions", "message_part"),
    [
        (
            "VoyageEmbeddingProvider",
            _build_voyage_provider,
            [[0.1, 0.2]],
            ["one", "two"],
            2,
            "embedding count mismatch",
        ),
        (
            "ZeroEntropyEmbeddingProvider",
            _build_zeroentropy_provider,
            [SimpleNamespace(embedding=[0.1, 0.2])],
            ["one", "two"],
            2,
            "embedding count mismatch",
        ),
        (
            "VoyageEmbeddingProvider",
            _build_voyage_provider,
            [[0.1], [0.2, 0.3]],
            ["one", "two"],
            2,
            "embedding dimension mismatch",
        ),
        (
            "ZeroEntropyEmbeddingProvider",
            _build_zeroentropy_provider,
            [SimpleNamespace(embedding=[0.1]), SimpleNamespace(embedding=[0.2, 0.3])],
            ["one", "two"],
            2,
            "embedding dimension mismatch",
        ),
        (
            "VoyageEmbeddingProvider",
            _build_voyage_provider,
            [[DangerousTypeName(), 0.2]],
            ["one"],
            2,
            "invalid embedding value",
        ),
        (
            "ZeroEntropyEmbeddingProvider",
            _build_zeroentropy_provider,
            [SimpleNamespace(embedding=[DangerousTypeName(), 0.2])],
            ["one"],
            2,
            "invalid embedding value",
        ),
        (
            "VoyageEmbeddingProvider",
            _build_voyage_provider,
            [_DangerousVector()],
            ["one"],
            2,
            "invalid embedding vector",
        ),
        (
            "ZeroEntropyEmbeddingProvider",
            _build_zeroentropy_provider,
            [SimpleNamespace(embedding=_DangerousVector())],
            ["one"],
            2,
            "invalid embedding vector",
        ),
    ],
)
def test_embed_sync_sanitizes_validation_errors(
    provider_name: str,
    build_provider: Any,
    rows: list[object],
    texts: list[str],
    dimensions: int,
    message_part: str,
) -> None:
    provider = build_provider(rows, dimensions)

    with pytest.raises(ValueError) as exc_info:
        provider._embed_sync(texts, "document")

    message = str(exc_info.value)
    assert provider_name in message
    assert message_part in message
    _sanitize_assertions(message)


def test_voyage_import_helper_rejects_incompatible_sdk_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "voyageai", SimpleNamespace())

    with pytest.raises(ImportError) as exc_info:
        voyage_provider_module._import_voyageai()

    message = str(exc_info.value)
    assert "voyageai.Client" in message
    assert "rag-core[voyage]" in message


def test_zeroentropy_import_helper_rejects_incompatible_sdk_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "zeroentropy", SimpleNamespace())

    with pytest.raises(ImportError) as exc_info:
        zeroentropy_provider_module._import_zeroentropy()

    message = str(exc_info.value)
    assert "zeroentropy.ZeroEntropy" in message
    assert "rag-core[zeroentropy]" in message
