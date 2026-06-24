from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import TYPE_CHECKING, TypedDict

from rag_core.config import (
    DEFAULT_RERANKER_PROVIDER,
    EmbeddingConfig,
    QdrantConfig,
    RerankerConfig,
)
from rag_core.core import Engine, Config
from rag_core.core_models import IngestedDocument
from rag_core.search.provider_protocols import ProviderHealth
from rag_core.search.providers.provider_health import (
    PROVIDER_HEALTH_KIND_EMBEDDING,
    build_healthy_provider_health,
)
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL
from rag_core.search.vector_models import SparseVector

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

_DEMO_EMBEDDING_DIMENSIONS = 64
_DEMO_COLLECTION_PREFIX = "rag_core_examples"
_DEMO_BYTES = b"Billing is due monthly and invoices can be paid by card or ACH."
_DEMO_FILENAME = "billing.txt"
_DEMO_MIME_TYPE = "text/plain"
_DEMO_NAMESPACE = "acme"
_DEMO_COLLECTION = "help-center"
_DEMO_QUERY = "How can I pay invoices?"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DemoHit(TypedDict):
    score: float
    title: str
    text: str


class DemoPayload(TypedDict):
    document_id: str
    chunk_count: int
    hits: list[DemoHit]


class DemoEmbeddingProvider:
    def __init__(self, *, dimensions: int = _DEMO_EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = max(1, dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "demo-dense-v1"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_dense_vector(text, dimensions=self._dimensions) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return _dense_vector(query, dimensions=self._dimensions)

    async def check_health(self) -> ProviderHealth:
        return build_healthy_provider_health(
            provider_name="demo",
            kind=PROVIDER_HEALTH_KIND_EMBEDDING,
            model_name=self.model_name,
            dimensions=self._dimensions,
        )


class DemoSparseEmbedder:
    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        return [_sparse_vector(text) for text in texts]

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        return [{PRIMARY_SPARSE_CHANNEL: vector} for vector in self.embed_texts(texts)]

    def embed_query(self, query: str) -> SparseVector:
        return _sparse_vector(query)

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        return {PRIMARY_SPARSE_CHANNEL: self.embed_query(query)}


def build_demo_core(
    *,
    store_collection: str,
    qdrant_location: str = ":memory:",
    event_sink: EventSink | None = None,
) -> Engine:
    return Engine(
        Config(
            qdrant=QdrantConfig(
                location=qdrant_location,
                store_collection=f"{_DEMO_COLLECTION_PREFIX}_{store_collection}",
                dimension_aware_collection=False,
            ),
            embedding=EmbeddingConfig(
                provider="demo",
                model="demo-dense-v1",
                dimensions=_DEMO_EMBEDDING_DIMENSIONS,
            ),
            reranker=RerankerConfig(provider=DEFAULT_RERANKER_PROVIDER),
        ),
        embedding_provider=DemoEmbeddingProvider(),
        sparse_embedder=DemoSparseEmbedder(),
        event_sink=event_sink,
    )


async def ingest_demo_billing_document(core: Engine) -> IngestedDocument:
    return await core.add_bytes(
        file_bytes=_DEMO_BYTES,
        filename=_DEMO_FILENAME,
        mime_type=_DEMO_MIME_TYPE,
        namespace=_DEMO_NAMESPACE,
        collection=_DEMO_COLLECTION,
    )


async def run_demo_app() -> DemoPayload:
    core = build_demo_core(store_collection="minimal_app")
    try:
        await core.ensure_ready()
        ingested = await ingest_demo_billing_document(core)
        hits = await core.search(
            query=_DEMO_QUERY,
            namespace=_DEMO_NAMESPACE,
            collections=[_DEMO_COLLECTION],
            limit=3,
            rerank=False,
        )
        return {
            "document_id": ingested.document_id,
            "chunk_count": ingested.chunk_count,
            "hits": [
                {
                    "score": hit.score,
                    "title": hit.title or hit.document_id or "unknown",
                    "text": hit.text,
                }
                for hit in hits
            ],
        }
    finally:
        await core.close()


def _dense_vector(text: str, *, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    for token in _tokens(text):
        values[zlib.adler32(token.encode("utf-8")) % dimensions] += 1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _sparse_vector(text: str) -> SparseVector:
    terms = Counter(_tokens(text))
    if not terms:
        return SparseVector(indices=[0], values=[0.0])
    merged: dict[int, float] = {}
    for term, count in terms.items():
        index = zlib.adler32(term.encode("utf-8")) % 100_000
        merged[index] = merged.get(index, 0.0) + float(count)
    sorted_items = sorted(merged.items())
    return SparseVector(
        indices=[index for index, _ in sorted_items],
        values=[value for _, value in sorted_items],
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
