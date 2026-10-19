"""One environment decoder for the Python facade and CLI overrides."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from collections.abc import Mapping

from rag_core.config import EmbeddingConfig, QdrantConfig, VectorStoreConfig
from rag_core.config.embedding_config import default_embedding_model
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.core_models import Config


def env_bool(env: Mapping[str, str], name: str, fallback: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return fallback
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def build_config_from_env(
    *, environ: Mapping[str, str] | None = None, defaults: Config | None = None
) -> Config:
    env = os.environ if environ is None else environ
    base = defaults or Config()

    def text(name: str, fallback: str | None = None) -> str | None:
        raw = env.get(name)
        return raw.strip() or None if raw is not None else fallback

    def integer(name: str, fallback: int | None) -> int | None:
        raw = env.get(name)
        if raw is None:
            return fallback
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None

    if text("RAG_CORE_VECTOR_STORE", "qdrant") != "qdrant":
        raise ValueError("Only Qdrant is built in; inject a VectorStore into Engine")
    provider = (
        text("RAG_CORE_EMBEDDING_PROVIDER", base.embedding.provider)
        or base.embedding.provider
    )
    model = text("RAG_CORE_EMBEDDING_MODEL") or (
        base.embedding.model
        if provider == base.embedding.provider
        else default_embedding_model(provider)
    )
    if provider not in {"local", "openai", "demo"}:
        raise ValueError(
            "Use embedding provider local, openai or demo, or inject one into Engine"
        )
    url, location = text("RAG_CORE_QDRANT_URL"), text("RAG_CORE_QDRANT_LOCATION")
    if url is None and location is None and defaults is not None:
        url, location = base.vector_store.qdrant.url, base.vector_store.qdrant.location
    if bool(url) == bool(location):
        raise ValueError(
            "Qdrant requires exactly one of RAG_CORE_QDRANT_URL or RAG_CORE_QDRANT_LOCATION"
        )
    qdrant = base.vector_store.qdrant
    batch_size = integer("RAG_CORE_EMBEDDING_BATCH_SIZE", base.embedding.batch_size)
    assert batch_size is not None
    return replace(
        base,
        hybrid=env_bool(env, "RAG_CORE_HYBRID", base.hybrid),
        vector_store=VectorStoreConfig(
            qdrant=QdrantConfig(
                url=url,
                location=location,
                api_key=text("RAG_CORE_QDRANT_API_KEY")
                or text("QDRANT_API_KEY")
                or qdrant.api_key,
                store_collection=text(
                    "RAG_CORE_QDRANT_COLLECTION", qdrant.store_collection
                )
                or qdrant.store_collection,
                dimension_aware_collection=env_bool(
                    env,
                    "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION",
                    qdrant.dimension_aware_collection,
                ),
            )
        ),
        embedding=EmbeddingConfig(
            provider=provider,
            model=model,
            dimensions=integer(
                "RAG_CORE_EMBEDDING_DIMENSIONS", base.embedding.dimensions
            ),
            api_key=text("RAG_CORE_EMBEDDING_API_KEY")
            or text("OPENAI_API_KEY")
            or base.embedding.api_key,
            base_url=text("RAG_CORE_EMBEDDING_BASE_URL", base.embedding.base_url),
            batch_size=batch_size,
        ),
        ingest=replace(
            base.ingest,
            processing_version=text(
                "RAG_CORE_PROCESSING_VERSION", DEFAULT_PROCESSING_VERSION
            )
            or DEFAULT_PROCESSING_VERSION,
            manifest_directory=Path(manifest)
            if (manifest := text("RAG_CORE_MANIFEST_DIR"))
            else base.ingest.manifest_directory,
        ),
    )
