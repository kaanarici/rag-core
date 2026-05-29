from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_VECTOR_STORE_PROVIDER,
    DEMO_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE_ENV,
    EMBEDDING_DIMENSIONS_ENV,
    EmbeddingConfig,
    IngestConfig,
    QDRANT_DIMENSION_AWARE_COLLECTION_ENV,
    QdrantConfig,
    RerankerConfig,
    TurboPufferVectorStoreConfig,
    VectorStoreConfig,
)
from rag_core.config.vector_store_config import (
    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
    TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV,
)
from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.config.env_access import get_env, parse_env_bool
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION

if TYPE_CHECKING:
    from rag_core.core_models import RAGCoreConfig


def build_rag_core_config_from_cli_args(
    config_type: type["RAGCoreConfig"],
    args: argparse.Namespace,
    *,
    manifest_dir: Path | None = None,
) -> "RAGCoreConfig":
    embedding_provider = (
        _arg(args, "embedding_provider", default=DEFAULT_EMBEDDING_PROVIDER)
        or DEFAULT_EMBEDDING_PROVIDER
    )
    embedding_model = _embedding_model(args, provider=embedding_provider)
    vector_store_provider = (
        _arg(args, "vector_store", default=DEFAULT_VECTOR_STORE_PROVIDER)
        or DEFAULT_VECTOR_STORE_PROVIDER
    )
    return config_type(
        qdrant=QdrantConfig(
            url=_arg(args, "qdrant_url"),
            location=_arg(args, "qdrant_location"),
            api_key=_arg(args, "qdrant_api_key", default="") or "",
            collection=_arg(args, "qdrant_collection", default=DEFAULT_QDRANT_COLLECTION)
            or DEFAULT_QDRANT_COLLECTION,
            dimension_aware_collection=_env_backed_bool_arg(
                args,
                "dimension_aware_collection",
                env_name=QDRANT_DIMENSION_AWARE_COLLECTION_ENV,
                default=DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
            ),
        ),
        vector_store=VectorStoreConfig(
            provider=vector_store_provider,
            turbopuffer=TurboPufferVectorStoreConfig(
                namespace=_arg(args, "turbopuffer_namespace"),
                api_key=_arg(args, "turbopuffer_api_key"),
                region=_arg(args, "turbopuffer_region"),
                base_url=_arg(args, "turbopuffer_base_url"),
                distance_metric=_arg(
                    args,
                    "turbopuffer_distance_metric",
                    default=DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
                )
                or DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
                delete_continuation_limit=_arg(
                    args,
                    "turbopuffer_delete_continuation_limit",
                )
                if _arg(args, "turbopuffer_delete_continuation_limit") is not None
                else _env_int(
                    TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV,
                    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
                ),
            ),
        ),
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            dimensions=_arg(args, "embedding_dimensions")
            if _arg(args, "embedding_dimensions") is not None
            else _env_optional_int(EMBEDDING_DIMENSIONS_ENV),
            batch_size=_arg(
                args,
                "embedding_batch_size",
            )
            if _arg(args, "embedding_batch_size") is not None
            else _env_int(
                EMBEDDING_BATCH_SIZE_ENV,
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
        ),
        reranker=RerankerConfig(
            provider=_arg(args, "reranker_provider", default=DEFAULT_RERANKER_PROVIDER)
            or DEFAULT_RERANKER_PROVIDER,
            model=_arg(args, "reranker_model"),
        ),
        ingest=IngestConfig(
            processing_version=_arg(
                args,
                "processing_version",
                default=DEFAULT_PROCESSING_VERSION,
            )
            or DEFAULT_PROCESSING_VERSION,
            manifest_directory=manifest_dir,
        ),
    )


def _arg(args: argparse.Namespace, name: str, *, default: Any = None) -> Any:
    return getattr(args, name, default)


def _embedding_model(args: argparse.Namespace, *, provider: str) -> str:
    model = _arg(args, "embedding_model")
    if isinstance(model, str) and model:
        return model
    if provider.strip().lower() == "demo":
        return DEMO_EMBEDDING_MODEL
    return DEFAULT_EMBEDDING_MODEL


def _env_optional_int(name: str) -> int | None:
    raw = get_env(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_int(name: str, default: int) -> int:
    value = _env_optional_int(name)
    return default if value is None else value


def _env_backed_bool_arg(
    args: argparse.Namespace,
    attr: str,
    *,
    env_name: str,
    default: bool,
) -> bool:
    value = _arg(args, attr)
    if value is not None:
        return bool(value)
    raw = get_env(env_name)
    if raw is None:
        return default
    parsed = parse_env_bool(raw)
    if parsed is None:
        raise ValueError(f"{env_name} must be a boolean")
    return parsed


def with_ingest_source_type(
    config: "RAGCoreConfig",
    *,
    source_type: str,
) -> "RAGCoreConfig":
    return replace(
        config,
        ingest=replace(config.ingest, source_type=source_type),
    )


__all__ = ["build_rag_core_config_from_cli_args", "with_ingest_source_type"]
