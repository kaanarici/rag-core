from __future__ import annotations

import argparse
import sys
from urllib.parse import parse_qsl, urlsplit

from rag_core.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_PGVECTOR_SCHEMA,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_VECTOR_STORE_PROVIDER,
    EMBEDDING_MODEL_ENV,
    EMBEDDING_PROVIDER_ENV,
    PROCESSING_VERSION_ENV,
    QDRANT_COLLECTION_ENV,
    QDRANT_LOCATION_ENV,
    QDRANT_URL_ENV,
    RERANKER_MODEL_ENV,
    RERANKER_PROVIDER_ENV,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    PGVECTOR_DSN_ENV,
    PGVECTOR_SCHEMA_ENV,
    PGVECTOR_TABLE_ENV,
)
from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.config.vector_store_config import (
    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
    TURBOPUFFER_BASE_URL_ENV,
    TURBOPUFFER_DISTANCE_METRIC_ENV,
    TURBOPUFFER_NAMESPACE_ENV,
    TURBOPUFFER_REGION_ENV,
    VECTOR_STORE_ENV,
)
from rag_core.config.env_access import (
    get_env,
    get_env_stripped,
)
from rag_core.provider_api_keys import QDRANT_API_KEY_ENVS, TURBOPUFFER_API_KEY_ENVS


class _WarnSensitiveFlagAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser
        setattr(namespace, self.dest, values)
        flag = option_string or f"--{self.dest.replace('_', '-')}"
        env_name = _SENSITIVE_FLAG_ENVS.get(flag)
        if env_name:
            print(
                f"warning: {flag} exposes credentials in shell history/process argv; prefer {env_name}",
                file=sys.stderr,
            )


class _WarnSensitiveUrlAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser
        setattr(namespace, self.dest, values)
        if isinstance(values, str) and _url_contains_inline_secret(values):
            flag = option_string or f"--{self.dest.replace('_', '-')}"
            print(
                f"warning: {flag} contains credentials in shell history/process argv; "
                f"prefer {QDRANT_URL_ENV} env var",
                file=sys.stderr,
            )


_SENSITIVE_FLAG_ENVS: dict[str, str] = {
    "--pgvector-dsn": f"{PGVECTOR_DSN_ENV} env var",
    "--qdrant-api-key": f"{QDRANT_API_KEY_ENVS[0]} env var",
    "--turbopuffer-api-key": f"{TURBOPUFFER_API_KEY_ENVS[0]} env var",
}
_SENSITIVE_URL_QUERY_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "key",
        "password",
        "secret",
        "token",
    }
)


def add_config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vector-store",
        choices=SUPPORTED_VECTOR_STORE_PROVIDERS,
        default=env_or_default(
            VECTOR_STORE_ENV,
            DEFAULT_VECTOR_STORE_PROVIDER,
        )
        .strip()
        .lower(),
        help=f"First-party vector store to assemble. Default: {DEFAULT_VECTOR_STORE_PROVIDER}.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=_env_or_none(QDRANT_URL_ENV),
        action=_WarnSensitiveUrlAction,
    )
    parser.add_argument(
        "--qdrant-location",
        default=_env_or_none(QDRANT_LOCATION_ENV),
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=_env_or_none(QDRANT_API_KEY_ENVS[0]),
        action=_WarnSensitiveFlagAction,
    )
    parser.add_argument(
        "--qdrant-collection",
        default=env_or_default(
            QDRANT_COLLECTION_ENV,
            DEFAULT_QDRANT_COLLECTION,
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        default=env_or_default(EMBEDDING_PROVIDER_ENV, DEFAULT_EMBEDDING_PROVIDER),
    )
    parser.add_argument(
        "--embedding-model",
        default=_env_or_none(EMBEDDING_MODEL_ENV),
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help=(
            "Dense document embedding batch size for indexing. "
            f"Default: {DEFAULT_EMBEDDING_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--reranker-provider",
        default=env_or_default(RERANKER_PROVIDER_ENV, DEFAULT_RERANKER_PROVIDER),
    )
    parser.add_argument("--reranker-model", default=_env_or_none(RERANKER_MODEL_ENV))
    parser.add_argument(
        "--processing-version",
        default=env_or_default(PROCESSING_VERSION_ENV, DEFAULT_PROCESSING_VERSION),
        help="Base processing version used for automatic reindex decisions.",
    )
    parser.add_argument(
        "--dimension-aware-collection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--pgvector-dsn",
        default=_env_or_none(PGVECTOR_DSN_ENV),
        action=_WarnSensitiveFlagAction,
        help="Postgres DSN for the pgvector vector store. Prefer the env var.",
    )
    parser.add_argument(
        "--pgvector-table",
        default=_env_or_none(PGVECTOR_TABLE_ENV),
        help="Physical pgvector table. Defaults to the resolved collection name.",
    )
    parser.add_argument(
        "--pgvector-schema",
        default=env_or_default(PGVECTOR_SCHEMA_ENV, DEFAULT_PGVECTOR_SCHEMA),
    )
    parser.add_argument(
        "--turbopuffer-namespace",
        default=_env_or_none(TURBOPUFFER_NAMESPACE_ENV),
        help="TurboPuffer namespace used as the physical vector-store collection.",
    )
    parser.add_argument(
        "--turbopuffer-api-key",
        default=_env_or_none(TURBOPUFFER_API_KEY_ENVS[0]),
        action=_WarnSensitiveFlagAction,
        help="TurboPuffer API key. Never printed by doctor output.",
    )
    parser.add_argument(
        "--turbopuffer-region",
        default=_env_or_none(TURBOPUFFER_REGION_ENV),
    )
    parser.add_argument(
        "--turbopuffer-base-url",
        default=_env_or_none(TURBOPUFFER_BASE_URL_ENV),
    )
    parser.add_argument(
        "--turbopuffer-distance-metric",
        default=env_or_default(
            TURBOPUFFER_DISTANCE_METRIC_ENV,
            DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
        ),
    )
    parser.add_argument(
        "--turbopuffer-delete-continuation-limit",
        type=int,
        default=None,
        help=(
            "Maximum partial delete-by-filter writes before TurboPuffer delete "
            "fails with continuation state. "
            f"Default: {DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT}."
        ),
    )


def env_or_default(name: str, default: str) -> str:
    return (get_env(name, default) or default).strip()


def _env_or_none(name: str) -> str | None:
    return get_env_stripped(name) or None


def _url_contains_inline_secret(raw_url: str) -> bool:
    parsed = urlsplit(raw_url)
    if parsed.username or parsed.password:
        return True
    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized = key.strip().lower()
        if normalized in _SENSITIVE_URL_QUERY_KEYS:
            return True
    return False


__all__ = ["add_config_flags", "env_or_default"]
