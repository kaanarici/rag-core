from __future__ import annotations

import argparse
import sys
from urllib.parse import parse_qsl, urlsplit

from rag_core.config import (
    DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
)
from rag_core.config.embedding_config import DEFAULT_EMBEDDING_BATCH_SIZE
from rag_core.config.env_access import (
    get_env,
    get_env_stripped,
)


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
                "prefer RAG_CORE_QDRANT_URL env var",
                file=sys.stderr,
            )


_SENSITIVE_FLAG_ENVS: dict[str, str] = {
    "--qdrant-api-key": "RAG_CORE_QDRANT_API_KEY env var",
    "--turbopuffer-api-key": "TURBOPUFFER_API_KEY env var",
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
        default=env_or_default("RAG_CORE_VECTOR_STORE", "qdrant").strip().lower(),
        help="First-party vector store to assemble. Default: qdrant.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=_env_or_none("RAG_CORE_QDRANT_URL"),
        action=_WarnSensitiveUrlAction,
    )
    parser.add_argument(
        "--qdrant-location",
        default=_env_or_none("RAG_CORE_QDRANT_LOCATION"),
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=_env_or_none("RAG_CORE_QDRANT_API_KEY"),
        action=_WarnSensitiveFlagAction,
    )
    parser.add_argument(
        "--qdrant-collection",
        default=env_or_default("RAG_CORE_QDRANT_COLLECTION", "rag_core_chunks"),
    )
    parser.add_argument(
        "--embedding-provider",
        default=env_or_default("RAG_CORE_EMBEDDING_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--embedding-model",
        default=env_or_default("RAG_CORE_EMBEDDING_MODEL", "text-embedding-3-large"),
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
        default=env_or_default("RAG_CORE_RERANKER_PROVIDER", "none"),
    )
    parser.add_argument(
        "--reranker-model", default=_env_or_none("RAG_CORE_RERANKER_MODEL")
    )
    parser.add_argument(
        "--processing-version",
        default=env_or_default("RAG_CORE_PROCESSING_VERSION", "rag_core_processing_v1"),
        help="Base processing version used for automatic reindex decisions.",
    )
    parser.add_argument(
        "--dimension-aware-collection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--turbopuffer-namespace",
        default=_env_or_none("RAG_CORE_TURBOPUFFER_NAMESPACE"),
        help="TurboPuffer namespace used as the physical vector-store collection.",
    )
    parser.add_argument(
        "--turbopuffer-api-key",
        default=_env_or_none("TURBOPUFFER_API_KEY"),
        action=_WarnSensitiveFlagAction,
        help="TurboPuffer API key. Never printed by doctor output.",
    )
    parser.add_argument(
        "--turbopuffer-region",
        default=_env_or_none("TURBOPUFFER_REGION"),
    )
    parser.add_argument(
        "--turbopuffer-base-url",
        default=_env_or_none("TURBOPUFFER_BASE_URL"),
    )
    parser.add_argument(
        "--turbopuffer-distance-metric",
        default=env_or_default(
            "RAG_CORE_TURBOPUFFER_DISTANCE_METRIC",
            "cosine_distance",
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
