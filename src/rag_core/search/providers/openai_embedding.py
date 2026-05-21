"""OpenAI embedding SDK helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any
from typing import Callable
from typing import cast

from rag_core.search.providers.embedding_results import safe_indexed_embedding_vectors

_BATCH_SIZE = 100


def fingerprint_provider_config(**values: str | None) -> str:
    payload = {
        key: normalized
        for key, value in values.items()
        if (normalized := normalize_optional_provider_config_value(value))
    }
    if not payload:
        return ""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def import_async_openai() -> type[Any]:
    try:
        module = importlib.import_module("openai")
    except ImportError as exc:
        raise ImportError(
            "openai package is required for OpenAI embedding provider. Install it with: pip install openai"
        ) from exc
    client_class = getattr(module, "AsyncOpenAI", None)
    if client_class is None:
        raise ImportError("openai package with AsyncOpenAI is required for OpenAI embedding provider.")
    return cast(type[Any], client_class)


def build_openai_client(
    async_openai_loader: Callable[[], type[Any]],
    *,
    api_key: str | None,
    base_url: str | None,
) -> Any:
    client_kwargs: dict[str, str] = {}
    normalized_api_key = normalize_optional_provider_config_value(api_key)
    normalized_base_url = normalize_optional_provider_config_value(base_url)
    if normalized_api_key:
        client_kwargs["api_key"] = normalized_api_key
    if normalized_base_url:
        client_kwargs["base_url"] = normalized_base_url
    return async_openai_loader()(**client_kwargs)


def normalize_optional_provider_config_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


async def embed_openai_texts(
    client: Any,
    *,
    model: str,
    dimensions: int,
    send_dimensions: bool,
    texts: list[str],
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        request: dict[str, object] = {"model": model, "input": batch}
        if send_dimensions:
            request["dimensions"] = dimensions
        response = await client.embeddings.create(**request)
        all_embeddings.extend(
            safe_indexed_embedding_vectors(
                rows=[
                    (getattr(row, "index", None), getattr(row, "embedding", None))
                    for row in getattr(response, "data", []) or []
                ],
                expected_count=len(batch),
                expected_dimensions=dimensions,
                provider_name="OpenAIEmbeddingProvider",
            )
        )
    return all_embeddings
