"""Qdrant client construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient


@dataclass(frozen=True)
class QdrantClientState:
    client: AsyncQdrantClient
    is_local: bool


def create_qdrant_client(
    *,
    url: str | None,
    api_key: str | None,
    location: str | None,
    timeout: int = 120,
) -> QdrantClientState:
    api_key = _normalize_api_key(api_key)
    if bool(url) == bool(location):
        raise ValueError(
            "QdrantVectorStore requires exactly one of url or location; "
            "use QdrantConfig(location=':memory:'), pass url=..., "
            "or inject vector_store=... into RAGCore."
        )

    if location is not None:
        if location != ":memory:":
            return QdrantClientState(
                client=AsyncQdrantClient(
                    path=location,
                    timeout=timeout,
                    check_compatibility=False,
                ),
                is_local=True,
            )
        return QdrantClientState(
            client=AsyncQdrantClient(
                location=location,
                timeout=timeout,
                check_compatibility=False,
            ),
            is_local=True,
        )

    if api_key is not None:
        return QdrantClientState(
            client=AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout),
            is_local=False,
        )

    return QdrantClientState(
        client=AsyncQdrantClient(url=url, timeout=timeout),
        is_local=False,
    )


def _normalize_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    stripped = api_key.strip()
    return stripped or None
