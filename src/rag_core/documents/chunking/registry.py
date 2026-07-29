"""Typed registry for the ChunkingStrategy provider category.

Built-ins (``markdown``, ``semantic``, ``code``) register at import time of
this module's owning chunker modules; the registry imports those modules
lazily on first lookup so the public aggregator stays optional-dep-free.
"""

from __future__ import annotations

from typing import Any

from rag_core.search.providers.registry import ProviderRegistry

from .protocol import ChunkingStrategy

CHUNKING_STRATEGIES: ProviderRegistry[ChunkingStrategy] = ProviderRegistry(
    "chunking_strategy"
).with_builtins("rag_core.documents.chunking.builtins")


def create_chunking_strategy(name: str, **kwargs: Any) -> ChunkingStrategy:
    """Resolve the ChunkingStrategy provider category from a config name."""
    return CHUNKING_STRATEGIES.create(name, **kwargs)
