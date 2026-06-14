"""Shared public retrieval defaults for facade, CLI, and runtime entrypoints."""

from __future__ import annotations

DEFAULT_CONTEXT_LIMIT = 8
DEFAULT_LOCAL_SEARCH_LIMIT = 5
DEFAULT_RERANK = False
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_USE_LEXICAL_SEARCH = True

__all__ = [
    "DEFAULT_CONTEXT_LIMIT",
    "DEFAULT_LOCAL_SEARCH_LIMIT",
    "DEFAULT_RERANK",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_USE_LEXICAL_SEARCH",
]
