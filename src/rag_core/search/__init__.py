from __future__ import annotations

from importlib import import_module

__all__ = (
    "And",
    "Boost",
    "Citation",
    "Context",
    "ContextSnippet",
    "DEFAULT_SEARCH_PROFILE",
    "DenseChannel",
    "Filter",
    "Geo",
    "In",
    "Mmr",
    "Not",
    "Or",
    "Prefetch",
    "PrefetchFusion",
    "PRIMARY_DENSE_QUERY_VECTOR",
    "QUERY_PLAN_PRESETS",
    "QueryPlan",
    "Range",
    "RerankBudget",
    "SEARCH_PROFILES",
    "SearchResult",
    "SparseChannel",
    "SparseVector",
    "SourceLocator",
    "SourcePreview",
    "Term",
    "UnsupportedQueryStage",
    "default_query_plan",
    "describe_query_plan",
    "describe_query_plan_presets",
    "describe_search_profile_catalog",
    "describe_search_profiles",
    "query_plan_preset",
    "search_profile",
)

# Curated public re-exports map each symbol to its OWNER module.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "And": ("rag_core.search.filters", "And"),
    "Boost": ("rag_core.search.query_plan", "Boost"),
    "Citation": ("rag_core.search.context_pack", "Citation"),
    "Context": ("rag_core.search.context_pack", "Context"),
    "ContextSnippet": ("rag_core.search.context_pack", "ContextSnippet"),
    "DEFAULT_SEARCH_PROFILE": ("rag_core.search.planning", "DEFAULT_SEARCH_PROFILE"),
    "DenseChannel": ("rag_core.search.query_plan", "DenseChannel"),
    "Filter": ("rag_core.search.filters", "Filter"),
    "Geo": ("rag_core.search.filters", "Geo"),
    "In": ("rag_core.search.filters", "In"),
    "Mmr": ("rag_core.search.query_plan", "Mmr"),
    "Not": ("rag_core.search.filters", "Not"),
    "Or": ("rag_core.search.filters", "Or"),
    "Prefetch": ("rag_core.search.query_plan", "Prefetch"),
    "PrefetchFusion": ("rag_core.search.query_plan", "PrefetchFusion"),
    "PRIMARY_DENSE_QUERY_VECTOR": (
        "rag_core.search.query_plan",
        "PRIMARY_DENSE_QUERY_VECTOR",
    ),
    "QUERY_PLAN_PRESETS": ("rag_core.search.planning", "QUERY_PLAN_PRESETS"),
    "QueryPlan": ("rag_core.search.query_plan", "QueryPlan"),
    "Range": ("rag_core.search.filters", "Range"),
    "RerankBudget": ("rag_core.search.request_models", "RerankBudget"),
    "SEARCH_PROFILES": ("rag_core.search.planning", "SEARCH_PROFILES"),
    "SearchResult": ("rag_core.search.vector_models", "SearchResult"),
    "SparseChannel": ("rag_core.search.query_plan", "SparseChannel"),
    "SparseVector": ("rag_core.search.vector_models", "SparseVector"),
    "SourceLocator": ("rag_core.search.context_pack", "SourceLocator"),
    "SourcePreview": ("rag_core.search.context_pack", "SourcePreview"),
    "Term": ("rag_core.search.filters", "Term"),
    "UnsupportedQueryStage": ("rag_core.search.query_plan", "UnsupportedQueryStage"),
    "default_query_plan": ("rag_core.search.planning", "default_query_plan"),
    "describe_query_plan": ("rag_core.search.planning", "describe_query_plan"),
    "describe_query_plan_presets": (
        "rag_core.search.planning",
        "describe_query_plan_presets",
    ),
    "describe_search_profile_catalog": (
        "rag_core.search.planning",
        "describe_search_profile_catalog",
    ),
    "describe_search_profiles": ("rag_core.search.planning", "describe_search_profiles"),
    "query_plan_preset": ("rag_core.search.planning", "query_plan_preset"),
    "search_profile": ("rag_core.search.planning", "search_profile"),
}

def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol = target
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return list(__all__)
