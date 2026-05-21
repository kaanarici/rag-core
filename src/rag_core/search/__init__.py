from __future__ import annotations

from importlib import import_module

__all__ = (
    "And",
    "Boost",
    "ContextSnippet",
    "DEFAULT_SEARCH_PROFILE",
    "DenseChannel",
    "Filter",
    "Geo",
    "In",
    "Mmr",
    "MetadataFilterCapabilities",
    "ModelContextPack",
    "Not",
    "Or",
    "Prefetch",
    "PrefetchFusion",
    "QUERY_PLAN_PRESETS",
    "QueryPlan",
    "Range",
    "RerankBudget",
    "SEARCH_PROFILES",
    "SearchQuery",
    "SearchRequest",
    "SearchResult",
    "SparseChannel",
    "SparseVector",
    "SourceLocator",
    "SourcePreview",
    "SourceReference",
    "Term",
    "UnsupportedQueryStage",
    "default_query_plan",
    "describe_query_plan_presets",
    "describe_retrieval_profiles",
    "describe_search_profiles",
    "query_plan_preset",
    "search_profile",
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "And": ("rag_core.search.types", "And"),
    "Boost": ("rag_core.search.query_plan", "Boost"),
    "ContextSnippet": ("rag_core.search.context_pack", "ContextSnippet"),
    "DEFAULT_SEARCH_PROFILE": ("rag_core.search.planning", "DEFAULT_SEARCH_PROFILE"),
    "DenseChannel": ("rag_core.search.query_plan", "DenseChannel"),
    "Filter": ("rag_core.search.types", "Filter"),
    "Geo": ("rag_core.search.types", "Geo"),
    "In": ("rag_core.search.types", "In"),
    "Mmr": ("rag_core.search.query_plan", "Mmr"),
    "MetadataFilterCapabilities": (
        "rag_core.search.types",
        "MetadataFilterCapabilities",
    ),
    "ModelContextPack": ("rag_core.search.context_pack", "ModelContextPack"),
    "Not": ("rag_core.search.types", "Not"),
    "Or": ("rag_core.search.types", "Or"),
    "Prefetch": ("rag_core.search.query_plan", "Prefetch"),
    "PrefetchFusion": ("rag_core.search.query_plan", "PrefetchFusion"),
    "QUERY_PLAN_PRESETS": ("rag_core.search.planning", "QUERY_PLAN_PRESETS"),
    "QueryPlan": ("rag_core.search.query_plan", "QueryPlan"),
    "Range": ("rag_core.search.types", "Range"),
    "RerankBudget": ("rag_core.search.types", "RerankBudget"),
    "SEARCH_PROFILES": ("rag_core.search.planning", "SEARCH_PROFILES"),
    "SearchQuery": ("rag_core.search.types", "SearchQuery"),
    "SearchRequest": ("rag_core.search.searcher", "SearchRequest"),
    "SearchResult": ("rag_core.search.types", "SearchResult"),
    "SparseChannel": ("rag_core.search.query_plan", "SparseChannel"),
    "SparseVector": ("rag_core.search.types", "SparseVector"),
    "SourceLocator": ("rag_core.search.context_pack", "SourceLocator"),
    "SourcePreview": ("rag_core.search.context_pack", "SourcePreview"),
    "SourceReference": ("rag_core.search.context_pack", "SourceReference"),
    "Term": ("rag_core.search.types", "Term"),
    "UnsupportedQueryStage": ("rag_core.search.query_plan", "UnsupportedQueryStage"),
    "default_query_plan": ("rag_core.search.planning", "default_query_plan"),
    "describe_query_plan_presets": (
        "rag_core.search.planning",
        "describe_query_plan_presets",
    ),
    "describe_retrieval_profiles": (
        "rag_core.search.planning",
        "describe_retrieval_profiles",
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
