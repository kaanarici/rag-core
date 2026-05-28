from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rag_core.config.vector_store_config import (
    QDRANT_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_VECTOR_STORE_PROVIDER,
)
from rag_core.retrieval_channels import DENSE_RETRIEVAL_CHANNEL, SPARSE_RETRIEVAL_CHANNEL
from rag_core.search.providers.diagnostic_support import (
    SUPPORT_DEFAULT,
    SUPPORT_FIRST_PARTY_OPTIONAL,
    SUPPORT_FIRST_PARTY_UTILITY,
    ProviderDiagnosticSupportLevel,
)
from rag_core.search.provider_protocols import (
    MetadataFilterCapabilities,
    QueryPlanCapabilities,
    StoreCapabilities,
)
from rag_core.search.sparse_channels import KNOWN_SPARSE_CHANNELS

MEMORY_VECTOR_STORE_PROVIDER = "memory"
QUERY_PLAN_CAPABILITY_DENSE: Final[str] = DENSE_RETRIEVAL_CHANNEL
QUERY_PLAN_CAPABILITY_SPARSE: Final[str] = SPARSE_RETRIEVAL_CHANNEL
QUERY_PLAN_CAPABILITY_HYBRID: Final[str] = "hybrid"
QUERY_PLAN_CAPABILITY_HYBRID_RRF: Final[str] = "hybrid_rrf"
QUERY_PLAN_CAPABILITY_HYBRID_DBSF: Final[str] = "hybrid_dbsf"
QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF: Final[str] = "hybrid_weighted_rrf"
QUERY_PLAN_CAPABILITY_MMR: Final[str] = "mmr"
QUERY_PLAN_CAPABILITY_NESTED_PREFETCH: Final[str] = "nested_prefetch"
QUERY_PLAN_CAPABILITY_BOOST: Final[str] = "boost"
QUERY_PLAN_CAPABILITY_FIELDS = (
    QUERY_PLAN_CAPABILITY_DENSE,
    QUERY_PLAN_CAPABILITY_SPARSE,
    QUERY_PLAN_CAPABILITY_HYBRID,
    QUERY_PLAN_CAPABILITY_HYBRID_RRF,
    QUERY_PLAN_CAPABILITY_HYBRID_DBSF,
    QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF,
    QUERY_PLAN_CAPABILITY_MMR,
    QUERY_PLAN_CAPABILITY_NESTED_PREFETCH,
    QUERY_PLAN_CAPABILITY_BOOST,
)
QUERY_PLAN_STAGE_CAPABILITY_FIELDS = tuple(
    field
    for field in QUERY_PLAN_CAPABILITY_FIELDS
    if field != QUERY_PLAN_CAPABILITY_HYBRID
)

METADATA_FILTER_CAPABILITY_TERM: Final[str] = "term"
METADATA_FILTER_CAPABILITY_IN: Final[str] = "in"
METADATA_FILTER_CAPABILITY_NUMERIC_RANGE: Final[str] = "numeric_range"
METADATA_FILTER_CAPABILITY_STRING_RANGE: Final[str] = "string_range"
METADATA_FILTER_CAPABILITY_GEO: Final[str] = "geo"
METADATA_FILTER_CAPABILITY_BOOLEAN: Final[str] = "boolean"
METADATA_FILTER_CAPABILITY_FIELDS = (
    METADATA_FILTER_CAPABILITY_TERM,
    METADATA_FILTER_CAPABILITY_IN,
    METADATA_FILTER_CAPABILITY_NUMERIC_RANGE,
    METADATA_FILTER_CAPABILITY_STRING_RANGE,
    METADATA_FILTER_CAPABILITY_GEO,
    METADATA_FILTER_CAPABILITY_BOOLEAN,
)


@dataclass(frozen=True)
class VectorStoreCapabilitySpec:
    query_plan: QueryPlanCapabilities
    metadata_filter: MetadataFilterCapabilities
    per_point_delete: bool = True
    document_record_lookup: bool = True

    def to_store_capabilities(
        self,
        *,
        dense_vector_dimensions: int | None = None,
        query_plan: QueryPlanCapabilities | None = None,
    ) -> StoreCapabilities:
        return StoreCapabilities(
            per_point_delete=self.per_point_delete,
            document_record_lookup=self.document_record_lookup,
            dense_vector_dimensions=dense_vector_dimensions,
            query_plan=query_plan or self.query_plan,
            metadata_filter=self.metadata_filter,
        )


@dataclass(frozen=True)
class VectorStoreProviderSpec:
    name: str
    docs_label: str
    docs_maturity: str
    docs_entrypoint: str
    diagnostic_support_level: ProviderDiagnosticSupportLevel
    capabilities: VectorStoreCapabilitySpec


def describe_query_plan_capabilities(
    capabilities: QueryPlanCapabilities,
) -> dict[str, bool]:
    values = {
        QUERY_PLAN_CAPABILITY_DENSE: capabilities.dense,
        QUERY_PLAN_CAPABILITY_SPARSE: capabilities.sparse,
        QUERY_PLAN_CAPABILITY_HYBRID: capabilities.hybrid,
        QUERY_PLAN_CAPABILITY_HYBRID_RRF: capabilities.hybrid_rrf,
        QUERY_PLAN_CAPABILITY_HYBRID_DBSF: capabilities.hybrid_dbsf,
        QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF: capabilities.hybrid_weighted_rrf,
        QUERY_PLAN_CAPABILITY_MMR: capabilities.mmr,
        QUERY_PLAN_CAPABILITY_NESTED_PREFETCH: capabilities.nested_prefetch,
        QUERY_PLAN_CAPABILITY_BOOST: capabilities.boost,
    }
    return {field: values[field] for field in QUERY_PLAN_CAPABILITY_FIELDS}


def describe_metadata_filter_capabilities(
    capabilities: MetadataFilterCapabilities,
) -> dict[str, bool]:
    values = {
        METADATA_FILTER_CAPABILITY_TERM: capabilities.term,
        METADATA_FILTER_CAPABILITY_IN: capabilities.in_,
        METADATA_FILTER_CAPABILITY_NUMERIC_RANGE: capabilities.numeric_range,
        METADATA_FILTER_CAPABILITY_STRING_RANGE: capabilities.string_range,
        METADATA_FILTER_CAPABILITY_GEO: capabilities.geo,
        METADATA_FILTER_CAPABILITY_BOOLEAN: capabilities.boolean,
    }
    return {field: values[field] for field in METADATA_FILTER_CAPABILITY_FIELDS}


MEMORY_VECTOR_STORE_CAPABILITY_SPEC = VectorStoreCapabilitySpec(
    query_plan=QueryPlanCapabilities(
        dense=True,
        sparse=True,
        hybrid_rrf=True,
    ),
    metadata_filter=MetadataFilterCapabilities(
        term=True,
        in_=True,
        numeric_range=True,
        string_range=True,
        geo=True,
        boolean=True,
    ),
)

QDRANT_VECTOR_STORE_CAPABILITY_SPEC = VectorStoreCapabilitySpec(
    query_plan=QueryPlanCapabilities(
        dense=True,
        sparse=True,
        hybrid_rrf=True,
        hybrid_dbsf=True,
        hybrid_weighted_rrf=True,
        mmr=True,
        boost=True,
        nested_prefetch=True,
    ),
    metadata_filter=MetadataFilterCapabilities(
        term=True,
        in_=True,
        numeric_range=True,
        string_range=False,
        geo=True,
        boolean=True,
    ),
)


def qdrant_query_plan_capabilities_for_sparse_names(
    sparse_names: set[str] | frozenset[str] | None,
) -> QueryPlanCapabilities:
    known_sparse_names = (
        sparse_names & KNOWN_SPARSE_CHANNELS if sparse_names is not None else None
    )
    if not known_sparse_names:
        return QueryPlanCapabilities(dense=True)
    return QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan


TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC = VectorStoreCapabilitySpec(
    query_plan=QueryPlanCapabilities(
        dense=True,
        sparse=True,
        hybrid_rrf=True,
    ),
    metadata_filter=MetadataFilterCapabilities(
        term=True,
        in_=True,
        numeric_range=True,
        string_range=True,
        geo=False,
        boolean=True,
    ),
)


QDRANT_VECTOR_STORE_PROVIDER_SPEC = VectorStoreProviderSpec(
    name=QDRANT_VECTOR_STORE_PROVIDER,
    docs_label="Qdrant",
    docs_maturity="first-party default",
    docs_entrypoint=(
        "`QdrantConfig`, `--qdrant-*`, "
        "`rag_core.search.providers.QdrantVectorStore`"
    ),
    diagnostic_support_level=SUPPORT_DEFAULT,
    capabilities=QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
)
TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC = VectorStoreProviderSpec(
    name=TURBOPUFFER_VECTOR_STORE_PROVIDER,
    docs_label="TurboPuffer",
    docs_maturity="first-party optional",
    docs_entrypoint=(
        "`--vector-store turbopuffer`, `uv sync --extra turbopuffer`, "
        "`rag_core.search.providers.turbopuffer_store.TurboPufferVectorStore`"
    ),
    diagnostic_support_level=SUPPORT_FIRST_PARTY_OPTIONAL,
    capabilities=TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC,
)
MEMORY_VECTOR_STORE_PROVIDER_SPEC = VectorStoreProviderSpec(
    name=MEMORY_VECTOR_STORE_PROVIDER,
    docs_label="In-memory",
    docs_maturity="utility",
    docs_entrypoint=(
        "`RAGCore(vector_store=...)` with "
        "`rag_core.search.providers.memory_store.InMemoryVectorStore`"
    ),
    diagnostic_support_level=SUPPORT_FIRST_PARTY_UTILITY,
    capabilities=MEMORY_VECTOR_STORE_CAPABILITY_SPEC,
)
BUILTIN_VECTOR_STORE_PROVIDER_SPECS = (
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
    TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC,
    MEMORY_VECTOR_STORE_PROVIDER_SPEC,
)
BUILTIN_VECTOR_STORE_PROVIDER_ORDER = tuple(
    spec.name for spec in BUILTIN_VECTOR_STORE_PROVIDER_SPECS
)
VECTOR_STORE_PROVIDER_SPECS_BY_NAME = {
    spec.name: spec for spec in BUILTIN_VECTOR_STORE_PROVIDER_SPECS
}


__all__ = [
    "BUILTIN_VECTOR_STORE_PROVIDER_SPECS",
    "BUILTIN_VECTOR_STORE_PROVIDER_ORDER",
    "METADATA_FILTER_CAPABILITY_BOOLEAN",
    "METADATA_FILTER_CAPABILITY_FIELDS",
    "METADATA_FILTER_CAPABILITY_GEO",
    "METADATA_FILTER_CAPABILITY_IN",
    "METADATA_FILTER_CAPABILITY_NUMERIC_RANGE",
    "METADATA_FILTER_CAPABILITY_STRING_RANGE",
    "METADATA_FILTER_CAPABILITY_TERM",
    "MEMORY_VECTOR_STORE_CAPABILITY_SPEC",
    "MEMORY_VECTOR_STORE_PROVIDER",
    "MEMORY_VECTOR_STORE_PROVIDER_SPEC",
    "QUERY_PLAN_CAPABILITY_BOOST",
    "QUERY_PLAN_CAPABILITY_DENSE",
    "QUERY_PLAN_CAPABILITY_FIELDS",
    "QUERY_PLAN_CAPABILITY_HYBRID",
    "QUERY_PLAN_CAPABILITY_HYBRID_DBSF",
    "QUERY_PLAN_CAPABILITY_HYBRID_RRF",
    "QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF",
    "QUERY_PLAN_CAPABILITY_MMR",
    "QUERY_PLAN_CAPABILITY_NESTED_PREFETCH",
    "QUERY_PLAN_CAPABILITY_SPARSE",
    "QUERY_PLAN_STAGE_CAPABILITY_FIELDS",
    "QDRANT_VECTOR_STORE_CAPABILITY_SPEC",
    "QDRANT_VECTOR_STORE_PROVIDER_SPEC",
    "TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC",
    "TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC",
    "VECTOR_STORE_PROVIDER_SPECS_BY_NAME",
    "VectorStoreCapabilitySpec",
    "VectorStoreProviderSpec",
    "describe_metadata_filter_capabilities",
    "describe_query_plan_capabilities",
    "qdrant_query_plan_capabilities_for_sparse_names",
]
