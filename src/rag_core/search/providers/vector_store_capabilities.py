from rag_core.search.provider_protocols import MetadataFilterCapabilities

QDRANT_METADATA_FILTERS = MetadataFilterCapabilities(
    term=True, in_=True, numeric_range=True, geo=True, boolean=True
)
