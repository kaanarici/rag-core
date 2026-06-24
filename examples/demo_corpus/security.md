# Security

Workspace data is scoped by collection, with namespace available for multi-tenant partitions. Applications should bind tenant scope before calling retrieval.
Models should provide the query and bounded retrieval options, not raw collection identifiers.
