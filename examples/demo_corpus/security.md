# Security

Workspace data is scoped by namespace and corpus. Applications should bind tenant scope before calling retrieval.
Models should provide the query and bounded retrieval options, not raw corpus identifiers.
