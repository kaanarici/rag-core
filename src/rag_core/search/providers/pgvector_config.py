from __future__ import annotations

import re
from dataclasses import dataclass

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

_POSTGRES_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


@dataclass(frozen=True)
class PgVectorConfig:
    dsn: str | None
    table_name: str
    schema: str
    dense_dimensions: int
    policy: VectorStorePolicy = DEFAULT_POLICY

    @property
    def qualified_table(self) -> str:
        return f"{quote_identifier(self.schema)}.{quote_identifier(self.table_name)}"

    @property
    def dense_index_name(self) -> str:
        base = f"{self.table_name}_dense_hnsw_idx"
        if len(base) <= 63:
            return base
        return f"{self.table_name[:48]}_dense_hnsw_idx"


def build_pgvector_config(
    *,
    dsn: str | None,
    table_name: str,
    schema: str,
    dense_dimensions: int,
    policy: VectorStorePolicy,
) -> PgVectorConfig:
    if dsn is not None and not dsn.strip():
        raise ValueError("PgVectorVectorStore dsn must be a non-empty string")
    if (
        isinstance(dense_dimensions, bool)
        or not isinstance(dense_dimensions, int)
        or dense_dimensions <= 0
    ):
        raise ValueError("PgVectorVectorStore dense_dimensions must be positive")
    return PgVectorConfig(
        dsn=dsn.strip() if dsn is not None else None,
        table_name=validate_pgvector_identifier(
            table_name,
            "PgVectorVectorStore table_name",
        ),
        schema=validate_pgvector_identifier(schema, "PgVectorVectorStore schema"),
        dense_dimensions=dense_dimensions,
        policy=policy,
    )


def validate_pgvector_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty PostgreSQL identifier")
    identifier = value.strip()
    if not _POSTGRES_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(
            f"{field_name} must match [A-Za-z_][A-Za-z0-9_]{{0,62}}"
        )
    return identifier


def quote_identifier(identifier: str) -> str:
    return f'"{validate_pgvector_identifier(identifier, "PostgreSQL identifier")}"'
