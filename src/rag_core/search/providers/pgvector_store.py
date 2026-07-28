"""pgvector vector store adapter."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from rag_core.search.document_records import (
    resolve_document_id_from_payload,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.planning import validate_query_plan_capabilities
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.provider_protocols import StoreCapabilities
from rag_core.search.query_plan import QueryPlan
from rag_core.search.request_models import DeleteFilter, SearchQuery, StoredDocumentRecord
from rag_core.search.vector_models import SearchResult, VectorPoint

from .chunk_lookup import validate_chunk_lookup_inputs
from .pgvector_config import PgVectorConfig, build_pgvector_config, quote_identifier
from .pgvector_filters import (
    build_chunk_lookup_where,
    build_delete_where,
    build_document_lookup_where,
    build_search_where,
)
from .pgvector_payloads import (
    point_to_pgvector_params,
    row_payload,
    row_to_search_result,
)
from .pgvector_query_plan import (
    resolve_pgvector_search_limit,
    validate_pgvector_query_plan,
)
from .registry import VECTOR_STORES
from .vector_dimensions import validate_point_dense_dimensions, validate_query_dense_dimensions
from .vector_store_capabilities import (
    PGVECTOR_VECTOR_STORE_CAPABILITY_SPEC,
    PGVECTOR_VECTOR_STORE_PROVIDER_SPEC,
)

if TYPE_CHECKING:
    import asyncpg

_PGVECTOR_TYPE_RE = re.compile(r"^vector(?:\((?P<dimensions>[1-9][0-9]*)\))?$")


class PgVectorAcquireContext(Protocol):
    async def __aenter__(self) -> "PgVectorConnection": ...

    async def __aexit__(self, *exc: object) -> None: ...


class PgVectorConnection(Protocol):
    async def execute(self, query: str, *args: object) -> object: ...

    async def executemany(
        self,
        command: str,
        args: Sequence[Sequence[object]],
    ) -> object: ...

    async def fetch(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None: ...

    async def fetchval(self, query: str, *args: object) -> object: ...


class PgVectorPool(Protocol):
    def acquire(self) -> PgVectorAcquireContext: ...

    async def close(self) -> object: ...


class PgVectorExtensionError(RuntimeError):
    """Raised when the connected database cannot provide the pgvector type."""


class PgVectorVectorStore:
    """Postgres/pgvector adapter for dense vector search and metadata filters."""

    provider_name = PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name

    def __init__(
        self,
        *,
        dsn: str | None = None,
        table_name: str,
        schema: str = "public",
        dense_dimensions: int,
        pool: PgVectorPool | None = None,
        policy: VectorStorePolicy = DEFAULT_POLICY,
    ) -> None:
        if pool is None and (dsn is None or not dsn.strip()):
            raise ValueError("PgVectorVectorStore requires dsn when pool is not provided")
        self._config: PgVectorConfig = build_pgvector_config(
            dsn=dsn,
            table_name=table_name,
            schema=schema,
            dense_dimensions=dense_dimensions,
            policy=policy,
        )
        self._policy = policy
        self._pool = pool
        self._owns_pool = pool is None
        self._ready = False
        self._ensure_lock = asyncio.Lock()

    async def __aenter__(self) -> "PgVectorVectorStore":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def capabilities(self) -> StoreCapabilities:
        return PGVECTOR_VECTOR_STORE_CAPABILITY_SPEC.to_store_capabilities(
            dense_vector_dimensions=self._config.dense_dimensions,
        )

    async def close(self) -> None:
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
        self._pool = None
        self._ready = False

    async def ensure_collection(self) -> None:
        async with self._ensure_lock:
            if self._ready:
                return
            pool = await self._ensure_pool()
            async with pool.acquire() as connection:
                await _ensure_pgvector_extension(connection)
                await connection.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(self._config.schema)}"
                )
                await connection.execute(_create_table_sql(self._config))
                await _verify_dense_column_dimensions(connection, self._config)
                await connection.execute(_create_dense_index_sql(self._config))
            self._ready = True

    async def check_health(self) -> dict[str, object]:
        try:
            await self.ensure_collection()
            pool = await self._ensure_pool()
            async with pool.acquire() as connection:
                count = await connection.fetchval(
                    f"SELECT COUNT(*) FROM {self._config.qualified_table}"
                )
            return {
                "healthy": True,
                "adapter": PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
                "points_count": _non_negative_int(count, "pgvector health count"),
                "schema": self._config.schema,
                "table": self._config.table_name,
            }
        except Exception:
            return {
                "healthy": False,
                "adapter": PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
                "schema": self._config.schema,
                "table": self._config.table_name,
            }

    def validate_query_plan(self, plan: QueryPlan) -> None:
        validate_query_plan_capabilities(
            plan,
            capabilities=PGVECTOR_VECTOR_STORE_CAPABILITY_SPEC.query_plan,
            provider_name=PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        validate_pgvector_query_plan(plan)

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        validate_point_dense_dimensions(
            points,
            dense_dimensions=self._config.dense_dimensions,
            provider_name=PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        await self.ensure_collection()
        pool = await self._ensure_pool()
        params = [
            point_to_pgvector_params(point, policy=self._policy)
            for point in points
        ]
        async with pool.acquire() as connection:
            await connection.executemany(_upsert_sql(self._config), params)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        namespace = query.namespace.strip()
        if not namespace:
            raise ValueError("namespace is required for search")
        if query.has_empty_allowlist():
            return []
        if not query.dense_vector:
            raise ValueError("pgvector dense query vector is required for search")
        validate_query_dense_dimensions(
            query.dense_vector,
            dense_dimensions=self._config.dense_dimensions,
            provider_name=PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        if query.query_plan is not None:
            self.validate_query_plan(query.query_plan)
        limit = resolve_pgvector_search_limit(query)
        where = build_search_where(
            query=query,
            namespace=namespace,
            policy=self._policy,
            start_index=2,
        )
        params = [query.dense_vector, *where.params, limit]
        limit_param = f"${len(params)}"
        sql = (
            "SELECT id, payload, 1.0 - (dense <=> $1::vector) AS score "
            f"FROM {self._config.qualified_table} "
            f"WHERE {where.sql} "
            "ORDER BY dense <=> $1::vector "
            f"LIMIT {limit_param}::integer"
        )
        await self.ensure_collection()
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(sql, *params)
        return [
            row_to_search_result(
                row,
                score=_float_value(row.get("score"), "pgvector search score"),
                policy=self._policy,
            )
            for row in rows
        ]

    async def delete(self, filter: DeleteFilter) -> None:
        namespace = (filter.namespace or "").strip()
        if not namespace:
            raise ValueError("namespace is required for delete")
        where = build_delete_where(
            filter_values=filter,
            namespace=namespace,
            policy=self._policy,
        )
        await self.ensure_collection()
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                f"DELETE FROM {self._config.qualified_table} WHERE {where.sql}",
                *where.params,
            )

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        await self.ensure_collection()
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                f"DELETE FROM {self._config.qualified_table} WHERE id = ANY($1::text[])",
                list(point_ids),
            )

    async def get_document_record(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        namespace_scoped, collection_scoped = validate_document_lookup_inputs(
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
        )
        where = build_document_lookup_where(
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
        )
        await self.ensure_collection()
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT id, payload FROM {self._config.qualified_table} "
                f"WHERE {where.sql} LIMIT 1",
                *where.params,
            )
            if row is None:
                return None
            payload = row_payload(row)
            resolved_document_id = resolve_document_id_from_payload(
                payload=payload,
                document_id_field=self._policy.document_id_field,
                fallback_document_id=document_id,
                invalid_message="pgvector document lookup returned invalid document_id",
                reject_blank=True,
            )
            if resolved_document_id is None:
                return None
            count_where = build_document_lookup_where(
                namespace=namespace_scoped,
                collection=collection_scoped,
                document_id=resolved_document_id,
                document_key=None,
                policy=self._policy,
            )
            count = await connection.fetchval(
                f"SELECT COUNT(*) FROM {self._config.qualified_table} "
                f"WHERE {count_where.sql}",
                *count_where.params,
            )
        return stored_document_record_from_payload(
            payload=payload,
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=resolved_document_id,
            chunk_count=_non_negative_int(count, "pgvector document chunk count"),
            policy=self._policy,
            invalid_field_message=(
                "search payload field must be a string: {field}"
            ),
        )

    async def get_chunks_by_index(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        chunk_indices: Sequence[int],
    ) -> list[SearchResult]:
        namespace_scoped, collection_scoped, document_scoped, indices = (
            validate_chunk_lookup_inputs(
                namespace=namespace,
                collection=collection,
                document_id=document_id,
                chunk_indices=chunk_indices,
            )
        )
        if not indices:
            return []
        where = build_chunk_lookup_where(
            namespace=namespace_scoped,
            collection=collection_scoped,
            document_id=document_scoped,
            chunk_indices=indices,
            policy=self._policy,
        )
        await self.ensure_collection()
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT id, payload FROM {self._config.qualified_table} "
                f"WHERE {where.sql} ORDER BY chunk_index ASC",
                *where.params,
            )
        results = [
            row_to_search_result(row, score=0.0, policy=self._policy)
            for row in rows
        ]
        return sorted(results, key=lambda result: result.chunk_index or 0)

    async def _ensure_pool(self) -> PgVectorPool:
        if self._pool is not None:
            return self._pool
        if self._config.dsn is None:
            raise ValueError("PgVectorVectorStore requires dsn when pool is not provided")
        await _ensure_extension_for_dsn(self._config.dsn)
        self._pool = await _create_asyncpg_pool(self._config.dsn)
        return self._pool


def _create_table_sql(config: PgVectorConfig) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {config.qualified_table} ("
        "id text PRIMARY KEY, "
        f"dense vector({config.dense_dimensions}) NOT NULL, "
        "payload jsonb NOT NULL, "
        "namespace text, "
        "collection text, "
        "document_id text, "
        "document_key text, "
        "content_sha256 text, "
        "processing_version text, "
        "content_type text, "
        "source_type text, "
        "chunk_index integer)"
    )


def _create_dense_index_sql(config: PgVectorConfig) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS {quote_identifier(config.dense_index_name)} "
        f"ON {config.qualified_table} USING hnsw (dense vector_cosine_ops)"
    )


async def _verify_dense_column_dimensions(
    connection: PgVectorConnection,
    config: PgVectorConfig,
) -> None:
    value = await connection.fetchval(
        """
        SELECT format_type(attribute.atttypid, attribute.atttypmod)
        FROM pg_attribute attribute
        JOIN pg_class relation ON relation.oid = attribute.attrelid
        JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = $1
          AND relation.relname = $2
          AND attribute.attname = 'dense'
          AND NOT attribute.attisdropped
        """,
        config.schema,
        config.table_name,
    )
    if not isinstance(value, str):
        raise ValueError(
            "pgvector table "
            f"{config.schema}.{config.table_name} is missing dense vector column; "
            "use a different table_name or migrate the existing collection"
        )
    found = _parse_pgvector_type_dimensions(value)
    if found != config.dense_dimensions:
        found_label = "unbounded" if found is None else str(found)
        raise ValueError(
            "pgvector table "
            f"{config.schema}.{config.table_name} dense vector dimension mismatch: "
            f"expected {config.dense_dimensions}, found {found_label}. "
            "Use a different table_name or migrate/reingest the existing collection."
        )


def _parse_pgvector_type_dimensions(value: str) -> int | None:
    match = _PGVECTOR_TYPE_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"pgvector dense column has unsupported type {value!r}")
    dimensions = match.group("dimensions")
    if dimensions is None:
        return None
    return int(dimensions)


def _upsert_sql(config: PgVectorConfig) -> str:
    return (
        f"INSERT INTO {config.qualified_table} "
        "(id, dense, payload, namespace, collection, document_id, document_key, "
        "content_sha256, processing_version, content_type, source_type, chunk_index) "
        "VALUES ($1, $2::vector, $3::jsonb, $4, $5, $6, $7, $8, $9, $10, $11, $12) "
        "ON CONFLICT (id) DO UPDATE SET "
        "dense = EXCLUDED.dense, "
        "payload = EXCLUDED.payload, "
        "namespace = EXCLUDED.namespace, "
        "collection = EXCLUDED.collection, "
        "document_id = EXCLUDED.document_id, "
        "document_key = EXCLUDED.document_key, "
        "content_sha256 = EXCLUDED.content_sha256, "
        "processing_version = EXCLUDED.processing_version, "
        "content_type = EXCLUDED.content_type, "
        "source_type = EXCLUDED.source_type, "
        "chunk_index = EXCLUDED.chunk_index"
    )


async def _create_asyncpg_pool(dsn: str) -> PgVectorPool:
    import asyncpg
    from pgvector.asyncpg import register_vector

    async def init(connection: "asyncpg.Connection") -> None:
        await register_vector(connection)

    pool: PgVectorPool = await asyncpg.create_pool(dsn=dsn, init=init)
    return pool


async def _ensure_extension_for_dsn(dsn: str) -> None:
    import asyncpg

    connection = await asyncpg.connect(dsn=dsn)
    try:
        await _ensure_pgvector_extension(connection)
    finally:
        await connection.close()


async def _ensure_pgvector_extension(connection: PgVectorConnection) -> None:
    exists = await connection.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
    )
    if exists is True:
        return
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:
        raise PgVectorExtensionError(
            "pgvector extension is not available in this Postgres database. "
            "Install the pgvector extension and run CREATE EXTENSION vector, "
            "or use a Postgres service/image with pgvector enabled."
        ) from exc
    exists = await connection.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
    )
    if exists is not True:
        raise PgVectorExtensionError(
            "pgvector extension did not become available after CREATE EXTENSION vector"
        )


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _float_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


VECTOR_STORES.register(
    PGVECTOR_VECTOR_STORE_PROVIDER_SPEC.name,
    lambda **kw: PgVectorVectorStore(**kw),
)
