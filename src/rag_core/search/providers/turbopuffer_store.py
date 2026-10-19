"""Optional TurboPuffer store using native BM25 and the async Python SDK."""

from __future__ import annotations
import asyncio
import math
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any
from rag_core.search.document_records import (
    document_record_fields,
    stored_document_record_from_payload,
    validate_document_lookup_inputs,
)
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy, scoped_uuid
from rag_core.search.document_versions import (
    KIND,
    REVISION,
    HEAD,
    CHUNK,
    Candidate,
    candidate,
    committed_search,
    delete_document as delete_document_version,
    head_id,
    parse_head,
    replace_document,
)
from rag_core.search.provider_protocols import (
    MetadataFilterCapabilities,
    StoreCapabilities,
)
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
    StoredDocumentRecord,
)
from rag_core.search.result_filters import result_matches_query
from rag_core.search.vector_models import SearchResult, VectorPoint
from .turbopuffer_payloads import (
    TURBOPUFFER_BM25_TEXT_FIELD,
    _delete_filter,
    _document_lookup_filter,
    _point_to_row,
    _required_response_rows,
    _required_row_id,
    _row_payload,
    _row_to_result,
    _schema,
    _search_filter,
    _validate_point_id,
    _validated_point_payload,
)
from .vector_dimensions import (
    validate_point_dense_dimensions,
    validate_query_dense_dimensions,
)


class TurboPufferVectorStore:
    provider_name = "turbopuffer"

    def __init__(
        self,
        *,
        namespace: str,
        dense_dimensions: int,
        region: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        namespace_client: Any = None,
        write_batch_size: int = 256,
        delete_continuation_limit: int = 100,
        policy: VectorStorePolicy = DEFAULT_POLICY,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", namespace):
            raise ValueError("invalid TurboPuffer namespace")
        for name, value in (
            ("dense_dimensions", dense_dimensions),
            ("write_batch_size", write_batch_size),
            ("delete_continuation_limit", delete_continuation_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.namespace = namespace
        self._dimensions = dense_dimensions
        self._region = region
        self._api_key = api_key
        self._client = client
        self._namespace_client = namespace_client
        self._owns_client = client is None and namespace_client is None
        self._batch_size = write_batch_size
        self._delete_limit = delete_continuation_limit
        self._policy = policy
        self._embedding_model: str | None = None
        self._identity_ready = False
        self._identity_lock = asyncio.Lock()

    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            per_point_delete=True,
            dense_vector_dimensions=self._dimensions,
            hybrid=True,
            lexical=True,
            native_lexical=True,
            metadata_filter=MetadataFilterCapabilities(
                term=True, in_=True, numeric_range=True, string_range=True, boolean=True
            ),
        )

    def _namespace(self) -> Any:
        if self._namespace_client is None:
            if self._client is None:
                try:
                    from turbopuffer import AsyncTurbopuffer
                except ImportError:
                    raise ImportError(
                        "Install rag-core[turbopuffer] to use TurboPuffer"
                    ) from None
                kwargs: dict[str, Any] = {}
                if self._api_key is not None:
                    kwargs["api_key"] = self._api_key
                if self._region is not None:
                    kwargs["region"] = self._region
                self._client = AsyncTurbopuffer(**kwargs)
            self._namespace_client = self._client.namespace(self.namespace)
        return self._namespace_client

    def bind_embedding_space(self, model_name: str) -> None:
        if self._embedding_model is not None and self._embedding_model != model_name:
            raise ValueError(
                "Store is bound to a different embedding model; use a fresh namespace"
            )
        self._embedding_model = model_name

    async def ensure_collection(self) -> None:
        ready = self._identity_ready
        if self._embedding_model is None or ready:
            return
        async with self._identity_lock:
            if self._identity_ready:
                return
            id = scoped_uuid("embedding-space-v1")
            expected = {
                "_rag_kind": "embedding_space_v1",
                "model": self._embedding_model,
                "dimensions": self._dimensions,
            }
            existing = (await self._read_heads([id])).get(id)
            if existing is None:
                try:
                    response = await self._namespace().query(
                        aggregate_by={"rows": ("Count",)}
                    )
                    count = (
                        response.aggregations.get("rows")
                        if isinstance(response.aggregations, Mapping)
                        else None
                    )
                except Exception as exc:
                    if getattr(exc, "status_code", None) != 404:
                        raise
                    count = 0
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("Namespace identity count is invalid")
                if count:
                    # Another initializer may have published after our first read.
                    existing = (await self._read_heads([id])).get(id)
                    if existing is None:
                        raise ValueError(
                            "Namespace has no embedding identity; reingest into a fresh namespace"
                        )
                else:
                    await self._namespace().write(
                        upsert_rows=[
                            {
                                "id": id,
                                "_rag_kind": "embedding_space_v1",
                                "rag_core_payload": json.dumps(expected),
                            }
                        ],
                        upsert_condition=("id", "Eq", None),
                        schema=_schema(self._dimensions, policy=self._policy),
                        distance_metric="cosine_distance",
                    )
                    existing = (await self._read_heads([id])).get(id)
            if existing != expected:
                raise ValueError(
                    "Namespace uses a different embedding model or dimensions; use a fresh namespace"
                )
            self._identity_ready = True

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = self._namespace_client = None

    async def check_health(self) -> dict[str, object]:
        try:
            await self._namespace().metadata()
            return {"provider": self.provider_name, "healthy": True}
        except Exception as exc:
            return {
                "provider": self.provider_name,
                "healthy": False,
                "error": type(exc).__name__,
            }

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        validate_point_dense_dimensions(
            points, dense_dimensions=self._dimensions, provider_name=self.provider_name
        )
        for point in points:
            _validated_point_payload(point)
        await self.ensure_collection()
        schema = _schema(self._dimensions, policy=self._policy)
        for start in range(0, len(points), self._batch_size):
            await self._namespace().write(
                upsert_rows=[
                    _point_to_row(point, policy=self._policy)
                    for point in points[start : start + self._batch_size]
                ],
                schema=schema,
                distance_metric="cosine_distance",
            )

    async def replace_document(
        self, record: StoredDocumentRecord, points: Sequence[VectorPoint]
    ) -> None:
        await self.ensure_collection()
        await replace_document(self, record, points, self._policy)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return await committed_search(
            query, self._search_candidates, self._read_heads, self._policy
        )

    async def _search_candidates(self, query: SearchQuery) -> list[Candidate]:
        if query.has_empty_allowlist():
            return []
        if query.mode != "lexical":
            validate_query_dense_dimensions(
                query.dense_vector,
                dense_dimensions=self._dimensions,
                provider_name=self.provider_name,
            )
        if query.mode != "dense" and not (query.lexical_query or "").strip():
            raise ValueError("TurboPuffer hybrid/lexical search requires lexical_query")
        await self.ensure_collection()
        filters = _search_filter(
            query=query, tenant_id=query.tenant_id, policy=self._policy
        )
        dense = {
            "rank_by": ("vector", "ANN", query.dense_vector),
            "filters": filters,
            "limit": query.limit,
            "include_attributes": True,
        }
        lexical = {
            **dense,
            "rank_by": (TURBOPUFFER_BM25_TEXT_FIELD, "BM25", query.lexical_query),
        }
        try:
            if query.mode == "hybrid":
                pool = query.candidate_limit or query.limit
                response = await self._namespace().multi_query(
                    queries=[{**dense, "limit": pool}, {**lexical, "limit": pool}]
                )
                if not isinstance(response.results, list) or len(response.results) != 2:
                    raise ValueError(
                        "TurboPuffer multi-query must return two result lists"
                    )
                groups = [
                    self._results(r, bm25=i == 1)
                    for i, r in enumerate(response.results)
                ]
                scores: dict[str, float] = {}
                hits: dict[str, Candidate] = {}
                for group in groups:
                    for rank, (hit, revision) in enumerate(group):
                        scores[hit.id] = scores.get(hit.id, 0) + 1 / (60 + rank + 1)
                        hits.setdefault(hit.id, (hit, revision))
                results = [
                    (replace(hits[key][0], score=scores[key]), hits[key][1])
                    for key in sorted(hits, key=lambda key: (-scores[key], key))
                ]
            else:
                response = await self._namespace().query(
                    **(lexical if query.mode == "lexical" else dense)
                )
                results = self._results(response, bm25=query.mode == "lexical")
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return []
            raise
        scope_query = replace(query, metadata_filter=None)
        return [
            (hit, revision)
            for hit, revision in results
            if result_matches_query(hit, scope_query)
        ][: query.limit]

    def _results(self, response: Any, *, bm25: bool) -> list[Candidate]:
        results = []
        for row in _required_response_rows(response, operation="search"):
            if bm25:
                payload = _row_payload(row)
                from rag_core.search.stored_payload import payload_to_result

                raw = (
                    row.get("$dist")
                    if isinstance(row, Mapping)
                    else getattr(row, "$dist", None)
                )
                if (
                    isinstance(raw, bool)
                    or not isinstance(raw, (int, float))
                    or not math.isfinite(raw)
                ):
                    raise ValueError("TurboPuffer BM25 result requires a finite score")
                hit = payload_to_result(
                    point_id=_required_row_id(row),
                    payload=payload,
                    score=float(raw),
                    policy=self._policy,
                )
            else:
                hit = _row_to_result(
                    row, distance_metric="cosine_distance", policy=self._policy
                )
            results.append(candidate(hit, _row_payload(row)))
        return results

    async def delete(self, filter: DeleteFilter) -> None:

        if filter.document_id is not None and filter.collection is not None:
            await self.ensure_collection()
            await delete_document_version(
                self,
                StoredDocumentRecord(
                    tenant_id=filter.tenant_id or "",
                    collection=filter.collection,
                    document_id=filter.document_id,
                ),
                self._policy,
            )
            # Unversioned low-level rows have no revision; managed rows are
            # removed only by their captured version, never by a broad rollback.
            await self._delete_legacy_document(filter)
            return
        scope = _delete_filter(
            filter_values=filter, tenant_id=filter.tenant_id or "", policy=self._policy
        )
        try:
            for _ in range(self._delete_limit):
                response = await self._namespace().write(
                    delete_by_filter=scope, delete_by_filter_allow_partial=True
                )
                if not isinstance(response.rows_remaining, bool):
                    raise ValueError(
                        "TurboPuffer delete returned invalid rows_remaining"
                    )
                if not response.rows_remaining:
                    return
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return
            raise
        raise RuntimeError(
            "TurboPuffer delete continuation limit reached; retry deletion"
        )

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        for start in range(0, len(point_ids), self._batch_size):
            await self._namespace().write(
                deletes=[
                    _validate_point_id(x)
                    for x in point_ids[start : start + self._batch_size]
                ]
            )

    async def _delete_legacy_document(self, values: DeleteFilter) -> None:
        scope = _delete_filter(
            filter_values=values, tenant_id=values.tenant_id or "", policy=self._policy
        )
        for _ in range(self._delete_limit):
            response = await self._namespace().write(
                delete_by_filter=("And", (scope, (KIND, "Eq", None))),
                delete_by_filter_allow_partial=True,
            )
            if not isinstance(response.rows_remaining, bool):
                raise ValueError("TurboPuffer delete returned invalid rows_remaining")
            if not response.rows_remaining:
                return
        raise RuntimeError("TurboPuffer legacy cleanup limit reached")

    async def _read_heads(self, ids: Sequence[str]) -> dict[str, dict[str, object]]:
        if not ids:
            return {}
        try:
            response = await self._namespace().query(
                rank_by=("id", "asc"),
                filters=("id", "In", list(ids)),
                limit=len(ids),
                include_attributes=True,
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return {}
            raise
        return {
            _required_row_id(row): _row_payload(row)
            for row in _required_response_rows(response, operation="document heads")
        }

    async def _publish_head(
        self, id: str, payload: dict[str, object], expected_revision: str | None
    ) -> bool:
        row = {
            "id": id,
            **payload,
            "rag_core_payload": json.dumps(payload, allow_nan=False),
        }
        if expected_revision is None:
            response = await self._namespace().write(
                upsert_rows=[row],
                upsert_condition=("id", "Eq", None),
                schema=_schema(self._dimensions, policy=self._policy),
                distance_metric="cosine_distance",
            )
        else:
            response = await self._namespace().write(
                patch_rows=[row], patch_condition=(REVISION, "Eq", expected_revision)
            )
        affected: object = response.rows_affected
        if isinstance(affected, bool) or not isinstance(affected, int) or affected not in (0, 1):
            raise ValueError("TurboPuffer returned an unknown publication outcome")
        if affected == 1:
            return True
        # The SDK may replay an applied conditional write after losing its ack.
        observed = (await self._read_heads([id])).get(id)
        if observed is None:
            raise RuntimeError("TurboPuffer publication outcome could not be verified")
        return parse_head(observed, self._policy).revision == payload[REVISION]

    def _revision_filter(self, record: StoredDocumentRecord, revision: str) -> object:
        scope = _document_lookup_filter(
            tenant_id=record.tenant_id,
            collection=record.collection,
            document_id=record.document_id,
            document_key=None,
            policy=self._policy,
        )
        return ("And", (scope, (KIND, "Eq", CHUNK), (REVISION, "Eq", revision)))

    async def _count_revision(self, record: StoredDocumentRecord, revision: str) -> int:
        response = await self._namespace().query(
            filters=self._revision_filter(record, revision),
            aggregate_by={"chunks": ("Count",)},
        )
        count = (
            response.aggregations.get("chunks")
            if isinstance(response.aggregations, Mapping)
            else None
        )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("TurboPuffer returned invalid revision chunk count")
        return count

    async def _delete_revision(
        self, record: StoredDocumentRecord, revision: str
    ) -> None:
        for _ in range(self._delete_limit):
            response = await self._namespace().write(
                delete_by_filter=self._revision_filter(record, revision),
                delete_by_filter_allow_partial=True,
            )
            if not isinstance(response.rows_remaining, bool):
                raise ValueError("TurboPuffer delete returned invalid rows_remaining")
            if not response.rows_remaining:
                return
        raise RuntimeError("TurboPuffer revision cleanup limit reached")

    async def get_document_record(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        tenant_id, collection = validate_document_lookup_inputs(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
        )
        await self.ensure_collection()
        if document_id is not None:
            id = head_id(tenant_id, collection, document_id)
            payload = (await self._read_heads([id])).get(id)
        else:
            scope = _document_lookup_filter(
                tenant_id=tenant_id,
                collection=collection,
                document_id=None,
                document_key=document_key,
                policy=self._policy,
            )
            try:
                response = await self._namespace().query(
                    rank_by=("id", "asc"),
                    filters=("And", (scope, (KIND, "Eq", HEAD))),
                    limit=2,
                    include_attributes=True,
                )
            except Exception as exc:
                if getattr(exc, "status_code", None) == 404:
                    return None
                raise
            rows = _required_response_rows(response, operation="document head lookup")
            if len(rows) > 1:
                raise ValueError("document key identifies multiple published documents")
            payload = _row_payload(rows[0]) if rows else None
        if payload is not None:
            record = parse_head(payload, self._policy)
            if (
                record.tenant_id != tenant_id
                or record.collection != collection
                or (document_id is not None and record.document_id != document_id)
                or (document_key is not None and record.document_key != document_key)
            ):
                raise ValueError("TurboPuffer document head returned a different scope")
            if record.deleted:
                return None
            assert record.revision is not None
            return replace(
                record, chunk_count=await self._count_revision(record, record.revision)
            )
        return await self._legacy_record(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
        )

    async def _legacy_record(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        tenant_id, collection = validate_document_lookup_inputs(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
        )
        scope = _document_lookup_filter(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
        )
        try:
            lookup = dict(
                rank_by=("id", "asc"),
                filters=scope,
                limit=1,
                include_attributes=document_record_fields(self._policy),
            )
            counted = None
            if document_id is not None:
                count_scope = _document_lookup_filter(
                    tenant_id=tenant_id,
                    collection=collection,
                    document_id=document_id,
                    document_key=None,
                    policy=self._policy,
                )
                combined = await self._namespace().multi_query(
                    queries=[
                        lookup,
                        {
                            "filters": count_scope,
                            "aggregate_by": {"chunk_count": ("Count",)},
                        },
                    ]
                )
                if not isinstance(combined.results, list) or len(combined.results) != 2:
                    raise ValueError(
                        "TurboPuffer document lookup must return a row and a count result"
                    )
                response, counted = combined.results
            else:
                response = await self._namespace().query(**lookup)
            rows = _required_response_rows(response, operation="document lookup")
            if not rows:
                return None
            sample = _row_payload(rows[0])
            resolved_id = sample.get(self._policy.document_id_field)
            if (
                not isinstance(resolved_id, str)
                or not resolved_id.strip()
                or sample.get(self._policy.tenant_id_field) != tenant_id
                or sample.get(self._policy.collection_field) != collection
                or document_id is not None
                and resolved_id != document_id
                or document_key is not None
                and sample.get(self._policy.document_key_field) != document_key
            ):
                raise ValueError(
                    "TurboPuffer document lookup returned a different scope"
                )
            if counted is None:
                scope = _document_lookup_filter(
                    tenant_id=tenant_id,
                    collection=collection,
                    document_id=resolved_id,
                    document_key=None,
                    policy=self._policy,
                )
                counted = await self._namespace().query(
                    filters=scope, aggregate_by={"chunk_count": ("Count",)}
                )
            count = (
                counted.aggregations.get("chunk_count")
                if isinstance(counted.aggregations, Mapping)
                else None
            )
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "TurboPuffer document lookup returned invalid chunk count"
                )
            return stored_document_record_from_payload(
                payload=sample,
                tenant_id=tenant_id,
                collection=collection,
                document_id=resolved_id,
                chunk_count=count,
                policy=self._policy,
                invalid_field_message="TurboPuffer document lookup returned invalid {field}",
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise
