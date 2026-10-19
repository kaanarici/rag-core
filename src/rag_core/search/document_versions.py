"""Publish complete, immutable document versions through a single head record.

Heads live beside chunks in the vector store. Staging never overwrites a live
version. A lost publish acknowledgement may leave either complete version live;
it must never trigger deletion of the potentially published version.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from rag_core.search.policy import VectorStorePolicy, scoped_uuid
from rag_core.search.request_models import SearchQuery, StoredDocumentRecord
from rag_core.search.vector_models import SearchResult, VectorPoint

KIND = "_rag_kind"
REVISION = "_rag_revision"
EXPECTED_CHUNKS = "_rag_chunks"
HEAD = "head_v2"
CHUNK = "chunk_v2"
Candidate = tuple[SearchResult, str | None]
logger = logging.getLogger(__name__)
_MAX_VISIBILITY_READS = 8


class ConcurrentDocumentUpdate(RuntimeError):
    """The document changed while a new version was being prepared; retry."""


class LegacyIndexError(ValueError):
    """An unversioned index must be rebuilt rather than guessed to be complete."""


class SearchVisibilityError(RuntimeError):
    """Bounded refill could not establish a committed result set; retry or repair."""

    def __init__(self, *, attempts: int, excluded_versions: int) -> None:
        self.attempts = attempts
        self.excluded_versions = excluded_versions
        super().__init__(
            f"search visibility budget exhausted after {attempts} reads "
            f"excluding {excluded_versions} unpublished versions; retry or rebuild the index"
        )


class VersionBackend(Protocol):
    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...
    async def get_document_record(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None: ...
    async def _read_heads(self, ids: Sequence[str]) -> dict[str, dict[str, object]]: ...
    async def _publish_head(
        self, id: str, payload: dict[str, object], expected_revision: str | None
    ) -> bool:
        """Return a confirmed applied/rejected outcome; raise if it is unknown."""
        ...
    async def _count_revision(
        self, record: StoredDocumentRecord, revision: str
    ) -> int: ...
    async def _delete_revision(
        self, record: StoredDocumentRecord, revision: str
    ) -> None: ...


def head_id(tenant_id: str, collection: str, document_id: str) -> str:
    return scoped_uuid("document-head-v2", tenant_id, collection, document_id)


def head_payload(
    record: StoredDocumentRecord, policy: VectorStorePolicy
) -> dict[str, object]:
    return {
        "_rag_deleted": record.deleted,
        "_rag_retired": list(record.retired_revisions),
        KIND: HEAD,
        REVISION: record.revision,
        EXPECTED_CHUNKS: record.expected_chunk_count,
        policy.tenant_id_field: record.tenant_id,
        policy.collection_field: record.collection,
        policy.document_id_field: record.document_id,
        policy.document_key_field: record.document_key,
        policy.content_sha256_field: record.content_sha256,
        policy.processing_version_field: record.processing_version,
        "ingest_fingerprint": record.ingest_fingerprint,
    }


def parse_head(
    payload: Mapping[str, object], policy: VectorStorePolicy
) -> StoredDocumentRecord:
    from rag_core.search.document_records import stored_document_record_from_payload

    revision = payload.get(REVISION)
    expected = payload.get(EXPECTED_CHUNKS)
    if (
        payload.get(KIND) != HEAD
        or not isinstance(revision, str)
        or len(revision) != 32
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("invalid document head revision")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ValueError("invalid document head chunk count")
    scope = [
        payload.get(field)
        for field in (
            policy.tenant_id_field,
            policy.collection_field,
            policy.document_id_field,
        )
    ]
    if any(not isinstance(value, str) or not value.strip() for value in scope):
        raise ValueError("invalid document head scope")
    retired = payload.get("_rag_retired", [])
    if (
        not isinstance(retired, (list, tuple))
        or len(retired) > 16
        or any(
            not isinstance(value, str)
            or len(value) != 32
            or any(c not in "0123456789abcdef" for c in value)
            or value == revision
            for value in retired
        )
    ):
        raise ValueError("invalid document head cleanup revisions")
    if not isinstance(payload.get("_rag_deleted", False), bool):
        raise ValueError("invalid document head deletion state")
    tenant, collection, document = scope
    assert (
        isinstance(tenant, str)
        and isinstance(collection, str)
        and isinstance(document, str)
    )
    return replace(
        stored_document_record_from_payload(
            payload=payload,
            tenant_id=tenant,
            collection=collection,
            document_id=document,
            chunk_count=expected,
            policy=policy,
            invalid_field_message="invalid document head {field}",
        ),
        revision=revision,
        expected_chunk_count=expected,
        committed=True,
        deleted=payload.get("_rag_deleted") is True,
        retired_revisions=tuple(retired),
    )


def candidate(result: SearchResult, payload: Mapping[str, object]) -> Candidate:
    revision = payload.get(REVISION)
    if KIND not in payload and REVISION not in payload:
        return result, None  # Raw low-level points, never eligible for ingest skip.
    if payload.get(KIND) != CHUNK or not isinstance(revision, str) or not revision:
        raise ValueError("invalid document chunk revision")
    return result, revision


async def finish_cleanup(
    backend: VersionBackend, head: StoredDocumentRecord, policy: VectorStorePolicy
) -> None:
    for revision in head.retired_revisions:
        await backend._delete_revision(head, revision)
    if head.retired_revisions:
        await backend._publish_head(
            head_id(head.tenant_id, head.collection, head.document_id),
            head_payload(replace(head, retired_revisions=()), policy),
            expected_revision=head.revision,
        )


def retirement_list(old: StoredDocumentRecord | None) -> tuple[str, ...]:
    if old is None:
        return ()
    values = [*old.retired_revisions]
    if old.revision is not None and not old.deleted:
        values.append(old.revision)
    return tuple(dict.fromkeys(values))


async def replace_document(
    backend: VersionBackend,
    record: StoredDocumentRecord,
    points: Sequence[VectorPoint],
    policy: VectorStorePolicy,
) -> None:
    if record.chunk_count != len(points) or len({point.id for point in points}) != len(
        points
    ):
        raise ValueError("document write requires one unique point per chunk")
    for point in points:
        if any(
            point.payload.get(field) != value
            for field, value in (
                (policy.tenant_id_field, record.tenant_id),
                (policy.collection_field, record.collection),
                (policy.document_id_field, record.document_id),
            )
        ):
            raise ValueError("document point scope differs from its head")
    published_id = head_id(record.tenant_id, record.collection, record.document_id)
    previous = (await backend._read_heads([published_id])).get(published_id)
    old = (
        parse_head(previous, policy)
        if previous is not None
        else await backend.get_document_record(
            tenant_id=record.tenant_id,
            collection=record.collection,
            document_id=record.document_id,
        )
    )
    if old is not None and (old.tenant_id, old.collection, old.document_id) != (
        record.tenant_id,
        record.collection,
        record.document_id,
    ):
        raise ValueError("document head returned a different scope")
    if old is not None and old.revision is None:
        raise LegacyIndexError(
            "Unversioned index: rebuild into a fresh index directory, Qdrant collection or TurboPuffer namespace; existing data was not changed"
        )
    if old is not None and len(retirement_list(old)) > 16:
        await finish_cleanup(backend, old, policy)
        old = replace(old, retired_revisions=())
    revision = uuid.uuid4().hex
    current = replace(
        record,
        revision=revision,
        expected_chunk_count=len(points),
        committed=True,
        deleted=False,
        retired_revisions=retirement_list(old),
    )
    staged = [
        replace(
            point,
            id=scoped_uuid(
                "document-chunk-v2",
                record.tenant_id,
                record.collection,
                record.document_id,
                revision,
                point.id,
            ),
            payload={**point.payload, KIND: CHUNK, REVISION: revision},
        )
        for point in points
    ]
    try:
        await backend.upsert(staged)
        count = await backend._count_revision(current, revision)
        if count != len(staged):
            raise RuntimeError(
                "incomplete staged document; previous version remains published"
            )
    except BaseException:
        try:
            await backend._delete_revision(current, revision)
        except Exception:
            logger.warning("Could not remove an unpublished document version")
        raise
    # Only a definite rejection permits cleanup after publication starts.
    applied = await backend._publish_head(
        head_id(record.tenant_id, record.collection, record.document_id),
        head_payload(current, policy),
        expected_revision=old.revision if old is not None and old.committed else None,
    )
    if applied is False:
        try:
            await backend._delete_revision(current, revision)
        except Exception:
            logger.warning("Could not remove a rejected document version")
        raise ConcurrentDocumentUpdate("document changed during publish; retry ingest")
    if applied is not True:
        raise RuntimeError("backend returned an unknown publication outcome")
    try:
        await finish_cleanup(backend, current, policy)
    except Exception:
        logger.warning(
            "Could not remove obsolete document versions; cleanup is recorded for retry"
        )


async def committed_search(
    query: SearchQuery,
    search: Callable[[SearchQuery], Awaitable[list[Candidate]]],
    read_heads: Callable[[Sequence[str]], Awaitable[dict[str, dict[str, object]]]],
    policy: VectorStorePolicy,
) -> list[SearchResult]:
    excluded = set(query.excluded_revisions)
    for attempt in range(1, _MAX_VISIBILITY_READS + 1):
        rows = await search(replace(query, excluded_revisions=tuple(sorted(excluded))))
        ids: dict[str, tuple[str, str, str]] = {}
        for hit, revision in rows:
            if not hit.tenant_id or not hit.collection or not hit.document_id:
                if revision is not None:
                    raise ValueError("versioned search result has no scope")
                continue
            scope = (hit.tenant_id, hit.collection, hit.document_id)
            ids[head_id(*scope)] = scope
        heads = await read_heads(list(ids)) if ids else {}
        active: dict[tuple[str, str, str], str | None] = {}
        for id, scope in ids.items():
            if id not in heads:
                active[scope] = None
                continue
            head = parse_head(heads[id], policy)
            if (head.tenant_id, head.collection, head.document_id) != scope:
                raise ValueError("document head returned a different scope")
            active[scope] = head.revision
        visible: list[SearchResult] = []
        rejected = set()
        for hit, revision in rows:
            if revision == active.get(
                (hit.tenant_id or "", hit.collection or "", hit.document_id or "")
            ):
                visible.append(hit)
            else:
                if revision is not None:
                    rejected.add(revision)
        if not rejected or len(visible) >= query.limit:
            return visible[: query.limit]
        # Requery without *whole* unfinished versions so they cannot consume
        # the candidate budget and hide the previously committed document.
        if rejected <= excluded:
            break
        excluded.update(rejected)
    raise SearchVisibilityError(attempts=attempt, excluded_versions=len(excluded))


async def delete_document(
    backend: VersionBackend, record: StoredDocumentRecord, policy: VectorStorePolicy
) -> None:
    """Keep a tombstone so a delayed/replayed publish cannot recreate a delete."""
    id = head_id(record.tenant_id, record.collection, record.document_id)
    previous = (await backend._read_heads([id])).get(id)
    old = (
        parse_head(previous, policy)
        if previous is not None
        else await backend.get_document_record(
            tenant_id=record.tenant_id,
            collection=record.collection,
            document_id=record.document_id,
        )
    )
    if old is not None and (old.tenant_id, old.collection, old.document_id) != (
        record.tenant_id,
        record.collection,
        record.document_id,
    ):
        raise ValueError("document head returned a different scope")
    if old is not None and len(retirement_list(old)) > 16:
        await finish_cleanup(backend, old, policy)
        old = replace(old, retired_revisions=())
    deleted = replace(
        record,
        revision=uuid.uuid4().hex,
        chunk_count=0,
        expected_chunk_count=0,
        committed=True,
        deleted=True,
        retired_revisions=retirement_list(old),
    )
    applied = await backend._publish_head(
        id,
        head_payload(deleted, policy),
        expected_revision=old.revision if old is not None and old.committed else None,
    )
    if applied is False:
        raise ConcurrentDocumentUpdate("document changed during deletion; retry")
    if applied is not True:
        raise RuntimeError("backend returned an unknown publication outcome")
    await finish_cleanup(backend, deleted, policy)
