"""Payload index helpers for Qdrant collection setup."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.policy import VectorStorePolicy


async def create_payload_indexes(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    policy: VectorStorePolicy,
) -> None:
    tenant_field = policy.tenant_payload_field
    for field_name, schema_type in collection_index_fields(policy):
        if tenant_field is not None and field_name == tenant_field:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=rest.KeywordIndexParams(
                    type=rest.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            continue
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )


def collection_index_fields(
    policy: VectorStorePolicy,
) -> tuple[tuple[str, rest.PayloadSchemaType], ...]:
    return (
        (policy.namespace_field, rest.PayloadSchemaType.KEYWORD),
        (policy.corpus_id_field, rest.PayloadSchemaType.KEYWORD),
        (policy.document_id_field, rest.PayloadSchemaType.KEYWORD),
        (policy.document_key_field, rest.PayloadSchemaType.KEYWORD),
        (policy.content_sha256_field, rest.PayloadSchemaType.KEYWORD),
        (policy.processing_version_field, rest.PayloadSchemaType.KEYWORD),
        (policy.content_type_field, rest.PayloadSchemaType.KEYWORD),
        (policy.source_type_field, rest.PayloadSchemaType.KEYWORD),
    )
