"""Tenant-payload-index hint plumbing into the Qdrant adapter.

``VectorStorePolicy.tenant_payload_field`` opts a single payload field into
Qdrant's multi-tenant optimization (``is_tenant=True``). The matching field
becomes a ``KeywordIndexParams`` with that flag set; everything else stays on
the default keyword schema; ``None`` keeps every field on the default schema.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from qdrant_client import models as rest

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.providers.qdrant_store import (
    CollectionConfig,
    create_collection,
)


class _FakeClient:
    def __init__(self) -> None:
        self.create_collection_calls: list[dict[str, object]] = []
        self.create_payload_index_calls: list[
            tuple[str, rest.PayloadSchemaType | rest.KeywordIndexParams]
        ] = []

    async def create_collection(self, **kwargs: object) -> None:
        self.create_collection_calls.append(kwargs)

    async def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: rest.PayloadSchemaType | rest.KeywordIndexParams,
        wait: bool = True,
    ) -> rest.UpdateResult:
        assert collection_name == "docs"
        assert wait is True
        self.create_payload_index_calls.append((field_name, field_schema))
        return rest.UpdateResult(status=rest.UpdateStatus.COMPLETED)


def _run_create(policy: VectorStorePolicy) -> _FakeClient:
    client = _FakeClient()
    asyncio.run(
        create_collection(
            client=cast(Any, client),
            config=CollectionConfig(
                collection_name="docs",
                dimensions=3072,
                quantization_enabled=False,
                is_local=False,
                policy=policy,
            ),
        )
    )
    return client


def test_create_collection_marks_tenant_payload_index_when_policy_sets_it() -> None:
    client = _run_create(VectorStorePolicy(tenant_payload_field="tenant_id"))

    by_field = dict(client.create_payload_index_calls)
    tenant_id_schema = by_field["tenant_id"]
    assert isinstance(tenant_id_schema, rest.KeywordIndexParams)
    assert tenant_id_schema.is_tenant is True
    assert tenant_id_schema.type == rest.KeywordIndexType.KEYWORD

    # Other policy-driven fields keep the default keyword schema so we don't
    # accidentally opt every field into multi-tenant mode.
    other_schema = by_field["collection"]
    assert not isinstance(other_schema, rest.KeywordIndexParams)
    assert other_schema == rest.PayloadSchemaType.KEYWORD


def test_create_collection_skips_tenant_hint_when_policy_field_is_none() -> None:
    client = _run_create(VectorStorePolicy())

    schemas = [schema for _, schema in client.create_payload_index_calls]
    assert all(not isinstance(schema, rest.KeywordIndexParams) for schema in schemas)


def test_create_collection_routes_tenant_hint_to_custom_field_name() -> None:
    client = _run_create(
        VectorStorePolicy(tenant_id_field="ns", tenant_payload_field="ns")
    )

    fields = [field_name for field_name, _ in client.create_payload_index_calls]
    assert "ns" in fields
    assert "tenant_id" not in fields
    ns_schema = dict(client.create_payload_index_calls)["ns"]
    assert isinstance(ns_schema, rest.KeywordIndexParams)
    assert ns_schema.is_tenant is True
