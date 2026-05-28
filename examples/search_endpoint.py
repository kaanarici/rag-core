"""Application endpoint helper for a document search tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence

from rag_core import RAGCore
from rag_core.contracts import (
    normalize_static_content_types,
    normalize_static_retrieval_scope,
    parse_search_user_documents_request,
    scope_document_ids,
    search_user_documents_tool_result,
    validate_bound_namespace,
)
from rag_core.demo import build_demo_core


async def seed_demo_corpus(core: RAGCore) -> None:
    await core.ingest_bytes(
        file_bytes=(
            b"Billing invoices are due monthly. Customers can pay by card, "
            b"ACH, or wire transfer from the billing portal."
        ),
        filename="billing.txt",
        mime_type="text/plain",
        namespace="acme",
        corpus_id="help-center",
        document_id="billing",
    )
    await core.ingest_bytes(
        file_bytes=b"Shipping usually takes 3-5 business days after fulfillment.",
        filename="shipping.txt",
        mime_type="text/plain",
        namespace="acme",
        corpus_id="help-center",
        document_id="shipping",
    )


async def search_user_documents_endpoint(
    core: RAGCore,
    payload: Mapping[str, object],
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    authorized_document_ids: Sequence[str] | None = None,
    authorized_content_types: Sequence[str] | None = None,
) -> dict[str, object]:
    if authorized_document_ids is None:
        raise ValueError("authorized_document_ids must be bound by the app endpoint")
    request = parse_search_user_documents_request(payload)
    normalized_namespace = validate_bound_namespace(namespace)
    corpus_ids_tuple, authorized_document_ids_tuple = normalize_static_retrieval_scope(
        corpus_ids=corpus_ids,
        document_ids=authorized_document_ids,
        limit=request.limit,
    )
    content_types_tuple = normalize_static_content_types(authorized_content_types)
    document_ids = scope_document_ids(
        requested=request.document_ids,
        configured=authorized_document_ids_tuple,
    )
    pack = await core.retrieve_context(
        query=request.query,
        namespace=normalized_namespace,
        corpus_ids=list(corpus_ids_tuple),
        limit=request.limit,
        content_types=(
            list(content_types_tuple) if content_types_tuple is not None else None
        ),
        document_ids=document_ids,
        rerank=request.rerank,
        use_lexical_search=request.use_lexical_search,
        max_chars=request.max_chars,
        max_tokens=request.max_tokens,
    )
    return search_user_documents_tool_result(pack)


async def run_demo() -> dict[str, object]:
    core = build_demo_core(collection="search_endpoint")
    async with core:
        await seed_demo_corpus(core)
        return await search_user_documents_endpoint(
            core,
            {
                "query": "How can a customer pay an invoice?",
                "limit": 3,
                "rerank": False,
                "max_chars": 1200,
            },
            namespace="acme",
            corpus_ids=["help-center"],
            authorized_document_ids=["billing", "shipping"],
        )


def main() -> None:
    print(json.dumps(asyncio.run(run_demo()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
