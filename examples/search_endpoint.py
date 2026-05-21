"""Application endpoint helper for a document search tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence

from rag_core import RAGCore
from rag_core.contracts import (
    parse_search_user_documents_request,
    search_user_documents_tool_result,
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
) -> dict[str, object]:
    if authorized_document_ids is None:
        raise ValueError("authorized_document_ids must be bound by the app endpoint")
    request = parse_search_user_documents_request(payload)
    document_ids = _authorized_document_ids(
        request.document_ids,
        authorized_document_ids=authorized_document_ids,
    )
    pack = await core.retrieve_context(
        query=request.query,
        namespace=namespace,
        corpus_ids=list(corpus_ids),
        limit=request.limit,
        document_ids=document_ids,
        rerank=request.rerank,
        use_lexical_search=request.use_lexical_search,
        max_chars=request.max_chars,
        max_tokens=request.max_tokens,
    )
    return search_user_documents_tool_result(pack)


def _authorized_document_ids(
    requested_document_ids: Sequence[str] | None,
    *,
    authorized_document_ids: Sequence[str],
) -> list[str]:
    if requested_document_ids is None:
        return list(authorized_document_ids)

    allowed = set(authorized_document_ids)
    selected = [document_id for document_id in requested_document_ids if document_id in allowed]
    if len(selected) != len(requested_document_ids):
        raise ValueError("document_ids include unauthorized documents")
    return selected


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
