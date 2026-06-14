from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.search.indexer_models import IndexResult
from rag_core.search.lexical_sidecar import LexicalSidecarRecord
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.stored_payload import payload_to_result

if TYPE_CHECKING:
    from rag_core.search.provider_protocols import SearchSidecar


def sync_search_sidecar(
    *,
    sidecar: "SearchSidecar | None",
    namespace: str,
    corpus_id: str,
    document_id: str,
    result: IndexResult,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> None:
    if sidecar is None:
        return
    sidecar.delete_document(
        namespace=namespace,
        document_id=document_id,
        corpus_id=corpus_id,
    )
    sidecar.upsert_records(
        [
            LexicalSidecarRecord(
                namespace=namespace,
                result=payload_to_result(
                    point_id=point_id,
                    payload=payload,
                    score=0.0,
                    policy=policy,
                ),
            )
            for point_id, payload in zip(result.point_ids, result.point_payloads, strict=True)
        ]
    )
