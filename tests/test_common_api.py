from __future__ import annotations

import asyncio
import uuid

import pytest

from rag_core import (
    Config,
    Document,
    Engine,
    Scope,
    SearchOptions,
)
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.demo import DemoEmbeddingProvider
from rag_core.evals import EvalCase, run_eval
from rag_core.search.query_plan import UnsupportedQueryStage

pytestmark = pytest.mark.integration


def test_common_path_is_dense_idempotent_and_duplicate_aware() -> None:
    async def scenario() -> None:
        scope = Scope(tenant_id="school", collection="economics")
        config = Config(
            qdrant=QdrantConfig(
                location=":memory:",
                store_collection=f"common_api_{uuid.uuid4().hex}",
                dimension_aware_collection=False,
            ),
            embedding=EmbeddingConfig(
                provider="demo",
                model="demo-dense-v1",
                dimensions=64,
            ),
        )
        async with Engine(config, embedding=DemoEmbeddingProvider()) as core:
            copied_text = (
                b"Spot exchange uses the current rate. A forward exchange "
                b"contract fixes a rate for a future date."
            )
            first = await core.ingest(
                Document(
                    id="copy-a",
                    key="current/chapter-11.txt",
                    content=copied_text,
                    content_type="text/plain",
                ),
                scope=scope,
            )
            second = await core.ingest(
                Document(
                    id="copy-b",
                    key="archive/chapter-11.txt",
                    content=copied_text,
                    content_type="text/plain",
                ),
                scope=scope,
            )
            unchanged = await core.ingest(
                Document(
                    id="copy-a",
                    key="current/chapter-11.txt",
                    content=copied_text,
                    content_type="text/plain",
                ),
                scope=scope,
            )
            await core.ingest(
                Document(
                    id="policy",
                    key="policy/tuition.txt",
                    content=b"Tuition invoices are paid by bank transfer.",
                    content_type="text/plain",
                ),
                scope=scope,
            )

            result = await core.search(
                "How do spot and forward exchange rates differ?",
                scope=scope,
                limit=3,
            )

            assert core._sparse is None
            assert core._store.capabilities.query_plan.dense is True
            assert core._store.capabilities.query_plan.sparse is False
            assert first.status == "created"
            assert second.status == "created"
            assert unchanged.status == "unchanged"
            assert unchanged.content_hash == first.content_hash
            assert result.answerability.status == "unknown"
            assert result.answerability.reason == "not_calibrated"
            assert result.answerability.calibration is None
            assert result.diagnostics["mode"] == "dense"
            assert result.diagnostics["duplicate_count"] == 1
            copied = next(
                item for item in result.evidence if item.document_id in {"copy-a", "copy-b"}
            )
            assert copied.locator["chunk_index"] == 0
            assert {source["document_id"] for source in copied.equivalent_sources} == {
                "copy-a",
                "copy-b",
            } - {copied.document_id}

            [eval_result] = await run_eval(
                core,
                [
                    EvalCase(
                        query="spot and forward exchange",
                        namespace=scope.tenant_id,
                        collections=(scope.collection,),
                        expected_ids=("copy-a", "copy-b"),
                    )
                ],
            )
            assert eval_result.retrieval_mode == "dense"
            assert eval_result.answerability_status == "unknown"
            assert eval_result.suppressed_duplicate_count == 1
            assert eval_result.duplicate_result_rate > 0
            assert eval_result.unique_document_count == len(
                set(eval_result.retrieved_ids)
            )

            calibrated = await core.search(
                "spot and forward exchange",
                scope=scope,
                limit=1,
                options=SearchOptions(
                    answerability_threshold=-1.0,
                    answerability_calibration="demo-dense-v1/economics-v1",
                ),
            )
            assert calibrated.answerability.status == "sufficient"
            assert (
                calibrated.answerability.calibration
                == "demo-dense-v1/economics-v1"
            )

            comparison = await core.search(
                "spot and forward exchange",
                scope=Scope(
                    tenant_id="school",
                    collection="economics",
                    document_ids=("copy-a", "copy-b"),
                ),
                limit=2,
                options=SearchOptions(
                    duplicate_policy="preserve",
                    max_results_per_document=None,
                ),
            )
            assert {item.document_id for item in comparison.evidence} == {
                "copy-a",
                "copy-b",
            }

            with pytest.raises(
                UnsupportedQueryStage,
                match="does not support hybrid RRF",
            ):
                await core.search(
                    "spot and forward exchange",
                    scope=scope,
                    options=SearchOptions(mode="hybrid"),
                )

            deleted = await core.delete("copy-b", scope=scope)
            assert deleted.vector_store_acked is True
            after_delete = await core.search(
                "spot and forward exchange",
                scope=Scope(
                    tenant_id="school",
                    collection="economics",
                    document_ids=("copy-b",),
                ),
            )
            assert after_delete.evidence == ()

    asyncio.run(scenario())
