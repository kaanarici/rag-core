from __future__ import annotations

import pytest

from rag_core.cli import main


def test_doctor_human_output_summarizes_provider_query_plan_support(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "doctor",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Vector Store Providers:" in output
    assert (
        "  * qdrant: support=default "
        "query_plan=dense,sparse,hybrid_rrf,hybrid_dbsf,"
        "hybrid_weighted_rrf,mmr,nested_prefetch,boost"
    ) in output
    assert "  - turbopuffer: support=first_party_optional query_plan=dense" in output
