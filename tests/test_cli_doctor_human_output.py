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
    assert (
        "  - turbopuffer: support=first_party_optional "
        "query_plan=dense"
    ) in output
    assert (
        "  - memory: support=first_party_utility "
        "query_plan=dense,sparse,hybrid_rrf"
    ) in output


def test_doctor_human_output_guides_unconfigured_first_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["doctor"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Next Steps:" in output
    assert (
        'rag-core local-search examples/demo_corpus "How can invoices be paid?"'
        in output
    )
    assert "`OPENAI_API_KEY`" in output
    assert "--embedding-provider demo --embedding-dimensions 64" in output
    assert "--qdrant-location :memory:" in output
    assert "--qdrant-url http://127.0.0.1:6333" in output


def test_doctor_human_output_skips_next_steps_for_ready_no_key_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "doctor",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "64",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Next Steps:" not in output
    assert "OPENAI_API_KEY" not in output


def test_doctor_human_output_reports_provider_health_when_checked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "doctor",
            "--check-store",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "64",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Store Health: healthy=True" in output
    assert "Embedding Health: healthy=True provider=demo model=demo-dense-v1" in output
    assert "Reranker Health: healthy=True provider=none model=none" in output
