from pathlib import Path

import pytest

from rag_core.cli import main


@pytest.mark.parametrize(
    "command",
    ("doctor", "ingest", "ingest-url", "search"),
)
def test_cli_help_uses_vector_store_language(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "configured Qdrant collection" not in output


def test_doctor_help_keeps_no_key_smoke_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["doctor", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert (
        "rag-core doctor --qdrant-location :memory: "
        "--embedding-provider demo --embedding-dimensions 64 --json"
    ) in output
    assert (
        "rag-core doctor --check-store --qdrant-location :memory: "
        "--embedding-provider demo --embedding-dimensions 64 --json"
    ) in output
    assert (
        "rag-core doctor --check-store --qdrant-location :memory: "
        "--embedding-model text-embedding-3-small --json"
    ) not in output


@pytest.mark.parametrize(
    "command",
    ("doctor", "ingest", "ingest-archive", "ingest-urls"),
)
def test_cli_help_reserves_runtime_language_for_serve(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [command, "--help"]
    if command == "ingest-urls":
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/docs\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    normalized = " ".join(output.split())
    assert "constructing the runtime" not in normalized
    assert "planned runtime shape" not in normalized
    if command != "doctor":
        assert "assembling RAGCore" in normalized


@pytest.mark.parametrize(
    "command",
    ("ingest", "ingest-archive", "ingest-url", "ingest-urls", "manifest"),
)
def test_source_cli_help_uses_one_metadata_placeholder(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--metadata KEY=VALUE" in output
    assert "--metadata METADATA" not in output


def test_cli_top_level_help_excludes_removed_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for removed in ("eval", "trace-summary"):
        assert f"  {removed} " not in f"\n{output}\n"
    assert "  local-eval " in f"\n{output}\n"
    assert "  serve " in f"\n{output}\n"


def test_public_docs_do_not_advertise_removed_cli_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/expectations.md", "docs/quickstart.md")
    )

    assert "rag-core eval" not in docs
    assert "rag-core trace-summary" not in docs


def test_retrieve_context_docs_do_not_advertise_rejected_json_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/quickstart.md",
            "src/rag_core/cli_help_examples.py",
        )
    )

    assert "retrieve-context" in text
    offenders = [
        line
        for line in text.splitlines()
        if "retrieve-context" in line and "--json" in line
    ]
    assert offenders == []
    assert "query.set_defaults(json=False)" in Path(
        "src/rag_core/cli_search_parser.py"
    ).read_text(encoding="utf-8")


def test_configured_cli_docs_keep_query_commands_on_same_store_and_model() -> None:
    docs = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/quickstart.md", "docs/providers.md")
    )

    assert "text-embedding-3-small --embedding-dimensions 1536" in docs
    assert (
        'retrieve-context "billing policy" --namespace acme --corpus-id help \\\n'
        "  --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536"
        in docs
    )
    assert (
        'retrieve-context "How can invoices be paid?" \\\n'
        "  --namespace acme --corpus-id help \\\n"
        "  --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536"
        in docs
    )
    assert (
        "uv run rag-core doctor --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536 --json"
        in docs
    )
    assert (
        'search "billing policy" --namespace acme --corpus-id help '
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small --embedding-dimensions 1536 --json"
        in docs
    )
    assert "http://localhost:6333" not in docs
    assert (
        "uv run rag-core doctor --check-store --qdrant-location :memory: \\\n"
        "  --embedding-provider demo --embedding-dimensions 64 --json" in docs
    )
    assert (
        "uv run rag-core ingest ./docs --namespace acme --corpus-id help \\\n"
        "  --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536"
        in docs
    )
    for line in docs.splitlines():
        if (
            "rag-core " in line
            and "--qdrant-url http://127.0.0.1:6333" in line
            and any(
                command in line
                for command in (" ingest ", " search ", " retrieve-context ")
            )
        ):
            assert "--embedding-provider openai" in line
            assert "--embedding-model text-embedding-3-small" in line
            assert "--embedding-dimensions 1536" in line


def test_cli_help_does_not_pair_process_local_qdrant_with_ingest_or_query() -> None:
    examples = Path("src/rag_core/cli_help_examples.py").read_text(encoding="utf-8")

    for command in ("ingest", "search", "retrieve-context"):
        offenders = [
            line
            for line in examples.splitlines()
            if f"rag-core {command} " in line and "--qdrant-location :memory:" in line
        ]
        assert offenders == []

    assert (
        "rag-core ingest ./docs --namespace acme --corpus-id help "
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small --embedding-dimensions 1536 --json"
        in examples
    )
    assert (
        'rag-core retrieve-context "billing policy" --namespace acme --corpus-id help '
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small --embedding-dimensions 1536"
        in examples
    )
    assert (
        'rag-core search "billing" --namespace acme --corpus-id help '
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small --embedding-dimensions 1536 "
        "--search-profile balanced --json" in examples
    )


def test_retrieval_default_docs_distinguish_profiles_from_lexical_sidecar() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/expectations.md",
            "docs/embed.md",
            "docs/stability.md",
            "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
        )
    )

    assert "capability-aware default query plan" in docs
    assert "Default retrieval is capability-aware" in docs
    assert "`balanced` when dense+sparse hybrid RRF is supported" in docs
    assert "capability-aware query embedding" in docs
    assert "use_lexical_search` is the request flag" in docs
    assert "configured lexical/exact-match expansion" in docs
    assert "Sparse query-plan channels and named search profiles are separate" in docs
    assert "configured sidecar/exact-match" not in docs
    assert "hybrid on" not in docs
    assert "Hybrid / lexical" not in docs
    assert "dense+sparse index into Qdrant" not in docs
    assert "run hybrid search" not in docs
    assert "provider-aware default query plan" not in docs
    assert "chunking, hybrid search" not in docs
    assert "index \u2192 hybrid search" not in docs
    assert (
        "Hybrid/lexical is on by default across library, CLI, HTTP, and tool contract"
        not in docs
    )
    assert "promise that every provider runs hybrid" in docs
    assert "The default in `rag-core`" not in docs
    assert "The lineage record" not in docs
