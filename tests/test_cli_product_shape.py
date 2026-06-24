from pathlib import Path

import pytest

from rag_core.cli import main


@pytest.mark.parametrize(
    "command",
    ("doctor", "add", "search", "context"),
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
    ("doctor", "add"),
)
def test_cli_help_reserves_runtime_language_for_serve(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [command, "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    normalized = " ".join(output.split())
    assert "constructing the runtime" not in normalized
    assert "planned runtime shape" not in normalized
    if command != "doctor":
        assert "assembling Engine" in normalized


@pytest.mark.parametrize(
    "command",
    ("add", "manifest"),
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


def test_cli_top_level_help_lists_primary_commands_and_aliases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    removed_commands = (
        "trace-summary",
        "ingest-" "archive",
        "ingest-" "url",
        "ingest-" "urls",
        "manifest-" "compact",
    )
    for removed in removed_commands:
        assert f"  {removed} " not in f"\n{output}\n"
    assert 'rag-core context "<question>" <folder>' in output
    assert "Advanced configured commands:" in output
    assert "add <source>|--url-list <file.txt>" in output
    assert 'search "<query>" [path]' in output
    assert 'context "<query>" [path]' in output
    assert "eval" in output
    assert "Deprecated aliases:" in output
    assert "ingest -> add" in output
    assert "local-search -> search" in output
    assert "local-eval -> eval" in output
    assert "  serve " in f"\n{output}\n"


def test_public_docs_do_not_advertise_removed_cli_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    docs_dir = root / "docs-site" / "content" / "docs"
    docs = "\n".join(
        text
        for text in (
            (root / "README.md").read_text(encoding="utf-8"),
            *(path.read_text(encoding="utf-8") for path in docs_dir.glob("*.mdx")),
        )
    )

    assert "rag-core trace-summary" not in docs


def test_context_docs_advertise_search_context_not_deleted_command() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "src/rag_core/cli/help_examples.py",
        )
    )

    assert "context" in text
    assert "retrieve-" "context" not in text


def test_configured_cli_docs_keep_query_commands_on_same_store_and_known_model() -> None:
    docs = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "src/rag_core/cli/help_examples.py",
        )
    )

    assert "text-embedding-3-small --embedding-dimensions 1536" not in docs
    assert "custom or unknown provider/model pairs" in docs
    assert (
        'context "billing policy" --collection help \\\n'
        "  --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small"
        in docs
    )
    assert "Known local and OpenAI models infer their dimensions" in docs
    assert (
        "rag-core doctor --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small --json"
        in docs
    )
    assert (
        'search "billing policy" --collection help '
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small --json"
        in docs
    )
    assert "http://localhost:6333" not in docs
    assert (
        "rag-core doctor --check-store --qdrant-location :memory: "
        "--embedding-provider demo --embedding-dimensions 64 --json" in docs
    )
    assert (
        "rag-core add ./docs --collection help \\\n"
        "  --qdrant-url http://127.0.0.1:6333 \\\n"
        "  --embedding-provider openai --embedding-model text-embedding-3-small"
        in docs
    )
    for line in docs.splitlines():
        if (
            "rag-core " in line
            and "--qdrant-url http://127.0.0.1:6333" in line
            and any(
                command in line
                for command in (" add ", " search ", " context ")
            )
        ):
            assert "--embedding-provider openai" in line
            assert "--embedding-model text-embedding-3-small" in line
            assert "--embedding-dimensions 1536" not in line


def test_cli_help_does_not_pair_process_local_qdrant_with_ingest_or_query() -> None:
    examples = Path("src/rag_core/cli/help_examples.py").read_text(encoding="utf-8")

    for command in ("add", "search", "context"):
        offenders = [
            line
            for line in examples.splitlines()
            if f"rag-core {command} " in line and "--qdrant-location :memory:" in line
        ]
        assert offenders == []

    assert (
        "rag-core add ./docs --collection help "
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small --json"
        in examples
    )
    assert (
        'rag-core context "billing policy" --collection help '
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small"
        in examples
    )
    assert (
        'rag-core search "billing" --collections help,policies '
        "--qdrant-url http://127.0.0.1:6333 --embedding-provider openai "
        "--embedding-model text-embedding-3-small "
        "--search-profile balanced --json" in examples
    )


def test_retrieval_default_docs_distinguish_profiles_from_lexical_sidecar() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs-site/content/docs/search.mdx",
            "docs-site/content/docs/how-retrieval-works.mdx",
            "docs-site/content/docs/rag-modes.mdx",
            "docs-site/content/docs/stability.mdx",
            "src/rag_core/search/pipeline/stages/hybrid_retrieve.py",
        )
    )

    normalized = " ".join(docs.split())

    # Default retrieval is described as capability-aware, picking balanced hybrid
    # RRF only when the store supports it.
    assert "Default retrieval is capability-aware" in docs
    assert "balanced" in docs
    assert "hybrid RRF" in docs

    # `use_lexical_search` is documented as a request-level lexical/exact-match
    # sidecar flag, kept distinct from query plans and named search profiles.
    assert "`use_lexical_search` is the request flag" in docs
    assert "exact-match sidecar" in docs
    assert "not a query-plan or search-profile selector" in normalized
    assert (
        "different thing from the `use_lexical_search` flag" in normalized
        or "not the portable exact-match sidecar" in normalized
    )

    # Guardrail: do not overclaim that hybrid is universally on; capability-aware
    # planning is not a promise that every provider runs hybrid.
    assert "promise that every provider runs hybrid" in docs
    assert (
        "Hybrid/lexical is on by default across library, CLI, HTTP, and tool contract"
        not in docs
    )
    assert "run hybrid search" not in docs
