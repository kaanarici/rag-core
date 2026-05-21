from __future__ import annotations

from rag_core.cli import _build_parser


def test_ingest_subparser_accepts_max_concurrency() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "ingest",
            "./docs",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--max-concurrency",
            "3",
        ]
    )

    assert args.command == "ingest"
    assert args.max_concurrency == 3
