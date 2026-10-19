from __future__ import annotations
import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-core",
        description="Index documents, retrieve evidence, and pack cited context.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "context", "add", "doctor", "eval", "mcp", "demo"):
        command = commands.add_parser(name)
        if name != "mcp":
            command.add_argument("--json", action="store_true")
        if name != "demo":
            command.add_argument("--events-jsonl", "--trace-jsonl", dest="events_jsonl")
        if name == "demo":
            continue
        _config_flags(command)
        if name in {"search", "context", "add", "mcp"}:
            command.add_argument("--tenant-id", default=None)
            command.add_argument("--collection", action="append", default=[])
        if name in {"search", "context"}:
            command.add_argument("text")
            command.add_argument("path", nargs="?")
            command.add_argument("--persist-dir")
            command.add_argument("--demo", action="store_true")
            command.add_argument("--limit", type=int)
            command.add_argument("--max-files", type=int)
            command.add_argument("--content-type", action="append", default=[])
            command.add_argument("--document-id", action="append", default=[])
            command.add_argument("--metadata-filter", action="append", default=[])
            command.add_argument("--mode", choices=("dense", "hybrid", "lexical"))
            command.add_argument(
                "--context-order", choices=("rank", "extrema"), default="rank"
            )
            command.add_argument("--max-context-chars", type=int)
            command.add_argument("--max-context-tokens", type=int)
            command.set_defaults(context=name == "context")
        elif name == "add":
            command.add_argument("path")
            command.add_argument("--metadata", action="append", default=[])
            command.add_argument("--force-reindex", action="store_true")
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--max-files", type=int)
            command.add_argument("--max-concurrency", type=int, default=1)
        elif name == "eval":
            command.add_argument("--demo", action="store_true")
            command.add_argument("--persist-dir")
            command.add_argument("path")
            command.add_argument("cases")
            command.add_argument("--max-files", type=int, default=200)
            command.add_argument("--max-concurrency", type=int, default=1)
            command.add_argument(
                "--mode", choices=("dense", "hybrid", "lexical"), default=None
            )
        elif name == "mcp":
            command.add_argument("--limit-cap", type=int, default=10)
    return parser


def _config_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--embedding-provider", choices=("local", "openai", "demo"))
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-location")
    parser.add_argument("--qdrant-collection")
    parser.add_argument("--qdrant-api-key", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dimension-aware-collection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--processing-version")
    parser.add_argument("--manifest-dir")
