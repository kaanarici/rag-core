from __future__ import annotations

import argparse

from rag_core.cli_config_parser import add_config_flags
from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_fetch_parser import add_fetch_flags


def add_remote_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    _add_ingest_url_command(
        subparsers,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    _add_ingest_urls_command(
        subparsers,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    _add_discover_remote_command(subparsers)


def _add_ingest_url_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    ingest_url = subparsers.add_parser(
        "ingest-url",
        help="Fetch and ingest one HTTPS URL into the configured vector store.",
    )
    ingest_url.description = (
        "Fetch and ingest one explicit HTTPS URL. Plain HTTP requires "
        "--fetch-allow-http. This command does not crawl, expand sitemaps, or "
        "batch local files."
    )
    add_config_flags(ingest_url)
    ingest_url.add_argument("url", help="HTTPS URL to fetch and ingest.")
    ingest_url.add_argument("--namespace", required=True)
    ingest_url.add_argument("--corpus-id", required=True)
    ingest_url.add_argument("--document-id")
    ingest_url.add_argument(
        "--force-reindex",
        action="store_true",
        help="Reindex the URL even when fetched content is unchanged.",
    )
    _add_manifest_dir_argument(
        ingest_url,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    ingest_url.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable metadata field applied to the ingested document.",
    )
    add_events_jsonl_flag(ingest_url)
    add_fetch_flags(ingest_url)
    ingest_url.add_argument("--json", action="store_true", help="Emit JSON output.")


def _add_ingest_urls_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    ingest_urls = subparsers.add_parser(
        "ingest-urls",
        help="Fetch and ingest explicit HTTPS URLs from a text file.",
    )
    ingest_urls.description = (
        "Fetch and ingest an explicit list of HTTPS URLs, one URL per line. "
        "Plain HTTP requires --fetch-allow-http. Blank lines and full-line "
        "comments are ignored. This command does not crawl, discover, or recurse."
    )
    add_config_flags(ingest_urls)
    ingest_urls.add_argument(
        "url_file", help="Text file with one HTTP(S) URL per line."
    )
    ingest_urls.add_argument("--namespace", required=True)
    ingest_urls.add_argument("--corpus-id", required=True)
    ingest_urls.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum URL ingests to run concurrently.",
    )
    ingest_urls.add_argument(
        "--plan-json",
        action="store_true",
        help=(
            "Print the redacted URL source-item plan, then exit without "
            "constructing the runtime or fetching URLs."
        ),
    )
    _add_manifest_dir_argument(
        ingest_urls,
        manifest_dir_default=manifest_dir_default,
        manifest_dir_help_default=manifest_dir_help_default,
    )
    ingest_urls.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Attach metadata as KEY=VALUE. Repeat for multiple fields.",
    )
    ingest_urls.add_argument(
        "--force-reindex",
        action="store_true",
        help="Reindex every URL even when fetched content is unchanged.",
    )
    add_events_jsonl_flag(ingest_urls)
    add_fetch_flags(ingest_urls)
    ingest_urls.add_argument("--json", action="store_true", help="Emit JSONL output.")


def _add_discover_remote_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    discover_remote = subparsers.add_parser(
        "discover-remote",
        help="Fetch one sitemap or llms.txt artifact and print discovered URLs.",
    )
    discover_remote.description = (
        "Fetch one explicit remote discovery artifact and print discovered HTTPS "
        "URLs. Plain HTTP requires --fetch-allow-http. Sitemap indexes are "
        "expanded into page URLs. This command does not fetch discovered page "
        "URLs, crawl, or ingest."
    )
    discover_remote.add_argument("url", help="HTTPS sitemap or llms.txt URL.")
    discover_remote.add_argument(
        "--kind",
        choices=("sitemap", "llms-txt"),
        required=True,
        help="Discovery artifact format.",
    )
    discover_remote.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Maximum discovered URLs to accept before failing.",
    )
    discover_remote.add_argument(
        "--max-sitemap-fetches",
        type=int,
        default=None,
        help="Maximum nested sitemap files to fetch while expanding a sitemap index.",
    )
    discover_remote.add_argument(
        "--output-url-file",
        help=(
            "Write redacted discovered URLs without query strings, one per line. "
            "Query-bearing URLs require --output-url-file-raw-queries for ingest-urls. "
            "Refuses to overwrite an existing file."
        ),
    )
    discover_remote.add_argument(
        "--output-url-file-raw-queries",
        action="store_true",
        help=(
            "Write raw query strings to --output-url-file. This can preserve "
            "query-distinct sources but may persist secrets from discovered URLs."
        ),
    )
    add_fetch_flags(discover_remote)
    discover_remote.add_argument(
        "--json", action="store_true", help="Emit JSON output."
    )


def _add_manifest_dir_argument(
    parser: argparse.ArgumentParser,
    *,
    manifest_dir_default: str,
    manifest_dir_help_default: str,
) -> None:
    parser.add_argument(
        "--manifest-dir",
        default=manifest_dir_default,
        help=(
            "Directory under which JSONL manifests are written. "
            f"Default: {manifest_dir_help_default}"
        ),
    )


__all__ = ["add_remote_commands"]
