from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import venv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the built rag-core wheel and run a consumer smoke app."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = args.dist_dir if args.dist_dir.is_absolute() else repo_root / args.dist_dir
    wheel = _find_wheel(dist_dir)
    temp_root = Path(tempfile.mkdtemp(prefix="rag-core-wheel-smoke-")).resolve()
    try:
        venv_dir = temp_root / "venv"
        app_dir = temp_root / "consumer-app"
        app_dir.mkdir()
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)

        python = _venv_bin(venv_dir, "python")
        _run(
            [
                str(python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                str(wheel.resolve()),
            ],
            cwd=app_dir,
        )

        app_path = app_dir / "consumer_smoke.py"
        app_path.write_text(_consumer_app(), encoding="utf-8")
        env = _clean_env()
        env["RAG_CORE_WHEEL_SMOKE_REPO_ROOT"] = str(repo_root)
        _run([str(python), str(app_path)], cwd=app_dir, env=env)
        _run([str(python), "-m", "rag_core.quickstart"], cwd=app_dir, env=env)
    finally:
        if args.keep_temp:
            print(f"kept wheel smoke directory: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def _venv_bin(venv_dir: Path, command: str) -> Path:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" and command != "pip" else ""
    return venv_dir / bin_dir / f"{command}{suffix}"


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def _consumer_app() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import asyncio
        import hashlib
        import json
        import os
        import subprocess
        import sys
        import zipfile
        from pathlib import Path

        import rag_core
        from rag_core.demo import build_demo_core
        from rag_core.fetch_security import validate_fetch_url
        from rag_core.fetching import FetchResponse


        _REMOTE_URL = "https://example.com/docs/remote-guide?token=secret"


        class StaticFetchClient:
            def fetch(self, url: str) -> FetchResponse:
                body = b"Remote guide pages can be fetched, parsed, indexed, and cited."
                validated = validate_fetch_url(url)
                return FetchResponse(
                    url=validated,
                    status_code=200,
                    content_type="text/plain",
                    content_length=len(body),
                    content_sha256=hashlib.sha256(body).hexdigest(),
                    body=body,
                    redirect_chain=(validated,),
                )


        def _is_relative_to(path: Path, parent: Path) -> bool:
            try:
                path.relative_to(parent)
            except ValueError:
                return False
            return True


        def _write_no_key_sources(app_dir: Path) -> tuple[Path, Path]:
            source_dir = app_dir / "source-smoke"
            source_dir.mkdir(exist_ok=True)
            local_file = source_dir / "local-guide.md"
            local_file.write_text(
                "# Local Guide\\n\\nLocal files can be parsed, indexed, and cited.",
                encoding="utf-8",
            )
            archive_path = source_dir / "docs.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "archive-guide.md",
                    "# Archive Guide\\n\\nZIP members are parsed, indexed, and cited.",
                )
            return local_file, archive_path


        async def _no_key_context_smoke(app_dir: Path) -> dict[str, object]:
            core = build_demo_core(collection="wheel_smoke_consumer")
            async with core:
                local_file, archive_path = _write_no_key_sources(app_dir)
                local_doc = await core.ingest_file(
                    local_file,
                    namespace="acme",
                    corpus_id="sources",
                    document_key="local-guide.md",
                )
                archive_result = await core.ingest_archive(
                    archive_path,
                    namespace="acme",
                    corpus_id="sources",
                )
                remote_doc = await core.ingest_url(
                    _REMOTE_URL,
                    namespace="acme",
                    corpus_id="sources",
                    fetch_client=StaticFetchClient(),
                )
                context = await core.retrieve_context(
                    query="Local files ZIP members remote guide pages indexed cited",
                    namespace="acme",
                    corpus_ids=["sources"],
                    limit=5,
                    rerank=False,
                    max_chars=1600,
                )

            context_text = context.as_text()
            if local_doc.chunk_count < 1:
                raise AssertionError("expected local file chunks")
            if archive_result.written_count < 1:
                raise AssertionError("expected archive member ingest")
            if remote_doc.chunk_count < 1:
                raise AssertionError("expected URL source chunks")
            if len(context.citations) < 3:
                raise AssertionError("expected retrieved context citations for all source types")
            lowered = context_text.lower()
            for term in ("local", "zip", "remote"):
                if term not in lowered:
                    raise AssertionError(f"expected retrieved context to include {term} source text")
            return {
                "local_document_key": local_doc.document_key,
                "archive_written_count": archive_result.written_count,
                "remote_document_key": remote_doc.document_key,
                "source_citation_count": len(context.citations),
            }


        async def _persistent_qdrant_smoke(app_dir: Path) -> int:
            qdrant_location = app_dir / "qdrant-data"
            collection = "wheel_smoke_persistent"
            async with build_demo_core(
                collection=collection,
                qdrant_location=str(qdrant_location),
            ) as writer:
                await writer.ingest_bytes(
                    file_bytes=b"Refund records stay searchable after the core restarts.",
                    filename="refunds.txt",
                    mime_type="text/plain",
                    namespace="acme",
                    corpus_id="help-center",
                    document_id="refunds",
                )

            async with build_demo_core(
                collection=collection,
                qdrant_location=str(qdrant_location),
            ) as reader:
                hits = await reader.search(
                    query="What stays searchable after restart?",
                    namespace="acme",
                    corpus_ids=["help-center"],
                    limit=3,
                    rerank=False,
                )
                context = await reader.retrieve_context(
                    query="What stays searchable after restart?",
                    namespace="acme",
                    corpus_ids=["help-center"],
                    limit=3,
                    rerank=False,
                    max_chars=1200,
                )

            context_text = context.as_text()
            if not hits:
                raise AssertionError("expected persistent Qdrant search hits")
            if not context.citations:
                raise AssertionError("expected persistent Qdrant citations")
            if "restart" not in context_text.lower():
                raise AssertionError("expected persistent Qdrant context after restart")
            return len(context.citations)


        async def main() -> None:
            repo_root = Path(os.environ["RAG_CORE_WHEEL_SMOKE_REPO_ROOT"]).resolve()
            imported_from = Path(rag_core.__file__).resolve()
            if _is_relative_to(imported_from, repo_root):
                raise AssertionError(f"rag_core imported from checkout: {imported_from}")

            source_smoke = await _no_key_context_smoke(Path.cwd())
            persistent_citation_count = await _persistent_qdrant_smoke(Path.cwd())

            doctor_name = "rag-core.exe" if os.name == "nt" else "rag-core"
            doctor = subprocess.run(
                [str(Path(sys.executable).parent / doctor_name), "doctor", "--json"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            doctor_payload = json.loads(doctor.stdout)
            if not isinstance(doctor_payload, dict) or "embedding" not in doctor_payload:
                raise AssertionError("expected doctor --json to emit embedding diagnostics")

            print(
                json.dumps(
                    {
                        "imported_from": str(imported_from),
                        **source_smoke,
                        "persistent_citation_count": persistent_citation_count,
                        "doctor_embedding_provider": doctor_payload["embedding"]["provider"],
                    },
                    sort_keys=True,
                )
            )


        if __name__ == "__main__":
            asyncio.run(main())
        """
    ).strip()


if __name__ == "__main__":
    main()
