from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from urllib.error import URLError
from urllib.request import urlopen
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
        _run(
            [
                str(python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                _wheel_requirement(wheel.resolve(), extras=("runtime",)),
            ],
            cwd=app_dir,
        )
        _installed_runtime_smoke(venv_dir, app_dir, repo_root, env=env)
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


def _wheel_requirement(wheel: Path, *, extras: tuple[str, ...] = ()) -> str:
    extra = f"[{','.join(extras)}]" if extras else ""
    return f"rag-core{extra} @ {wheel.as_uri()}"


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _installed_runtime_smoke(
    venv_dir: Path,
    app_dir: Path,
    repo_root: Path,
    *,
    env: dict[str, str],
) -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    stdout_path = app_dir / "serve.stdout.log"
    stderr_path = app_dir / "serve.stderr.log"
    command = [
        str(_venv_bin(venv_dir, "rag-core")),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ingest-root",
        str(repo_root),
        "--job-db-path",
        str(app_dir / "runtime-jobs.sqlite3"),
        "--qdrant-location",
        ":memory:",
        "--embedding-provider",
        "demo",
        "--embedding-model",
        "demo-dense-v1",
        "--embedding-dimensions",
        "64",
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        server = subprocess.Popen(
            command,
            cwd=app_dir,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

    try:
        _wait_for_ready(server, base_url, stdout_path, stderr_path)
        smoke_env = env.copy()
        smoke_env["BASE_URL"] = base_url
        smoke_env["INGEST_PATH"] = str(repo_root / "examples" / "demo_corpus" / "billing.md")
        _run(
            [str(repo_root / "scripts" / "self_host_smoke.sh")],
            cwd=repo_root,
            env=smoke_env,
        )
        print("installed runtime extra smoke passed")
    finally:
        _stop_server(server)


def _wait_for_ready(
    server: subprocess.Popen[str],
    base_url: str,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    for _ in range(60):
        if server.poll() is not None:
            _print_server_logs(stdout_path, stderr_path)
            raise SystemExit(f"installed rag-core serve exited early: {server.returncode}")
        try:
            with urlopen(f"{base_url}/health/ready", timeout=1) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(0.5)
    _print_server_logs(stdout_path, stderr_path)
    raise SystemExit("installed rag-core serve did not become ready")


def _stop_server(server: subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)


def _print_server_logs(stdout_path: Path, stderr_path: Path) -> None:
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if stdout:
        print(stdout, file=sys.stdout)
    if stderr:
        print(stderr, file=sys.stderr)


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
        import rag_core.integrations as integrations
        from rag_core.demo import build_demo_core
        from rag_core.evals import (
            add_quality_gate,
            eval_exit_code,
            eval_report,
            load_cases,
            run_eval,
        )
        from rag_core.fetch_security import validate_fetch_url
        from rag_core.fetching import FetchResponse
        from rag_core.search import search_profile


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


        def _console_script_path(command: str) -> Path:
            suffix = ".exe" if os.name == "nt" else ""
            return Path(sys.executable).parent / f"{command}{suffix}"


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

            context_text = context.as_prompt_text()
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

            context_text = context.as_prompt_text()
            if not hits:
                raise AssertionError("expected persistent Qdrant search hits")
            if not context.citations:
                raise AssertionError("expected persistent Qdrant citations")
            if "restart" not in context_text.lower():
                raise AssertionError("expected persistent Qdrant context after restart")
            return len(context.citations)


        def _integration_import_smoke() -> dict[str, object]:
            expected_exports = (
                "LangChainNotInstalledError",
                "LangChainRetrieverConfig",
                "build_langchain_retriever",
                "build_retrieve_context_tool",
                "create_langchain_context_tool",
                "create_langchain_retriever_tool",
                "langchain",
                "openai_agents",
            )
            if integrations.__all__ != expected_exports:
                raise AssertionError(
                    f"unexpected integration exports: {integrations.__all__!r}"
                )
            if integrations.langchain.__name__ != "rag_core.integrations.langchain":
                raise AssertionError("expected lazy langchain module export")
            if integrations.openai_agents.__name__ != "rag_core.integrations.openai_agents":
                raise AssertionError("expected lazy openai_agents module export")

            try:
                integrations.build_langchain_retriever(
                    object(),
                    namespace="acme",
                    corpus_ids=["help-center"],
                )
            except integrations.LangChainNotInstalledError as exc:
                if "langchain" not in str(exc).lower():
                    raise AssertionError(f"unexpected LangChain error: {exc}") from exc
            else:
                raise AssertionError("expected LangChain extra to be required")

            class _RetrieveContextCore:
                async def retrieve_context(self, **kwargs: object) -> object:
                    del kwargs
                    raise AssertionError("openai-agents builder should fail before retrieval")

            try:
                integrations.build_retrieve_context_tool(
                    _RetrieveContextCore(),
                    namespace="acme",
                    corpus_ids=["help-center"],
                )
            except ImportError as exc:
                if "openai-agents" not in str(exc):
                    raise AssertionError(f"unexpected OpenAI Agents error: {exc}") from exc
            else:
                raise AssertionError("expected OpenAI Agents extra to be required")

            return {
                "integration_export_count": len(expected_exports),
                "integration_optional_extra_errors": ["langchain", "openai-agents"],
            }


        async def _eval_smoke(app_dir: Path) -> dict[str, object]:
            cases_path = app_dir / "eval_cases.jsonl"
            cases_path.write_text(
                json.dumps(
                    {
                        "case_id": "wheel/billing",
                        "query": "how can invoices be paid",
                        "namespace": "acme",
                        "corpus_ids": ["help-center"],
                        "expected_ids": ["billing-eval"],
                    }
                )
                + "\\n",
                encoding="utf-8",
            )
            async with build_demo_core(collection="wheel_smoke_eval") as core:
                await core.ingest_bytes(
                    file_bytes=b"Billing invoices can be paid by card or ACH.",
                    filename="billing.md",
                    mime_type="text/markdown",
                    namespace="acme",
                    corpus_id="help-center",
                    document_id="billing-eval",
                )
                results = await run_eval(
                    core,
                    load_cases(cases_path),
                    query_plan=search_profile("balanced", limit=5),
                )

            report = eval_report(results, run={"mode": "wheel_smoke"})
            add_quality_gate(
                report,
                {"eval": report},
                {"recall_at_5": {"minimum": 1.0}, "mrr": {"minimum": 1.0}},
            )
            if eval_exit_code(report) != 0:
                raise AssertionError(f"expected wheel eval smoke to pass: {report}")
            gate = report.get("quality_gate")
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise AssertionError(f"expected passing quality gate: {report}")
            return {
                "eval_case_count": report["case_count"],
                "eval_quality_gate_passed": gate["passed"],
            }


        def _installed_cli_smoke(app_dir: Path) -> dict[str, object]:
            cli = _console_script_path("rag-core")
            doctor = subprocess.run(
                [str(cli), "doctor", "--json"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            doctor_payload = json.loads(doctor.stdout)
            if not isinstance(doctor_payload, dict) or "embedding" not in doctor_payload:
                raise AssertionError("expected doctor --json to emit embedding diagnostics")

            local_search = subprocess.run(
                [
                    str(cli),
                    "local-search",
                    str(app_dir / "source-smoke"),
                    "Local files parsed indexed cited",
                    "--json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            search_payload = json.loads(local_search.stdout)
            hits = search_payload.get("hits")
            if not isinstance(hits, list) or not hits:
                raise AssertionError("expected installed local-search CLI hits")
            if search_payload.get("indexed_count", 0) < 1:
                raise AssertionError("expected installed local-search CLI to index files")

            local_eval_cases = app_dir / "local_eval_cases.jsonl"
            local_eval_cases.write_text(
                json.dumps(
                    {
                        "case_id": "wheel/local-eval",
                        "query": "how are local files parsed indexed cited",
                        "namespace": "acme",
                        "corpus_ids": ["sources"],
                        "expected_ids": ["local-guide.md"],
                    }
                )
                + "\\n",
                encoding="utf-8",
            )
            local_eval = subprocess.run(
                [
                    str(cli),
                    "local-eval",
                    str(app_dir / "source-smoke"),
                    str(local_eval_cases),
                    "--min-recall-at-5",
                    "1",
                    "--min-mrr",
                    "1",
                    "--json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            eval_payload = json.loads(local_eval.stdout)
            gate = eval_payload.get("quality_gate")
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise AssertionError("expected installed local-eval quality gate to pass")

            return {
                "doctor_embedding_provider": doctor_payload["embedding"]["provider"],
                "installed_cli_local_search_hits": len(hits),
                "installed_cli_local_eval_cases": eval_payload["case_count"],
            }


        async def main() -> None:
            repo_root = Path(os.environ["RAG_CORE_WHEEL_SMOKE_REPO_ROOT"]).resolve()
            imported_from = Path(rag_core.__file__).resolve()
            if _is_relative_to(imported_from, repo_root):
                raise AssertionError(f"rag_core imported from checkout: {imported_from}")

            source_smoke = await _no_key_context_smoke(Path.cwd())
            persistent_citation_count = await _persistent_qdrant_smoke(Path.cwd())
            integration_smoke = _integration_import_smoke()
            eval_smoke = await _eval_smoke(Path.cwd())
            cli_smoke = _installed_cli_smoke(Path.cwd())

            print(
                json.dumps(
                    {
                        "imported_from": str(imported_from),
                        **source_smoke,
                        **integration_smoke,
                        **eval_smoke,
                        **cli_smoke,
                        "persistent_citation_count": persistent_citation_count,
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
