from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib


@dataclass(frozen=True)
class Probe:
    summary: str
    code: str


PROBES: dict[str, Probe] = {
    "semantic": Probe(
        "imports magika/sentence-transformers/tree-sitter and constructs semantic/code chunkers",
        """
        import importlib

        importlib.import_module("magika")
        importlib.import_module("sentence_transformers")
        importlib.import_module("tree_sitter_language_pack")

        from rag_core.documents.chunking.code import CodeChunker
        from rag_core.documents.chunking.code_runtime import tree_sitter_runtime_available
        from rag_core.documents.chunking.protocol import ChunkConfig
        from rag_core.documents.chunking.semantic import SemanticChunker

        if not tree_sitter_runtime_available():
            raise AssertionError("expected tree-sitter runtime to be available")

        code = "def alpha():\\n    return 1\\n\\ndef beta():\\n    return 2\\n"
        code_chunks = CodeChunker(
            language="python",
            enable_magika_detection=False,
        ).chunk(code, ChunkConfig(max_chars=80, overlap=0, strategy="code"))
        if not code_chunks:
            raise AssertionError("expected code chunker construction to produce chunks")

        chunks = SemanticChunker(enable_local_model=False).chunk(
            "Alpha sentence. Beta sentence.",
            ChunkConfig(max_chars=80, overlap=0, strategy="semantic"),
        )
        if not chunks:
            raise AssertionError("expected semantic chunker construction to produce chunks")

        result = {
            "extra": "semantic",
            "code_chunks": len(code_chunks),
            "semantic_chunks": len(chunks),
        }
        """,
    ),
    "html": Probe(
        "imports html-to-markdown and constructs the HTML converter surface",
        """
        from html_to_markdown import convert

        from rag_core.documents.converters.html_converter import HtmlConverter

        converted = convert(
            "<main><h1>Billing</h1><p>Invoices can be paid by ACH.</p></main>"
        )
        content = getattr(converted, "content", converted)
        if not isinstance(content, str) or "Billing" not in content or "ACH" not in content:
            raise AssertionError(f"expected html-to-markdown content, got {content!r}")

        converter = HtmlConverter()
        if converter.format_name != "html":
            raise AssertionError(f"unexpected converter {converter.format_name!r}")

        result = {"extra": "html", "converted_chars": len(content)}
        """,
    ),
    "pdf": Probe(
        "imports pdf-inspector and confirms the in-process PDF adapter is selected",
        """
        import asyncio
        import importlib

        pdf_inspector = importlib.import_module("pdf_inspector")

        from rag_core.documents.converters.pdf_converter import PdfConverter

        pdf_path = repo_root / "tests" / "fixtures" / "real_documents" / "apache_tika" / "testPDF.pdf"
        pdf_bytes = pdf_path.read_bytes()
        raw = pdf_inspector.process_pdf_bytes(pdf_bytes)
        if not getattr(raw, "markdown", "").strip():
            raise AssertionError("expected pdf-inspector wheel to return markdown")

        converted = asyncio.run(
            PdfConverter().convert(pdf_bytes, "testPDF.pdf", "application/pdf")
        )
        if converted.metadata.get("inspector_adapter") != "wheel":
            raise AssertionError(f"expected wheel adapter, got {converted.metadata!r}")
        if not converted.content.strip():
            raise AssertionError("expected converted PDF markdown")

        result = {
            "extra": "pdf",
            "pdf_type": getattr(raw, "pdf_type", None),
            "markdown_chars": len(converted.content),
        }
        """,
    ),
    "rerank": Probe(
        "imports cohere and constructs Cohere embedding/rerank adapters with a fake local key",
        """
        import importlib

        importlib.import_module("cohere")

        from rag_core.search.providers.cohere import (
            CohereEmbeddingProvider,
            CohereReranker,
        )

        embedder = CohereEmbeddingProvider(api_key="test-key")
        reranker = CohereReranker(api_key="test-key")
        if embedder.provider_name != "cohere" or reranker.provider_name != "cohere":
            raise AssertionError("expected Cohere provider adapters")

        result = {
            "extra": "rerank",
            "embedding_model": embedder.model_name,
            "reranker_model": reranker.model_name,
        }
        """,
    ),
    "voyage": Probe(
        "imports voyageai and constructs Voyage embedding/rerank adapters with a fake local key",
        """
        import importlib

        importlib.import_module("voyageai")

        from rag_core.search.providers.voyage import (
            VoyageEmbeddingProvider,
            VoyageReranker,
        )

        embedder = VoyageEmbeddingProvider(api_key="test-key", dimensions=1024)
        reranker = VoyageReranker(api_key="test-key")
        if embedder.provider_name != "voyage" or reranker.provider_name != "voyage":
            raise AssertionError("expected Voyage provider adapters")

        result = {
            "extra": "voyage",
            "embedding_model": embedder.model_name,
            "reranker_model": reranker.model_name,
        }
        """,
    ),
    "zeroentropy": Probe(
        "imports zeroentropy and constructs ZeroEntropy embedding/rerank adapters with a fake local key",
        """
        import importlib

        importlib.import_module("zeroentropy")

        from rag_core.search.providers.zeroentropy import (
            ZeroEntropyEmbeddingProvider,
            ZeroEntropyReranker,
        )

        embedder = ZeroEntropyEmbeddingProvider(api_key="test-key", dimensions=2560)
        reranker = ZeroEntropyReranker(api_key="test-key")
        if embedder.provider_name != "zeroentropy" or reranker.provider_name != "zeroentropy":
            raise AssertionError("expected ZeroEntropy provider adapters")

        result = {
            "extra": "zeroentropy",
            "embedding_model": embedder.model_name,
            "reranker_model": reranker.model_name,
        }
        """,
    ),
    "opentelemetry": Probe(
        "constructs OpenTelemetrySink and emits a local span event",
        """
        from rag_core.events import OpenTelemetrySink, SearchPlanned

        sink = OpenTelemetrySink()
        sink.emit(SearchPlanned(namespace="acme", collections=("docs",), limit=3))
        if sink.provider_name != "opentelemetry":
            raise AssertionError(f"unexpected sink provider {sink.provider_name!r}")

        result = {
            "extra": "opentelemetry",
            "provider": sink.provider_name,
            "failure_count": sink.failure_count,
        }
        """,
    ),
    "anthropic": Probe(
        "constructs AnthropicChunkContextualizer with a fake local key",
        """
        from rag_core.documents.contextualizer_adapters import AnthropicChunkContextualizer

        contextualizer = AnthropicChunkContextualizer(api_key="test-key")
        if not contextualizer.contextualizer_id.startswith("anthropic:"):
            raise AssertionError(
                f"unexpected contextualizer id {contextualizer.contextualizer_id!r}"
            )

        result = {
            "extra": "anthropic",
            "contextualizer_id": contextualizer.contextualizer_id,
        }
        """,
    ),
    "langchain": Probe(
        "constructs LangChain retriever and context-tool adapters",
        """
        from rag_core.integrations.langchain import (
            build_langchain_retriever,
            create_langchain_context_tool,
        )

        class Core:
            async def search(self, **kwargs):
                raise AssertionError("probe should not call search")

            async def context(self, **kwargs):
                raise AssertionError("probe should not call retrieve_context")

        core = Core()
        retriever = build_langchain_retriever(
            core,
            namespace="acme",
            collections=["docs"],
            rerank=False,
        )
        tool = create_langchain_context_tool(
            core,
            namespace="acme",
            collections=["docs"],
            rerank=False,
        )
        if retriever is None or tool is None:
            raise AssertionError("expected LangChain adapter objects")

        result = {
            "extra": "langchain",
            "retriever_type": type(retriever).__name__,
            "tool_type": type(tool).__name__,
        }
        """,
    ),
    "openai-agents": Probe(
        "constructs the OpenAI Agents retrieval tool adapter",
        """
        from rag_core.integrations.openai_agents import build_retrieve_context_tool

        class Core:
            async def context(self, **kwargs):
                raise AssertionError("probe should not call retrieve_context")

        tool = build_retrieve_context_tool(
            Core(),
            namespace="acme",
            collections=["docs"],
            tool_name="search_docs",
        )
        if tool is None:
            raise AssertionError("expected OpenAI Agents tool object")

        result = {"extra": "openai-agents", "tool_type": type(tool).__name__}
        """,
    ),
    "mcp": Probe(
        "constructs the MCP server adapter and lists tools without stdio",
        """
        import asyncio

        from mcp import types

        from rag_core.integrations.mcp_server import build_mcp_server

        class Core:
            async def search(self, **kwargs):
                del kwargs
                raise AssertionError("probe should not call search")

            async def context(self, **kwargs):
                del kwargs
                raise AssertionError("probe should not call retrieve_context")

        server = build_mcp_server(
            Core(),
            namespace="acme",
            collections=["docs"],
        )

        async def list_tools():
            response = await server.request_handlers[types.ListToolsRequest](
                types.ListToolsRequest()
            )
            return response.root.tools

        tools = asyncio.run(list_tools())
        names = [tool.name for tool in tools]
        if names != ["search_user_documents"]:
            raise AssertionError(f"unexpected MCP tools: {names!r}")

        result = {"extra": "mcp", "tools": names}
        """,
    ),
    "turbopuffer": Probe(
        "imports turbopuffer and constructs TurboPufferVectorStore with a local fake namespace",
        """
        import importlib

        turbopuffer = importlib.import_module("turbopuffer")
        if not callable(getattr(turbopuffer, "AsyncTurbopuffer", None)):
            raise AssertionError("expected turbopuffer.AsyncTurbopuffer")

        from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore

        class Namespace:
            async def metadata(self):
                return {"ok": True}

            async def write(self, **kwargs):
                return None

            async def query(self, **kwargs):
                return []

        store = TurboPufferVectorStore(
            namespace="rag-core-extra-smoke",
            dense_dimensions=3,
            namespace_client=Namespace(),
        )
        capabilities = store.capabilities
        if capabilities.dense_vector_dimensions != 3:
            raise AssertionError(f"unexpected dimensions {capabilities!r}")

        result = {
            "extra": "turbopuffer",
            "dense_dimensions": capabilities.dense_vector_dimensions,
        }
        """,
    ),
    "pgvector": Probe(
        "imports asyncpg/pgvector and constructs PgVectorVectorStore with a local fake pool",
        """
        import importlib

        asyncpg = importlib.import_module("asyncpg")
        pgvector_asyncpg = importlib.import_module("pgvector.asyncpg")
        if not callable(getattr(asyncpg, "create_pool", None)):
            raise AssertionError("expected asyncpg.create_pool")
        if not callable(getattr(pgvector_asyncpg, "register_vector", None)):
            raise AssertionError("expected pgvector.asyncpg.register_vector")

        from rag_core.search.providers.pgvector_store import PgVectorVectorStore

        class Acquire:
            async def __aenter__(self):
                return Connection()

            async def __aexit__(self, *exc):
                return None

        class Connection:
            async def execute(self, query, *args):
                return "OK"

            async def executemany(self, command, args):
                return "OK"

            async def fetch(self, query, *args):
                return []

            async def fetchrow(self, query, *args):
                return None

            async def fetchval(self, query, *args):
                return True if "pg_extension" in query else 0

        class Pool:
            def acquire(self):
                return Acquire()

            async def close(self):
                return None

        store = PgVectorVectorStore(
            table_name="rag_core_extra_smoke",
            dense_dimensions=3,
            pool=Pool(),
        )
        capabilities = store.capabilities
        if capabilities.dense_vector_dimensions != 3:
            raise AssertionError(f"unexpected dimensions {capabilities!r}")

        result = {
            "extra": "pgvector",
            "dense_dimensions": capabilities.dense_vector_dimensions,
        }
        """,
    ),
}

SKIPPED_EXTRAS = {
    "runtime": "runtime extra is already covered by scripts/wheel_smoke.py",
}


class SmokeFailure(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the built rag-core wheel with declared extras and probe adapters."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--extras",
        nargs="+",
        help="Extras to test, separated by spaces or commas. Default: all pyproject extras.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pyproject_extras = _read_pyproject_extras(repo_root / "pyproject.toml")
    _ensure_probe_coverage(pyproject_extras)
    requested_extras = _requested_extras(args.extras, pyproject_extras)

    dist_dir = args.dist_dir if args.dist_dir.is_absolute() else repo_root / args.dist_dir
    wheel = _find_or_build_wheel(dist_dir, repo_root=repo_root)
    temp_root = Path(tempfile.mkdtemp(prefix="rag-core-wheel-extra-smoke-")).resolve()
    failures: list[str] = []

    print(f"wheel extra smoke using {wheel.resolve()}")
    try:
        for extra in requested_extras:
            skip_reason = SKIPPED_EXTRAS.get(extra)
            if skip_reason is not None:
                print(f"extra {extra}: SKIP - {skip_reason}")
                continue

            probe = PROBES[extra]
            print(f"extra {extra}: {probe.summary}")
            try:
                _run_extra_probe(
                    extra=extra,
                    probe=probe,
                    wheel=wheel.resolve(),
                    repo_root=repo_root,
                    temp_root=temp_root,
                )
            except SmokeFailure as exc:
                failures.append(extra)
                print(f"extra {extra}: FAIL - {exc}", file=sys.stderr)
            else:
                print(f"extra {extra}: PASS")
    finally:
        if args.keep_temp:
            print(f"kept wheel extra smoke directory: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    if failures:
        raise SystemExit(f"wheel extra smoke failed: {', '.join(failures)}")


def _read_pyproject_extras(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = data.get("project", {}).get("optional-dependencies", {})
    if not isinstance(optional, dict) or not optional:
        raise SystemExit("expected [project.optional-dependencies] in pyproject.toml")
    return list(optional)


def _ensure_probe_coverage(pyproject_extras: list[str]) -> None:
    known = set(PROBES) | set(SKIPPED_EXTRAS)
    unknown = sorted(set(pyproject_extras) - known)
    if unknown:
        raise SystemExit(
            "pyproject declares extras without wheel_extra_smoke probes: "
            + ", ".join(unknown)
        )


def _requested_extras(raw: list[str] | None, pyproject_extras: list[str]) -> list[str]:
    if raw is None:
        return pyproject_extras

    requested = [extra for item in raw for extra in item.split(",") if extra]
    unknown = sorted(set(requested) - set(pyproject_extras))
    if unknown:
        raise SystemExit(
            "requested extras are not declared in pyproject.toml: " + ", ".join(unknown)
        )
    return requested


def _find_or_build_wheel(dist_dir: Path, *, repo_root: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        print(f"no wheel found in {dist_dir}; running uv build")
        _run(["uv", "build", "--out-dir", str(dist_dir)], cwd=repo_root)
        wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def _run_extra_probe(
    *,
    extra: str,
    probe: Probe,
    wheel: Path,
    repo_root: Path,
    temp_root: Path,
) -> None:
    extra_root = temp_root / _safe_dir_name(extra)
    venv_dir = extra_root / "venv"
    app_dir = extra_root / "consumer-app"
    app_dir.mkdir(parents=True)
    _run([sys.executable, "-m", "venv", "--clear", str(venv_dir)], cwd=extra_root)

    python = _venv_bin(venv_dir, "python")
    install_env = _clean_env()
    install_env["PIP_CACHE_DIR"] = str(temp_root / "pip-cache")
    install_env["CARGO_HOME"] = str(temp_root / "cargo-home")
    print(f"extra {extra}: installing {_wheel_requirement(wheel, extras=(extra,))}")
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            _wheel_requirement(wheel, extras=(extra,)),
        ],
        cwd=app_dir,
        env=install_env,
    )

    probe_path = app_dir / "extra_probe.py"
    probe_path.write_text(_probe_script(probe), encoding="utf-8")
    env = _clean_env()
    env["RAG_CORE_WHEEL_EXTRA_SMOKE_REPO_ROOT"] = str(repo_root)
    _run([str(python), str(probe_path)], cwd=app_dir, env=env)


def _probe_script(probe: Probe) -> str:
    header = textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import os
        from pathlib import Path

        import rag_core


        def _is_relative_to(path: Path, parent: Path) -> bool:
            try:
                path.relative_to(parent)
            except ValueError:
                return False
            return True


        repo_root = Path(os.environ["RAG_CORE_WHEEL_EXTRA_SMOKE_REPO_ROOT"]).resolve()
        imported_from = Path(rag_core.__file__).resolve()
        if _is_relative_to(imported_from, repo_root):
            raise AssertionError(f"rag_core imported from checkout: {imported_from}")

        result = {}
        """
    ).strip()
    footer = textwrap.dedent(
        """
        result["imported_from"] = str(imported_from)
        print(json.dumps(result, sort_keys=True))
        """
    ).strip()
    return "\n".join([header, textwrap.dedent(probe.code).strip(), footer])


def _safe_dir_name(extra: str) -> str:
    return extra.replace("-", "_")


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
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SmokeFailure(
            f"command exited {completed.returncode}: {shlex.join(command)}"
        )
    return completed


if __name__ == "__main__":
    main()
