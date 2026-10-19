"""Install the wheel into a fresh consumer environment and use real local models."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CONSUMER = r"""
from pathlib import Path
import asyncio, json, subprocess, sys
import rag_core
from rag_core import Config, Document, RAGCore
from rag_core.easy import index

root = Path.cwd()
assert not Path(rag_core.__file__).is_relative_to(Path(sys.argv[1]))
folder = root / "documents"
folder.mkdir()
(folder / "billing.md").write_text("Invoices can be paid by card or ACH bank transfer.")
(folder / "security.md").write_text("Account passwords must have at least twelve characters.")
with index(folder, persist_dir=root / "indexes") as docs:
    assert "billing.md" in docs.search("How can invoices be paid?").evidence[0].document_key
    assert "Citations:" in docs.context("How can invoices be paid?")
with index(folder, persist_dir=root / "indexes") as docs:
    assert {r.ingest_state for r in docs.refresh().succeeded} == {"unchanged"}
for command in ["search", "context"]:
    result = subprocess.run([sys.executable, "-m", "rag_core", command,
        "How can invoices be paid?", str(folder), "--persist-dir", str(root/"indexes"), "--json"],
        check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["hits"] if command == "search" else payload["context_text"]
async def app():
    config = Config.local(root / "app-index", hybrid=True)
    async with RAGCore(config, tenant_id="acme", collection="docs") as rag:
        result = await rag.ingest(Document(key="billing.md", content=b"Pay invoices by ACH.", content_type="text/markdown"))
        assert result.status == "created"
        assert (await rag.search("invoice payment")).evidence
        assert (await rag.tool()("invoice payment"))["context_text"]
    async with RAGCore(config, tenant_id="acme", collection="docs") as rag:
        result = await rag.ingest(Document(key="billing.md", content=b"Pay invoices by ACH.", content_type="text/markdown"))
        assert result.status == "unchanged"
asyncio.run(app())
print(json.dumps({"installed_module": rag_core.__file__, "real_models": True,
    "python_folder_reopen": True, "cli_search_context": True, "app_hybrid_tool_reopen": True}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    (wheel,) = args.dist_dir.resolve().glob("*.whl")
    root = Path(tempfile.mkdtemp(prefix="rag-core-consumer-")).resolve()
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["RAG_CORE_INDEX_DIR"] = str(root / "indexes")
    python = root / "venv/bin/python"
    try:
        subprocess.run(
            ["uv", "venv", str(root / "venv"), "--python", sys.executable],
            check=True,
            env=env,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)],
            check=True,
            env=env,
        )
        app = root / "consumer.py"
        app.write_text(CONSUMER)
        subprocess.run(
            [str(python), str(app), str(repo)], cwd=root, env=env, check=True
        )
    finally:
        if args.keep_temp:
            print(f"Consumer evidence retained at {root}")
        else:
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
