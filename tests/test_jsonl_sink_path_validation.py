from __future__ import annotations

from pathlib import Path

import pytest

from rag_core.events.sinks import JsonlSink
from rag_core.events.types import IngestStarted


def test_jsonl_sink_validates_directory_path_before_runtime_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="events JSONL path must be a file"):
        JsonlSink(tmp_path)


def test_jsonl_sink_prepares_parent_and_file_on_construction(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "events.jsonl"

    sink = JsonlSink(path)

    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
    sink.emit(IngestStarted(filename="a.txt"))
    assert "ingest.started" in path.read_text(encoding="utf-8")
