from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

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


def test_jsonl_sink_reuses_one_handle(tmp_path: Path) -> None:
    """Five emits must open the file handle exactly once."""
    path = tmp_path / "events.jsonl"
    open_call_count = 0
    _real_open = __import__("rag_core.private_files", fromlist=["open_private_append_handle"]).open_private_append_handle

    def counting_open(p: Any, *, reject_symlink: bool = False) -> Any:
        nonlocal open_call_count
        open_call_count += 1
        return _real_open(p, reject_symlink=reject_symlink)

    with patch("rag_core.events.sinks.open_private_append_handle", counting_open):
        sink = JsonlSink(path)
        for _ in range(5):
            sink.emit(IngestStarted(filename="a.txt"))

    assert open_call_count == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5


def test_jsonl_sink_close_is_idempotent_and_reopens(tmp_path: Path) -> None:
    """close() twice must not raise; a subsequent emit must land on disk."""
    path = tmp_path / "events.jsonl"
    reopen_count = 0
    _real_open = __import__("rag_core.private_files", fromlist=["open_private_append_handle"]).open_private_append_handle

    def counting_open(p: Any, *, reject_symlink: bool = False) -> Any:
        nonlocal reopen_count
        reopen_count += 1
        return _real_open(p, reject_symlink=reject_symlink)

    with patch("rag_core.events.sinks.open_private_append_handle", counting_open):
        sink = JsonlSink(path)
        sink.emit(IngestStarted(filename="first.txt"))
        sink.close()
        sink.close()  # second close must not raise
        sink.emit(IngestStarted(filename="second.txt"))

    # first open + reopen after close = 2
    assert reopen_count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_jsonl_sink_write_failure_counts_and_recovers(tmp_path: Path) -> None:
    """A single forced write error increments failure_count by 1; the next emit recovers."""
    path = tmp_path / "events.jsonl"

    sink = JsonlSink(path)
    # Force an error by closing the underlying fd from under the handle.
    sink.emit(IngestStarted(filename="prime.txt"))  # lazily opens handle
    assert sink._handle is not None
    # Sabotage: close the file object so the next write raises.
    handle_before = sink._handle
    handle_before.close()

    sink.emit(IngestStarted(filename="bad.txt"))  # must swallow the error
    assert sink.failure_count == 1
    # Handle must be dropped after the error.
    handle_after_bad: object = sink._handle
    assert handle_after_bad is None

    sink.emit(IngestStarted(filename="recovery.txt"))  # must reopen and succeed
    assert sink.failure_count == 1  # no additional failure

    lines = path.read_text(encoding="utf-8").splitlines()
    # "prime.txt" event + "recovery.txt" event; "bad.txt" was dropped
    assert len(lines) == 2
