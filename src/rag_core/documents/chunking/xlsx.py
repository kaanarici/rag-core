"""Token-budgeted row windows for normalized XLSX markdown."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from rag_core.config.chunking_config import XLSX_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk, estimate_token_count
from rag_core.documents.xlsx_protocol import extract_xlsx_sheet_marker

from .budget import fits_chunk_budget, token_budget_for_char_limit
from .markdown import MarkdownChunker
from .protocol import ChunkConfig
from .xlsx_fragments import header_units, split_payload, split_semantic_payload

_SHEET_HEADING_RE = re.compile(
    r"^## Sheet:\s+(.+?)(?:\s+\(Rows\s+([1-9]\d*)-([1-9]\d*)\))?\s*$",
    re.MULTILINE,
)
_ROW_MARKER_RE = re.compile(r"\s*<!-- rag-core-xlsx-row:([1-9]\d*) -->\s*$")


@dataclass(frozen=True)
class _Row:
    number: int
    text: str
    token_count: int


@dataclass(frozen=True)
class _Context:
    text: str
    header_truncated: bool


@dataclass(frozen=True)
class _ParsedTable:
    header: str
    separator: str
    header_row: int
    rows: list[_Row]
    tail_start: int


@dataclass(frozen=True)
class _SheetTable:
    name: str
    header: str
    separator: str
    config: ChunkConfig

    def context(
        self,
        start_row: int,
        end_row: int,
        *,
        reserve: str = "",
        include_header: bool = True,
    ) -> _Context:
        heading = f"## Sheet: {self.name} (Rows {start_row}-{end_row})"
        header = self.header if include_header else ""
        full = f"{heading}\n\n{header}\n{self.separator}" if header else heading
        if fits_chunk_budget(f"{full}{reserve}", max_chars=self.config.max_chars):
            return _Context(full, False)

        base = f"## Sheet (Rows {start_row}-{end_row})"
        if not fits_chunk_budget(f"{base}{reserve}", max_chars=self.config.max_chars):
            raise ValueError("XLSX provenance does not fit configured budget")

        suffix = f" (Rows {start_row}-{end_row})"
        bounded_name = _largest_prefix(
            self.name,
            builder=lambda value: f"## Sheet: {value}{suffix}{reserve}",
            max_chars=self.config.max_chars,
        )
        context = f"## Sheet: {bounded_name}{suffix}" if bounded_name else base
        header_payload = header.strip().strip("|").strip()
        if header_payload:
            bounded_header = _largest_prefix(
                header_payload,
                builder=lambda value: f"{context}\n\nHeader: {value}{reserve}",
                max_chars=self.config.max_chars,
            )
            if bounded_header:
                context = f"{context}\n\nHeader: {bounded_header}"
        return _Context(context, bool(header))

    def make_chunk(
        self,
        *,
        text: str,
        context: _Context,
        start_row: int,
        end_row: int,
        data_text: str,
        metadata: dict[str, object] | None = None,
    ) -> PreparedChunk:
        chunk_metadata: dict[str, object] = {
            "chunking_strategy": XLSX_CHUNKING_STRATEGY,
            "offset_reconstruction": "unreliable",
            "row_range": f"{start_row}-{end_row}",
            "section_path": f"Sheet: {self.name}",
            "section_title": f"Sheet: {self.name}",
            "sheet_name": self.name,
            "xlsx_context_token_count": estimate_token_count(context.text),
            "xlsx_data_token_count": estimate_token_count(data_text),
        }
        if context.header_truncated:
            chunk_metadata["xlsx_header_context_truncated"] = True
        if metadata:
            chunk_metadata.update(metadata)
        return PreparedChunk(
            chunk_index=0,
            text=text,
            embedding_text=text,
            word_count=len(text.split()),
            start_char=None,
            end_char=None,
            token_count=estimate_token_count(text),
            chunking_strategy=XLSX_CHUNKING_STRATEGY,
            metadata=chunk_metadata,
        )


class XlsxChunker:
    """Build row windows with repeated sheet and table-header context."""

    def chunk(self, text: str, config: ChunkConfig) -> list[PreparedChunk]:
        if not text:
            return []
        matches = list(_SHEET_HEADING_RE.finditer(text))
        if not matches:
            return MarkdownChunker().chunk(text, config)

        chunks = MarkdownChunker().chunk(text[: matches[0].start()], config)
        emitted_headers: set[tuple[str, str, int]] = set()
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            chunks.extend(
                _chunk_sheet(
                    text[start:end].rstrip(),
                    sheet_name=match.group(1).strip(),
                    declared_start=(
                        int(match.group(2)) if match.group(2) is not None else None
                    ),
                    declared_end=(
                        int(match.group(3)) if match.group(3) is not None else None
                    ),
                    config=config,
                    emitted_headers=emitted_headers,
                )
            )

        chunks = [
            replace(chunk, chunk_index=index) for index, chunk in enumerate(chunks)
        ]
        if any(
            not fits_chunk_budget(chunk.text, max_chars=config.max_chars)
            for chunk in chunks
        ):
            raise ValueError("XLSX chunk exceeded configured budget")
        return chunks


def _chunk_sheet(
    section: str,
    *,
    sheet_name: str,
    declared_start: int | None,
    declared_end: int | None,
    config: ChunkConfig,
    emitted_headers: set[tuple[str, str, int]],
) -> list[PreparedChunk]:
    section, marker = extract_xlsx_sheet_marker(section)
    if marker is not None:
        sheet_name = marker.sheet_name
        declared_start = marker.start_row
        declared_end = marker.end_row
    lines = section.splitlines()
    parsed = _parse_table(lines)
    if parsed is None:
        return _non_table_chunks(
            section,
            sheet_name=sheet_name,
            declared_start=declared_start,
            declared_end=declared_end,
            config=config,
        )

    table = _SheetTable(
        name=sheet_name,
        header=parsed.header,
        separator=parsed.separator,
        config=config,
    )
    if parsed.rows:
        row_chunks = _row_chunks(
            table,
            rows=parsed.rows,
            header_row=parsed.header_row,
            declared_start=declared_start,
        )
        header_key = (sheet_name, parsed.header, parsed.header_row)
        if (
            any(
                chunk.metadata.get("xlsx_header_context_truncated") is True
                for chunk in row_chunks
            )
            and header_key not in emitted_headers
        ):
            emitted_headers.add(header_key)
            chunks = [
                *_header_chunks(
                    table,
                    start_row=parsed.header_row,
                    end_row=parsed.header_row,
                ),
                *row_chunks,
            ]
        else:
            chunks = row_chunks
    else:
        start_row = declared_start or parsed.header_row
        chunks = _header_chunks(
            table,
            start_row=start_row,
            end_row=declared_end or parsed.header_row,
        )

    tail = "\n".join(lines[parsed.tail_start :]).strip()
    if tail:
        _append_tail(chunks, tail=tail, table=table)
    return chunks


def _parse_table(lines: list[str]) -> _ParsedTable | None:
    table_index = _table_start(lines)
    if table_index is None:
        return None
    header, header_row = _strip_row_marker(lines[table_index])
    if header_row is None:
        return None

    rows: list[_Row] = []
    cursor = table_index + 2
    while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
        text, number = _strip_row_marker(lines[cursor])
        if number is None:
            break
        rows.append(_Row(number, text, estimate_token_count(text)))
        cursor += 1
    return _ParsedTable(
        header=header,
        separator=lines[table_index + 1].rstrip(),
        header_row=header_row,
        rows=rows,
        tail_start=cursor,
    )


def _row_chunks(
    table: _SheetTable,
    *,
    rows: list[_Row],
    header_row: int,
    declared_start: int | None,
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    active: list[_Row] = []
    active_start: int | None = None
    active_chars = active_tokens = 0
    first_window = True
    token_budget = token_budget_for_char_limit(table.config.max_chars)

    for row in rows:
        start_row = active_start or _window_start(
            row.number,
            first_window=first_window,
            header_row=header_row,
            declared_start=declared_start,
        )
        context = table.context(start_row, row.number)
        candidate_chars = active_chars + len(row.text) + (1 if active else 0)
        candidate_tokens = active_tokens + row.token_count
        if (
            len(context.text) + 1 + candidate_chars <= table.config.max_chars
            and estimate_token_count(context.text) + candidate_tokens <= token_budget
        ):
            active_start = start_row
            active.append(row)
            active_chars = candidate_chars
            active_tokens = candidate_tokens
            continue

        if active:
            assert active_start is not None
            chunks.append(
                _window_chunk(
                    table,
                    active,
                    start_row=active_start,
                )
            )
            first_window = False
            active = []
            active_start = None
            active_chars = active_tokens = 0

        row_start = _window_start(
            row.number,
            first_window=first_window,
            header_row=header_row,
            declared_start=declared_start,
        )
        row_chunks = _single_row_chunks(table, row=row, start_row=row_start)
        if len(row_chunks) == 1 and "row_fragment" not in row_chunks[0].metadata:
            active = [row]
            active_start = row_start
            active_chars = len(row.text)
            active_tokens = row.token_count
        else:
            chunks.extend(row_chunks)
            first_window = False

    if active:
        assert active_start is not None
        chunks.append(
            _window_chunk(
                table,
                active,
                start_row=active_start,
            )
        )
    return chunks


def _window_chunk(
    table: _SheetTable,
    rows: list[_Row],
    *,
    start_row: int,
) -> PreparedChunk:
    end_row = rows[-1].number
    context = table.context(start_row, end_row)
    data = "\n".join(row.text for row in rows)
    return table.make_chunk(
        text=f"{context.text}\n{data}",
        context=context,
        start_row=start_row,
        end_row=end_row,
        data_text=data,
    )


def _single_row_chunks(
    table: _SheetTable,
    *,
    row: _Row,
    start_row: int,
) -> list[PreparedChunk]:
    context = table.context(start_row, row.number)
    text = f"{context.text}\n{row.text}"
    if fits_chunk_budget(text, max_chars=table.config.max_chars):
        return [
            table.make_chunk(
                text=text,
                context=context,
                start_row=start_row,
                end_row=row.number,
                data_text=row.text,
            )
        ]
    return _fragment_chunks(
        table,
        payload=row.text.strip().strip("|").strip(),
        label_prefix=f"Row {row.number} fragment",
        start_row=start_row,
        end_row=row.number,
        metadata_key="row_fragment",
        include_header=True,
    )


def _header_chunks(
    table: _SheetTable,
    *,
    start_row: int,
    end_row: int,
) -> list[PreparedChunk]:
    payload = table.header.strip().strip("|").strip()
    return _fragment_chunks(
        table,
        payload=payload,
        label_prefix="Header fragment",
        start_row=start_row,
        end_row=end_row,
        metadata_key="header_fragment",
        include_header=False,
        semantic_units=header_units(payload),
    )


def _fragment_chunks(
    table: _SheetTable,
    *,
    payload: str,
    label_prefix: str,
    start_row: int,
    end_row: int,
    metadata_key: str,
    include_header: bool,
    semantic_units: list[str] | None = None,
) -> list[PreparedChunk]:
    if not payload:
        context = table.context(
            start_row,
            end_row,
            include_header=include_header,
        )
        return [
            table.make_chunk(
                text=context.text,
                context=context,
                start_row=start_row,
                end_row=end_row,
                data_text="",
                metadata={metadata_key: "1/1"},
            )
        ]
    digit_count = len(str(max(1, len(payload))))
    upper_label = f"{label_prefix} {'9' * digit_count}/{'9' * digit_count}: "
    payload_reserve = "界" * max(
        1,
        token_budget_for_char_limit(table.config.max_chars) // 2,
    )
    context = table.context(
        start_row,
        end_row,
        reserve=f"\n{upper_label}{payload_reserve}",
        include_header=include_header,
    )
    prefix = f"{context.text}\n{upper_label}"
    fragments = (
        split_semantic_payload(
            semantic_units,
            prefix=prefix,
            max_chars=table.config.max_chars,
        )
        if semantic_units is not None
        else split_payload(
            payload,
            prefix=prefix,
            max_chars=table.config.max_chars,
        )
    )
    count = len(fragments)
    return [
        table.make_chunk(
            text=(f"{context.text}\n{label_prefix} {index}/{count}: {fragment}"),
            context=context,
            start_row=start_row,
            end_row=end_row,
            data_text=fragment,
            metadata={metadata_key: f"{index}/{count}"},
        )
        for index, fragment in enumerate(fragments, start=1)
    ]


def _non_table_chunks(
    section: str,
    *,
    sheet_name: str,
    declared_start: int | None,
    declared_end: int | None,
    config: ChunkConfig,
) -> list[PreparedChunk]:
    row_metadata = (
        {"row_range": f"{declared_start}-{declared_end}"}
        if declared_start is not None and declared_end is not None
        else {}
    )
    return [
        replace(
            chunk,
            start_char=None,
            end_char=None,
            metadata={
                **{
                    key: value
                    for key, value in chunk.metadata.items()
                    if key != "row_range"
                },
                "offset_reconstruction": "unreliable",
                "section_path": f"Sheet: {sheet_name}",
                "section_title": f"Sheet: {sheet_name}",
                "sheet_name": sheet_name,
                **row_metadata,
            },
        )
        for chunk in MarkdownChunker().chunk(section, config)
    ]


def _append_tail(
    chunks: list[PreparedChunk],
    *,
    tail: str,
    table: _SheetTable,
) -> None:
    if chunks and fits_chunk_budget(
        f"{chunks[-1].text}\n\n{tail}",
        max_chars=table.config.max_chars,
    ):
        previous = chunks[-1]
        combined = f"{previous.text}\n\n{tail}"
        chunks[-1] = replace(
            previous,
            text=combined,
            embedding_text=combined,
            word_count=len(combined.split()),
            token_count=estimate_token_count(combined),
        )
        return

    chunks.extend(
        _non_table_chunks(
            f"## Sheet: {table.name}\n\n{tail}",
            sheet_name=table.name,
            declared_start=None,
            declared_end=None,
            config=table.config,
        )
    )


def _largest_prefix(
    value: str,
    *,
    builder: Callable[[str], str],
    max_chars: int,
) -> str:
    low, high, best = 0, min(len(value), max_chars), 0
    while low <= high:
        middle = (low + high) // 2
        if fits_chunk_budget(builder(value[:middle]), max_chars=max_chars):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return value[:best]


def _window_start(
    first_row: int,
    *,
    first_window: bool,
    header_row: int,
    declared_start: int | None,
) -> int:
    return (
        first_row if not first_window else declared_start or min(header_row, first_row)
    )


def _table_start(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1):
        if not lines[index].lstrip().startswith("|"):
            continue
        cells = [
            cell.strip() for cell in lines[index + 1].strip().strip("|").split("|")
        ]
        if cells and all(cell and set(cell) <= {"-", ":"} for cell in cells):
            return index
    return None


def _strip_row_marker(line: str) -> tuple[str, int | None]:
    match = _ROW_MARKER_RE.search(line)
    if match is None:
        return line.rstrip(), None
    return line[: match.start()].rstrip(), int(match.group(1))


__all__ = ["XlsxChunker"]
