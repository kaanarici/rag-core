from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PgVectorSqlCall:
    sql: str
    params: tuple[object, ...]


class PgVectorFakeAcquire:
    def __init__(self, connection: "PgVectorFakeConnection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "PgVectorFakeConnection":
        return self._connection

    async def __aexit__(self, *exc: object) -> None:
        return None


class PgVectorFakePool:
    def __init__(self, connection: "PgVectorFakeConnection | None" = None) -> None:
        self.connection = connection or PgVectorFakeConnection()
        self.close_calls = 0

    def acquire(self) -> PgVectorFakeAcquire:
        return PgVectorFakeAcquire(self.connection)

    async def close(self) -> None:
        self.close_calls += 1


class PgVectorFakeConnection:
    def __init__(
        self,
        *,
        extension_available: bool = True,
        dense_dimensions: int | None = 3,
    ) -> None:
        self.extension_available = extension_available
        self.dense_dimensions = dense_dimensions
        self.execute_calls: list[PgVectorSqlCall] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []
        self.fetch_calls: list[PgVectorSqlCall] = []
        self.fetchrow_calls: list[PgVectorSqlCall] = []
        self.fetchval_calls: list[PgVectorSqlCall] = []
        self.rows: dict[str, dict[str, object]] = {}
        self.fail_create_extension = False

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append(PgVectorSqlCall(query, tuple(args)))
        if query == "CREATE EXTENSION IF NOT EXISTS vector":
            if self.fail_create_extension:
                raise RuntimeError("permission denied")
            self.extension_available = True
        if query.startswith("DELETE FROM") and "id = ANY" in query:
            point_ids = _string_list(args[0])
            for point_id in point_ids:
                self.rows.pop(point_id, None)
        if query.startswith("DELETE FROM") and "id = ANY" not in query:
            namespace, corpus_id, document_id = _scope_args(args)
            self.rows = {
                point_id: row
                for point_id, row in self.rows.items()
                if not _matches_scope(row, namespace, corpus_id, document_id)
            }
        return "OK"

    async def executemany(
        self,
        command: str,
        args: Sequence[Sequence[object]],
    ) -> str:
        params = [tuple(row) for row in args]
        self.executemany_calls.append((command, params))
        for row in params:
            payload = json.loads(_required_str(row[2]))
            self.rows[_required_str(row[0])] = {
                "id": row[0],
                "dense": list(row[1]) if isinstance(row[1], list) else row[1],
                "payload": payload,
                "namespace": row[3],
                "corpus_id": row[4],
                "document_id": row[5],
                "document_key": row[6],
                "chunk_index": row[11],
            }
        return "OK"

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        self.fetch_calls.append(PgVectorSqlCall(query, tuple(args)))
        if "ORDER BY chunk_index ASC" in query:
            namespace, corpus_id, document_id, indices = args
            wanted = set(_int_list(indices))
            rows = [
                _result_row(row, score=0.0)
                for row in self.rows.values()
                if _matches_scope(row, namespace, corpus_id, document_id)
                and row.get("chunk_index") in wanted
            ]
            return sorted(rows, key=_chunk_index)
        if "ORDER BY dense <=>" in query:
            namespace = args[1] if len(args) > 1 else None
            corpus_ids = args[2] if len(args) > 2 and isinstance(args[2], list) else None
            rows = [
                _result_row(row, score=0.8)
                for row in self.rows.values()
                if row.get("namespace") == namespace
                and (corpus_ids is None or row.get("corpus_id") in corpus_ids)
            ]
            _raise_on_unsafe_numeric_cast(query, args, rows)
            rows = _apply_numeric_range_subset(query, args, rows)
            return rows[: _limit_arg(args)]
        return []

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetchrow_calls.append(PgVectorSqlCall(query, tuple(args)))
        namespace = args[0] if len(args) > 0 else None
        corpus_id = args[1] if len(args) > 1 else None
        lookup = args[2] if len(args) > 2 else None
        for row in self.rows.values():
            if row.get("namespace") != namespace or row.get("corpus_id") != corpus_id:
                continue
            if '"document_key"' in query:
                if row.get("document_key") == lookup:
                    return _result_row(row, score=0.0)
            elif row.get("document_id") == lookup:
                return _result_row(row, score=0.0)
        return None

    async def fetchval(self, query: str, *args: object) -> object:
        self.fetchval_calls.append(PgVectorSqlCall(query, tuple(args)))
        if "pg_extension" in query:
            return self.extension_available
        if "format_type" in query and "pg_attribute" in query:
            if self.dense_dimensions is None:
                return None
            return f"vector({self.dense_dimensions})"
        if "COUNT(*)" in query and args:
            namespace, corpus_id, document_id = _scope_args(args)
            return sum(
                1
                for row in self.rows.values()
                if _matches_scope(row, namespace, corpus_id, document_id)
            )
        if "COUNT(*)" in query:
            return len(self.rows)
        return None


def _result_row(row: Mapping[str, object], *, score: float) -> Mapping[str, object]:
    return {
        "id": row["id"],
        "payload": row["payload"],
        "score": score,
    }


def _payload(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("payload")
    return value if isinstance(value, Mapping) else {}


def _raise_on_unsafe_numeric_cast(
    query: str,
    args: Sequence[object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    if "::double precision" not in query or "jsonb_typeof" in query:
        return
    field = _numeric_range_field(args)
    if field is None:
        return
    for row in rows:
        value = _payload(row).get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise RuntimeError("invalid input syntax for type double precision")


def _apply_numeric_range_subset(
    query: str,
    args: Sequence[object],
    rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    field = _numeric_range_field(args)
    if field is None or "double precision" not in query:
        return list(rows)
    bound = _numeric_range_bound(args, field)
    if bound is None:
        return list(rows)
    matched: list[Mapping[str, object]] = []
    for row in rows:
        value = _numeric_payload_value(_payload(row).get(field))
        if value is not None and value >= bound:
            matched.append(row)
    return matched


def _numeric_range_field(args: Sequence[object]) -> str | None:
    for value in args:
        if value == "score":
            return "score"
    return None


def _numeric_range_bound(args: Sequence[object], field: str) -> float | None:
    seen_field = False
    for value in args:
        if value == field:
            seen_field = True
            continue
        if seen_field and not isinstance(value, bool) and isinstance(value, (int, float)):
            return float(value)
    return None


def _numeric_payload_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _chunk_index(row: Mapping[str, object]) -> int:
    value = _payload(row).get("chunk_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _scope_args(args: Sequence[object]) -> tuple[object, object | None, object | None]:
    namespace = args[0] if args else None
    corpus_id = args[1] if len(args) > 1 else None
    document_id = args[2] if len(args) > 2 else None
    return namespace, corpus_id, document_id


def _matches_scope(
    row: Mapping[str, object],
    namespace: object,
    corpus_id: object | None,
    document_id: object | None,
) -> bool:
    if row.get("namespace") != namespace:
        return False
    if corpus_id is not None and row.get("corpus_id") != corpus_id:
        return False
    if document_id is not None and row.get("document_id") != document_id:
        return False
    return True


def _required_str(value: object) -> str:
    if not isinstance(value, str):
        raise AssertionError(f"expected string, got {type(value).__name__}")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise AssertionError(f"expected list, got {type(value).__name__}")
    return [str(item) for item in value]


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        raise AssertionError(f"expected list, got {type(value).__name__}")
    return [int(item) for item in value]


def _limit_arg(args: Sequence[object]) -> int:
    if not args:
        return 10
    value = args[-1]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 10
