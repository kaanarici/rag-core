"""Small stateful HTTP fixture for SDK publication tests, not hosted proof."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any

import httpx

from rag_core.search.document_versions import HEAD, KIND


def matches(row: dict[str, Any], condition: Any) -> bool:
    if condition is None:
        return True
    if condition[0] == "And":
        return all(matches(row, child) for child in condition[1])
    if condition[0] == "Not":
        return not matches(row, condition[1])
    key, operator, value = condition
    if operator == "Eq":
        return bool(row.get(key) == value)
    if operator == "In":
        return row.get(key) in value
    raise AssertionError(f"unsupported fixture filter: {condition!r}")


def vector(value: Any) -> list[float]:
    if isinstance(value, str):
        raw = base64.b64decode(value)
        return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    return list(value)


class PublicationHTTP:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.head_requests: list[dict[str, Any]] = []
        self.head_outcomes: list[int] = []
        self.fail_next_head_ack = False
        self.pause_next_head_ack = False
        self.head_applied = asyncio.Event()
        self.release_ack = asyncio.Event()

    def query(self, body: dict[str, Any]) -> dict[str, Any]:
        rows = [
            dict(row) for row in self.rows.values()
            if matches(row, body.get("filters"))
        ]
        if "aggregate_by" in body:
            return {"aggregations": {key: len(rows) for key in body["aggregate_by"]}}
        rank = body["rank_by"]
        if rank[0] == "vector":
            query = vector(rank[2])
            rows = [row for row in rows if "vector" in row]
            for row in rows:
                # Test vectors have unit length, so this is exact cosine distance.
                row["$dist"] = 1 - sum(
                    a * b for a, b in zip(query, vector(row["vector"]), strict=True)
                )
            rows.sort(key=lambda row: (row["$dist"], row["id"]))
        else:
            assert rank == ["id", "asc"]
            rows.sort(key=lambda row: row["id"])
        rows = rows[:body["limit"]]
        attrs = body.get("include_attributes", False)
        if attrs is not True:
            fields = {"id", "$dist", *(attrs if isinstance(attrs, list) else [])}
            rows = [{k: v for k, v in row.items() if k in fields} for row in rows]
        return {"rows": rows}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/query"):
            response = (
                {"results": [self.query(query) for query in body["queries"]]}
                if "queries" in body else self.query(body)
            )
            return httpx.Response(200, json=response)
        affected = 0
        writes = [*body.get("upsert_rows", []), *body.get("patch_rows", [])]
        head_write = any(row.get(KIND) == HEAD for row in writes)
        for row in body.get("upsert_rows", []):
            old = self.rows.get(row["id"])
            if old is None or matches(old, body.get("upsert_condition")):
                self.rows[row["id"]] = dict(row)
                affected += 1
        for row in body.get("patch_rows", []):
            old = self.rows.get(row["id"])
            if old is not None and matches(old, body.get("patch_condition")):
                old.update(row)
                affected += 1
        for id in body.get("deletes", []):
            affected += self.rows.pop(id, None) is not None
        if "delete_by_filter" in body:
            deleted = [
                id for id, row in self.rows.items()
                if matches(row, body["delete_by_filter"])
            ]
            for id in deleted:
                del self.rows[id]
            affected += len(deleted)
        if head_write:
            self.head_requests.append(body)
            self.head_outcomes.append(affected)
            if self.fail_next_head_ack:
                assert affected == 1
                self.fail_next_head_ack = False
                self.head_applied.set()
                return httpx.Response(
                    503, json={"error": "ack lost after applying head"},
                    headers={"retry-after-ms": "1"},
                )
            if self.pause_next_head_ack:
                assert affected == 1
                self.pause_next_head_ack = False
                self.head_applied.set()
                await self.release_ack.wait()
        return httpx.Response(
            200, json={"rows_affected": affected, "rows_remaining": False}
        )
