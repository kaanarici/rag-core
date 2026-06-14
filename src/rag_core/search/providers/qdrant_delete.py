from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import DeleteFilter

from .qdrant_filters import build_delete_filter
from .qdrant_payloads import _qdrant_point_id


async def delete_qdrant_filter(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    filter_values: DeleteFilter,
    namespace: str,
    policy: VectorStorePolicy,
) -> None:
    await client.delete(
        collection_name=collection_name,
        points_selector=rest.FilterSelector(
            filter=build_delete_filter(
                filter_values=filter_values,
                namespace=namespace,
                policy=policy,
            ),
        ),
    )


async def delete_qdrant_point_ids(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    point_ids: Sequence[str],
) -> None:
    qdrant_point_ids = [_qdrant_point_id(point_id) for point_id in point_ids]
    await client.delete(
        collection_name=collection_name,
        points_selector=rest.PointIdsList(points=cast(Any, qdrant_point_ids)),
    )
