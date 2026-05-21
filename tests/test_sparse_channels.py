from rag_core.search.sparse_channels import (
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
    merge_sparse_channels,
    primary_sparse_channel,
    single_sparse_channel,
)


def test_sparse_channel_helpers_keep_primary_channel_canonical() -> None:
    merged = merge_sparse_channels("primary", {SECONDARY_SPARSE_CHANNEL: "secondary"})

    assert merged == {
        PRIMARY_SPARSE_CHANNEL: "primary",
        SECONDARY_SPARSE_CHANNEL: "secondary",
    }
    assert primary_sparse_channel(merged, missing_message="missing") == "primary"


def test_sparse_channel_helpers_fall_back_to_first_available_channel() -> None:
    channels = {SECONDARY_SPARSE_CHANNEL: "secondary"}

    assert primary_sparse_channel(channels, missing_message="missing") == "secondary"


def test_single_sparse_channel_wraps_primary_channel() -> None:
    assert single_sparse_channel("vector") == {PRIMARY_SPARSE_CHANNEL: "vector"}
