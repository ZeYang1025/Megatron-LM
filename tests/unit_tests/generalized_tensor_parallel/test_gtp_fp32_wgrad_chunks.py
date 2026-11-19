# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Column chunking of the opt-in FP32-accumulating GTP wgrad reduction (CPU only, no GPU).

pytest tests/unit_tests/generalized_tensor_parallel/test_gtp_fp32_wgrad_chunks.py
"""

import pytest

pytest.importorskip("torch")

from megatron.core.tensor_parallel.generalized_tensor_parallelism import (  # noqa: E402
    _FP32_WGRAD_REDUCTION_CHUNK_BYTES,
    _fp32_wgrad_reduction_column_chunks,
)


@pytest.mark.parametrize("num_cols", [1, 2, 7, 8, 63, 7168])
@pytest.mark.parametrize("bytes_per_col", [1, 3, 100, 737280])
@pytest.mark.parametrize("max_chunk_bytes", [0, 1, 99, 300, 512 * 1024**2])
def test_chunks_cover_columns_under_the_cap(num_cols, bytes_per_col, max_chunk_bytes):
    chunks = _fp32_wgrad_reduction_column_chunks(num_cols, bytes_per_col, max_chunk_bytes)
    covered = [col for start, end in chunks for col in range(start, end)]
    assert covered == list(range(num_cols)), chunks
    widths = [end - start for start, end in chunks]
    max_cols = max(1, max_chunk_bytes // bytes_per_col)
    assert max(widths) <= max_cols, (widths, max_cols)
    # The fewest chunks the cap allows, with widths balanced to within one column.
    assert len(chunks) == -(-num_cols // max_cols)
    assert max(widths) - min(widths) <= 1


def test_no_columns_gives_no_chunks():
    assert _fp32_wgrad_reduction_column_chunks(0, 8, 64) == []


def test_rejects_non_positive_column_bytes():
    with pytest.raises(ValueError):
        _fp32_wgrad_reduction_column_chunks(4, 0, 64)


def test_kimi_k3_embedding_chunking():
    """163840 x 7168 BF16 embedding at GTP 8: ten chunks of ~504 MiB scratch each."""
    rows_padded, cols, gtp_size, element_size = 163840, 7168, 8, 2
    bytes_per_col = rows_padded * 2 * element_size + (rows_padded // gtp_size) * 4
    chunks = _fp32_wgrad_reduction_column_chunks(
        cols, bytes_per_col, _FP32_WGRAD_REDUCTION_CHUNK_BYTES
    )
    assert len(chunks) == 10
    assert {end - start for start, end in chunks} == {716, 717}
    assert max(end - start for start, end in chunks) * bytes_per_col <= 512 * 1024**2
