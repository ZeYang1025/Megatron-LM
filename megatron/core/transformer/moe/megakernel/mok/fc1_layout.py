# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Index math for repacking MCore's combined routed FC1 into MoK's recompute layout.

MCore owns routed FC1 as one combined ``[2I, H]`` matrix per expert: gate rows ``[0, I)`` and
up rows ``[I, 2I)``. MoK's forward consumes that layout directly (gate and up alias one payload
and MoK applies the up row offset), but ``mok.functional.recompute_forward_context`` accepts only
contiguous single-grouped gate and up weights:

* data ``[E, I, H]`` each, and
* in MXFP8, tcgen05 scale-factor tensors ``[E * I / 128, H / 128, 32, 16]`` each, in which
  expert ``e``'s 128-row block ``b`` is row block ``e * I / 128 + b``.

The tcgen05 layout stores one ``[32, 16]`` byte tile per 128-row x 128-column block, so a
128-row block of an expert's ``[2I / 128, H / 128, 32, 16]`` scale is self-contained and the
gate/up split is a row-block slice. This module holds only that index math; it imports nothing
outside the standard library so that it can be unit tested without a GPU stack.
"""

from __future__ import annotations

# Rows (and columns) covered by one tcgen05 scale-factor tile.
SCALE_BLOCK_ROWS = 128
SCALE_BLOCK_COLS = 128
# One E8M0 scale per 32 elements along the reduction dimension.
MXFP8_BLOCK_SIZE = 32
# Bytes of one [32, 16] tcgen05 scale tile (128 rows x 4 scales).
SCALE_TILE_BYTES = 32 * 16


def _check_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_scale_aligned(name: str, value: int) -> None:
    _check_positive_int(name, value)
    if value % SCALE_BLOCK_ROWS != 0:
        raise ValueError(
            f"{name} must be divisible by {SCALE_BLOCK_ROWS} for the MXFP8 tcgen05 scale layout, "
            f"got {value}"
        )


def gate_up_row_slices(intermediate_size: int) -> tuple[slice, slice]:
    """Rows of one expert's combined ``[2I, H]`` FC1 that hold gate and up."""
    _check_positive_int("intermediate_size", intermediate_size)
    return slice(0, intermediate_size), slice(intermediate_size, 2 * intermediate_size)


def gate_up_scale_block_slices(intermediate_size: int) -> tuple[slice, slice]:
    """Row blocks of one expert's ``[2I / 128, H / 128, 32, 16]`` scale holding gate and up."""
    _check_scale_aligned("intermediate_size", intermediate_size)
    blocks = intermediate_size // SCALE_BLOCK_ROWS
    return slice(0, blocks), slice(blocks, 2 * blocks)


def expert_scale_block_slice(expert: int, intermediate_size: int) -> slice:
    """Row blocks of expert ``expert`` in a single-grouped ``[E * 2I / 128, ...]`` FC1 scale."""
    if type(expert) is not int or expert < 0:
        raise ValueError(f"expert must be a non-negative integer, got {expert!r}")
    _check_scale_aligned("intermediate_size", intermediate_size)
    blocks_per_expert = 2 * intermediate_size // SCALE_BLOCK_ROWS
    return slice(expert * blocks_per_expert, (expert + 1) * blocks_per_expert)


def packed_gate_up_shapes(
    num_experts: int, intermediate_size: int, hidden_size: int, *, mxfp8: bool
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Shapes of one packed gate (or up) data tensor and its MXFP8 scale (None for BF16)."""
    _check_positive_int("num_experts", num_experts)
    _check_positive_int("intermediate_size", intermediate_size)
    _check_positive_int("hidden_size", hidden_size)
    data_shape = (num_experts, intermediate_size, hidden_size)
    if not mxfp8:
        return data_shape, None
    _check_scale_aligned("intermediate_size", intermediate_size)
    _check_scale_aligned("hidden_size", hidden_size)
    scale_shape = (
        num_experts * intermediate_size // SCALE_BLOCK_ROWS,
        hidden_size // SCALE_BLOCK_COLS,
        32,
        16,
    )
    return data_shape, scale_shape


def packed_gate_up_nbytes(
    num_experts: int, intermediate_size: int, hidden_size: int, *, mxfp8: bool
) -> int:
    """Bytes of the transient packed gate and up copies (data and scales) one rebuild allocates."""
    data_shape, scale_shape = packed_gate_up_shapes(
        num_experts, intermediate_size, hidden_size, mxfp8=mxfp8
    )
    element_bytes = 1 if mxfp8 else 2
    nbytes = num_experts * intermediate_size * hidden_size * element_bytes
    if scale_shape is not None:
        nbytes += scale_shape[0] * scale_shape[1] * SCALE_TILE_BYTES
    return 2 * nbytes


def routed_forward_context_nbytes(
    macrobatch_size: int, hidden_size: int, intermediate_size: int, *, mxfp8: bool
) -> int:
    """Bytes of MoK's routed forward context, which is sized by ``macrobatch_size`` rows.

    MXFP8 keeps FP8 copies of the dispatched input (transposed), gate, up and the activation
    output (transposed), each with one E8M0 scale per 32 elements: ``(H + 3 I) * 33 / 32`` bytes
    per row, e.g. 13,200 B for Kimi K3's latent ``H = 3584`` and ``I = 3072``. BF16 keeps the same
    four matrices in BF16: ``2 * (H + 3 I)`` bytes per row. A MoK-fused shared expert adds its
    own BF16 gate/up/activation, which this function does not count.
    """
    _check_positive_int("macrobatch_size", macrobatch_size)
    _check_positive_int("hidden_size", hidden_size)
    _check_positive_int("intermediate_size", intermediate_size)
    elements = macrobatch_size * (hidden_size + 3 * intermediate_size)
    if not mxfp8:
        return 2 * elements
    if elements % MXFP8_BLOCK_SIZE != 0:
        raise ValueError("MXFP8 context dimensions must be divisible by the 32-element block")
    return elements + elements // MXFP8_BLOCK_SIZE


def valid_context_rows(num_tokens: int, macrobatch_size: int) -> int:
    """Rows of a routed forward context that MoK writes: the first macrobatch's padded rows.

    MoK's forward processes macrobatches last to first and leaves macrobatch 0 in the context
    buffers; its rebuild computes only macrobatch 0. Rows past MoK's padded routed-token count
    are never written (nor read by backward) and hold arbitrary values.
    """
    if type(num_tokens) is not int or num_tokens < 0:
        raise ValueError(f"num_tokens must be a non-negative integer, got {num_tokens!r}")
    _check_positive_int("macrobatch_size", macrobatch_size)
    return min(num_tokens, macrobatch_size)
