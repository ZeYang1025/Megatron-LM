# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Torch-free checks of the FC1 repacking that MoK's forward-context rebuild relies on.

``moe_mok_recompute_forward_context`` rebuilds MoK's forward context from routed FC1 packed out of
MCore's combined ``[2I, H]`` expert weights. These tests model MoK's weight and scale addressing
(mirrored from mixture-of-kittens d59188a, csrc/megakernel/types.cuh and forward.cuh /
recompute_forward_context.cuh) with labelled rows and scale blocks, and check that the packed
layout makes the rebuild read exactly the rows and scale blocks the forward read.

The module under test is loaded from its file, so neither torch nor a GPU is needed. Where torch
is installed, run it like any unit test. Without torch, ``tests/unit_tests/__init__.py`` cannot be
imported, so run the file from outside the package (the symlink keeps the repository path)::

    mkdir -p /tmp/mok_fc1_layout
    ln -sf "$PWD/tests/unit_tests/transformer/moe/test_mok_fc1_layout.py" /tmp/mok_fc1_layout/
    python -m pytest -p no:cacheprovider /tmp/mok_fc1_layout/test_mok_fc1_layout.py
"""

import importlib.util
from pathlib import Path

import pytest

_LAYOUT_PATH = (
    Path(__file__).resolve().parents[4]
    / "megatron/core/transformer/moe/megakernel/mok/fc1_layout.py"
)
_spec = importlib.util.spec_from_file_location("_mok_fc1_layout_under_test", _LAYOUT_PATH)
fc1_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc1_layout)

# MoK constants (types.cuh): a weight tile covers MLP_WEIGHT_ROWS_PER_CTA = MLP_Nb / CLUSTER_SIZE
# rows; a tcgen05 scale tile covers QUANT_Mb = 128 rows.
MLP_WEIGHT_ROWS_PER_CTA = 256 // 2
QUANT_MB = 128

# Kimi-K3 per-rank MoE shape at EP64: 14 local experts, latent hidden 3584, expert FFN 3072.
K3_EXPERTS, K3_HIDDEN, K3_FFN = 14, 3584, 3072


def _physical_row_block(
    expert, logical_block, *, blocks_per_expert, block_offset, per_expert_table
):
    """routed_scale_view::physical_row_block in types.cuh."""
    return (0 if per_expert_table else expert * blocks_per_expert) + block_offset + logical_block


def _forward_scale_block(scale_storage, expert, block, which, intermediate_size, split):
    """Scale block MoK's forward reads for (expert, logical gate/up block), combined FC1.

    forward.cuh passes the combined FC1 scale with physical_row_blocks_per_expert = 2I / 128 and
    row_block_offset 0 for gate, I / 128 for up. Split weights select one expert's scale tensor
    through the descriptor table; single-grouped weights index one [E * 2I / 128] tensor.
    """
    offset = 0 if which == "gate" else intermediate_size // QUANT_MB
    physical = _physical_row_block(
        expert,
        block,
        blocks_per_expert=2 * intermediate_size // QUANT_MB,
        block_offset=offset,
        per_expert_table=split,
    )
    return scale_storage[expert][physical] if split else scale_storage[physical]


def _rebuild_scale_block(packed_scale, expert, block, intermediate_size):
    """Scale block MoK's recompute reads: contiguous [E * I / 128], I / 128 blocks per expert."""
    physical = _physical_row_block(
        expert,
        block,
        blocks_per_expert=intermediate_size // QUANT_MB,
        block_offset=0,
        per_expert_table=False,
    )
    return packed_scale[physical]


def _forward_weight_row(payloads, expert, row, which, intermediate_size):
    """FC1 row MoK's forward reads: combined up starts I / MLP_WEIGHT_ROWS_PER_CTA tiles down."""
    tile_offset = 0 if which == "gate" else intermediate_size // MLP_WEIGHT_ROWS_PER_CTA
    return payloads[expert][tile_offset * MLP_WEIGHT_ROWS_PER_CTA + row]


def _pack(expert_rows, expert_scales, intermediate_size):
    """The list equivalent of weights.pack_fc1_gate_up (stack data, concatenate scale blocks)."""
    gate_rows, up_rows = fc1_layout.gate_up_row_slices(intermediate_size)
    gate = [rows[gate_rows] for rows in expert_rows]
    up = [rows[up_rows] for rows in expert_rows]
    gate_blocks, up_blocks = fc1_layout.gate_up_scale_block_slices(intermediate_size)
    gate_scale = [block for scale in expert_scales for block in scale[gate_blocks]]
    up_scale = [block for scale in expert_scales for block in scale[up_blocks]]
    return gate, up, gate_scale, up_scale


@pytest.mark.parametrize("num_experts", [1, 3, K3_EXPERTS])
@pytest.mark.parametrize("intermediate_size", [128, 256, 384])
@pytest.mark.parametrize("split", [True, False], ids=["split", "single_grouped"])
def test_packed_fc1_reads_what_forward_reads(num_experts, intermediate_size, split):
    rows = 2 * intermediate_size
    blocks = rows // QUANT_MB
    payloads = [[("row", expert, row) for row in range(rows)] for expert in range(num_experts)]
    if split:
        scale_storage = [
            [("block", expert, block) for block in range(blocks)] for expert in range(num_experts)
        ]
        expert_scales = scale_storage
    else:
        # One [E * 2I / 128] tensor; Megatron cuts it into per-expert views before packing.
        scale_storage = [
            ("block", expert, block) for expert in range(num_experts) for block in range(blocks)
        ]
        expert_scales = [
            scale_storage[fc1_layout.expert_scale_block_slice(expert, intermediate_size)]
            for expert in range(num_experts)
        ]

    gate, up, gate_scale, up_scale = _pack(payloads, expert_scales, intermediate_size)

    assert len(gate) == len(up) == num_experts
    assert all(len(rows_) == intermediate_size for rows_ in gate + up)
    data_shape, scale_shape = fc1_layout.packed_gate_up_shapes(
        num_experts, intermediate_size, 256, mxfp8=True
    )
    assert data_shape == (num_experts, intermediate_size, 256)
    assert len(gate_scale) == len(up_scale) == scale_shape[0]
    for expert in range(num_experts):
        for row in range(intermediate_size):
            assert gate[expert][row] == _forward_weight_row(
                payloads, expert, row, "gate", intermediate_size
            )
            assert up[expert][row] == _forward_weight_row(
                payloads, expert, row, "up", intermediate_size
            )
        for block in range(intermediate_size // QUANT_MB):
            for which, packed in (("gate", gate_scale), ("up", up_scale)):
                assert _rebuild_scale_block(
                    packed, expert, block, intermediate_size
                ) == _forward_scale_block(
                    scale_storage, expert, block, which, intermediate_size, split
                ), (which, expert, block)


def test_k3_slices_and_shapes():
    assert fc1_layout.gate_up_row_slices(K3_FFN) == (slice(0, 3072), slice(3072, 6144))
    assert fc1_layout.gate_up_scale_block_slices(K3_FFN) == (slice(0, 24), slice(24, 48))
    assert fc1_layout.expert_scale_block_slice(13, K3_FFN) == slice(13 * 48, 14 * 48)
    assert fc1_layout.packed_gate_up_shapes(K3_EXPERTS, K3_FFN, K3_HIDDEN, mxfp8=True) == (
        (14, 3072, 3584),
        (14 * 24, 28, 32, 16),
    )
    assert fc1_layout.packed_gate_up_shapes(K3_EXPERTS, K3_FFN, K3_HIDDEN, mxfp8=False) == (
        (14, 3072, 3584),
        None,
    )
    # The rebuild's transient packed FC1 copy at the K3 shape: ~0.32 GB in MXFP8.
    payload = 2 * K3_EXPERTS * K3_FFN * K3_HIDDEN
    assert fc1_layout.packed_gate_up_nbytes(
        K3_EXPERTS, K3_FFN, K3_HIDDEN, mxfp8=True
    ) == payload + payload // 32 == 317_915_136
    assert fc1_layout.packed_gate_up_nbytes(K3_EXPERTS, K3_FFN, K3_HIDDEN, mxfp8=False) == (
        2 * payload
    )


def test_k3_forward_context_size():
    # 13,200 B per macrobatch row in MXFP8; 1.61 GiB at MoK's default macrobatch of 131072.
    assert fc1_layout.routed_forward_context_nbytes(1, K3_HIDDEN, K3_FFN, mxfp8=False) == 25_600
    assert fc1_layout.routed_forward_context_nbytes(128, K3_HIDDEN, K3_FFN, mxfp8=True) == (
        128 * 13_200
    )
    default = fc1_layout.routed_forward_context_nbytes(131072, K3_HIDDEN, K3_FFN, mxfp8=True)
    assert default == 131072 * 13_200
    assert round(default / 2**30, 3) == 1.611
    # Per-tensor bytes of MoK's MXFP8 context (ops.py docstrings) add up to the same total.
    rows, h, i = 4096, 512, 768
    tensors = [
        h * rows,  # x_fp8_t_routed [H, M]
        (h // 128) * (rows // 128) * 512,  # x_sc_t_routed [H/128, M/128, 32, 16]
        rows * i,  # gate_fp8_routed [M, I]
        (rows // 128) * (i // 128) * 512,  # gate_sc_routed
        rows * i,  # up_fp8_routed
        (rows // 128) * (i // 128) * 512,  # up_sc_routed
        i * rows,  # hidden_fp8_t_routed [I, M]
        (i // 128) * (rows // 128) * 512,  # hidden_sc_t_routed
    ]
    assert fc1_layout.routed_forward_context_nbytes(rows, h, i, mxfp8=True) == sum(tensors)


def test_valid_context_rows():
    assert fc1_layout.valid_context_rows(133_120, 131_072) == 131_072
    assert fc1_layout.valid_context_rows(4_352, 8_192) == 4_352
    assert fc1_layout.valid_context_rows(0, 256) == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: fc1_layout.gate_up_scale_block_slices(3000),
        lambda: fc1_layout.gate_up_scale_block_slices(0),
        lambda: fc1_layout.gate_up_row_slices(-1),
        lambda: fc1_layout.gate_up_row_slices(3072.0),
        lambda: fc1_layout.expert_scale_block_slice(-1, 3072),
        lambda: fc1_layout.packed_gate_up_shapes(14, 3072, 3000, mxfp8=True),
        lambda: fc1_layout.routed_forward_context_nbytes(0, 3584, 3072, mxfp8=True),
        lambda: fc1_layout.valid_context_rows(-1, 256),
    ],
)
def test_rejects_invalid_dimensions(call):
    with pytest.raises(ValueError):
        call()
