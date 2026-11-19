# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Index arithmetic for GTP local checkpoint fragments.

Pure Python without a torch import, so the layouts can be checked exhaustively without a GPU.
``dist_checkpointing.gtp`` turns :func:`gtp_projection_pieces` into model-weight fragments and
``dist_checkpointing.optimizer`` turns :func:`flat_range_chunks` into optimizer-state fragments.
"""

import math
from typing import List, NamedTuple, Sequence, Tuple


class ProjectionPiece(NamedTuple):
    """Intersection of a physical GTP shard with one logical section of a fused projection."""

    start: int
    """First row of the piece within the physical shard."""
    stop: int
    """One past the last row of the piece within the physical shard."""
    suffix: str
    """Suffix appended to the projection's checkpoint key."""
    global_offset: int
    """Dim-0 offset of the piece in that checkpoint tensor."""
    global_rows: int
    """Dim-0 size of that checkpoint tensor."""


def gtp_valid_rows(total_rows: int, shard_rows: int, gtp_rank: int) -> int:
    """Rows of GTP shard ``gtp_rank`` that hold data; the rest is the zero alignment tail."""
    return max(0, min(shard_rows, total_rows - gtp_rank * shard_rows))


def gtp_projection_pieces(
    sections: Sequence[int],
    names: Sequence[str],
    shard_rows: int,
    gtp_rank: int,
    tp_rank: int,
    tp_size: int,
    *,
    glu: bool = False,
    singleton_local_shards: bool = False,
) -> List[ProjectionPiece]:
    """Map a physical GTP shard of a fused TP-local projection onto checkpoint coordinates.

    The TP-local projection stacks ``sections`` along dim 0, named by ``names``. GTP rank
    ``gtp_rank`` holds its rows ``[gtp_rank * shard_rows, (gtp_rank + 1) * shard_rows)``, with
    zero alignment rows past ``sum(sections)``. The result lists, in row order, the piece of
    every section that the shard covers; pad rows belong to no piece.

    The coordinates are those of the same projection's checkpoint without GTP, so either run
    can load the other's checkpoint:

    - Split projection (``glu=False``, ``_split_tensor_factory``): section ``name`` is the
      tensor ``<key>.<name>`` with ``tp_size * size`` rows, ordered by TP rank.
    - Gated linear unit (``glu=True``, sections ``[gate, up]``, ``apply_swiglu_sharded_factory``):
      one tensor ``<key>`` holding the gate rows of every TP rank followed by their up rows.
      With ``singleton_local_shards`` the halves are the tensors ``<key>_<name>`` instead.

    Raises:
        ValueError: names do not match the sections, a section is empty, or GLU halves differ.
    """
    if len(sections) != len(names) or any(size <= 0 for size in sections):
        raise ValueError(
            f'Split names {list(names)} must match positive section sizes {list(sections)}'
        )
    if glu and (len(sections) != 2 or sections[0] != sections[1]):
        raise ValueError(f'GLU requires equal gate and up sections, got {list(sections)}')
    shard_start = gtp_rank * shard_rows
    shard_end = shard_start + gtp_valid_rows(sum(sections), shard_rows, gtp_rank)
    pieces = []
    section_start = 0
    for index, (size, name) in enumerate(zip(sections, names)):
        piece_start = max(section_start, shard_start)
        piece_end = min(section_start + size, shard_end)
        if piece_start < piece_end:
            offset = tp_rank * size + piece_start - section_start
            global_rows = tp_size * size
            suffix = f'.{name}'
            if glu:
                suffix = f'_{name}' if singleton_local_shards else ''
                if not singleton_local_shards:
                    offset += index * global_rows
                    global_rows *= 2
            pieces.append(
                ProjectionPiece(
                    piece_start - shard_start, piece_end - shard_start, suffix, offset, global_rows
                )
            )
        section_start += size
    return pieces


class FlatChunk(NamedTuple):
    """A run of consecutive elements of a C-contiguous tensor that also forms a box."""

    start: int
    """Flat index of the first element."""
    coordinates: Tuple[int, ...]
    """Multi-index of the first element."""
    shape: Tuple[int, ...]
    """Box shape; leading singleton axes place it inside the tensor."""


def flat_range_chunks(shape: Sequence[int], start: int, stop: int) -> List[FlatChunk]:
    """Split the elements ``[start, stop)`` of a C-contiguous tensor of ``shape`` into boxes.

    Each box starts at the current element and spans whole sub-tensors along the outermost
    axis that allows it: whole rows if the run starts at a row boundary and covers at least
    one row, else part of a single row. Boxes come in flat order and tile the range exactly,
    so concatenating their flattened values restores the range.

    Raises:
        ValueError: the range does not lie within the tensor.
    """
    shape = tuple(shape)
    if not 0 <= start <= stop <= math.prod(shape):
        raise ValueError(f'Invalid flat range [{start}, {stop}) for shape {shape}')
    strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
    chunks = []
    cursor = start
    while cursor < stop:
        coordinates = tuple((cursor // stride) % size for stride, size in zip(strides, shape))
        chunk_shape, count = (), 1
        for axis, stride in enumerate(strides):
            if any(coordinates[axis + 1 :]):
                continue
            extent = min(shape[axis] - coordinates[axis], (stop - cursor) // stride)
            if extent:
                chunk_shape = (1,) * axis + (extent,) + shape[axis + 1 :]
                count = extent * stride
                break
        chunks.append(FlatChunk(cursor, coordinates, chunk_shape))
        cursor += count
    return chunks
