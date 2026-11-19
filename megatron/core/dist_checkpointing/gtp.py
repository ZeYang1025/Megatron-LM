# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local checkpoint fragments for GTP-sharded fused projections (no parameter collectives).

A GTP rank holds a contiguous row slice of a TP-local fused projection, for example the
packed Q/K/V ``in_proj`` of KDA/GDN or the ``[gate; up]`` FC1 of a gated MLP. The slice
boundaries need not line up with the projection's logical sections, so the section split used
without GTP cannot be applied to the shard. Each rank instead saves views of the
intersections of its own rows with the sections, under the keys and coordinates that the
non-GTP checkpoint uses. GTP and non-GTP runs can therefore load each other's checkpoints,
and no rank gathers the projection.

Optimizer state goes through a companion factory bound to the live parameter
(``ShardedTensorFactory.optimizer_factory``, picked up by ``for_optimizer``). It maps the
FP32 master and each moment onto the same logical fragments, again without gathering.

Adapted from PR #8109 (eb0f8e0f6), which extends the gated-delta-product fragments of #8530
(``ssm/gated_delta_product.py``). Gated-delta-product keeps its own helper and its optimizer
fallback; Mamba keeps its gather.
"""

from dataclasses import replace
from typing import List, Optional

import torch

from .gtp_layout import gtp_projection_pieces, gtp_valid_rows
from .mapping import LocalNonpersistentObject, ReplicaId, ShardedTensor, ShardedTensorFactory


def attach_optimizer_factory(
    factory: ShardedTensorFactory, weight: torch.Tensor
) -> ShardedTensorFactory:
    """Bind the model parameter's optimizer state to the factory's local logical fragments.

    The companion maps a physical optimizer tensor of ``weight``'s shape (or a flat slice of
    it) onto the ShardedTensors that ``factory`` builds, in order. Optimizer code reaches it
    through :meth:`ShardedTensorFactory.for_optimizer`.
    """
    from .optimizer import _make_sharded_optimizer_factory

    parts = [part for part in factory.build() if isinstance(part, ShardedTensor)]
    factory.optimizer_factory = _make_sharded_optimizer_factory(
        parts, weight, factory.key, physical_numel=weight.numel(), replica_id=factory.replica_id
    )
    return factory


def _gtp_split_tensor_factory(
    orig_sh_ten: ShardedTensor,
    split_sections: List[int],
    split_names: List[str],
    gtp_remat_rank: int,
    tp_rank: int,
    tp_size: int,
    *,
    weight: Optional[torch.Tensor] = None,
    glu: bool = False,
    singleton_local_shards: bool = False,
) -> ShardedTensorFactory:
    """Save the intersections of a physical GTP shard with TP-local logical sections.

    ``orig_sh_ten.data`` is this rank's physical shard: rows ``[gtp_remat_rank * R,
    (gtp_remat_rank + 1) * R)`` of the TP-local tensor, where R is the shard's row count, with
    zero alignment rows past ``sum(split_sections)``. Model fragments are views of the shard;
    pad rows are never saved, and loading rebuilds the shard with a zero tail. See
    :func:`gtp_projection_pieces` for the checkpoint coordinates.

    Args:
        orig_sh_ten: the projection's ShardedTensor. Only its key, replica_id and prepended
            axes are reused; dim-0 coordinates are recomputed from the logical sections.
        split_sections: TP-local logical row count of each section.
        split_names: section names, used as key suffixes (``w``/``v`` for a GLU).
        gtp_remat_rank: rank in the projection's GTP group.
        tp_rank: rank in the tensor-parallel group.
        tp_size: size of the tensor-parallel group.
        weight: the live parameter. When given, a source-bound optimizer companion is
            attached (see :func:`attach_optimizer_factory`).
        glu: sections are the equal gate and up halves of a gated linear unit.
        singleton_local_shards: store GLU gate and up under separate keys.
    """
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    data = orig_sh_ten.data
    if weight is not None and tuple(data.shape) != tuple(weight.shape):
        raise ValueError(
            f'Checkpoint entry {orig_sh_ten.key} has shape {tuple(data.shape)}, but its GTP '
            f'parameter shard has shape {tuple(weight.shape)}'
        )
    metadata = orig_sh_ten.without_data()
    prepend = metadata.prepend_axis_num
    local_rows, *trailing_shape = data.shape
    trailing_shape = tuple(trailing_shape)
    valid_rows = gtp_valid_rows(sum(split_sections), local_rows, gtp_remat_rank)
    base_key = orig_sh_ten.key
    pieces = []
    for piece in gtp_projection_pieces(
        split_sections,
        split_names,
        local_rows,
        gtp_remat_rank,
        tp_rank,
        tp_size,
        glu=glu,
        singleton_local_shards=singleton_local_shards,
    ):
        template = replace(
            metadata,
            key=base_key + piece.suffix,
            local_shape=(piece.stop - piece.start,) + trailing_shape,
            global_shape=metadata.global_shape[:prepend] + (piece.global_rows,) + trailing_shape,
            global_offset=metadata.global_offset[:prepend]
            + (piece.global_offset,)
            + (0,) * len(trailing_shape),
            axis_fragmentations=None,
            flattened_range=None,
            allow_shape_mismatch=False,
        )
        pieces.append((slice(piece.start, piece.stop), template))

    @torch.no_grad()
    def build(key: str, tensor: torch.Tensor, replica_id: ReplicaId, flattened_range):
        if flattened_range is not None:
            raise ValueError(
                f'{key}: GTP projection fragments cover whole rows; map flat optimizer '
                'fragments through the optimizer companion (ShardedTensorFactory.for_optimizer)'
            )
        if not pieces:
            # A padding-only shard saves nothing. An empty list would be dropped by checkpoint
            # preprocessing, leaving no value to merge on load, so keep a local zero-element
            # tensor that preserves the key and device.
            return [LocalNonpersistentObject(tensor.new_empty((0,) + trailing_shape))]
        return [
            replace(
                template,
                key=key + template.key[len(base_key) :],
                data=tensor[rows],
                dtype=tensor.dtype,
                replica_id=replica_id,
            )
            for rows, template in pieces
        ]

    @torch.no_grad()
    def merge(parts):
        # Rebuild the physical shard: the loaded rows, then zeros for the alignment tail, as
        # at initialization. Pad rows share MXFP8 scaling blocks with real rows.
        merged = parts[0].new_empty((local_rows,) + trailing_shape)
        if valid_rows:
            merged[:valid_rows] = torch.cat(parts)
        merged[valid_rows:].zero_()
        return merged

    # A fresh view: optimizer code matches parameters by identity, and must reach this
    # projection through the source-bound companion, never through the model fragments.
    factory = ShardedTensorFactory(
        base_key, data.view_as(data), build, merge, orig_sh_ten.replica_id
    )
    if weight is not None:
        attach_optimizer_factory(factory, weight)
    return factory


def gtp_logical_rows(weight: Optional[torch.Tensor]) -> Optional[int]:
    """TP-local logical row count of a GTP weight shard, or None if ``weight`` is not one.

    That is the shard's rows times the GTP group size, minus the zero alignment tail.
    """
    from megatron.core.tensor_parallel import gtp_api

    if weight is None or not gtp_api.HAVE_GTP or not gtp_api.is_gtp_param(weight):
        return None
    return weight.shape[0] * weight.group.size() - int(getattr(weight, 'pad_length', 0))


def split_gtp_projection(
    entry: ShardedTensor,
    weight: torch.Tensor,
    sections: List[int],
    names: List[str],
    tp_group: Optional[torch.distributed.ProcessGroup],
    *,
    glu: bool = False,
    singleton_local_shards: bool = False,
) -> Optional[ShardedTensorFactory]:
    """Local fragment factory for a GTP-sharded fused projection, or None for other weights.

    Args:
        entry: the ShardedTensor that the projection's linear layer emits for ``weight``.
        weight: the live parameter.
        sections: TP-local logical row count of each section; they must cover the weight.
        names: section names (checkpoint key suffixes).
        tp_group: tensor-parallel group the projection is sharded over.
        glu: sections are the gate and up halves of a gated linear unit.
        singleton_local_shards: store GLU gate and up under separate keys.

    Raises:
        ValueError: the sections do not cover the weight's logical rows.
    """
    from megatron.core.utils import get_pg_rank, get_pg_size

    logical_rows = gtp_logical_rows(weight)
    if logical_rows is None:
        return None
    if sum(sections) != logical_rows:
        raise ValueError(
            f'Projection sections {list(sections)} of {entry.key} do not cover its '
            f'{logical_rows} TP-local logical rows'
        )
    return _gtp_split_tensor_factory(
        entry,
        list(sections),
        list(names),
        weight.group.rank(),
        get_pg_rank(tp_group),
        get_pg_size(tp_group),
        weight=weight,
        glu=glu,
        singleton_local_shards=singleton_local_shards,
    )
