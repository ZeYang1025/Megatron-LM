# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP local checkpoint fragments: layouts, optimizer companions and master indices (CPU).

Adapted from PR #8109 (eb0f8e0f6, test_gtp_local_checkpoint.py). The multi-GPU DCP and real
module tests are in tests/unit_tests/generalized_tensor_parallel/test_gtp_local_checkpoint.py.
"""

import itertools
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from megatron.core.dist_checkpointing.gtp import (  # noqa: E402
    _gtp_split_tensor_factory,
    gtp_logical_rows,
    split_gtp_projection,
)
from megatron.core.dist_checkpointing.gtp_layout import (  # noqa: E402
    flat_range_chunks,
    gtp_projection_pieces,
    gtp_valid_rows,
)
from megatron.core.dist_checkpointing.mapping import ShardedTensor  # noqa: E402
from megatron.core.dist_checkpointing.optimizer import (  # noqa: E402
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.ssm.utils import _split_tensor_factory  # noqa: E402
from megatron.core.transformer.mlp import (  # noqa: E402
    apply_gtp_swiglu_sharded_factory,
    apply_swiglu_sharded_factory,
)

ALIGNMENT = 8  # rows per GTP alignment tile in these tests


def make_projection(kind, tp, gtp, rank, *, device='cpu', dtype=torch.float32, scribble=False):
    """Build one rank's fused projection weight and its checkpoint factory, as a4 does.

    ``kind`` is ``kda`` (packed Q/K/V of unequal sections), ``glu`` (gate/up) or ``expert``
    (GLU under a prepended expert axis). Values are distinct across TP ranks and sections,
    so any mislabeled row changes the loaded result.

    GTP entries use the padded-grid metadata of ``make_tp_sharded_tensor_for_checkpoint``
    (composite offset ``tp_rank * gtp + gtp_rank`` over ``tp * gtp`` fragments, pad rows
    included); the factory recomputes logical coordinates from the sections. Non-GTP layouts
    use the unchanged production factories, which have no optimizer companion.

    Returns:
        (factory, weight, expected physical shard, number of rows holding data)
    """
    global_sizes = [12, 8, 12] if kind == 'kda' else [12, 12]
    full_parts = []
    cursor = 0
    for size in global_sizes:
        full_parts.append(
            torch.arange(cursor + 1, cursor + size * 3 + 1, device=device).reshape(size, 3)
        )
        cursor += size * 3
    tp_rank, gtp_rank = rank % (tp * gtp) // gtp, rank % gtp
    dp_rank = rank // (tp * gtp)
    logical = torch.cat([part.chunk(tp)[tp_rank] for part in full_parts]).to(dtype)
    rows = logical.shape[0]
    if gtp > 1:
        rows = math.ceil(rows / (gtp * ALIGNMENT)) * ALIGNMENT
    start = rows * gtp_rank
    valid = max(0, min(rows, logical.shape[0] - start))
    expected = logical.new_zeros(rows, 3)
    expected[:valid] = logical[start : start + valid]
    weight = torch.nn.Parameter(expected.clone())
    if scribble:
        with torch.no_grad():
            weight.fill_(-999)
    offsets = ((0, 0, 1),) if kind == 'expert' else ()
    key = 'layers.0.in_proj.weight' if kind == 'kda' else 'layers.0.fc1.weight'
    sections = [size // tp for size in global_sizes]
    names = ['query', 'key', 'value'] if kind == 'kda' else ['w', 'v']
    if gtp > 1:
        entry = ShardedTensor.from_rank_offsets(
            key,
            weight,
            *offsets,
            (len(offsets), tp_rank * gtp + gtp_rank, tp * gtp),
            prepend_axis_num=len(offsets),
            replica_id=(0, 0, dp_rank),
        )
        factory = _gtp_split_tensor_factory(
            entry, sections, names, gtp_rank, tp_rank, tp, weight=weight, glu=kind != 'kda'
        )
    else:
        entry = ShardedTensor.from_rank_offsets(
            key,
            weight,
            *offsets,
            (len(offsets), tp_rank, tp),
            prepend_axis_num=len(offsets),
            replica_id=(0, 0, dp_rank),
        )
        if kind == 'kda':
            factory = _split_tensor_factory(entry, sections, names, 0)
        else:
            factory = apply_swiglu_sharded_factory(entry, offsets)
    return factory, weight, expected, valid


# ---------------------------------------------------------------------------------------------
# Pure layout arithmetic (gtp_layout has no torch dependency)
# ---------------------------------------------------------------------------------------------


def _non_gtp_coordinate(sections, names, glu, singleton, tp_rank, tp_size, row):
    """Checkpoint (key suffix, global row, global rows) of TP-local ``row`` without GTP."""
    section_start = 0
    for index, (size, name) in enumerate(zip(sections, names)):
        if row < section_start + size:
            local = row - section_start
            if not glu:
                return f'.{name}', tp_rank * size + local, tp_size * size
            if singleton:
                return f'_{name}', tp_rank * size + local, tp_size * size
            return '', (index * tp_size + tp_rank) * size + local, 2 * tp_size * size
        section_start += size
    raise AssertionError(row)


@pytest.mark.parametrize('glu,singleton', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('tp_size', [1, 2, 4])
@pytest.mark.parametrize('gtp_size', [1, 2, 3, 8])
@pytest.mark.parametrize('alignment', [1, 4, 16])
def test_projection_pieces_match_non_gtp_layout(glu, singleton, tp_size, gtp_size, alignment):
    """Every logical row is saved once, in shard order, at its non-GTP checkpoint coordinate."""
    for sections in ([6, 6], [10, 10], [33, 33]) if glu else ([12, 8, 12], [5, 7, 1, 9], [64]):
        names = ['w', 'v'] if glu else [f's{i}' for i in range(len(sections))]
        total = sum(sections)
        shard_rows = math.ceil(total / (gtp_size * alignment)) * alignment
        for tp_rank in range(tp_size):
            saved = []
            for gtp_rank in range(gtp_size):
                pieces = gtp_projection_pieces(
                    sections,
                    names,
                    shard_rows,
                    gtp_rank,
                    tp_rank,
                    tp_size,
                    glu=glu,
                    singleton_local_shards=singleton,
                )
                cursor = 0
                for piece in pieces:
                    assert piece.start == cursor < piece.stop
                    cursor = piece.stop
                    for row in range(piece.start, piece.stop):
                        logical = gtp_rank * shard_rows + row
                        assert (
                            piece.suffix,
                            piece.global_offset + row - piece.start,
                            piece.global_rows,
                        ) == _non_gtp_coordinate(
                            sections, names, glu, singleton, tp_rank, tp_size, logical
                        )
                        saved.append(logical)
                assert cursor == gtp_valid_rows(total, shard_rows, gtp_rank)
            assert sorted(saved) == list(range(total))


@pytest.mark.parametrize('shape', [(), (5,), (3, 4), (2, 3, 4), (4, 1, 3)])
def test_flat_range_chunks_tile_range_with_boxes(shape):
    """Chunks tile a flat range in order; each chunk is a box of consecutive elements."""
    numel = math.prod(shape)
    strides = [math.prod(shape[axis + 1 :]) for axis in range(len(shape))]
    for start in range(numel + 1):
        for stop in range(start, numel + 1):
            cursor = start
            for chunk in flat_range_chunks(shape, start, stop):
                assert chunk.start == cursor
                assert len(chunk.shape) == len(shape)
                elements = [
                    sum((c + i) * s for c, i, s in zip(chunk.coordinates, index, strides))
                    for index in itertools.product(*[range(n) for n in chunk.shape])
                ]
                assert elements == list(range(cursor, cursor + len(elements)))
                cursor += len(elements)
            assert cursor == stop


def test_layout_rejects_invalid_input():
    with pytest.raises(ValueError):
        gtp_projection_pieces([4, 4], ['a'], 4, 0, 0, 1)
    with pytest.raises(ValueError):
        gtp_projection_pieces([4, 0], ['a', 'b'], 4, 0, 0, 1)
    with pytest.raises(ValueError, match='GLU'):
        gtp_projection_pieces([4, 6], ['w', 'v'], 4, 0, 0, 1, glu=True)
    with pytest.raises(ValueError):
        flat_range_chunks((3, 2), 2, 7)


# ---------------------------------------------------------------------------------------------
# Model fragments and optimizer companions
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize('kind', ['kda', 'glu', 'expert'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_local_fragments_and_flat_optimizer_slices(kind, dtype):
    """All/partial padding, TP offsets, optimizer identity and non-row-aligned flat slices."""
    for rank in range(8):
        with mock.patch.object(
            dist, 'all_gather_into_tensor', side_effect=AssertionError('gather')
        ):
            factory, weight, expected, valid = make_projection(kind, 2, 4, rank, dtype=dtype)
            chunks = factory.build()
        tensors = [chunk for chunk in chunks if isinstance(chunk, ShardedTensor)]
        assert sum(chunk.data.numel() for chunk in tensors) == valid * 3
        for chunk in tensors:
            assert chunk.data.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        values = [
            chunk.data if isinstance(chunk, ShardedTensor) else chunk.unwrap() for chunk in chunks
        ]
        torch.testing.assert_close(factory.merge_fn(values), expected, rtol=0, atol=0)
        metadata = get_param_id_to_sharded_param_map({'model': factory}, [weight])[0]
        assert metadata.data is weight
        for start, stop in [(0, weight.numel()), (1, 5), (3, weight.numel() - 1)]:
            state = torch.arange(start, stop, dtype=torch.float32) + 100
            optimizer = replace(
                metadata,
                key='optimizer.momentum.' + factory.key,
                data=state,
                flattened_range=slice(start, stop),
            )
            built = optimizer.build()
            loaded = {
                'layout': built['layout'].unwrap(),
                'chunks': [part.data for part in built.get('chunks', [])],
            }
            reference = state.clone()
            reference[max(0, valid * 3 - start) :].zero_()
            torch.testing.assert_close(optimizer.merge_fn(loaded), reference, rtol=0, atol=0)
            if not loaded['chunks']:
                # An empty list would leak into the rank-0 common state; it is omitted.
                assert 'chunks' not in built
            for part in built.get('chunks', []):
                assert part.key.startswith('optimizer.momentum.')
                assert part.data.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()


def test_padding_only_rank_saves_no_common_state():
    """A padding-only GTP rank contributes no tensors and no common state for its fragments."""
    from megatron.core.dist_checkpointing.state_dict_utils import save_preprocess

    for rank in range(8):
        factory, weight, _, valid = make_projection('glu', 2, 4, rank)
        metadata = get_param_id_to_sharded_param_map({'w': factory}, [weight])[0]
        state_dict = {
            'w': factory,
            'state': {
                0: {
                    'exp_avg': make_sharded_optimizer_tensor(
                        metadata, weight.detach().float() + 5, 'optimizer.state.exp_avg'
                    )
                }
            },
        }
        sharded, common = save_preprocess(state_dict, validate_access_integrity=False)
        assert common == {}, (rank, common)
        if not valid:
            assert sharded == {}, (rank, sharded)


def _fake_gtp_weight(rows, columns, gtp_rank, gtp_size, pad_length):
    weight = torch.nn.Parameter(torch.arange(rows * columns, dtype=torch.float32).reshape(rows, -1))
    weight.is_gtp_weight_remat = True
    weight.group = SimpleNamespace(rank=lambda: gtp_rank, size=lambda: gtp_size)
    weight.pad_length = pad_length
    return weight


@pytest.fixture
def gtp_api_enabled(monkeypatch):
    """Recognize GTP-tagged params without the GTP TransformerEngine build."""
    from megatron.core.tensor_parallel import gtp_api

    monkeypatch.setattr(gtp_api, 'HAVE_GTP', True)
    monkeypatch.setattr(
        gtp_api,
        'is_gtp_param',
        lambda param: getattr(param, 'is_gtp_weight_remat', False),
        raising=False,
    )


def test_production_entry_points_route_gtp_params_only(gtp_api_enabled):
    """GTP weights take local fragments; every other weight keeps the existing factories."""
    plain = torch.nn.Parameter(torch.zeros(6, 3))
    plain_entry = ShardedTensor.from_rank_offsets('fc1.weight', plain, (0, 0, 1))
    assert gtp_logical_rows(plain) is None and gtp_logical_rows(None) is None
    assert split_gtp_projection(plain_entry, plain, [2, 2, 2], ['a', 'b', 'c'], None) is None
    factory = apply_gtp_swiglu_sharded_factory(plain_entry, plain, ())
    assert factory.optimizer_factory is None
    assert [part.global_offset for part in factory.build()] == [(0, 0), (3, 0)]

    # GTP rank 1 of 2: rows [8, 16) of 14 logical rows, the last two of them padding.
    weight = _fake_gtp_weight(8, 3, 1, 2, 2)
    entry = ShardedTensor.from_rank_offsets('fc1.weight', weight, (0, 1, 2))
    assert gtp_logical_rows(weight) == 14
    factory = apply_gtp_swiglu_sharded_factory(entry, weight, ())
    assert factory.for_optimizer().data is weight
    parts = factory.build()
    # Up rows 1..6 of 7, stored after all 7 gate rows; padding is not saved.
    assert [(p.key, p.global_offset, p.local_shape) for p in parts] == [
        ('fc1.weight', (8, 0), (6, 3))
    ]
    with pytest.raises(ValueError, match='do not cover'):
        split_gtp_projection(entry, weight, [6, 6], ['a', 'b'], None)
    with pytest.raises(ValueError, match='shape'):
        split_gtp_projection(
            ShardedTensor.from_rank_offsets('x', torch.zeros(4, 3), (0, 1, 2)),
            weight,
            [7, 7],
            ['a', 'b'],
            None,
        )
    weight.pad_length = 3
    with pytest.raises(ValueError, match='odd'):
        apply_gtp_swiglu_sharded_factory(entry, weight, ())


def test_for_optimizer_tracks_key_and_replica_rewrites():
    """Layer-offset key rewrites and PP/DP replica updates reach the optimizer companion."""
    # TP1 x GTP2 rank 0 holds rows [0, 16) of the 12 + 8 + 12 query/key/value rows.
    factory, weight, _, _ = make_projection('kda', 1, 2, 0)
    factory.key = 'decoder.layers.27.self_attention.in_proj.weight'
    factory.replica_id = (1, 0, 3)
    companion = factory.for_optimizer()
    assert companion.key == factory.key and companion.data is weight
    assert companion.replica_id == (1, 0, 3)
    state = make_sharded_optimizer_tensor(factory, torch.zeros_like(weight), 'optimizer.state.m')
    parts = state.build()['chunks']
    assert [(part.key, part.global_offset, part.local_shape) for part in parts] == [
        (f'optimizer.state.m.{factory.key}.query', (0, 0), (12, 3)),
        (f'optimizer.state.m.{factory.key}.key', (0, 0), (4, 3)),
    ]
    assert all(part.replica_id == (1, 0, 3) for part in parts)
    stripped = factory.without_data()
    assert stripped.data is None and stripped.optimizer_factory.data is None


# ---------------------------------------------------------------------------------------------
# Float16 / LayerWise master indices
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("master_layout", ["factories", "legacy_lists", "indexed_lists"])
@pytest.mark.parametrize("layerwise", [False, True])
def test_mixed_master_factories_preserve_parameter_indices_through_load(master_layout, layerwise):
    """Only the middle master is built by a companion; its neighbors must not shift.

    Load preprocessing moves the companion's local-only layout object out of a master list
    without its position, so the masters of such a group are keyed by parameter index.
    """
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, merge
    from megatron.core.dist_checkpointing.mapping import apply_factory_merges
    from megatron.core.dist_checkpointing.state_dict_utils import load_preprocess
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    model_params = [torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16)) for _ in range(3)]
    masters = [torch.nn.Parameter(torch.full((4, 3), float(index + 11))) for index in range(3)]
    wrapper = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    wrapper.optimizer = torch.optim.SGD(masters, lr=0.1, momentum=0.9)
    wrapper.config = SimpleNamespace(fp16=False)
    wrapper.grad_scaler = None
    wrapper._optimizer_state_offloader = None
    wrapper.float16_groups = [model_params]
    wrapper.fp32_from_float16_groups = [masters]
    wrapper.fp32_from_fp32_groups = [[]]
    wrapper.synchronize_optimizer_state_for_checkpoint = lambda: None
    model_state = {}
    for index, param in enumerate(model_params):
        key = "layer%d.weight" % index
        model_state[key] = ShardedTensor.from_rank_offsets(key, param)
        wrapper.optimizer.state[masters[index]]["momentum_buffer"] = torch.full(
            (4, 3), float(index + 101)
        )
    live = model_params[1]
    live.is_gtp_weight_remat = True
    physical = ShardedTensor.from_rank_offsets("layer1.weight", live, (0, 0, 2))
    model_state["layer1.weight"] = _gtp_split_tensor_factory(
        physical, [4, 4], ['w', 'v'], 0, 0, 1, weight=live, glu=True
    )

    owner = wrapper
    if layerwise:
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        owner = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
        owner.chained_optimizers = [wrapper]
        owner.pg_collection = SimpleNamespace()
        owner.model_chunks = []
    if master_layout != "factories":
        restored = wrapper.state_dict()
        restored["fp32_from_fp16_params"] = [[param.detach().clone() for param in masters]]
    else:
        with mock.patch(
            "megatron.core.optimizer.layer_wise_optimizer._all_gather_param_group_metadata",
            side_effect=lambda group, pg: [group],
        ):
            sharded = owner.sharded_state_dict(model_state)
        assert isinstance(sharded["fp32_from_fp16_params"][0], dict)
        remaining, nonpersistent, factories = load_preprocess(sharded)
        loaded = dict_list_map_outplace(
            lambda value: value.data.clone() if isinstance(value, ShardedTensor) else value,
            remaining,
        )
        merge(nonpersistent, loaded)
        apply_factory_merges(nonpersistent, factories)
        restored = nonpersistent
    if master_layout == "indexed_lists":
        restored["fp32_from_fp16_params"] = dict(enumerate(restored["fp32_from_fp16_params"]))
    for param in masters:
        param.data.fill_(-999)
    owner.load_state_dict(restored)
    for index, param in enumerate(masters):
        torch.testing.assert_close(param, torch.full_like(param, index + 11), rtol=0, atol=0)
        torch.testing.assert_close(
            wrapper.optimizer.state[param]["momentum_buffer"],
            torch.full_like(param, index + 101),
            rtol=0,
            atol=0,
        )


def test_masters_without_companions_keep_list_layout():
    """Optimizers whose masters use no companion keep a4's list-per-group layout."""
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    model_params = [torch.nn.Parameter(torch.zeros(6, 3, dtype=torch.bfloat16)) for _ in range(2)]
    masters = [torch.nn.Parameter(torch.ones(6, 3)) for _ in range(2)]
    wrapper = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    wrapper.optimizer = torch.optim.SGD(masters, lr=0.1)
    wrapper.config = SimpleNamespace(fp16=False)
    wrapper.grad_scaler = None
    wrapper._optimizer_state_offloader = None
    wrapper.float16_groups = [model_params]
    wrapper.fp32_from_float16_groups = [masters]
    wrapper.fp32_from_fp32_groups = [[]]
    wrapper.synchronize_optimizer_state_for_checkpoint = lambda: None
    entry = ShardedTensor.from_rank_offsets('layer1.weight', model_params[1], (0, 0, 1))
    model_state = {
        'layer0.weight': ShardedTensor.from_rank_offsets('layer0.weight', model_params[0]),
        # A non-GTP split factory, as KDA uses without GTP.
        'layer1.weight': _split_tensor_factory(entry, [2, 4], ['a', 'b'], 0),
    }
    sharded = wrapper.sharded_state_dict(model_state)
    group = sharded['fp32_from_fp16_params'][0]
    assert isinstance(group, list) and len(group) == 2


def test_float16_load_rejects_mismatched_master_indices():
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    masters = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    wrapper = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    wrapper.optimizer = torch.optim.SGD(masters, lr=0.1)
    wrapper.config = SimpleNamespace(fp16=False)
    wrapper.grad_scaler = None
    wrapper._optimizer_state_offloader = None
    wrapper.fp32_from_float16_groups = [masters]
    state = wrapper.state_dict()
    state['fp32_from_fp16_params'] = {0: {0: torch.zeros(2), 2: torch.zeros(2)}}
    with pytest.raises(ValueError, match='indexed'):
        wrapper.load_state_dict(state)
