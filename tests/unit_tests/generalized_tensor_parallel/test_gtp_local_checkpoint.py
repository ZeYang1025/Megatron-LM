# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP local checkpoint fragments on GPUs: DCP topology changes and real Kimi-K3 modules.

Adapted from PR #8109 (eb0f8e0f6). The CPU-only layout, companion and master-index tests are
in tests/unit_tests/dist_checkpointing/test_gtp_local_fragments.py. Run on 4 GPUs:

    torchrun --nproc-per-node 4 -m pytest -x \
        tests/unit_tests/generalized_tensor_parallel/test_gtp_local_checkpoint.py
"""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine")
if not torch.cuda.is_available():
    pytest.skip("GTP checkpoint tests need CUDA", allow_module_level=True)

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP  # noqa: E402

if not HAVE_GTP:
    pytest.skip("GTP requires TE with hook registry", allow_module_level=True)

import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from megatron.core import dist_checkpointing  # noqa: E402
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.dist_checkpointing.mapping import ShardedTensor  # noqa: E402
from megatron.core.dist_checkpointing.optimizer import (  # noqa: E402
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (  # noqa: E402
    GTP_CONFIG,
    GTPShardedParam,
    update_gtp_config,
)
from megatron.core.tensor_parallel.gtp_api import is_gtp_param  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from tests.unit_tests.dist_checkpointing import TempNamedDir  # noqa: E402
from tests.unit_tests.dist_checkpointing.test_gtp_local_fragments import (  # noqa: E402
    make_projection,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: E402,F401
    _torchrun_dist_init,
    reset_gtp_globals,
)

_PGS = ['tp', 'cp', 'gtp_remat', 'dp_cp', 'dp_cp_gtp_remat']


def _require_world_size_multiple(n):
    if dist.get_world_size() % n:
        pytest.skip(
            f"Requires a world size divisible by {n}, got {dist.get_world_size()} "
            f"(launch with torchrun --nproc-per-node={n})"
        )


@pytest.mark.parametrize('kind', ['kda', 'glu', 'expert'])
@pytest.mark.parametrize(
    'source,target',
    [((2, 1), (1, 4)), ((1, 4), (2, 1)), ((2, 2), (1, 4)), ((1, 1), (1, 4)), ((1, 4), (1, 2))],
)
def test_cross_layout_model_and_optimizer(kind, source, target, tmp_path_dist_ckpt):
    """Real DCP cross-loads model, FP32 master and momentum, then continues an update.

    Layouts are (TP, GTP); the non-GTP layouts use the unchanged factories without an
    optimizer companion, so this also checks that both checkpoint layouts are compatible.
    """
    _require_world_size_multiple(4)
    rank = dist.get_rank()
    factory, weight, _, _ = make_projection(kind, *source, rank, device='cuda')
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)
    weight.grad = torch.full_like(weight, 2)
    optimizer.step()
    # Master and momentum differ from the model values: identity-only or zero-state
    # mappings cannot pass.
    metadata = get_param_id_to_sharded_param_map({'weight': factory}, [weight])[0]
    state = optimizer.state[weight]['momentum_buffer']
    checkpoint = {
        'weight': factory,
        'master': make_sharded_optimizer_tensor(metadata, weight.detach().float() + 1000, 'master'),
        'momentum': make_sharded_optimizer_tensor(metadata, state, 'momentum'),
    }
    name = f'gtp_local_{kind}_tp{source[0]}gtp{source[1]}_to_tp{target[0]}gtp{target[1]}'
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as ckpt_dir:
        dist_checkpointing.save(checkpoint, ckpt_dir)
        factory, weight, expected, valid = make_projection(
            kind, *target, rank, device='cuda', scribble=True
        )
        metadata = get_param_id_to_sharded_param_map({'weight': factory}, [weight])[0]
        loaded = dist_checkpointing.load(
            {
                'weight': factory,
                'master': make_sharded_optimizer_tensor(
                    metadata, torch.empty_like(weight), 'master'
                ),
                'momentum': make_sharded_optimizer_tensor(
                    metadata, torch.empty_like(weight), 'momentum'
                ),
            },
            ckpt_dir,
        )
    for key, delta in [('weight', -0.02), ('master', 999.98), ('momentum', None)]:
        reference = expected + delta if delta is not None else torch.full_like(expected, 2)
        reference[valid:].zero_()
        torch.testing.assert_close(loaded[key], reference, rtol=1e-6, atol=1e-5)
    with torch.no_grad():
        weight.copy_(loaded['weight'])
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)
    optimizer.load_state_dict(
        {
            'state': {0: {'momentum_buffer': loaded['momentum']}},
            'param_groups': optimizer.state_dict()['param_groups'],
        }
    )
    weight.grad = torch.full_like(weight, 3)
    optimizer.step()
    torch.testing.assert_close(
        weight[:valid], expected[:valid] - 0.02 - 0.048, rtol=1e-6, atol=1e-5
    )


def _build_module(kind, pg):
    """Build a small KDA, dense MLP or shared-expert MLP; return (module, key, weight).

    The sizes put GTP2 shard boundaries inside the packed Q/K/V sections of the KDA in_proj
    (768 rows, 384 per shard) and pad the GLU fc1 (776 rows, 400 per shard with alignment 32).
    """
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    if kind == 'kda':
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
        from megatron.core.ssm.kda_layer_config import KDALayerConfig

        if not HAVE_FLA_KDA:
            pytest.skip("FLA with KDA support is not installed.")
        config = KDALayerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=4,
            num_query_groups=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            # Kimi-K3 layout: separate F/G projections, a Q/K/V-only packed in_proj.
            kda_f_lora_rank=64,
            kda_safe_gate=True,
            kda_lower_bound=-5.0,
            normalization="RMSNorm",
            activation_func=F.silu,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=False,
            experimental_attention_variant="kda",
            is_hybrid_model=True,
            transformer_impl="transformer_engine",
        )
        spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
        model = KimiDeltaAttention(
            config, submodules=spec.submodules, layer_number=1, pg_collection=pg
        ).cuda()
        return model, 'in_proj.weight', model.in_proj.weight
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=388,
        moe_shared_expert_intermediate_size=388,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
    )
    if kind == 'shared':
        submodules = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
        )
        model = SharedExpertMLP(config, submodules, gate=False, pg_collection=pg)
    else:
        submodules = MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        )
        model = MLP(
            config,
            submodules,
            ffn_hidden_size=config.ffn_hidden_size,
            tp_group=pg.tp,
            pg_collection=pg,
        )
    model = model.cuda()
    return model, 'linear_fc1.weight', model.linear_fc1.weight


def _init_parallel(gtp_remat_size):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=gtp_remat_size,
    )
    model_parallel_cuda_manual_seed(42)
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=_PGS)
    # The production default: data-parallel groups include the GTP axis.
    return pg, {'dp_cp_group': pg.dp_cp_gtp_remat}


@pytest.fixture
def _gtp_alignment():
    old_pad = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=16)
    yield
    update_gtp_config(pad_for_alignment=old_pad)
    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state.clear()


@pytest.mark.parametrize('kind', ['kda', 'dense', 'shared'])
def test_real_kimi_module_uses_local_fragments(kind, _gtp_alignment):
    """KDA in_proj and dense/shared GLU fc1 save local fragments bound to the live shard."""
    _require_world_size_multiple(2)
    pg, metadata = _init_parallel(gtp_remat_size=2)
    model, key, weight = _build_module(kind, pg)
    with mock.patch.object(
        dist, 'all_gather_into_tensor', side_effect=AssertionError('checkpoint gathered')
    ):
        sd = model.sharded_state_dict(metadata=metadata)
    assert is_gtp_param(weight)
    factory = sd[key]
    assert factory.for_optimizer().data is weight
    chunks = factory.build()
    assert (
        sum(part.data.numel() for part in chunks if isinstance(part, ShardedTensor))
        <= weight.numel()
    )
    for part in chunks:
        if isinstance(part, ShardedTensor):
            assert part.data.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    indexed = get_param_id_to_sharded_param_map(sd, [weight])
    assert indexed[0].key == factory.key
    assert indexed[0].data is weight


def _gathered_logical(weight):
    """The TP-local logical weight: all GTP shards gathered, alignment padding stripped."""
    shards = [torch.empty_like(weight.data) for _ in range(weight.group.size())]
    dist.all_gather(shards, weight.data.contiguous(), group=weight.group)
    logical_rows = weight.shape[0] * weight.group.size() - weight.pad_length
    return torch.cat(shards)[:logical_rows].clone()


@pytest.mark.parametrize('kind', ['kda', 'dense', 'shared'])
def test_real_module_loads_across_gtp(kind, _gtp_alignment, tmp_path_dist_ckpt):
    """GTP2 checkpoints load without GTP and non-GTP checkpoints load into GTP2.

    The second direction is how a GTP run starts from an existing (non-GTP) checkpoint.
    """
    _require_world_size_multiple(2)

    def projection_only(model, metadata, key):
        return {key: model.sharded_state_dict(metadata=metadata)[key]}

    # GTP save -> non-GTP load.
    pg, metadata = _init_parallel(gtp_remat_size=2)
    model, key, weight = _build_module(kind, pg)
    golden = _gathered_logical(weight)
    with TempNamedDir(tmp_path_dist_ckpt / f'gtp_local_{kind}_gtp_to_plain', sync=True) as path:
        dist_checkpointing.save(projection_only(model, metadata, key), path)
        pg, metadata = _init_parallel(gtp_remat_size=1)
        model, key, weight = _build_module(kind, pg)
        assert not is_gtp_param(weight)
        with torch.no_grad():
            weight.fill_(-999.0)
        loaded = dist_checkpointing.load(projection_only(model, metadata, key), path)
    torch.testing.assert_close(loaded[key], golden, rtol=0, atol=0)

    # Non-GTP save -> GTP load.
    pg, metadata = _init_parallel(gtp_remat_size=1)
    model, key, weight = _build_module(kind, pg)
    golden = weight.detach().clone()
    with TempNamedDir(tmp_path_dist_ckpt / f'gtp_local_{kind}_plain_to_gtp', sync=True) as path:
        dist_checkpointing.save(projection_only(model, metadata, key), path)
        pg, metadata = _init_parallel(gtp_remat_size=2)
        model, key, weight = _build_module(kind, pg)
        assert is_gtp_param(weight)
        with torch.no_grad():
            weight.fill_(-999.0)
        loaded = dist_checkpointing.load(projection_only(model, metadata, key), path)
    local_rows = weight.shape[0]
    start = weight.group.rank() * local_rows
    valid = max(0, min(local_rows, golden.shape[0] - start))
    merged = loaded[key]
    assert tuple(merged.shape) == tuple(weight.shape)
    torch.testing.assert_close(merged[:valid], golden[start : start + valid], rtol=0, atol=0)
    assert torch.equal(merged[valid:], torch.zeros_like(merged[valid:]))
