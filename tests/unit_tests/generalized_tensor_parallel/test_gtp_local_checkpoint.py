# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP local checkpoint fragments on GPUs: DCP topology changes and real Kimi-K3 modules.

Adapted from PR #8109 (eb0f8e0f6). The last section runs the production optimizer end to
end: a GPT model whose dense GLU fc1 is GTP-sharded, the LayerWise Muon optimizer that
Kimi-K3 trains with, and model plus optimizer saved and loaded together through DCP. The
CPU-only layout, companion and master-index tests are in
tests/unit_tests/dist_checkpointing/test_gtp_local_fragments.py. Run on 4 GPUs:

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
from megatron.core.dist_checkpointing.dict_utils import nested_values  # noqa: E402
from megatron.core.dist_checkpointing.gtp import gtp_logical_rows  # noqa: E402
from megatron.core.dist_checkpointing.mapping import (  # noqa: E402
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.optimizer import (  # noqa: E402
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.dist_checkpointing.serialization import (  # noqa: E402
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (  # noqa: E402
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (  # noqa: E402
    GTP_CONFIG,
    GTPShardedParam,
    update_gtp_config,
)
from megatron.core.tensor_parallel.gtp_api import is_gtp_param  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from tests.unit_tests.dist_checkpointing import (  # noqa: E402
    TempNamedDir,
    setup_model_and_optimizer,
)
from tests.unit_tests.dist_checkpointing.test_gtp_local_fragments import (  # noqa: E402
    make_projection,
)
from tests.unit_tests.dist_checkpointing.utils import HAVE_NVRX  # noqa: E402
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: E402,F401
    _torchrun_dist_init,
    reset_gtp_globals,
)
from tests.unit_tests.test_utilities import Utils  # noqa: E402

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
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=gtp_remat_size
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


@pytest.mark.parametrize('kind', ['kda', 'dense', 'shared'])
@pytest.mark.parametrize('source,target', [(2, 1), (1, 2)])
def test_real_module_optimizer_state_loads_across_gtp(
    kind, source, target, _gtp_alignment, tmp_path_dist_ckpt
):
    """A moment of the real KDA in_proj and GLU fc1 loads across GTP and non-GTP runs.

    The moment is mapped the way the Float16 optimizer maps its state
    (get_param_id_to_sharded_param_map, make_sharded_optimizer_tensor): through the companion
    under GTP, through the existing factories without it. Its values differ from the weight
    but are equal on data-parallel replicas; pad rows are not saved and load as zeros.
    """
    _require_world_size_multiple(2)

    def momentum_entry(model, metadata, key, weight, state):
        model_sd = {key: model.sharded_state_dict(metadata=metadata)[key]}
        param_map = get_param_id_to_sharded_param_map(model_sd, [weight])
        return make_sharded_optimizer_tensor(param_map[0], state, 'optimizer.state.momentum')

    pg, metadata = _init_parallel(gtp_remat_size=source)
    model, key, weight = _build_module(kind, pg)
    assert is_gtp_param(weight) == (source > 1)
    golden = _logical_weight(weight).float() * 3 + 0.5
    name = f'gtp_local_{kind}_momentum_gtp{source}_to_gtp{target}'
    with TempNamedDir(tmp_path_dist_ckpt / name, sync=True) as path:
        state = weight.detach().float() * 3 + 0.5
        entry = momentum_entry(model, metadata, key, weight, state)
        dist_checkpointing.save({'momentum': entry}, path)

        pg, metadata = _init_parallel(gtp_remat_size=target)
        model, key, weight = _build_module(kind, pg)
        assert is_gtp_param(weight) == (target > 1)
        state = torch.full(weight.shape, -999.0, dtype=torch.float32, device=weight.device)
        entry = momentum_entry(model, metadata, key, weight, state)
        loaded = dist_checkpointing.load({'momentum': entry}, path)
    gtp_rank = weight.group.rank() if target > 1 else 0
    expected = _physical_shard(golden, weight.shape[0], gtp_rank)
    _assert_equal(loaded['momentum'], expected, f'{kind} momentum')


# ---------------------------------------------------------------------------------------------
# The production optimizer end to end: LayerWise Muon, model and optimizer through DCP
# ---------------------------------------------------------------------------------------------

_GPT_LAYERS = 2
# Checkpoint key of the GLU fc1 weight; the state-dict keys carry the layer index instead.
_FC1_KEY = 'decoder.layers.mlp.linear_fc1.weight'


@pytest.fixture
def _muon_checkpoint_run(run_config):
    """Skip without the Muon or checkpoint-test packages; restore the default parallel state."""
    from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS

    if not HAVE_EMERGING_OPTIMIZERS:
        pytest.skip("Muon needs the emerging-optimizers package")
    if not HAVE_NVRX:
        pytest.skip("setup_model_and_optimizer needs nvidia-resiliency-ext")
    yield
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel()


def _set_gtp_degree(gtp_remat_size):
    """Re-initialize model parallelism as TP1/PP1 with the given GTP degree."""
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=gtp_remat_size)
    model_parallel_cuda_manual_seed(2)


def _initialize_glu_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    """Dense GPT with a gated MLP on the TE spec, built by ``setup_model_and_optimizer``.

    Every GTP-sharded output dimension (QKV 384, attention projection and fc2 128, fc1 512,
    vocabulary 128) is a multiple of 64, so GTP2 shards carry no alignment padding (16 or 32
    rows per shard). Weights without an optimizer companion then keep the same checkpoint
    shape at every GTP degree, and the GTP-degree test can load the whole optimizer.
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Passed through training.get_model but not part of TransformerConfig.
    config_kwargs.pop('pg_collection', None)
    config_kwargs.pop('config', None)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=_GPT_LAYERS,
        hidden_size=128,
        num_attention_heads=8,
        kv_channels=16,
        ffn_hidden_size=256,
        gated_linear_unit=True,
        add_bias_linear=False,
        use_cpu_initialization=False,
        **config_kwargs,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
    )
    model.bfloat16()
    with torch.no_grad():
        for param in model.parameters():
            param.random_()
    return model


def _build_glu_gpt(seed, *, with_distributed_optimizer):
    """Build the model and its LayerWise Muon optimizer; return (unwrapped model, optimizer).

    ``with_distributed_optimizer`` selects the Kimi-K3 layout: LayerWise Muon for the matrices
    next to a DistributedOptimizer for embeddings and norms. Without it LayerWise holds every
    parameter, so all optimizer state is model-space and can change data-parallel size.
    """
    model, optimizer = setup_model_and_optimizer(
        seed=seed,
        tp=1,
        pp=1,
        bf16=True,
        dist_opt=with_distributed_optimizer,
        use_param_layout=with_distributed_optimizer,
        initialize_fn=_initialize_glu_gpt_model,
        optimizer='dist_muon',
    )
    return model[0], optimizer


def _fc1_weights(model):
    return [layer.mlp.linear_fc1.weight for layer in model.decoder.layers]


def _float16_optimizers(optimizer):
    """The Float16OptimizerWithFloat16Params instances of a (chained) optimizer, in order."""
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    if isinstance(optimizer, Float16OptimizerWithFloat16Params):
        return [optimizer]
    return [
        child
        for member in getattr(optimizer, 'chained_optimizers', ())
        for child in _float16_optimizers(member)
    ]


def _main_param_states(optimizer):
    """Yield (model param, {'fp32_param': master, state name: moment}) per float16 param.

    LayerWise keeps a param's master and moments only on the data-parallel rank that owns it.
    """
    for child in _float16_optimizers(optimizer):
        if child.is_stub_optimizer:
            continue
        for model_group, main_group in zip(child.float16_groups, child.fp32_from_float16_groups):
            for param, main_param in zip(model_group, main_group):
                tensors = {'fp32_param': main_param}
                for name, value in child.optimizer.state[main_param].items():
                    if torch.is_tensor(value) and value.shape == main_param.shape:
                        tensors[name] = value
                yield param, tensors


def _perturb_optimizer_state(optimizer):
    """Make every master differ from its weight and every moment nonzero.

    Masters start as copies of the weights and Muon momentum starts at zero, so a load that
    returned the weights or zeros would otherwise pass.
    """
    with torch.no_grad():
        for _, tensors in _main_param_states(optimizer):
            for value in tensors.values():
                value.add_(torch.randn_like(value))


def _optimizer_snapshot(optimizer):
    """CPU copies of this rank's masters and moments, in optimizer order."""
    return [
        {name: value.detach().cpu().clone() for name, value in tensors.items()}
        for _, tensors in _main_param_states(optimizer)
    ]


def _owned_fc1_state(model, optimizer):
    """{layer: {name: CPU tensor}} for the fc1 weights whose optimizer state this rank owns."""
    layers = {id(weight): layer for layer, weight in enumerate(_fc1_weights(model))}
    return {
        layers[id(param)]: {name: value.detach().cpu().clone() for name, value in tensors.items()}
        for param, tensors in _main_param_states(optimizer)
        if id(param) in layers
    }


def _logical_weight(weight):
    """The TP-local logical weight on CPU: gathered over GTP, alignment padding stripped."""
    if is_gtp_param(weight):
        return _gathered_logical(weight).cpu()
    return weight.detach().cpu().clone()


def _logical_fc1_state(model, optimizer):
    """Every layer's TP-local logical fc1 master and moments, assembled from all ranks.

    Each GTP shard's state lives on exactly one data-parallel rank: gather the owned shards
    everywhere, check that every shard has exactly one owner, and stitch the shards together.
    """
    weights = _fc1_weights(model)
    gtp = is_gtp_param(weights[0])
    gtp_size = weights[0].group.size() if gtp else 1
    gtp_rank = weights[0].group.rank() if gtp else 0
    owned = {
        (layer, gtp_rank): state for layer, state in _owned_fc1_state(model, optimizer).items()
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, owned)
    shards = {}
    for rank_shards in gathered:
        for shard, state in rank_shards.items():
            assert shard not in shards, f'fc1 shard (layer, GTP rank) {shard} has two owners'
            shards[shard] = state
    expected = {(layer, rank) for layer in range(len(weights)) for rank in range(gtp_size)}
    assert set(shards) == expected, sorted(shards)
    logical = {}
    for layer, weight in enumerate(weights):
        rows = gtp_logical_rows(weight) if gtp else weight.shape[0]
        parts = [shards[(layer, rank)] for rank in range(gtp_size)]
        logical[layer] = {
            name: torch.cat([part[name] for part in parts])[:rows] for name in parts[0]
        }
    return logical


def _physical_shard(logical, rows, gtp_rank):
    """GTP shard ``gtp_rank`` of ``rows`` rows: its slice of ``logical``, then a zero tail."""
    shard = logical.new_zeros((rows,) + tuple(logical.shape[1:]))
    start = gtp_rank * rows
    valid = max(0, min(rows, logical.shape[0] - start))
    shard[:valid] = logical[start : start + valid]
    return shard


def _assert_equal(actual, expected, what):
    """Bitwise equality with a readable failure message."""
    actual = actual.cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (
        what,
        tuple(actual.shape),
        tuple(expected.shape),
        actual.dtype,
        expected.dtype,
    )
    differ = int((actual != expected).sum())
    assert differ == 0, f'{what}: {differ} of {expected.numel()} elements differ'


def _master_groups(state_dict):
    """Each per-group container of ``fp32_from_fp16_params`` in a (chained) optimizer state."""
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            if key == 'fp32_from_fp16_params' and isinstance(value, dict):
                yield from value.values()
            else:
                yield from _master_groups(value)
    elif isinstance(state_dict, list):
        for value in state_dict:
            yield from _master_groups(value)


def _is_fc1_entry(value):
    return isinstance(value, ShardedTensorFactory) and value.key.endswith(_FC1_KEY)


def _assert_fc1_optimizer_entries(optimizer_sharded_sd, num_shards):
    """fc1 masters and moments go through the companion, and masters are keyed by index.

    Over all ranks there is exactly one master entry per (layer, GTP shard).
    """
    entries = [value for value in nested_values(optimizer_sharded_sd) if _is_fc1_entry(value)]
    masters = sum('optimizer.state.fp32_param.' in entry.key for entry in entries)
    counts = torch.tensor([masters, len(entries) - masters], device='cuda')
    dist.all_reduce(counts)
    assert counts[0].item() == num_shards, counts
    assert counts[1].item() >= num_shards, counts
    for group in _master_groups(optimizer_sharded_sd):
        if isinstance(group, dict):
            values = list(group.values())
        elif isinstance(group, list):
            values = group
        else:  # an empty group, which stays out of the checkpoint
            continue
        if any(_is_fc1_entry(value) for value in values):
            assert isinstance(group, dict), type(group)
            assert all(isinstance(index, int) for index in group), list(group)


def _assert_checkpoint_holds_logical_fc1(plain, logical_weights, logical_state):
    """The checkpoint's fc1 weight, master and moments are the logical (non-GTP) tensors."""
    for layer, weight in enumerate(logical_weights):
        _assert_equal(plain[_FC1_KEY][layer], weight, f'checkpoint fc1 weight, layer {layer}')
    suffix = f'.{_FC1_KEY}'
    state_keys = {
        key.split('optimizer.state.', 1)[1][: -len(suffix)]: key
        for key in plain
        if key.endswith(suffix) and 'optimizer.state.' in key
    }
    assert set(state_keys) == set(logical_state[0]), (sorted(state_keys), sorted(logical_state[0]))
    for name, key in state_keys.items():
        for layer, state in logical_state.items():
            _assert_equal(plain[key][layer], state[name], f'checkpoint {key}, layer {layer}')


def test_layerwise_muon_checkpoint_roundtrip_with_gtp_glu(_muon_checkpoint_run, tmp_path_dist_ckpt):
    """GTP2 GLU fc1 masters and Muon moments round-trip through the Kimi-K3 optimizer layout.

    LayerWise Muon next to a DistributedOptimizer (dp_reshardable), with model and optimizer
    saved together through DCP and the fully parallel save wrapper, as training does. A fresh
    model and optimizer must restore every master and moment bit-exactly, the checkpoint must
    hold the logical (non-GTP) fc1 tensors, and re-saving the loaded optimizer must reproduce
    the optimizer checkpoint.
    """
    _require_world_size_multiple(2)
    gtp_size = 2
    metadata = {'distrib_optim_sharding_type': 'dp_reshardable'}
    _set_gtp_degree(gtp_size)
    model_a, optimizer_a = _build_glu_gpt(seed=2, with_distributed_optimizer=True)
    weights_a = _fc1_weights(model_a)
    assert all(is_gtp_param(weight) for weight in weights_a)
    _perturb_optimizer_state(optimizer_a)
    expected_state = _optimizer_snapshot(optimizer_a)
    shards_a = [weight.detach().cpu().clone() for weight in weights_a]
    logical_weights = [_logical_weight(weight) for weight in weights_a]
    logical_state = _logical_fc1_state(model_a, optimizer_a)

    model_sd_a = model_a.sharded_state_dict()
    for layer, weight in enumerate(weights_a):
        entry = model_sd_a[f'decoder.layers.{layer}.mlp.linear_fc1.weight']
        assert isinstance(entry, ShardedTensorFactory) and entry.key == _FC1_KEY, entry
        assert entry.for_optimizer().data is weight
    optim_sd_a = optimizer_a.sharded_state_dict(model_sd_a, metadata=metadata)
    _assert_fc1_optimizer_entries(optim_sd_a, num_shards=_GPT_LAYERS * gtp_size)

    with (
        TempNamedDir(tmp_path_dist_ckpt / 'gtp_glu_muon_a', sync=True) as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / 'gtp_glu_muon_b', sync=True) as ckpt_b,
    ):
        # Training saves through this wrapper over the data-parallel group (GTP included).
        save_strategy = FullyParallelSaveStrategyWrapper(
            get_default_save_sharded_strategy(),
            ps.get_data_parallel_group(with_context_parallel=True),
        )
        dist_checkpointing.save(
            {'model': model_sd_a, 'optimizer': optim_sd_a}, ckpt_a, save_strategy
        )

        model_b, optimizer_b = _build_glu_gpt(seed=3, with_distributed_optimizer=True)
        model_sd_b = model_b.sharded_state_dict()
        optim_template_b = optimizer_b.sharded_state_dict(
            model_sd_b, is_loading=True, metadata=metadata
        )
        loaded = dist_checkpointing.load(
            {'model': model_sd_b, 'optimizer': optim_template_b}, ckpt_a
        )
        optimizer_b.load_state_dict(loaded['optimizer'])
        for layer, shard in enumerate(shards_a):
            loaded_shard = loaded['model'][f'decoder.layers.{layer}.mlp.linear_fc1.weight']
            _assert_equal(loaded_shard, shard, f'loaded fc1 weight, layer {layer}')
        restored_state = _optimizer_snapshot(optimizer_b)
        assert len(restored_state) == len(expected_state)
        for index, (restored, expected) in enumerate(zip(restored_state, expected_state)):
            assert restored.keys() == expected.keys(), (index, sorted(restored), sorted(expected))
            for name in expected:
                _assert_equal(restored[name], expected[name], f'restored param {index} {name}')

        model_sd_b = model_b.sharded_state_dict()
        dist_checkpointing.save(
            optimizer_b.sharded_state_dict(model_sd_b, metadata=metadata), ckpt_b
        )

        _set_gtp_degree(1)
        plain_a = dist_checkpointing.load_plain_tensors(ckpt_a)
        plain_b = dist_checkpointing.load_plain_tensors(ckpt_b)

    _assert_checkpoint_holds_logical_fc1(plain_a, logical_weights, logical_state)
    optimizer_keys = {key for key in plain_a if key.startswith('chained_')}
    assert optimizer_keys == set(plain_b), sorted(optimizer_keys ^ set(plain_b))
    for key in sorted(optimizer_keys):
        _assert_equal(plain_b[key], plain_a[key].cpu(), f're-saved {key}')


@pytest.mark.parametrize('source_gtp,target_gtp', [(2, 1), (1, 2)])
def test_layerwise_muon_fc1_state_loads_across_gtp_degrees(
    source_gtp, target_gtp, _muon_checkpoint_run, tmp_path_dist_ckpt
):
    """fc1 weight, master and Muon moments load across GTP degrees, in both directions.

    GTP2 -> GTP1 loads a GTP checkpoint without GTP; GTP1 -> GTP2 starts a GTP run from a
    non-GTP checkpoint. LayerWise holds every parameter here: the DistributedOptimizer of the
    Kimi-K3 layout saves dp_reshardable state, which cannot change data-parallel size, and a
    GTP-degree change at a fixed world size changes it.
    """
    _require_world_size_multiple(2)
    _set_gtp_degree(source_gtp)
    model, optimizer = _build_glu_gpt(seed=2, with_distributed_optimizer=False)
    weights = _fc1_weights(model)
    assert all(is_gtp_param(weight) == (source_gtp > 1) for weight in weights)
    _perturb_optimizer_state(optimizer)
    logical_weights = [_logical_weight(weight) for weight in weights]
    logical_state = _logical_fc1_state(model, optimizer)
    ckpt_name = f'gtp_glu_muon_gtp{source_gtp}_to_gtp{target_gtp}'
    with TempNamedDir(tmp_path_dist_ckpt / ckpt_name, sync=True) as ckpt_dir:
        model_sd = model.sharded_state_dict()
        dist_checkpointing.save(
            {'model': model_sd, 'optimizer': optimizer.sharded_state_dict(model_sd)}, ckpt_dir
        )

        _set_gtp_degree(target_gtp)
        model, optimizer = _build_glu_gpt(seed=3, with_distributed_optimizer=False)
        weights = _fc1_weights(model)
        assert all(is_gtp_param(weight) == (target_gtp > 1) for weight in weights)
        model_sd = model.sharded_state_dict()
        optim_template = optimizer.sharded_state_dict(model_sd, is_loading=True)
        loaded = dist_checkpointing.load({'model': model_sd, 'optimizer': optim_template}, ckpt_dir)
        optimizer.load_state_dict(loaded['optimizer'])

    gtp_rank = weights[0].group.rank() if target_gtp > 1 else 0
    for layer, weight in enumerate(weights):
        expected = _physical_shard(logical_weights[layer], weight.shape[0], gtp_rank)
        loaded_weight = loaded['model'][f'decoder.layers.{layer}.mlp.linear_fc1.weight']
        _assert_equal(loaded_weight, expected, f'loaded fc1 weight, layer {layer}')
    owned = _owned_fc1_state(model, optimizer)
    for layer, state in owned.items():
        assert state.keys() == logical_state[layer].keys(), (
            sorted(state),
            sorted(logical_state[layer]),
        )
        for name, value in state.items():
            expected = _physical_shard(
                logical_state[layer][name], weights[layer].shape[0], gtp_rank
            )
            _assert_equal(value, expected, f'loaded fc1 {name}, layer {layer}')
    owned_count = torch.tensor([len(owned)], device='cuda')
    dist.all_reduce(owned_count)
    assert owned_count.item() == _GPT_LAYERS * target_gtp, owned_count
