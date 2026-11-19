# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real MoK next to dense GTP-sharded native modules: dense GTP must not change the numbers.

Kimi-K3 runs routed-only MoK at EP64 with dense GTP8 and expert GTP 1, so a rank's dense GTP peers
are other EP ranks holding other experts. This test reproduces that layout on one four-GPU
Blackwell node: EP4 with dense GTP4, and a latent MoE layer whose latent down-projection, RMSNorm
up-projection and native shared experts surround the megakernel call.

* Phase A builds the layer without GTP (dense DP4) and gives every parameter deterministic values.
* Phase B builds it with dense GTP4 (``tensor_parallel_num_weight_shards=4``), loads the same values
  (its own shard of each GTP weight), and runs the same per-rank microbatches.

Both phases run two microbatches through MCore DDP and the production gradient finalization: the
GTP fence, the DP gradient sync, and the gtp_remat AVG of replicated parameters. Phase B must
reproduce phase A's outputs, input gradients and fully reduced main_grads (GTP shards gathered).
The dense GEMMs see the same gathered weights, the megakernel sees the same routed weights, and both
phases average every gradient over the same four data-parallel token streams.

Run on one GB200/GB300 node inside the MoK container (four ranks):

    torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/moe/test_mok_gtp.py
"""

import inspect

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situlu
from tests.unit_tests.test_utilities import Utils

EP = 4
GTP = 4
HIDDEN = 512
LATENT = 256  # MoK's hidden size: divisible by 256
FFN = 256  # routed intermediate size: divisible by 256
SHARED_FFN = 512  # native shared experts, as Kimi-K3's 2 x routed width
NUM_EXPERTS = 8
TOPK = 2
TOKENS = 512  # MoK needs at least 512 local tokens, divisible by 256
MICROBATCHES = 2


def _require_mok_node(mxfp8):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MoK requires Blackwell SM100/SM103")
    if torch.cuda.device_count() < EP:
        pytest.skip(f"Requires {EP} CUDA devices")
    if Utils.world_size != EP:
        pytest.skip(f"Run with torchrun --nproc-per-node {EP}")
    functional = pytest.importorskip("mok.functional")
    if "situ_beta" not in inspect.signature(functional.forward).parameters:
        pytest.skip("MoK build with routed-only and SiTU support is required")
    te = pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

    if not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    if mxfp8:
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(f"MXFP8 not available: {reason}")


def _config(*, gtp, mxfp8, recompute, latent_opt_in):
    """Kimi-K3's MoE structure at the sizes of the MoK EP4 replay test (test_moe_kernels.py)."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    precision = dict(fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True) if mxfp8 else {}
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=FFN,
        moe_shared_expert_intermediate_size=SHARED_FFN,
        moe_latent_size=LATENT,
        moe_latent_up_projection_rmsnorm=True,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=situlu,
        situ_glu_beta1=2.0,
        situ_glu_beta2=8.0,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
        # Dense GTP only: expert GTP stays 1 (MoK reads routed weights itself).
        tensor_parallel_num_weight_shards=gtp,
        gtp_remat_opt_in_modules=["moe_latent_proj"] if latent_opt_in else [],
        recompute_granularity="selective" if recompute else None,
        recompute_modules=[recompute] if recompute else None,
        moe_megakernel_backend="mok",
        moe_megakernel_backend_config={
            "shared_expert_mode": "native",
            "fwd_num_comm_sms": 2,
            "bwd_num_comm_sms": 2,
            "minibatch_size": 512,
            "macrobatch_size": 4096,
            "schedule_capacity_multiplier": 1.5,
        },
        **precision,
    )


def _deterministic_weights(layer, ep_rank):
    """BF16 values for every parameter: dense ones equal on all ranks, experts per EP rank."""
    weights = {}
    for index, (name, param) in enumerate(sorted(layer.named_parameters())):
        is_expert = not getattr(param, "allreduce", True)
        generator = torch.Generator(device="cuda").manual_seed(
            9000 + index + (1_000_000 * (1 + ep_rank) if is_expert else 0)
        )
        value = torch.randn(tuple(param.shape), generator=generator, device="cuda") / 16
        if param.dim() == 1:  # the RMSNorm scale of the latent up-projection
            value = value + 1
        weights[name] = value.to(torch.bfloat16)
    return weights


def _full_main_grad(param, gtp_group):
    """FP32 copy of a reduced gradient; GTP shards are gathered and stripped of padding."""
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param

    grad = param.main_grad.detach().float()
    if not is_gtp_param(param):
        return grad.clone()
    parts = [torch.empty_like(grad) for _ in range(gtp_group.size())]
    torch.distributed.all_gather(parts, grad.contiguous(), group=gtp_group)
    full = torch.cat(parts)
    return full[: full.shape[0] - param.pad_length].clone()


def _run_phase(*, gtp, mxfp8, recompute, latent_opt_in, batches, weights, monkeypatch):
    """Build, load, run two microbatches and finalize gradients like production."""
    from mok import functional

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_replicated_grads_over_gtp_remat_group,
    )
    from megatron.core.fp8_utils import get_fp8_context, is_float8tensor
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.gtp_api import (
        GTP_CONFIG,
        classify_gtp_remat_chains,
        gtp_native_fp8_load_context,
        is_gtp_param,
        wait_for_gtp_grad_reduction_on_current_stream,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.utils import resolve_gtp_pad_for_alignment

    Utils.initialize_model_parallel(expert_model_parallel_size=EP, gtp_remat_size=gtp)
    # The shard alignment configure_gtp_remat_from_recipe sets before a production model build.
    monkeypatch.setattr(
        GTP_CONFIG,
        "pad_for_alignment",
        resolve_gtp_pad_for_alignment(fp8_recipe="mxfp8" if mxfp8 else None, fp8=mxfp8),
    )
    model_parallel_cuda_manual_seed(1234)
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    config = _config(gtp=gtp, mxfp8=mxfp8, recompute=recompute, latent_opt_in=latent_opt_in)
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=NUM_EXPERTS, moe_grouped_gemm=True
        ).mlp
    )
    with get_fp8_context(config, is_init=True):
        layer = MoELayer(config, submodules, layer_number=1, pg_collection=pgc)

    assert config.gtp_weight_remat_size == gtp and config.expert_gtp_weight_remat_size == 1
    assert not layer.megakernel_fuses_shared_expert
    assert not any(is_gtp_param(p) for p in layer.megakernel_experts.parameters())
    assert not is_gtp_param(layer.router.weight)
    assert not is_gtp_param(layer.fc2_latent_proj.layer_norm_weight)
    for weight in (layer.shared_experts.linear_fc1.weight, layer.shared_experts.linear_fc2.weight):
        assert is_gtp_param(weight) == (gtp > 1)
    for weight in (layer.fc1_latent_proj.weight, layer.fc2_latent_proj.weight):
        assert is_gtp_param(weight) == (gtp > 1 and latent_opt_in)

    if weights is None:
        weights = _deterministic_weights(layer, pgc.ep.rank())
    gtp_rank = pgc.gtp_remat.rank() if gtp > 1 else 0
    with torch.no_grad(), gtp_native_fp8_load_context(layer):
        for name, param in layer.named_parameters():
            value = weights[name]
            if is_gtp_param(param):
                rows = param.shape[0]
                value = F.pad(value, (0, 0, 0, rows * gtp - value.shape[0]))
                value = value[gtp_rank * rows : (gtp_rank + 1) * rows]
            if is_float8tensor(param):
                param.copy_(value)  # re-quantizes into the native MXFP8 storage
            else:
                param.data.copy_(value)

    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            fp8_param_gather=mxfp8,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
        ),
        layer,
        pg_collection=pgc,
    )
    if gtp > 1:
        # As in production, after the DDP wrap and before the first forward.
        classify_gtp_remat_chains(ddp)

    ddp.zero_grad_buffer()
    outputs, input_grads = [], []
    for x, upstream in batches:
        x = x.clone().requires_grad_()
        with get_fp8_context(config):
            output, _ = ddp(x)
        output.backward(upstream)
        outputs.append(output.detach().float())
        input_grads.append(x.grad.detach().float())

    # finalize_model_grads' order: GTP fence, DP sync, gtp_remat AVG of replicated params.
    if gtp > 1:
        wait_for_gtp_grad_reduction_on_current_stream()
    ddp.finish_grad_sync()
    _allreduce_replicated_grads_over_gtp_remat_group([ddp], pgc.gtp_remat, pgc.expt_gtp_remat)
    main_grads = {
        name: _full_main_grad(param, pgc.gtp_remat) for name, param in layer.named_parameters()
    }
    torch.cuda.synchronize()
    # Collective over the EP group's symmetric memory: every rank, before the groups go away.
    functional.clear_workspace_cache()
    return weights, outputs, input_grads, main_grads


def _assert_matches(actual, expected, name, tolerance):
    actual, expected = actual.float(), expected.float()
    assert actual.shape == expected.shape, f"{name}: {tuple(actual.shape)} vs {expected.shape}"
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    delta = (actual - expected).abs()
    normalized_l1 = (delta.sum() / expected.abs().sum().clamp_min(1e-12)).item()
    assert normalized_l1 <= tolerance, (
        f"{name}: GTP run differs from the no-GTP run, normalized L1 {normalized_l1:.3e} > "
        f"{tolerance:.0e} (max abs {delta.max().item():.3e})"
    )


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("precision", "recompute", "latent_opt_in"),
    [
        pytest.param("bf16", None, False, id="bf16"),
        pytest.param("bf16", None, True, id="bf16-latent_gtp"),
        pytest.param("mxfp8", None, False, id="mxfp8"),
        pytest.param("mxfp8", None, True, id="mxfp8-latent_gtp"),
        # The Kimi-K3 GTP8 recipe: MoE recompute off, native shared experts recomputed.
        pytest.param("mxfp8", "shared_experts", True, id="mxfp8-latent_gtp-shared_recompute"),
        pytest.param("mxfp8", "moe", True, id="mxfp8-latent_gtp-moe_recompute"),
    ],
)
def test_mok_with_dense_gtp_matches_no_gtp(precision, recompute, latent_opt_in, monkeypatch):
    """BF16 recompute is not covered: on this branch MCore's native checkpoint (used for BF16) is
    not recognized as a GTP recompute phase (upstream 93fbedfd8), independently of MoK."""
    mxfp8 = precision == "mxfp8"
    _require_mok_node(mxfp8)
    from mok import functional

    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        _GTP_PARAMS,
        reset_gtp_state,
    )

    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(4321 + rank)
    batches = [
        (
            torch.randn(TOKENS, 1, HIDDEN, generator=generator, device="cuda").to(torch.bfloat16),
            (torch.randn(TOKENS, 1, HIDDEN, generator=generator, device="cuda") / 16).to(
                torch.bfloat16
            ),
        )
        for _ in range(MICROBATCHES)
    ]
    common = dict(
        mxfp8=mxfp8,
        recompute=recompute,
        latent_opt_in=latent_opt_in,
        batches=batches,
        monkeypatch=monkeypatch,
    )
    try:
        weights, ref_outputs, ref_input_grads, ref_main_grads = _run_phase(
            gtp=1, weights=None, **common
        )
        _, outputs, input_grads, main_grads = _run_phase(gtp=GTP, weights=weights, **common)
    finally:
        try:
            # A phase that failed after building its workspace must not leave it cached for a
            # later test, whose groups will differ (a no-op after a successful phase).
            functional.clear_workspace_cache()
        finally:
            reset_gtp_state()
            _GTP_PARAMS.clear()
            Utils.destroy_model_parallel()

    tolerance = 3e-2 if mxfp8 else 1e-2
    for microbatch in range(MICROBATCHES):
        _assert_matches(
            outputs[microbatch], ref_outputs[microbatch], f"output[{microbatch}]", tolerance
        )
        _assert_matches(
            input_grads[microbatch],
            ref_input_grads[microbatch],
            f"input gradient[{microbatch}]",
            tolerance,
        )
    assert set(main_grads) == set(ref_main_grads)
    for name, expected in ref_main_grads.items():
        assert expected.abs().sum() > 0, f"{name}: no gradient reached it; the test is vacuous"
        _assert_matches(main_grads[name], expected, f"main_grad {name}", tolerance)
