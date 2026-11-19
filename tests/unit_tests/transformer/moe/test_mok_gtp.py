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

The option-E cases run phase B with MoK forward-context recompute
(``moe_mok_recompute_forward_context``), as the Kimi-K3 option-E GTP recipes do: each microbatch's
MoE forward drops MoK's saved context and its backward rebuilds it, next to GTP's gathers and
reduce-scatters and the te_checkpoint replay of the shared experts. Phase A stays the option-off
reference. A phase C then runs the same GTP4 layer with the option off, and phase B must equal it
bit for bit in outputs, input gradients and routed-expert gradients: the rebuilt context equals the
dropped one, and none of these passes an NCCL reduction at EP4 on four GPUs (dense dgrads use
exactly gathered weights, and expert DP is 1). Dense main_grads are reduce-scattered, so they are
compared with phase A's at the tolerance only.

Run on one GB200/GB300 node inside the MoK container (four ranks):

    torchrun --nproc-per-node 4 -m pytest -q tests/unit_tests/transformer/moe/test_mok_gtp.py
"""

import contextlib
import inspect
from typing import NamedTuple
from unittest.mock import patch

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


def _require_mok_node(mxfp8, option_e=False):
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
    rebuild = getattr(functional, "recompute_forward_context", None)
    if option_e and (rebuild is None or "situ_beta" not in inspect.signature(rebuild).parameters):
        pytest.skip("MoK build with a SiTU recompute_forward_context (d59188a) is required")
    te = pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

    if not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    if mxfp8:
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(f"MXFP8 not available: {reason}")


def _config(*, gtp, mxfp8, recompute, latent_opt_in, option_e=False):
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
        # Option E: drop MoK's forward context after forward and rebuild it in backward.
        moe_mok_recompute_forward_context=option_e,
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


class _Phase(NamedTuple):
    weights: dict
    outputs: list
    input_grads: list
    main_grads: dict
    # Routed-expert parameters: MoK alone computes their gradients (expert DP is 1 here).
    expert_names: frozenset


def _run_phase(
    *, gtp, mxfp8, recompute, latent_opt_in, batches, weights, monkeypatch, option_e=False
):
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
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        _GTP_PARAMS,
        reset_gtp_state,
    )
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

    # Each phase builds a fresh model: drop an earlier GTP phase's chain state and parameters.
    reset_gtp_state()
    _GTP_PARAMS.clear()
    Utils.initialize_model_parallel(expert_model_parallel_size=EP, gtp_remat_size=gtp)
    # The shard alignment configure_gtp_remat_from_recipe sets before a production model build.
    monkeypatch.setattr(
        GTP_CONFIG,
        "pad_for_alignment",
        resolve_gtp_pad_for_alignment(fp8_recipe="mxfp8" if mxfp8 else None, fp8=mxfp8),
    )
    model_parallel_cuda_manual_seed(1234)
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    config = _config(
        gtp=gtp, mxfp8=mxfp8, recompute=recompute, latent_opt_in=latent_opt_in, option_e=option_e
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=NUM_EXPERTS, moe_grouped_gemm=True
        ).mlp
    )
    with get_fp8_context(config, is_init=True):
        layer = MoELayer(config, submodules, layer_number=1, pg_collection=pgc)

    assert config.gtp_weight_remat_size == gtp and config.expert_gtp_weight_remat_size == 1
    assert not layer.megakernel_fuses_shared_expert
    assert layer.megakernel_experts.recompute_forward_context is option_e
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

    # Count MoK's context rebuilds. Installed after the build: with option E on, the backend checks
    # the real function's signature when it is constructed.
    rebuilds = []
    real_rebuild = getattr(functional, "recompute_forward_context", None)
    count_rebuilds = contextlib.nullcontext()
    if real_rebuild is not None:

        def counting_rebuild(*args, **kwargs):
            rebuilds.append(None)
            return real_rebuild(*args, **kwargs)

        count_rebuilds = patch.object(functional, "recompute_forward_context", counting_rebuild)

    ddp.zero_grad_buffer()
    outputs, input_grads = [], []
    with count_rebuilds:
        for x, upstream in batches:
            x = x.clone().requires_grad_()
            with get_fp8_context(config):
                output, _ = ddp(x)
            output.backward(upstream)
            outputs.append(output.detach().float())
            input_grads.append(x.grad.detach().float())
    # Option E rebuilds the one MoK layer's context once per microbatch, in its backward (the
    # shared-expert checkpoint replays only the shared experts); without it, never.
    assert len(rebuilds) == (len(batches) if option_e else 0), len(rebuilds)

    # finalize_model_grads' order: GTP fence, DP sync, gtp_remat AVG of replicated params.
    if gtp > 1:
        wait_for_gtp_grad_reduction_on_current_stream()
    ddp.finish_grad_sync()
    _allreduce_replicated_grads_over_gtp_remat_group([ddp], pgc.gtp_remat, pgc.expt_gtp_remat)
    main_grads = {
        name: _full_main_grad(param, pgc.gtp_remat) for name, param in layer.named_parameters()
    }
    expert_names = frozenset(
        name for name, param in layer.named_parameters() if not getattr(param, "allreduce", True)
    )
    torch.cuda.synchronize()
    # Collective over the EP group's symmetric memory: every rank, before the groups go away.
    functional.clear_workspace_cache()
    return _Phase(weights, outputs, input_grads, main_grads, expert_names)


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


def _assert_bitwise_equal(actual, expected, name):
    """Option E against its option-off twin on the same GTP layer: equal bit for bit."""
    assert actual.dtype == expected.dtype == torch.float32, name
    assert actual.shape == expected.shape, f"{name}: {tuple(actual.shape)} vs {expected.shape}"
    differ = int((actual.view(torch.int32) != expected.view(torch.int32)).sum().item())
    assert differ == 0, (
        f"{name}: {differ} of {expected.numel()} elements differ bitwise between the GTP run "
        "with MoK forward-context recompute and the same run without it; the rebuilt context "
        "must equal the dropped one"
    )


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("precision", "recompute", "latent_opt_in", "option_e"),
    [
        pytest.param("bf16", None, False, False, id="bf16"),
        pytest.param("bf16", None, True, False, id="bf16-latent_gtp"),
        pytest.param("mxfp8", None, False, False, id="mxfp8"),
        pytest.param("mxfp8", None, True, False, id="mxfp8-latent_gtp"),
        # The Kimi-K3 GTP8 recipes turn MoE recompute off and recompute the native shared
        # experts. F4/P5 keep MoK's forward context and GTP-shard the latent projections.
        pytest.param(
            "mxfp8", "shared_experts", True, False, id="mxfp8-latent_gtp-shared_recompute"
        ),
        # F5/P6 rebuild it in backward instead (option E) and keep the latent projections
        # replicated. The option-off twin tells a failure of that GTP layout from an option-E one.
        pytest.param("mxfp8", "shared_experts", False, True, id="mxfp8-shared_recompute-opte"),
        pytest.param("mxfp8", "shared_experts", False, False, id="mxfp8-shared_recompute"),
        # F5/P6 with the latent opt-in their headers name as the next step.
        pytest.param(
            "mxfp8", "shared_experts", True, True, id="mxfp8-latent_gtp-shared_recompute-opte"
        ),
        pytest.param("mxfp8", "moe", True, False, id="mxfp8-latent_gtp-moe_recompute"),
    ],
)
def test_mok_with_dense_gtp_matches_no_gtp(
    precision, recompute, latent_opt_in, option_e, monkeypatch
):
    """BF16 recompute is not covered: on this branch MCore's native checkpoint (used for BF16) is
    not recognized as a GTP recompute phase (upstream 93fbedfd8), independently of MoK."""
    mxfp8 = precision == "mxfp8"
    _require_mok_node(mxfp8, option_e)
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
        # Phase A, the reference, never uses option E.
        reference = _run_phase(gtp=1, weights=None, **common)
        run = _run_phase(gtp=GTP, weights=reference.weights, option_e=option_e, **common)
        # Phase C: the same GTP layer with MoK's context kept.
        kept = _run_phase(gtp=GTP, weights=reference.weights, **common) if option_e else None
    finally:
        try:
            # A phase that failed after building its workspace must not leave it cached for a
            # later test, whose groups will differ (a no-op after a successful phase).
            functional.clear_workspace_cache()
        finally:
            reset_gtp_state()
            _GTP_PARAMS.clear()
            Utils.destroy_model_parallel()

    if kept is not None:
        # First, so that an option-E failure is not reported as a GTP tolerance miss.
        assert kept.expert_names and kept.expert_names == run.expert_names
        for microbatch in range(MICROBATCHES):
            _assert_bitwise_equal(
                run.outputs[microbatch], kept.outputs[microbatch], f"output[{microbatch}]"
            )
            _assert_bitwise_equal(
                run.input_grads[microbatch],
                kept.input_grads[microbatch],
                f"input gradient[{microbatch}]",
            )
        for name in sorted(kept.expert_names):
            _assert_bitwise_equal(run.main_grads[name], kept.main_grads[name], f"main_grad {name}")

    tolerance = 3e-2 if mxfp8 else 1e-2
    for microbatch in range(MICROBATCHES):
        _assert_matches(
            run.outputs[microbatch],
            reference.outputs[microbatch],
            f"output[{microbatch}]",
            tolerance,
        )
        _assert_matches(
            run.input_grads[microbatch],
            reference.input_grads[microbatch],
            f"input gradient[{microbatch}]",
            tolerance,
        )
    assert set(run.main_grads) == set(reference.main_grads)
    for name, expected in reference.main_grads.items():
        assert expected.abs().sum() > 0, f"{name}: no gradient reached it; the test is vacuous"
        _assert_matches(run.main_grads[name], expected, f"main_grad {name}", tolerance)
