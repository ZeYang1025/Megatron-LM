# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kimi's fused latent up-projection must retain full RMSNorm and linear semantics under GTP.

Ported from PR #8109 (a717f9e76) and extended to the fused weight-gradient accumulation and the
native MXFP8 GTP shard (fp8_param_gather) that Kimi-K3 trains with. Run with four ranks:

    torchrun --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/generalized_tensor_parallel/test_kimi_latent_rmsnorm_gtp.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.extensions.transformer_engine import TERMSNormDuplicatedLinear
from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import (
    GTP_CONFIG,
    dequantize_gtp_native_fp8,
    is_gtp_param,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _requires_mxfp8,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


@pytest.mark.parametrize("fuse_wgrad", [False, True])
@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("output_size", [80, 128])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_latent_projection_matches_full_matrix(
    output_size, bias, per_token_loss, fuse_wgrad, monkeypatch
):
    """Check padding, replicated norm/bias, forward, dgrad, and the global wgrad."""
    _requires_multi_gpu(4)
    # This standalone layer has no training driver to configure global GTP state.
    monkeypatch.setattr(GTP_CONFIG, "calculate_per_token_loss", per_token_loss)
    world = dist.get_world_size()
    if world != 4:
        pytest.skip("Run with four torchrun ranks")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(gtp_remat_size=world)
    try:
        _set_random_seed(1234)
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            params_dtype=torch.float32,
            gradient_accumulation_fusion=fuse_wgrad,
            add_bias_linear=bias,
            gtp_weight_remat_size=world,
        )
        layer = TERMSNormDuplicatedLinear(
            64,
            output_size,
            parallel_mode="duplicated",
            config=config,
            init_method=config.init_method,
            bias=bias,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            tp_group=pgc.tp,
            gtp_remat_group=pgc.gtp_remat,
            gtp_replica_group=pgc.dp_cp,
        )
        assert is_gtp_param(layer.weight)
        assert not is_gtp_param(layer.layer_norm_weight)
        assert layer.layer_norm_weight.shape == (64,)
        assert layer.out_features == output_size
        assert layer.weight.gtp_replica_group is pgc.dp_cp
        if bias:
            assert not is_gtp_param(layer.bias)
            assert layer.bias.shape == (output_size,)

        shards = [torch.empty_like(layer.weight) for _ in range(world)]
        dist.all_gather(shards, layer.weight.detach(), group=pgc.gtp_remat)
        weight = torch.cat(shards)[:output_size].detach().requires_grad_()
        norm = layer.layer_norm_weight.detach().clone().requires_grad_()
        ref_bias = layer.bias.detach().clone().requires_grad_() if bias else None
        torch.manual_seed(100 + dist.get_rank())
        x = torch.randn(16, 1, 64, device="cuda", requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        layer.weight.main_grad = torch.zeros_like(layer.weight)
        if fuse_wgrad:
            # Stand in for DDP, which owns this flag under fused accumulation.
            layer.weight.grad_added_to_main_grad = False
        placeholder_shapes = []
        layer.weight.register_hook(lambda grad: placeholder_shapes.append(tuple(grad.shape)))

        actual, _ = layer(x)
        expected = F.linear(
            F.rms_norm(ref_x, (64,), norm, config.layernorm_epsilon), weight, ref_bias
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        grad = torch.randn_like(expected)
        actual.backward(grad)
        expected.backward(grad)
        wait_for_gtp_grad_reduction_on_current_stream()
        torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(layer.layer_norm_weight.grad, norm.grad, atol=2e-5, rtol=2e-5)
        if bias:
            torch.testing.assert_close(layer.bias.grad, ref_bias.grad, atol=2e-5, rtol=2e-5)
        # Any placeholder autograd sees must be shaped like the shard (TE before be880cc2 needs
        # megatron/core/extensions/te_gtp_compat.py for that under fused accumulation).
        assert all(shape == tuple(layer.weight.shape) for shape in placeholder_shapes)

        dist.all_reduce(weight.grad, group=pgc.gtp_remat)
        # Default GTP returns the DP mean; token-loss mode returns a sum for the
        # training driver to normalize by the global token count.
        if not per_token_loss:
            weight.grad.div_(world)
        padded = F.pad(weight.grad, (0, 0, 0, layer.weight.pad_length))
        expected_shard = padded.chunk(world)[pgc.gtp_remat.rank()]
        torch.testing.assert_close(layer.weight.main_grad, expected_shard, atol=3e-5, rtol=3e-5)
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()


def _all_gather_rows(tensor, group):
    """Concatenate a dim-0 shard across ``group`` (a plain copy, also of a GTP shard)."""
    local = tensor.detach().clone()
    parts = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts, local, group=group)
    return torch.cat(parts)


def _assert_mxfp8_close(actual, expected, name, tolerance=0.1):
    """Relative L2 error against a high-precision reference.

    MXFP8 quantizes GEMM operands per 32-element block (about 2-4% relative error per element),
    so a correct result lands well under ``tolerance``. A sharding error (wrong rows, misplaced
    padding, a missing or doubled reduction) is O(1).
    """
    actual, expected = actual.detach().float(), expected.detach().float()
    assert actual.shape == expected.shape, f"{name}: {actual.shape} vs {expected.shape}"
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    error = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    assert error < tolerance, f"{name}: relative L2 error {error:.4f} >= {tolerance}"


@pytest.mark.parametrize("per_token_loss", [False, True])
# 128: no padding (as for Kimi-K3's 7168/3584 widths at GTP8); 160: one partly padded and one
# padding-only shard. MXFP8 needs the logical width to be a multiple of 32 for dgrad.
@pytest.mark.parametrize("output_size", [128, 160])
def test_fused_latent_projection_mxfp8_matches_dequantized_reference(
    output_size, per_token_loss, monkeypatch
):
    """Native MXFP8 GTP shard (fp8_model_init), MXFP8 compute and fused wgrad into FP32 main_grad,
    against a reference that uses the dequantized gathered weight in FP32."""
    _requires_multi_gpu(4)
    _requires_mxfp8()
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import fp8_autocast, fp8_model_init

    monkeypatch.setattr(GTP_CONFIG, "calculate_per_token_loss", per_token_loss)
    # The production MXFP8 alignment (configure_gtp_remat_from_recipe): 32-row shard tiles.
    monkeypatch.setattr(GTP_CONFIG, "pad_for_alignment", 32)
    world = dist.get_world_size()
    if world != 4:
        pytest.skip("Run with four torchrun ranks")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(gtp_remat_size=world)
    try:
        _set_random_seed(1234)
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=True,
            add_bias_linear=False,
            gtp_weight_remat_size=world,
        )
        recipe = MXFP8BlockScaling()
        with fp8_model_init(enabled=True, recipe=recipe):
            layer = TERMSNormDuplicatedLinear(
                64,
                output_size,
                parallel_mode="duplicated",
                config=config,
                init_method=config.output_layer_init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                tp_group=pgc.tp,
                gtp_remat_group=pgc.gtp_remat,
                gtp_replica_group=pgc.dp_cp,
            )
        assert is_gtp_param(layer.weight) and is_mxfp8tensor(layer.weight)
        assert not is_gtp_param(layer.layer_norm_weight)
        assert layer.layer_norm_weight.shape == (64,)
        assert layer.out_features == output_size
        padded_rows = layer.weight.shape[0] * world
        assert padded_rows - layer.weight.pad_length == output_size
        assert layer.weight.shape[0] % 32 == 0

        weight = (
            _all_gather_rows(dequantize_gtp_native_fp8(layer.weight), pgc.gtp_remat)[:output_size]
            .float()
            .requires_grad_()
        )
        norm = layer.layer_norm_weight.detach().float().clone().requires_grad_()
        torch.manual_seed(100 + dist.get_rank())
        # 32 tokens: the MXFP8 wgrad GEMM reduces over tokens in 32-element blocks.
        x = torch.randn(32, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref_x = x.detach().float().clone().requires_grad_()
        layer.weight.main_grad = torch.zeros(
            layer.weight.shape, dtype=torch.float32, device="cuda"
        )
        # Stand in for DDP, which owns this flag under fused accumulation.
        layer.weight.grad_added_to_main_grad = False
        placeholder_shapes = []
        layer.weight.register_hook(lambda grad: placeholder_shapes.append(tuple(grad.shape)))

        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            actual, _ = layer(x)
        expected = F.linear(F.rms_norm(ref_x, (64,), norm, config.layernorm_epsilon), weight)
        _assert_mxfp8_close(actual, expected, "output")
        grad = torch.randn_like(expected)
        actual.backward(grad.to(actual.dtype))
        expected.backward(grad)
        wait_for_gtp_grad_reduction_on_current_stream()
        _assert_mxfp8_close(x.grad, ref_x.grad, "input gradient")
        _assert_mxfp8_close(layer.layer_norm_weight.grad, norm.grad, "RMSNorm scale gradient")
        assert all(shape == tuple(layer.weight.shape) for shape in placeholder_shapes)

        dist.all_reduce(weight.grad, group=pgc.gtp_remat)
        if not per_token_loss:
            weight.grad.div_(world)
        main_grad = _all_gather_rows(layer.weight.main_grad, pgc.gtp_remat)
        _assert_mxfp8_close(main_grad[:output_size], weight.grad, "GTP-reduced weight gradient")
        # Alignment rows never receive gradient.
        assert torch.count_nonzero(main_grad[output_size:]) == 0
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()
