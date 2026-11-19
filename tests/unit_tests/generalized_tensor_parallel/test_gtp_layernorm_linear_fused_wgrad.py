# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP-sharded TELayerNormColumnParallelLinear with fused weight-gradient accumulation.

Kimi-K3 trains with gradient_accumulation_fusion=True (MoK requires it), so each GTP-sharded
TELayerNormColumnParallelLinear (MLA linear_q_up_proj / linear_kv_up_proj with the fused q/kv
norm, the stage-0 dense MLP linear_fc1) takes TE's fused-wgrad branch. Its backward must not
hand autograd an unsharded placeholder gradient, and the reduce-scattered weight gradient must
land in main_grad. Run with four ranks:

    torchrun --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/generalized_tensor_parallel/test_gtp_layernorm_linear_fused_wgrad.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import (
    is_gtp_param,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_gtp_globals,
)

HIDDEN = 64
TOKENS = 16
MICROBATCHES = 2


def _worker(rank, world_size, port, output_size, params_dtype):
    del port
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(gtp_remat_size=world_size)
    try:
        _set_random_seed(1234)
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            params_dtype=params_dtype,
            bf16=params_dtype == torch.bfloat16,
            normalization="RMSNorm",
            add_bias_linear=False,
            gradient_accumulation_fusion=True,
            gtp_weight_remat_size=world_size,
        )
        layer = TELayerNormColumnParallelLinear(
            HIDDEN,
            output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            pg_collection=pgc,
        )
        assert is_gtp_param(layer.weight)
        assert not is_gtp_param(layer.layer_norm_weight)
        assert layer.layer_norm_weight.shape == (HIDDEN,)

        # Stand in for DDP: an FP32 main_grad, plus the fused-accumulation flag reset below.
        layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=torch.float32, device="cuda")

        shards = [torch.empty_like(layer.weight) for _ in range(world_size)]
        dist.all_gather(shards, layer.weight.detach(), group=pgc.gtp_remat)
        weight = torch.cat(shards)[:output_size].float().requires_grad_()
        norm = layer.layer_norm_weight.detach().float().clone().requires_grad_()

        placeholder_shapes = []
        layer.weight.register_hook(lambda grad: placeholder_shapes.append(tuple(grad.shape)))
        tol = 2e-5 if params_dtype == torch.float32 else 3e-2
        wgrad_tol = 3e-5 if params_dtype == torch.float32 else 3e-2
        torch.manual_seed(100 + rank)
        for _ in range(MICROBATCHES):
            layer.weight.grad_added_to_main_grad = False
            x = torch.randn(TOKENS, 1, HIDDEN, device="cuda", dtype=params_dtype)
            x.requires_grad_()
            ref_x = x.detach().float().requires_grad_()
            actual, _ = layer(x)
            expected = F.linear(
                F.rms_norm(ref_x, (HIDDEN,), norm, config.layernorm_epsilon), weight
            )
            # Representable in the compute dtype, so both sides see the same upstream gradient.
            grad = torch.randn_like(expected).to(params_dtype)
            actual.backward(grad)
            expected.backward(grad.float())
            wait_for_gtp_grad_reduction_on_current_stream()

            torch.testing.assert_close(actual.float(), expected, atol=tol, rtol=tol)
            torch.testing.assert_close(x.grad.float(), ref_x.grad, atol=tol, rtol=tol)
            # DDP drops the placeholder after its hook; keep .grad from accumulating it.
            layer.weight.grad = None

        # TE 0a2ebe94 returns no placeholder for a GTP weight; later TE returns one, which must
        # be shaped like the shard (te_gtp_compat for TE before be880cc2).
        assert all(shape == tuple(layer.weight.shape) for shape in placeholder_shapes)
        torch.testing.assert_close(
            layer.layer_norm_weight.grad.float(), norm.grad, atol=tol, rtol=tol
        )

        # main_grad holds this rank's GTP shard of the gtp_remat mean of the accumulated wgrad.
        dist.all_reduce(weight.grad, group=pgc.gtp_remat)
        weight.grad.div_(world_size)
        padded = F.pad(weight.grad, (0, 0, 0, layer.weight.pad_length))
        expected_shard = padded.chunk(world_size)[pgc.gtp_remat.rank()]
        torch.testing.assert_close(
            layer.weight.main_grad, expected_shard, atol=wgrad_tol, rtol=wgrad_tol
        )
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()


@pytest.mark.parametrize("params_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_size", [96, 128])
def test_gtp_layernorm_linear_fused_wgrad(output_size, params_dtype):
    _requires_multi_gpu(4)
    _run_distributed(_worker, 4, output_size, params_dtype)
