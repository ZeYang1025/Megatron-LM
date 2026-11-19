# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP_remat shards of weights duplicated across TP must initialize identically on TP peers.

The MoE latent projections (``fc1_latent_proj`` as ``TELinear(parallel_mode="duplicated")`` and,
with ``moe_latent_up_projection_rmsnorm``, ``fc2_latent_proj`` as ``TERMSNormDuplicatedLinear``)
are replicated across TP and GTP_remat-sharded by the ``moe_latent_proj`` opt-in. TP peers with the
same GTP_remat rank hold the same shard: nothing re-syncs them after init (their gradients are
summed over TP), and a checkpoint keeps only TP rank 0's copy. Run with four ranks (TP2 x GTP2):

    torchrun --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/generalized_tensor_parallel/test_gtp_duplicated_init.py
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear, TERMSNormDuplicatedLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import is_gtp_param
from megatron.core.tensor_parallel.random import get_gtp_remat_rng_tracker_name
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

TP = 2
GTP = 2


def _all_gather(tensor, group):
    # clone() returns a plain contiguous tensor, also for a GTP shard (a Parameter subclass).
    local = tensor.detach().clone()
    parts = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts, local, group=group)
    return parts


@pytest.mark.parametrize("params_dtype", [torch.float32, torch.bfloat16])
def test_duplicated_latent_projection_shards_match_across_tp(params_dtype):
    _requires_multi_gpu(TP * GTP)
    if dist.get_world_size() != TP * GTP:
        pytest.skip(f"Run with {TP * GTP} torchrun ranks")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP, gtp_remat_size=GTP)
    try:
        _set_random_seed(1234)
        pgc = ProcessGroupCollection.use_mpu_process_groups()
        assert pgc.tp.size() == TP and pgc.gtp_remat.size() == GTP
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            tensor_model_parallel_size=TP,
            sequence_parallel=True,
            tensor_parallel_num_weight_shards=TP * GTP,
            params_dtype=params_dtype,
            add_bias_linear=False,
        )
        common = dict(
            parallel_mode="duplicated",
            config=config,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            gtp_remat_group=pgc.gtp_remat,
            gtp_replica_group=pgc.dp_cp,
        )
        # The shapes and init methods MoELayer uses for its latent projections.
        fc1_latent_proj = TELinear(128, 64, init_method=config.init_method, **common)
        fc2_latent_proj = TERMSNormDuplicatedLinear(
            64, 128, init_method=config.output_layer_init_method, tp_group=pgc.tp, **common
        )

        duplicated_stream = get_gtp_remat_rng_tracker_name(duplicated=True)
        layers = (("fc1_latent_proj", fc1_latent_proj), ("fc2_latent_proj", fc2_latent_proj))
        for name, layer in layers:
            assert is_gtp_param(layer.weight), name
            assert layer.rng_tracker_name == duplicated_stream, name
            tp_copies = _all_gather(layer.weight, pgc.tp)
            for tp_rank, copy in enumerate(tp_copies[1:], start=1):
                assert torch.equal(copy, tp_copies[0]), (
                    f"{name}: TP rank {tp_rank} initialized its GTP_remat shard differently from "
                    "TP rank 0, although the weight is duplicated across TP"
                )
            gtp_shards = _all_gather(layer.weight, pgc.gtp_remat)
            assert not torch.equal(
                gtp_shards[0], gtp_shards[1]
            ), f"{name}: GTP_remat ranks drew identical shards"
        # The RMSNorm scale is not GTP-sharded and stays replicated everywhere.
        assert not is_gtp_param(fc2_latent_proj.layer_norm_weight)
        norm_copies = _all_gather(fc2_latent_proj.layer_norm_weight, pgc.tp)
        assert torch.equal(norm_copies[0], norm_copies[1])
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()
