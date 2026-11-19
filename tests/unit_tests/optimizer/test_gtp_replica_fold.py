# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""LayerWise optimizer-state replica folding with dense GTP and expert GTP 1.

CPU only: the fold is computed from parameter attributes and process-group ranks.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.layer_wise_optimizer import _build_gtp_replica_fold, _fold_replica_id
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

pytestmark = pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TransformerEngine >= 2.19")


class _Group:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _model():
    model = torch.nn.Module()
    # Replicated across the dense gtp_remat group (like the MoE router).
    model.router = torch.nn.Linear(4, 2, bias=False)
    # A dense GTP shard: each peer holds distinct rows, so it is never folded.
    model.shared = torch.nn.Linear(4, 2, bias=False)
    model.shared.weight.is_gtp_weight_remat = True
    # A routed expert: expert-parallel (allreduce=False), one owner per expert.
    model.experts = torch.nn.Module()
    model.experts.weight0 = torch.nn.Parameter(torch.zeros(2, 4))
    model.experts.weight0.allreduce = False
    return model


@pytest.mark.parametrize("egtp_group", [None, _Group(rank=0, size=1)])
def test_dense_gtp_folds_only_replicated_dense_params(egtp_group):
    """With dense GTP > 1 and expert GTP 1, experts must keep their own main replica."""
    pg_collection = SimpleNamespace(gtp_remat=_Group(rank=2, size=4), expt_gtp_remat=egtp_group)

    fold = _build_gtp_replica_fold(pg_collection, [_model()])

    assert fold == {"router.weight": (2, 4)}
    expert_key = "optimizer.state.exp_avg.experts.weight0"
    assert _fold_replica_id((0, 0, 3), expert_key, fold) == (0, 0, 0)
    router_key = "optimizer.state.exp_avg.router.weight"
    assert _fold_replica_id((0, 0, 3), router_key, fold) == (0, 2, 0)


def test_expert_gtp_folds_replicated_expert_params():
    """Expert params replicated across an expert GTP group fold with that group's rank."""
    model = _model()
    pg_collection = SimpleNamespace(
        gtp_remat=_Group(rank=2, size=4), expt_gtp_remat=_Group(rank=1, size=2)
    )

    fold = _build_gtp_replica_fold(pg_collection, [model])

    assert fold == {"router.weight": (2, 4), "experts.weight0": (1, 2)}
