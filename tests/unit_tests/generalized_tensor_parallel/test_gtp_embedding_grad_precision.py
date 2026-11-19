# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP embedding gradients keep FP32 accumulation under gtp_embedding_wgrad_fp32.

Adapted from PR #8109 (a717f9e76), which promoted every low-precision wgrad to a pooled FP32
copy. Here the FP32 reduction is opt-in and chunked, so the cases are: the opt-in path and the
existing fp32-accumulation reduce-scatter must match an FP32 all-reduce reference exactly, and
the default reduce-scatter in the wgrad dtype must not (which shows the data tells them apart).
Every case also checks that each weight fires grad-ready exactly once per backward, and one case
opts in only one of two chained weights, which is how the embedding sits among untagged linears.

Four ranks on one node:
    torchrun --nproc-per-node 4 -m pytest -x -v \
        tests/unit_tests/generalized_tensor_parallel/test_gtp_embedding_grad_precision.py
"""

import functools
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine")

import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module  # noqa: E402
from megatron.core._rank_utils import get_default_log_ranks  # noqa: E402
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (  # noqa: E402
    GTP_CONFIG,
    GTPEmbeddingWeight,
    reset_gtp_state,
    update_gtp_config,
    wrap_module_params_gtp,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: E402,F401
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)
from tests.unit_tests.test_utilities import Utils  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

WORLD_SIZE = 4
HIDDEN = 8
# One coefficient per rank. Every local gradient is representable in BF16/FP16, but the mean
# across ranks is not, so any rounding to the wgrad dtype after reduction loses real bits.
RANK_COEFFICIENTS = [1.0, 2**-7, 2**-8, 2**-9]
_GTP_CONFIG_KEYS = (
    "pad_for_alignment",
    "async_reduction",
    "calculate_per_token_loss",
    "reduce_scatter_with_fp32_accumulation",
)


@pytest.fixture
def restore_gtp_config():
    old = {key: getattr(GTP_CONFIG, key) for key in _GTP_CONFIG_KEYS}
    yield
    torch.cuda.synchronize()
    dist.barrier()
    reset_gtp_state()
    update_gtp_config(**old)


def _require_four_ranks():
    if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        pytest.skip(f"Requires {WORLD_SIZE} ranks (torchrun --nproc-per-node {WORLD_SIZE})")


def _upstream(num_rows):
    """Per-rank output gradient, distinct in every row and column so a misplaced row or column
    chunk changes the result. Its values (at most 256 * coefficient) are exact in BF16/FP16."""
    assert num_rows * HIDDEN <= 256
    values = torch.arange(1, num_rows * HIDDEN + 1, device="cuda", dtype=torch.float32)
    return values.view(HIDDEN, num_rows).t().contiguous() * RANK_COEFFICIENTS[dist.get_rank()]


def _expected_shard(local_grad, pad_rows, mean):
    """This rank's shard of the FP32 sum (or mean) over ranks of the local [rows, hidden] grads."""
    total = local_grad.float().clone()
    dist.all_reduce(total)
    if mean:
        total /= dist.get_world_size()
    total = F.pad(total, (0, 0, 0, pad_rows))
    return total.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]


def _finish_pending_reduce_scatters(weights):
    """Finalize an async reduce-scatter the backward cascade left in flight, if any."""
    for weight in weights:
        if weight._wgrad_rs_handle is not None:
            weight._wait_reduce_scatter(finalize_grad=True)
    torch.cuda.synchronize()


def _count_grad_ready(counts, index):
    counts[index] += 1


def _count_grad_ready_fires(weights):
    """Register DDP's grad-ready hook as a counter. GTP fires it itself, once a weight's reduced
    gradient is in main_grad, so it shows each weight was finalized exactly once."""
    counts = [0] * len(weights)
    for index, weight in enumerate(weights):
        weight.register_grad_accum_hook(None, functools.partial(_count_grad_ready, counts, index))
    return counts


def _run_two_embeddings(dtype, num_rows, fp32_embedding_wgrad, after_backward=None):
    """Two GTP embedding tables read by one loss, chained in that order: the first is the chain
    head (its reduce-scatter is sync) and the second has a predecessor (its reduce-scatter is
    async when async_reduction is on, and the first's backward cascade finalizes it).

    ``fp32_embedding_wgrad`` opts both weights in (a bool) or each one separately (a pair).
    ``after_backward(weights)`` runs right after ``.backward()``, before any reduce-scatter still
    in flight is finalized.

    Returns:
        (weights, expected main_grad shards, grad-ready fires per weight).
    """
    world_size = dist.get_world_size()
    if isinstance(fp32_embedding_wgrad, bool):
        fp32_embedding_wgrad = (fp32_embedding_wgrad, fp32_embedding_wgrad)
    initial = torch.arange(num_rows * HIDDEN, device="cuda", dtype=torch.float32)
    initial = (initial.view(num_rows, HIDDEN) / 100).to(dtype)
    indices = torch.arange(num_rows, device="cuda")
    upstream = _upstream(num_rows)

    reference = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    reference_out = [F.embedding(indices, p) for p in reference]
    (reference_out[0].float() * upstream - reference_out[1].float() * upstream / 2).sum().backward()

    module = torch.nn.Module()
    module.first = torch.nn.Parameter(initial.clone())
    module.second = torch.nn.Parameter(initial.clone())
    wrap_module_params_gtp(module, ["first", "second"], dist.group.WORLD)
    weights = [module.first, module.second]
    pad_rows = weights[0].pad_length
    assert weights[0].shape[0] * world_size == num_rows + pad_rows
    for p, tagged in zip(weights, fp32_embedding_wgrad):
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        p._wgrad_reduce_in_fp32 = tagged
    grad_ready = _count_grad_ready_fires(weights)
    outputs = [F.embedding(indices, GTPEmbeddingWeight.apply(p)) for p in weights]
    for actual, baseline in zip(outputs, reference_out):
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    (outputs[0].float() * upstream - outputs[1].float() * upstream / 2).sum().backward()
    if after_backward is not None:
        after_backward(weights)
    _finish_pending_reduce_scatters(weights)

    mean = not GTP_CONFIG.calculate_per_token_loss
    expected = [_expected_shard(p.grad, pad_rows, mean) for p in reference]
    return weights, expected, grad_ready


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("async_reduction", [False, True])
@pytest.mark.parametrize("path", ["fp32_embedding_wgrad", "fp32_accum_rs", "default"])
def test_embedding_wgrad_reduction_precision(restore_gtp_config, dtype, async_reduction, path):
    _require_four_ranks()
    update_gtp_config(
        pad_for_alignment=1,
        async_reduction=async_reduction,
        calculate_per_token_loss=False,
        reduce_scatter_with_fp32_accumulation=(path == "fp32_accum_rs"),
    )
    weights, expected, grad_ready = _run_two_embeddings(
        dtype, num_rows=16, fp32_embedding_wgrad=(path == "fp32_embedding_wgrad")
    )
    assert grad_ready == [1, 1]
    for p, wanted in zip(weights, expected):
        assert p.main_grad.dtype == torch.float32
        if path == "default":
            # Reduce-scattered in the wgrad dtype: the sum is rounded to BF16/FP16.
            assert not torch.equal(p.main_grad, wanted)
            torch.testing.assert_close(p.main_grad, wanted, rtol=1e-2, atol=0)
        else:
            torch.testing.assert_close(p.main_grad, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tagged", [0, 1], ids=["chain_head", "mid_chain"])
def test_fp32_embedding_wgrad_next_to_an_untagged_weight(restore_gtp_config, dtype, tagged):
    """Only one of the two chained weights opts in, which is how the embedding sits in K3.

    chain_head (stage 0): the tagged weight heads the chain and its untagged successor's async
    reduce-scatter is still in flight on the same communicator while the FP32 path runs its
    all-to-alls; the cascade after the FP32 path must still finalize that reduce-scatter.

    mid_chain (the MTP stage): the tagged weight has a predecessor, so its reduce-scatter would
    have been async. The FP32 path runs synchronously instead and leaves no handle, so the
    predecessor's cascade finds nothing to drain.

    Either way nothing is left in flight when backward returns, each weight fires grad-ready
    exactly once, the tagged weight is exact and the untagged one keeps the low-precision result.
    """
    _require_four_ranks()
    update_gtp_config(
        pad_for_alignment=1,
        async_reduction=True,
        calculate_per_token_loss=False,
        reduce_scatter_with_fp32_accumulation=False,
    )
    in_flight_after_backward = []

    def record_in_flight(weights):
        in_flight_after_backward.extend(p._wgrad_rs_handle is not None for p in weights)

    weights, expected, grad_ready = _run_two_embeddings(
        dtype,
        num_rows=16,
        fp32_embedding_wgrad=(tagged == 0, tagged == 1),
        after_backward=record_in_flight,
    )
    # Only an async reduce-scatter reserves an output ticket (the sync chain head and the FP32
    # path never do), so this shows chain_head really ran the FP32 path while the successor's
    # reduce-scatter was in flight, and mid_chain really replaced an async one.
    assert (weights[1]._rs_ticket is not None) == (tagged == 0)
    assert weights[0]._rs_ticket is None
    assert in_flight_after_backward == [False, False], "backward left a reduce-scatter in flight"
    assert grad_ready == [1, 1]
    for index, (p, wanted) in enumerate(zip(weights, expected)):
        if index == tagged:
            torch.testing.assert_close(p.main_grad, wanted, rtol=0, atol=0)
        else:
            assert not torch.equal(p.main_grad, wanted)
            torch.testing.assert_close(p.main_grad, wanted, rtol=1e-2, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("async_reduction", [False, True])
@pytest.mark.parametrize("per_token_loss", [False, True])
def test_fp32_embedding_wgrad_chunked_and_padded(
    restore_gtp_config, monkeypatch, dtype, async_reduction, per_token_loss
):
    """18 vocab rows pad to 20 over four ranks (rank 3 owns two pad rows), and a 300-byte cap
    splits the 8 hidden columns into chunks of 3, 3 and 2 (100 bytes per column: 20 padded rows
    x 2 x 2 B of send and receive, plus 5 shard rows x 4 B of FP32 sum). Under per-token loss
    the gtp_remat axis sums instead of averaging."""
    _require_four_ranks()
    update_gtp_config(
        pad_for_alignment=1,
        async_reduction=async_reduction,
        calculate_per_token_loss=per_token_loss,
        reduce_scatter_with_fp32_accumulation=False,
    )
    monkeypatch.setattr(gtp_module, "_FP32_WGRAD_REDUCTION_CHUNK_BYTES", 300)
    assert gtp_module._fp32_wgrad_reduction_column_chunks(HIDDEN, 100, 300) == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]
    weights, expected, grad_ready = _run_two_embeddings(
        dtype, num_rows=18, fp32_embedding_wgrad=True
    )
    assert weights[0].pad_length == 2
    assert grad_ready == [1, 1]
    for p, wanted in zip(weights, expected):
        torch.testing.assert_close(p.main_grad, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("fp32_embedding_wgrad", [False, True])
def test_vocab_parallel_embedding_wires_the_config_flag(restore_gtp_config, fp32_embedding_wgrad):
    """TransformerConfig.gtp_embedding_wgrad_fp32 reaches the GTP embedding weight of a
    VocabParallelEmbedding and, when set, makes its main_grad the exact FP32 mean."""
    _require_four_ranks()
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    update_gtp_config(
        pad_for_alignment=1,
        async_reduction=True,
        calculate_per_token_loss=False,
        reduce_scatter_with_fp32_accumulation=False,
    )
    num_rows = 18
    Utils.initialize_model_parallel(gtp_remat_size=WORLD_SIZE)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=1,
            params_dtype=torch.bfloat16,
            perform_initialization=False,
            tensor_parallel_num_weight_shards=WORLD_SIZE,
            gtp_embedding_wgrad_fp32=fp32_embedding_wgrad,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            embedding = VocabParallelEmbedding(
                num_rows, HIDDEN, init_method=torch.nn.init.zeros_, config=config
            )
        assert not [w for w in caught if "gtp_embedding_wgrad_fp32" in str(w.message)]
        weight = embedding.weight
        assert gtp_module.is_gtp_param(weight)
        assert weight._wgrad_reduce_in_fp32 is fp32_embedding_wgrad
        with torch.no_grad():
            weight.zero_()
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        grad_ready = _count_grad_ready_fires([weight])

        upstream = _upstream(num_rows)
        output = embedding(torch.arange(num_rows, device="cuda").view(1, num_rows))
        (output.float() * upstream.view(1, num_rows, HIDDEN)).sum().backward()
        _finish_pending_reduce_scatters([weight])

        assert grad_ready == [1]
        # Every row is looked up once, so the local wgrad is the upstream gradient itself.
        wanted = _expected_shard(upstream.to(torch.bfloat16), weight.pad_length, mean=True)
        if fp32_embedding_wgrad:
            torch.testing.assert_close(weight.main_grad, wanted, rtol=0, atol=0)
        else:
            assert not torch.equal(weight.main_grad, wanted)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        Utils.initialize_model_parallel()


@pytest.mark.parametrize("case", ["gtp_off", "fp32_params"])
def test_flag_warns_where_it_has_no_effect(restore_gtp_config, case):
    """gtp_embedding_wgrad_fp32 does nothing without a GTP group to reduce over, or for FP32
    weights (their gradient already reduces in FP32). VocabParallelEmbedding, which decides
    whether the embedding is GTP-sharded, says so once per job (on the default log rank)."""
    _require_four_ranks()
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    gtp = case == "fp32_params"
    Utils.initialize_model_parallel(gtp_remat_size=WORLD_SIZE if gtp else 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=1,
            params_dtype=torch.float32 if gtp else torch.bfloat16,
            perform_initialization=False,
            tensor_parallel_num_weight_shards=WORLD_SIZE if gtp else None,
            gtp_embedding_wgrad_fp32=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            embedding = VocabParallelEmbedding(
                16, HIDDEN, init_method=torch.nn.init.zeros_, config=config
            )
        assert bool(gtp_module.is_gtp_param(embedding.weight)) is gtp
        no_effect = [
            w for w in caught if "gtp_embedding_wgrad_fp32 has no effect" in str(w.message)
        ]
        expected = 1 if dist.get_rank() in get_default_log_ranks() else 0
        assert len(no_effect) == expected, [str(w.message) for w in caught]
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        Utils.initialize_model_parallel()
