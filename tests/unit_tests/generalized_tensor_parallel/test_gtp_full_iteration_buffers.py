# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    reset_gtp_globals,
)
from tests.unit_tests.test_utilities import Utils


def test_capture_does_not_borrow_eager_wgrad_storage(monkeypatch):
    """A graph must not keep a scratch address that eager work can later overwrite."""
    Utils.initialize_distributed()
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {})
    eager = gtp._wgrad_pool_get((1024,), torch.float32, "cuda")
    gtp._wgrad_pool_put(eager)
    result = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch = gtp._wgrad_pool_get(
            eager.shape, eager.dtype, eager.device, graph_capture_safe=True
        )
        scratch.fill_(3)
        result.copy_(scratch)
    assert scratch.data_ptr() != eager.data_ptr()
    assert gtp._wgrad_pool_get(eager.shape, eager.dtype, eager.device) is eager
    for _ in range(3):
        eager.fill_(9)
        graph.replay()
        torch.testing.assert_close(result, torch.full_like(result, 3))


def test_plain_wgrad_reuse_waits_for_async_reader(monkeypatch):
    """Returning a buffer to the manual pool must not race its side-stream reader."""
    Utils.initialize_distributed()
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {})
    reader = torch.cuda.Stream()
    output = torch.empty(1024, device="cuda")
    for value in range(3):
        buffer = gtp._wgrad_pool_get(output.shape, output.dtype, output.device)
        buffer.fill_(value)
        reader.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(reader):
            torch.cuda._sleep(1000000)
            output.copy_(buffer)
            ready = torch.cuda.Event()
            ready.record()
        gtp._wgrad_pool_put(buffer, ready_event=ready)
        reused = gtp._wgrad_pool_get(output.shape, output.dtype, output.device)
        assert reused.data_ptr() == buffer.data_ptr()
        reused.fill_(-1)
        torch.testing.assert_close(output, torch.full_like(output, value))
        gtp._wgrad_pool_put(reused)


@pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TransformerEngine >= 2.19")
def test_full_iteration_fp32_accumulation_scratch_lifetime(monkeypatch):
    """Deferred FP32 sums must survive scratch release and subsequent allocation pressure."""
    Utils.initialize_distributed()
    group = torch.distributed.group.WORLD
    world_size = group.size()
    if world_size <= 2:
        pytest.skip("FP32-accumulation reduce-scatter requires more than two ranks")
    assert not gtp.is_gtp_symm_pool_registered(group)
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {})
    monkeypatch.setattr(gtp, "_FULL_ITERATION", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
    monkeypatch.setattr("megatron.core.full_cuda_graph.get_all_rng_states", lambda: {})
    layer = _make_gtp_linear(128, 32 * world_size, group)
    gtp.classify_gtp_chains(layer)
    weight = layer.weight
    weight.main_grad = torch.zeros_like(weight)
    source = torch.empty(weight._unsharded_shape, dtype=torch.bfloat16, device="cuda")
    source.fill_(group.rank())
    calls = []

    def forward_backward_func(**_):
        weight.main_grad.zero_()
        buffer = weight.get_wgrad_tensor()
        buffer.copy_(source)
        weight.wgrad_reduce_scatter(buffer, async_op=True)
        del buffer
        # This list is populated only by the non-registered all-to-all/FP32-sum path.
        assert len(weight._rs_a2a_bufs) == 1
        calls.append(torch.cuda.is_current_stream_capturing())
        rs_stream = gtp.get_rs_stream(weight.chain_id, group)
        with torch.cuda.stream(rs_stream):
            torch.cuda._sleep(5_000_000)
        weight._wait_reduce_scatter(finalize_grad=True)
        assert weight._rs_a2a_bufs is None and weight._wgrad_input_bufs is None
        # Do not join the RS stream until after these overwrites. Incorrectly freed
        # scratch can be recycled while its deferred FP32 sum is still reading it.
        pressure = [torch.empty_like(source) for _ in range(8)]
        for allocation in pressure:
            allocation.fill_(-100)
        torch.cuda.current_stream().wait_stream(rs_stream)
        return [{"loss": weight.main_grad}]

    wrapper = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)
    wrapper.reset_cuda_graph()
    kwargs = dict(
        model=[torch.nn.Identity()],
        data_iterator=None,
        num_microbatches=1,
        seq_length=1,
        forward_only=False,
    )
    try:
        wrapper(**kwargs)
        wrapper(**kwargs)
        for step in range(10):
            source.fill_(group.rank() + step)
            result = wrapper(**kwargs)
            torch.cuda.synchronize()
            expected = step + (world_size - 1) / 2
            torch.testing.assert_close(
                result[0]["loss"], torch.full_like(weight.main_grad, expected), atol=0, rtol=0
            )
        assert calls == [False, True], "Replay must not repeat Python scratch allocation"
    finally:
        torch.cuda.synchronize()
        wrapper.reset_cuda_graph()
