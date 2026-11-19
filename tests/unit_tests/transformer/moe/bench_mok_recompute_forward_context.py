# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Time the forward-context rebuild of ``moe_mok_recompute_forward_context`` at the K3 shape.

Run on one GB200/GB300 node inside the Kimi-K3 container, from this Megatron-LM checkout (needs
only the installed ``mok`` package and this checkout on ``PYTHONPATH``)::

    PYTHONPATH=$PWD torchrun --nproc-per-node 4 \
        tests/unit_tests/transformer/moe/bench_mok_recompute_forward_context.py

By default every rank gets the Kimi-K3 per-rank MoE shape at EP64: 8192 tokens, 14 local
experts (56 on 4 GPUs, 896 on 64), top-16 routing balanced over the experts as with forced load
balancing, latent hidden 3584, expert FFN 3072, MXFP8 split per-expert weights in MCore's
combined FC1 layout and SiTU-GLU (4, 25), i.e. what ``_MoKAutograd`` passes MoK for K3. The
default MoK config is the tuned one the Kimi-K3 8K option-E recipes launch with (the
``..._mok_tuned_opte_f73`` recipes set it or inherit it from ``..._mok_tuned``, as the
``..._mok_tuned_opte_a4`` ones inherit it from ``..._mok_tuned_a4``): comm SMs 32/40, minibatch
4096, schedule capacity multiplier 0.0625 and macrobatch 262144, measured first;
``MACROBATCH_SIZES`` adds 139264 (one macrobatch holds every padded routed row at K3, 133,120
when balanced) and MoK's default 131072. ``MOK_CONFIG='{}'`` measures MoK's default config
instead. For each ``MACROBATCH_SIZES`` entry it reports, as the max over ranks of the median over
``ITERS`` timed calls:

* ``forward``: ``build_schedule`` + ``functional.forward``, what MCore's MoK forward runs;
* ``pack``: ``weights.pack_fc1_gate_up`` of the local experts' FC1 rows and scale blocks;
* ``recompute``: ``functional.recompute_forward_context`` from the packed weights;
* ``rebuild``: pack + recompute, the time the option adds to every MoE backward;
* ``backward``: ``functional.backward`` into FP32 main grads, as MCore runs it;

plus the context and packed-FC1 sizes, the rebuild's peak memory above its start, and a bitwise
check of the rebuilt context against the forward's. The summary scales the rebuild by
``LAYERS x MICROBATCHES`` (default 23 x 32, stage 0 of the 8K TP1 PP4 recipe) and sets it
against MoE recompute at ``MOE_RECOMPUTE_MS`` per layer (8.19 ms, measured on the proxy).

Environment: NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_LOCAL_EXPERTS, TOPK,
MACROBATCH_SIZES (comma-separated, default 262144,139264,131072), MOK_CONFIG (JSON of the other
MoKConfig fields, default the recipes' tuned config above), ROUTING (balanced | random),
PRECISION (mxfp8 | bf16), ACTIVATION (situ | silu), WARMUP, ITERS, LAYERS, MICROBATCHES,
MOE_RECOMPUTE_MS.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.megakernel.mok import fc1_layout
from megatron.core.transformer.moe.megakernel.mok.weights import pack_fc1_gate_up

NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 8192))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 3584))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 14))
TOPK = int(os.environ.get("TOPK", 16))
MACROBATCH_SIZES = [
    int(size) for size in os.environ.get("MACROBATCH_SIZES", "262144,139264,131072").split(",")
]
# The tuned MoK config of the Kimi-K3 8K recipes (macrobatch_size comes from MACROBATCH_SIZES).
RECIPE_MOK_CONFIG = {
    "fwd_num_comm_sms": 32,
    "bwd_num_comm_sms": 40,
    "minibatch_size": 4096,
    "schedule_capacity_multiplier": 0.0625,
}
MOK_CONFIG = json.loads(os.environ.get("MOK_CONFIG", json.dumps(RECIPE_MOK_CONFIG)))
ROUTING = os.environ.get("ROUTING", "balanced")
PRECISION = os.environ.get("PRECISION", "mxfp8")
ACTIVATION = os.environ.get("ACTIVATION", "situ")
WARMUP = int(os.environ.get("WARMUP", 5))
ITERS = int(os.environ.get("ITERS", 20))
LAYERS = int(os.environ.get("LAYERS", 23))
MICROBATCHES = int(os.environ.get("MICROBATCHES", 32))
MOE_RECOMPUTE_MS = float(os.environ.get("MOE_RECOMPUTE_MS", 8.19))
SITU_BETA = (4.0, 25.0)
GIB = 1024**3


def _log(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def _max_over_ranks(value: float, device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _median_ms(setup, timed, device) -> float:
    """Median of ITERS timed calls of ``timed(setup())`` after WARMUP untimed ones.

    Every rank enters each call together (MoK calls are EP collectives with device barriers),
    and ``setup`` runs outside the timed region.
    """
    times = []
    for iteration in range(WARMUP + ITERS):
        state = setup()
        torch.cuda.synchronize(device)
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = timed(state)
        end.record()
        torch.cuda.synchronize(device)
        del result, state
        if iteration >= WARMUP:
            times.append(start.elapsed_time(end))
    return _max_over_ranks(statistics.median(times), device)


def _routes(rank: int, world_size: int, device):
    num_experts = NUM_LOCAL_EXPERTS * world_size
    if ROUTING == "balanced":
        # Round-robin over global token slots: TOPK consecutive (distinct) experts per token and
        # an even load per expert, as with forced load balancing.
        tokens = torch.arange(NUM_LOCAL_TOKENS, device=device) + rank * NUM_LOCAL_TOKENS
        slots = tokens[:, None] * TOPK + torch.arange(TOPK, device=device)[None, :]
        top_experts = (slots % num_experts).to(torch.int32)
    elif ROUTING == "random":
        generator = torch.Generator(device=device).manual_seed(7 + rank)
        scores = torch.rand(NUM_LOCAL_TOKENS, num_experts, generator=generator, device=device)
        top_experts = scores.topk(TOPK, dim=-1).indices.to(torch.int32)
    else:
        raise ValueError(f"ROUTING must be balanced or random, got {ROUTING!r}")
    if TOPK > num_experts:
        raise ValueError("TOPK must not exceed the number of experts")
    generator = torch.Generator(device=device).manual_seed(11 + rank)
    router_weights = torch.rand(NUM_LOCAL_TOKENS, TOPK, generator=generator, device=device)
    return top_experts.contiguous(), router_weights


def _split_weights(functional, ops, experts):
    """MCore-style split routed weights: one allocation per expert plus MoK descriptor tables."""
    if PRECISION == "bf16":
        table = ops.make_routed_weight_storage_table_bf16(experts)
        return functional.SplitRoutedWeight(experts[0], table), experts, None
    quantized = [ops.mxfp8_quantize(expert, True, True) for expert in experts]
    rows = [q[0] for q in quantized]
    row_scales = [q[1] for q in quantized]
    # TE keeps a native columnwise payload in the logical [M, K] shape.
    columns = [q[2].transpose(-2, -1).contiguous() for q in quantized]
    column_scales = [q[3] for q in quantized]
    weight = functional.SplitRoutedWeight(
        data=rows[0],
        storage_table=ops.make_routed_weight_storage_table_mxfp8(rows),
        scale=row_scales[0],
        scale_storage_table=ops.make_routed_scale_storage_table(row_scales),
        scale_tensors=tuple(row_scales),
        transposed_data=columns[0],
        transposed_scale=column_scales[0],
        transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(columns),
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(column_scales),
        transposed_scale_tensors=tuple(column_scales),
        native_columnwise=True,
    )
    return weight, rows, row_scales


def _context_tensors(context):
    for field in dataclasses.fields(context):
        value = getattr(context, field.name)
        for tensor in value if isinstance(value, tuple) else (value,):
            if isinstance(tensor, torch.Tensor):
                yield field.name, tensor


def _context_bytes(context) -> int:
    return sum(tensor.numel() * tensor.element_size() for _, tensor in _context_tensors(context))


def _valid_rows_equal(forward_context, rebuilt_context, schedule, macrobatch_size) -> bool:
    rows = fc1_layout.valid_context_rows(int(schedule.num_tokens.item()), macrobatch_size)
    blocks = rows // fc1_layout.SCALE_BLOCK_ROWS
    equal = True
    for name, transposed in (
        ("x_routed", True),
        ("gate_routed", False),
        ("up_routed", False),
        ("hidden_routed", True),
    ):
        expected, actual = getattr(forward_context, name), getattr(rebuilt_context, name)
        mxfp8 = isinstance(expected, tuple)
        pairs = zip(expected, actual) if mxfp8 else [(expected, actual)]
        for index, (reference, result) in enumerate(pairs):
            limit = blocks if index == 1 else rows
            # MXFP8 keeps x and the activation output transposed; BF16 keeps all row-major.
            if mxfp8 and transposed:
                reference, result = reference[:, :limit], result[:, :limit]
            else:
                reference, result = reference[:limit], result[:limit]
            width = {1: torch.uint8, 2: torch.int16}[reference.element_size()]
            equal &= torch.equal(reference.view(width), result.view(width))
    return equal


def main() -> None:
    from mok import functional, ops

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if PRECISION not in ("mxfp8", "bf16") or ACTIVATION not in ("situ", "silu"):
        raise ValueError("PRECISION must be mxfp8 or bf16 and ACTIVATION situ or silu")
    activation_kwargs = {"situ_beta": SITU_BETA} if ACTIVATION == "situ" else {}

    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn(
        NUM_LOCAL_TOKENS, HIDDEN_DIM, generator=generator, device=device, dtype=torch.bfloat16
    )
    d_output = torch.randn(x.shape, generator=generator, device=device, dtype=torch.bfloat16) / 16
    top_experts, router_weights = _routes(rank, world_size, device)
    weight_options = dict(generator=generator, device=device, dtype=torch.bfloat16)
    fc1 = [
        torch.randn(2 * INTERMEDIATE_DIM, HIDDEN_DIM, **weight_options) / 64
        for _ in range(NUM_LOCAL_EXPERTS)
    ]
    fc2 = [
        torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, **weight_options) / 64
        for _ in range(NUM_LOCAL_EXPERTS)
    ]
    split_fc1, fc1_payloads, fc1_scales = _split_weights(functional, ops, fc1)
    split_fc2, _, _ = _split_weights(functional, ops, fc2)
    del fc1, fc2
    fc1_main_grads = [
        torch.zeros(2 * INTERMEDIATE_DIM, HIDDEN_DIM, dtype=torch.float32, device=device)
        for _ in range(NUM_LOCAL_EXPERTS)
    ]
    fc2_main_grads = [
        torch.zeros(HIDDEN_DIM, INTERMEDIATE_DIM, dtype=torch.float32, device=device)
        for _ in range(NUM_LOCAL_EXPERTS)
    ]
    fc1_grad_table = ops.make_routed_d_weight_storage_table(fc1_main_grads)
    fc2_grad_table = ops.make_routed_d_weight_storage_table(fc2_main_grads)
    # MoK ABI order (shared gate, routed gate, shared up, routed up, shared down, routed down);
    # gate and up share the combined-FC1 buffers and table, as runtime.py passes them.
    main_grads = (None, fc1_main_grads[0], None, fc1_main_grads[0], None, fc2_main_grads[0])
    main_grad_tables = (fc1_grad_table, fc1_grad_table, fc2_grad_table)

    _log(
        rank,
        f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_LOCAL_EXPERTS * world_size} "
        f"({NUM_LOCAL_EXPERTS}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM} "
        f"world={world_size} precision={PRECISION} activation={ACTIVATION} routing={ROUTING} "
        f"config={functional.MoKConfig(**MOK_CONFIG)}",
    )
    packed_bytes = fc1_layout.packed_gate_up_nbytes(
        NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, mxfp8=PRECISION == "mxfp8"
    )
    summary = []
    for macrobatch_size in MACROBATCH_SIZES:
        config = functional.MoKConfig(**{**MOK_CONFIG, "macrobatch_size": macrobatch_size})
        functional.clear_workspace_cache()
        workspace = functional.get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=NUM_LOCAL_TOKENS,
            hidden_size=HIDDEN_DIM,
            topk=TOPK,
        )

        def run_forward(_state=None):
            schedule = functional.build_schedule(
                workspace, config, top_experts, num_local_experts=NUM_LOCAL_EXPERTS
            )
            output, context = functional.forward(
                config,
                workspace,
                schedule,
                x,
                router_weights,
                None,
                None,
                None,
                split_fc1,
                split_fc1,
                split_fc2,
                **activation_kwargs,
            )
            return schedule, context, output

        def pack(_state=None):
            return pack_fc1_gate_up(fc1_payloads, fc1_scales, INTERMEDIATE_DIM)

        def recompute(schedule, packed):
            gate, up = packed
            return functional.recompute_forward_context(
                config, workspace, schedule, x, None, None, gate, up, **activation_kwargs
            )

        def rebuild(schedule):
            return recompute(schedule, pack())

        def backward(schedule, context):
            return functional.backward(
                config,
                workspace,
                schedule,
                context,
                d_output,
                x,
                router_weights,
                None,
                None,
                None,
                split_fc1,
                split_fc1,
                split_fc2,
                main_grads=main_grads,
                main_grad_storage_tables=main_grad_tables,
            )

        def schedule_only():
            return functional.build_schedule(
                workspace, config, top_experts, num_local_experts=NUM_LOCAL_EXPERTS
            )

        def forward_state():
            schedule, context, _ = run_forward()
            return schedule, context

        forward_ms = _median_ms(lambda: None, run_forward, device)
        pack_ms = _median_ms(lambda: None, pack, device)
        recompute_ms = _median_ms(
            lambda: (schedule_only(), pack()), lambda state: recompute(*state), device
        )
        rebuild_ms = _median_ms(schedule_only, rebuild, device)
        backward_ms = _median_ms(forward_state, lambda s: backward(*s), device)

        # Correctness and memory of one rebuild.
        schedule, forward_context, _ = run_forward()
        context_bytes = _context_bytes(forward_context)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        rebuilt = rebuild(schedule)
        torch.cuda.synchronize(device)
        rebuild_peak = torch.cuda.max_memory_allocated(device) - start
        equal = _valid_rows_equal(forward_context, rebuilt, schedule, macrobatch_size)
        all_equal = _max_over_ranks(0.0 if equal else 1.0, device) == 0.0
        routed_rows = int(schedule.num_tokens.item())
        max_rows = _max_over_ranks(routed_rows, device)
        max_context = _max_over_ranks(context_bytes, device)
        max_peak = _max_over_ranks(rebuild_peak, device)
        del schedule, forward_context, rebuilt

        _log(
            rank,
            f"macrobatch={macrobatch_size}: forward {forward_ms:.3f} ms, pack {pack_ms:.3f} ms, "
            f"recompute {recompute_ms:.3f} ms, rebuild (pack + recompute) {rebuild_ms:.3f} ms, "
            f"backward {backward_ms:.3f} ms; padded routed rows/rank max {max_rows:.0f} "
            f"({max_rows / macrobatch_size:.3f} macrobatches); context {max_context / GIB:.3f} "
            f"GiB, packed FC1 {packed_bytes / GIB:.3f} GiB, rebuild peak above start "
            f"{max_peak / GIB:.3f} GiB; rebuilt context bitwise equal on all ranks: {all_equal}",
        )
        summary.append(
            (
                macrobatch_size,
                forward_ms,
                pack_ms,
                recompute_ms,
                rebuild_ms,
                backward_ms,
                max_context,
            )
        )

    _log(
        rank,
        "summary: macrobatch, forward ms, pack ms, recompute ms, rebuild ms, backward ms, "
        f"context GiB, rebuild s/step at {LAYERS} layers x {MICROBATCHES} microbatches, "
        f"saved s/step vs {MOE_RECOMPUTE_MS} ms MoE recompute",
    )
    for (
        macrobatch_size,
        forward_ms,
        pack_ms,
        recompute_ms,
        rebuild_ms,
        backward_ms,
        context,
    ) in summary:
        per_step = rebuild_ms * LAYERS * MICROBATCHES / 1000
        saved = (MOE_RECOMPUTE_MS - rebuild_ms) * LAYERS * MICROBATCHES / 1000
        _log(
            rank,
            f"summary: {macrobatch_size}, {forward_ms:.3f}, {pack_ms:.3f}, {recompute_ms:.3f}, "
            f"{rebuild_ms:.3f}, {backward_ms:.3f}, {context / GIB:.3f}, {per_step:.2f}, "
            f"{saved:.2f}",
        )
    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
