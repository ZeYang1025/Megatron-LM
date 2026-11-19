# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoK forward-context drop and rebuild (``moe_mok_recompute_forward_context``).

With the option on, ``_MoKAutograd`` releases MoK's saved forward context after forward and
rebuilds it before backward with ``mok.functional.recompute_forward_context``, from routed FC1
packed out of MCore's combined per-expert weights. These tests check that the rebuilt context
equals the dropped one bit for bit, that gradients are bitwise identical with the option off and
on (also with two layers and three microbatches interleaved in 1F1B order on MoK's shared
workspace), that forward keeps exactly the context's bytes less, and that a call replayed by an
activation checkpoint keeps its context instead of rebuilding it. The CPU tests of the adapter
wiring, with MoK mocked, are in test_mok_megakernel.py.

The config tests need only torch. The GPU tests need a Blackwell GPU (SM100/SM103), a
mixture-of-kittens build with ``functional.recompute_forward_context`` (d59188a or newer) and
Transformer Engine, and skip otherwise. They run at EP = world size. On one GB300 node inside
the Kimi-K3 container, from the Megatron-LM checkout::

    torchrun --nproc-per-node 4 -m pytest -x \
        tests/unit_tests/transformer/moe/test_mok_recompute_forward_context.py

``python -m pytest`` on one GPU runs the same tests at EP1.
"""

import contextlib
import dataclasses
import re
from typing import NamedTuple, Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situlu
from megatron.core.transformer.moe.megakernel.mok import fc1_layout
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

NUM_LOCAL_EXPERTS = 4
TOPK = 4
NUM_TOKENS = 1024
HIDDEN = 512
FFN = 512
MINIBATCH = 256
SITU_BETA = (4.0, 25.0)
# Each rank receives about NUM_TOKENS * TOPK = 4096 routed rows plus per-expert padding to a
# multiple of 256. MoK saves (and the option rebuilds) only the first macrobatch and replays the
# others inside its backward, so 2048 exercises that interplay and 8192 holds every row.
MULTI_MACROBATCH = 2048
SINGLE_MACROBATCH = 8192
_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


# --------------------------------------------------------------------------------------------
# Config and CLI (torch only)
# --------------------------------------------------------------------------------------------


def _mok_config(**overrides):
    values = {
        "num_layers": 2,
        "bf16": True,
        "moe_grouped_gemm": True,
        "hidden_size": 128,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 256,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "add_bias_linear": False,
        "gradient_accumulation_fusion": True,
        "moe_megakernel_backend": "mok",
        "moe_mok_recompute_forward_context": True,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_option_defaults_off():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    assert config.moe_mok_recompute_forward_context is False


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": True, "activation_func": situlu},
        # MoK-fused, ungated shared expert: the rebuild recomputes it too.
        {"moe_shared_expert_intermediate_size": 256},
        # A gated shared expert that MCore runs natively leaves MoK's call ungated.
        {
            "moe_shared_expert_intermediate_size": 256,
            "moe_shared_expert_gate": True,
            "moe_megakernel_backend_config": {"shared_expert_mode": "native"},
        },
    ],
)
def test_config_accepts_supported_mok_layers(overrides):
    assert _mok_config(**overrides).moe_mok_recompute_forward_context


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"moe_megakernel_backend": None}, "requires moe_megakernel_backend='mok'"),
        (
            {"moe_shared_expert_intermediate_size": 256, "moe_shared_expert_gate": True},
            "does not support a MOK-fused gated shared expert",
        ),
    ],
)
def test_config_rejects_unsupported_layers(overrides, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        _mok_config(**overrides)


def test_config_warns_when_moe_recompute_replays_the_call(monkeypatch):
    import megatron.core.transformer.transformer_config as transformer_config

    messages = []
    monkeypatch.setattr(
        transformer_config,
        "warn_single_rank",
        lambda message, *args, **kwargs: messages.append(message),
    )
    _mok_config()
    assert not [m for m in messages if "moe_mok_recompute_forward_context" in m]
    _mok_config(recompute_granularity="selective", recompute_modules=["moe"])
    assert [m for m in messages if "moe_mok_recompute_forward_context" in m]


def test_cli_flag_reaches_the_config_field():
    import argparse

    from megatron.training.arguments import _add_network_size_args

    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_network_size_args(parser)
    args, _ = parser.parse_known_args(["--moe-mok-recompute-forward-context"])
    assert args.moe_mok_recompute_forward_context is True
    args, _ = parser.parse_known_args([])
    assert args.moe_mok_recompute_forward_context is False


# --------------------------------------------------------------------------------------------
# GPU helpers
# --------------------------------------------------------------------------------------------


def _require_mok():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MoK requires a Blackwell SM100/SM103 GPU")
    if Utils.world_size not in (1, 4, 8, 16, 32, 64):
        pytest.skip("MoK's EP size (the world size here) must be one of 1, 4, 8, 16, 32, 64")
    functional = pytest.importorskip("mok.functional")
    ops = pytest.importorskip("mok.ops")
    if not hasattr(functional, "recompute_forward_context"):
        pytest.skip("the installed MoK lacks functional.recompute_forward_context")
    return functional, ops


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a tensor as same-width integers so equality is bitwise."""
    return tensor.detach().view(_INT_VIEW[tensor.element_size()])


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert actual.shape == expected.shape, f"{name}: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: {actual.dtype} != {expected.dtype}"
    mismatches = int((_bits(actual) != _bits(expected)).sum().item())
    assert mismatches == 0, f"{name}: {mismatches} of {actual.numel()} elements differ bitwise"


def _context_valid_views(context, schedule, macrobatch_size: int) -> dict[str, torch.Tensor]:
    """Copy the rows MoK writes of every context tensor (MoK's backward reuses the buffers).

    MXFP8 keeps x and the activation output transposed ([cols, rows]) with scales blocked
    [cols / 128, rows / 128, 32, 16]; gate and up are row-major with [rows / 128, ...] scales.
    BF16 keeps every routed matrix row-major. Rows past MoK's padded routed-token count are never
    written or read and are excluded, as in MoK's own forward/recompute equality test.
    """
    rows = fc1_layout.valid_context_rows(int(schedule.num_tokens.item()), macrobatch_size)
    blocks = rows // fc1_layout.SCALE_BLOCK_ROWS
    views = {}
    for name, transposed in (
        ("x_routed", True),
        ("gate_routed", False),
        ("up_routed", False),
        ("hidden_routed", True),
    ):
        value = getattr(context, name)
        if isinstance(value, tuple):
            data, scale = value
            views[name] = data[:, :rows] if transposed else data[:rows]
            views[f"{name}.scale"] = scale[:, :blocks] if transposed else scale[:blocks]
        else:
            views[name] = value[:rows]
    for name in ("gate_shared", "up_shared", "hidden_shared"):
        value = getattr(context, name)
        if value is not None:
            views[name] = value
    return {name: _bits(view).clone() for name, view in views.items()}


def _context_nbytes(context) -> int:
    total = 0
    for field in dataclasses.fields(context):
        value = getattr(context, field.name)
        for tensor in value if isinstance(value, tuple) else (value,):
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
    return total


def _live_bytes() -> tuple[int, bool]:
    """Bytes currently requested from the caching allocator; True when exact (not rounded)."""
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    if "requested_bytes.all.current" in stats:
        return stats["requested_bytes.all.current"], True
    return torch.cuda.memory_allocated(), False


def _mok_autograd_node(output: torch.Tensor):
    pending, seen = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if type(node).__name__ == "_MoKAutogradBackward":
            return node
        pending.extend(next_node for next_node, _ in node.next_functions)
    return None


def _make_routes(generator, num_experts: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """TOPK distinct experts per token, as top-k indices [tokens, TOPK] and routing weights."""
    scores = torch.rand(NUM_TOKENS, num_experts, generator=generator, device=device)
    top_experts = scores.topk(TOPK, dim=-1).indices
    router_weights = torch.rand(NUM_TOKENS, TOPK, generator=generator, device=device)
    return top_experts, router_weights


# --------------------------------------------------------------------------------------------
# MoK level: packed FC1 rebuild == forward context, with MCore's split combined-FC1 weights
# --------------------------------------------------------------------------------------------


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("macrobatch_size", [MULTI_MACROBATCH, SINGLE_MACROBATCH])
@pytest.mark.parametrize("activation", ["situ", "silu"])
@pytest.mark.parametrize("precision", ["mxfp8", "bf16"])
def test_packed_fc1_rebuild_equals_forward_context(precision, activation, macrobatch_size):
    """MoK's forward reads split per-expert [2I, H] FC1 through descriptor tables, exactly as
    MCore passes it; the rebuild reads the packed contiguous copy. Their contexts must match."""
    functional, ops = _require_mok()
    from megatron.core.transformer.moe.megakernel.mok.weights import pack_fc1_gate_up

    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    ep = torch.distributed.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=MINIBATCH,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=2.0,
    )
    try:
        workspace = functional.get_workspace(
            config,
            torch.distributed.group.WORLD,
            device=device,
            num_local_tokens=NUM_TOKENS,
            hidden_size=HIDDEN,
            topk=TOPK,
        )
        generator = torch.Generator(device=device).manual_seed(2026 + rank)
        x = torch.randn(
            NUM_TOKENS, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16
        )
        top_experts, router_weights = _make_routes(generator, NUM_LOCAL_EXPERTS * ep, device)
        fc1 = [
            torch.randn(2 * FFN, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
            / 16
            for _ in range(NUM_LOCAL_EXPERTS)
        ]
        fc2 = [
            torch.randn(HIDDEN, FFN, generator=generator, device=device, dtype=torch.bfloat16) / 16
            for _ in range(NUM_LOCAL_EXPERTS)
        ]
        if precision == "mxfp8":

            def split(experts):
                quantized = [ops.mxfp8_quantize(expert, True, False) for expert in experts]
                data = [q[0] for q in quantized]
                scales = [q[1] for q in quantized]
                weight = functional.SplitRoutedWeight(
                    data=data[0],
                    storage_table=ops.make_routed_weight_storage_table_mxfp8(data),
                    scale=scales[0],
                    scale_storage_table=ops.make_routed_scale_storage_table(scales),
                    scale_tensors=tuple(scales),
                )
                return weight, data, scales

            split_fc1, payloads, scales = split(fc1)
            split_fc2, _, _ = split(fc2)
        else:
            payloads, scales = fc1, None
            split_fc1 = functional.SplitRoutedWeight(
                fc1[0], ops.make_routed_weight_storage_table_bf16(fc1)
            )
            split_fc2 = functional.SplitRoutedWeight(
                fc2[0], ops.make_routed_weight_storage_table_bf16(fc2)
            )
        activation_kwargs = {"situ_beta": SITU_BETA} if activation == "situ" else {}
        schedule = functional.build_schedule(
            workspace, config, top_experts, num_local_experts=NUM_LOCAL_EXPERTS
        )
        # Gate and up alias one combined payload, as runtime._gate_up_weight_arguments passes it.
        _, forward_context = functional.forward(
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
        expected = _context_valid_views(forward_context, schedule, macrobatch_size)
        assert _context_nbytes(forward_context) == fc1_layout.routed_forward_context_nbytes(
            macrobatch_size, HIDDEN, FFN, mxfp8=precision == "mxfp8"
        )
        del forward_context

        gate, up = pack_fc1_gate_up(payloads, scales, FFN)
        gate_data, up_data = (gate[0], up[0]) if precision == "mxfp8" else (gate, up)
        for expert, payload in enumerate(payloads):
            _assert_bitwise_equal(gate_data[expert], payload[:FFN], f"packed gate {expert}")
            _assert_bitwise_equal(up_data[expert], payload[FFN:], f"packed up {expert}")
        if precision == "mxfp8":
            blocks = FFN // fc1_layout.SCALE_BLOCK_ROWS
            for expert, scale in enumerate(scales):
                rows = slice(expert * blocks, (expert + 1) * blocks)
                _assert_bitwise_equal(gate[1][rows], scale[:blocks], f"packed gate scale {expert}")
                _assert_bitwise_equal(up[1][rows], scale[blocks:], f"packed up scale {expert}")

        rebuilt = functional.recompute_forward_context(
            config, workspace, schedule, x, None, None, gate, up, **activation_kwargs
        )
        actual = _context_valid_views(rebuilt, schedule, macrobatch_size)
        assert actual.keys() == expected.keys()
        for name, tensor in expected.items():
            _assert_bitwise_equal(actual[name], tensor, name)
    finally:
        functional.clear_workspace_cache()


# --------------------------------------------------------------------------------------------
# MCore level: MoKMegakernel with the option off and on
# --------------------------------------------------------------------------------------------


class _Run(NamedTuple):
    results: dict
    kept_bytes: int
    exact_bytes: bool
    rebuilds: int
    forward_context_kept: Optional[bool]
    forward_context_nbytes: int
    forward_views: Optional[dict]
    backward_views: Optional[dict]


def _build_mok_layers(precision, activation, macrobatch_size, shared, single_grouped, count=1):
    """Build ``count`` MoE layers on one EP group, so that their MOK calls share one workspace."""
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules

    ep = Utils.world_size
    Utils.initialize_model_parallel(expert_model_parallel_size=ep)
    model_parallel_cuda_manual_seed(123)
    num_experts = NUM_LOCAL_EXPERTS * ep
    values = dict(
        num_layers=count,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=FFN,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=situlu if activation == "situ" else F.silu,
        situ_glu_beta1=SITU_BETA[0],
        situ_glu_beta2=SITU_BETA[1],
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        expert_model_parallel_size=ep,
        moe_megakernel_backend="mok",
        # Built with the option on, so the backend checks MoK's rebuild support; the tests then
        # switch it per run through the backend's recompute_forward_context property.
        moe_mok_recompute_forward_context=True,
        moe_megakernel_backend_config=dict(
            fwd_num_comm_sms=2,
            bwd_num_comm_sms=2,
            minibatch_size=MINIBATCH,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=2.0,
        ),
    )
    if shared == "fused":
        # Compatible structure (TP1, no latent, routed intermediate size): MoK fuses it.
        values["moe_shared_expert_intermediate_size"] = FFN
    if precision == "mxfp8":
        values.update(fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True)
    if single_grouped:
        values.update(moe_single_grouped_weight=True, moe_use_grouped_tensor=True)
    config = TransformerConfig(**values)
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_experts, moe_grouped_gemm=True
        ).mlp
    )
    layers = []
    for layer_number in range(1, count + 1):
        init_context = (
            get_fp8_context(config, is_init=True)
            if precision == "mxfp8"
            else contextlib.nullcontext()
        )
        with init_context:
            layer = MoELayer(
                config,
                submodules,
                layer_number=layer_number,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
        layer = layer.cuda()
        backend = layer.megakernel_experts
        assert backend.recompute_forward_context
        parameters = dict(backend.named_parameters())
        for parameter in parameters.values():
            # DDP would own these; MoK accumulates weight gradients into them.
            parameter.main_grad = torch.zeros(parameter.shape, dtype=torch.float32, device="cuda")
        # Callers keep the layer: it owns the native expert modules the backend aliases.
        layers.append((layer, backend, parameters))
    return layers


def _build_mok_layer(precision, activation, macrobatch_size, shared, single_grouped):
    return _build_mok_layers(precision, activation, macrobatch_size, shared, single_grouped)[0]


def _layer_inputs(num_experts):
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(4242 + torch.distributed.get_rank())
    x = torch.randn(NUM_TOKENS, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
    top_experts, _ = _make_routes(generator, num_experts, device)
    routing_map = torch.zeros(NUM_TOKENS, num_experts, dtype=torch.bool, device=device)
    routing_map.scatter_(1, top_experts, True)
    probs = torch.rand(NUM_TOKENS, num_experts, generator=generator, device=device)
    upstream = (
        torch.randn(NUM_TOKENS, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
        / 16
    )
    return x, probs, routing_map, upstream


def _run(functional, backend, parameters, inputs, *, rebuild, checkpointed=False):
    """One forward and backward through the MoK backend; returns results and observations."""
    from megatron.core import tensor_parallel

    x, probs, routing_map, upstream = inputs
    backend.recompute_forward_context = rebuild
    for parameter in parameters.values():
        parameter.main_grad.zero_()
        parameter.grad = None
        parameter.grad_added_to_main_grad = False
    x = x.detach().clone().requires_grad_()
    probs = probs.detach().clone().requires_grad_()

    calls = {"rebuild": 0}
    captured = {}
    real_rebuild = functional.recompute_forward_context
    real_backward = functional.backward

    def rebuild_spy(*args, **kwargs):
        calls["rebuild"] += 1
        return real_rebuild(*args, **kwargs)

    def backward_spy(config, workspace, schedule, forward_context, *args, **kwargs):
        captured["views"] = _context_valid_views(forward_context, schedule, config.macrobatch_size)
        return real_backward(config, workspace, schedule, forward_context, *args, **kwargs)

    with (
        patch.object(functional, "recompute_forward_context", rebuild_spy),
        patch.object(functional, "backward", backward_spy),
    ):
        before, exact = _live_bytes()
        if checkpointed:
            output = tensor_parallel.checkpoint(
                lambda hidden, router_probs: backend(hidden, router_probs, routing_map),
                False,
                x,
                probs,
            )
        else:
            output = backend(x, probs, routing_map)
        after, _ = _live_bytes()
        node = None if checkpointed else _mok_autograd_node(output)
        forward_context = None if node is None else node.forward_context
        forward_views = None
        context_nbytes = 0
        if forward_context is not None:
            forward_views = _context_valid_views(
                forward_context, node.schedule, backend.mok_config.macrobatch_size
            )
            context_nbytes = _context_nbytes(forward_context)
        del forward_context
        output.backward(upstream)
    torch.cuda.synchronize()

    results = {"output": output.detach(), "x.grad": x.grad, "probs.grad": probs.grad}
    for name, parameter in parameters.items():
        results[f"{name}.main_grad"] = parameter.main_grad.clone()
        assert parameter.grad_added_to_main_grad, name
        parameter.grad = None
    return _Run(
        results=results,
        kept_bytes=after - before,
        exact_bytes=exact,
        rebuilds=calls["rebuild"],
        forward_context_kept=None if node is None else node.forward_context is not None,
        forward_context_nbytes=context_nbytes,
        forward_views=forward_views,
        backward_views=captured.get("views"),
    )


def _assert_same_results(actual, expected, what: str) -> None:
    """Bitwise comparison of two runs' ``results`` (a _Run or a _PipelineRun)."""
    assert actual.results.keys() == expected.results.keys()
    for name, tensor in expected.results.items():
        _assert_bitwise_equal(actual.results[name], tensor, f"{what}: {name}")


def _require_mok_layer(single_grouped):
    """Skip unless MoK, Transformer Engine and (if asked) single-grouped weights are usable."""
    functional, _ = _require_mok()
    pytest.importorskip("transformer_engine.pytorch")
    if single_grouped and not is_te_min_version("2.14.0"):
        pytest.skip("single-grouped MoE weights need Transformer Engine 2.14+")
    return functional


_LAYER_CASES = [
    # The Kimi-K3 configuration first: MXFP8 split weights, SiTU-GLU, routed-only MoK.
    pytest.param("mxfp8", "situ", MULTI_MACROBATCH, None, False, id="mxfp8-situ-multi-split"),
    pytest.param("mxfp8", "situ", SINGLE_MACROBATCH, None, False, id="mxfp8-situ-single-split"),
    pytest.param("mxfp8", "silu", MULTI_MACROBATCH, None, False, id="mxfp8-silu-multi-split"),
    pytest.param("bf16", "situ", MULTI_MACROBATCH, None, False, id="bf16-situ-multi-split"),
    pytest.param("bf16", "silu", SINGLE_MACROBATCH, None, False, id="bf16-silu-single-split"),
    pytest.param("mxfp8", "silu", MULTI_MACROBATCH, "fused", False, id="mxfp8-silu-multi-fused"),
    pytest.param("bf16", "situ", SINGLE_MACROBATCH, "fused", False, id="bf16-situ-single-fused"),
    pytest.param(
        "mxfp8", "situ", MULTI_MACROBATCH, None, True, id="mxfp8-situ-multi-single_grouped"
    ),
    pytest.param("bf16", "silu", MULTI_MACROBATCH, None, True, id="bf16-silu-multi-single_grouped"),
]


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("precision", "activation", "macrobatch_size", "shared", "single_grouped"), _LAYER_CASES
)
def test_option_is_bitwise_identical_and_drops_the_context(
    precision, activation, macrobatch_size, shared, single_grouped
):
    functional = _require_mok_layer(single_grouped)
    try:
        layer, backend, parameters = _build_mok_layer(
            precision, activation, macrobatch_size, shared, single_grouped
        )
        assert backend.use_mxfp8_weights == (precision == "mxfp8")
        assert backend.native_single_grouped_weights == single_grouped
        assert (backend.shared_fc1_weight is not None) == (shared == "fused")
        inputs = _layer_inputs(layer.config.num_moe_experts)

        # The first call builds MoK's workspace and MCore's cached weight views and descriptor
        # tables; the second establishes that the baseline itself is bitwise reproducible.
        warmup = _run(functional, backend, parameters, inputs, rebuild=False)
        off = _run(functional, backend, parameters, inputs, rebuild=False)
        _assert_same_results(off, warmup, "option off, rerun (MoK determinism)")
        on = _run(functional, backend, parameters, inputs, rebuild=True)
        _assert_same_results(on, off, "option on vs off")

        assert off.rebuilds == 0 and off.forward_context_kept
        assert on.rebuilds == 1 and on.forward_context_kept is False
        # The context MoK's backward consumed with the option on is the rebuilt one; it equals
        # the context forward kept with the option off on every row MoK wrote.
        assert on.backward_views.keys() == off.forward_views.keys()
        for name, tensor in off.forward_views.items():
            _assert_bitwise_equal(on.backward_views[name], tensor, f"rebuilt context {name}")

        # Forward keeps exactly the context's bytes less: 13,200 B per macrobatch row at the
        # Kimi-K3 shape in MXFP8, (H + 3 I) * 33 / 32 in general, plus a fused shared expert's
        # BF16 gate/up/activation.
        expected = fc1_layout.routed_forward_context_nbytes(
            macrobatch_size, HIDDEN, FFN, mxfp8=precision == "mxfp8"
        )
        if shared == "fused":
            expected += 3 * NUM_TOKENS * FFN * 2
        assert off.forward_context_nbytes == expected
        saved = off.kept_bytes - on.kept_bytes
        if off.exact_bytes:
            assert abs(saved - expected) <= 4096, (saved, expected)
        else:
            # memory_allocated rounds blocks; allow 2 MiB of allocator slack per context tensor.
            assert expected <= saved <= expected + 8 * (2 << 20), (saved, expected)
    finally:
        functional.clear_workspace_cache()
        Utils.destroy_model_parallel()


# --------------------------------------------------------------------------------------------
# Training lifecycle: several layers and microbatches in flight on one shared MoK workspace
# --------------------------------------------------------------------------------------------

# One pipeline stage's 1F1B order with two microbatches in flight: F<m> runs microbatch m through
# every layer, B<m> backpropagates it through every layer in reverse.
_PIPELINE_SCHEDULE = ("F0", "F1", "B0", "F2", "B1", "B2")
_PIPELINE_MICROBATCHES = 3
_PIPELINE_IN_FLIGHT = 2


class _PipelineRun(NamedTuple):
    results: dict
    kept_bytes: int
    exact_bytes: bool
    rebuilds: int


def _pipeline_inputs(num_experts: int, num_layers: int):
    """Per microbatch: the first layer's input, per-layer routes and the output gradient."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(777 + torch.distributed.get_rank())
    microbatches = []
    for _ in range(_PIPELINE_MICROBATCHES):
        x = torch.randn(
            NUM_TOKENS, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16
        )
        routes = []
        for _ in range(num_layers):
            top_experts, _ = _make_routes(generator, num_experts, device)
            routing_map = torch.zeros(NUM_TOKENS, num_experts, dtype=torch.bool, device=device)
            routing_map.scatter_(1, top_experts, True)
            probs = torch.rand(NUM_TOKENS, num_experts, generator=generator, device=device)
            routes.append((probs, routing_map))
        upstream = (
            torch.randn(
                NUM_TOKENS, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16
            )
            / 16
        )
        microbatches.append((x, routes, upstream))
    return microbatches


def _run_pipeline(functional, layers, microbatches, *, rebuild) -> _PipelineRun:
    """Run the microbatches through the chained layers in _PIPELINE_SCHEDULE order.

    Every MoK call after the first finds the shared workspace last used by another layer or
    microbatch, and each layer's rebuild runs after other layers' and microbatches' forwards,
    rebuilds and backwards, in the order 1F1B uses them in training.
    """
    parameters = {}
    for index, (_, backend, layer_parameters) in enumerate(layers):
        backend.recompute_forward_context = rebuild
        for name, parameter in layer_parameters.items():
            parameter.main_grad.zero_()
            parameter.grad = None
            parameter.grad_added_to_main_grad = False
            parameters[f"layer{index}.{name}"] = parameter
    # As MCore DDP's backward post-hook does: MoK accumulated the gradient into main_grad, and
    # .grad holds a sentinel aliasing the parameter that must not accumulate across microbatches.
    # The accumulator nodes are kept alive for the hooks to stay registered.
    accumulators, handles = [], []
    for parameter in parameters.values():
        accumulator = parameter.expand_as(parameter).grad_fn.next_functions[0][0]
        handles.append(
            accumulator.register_hook(
                lambda *_, parameter=parameter: setattr(parameter, "grad", None)
            )
        )
        accumulators.append(accumulator)

    calls = {"rebuild": 0}
    real_rebuild = functional.recompute_forward_context

    def rebuild_spy(*args, **kwargs):
        calls["rebuild"] += 1
        return real_rebuild(*args, **kwargs)

    in_flight, results = {}, {}

    def forward(microbatch):
        x, routes, upstream = microbatches[microbatch]
        x = x.detach().clone().requires_grad_()
        probs = [router_probs.detach().clone().requires_grad_() for router_probs, _ in routes]
        hidden = x
        for (_, backend, _), router_probs, (_, routing_map) in zip(layers, probs, routes):
            hidden = backend(hidden, router_probs, routing_map)
        in_flight[microbatch] = (hidden, x, probs, upstream)

    def backward(microbatch):
        output, x, probs, upstream = in_flight.pop(microbatch)
        output.backward(upstream)
        results[f"mb{microbatch}.output"] = output.detach()
        results[f"mb{microbatch}.x.grad"] = x.grad
        for index, router_probs in enumerate(probs):
            results[f"mb{microbatch}.layer{index}.probs.grad"] = router_probs.grad

    kept_bytes = None
    try:
        with patch.object(functional, "recompute_forward_context", rebuild_spy):
            before, exact = _live_bytes()
            for step in _PIPELINE_SCHEDULE:
                (forward if step[0] == "F" else backward)(int(step[1:]))
                if len(in_flight) == _PIPELINE_IN_FLIGHT and kept_bytes is None:
                    kept_bytes = _live_bytes()[0] - before
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    del accumulators
    assert not in_flight
    for name, parameter in parameters.items():
        assert parameter.grad_added_to_main_grad, name
        results[f"{name}.main_grad"] = parameter.main_grad.clone()
    return _PipelineRun(
        results=results, kept_bytes=kept_bytes, exact_bytes=exact, rebuilds=calls["rebuild"]
    )


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("precision", "activation"),
    [pytest.param("mxfp8", "situ", id="mxfp8-situ"), pytest.param("bf16", "silu", id="bf16-silu")],
)
def test_option_is_bitwise_identical_across_interleaved_layers_and_microbatches(
    precision, activation
):
    """Two chained MoE layers share MoK's workspace; three microbatches run in 1F1B order with
    two in flight. Option on and off must give bitwise-identical outputs and gradients, and the
    option must keep exactly the in-flight contexts' bytes less."""
    functional = _require_mok_layer(False)
    try:
        layers = _build_mok_layers(precision, activation, MULTI_MACROBATCH, None, False, count=2)
        microbatches = _pipeline_inputs(layers[0][0].config.num_moe_experts, len(layers))

        # The first run builds MoK's workspace and MCore's cached weight views; the second checks
        # that the baseline itself is bitwise reproducible under this interleaving.
        warmup = _run_pipeline(functional, layers, microbatches, rebuild=False)
        off = _run_pipeline(functional, layers, microbatches, rebuild=False)
        _assert_same_results(off, warmup, "pipeline, option off, rerun (MoK determinism)")
        on = _run_pipeline(functional, layers, microbatches, rebuild=True)
        _assert_same_results(on, off, "pipeline, option on vs off")

        assert off.rebuilds == 0
        assert on.rebuilds == len(layers) * _PIPELINE_MICROBATCHES
        # With two microbatches in flight, the option keeps every layer's two contexts less.
        expected = (
            _PIPELINE_IN_FLIGHT
            * len(layers)
            * fc1_layout.routed_forward_context_nbytes(
                MULTI_MACROBATCH, HIDDEN, FFN, mxfp8=precision == "mxfp8"
            )
        )
        contexts = _PIPELINE_IN_FLIGHT * len(layers)
        saved = off.kept_bytes - on.kept_bytes
        if off.exact_bytes:
            assert abs(saved - expected) <= contexts * 4096, (saved, expected)
        else:
            # memory_allocated rounds blocks; allow 2 MiB of slack per context tensor.
            assert expected <= saved <= expected + contexts * 8 * (2 << 20), (saved, expected)
    finally:
        functional.clear_workspace_cache()
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
def test_checkpoint_replay_keeps_the_context():
    """'moe' recompute replays the MoK call inside the checkpoint's backward; with the option on,
    that call keeps its context (no rebuild) and gradients stay bitwise identical."""
    functional = _require_mok_layer(False)
    try:
        layer, backend, parameters = _build_mok_layer(
            "mxfp8", "situ", MULTI_MACROBATCH, None, False
        )
        inputs = _layer_inputs(layer.config.num_moe_experts)
        _run(functional, backend, parameters, inputs, rebuild=False, checkpointed=True)
        off = _run(functional, backend, parameters, inputs, rebuild=False, checkpointed=True)
        on = _run(functional, backend, parameters, inputs, rebuild=True, checkpointed=True)
        assert on.rebuilds == 0
        _assert_same_results(on, off, "checkpointed, option on vs off")
        plain = _run(functional, backend, parameters, inputs, rebuild=True)
        assert plain.rebuilds == 1
        _assert_same_results(plain, off, "uncheckpointed option on vs checkpointed off")
    finally:
        functional.clear_workspace_cache()
        Utils.destroy_model_parallel()
