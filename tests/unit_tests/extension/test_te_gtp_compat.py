# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TE LayerNormLinear placeholder weight gradients for GTP-sharded (distributed) weights.

The CPU tests drive the shim through stand-in autograd Functions that reproduce the TE code
shapes involved: the affected builds (TE 76bea96c..be880cc2^), whose fused-wgrad placeholder is
sized from the full GEMM scratch while the origin weight is the shard, and the older 0a2ebe94
shape, whose origin weight is the gathered tensor. The CUDA test runs the installed TE.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.te_gtp_compat import (
    ALREADY_PATCHED,
    NOT_NEEDED,
    PATCHED,
    UNSUPPORTED,
    apply_layernorm_linear_distributed_wgrad_fix,
    placeholder_mismatches_weight,
    source_sizes_placeholder_from_main_grad,
    weight_grad_index,
)

_BUGGY_SOURCE = """
            if getattr(origin_weight, "zero_out_wgrad", False):
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight.dtype,
                    zero=True,
                )
"""
_FIXED_SOURCE = _BUGGY_SOURCE.replace("main_grad.shape", "origin_weight.shape")

_PLACEHOLDERS = {}


def _dummy_wgrad(shape, dtype, zero=False, zero_tensor=False):
    """CPU stand-in for TE's cached ``get_dummy_wgrad``."""
    key = (*shape, dtype)
    if key not in _PLACEHOLDERS:
        _PLACEHOLDERS[key] = torch.full(shape, 7.0, dtype=dtype)
    if zero:
        _PLACEHOLDERS[key].zero_()
    return _PLACEHOLDERS[key].detach()


class _ShardedParam(torch.nn.Parameter):
    """A DistributedWeight-like leaf: the shard is the parameter, the gather is ``full``."""

    is_distributed_weight = True


def _sharded_weight(rows=4, cols=3, shards=2, **attrs):
    full = torch.randn(rows, cols)
    weight = _ShardedParam(full[: rows // shards].clone())
    weight.full = full
    weight.scratch = torch.zeros(rows, cols)
    weight.grad_added_to_main_grad = False
    for name, value in attrs.items():
        setattr(weight, name, value)
    return weight


def _make_function(origin_is_gathered=False):
    """Build a stand-in for TE's ``_LayerNormLinear`` (refactored layout, weight at index 3)."""

    class _FusedWgradLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, ln_weight, ln_bias, weight, bias, fwd_args):
            gathered = weight.full.detach().requires_grad_(True)
            origin = gathered if origin_is_gathered else weight
            ctx.backward_objects = SimpleNamespace(
                fuse_wgrad_accumulation=True,
                requires_wgrad=True,
                origin_weight_ref=weakref.ref(origin),
                scratch=weight.scratch,
                inp=inp.detach(),
                gathered=gathered,
            )
            return inp @ gathered.detach().t()

        @staticmethod
        def backward(ctx, grad_output):
            args = ctx.backward_objects
            ctx.backward_objects = None
            origin_weight = args.origin_weight_ref()
            args.origin_weight_ref = None
            main_grad = args.scratch
            main_grad.copy_(grad_output.t() @ args.inp)
            wgrad = None
            if hasattr(origin_weight, "grad_added_to_main_grad"):
                origin_weight.grad_added_to_main_grad = True
                wgrad = _dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight.dtype,
                    zero=getattr(origin_weight, "zero_out_wgrad", False),
                )
            return grad_output @ args.gathered.detach(), None, None, wgrad, None, None

    return _FusedWgradLinear


def _run(function, weight):
    inp = torch.randn(5, weight.full.shape[1], requires_grad=True)
    function.apply(inp, None, None, weight, None, None).sum().backward()
    return inp


def test_source_detection():
    assert source_sizes_placeholder_from_main_grad(_BUGGY_SOURCE)
    assert not source_sizes_placeholder_from_main_grad(_FIXED_SOURCE)


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        # TE 0a2ebe94 _LayerNormLinear.forward
        (
            ["ctx", "inp", "ln_weight", "ln_bias", "weight", "weight_workspace", "bias", "args"],
            3,
        ),
        # TE after PR #3493
        (["ctx", "inp", "ln_weight", "ln_bias", "weight", "bias", "fwd_args"], 3),
        (["ctx", "weight", "inp"], 0),
        (["ctx", "inp", "bias"], None),
    ],
)
def test_weight_grad_index(parameters, expected):
    assert weight_grad_index([(name, "POSITIONAL_OR_KEYWORD") for name in parameters]) == expected


def test_weight_grad_index_requires_positional_weight():
    parameters = [("ctx", "POSITIONAL_OR_KEYWORD"), ("args", "VAR_POSITIONAL")]
    assert weight_grad_index(parameters + [("weight", "KEYWORD_ONLY")]) is None


def test_placeholder_mismatch():
    assert placeholder_mismatches_weight((4, 3), (2, 3))
    assert not placeholder_mismatches_weight(torch.Size([2, 3]), (2, 3))


def test_affected_build_fails_without_the_fix():
    with pytest.raises(RuntimeError, match="invalid gradient at index 3"):
        _run(_make_function(), _sharded_weight())


@pytest.mark.parametrize("zero_out_wgrad", [False, True])
def test_fix_returns_shard_shaped_placeholder(zero_out_wgrad):
    function = _make_function()
    outcome = apply_layernorm_linear_distributed_wgrad_fix(
        function, _dummy_wgrad, source=_BUGGY_SOURCE
    )
    assert outcome == PATCHED
    weight = _sharded_weight(zero_out_wgrad=zero_out_wgrad)

    inp = _run(function, weight)

    assert weight.grad_added_to_main_grad
    assert weight.grad.shape == weight.shape
    if zero_out_wgrad:
        assert torch.count_nonzero(weight.grad) == 0
    # dgrad and the real weight gradient (left in main_grad) are untouched.
    torch.testing.assert_close(inp.grad, weight.full.sum(dim=0).expand_as(inp))
    torch.testing.assert_close(weight.scratch, inp.detach().sum(dim=0).expand_as(weight.scratch))


def test_fix_evicts_the_unsharded_placeholder_from_the_cache():
    """The replaced full-size placeholder must not stay cached; other entries stay."""
    function = _make_function()
    outcome = apply_layernorm_linear_distributed_wgrad_fix(
        function, _dummy_wgrad, source=_BUGGY_SOURCE, dummy_wgrad_cache=_PLACEHOLDERS
    )
    assert outcome == PATCHED
    unrelated_key = (4, 3, torch.float64)
    unrelated = _PLACEHOLDERS.setdefault(unrelated_key, torch.zeros(4, 3, dtype=torch.float64))
    weight = _sharded_weight()

    for _ in range(2):  # TE re-creates the full-size placeholder each backward; it goes again.
        _run(function, weight)
        assert weight.grad.shape == weight.shape
        assert (4, 3, torch.float32) not in _PLACEHOLDERS
        assert (2, 3, torch.float32) in _PLACEHOLDERS
        assert _PLACEHOLDERS[unrelated_key] is unrelated
        weight.grad = None


def test_fix_is_idempotent_and_keeps_original_backward():
    function = _make_function()
    original = function.backward
    outcome = apply_layernorm_linear_distributed_wgrad_fix(
        function, _dummy_wgrad, source=_BUGGY_SOURCE
    )
    assert outcome == PATCHED
    patched = function.backward
    outcome = apply_layernorm_linear_distributed_wgrad_fix(
        function, _dummy_wgrad, source=_BUGGY_SOURCE
    )
    assert outcome == ALREADY_PATCHED
    assert function.backward is patched
    assert patched.__wrapped__ is original


def test_fixed_source_is_not_patched():
    function = _make_function()
    original = function.backward
    outcome = apply_layernorm_linear_distributed_wgrad_fix(
        function, _dummy_wgrad, source=_FIXED_SOURCE
    )
    assert outcome == NOT_NEEDED
    assert function.backward is original


def test_forward_without_weight_input_is_left_alone():
    class _NoWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tensors):
            return tensors[0]

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    original = _NoWeight.backward
    outcome = apply_layernorm_linear_distributed_wgrad_fix(
        _NoWeight, _dummy_wgrad, source=_BUGGY_SOURCE
    )
    assert outcome == UNSUPPORTED
    assert _NoWeight.backward is original


def test_gathered_origin_weight_keeps_none_gradient():
    """TE 0a2ebe94 references the gathered weight and returns no placeholder at all."""
    function = _make_function(origin_is_gathered=True)
    apply_layernorm_linear_distributed_wgrad_fix(function, _dummy_wgrad, source=_BUGGY_SOURCE)
    weight = _sharded_weight()

    _run(function, weight)

    assert weight.grad is None
    assert not weight.grad_added_to_main_grad


def test_plain_weight_placeholder_is_untouched():
    """Non-distributed weights keep TE's own placeholder object."""
    function = _make_function()
    apply_layernorm_linear_distributed_wgrad_fix(function, _dummy_wgrad, source=_BUGGY_SOURCE)
    full = torch.randn(4, 3)
    weight = torch.nn.Parameter(full.clone())
    weight.full = full
    weight.scratch = torch.zeros(4, 3)
    weight.grad_added_to_main_grad = False
    seen = []
    weight.register_hook(lambda grad: seen.append(grad.data_ptr()))

    _run(function, weight)

    assert seen == [_PLACEHOLDERS[(4, 3, torch.float32)].data_ptr()]


def test_no_grad_forward_does_not_touch_the_shim():
    function = _make_function()
    apply_layernorm_linear_distributed_wgrad_fix(function, _dummy_wgrad, source=_BUGGY_SOURCE)
    weight = _sharded_weight()
    with torch.no_grad():
        out = function.apply(torch.randn(5, 3), None, None, weight, None, None)
    assert out.shape == (5, 4)
    assert not out.requires_grad


# ---------------------------------------------------------------------------
# Installed Transformer Engine (single GPU)
# ---------------------------------------------------------------------------


class _TEShardedWeight(torch.nn.Parameter):
    """DistributedWeight stub: ``main_grad`` is shard-shaped, ``grad_buffer()`` is the full
    GEMM scratch (TE be880cc2's regression fixture, adapted to accept either TE layout)."""

    is_distributed_weight = True

    def __new__(cls, full_weight, shards):
        shard = full_weight[: full_weight.shape[0] // shards].clone()
        return super().__new__(cls, shard, requires_grad=True)

    def __init__(self, full_weight, shards):
        self.full_weight = full_weight.detach().clone()
        self.main_grad = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.scratch = torch.empty_like(self.full_weight, dtype=torch.float32)
        self.grad_added_to_main_grad = False

    def materialize_group_for_forward(self):
        return self.full_weight.detach().requires_grad_(True)

    def materialize_group_for_backward(self, **kwargs):
        return self.full_weight.detach().requires_grad_(True)

    def grad_buffer(self):
        return self.scratch

    def finalize_group_grads(self, gradient, **kwargs):
        assert gradient.shape == self.full_weight.shape
        self.main_grad.add_(gradient[: self.shape[0]])
        return torch.zeros(self.shape, dtype=self.dtype, device=self.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("shards", [1, 2])
@pytest.mark.parametrize("zero_out_wgrad", [False, True])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_installed_te_layernorm_linear_with_sharded_weight(shards, zero_out_wgrad, normalization):
    """Backward through TE LayerNormLinear with a sharded fused-wgrad weight succeeds and the
    autograd placeholder, if TE returns one, is shaped like the shard."""
    te = pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("transformer_engine.pytorch.distributed_weight")

    outcome = apply_layernorm_linear_distributed_wgrad_fix()
    assert outcome in (PATCHED, ALREADY_PATCHED, NOT_NEEDED), outcome

    torch.manual_seed(271828)
    module, reference = (
        te.LayerNormLinear(
            128,
            256,
            normalization=normalization,
            bias=False,
            fuse_wgrad_accumulation=True,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(2)
    )
    reference.load_state_dict(module.state_dict())
    module.weight = _TEShardedWeight(module.weight.detach(), shards)
    reference.weight.main_grad = torch.zeros_like(reference.weight, dtype=torch.float32)
    reference.weight.grad_added_to_main_grad = False
    for weight in (module.weight, reference.weight):
        weight.zero_out_wgrad = zero_out_wgrad

    placeholder_shapes = []
    module.weight.register_hook(lambda grad: placeholder_shapes.append(tuple(grad.shape)))
    for microbatch in range(2):
        inputs = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        actual_input = inputs.clone().requires_grad_(True)
        reference_input = inputs.clone().requires_grad_(True)
        actual_output = module(actual_input, is_first_microbatch=microbatch == 0)
        reference_output = reference(reference_input, is_first_microbatch=microbatch == 0)
        actual_output.float().square().mean().backward()
        reference_output.float().square().mean().backward()

        for actual, expected in (
            (actual_output, reference_output),
            (actual_input.grad, reference_input.grad),
            (module.weight.main_grad, reference.weight.main_grad[: module.weight.shape[0]]),
            (module.layer_norm_weight.grad, reference.layer_norm_weight.grad),
        ):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=1e-5)
        # TE 0a2ebe94 returns no placeholder for a distributed weight; later builds return one,
        # which must be shaped like the shard (this shim, or TE be880cc2 and later).
        assert all(shape == tuple(module.weight.shape) for shape in placeholder_shapes)
        # DDP discards the placeholder after its hook.
        module.weight.grad = None
        reference.weight.grad = None
