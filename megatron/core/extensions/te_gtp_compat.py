# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transformer Engine compatibility shim for GTP-sharded ``LayerNormLinear`` weights.

When weight-gradient accumulation is fused into ``main_grad`` (Megatron's
``grad_added_to_main_grad`` convention), TE's ``_LayerNormLinear`` backward returns a placeholder
weight gradient to autograd. Before TE be880cc2 (PR #3748) it sized that placeholder from the
GEMM gradient buffer. For a ``DistributedWeight`` such as a GTP shard that buffer is the full,
unsharded scratch from ``grad_buffer()``, while autograd expects a gradient shaped like the local
shard, so backward fails with "returned an invalid gradient at index 3".

A TE build is affected only when both hold:

* the placeholder is sized from ``main_grad`` (every build before be880cc2), and
* the backward's origin weight is the parameter itself. Builds from 76bea96c (PR #3493) on keep a
  weak reference to the parameter. Earlier builds, such as 0a2ebe94, keep it to the gathered
  weight, which carries no ``grad_added_to_main_grad``; they skip the placeholder branch and
  return ``None``, which GTP accepts because it fires DDP's grad-ready hook itself.

:func:`apply_layernorm_linear_distributed_wgrad_fix` patches ``_LayerNormLinear.backward`` when
the installed source still sizes the placeholder from ``main_grad``. The wrapper reads the origin
weight before TE clears its reference and replaces the placeholder only when the origin weight is
a distributed weight whose shape differs from it, using TE's own ``get_dummy_wgrad``. It never
changes a gradient on builds where that branch is unreachable, and it is not installed on fixed
builds.

TE caches placeholders by shape, so on an affected build each replaced, unsharded placeholder
would otherwise stay cached (one full-size buffer per distinct GTP ``LayerNormLinear`` shape for
the whole run). The wrapper drops that cache entry after replacing it; TE allocates the buffer
again, transiently, on its next backward.

The helpers above the patch function are pure Python so they can be tested without torch.
"""

import inspect
import logging
import re
from typing import Callable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The pre-be880cc2 placeholder: ``get_dummy_wgrad(list(main_grad.shape), ...)``.
_MAIN_GRAD_SHAPED_PLACEHOLDER = re.compile(r"get_dummy_wgrad\(\s*list\(\s*main_grad\.shape\s*\)")

# Set on the installed wrapper so a second call (or a reloaded caller) never wraps twice.
_FIX_MARKER = "_megatron_gtp_distributed_wgrad_fix"

_POSITIONAL_KINDS = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")

# Outcomes of apply_layernorm_linear_distributed_wgrad_fix().
PATCHED = "patched"
ALREADY_PATCHED = "already-patched"
NOT_NEEDED = "not-needed"
UNAVAILABLE = "unavailable"
UNSUPPORTED = "unsupported"

# Decisions that leave a class unpatched, so its source is inspected once per process. A patched
# class is recognized by the marker on its backward instead.
_unpatched_outcomes = {}


def source_sizes_placeholder_from_main_grad(source: str) -> bool:
    """True if TE source builds the fused-wgrad placeholder from ``main_grad``'s shape."""
    return _MAIN_GRAD_SHAPED_PLACEHOLDER.search(source) is not None


def weight_grad_index(
    forward_parameters: Sequence[Tuple[str, str]], weight_name: str = "weight"
) -> Optional[int]:
    """Return the weight's slot in an autograd Function's backward outputs.

    ``forward_parameters`` lists ``(name, kind)`` for ``Function.forward``, ``ctx`` first, where
    ``kind`` is the ``inspect.Parameter`` kind name. Backward returns one gradient per positional
    forward input after ``ctx``, so the weight's slot is its position minus one. Returns ``None``
    when the weight is not a plain positional input (for example behind ``*args``).
    """
    names = []
    for name, kind in forward_parameters:
        if kind not in _POSITIONAL_KINDS:
            break
        names.append(name)
    if weight_name not in names[1:]:
        return None
    return names.index(weight_name, 1) - 1


def placeholder_mismatches_weight(placeholder_shape, weight_shape) -> bool:
    """True if a placeholder weight gradient is not shaped like its parameter."""
    return tuple(placeholder_shape) != tuple(weight_shape)


def _origin_weight_before_backward(ctx):
    """Resolve the weight TE's backward reads through its weak reference, before TE clears it.

    TE keeps the backward state on ``ctx`` (builds before PR #3493) or on the
    ``ctx.backward_objects`` dataclass (later builds). Returns ``None`` unless the forward recorded
    a fused-accumulation weight gradient.
    """
    state = getattr(ctx, "backward_objects", None)
    if state is None:
        state = ctx
    if not getattr(state, "fuse_wgrad_accumulation", False):
        return None
    if not getattr(state, "requires_wgrad", False):
        return None
    ref = getattr(state, "origin_weight_ref", None)
    return ref() if ref is not None else None


def _storage_address(tensor) -> int:
    """Address of a tensor's storage, or 0 if it has none (e.g. a ZeroTensor placeholder)."""
    try:
        if tensor._is_zerotensor():
            return 0
        return tensor.untyped_storage().data_ptr()
    except (AttributeError, NotImplementedError, RuntimeError):
        return 0


def _evict_cached_placeholder(cache, placeholder) -> None:
    """Drop the entry of TE's placeholder cache that backs ``placeholder``, if any.

    TE returns a detached view of the cached tensor, so the entry is found by storage, not by key
    (the key layout differs between TE builds). Only the replaced placeholder's own entry goes.
    """
    if not isinstance(cache, dict):
        return
    address = _storage_address(placeholder)
    if address == 0:
        return
    for key, cached in list(cache.items()):
        if _storage_address(cached) == address and tuple(cached.shape) == tuple(placeholder.shape):
            del cache[key]


def _accepts_keyword(function: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _wrap_backward(
    backward: Callable, index: int, get_dummy_wgrad: Callable, dummy_wgrad_cache=None
) -> Callable:
    """Return ``backward`` with a shard-shaped placeholder for distributed weights.

    ``dummy_wgrad_cache`` is the dict behind ``get_dummy_wgrad`` (TE's ``_dummy_wgrads``); the
    replaced full-size placeholder's entry is dropped from it. ``None`` skips that step.
    """
    supports_zero_tensor = _accepts_keyword(get_dummy_wgrad, "zero_tensor")

    def backward_with_shard_shaped_placeholder(ctx, *grad_outputs):
        origin_weight = _origin_weight_before_backward(ctx)
        grads = backward(ctx, *grad_outputs)
        if origin_weight is None or not getattr(origin_weight, "is_distributed_weight", False):
            return grads
        if not isinstance(grads, (tuple, list)) or index >= len(grads):
            return grads
        placeholder = grads[index]
        if placeholder is None or not placeholder_mismatches_weight(
            placeholder.shape, origin_weight.shape
        ):
            return grads
        # Same flags TE uses for its own placeholder; the real gradient is already in main_grad.
        kwargs = {}
        if getattr(origin_weight, "zero_out_wgrad", False):
            kwargs["zero"] = True
        elif supports_zero_tensor:
            kwargs["zero_tensor"] = getattr(origin_weight, "zero_tensor_dummy_wgrad", False)
        grads = list(grads)
        grads[index] = get_dummy_wgrad(list(origin_weight.shape), origin_weight.dtype, **kwargs)
        # Do not keep TE's unsharded placeholder cached for the rest of the run.
        _evict_cached_placeholder(dummy_wgrad_cache, placeholder)
        return tuple(grads)

    backward_with_shard_shaped_placeholder.__doc__ = backward.__doc__
    backward_with_shard_shaped_placeholder.__wrapped__ = backward
    setattr(backward_with_shard_shaped_placeholder, _FIX_MARKER, True)
    return backward_with_shard_shaped_placeholder


def _log_rank_0(level: int, message: str) -> None:
    try:
        import torch

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
    except ImportError:
        pass
    logger.log(level, message)


def apply_layernorm_linear_distributed_wgrad_fix(
    function_cls=None, get_dummy_wgrad=None, source: Optional[str] = None, dummy_wgrad_cache=None
) -> str:
    """Patch TE's ``_LayerNormLinear.backward`` once if the installed TE needs it.

    Idempotent; call it whenever a GTP-sharded TE ``LayerNormLinear`` is built. The arguments
    default to the installed TE and exist for tests.

    Args:
        function_cls: The autograd Function to patch (default: TE's ``_LayerNormLinear``).
        get_dummy_wgrad: Placeholder factory (default: the one TE's module uses).
        source: Source to inspect for the bug (default: the module defining ``function_cls``).
        dummy_wgrad_cache: The placeholder cache behind ``get_dummy_wgrad`` (default: the
            ``_dummy_wgrads`` dict of the module defining it, when there is one).

    Returns:
        One of ``"patched"``, ``"already-patched"``, ``"not-needed"`` (fixed TE),
        ``"unavailable"`` (no TE ``LayerNormLinear``) or ``"unsupported"`` (the weight gradient
        cannot be located; a warning is logged and TE is left unchanged).
    """
    module = None
    if function_cls is None:
        try:
            from transformer_engine.pytorch.module import layernorm_linear as module
        except ImportError:
            return UNAVAILABLE
        function_cls = getattr(module, "_LayerNormLinear", None)
        if function_cls is None:
            return UNAVAILABLE
    else:
        module = inspect.getmodule(function_cls)

    backward = inspect.getattr_static(function_cls, "backward", None)
    if isinstance(backward, staticmethod):
        backward = backward.__func__
    if backward is None:
        return UNAVAILABLE
    if getattr(backward, _FIX_MARKER, False):
        return ALREADY_PATCHED
    if function_cls in _unpatched_outcomes:
        return _unpatched_outcomes[function_cls]

    if source is None:
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            # Without source the wrapper still only acts on a mismatched placeholder.
            source = None
    if source is not None and not source_sizes_placeholder_from_main_grad(source):
        _unpatched_outcomes[function_cls] = NOT_NEEDED
        return NOT_NEEDED

    if get_dummy_wgrad is None:
        get_dummy_wgrad = getattr(module, "get_dummy_wgrad", None)
        if get_dummy_wgrad is None:
            from transformer_engine.pytorch.module.base import get_dummy_wgrad
    if dummy_wgrad_cache is None:
        dummy_wgrad_cache = getattr(inspect.getmodule(get_dummy_wgrad), "_dummy_wgrads", None)

    forward = inspect.getattr_static(function_cls, "forward", None)
    if isinstance(forward, staticmethod):
        forward = forward.__func__
    index = None
    if forward is not None:
        index = weight_grad_index(
            [(p.name, p.kind.name) for p in inspect.signature(forward).parameters.values()]
        )
    if index is None:
        _unpatched_outcomes[function_cls] = UNSUPPORTED
        _log_rank_0(
            logging.WARNING,
            f"{function_cls.__qualname__}.forward has no positional 'weight' input; the GTP "
            "placeholder-gradient shim for TE LayerNormLinear is not installed. A GTP-sharded "
            "LayerNormLinear may fail in backward on this TE build (fixed upstream by TE "
            "be880cc2).",
        )
        return UNSUPPORTED

    function_cls.backward = staticmethod(
        _wrap_backward(backward, index, get_dummy_wgrad, dummy_wgrad_cache)
    )
    _log_rank_0(
        logging.INFO,
        f"Wrapped {function_cls.__qualname__}.backward (TE before be880cc2): a placeholder weight "
        "gradient that this TE sizes from the unsharded GEMM scratch of a GTP-sharded weight is "
        "replaced by a shard-shaped one; all other gradients pass through unchanged.",
    )
    return PATCHED
