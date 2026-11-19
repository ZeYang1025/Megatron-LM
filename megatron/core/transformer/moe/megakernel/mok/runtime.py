# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK functional runtime and its MCore autograd bridge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import torch

from megatron.core._rank_utils import log_single_rank
from megatron.core.transformer.moe.megakernel.mok.fc1_layout import routed_forward_context_nbytes
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient as _finish_weight_gradient,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    main_grad_buffer as _main_grad_buffer,
)

if TYPE_CHECKING:
    from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel

logger = logging.getLogger(__name__)

# Whether this process has logged that moe_mok_recompute_forward_context released a forward
# context; the confirmation is logged once per run, not per layer and microbatch.
_forward_context_release_logged = False


def _gate_up_weight_arguments(shared_fc1: torch.Tensor | None, routed_fc1, intermediate_size: int):
    """Adapt MCore-owned combined FC1 weights to MOK's gate/up API.

    Shared FC1 is a dense two-dimensional parameter, so gate and up are
    ordinary zero-copy row views. Routed FC1 may span multiple experts or
    descriptor-backed allocations; passing the same combined payload for gate
    and up preserves its storage metadata, and MOK applies the existing logical
    gate/up offsets when the two arguments alias.
    """
    if shared_fc1 is None:
        return None, None, routed_fc1, routed_fc1
    expected_rows = 2 * intermediate_size
    if shared_fc1.ndim != 2 or shared_fc1.shape[0] != expected_rows:
        raise RuntimeError(
            "MOK shared FC1 shape mismatch: "
            f"got {tuple(shared_fc1.shape)}, expected [{expected_rows}, hidden_size]"
        )
    shared_gate = shared_fc1.narrow(0, 0, intermediate_size)
    shared_up = shared_fc1.narrow(0, intermediate_size, intermediate_size)
    return shared_gate, shared_up, routed_fc1, routed_fc1


def _gate_up_main_grad_arguments(main_grads, main_grad_storage_tables, intermediate_size: int):
    """Adapt canonical FC1/FC2 main-grad buffers to MOK's six-buffer order."""
    shared_fc1, routed_fc1, shared_fc2, routed_fc2 = main_grads
    shared_gate, shared_up, routed_gate, routed_up = _gate_up_weight_arguments(
        shared_fc1, routed_fc1, intermediate_size
    )
    split_main_grads = (shared_gate, routed_gate, shared_up, routed_up, shared_fc2, routed_fc2)
    if main_grad_storage_tables is None:
        return split_main_grads, None
    routed_fc1_table, routed_fc2_table = main_grad_storage_tables
    return split_main_grads, (routed_fc1_table, routed_fc1_table, routed_fc2_table)


def _in_autograd_backward() -> bool:
    """Return whether this thread is executing an autograd GraphTask.

    An activation checkpoint replays its region's forward inside the checkpoint node's backward
    and backpropagates the replayed region right after it.
    """
    return torch._C._current_graph_task_id() != -1


def _log_first_forward_context_release(module: "MoKMegakernel") -> None:
    """Log once per process, on the default log rank, that forward released MOK's context.

    Confirms in the run log that moe_mok_recompute_forward_context acts at runtime: a MoE
    forward that autograd will backpropagate released its MOK forward context, which the layer
    rebuilds right before its backward. Host-side only; later calls return at once.
    """
    global _forward_context_release_logged
    if _forward_context_release_logged:
        return
    _forward_context_release_logged = True
    macrobatch_size = getattr(module.mok_config, "macrobatch_size", None)
    try:
        context_bytes = routed_forward_context_nbytes(
            macrobatch_size,
            module.hidden_size,
            module.intermediate_size,
            mxfp8=module.use_mxfp8_weights,
        )
        size = (
            f"routed part: {macrobatch_size} rows, {context_bytes / 2**30:.2f} GiB per layer "
            "and microbatch"
        )
    except ValueError:
        # No valid macrobatch_size in the MOK config: confirm the release without its size.
        size = "size unknown"
    log_single_rank(
        logger,
        logging.INFO,
        "MOK forward-context recompute (moe_mok_recompute_forward_context) is active: MoE "
        "forward released MOK's forward context (%s), and each layer rebuilds it right before "
        "its backward. Logged once, at the first release.",
        size,
    )


def _rebuild_forward_context(
    module: "MoKMegakernel", workspace, schedule, x: torch.Tensor, shared_fc1, fc1_weight_view
):
    """Recompute the MOK forward context that forward dropped (moe_mok_recompute_forward_context).

    MOK's ``recompute_forward_context`` reruns dispatch, FC1 and the activation for the first
    macrobatch from the forward's input, schedule, config and weights. MOK's kernels are
    deterministic and only bytes move when FC1 is packed, so the rebuilt context equals the
    dropped one on every row MOK wrote.

    The rebuild is an expert-parallel collective: it copies ``x`` into the symmetric dispatch
    buffer and reads the peers' buffers between MOK device barriers. Every EP rank must therefore
    reach it for the same layer and microbatch in the same order. It runs inside this autograd
    node's backward right before MOK's backward, itself an EP collective, so it inherits the
    order MOK already relies on: the EP ranks of a pipeline stage run the same schedule, so they
    backpropagate the same layers and microbatches in the same order. The MCore features that
    would reorder MoE backward work (EP A2A overlap, delayed wgrad) are rejected for MOK by
    TransformerConfig, and whether a call rebuilds depends only on the layer's
    ``recompute_forward_context`` setting (from the config) and on whether the call replays an
    activation-recompute region, which is the same on every EP rank.
    """
    from mok import functional

    routed_gate, routed_up = module.routed_fc1_recompute_weights(fc1_weight_view)
    shared_gate, shared_up, _, _ = _gate_up_weight_arguments(
        shared_fc1, None, module.intermediate_size
    )
    activation_kwargs = {} if module.situ_beta is None else {"situ_beta": module.situ_beta}
    forward_context = functional.recompute_forward_context(
        module.mok_config,
        workspace,
        schedule,
        x,
        shared_gate,
        shared_up,
        routed_gate,
        routed_up,
        swiglu_limit=module.swiglu_limit,
        **activation_kwargs,
    )
    # The packed FC1 copies are dead once the rebuild is enqueued. Dropping the last references
    # returns their blocks to this stream's pool, where any reuse is ordered after the rebuild
    # kernel, so MOK's backward can take the memory without record_stream.
    del routed_gate, routed_up
    return forward_context


class _MoKWeightArguments(NamedTuple):
    """Non-owning references in the external MOK forward/backward ABI order.

    Routed arguments may be tensors, MXFP8 tuples, or split-weight descriptors.
    This transient tuple neither copies payloads nor registers/caches parameters.
    """

    shared_gate: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor
    routed_gate: object
    routed_up: object
    routed_down: object

    @classmethod
    def from_combined(cls, shared_fc1, shared_fc2, routed_fc1, routed_fc2, intermediate_size):
        """Split only shared FC1 while retaining the caller's routed weight views."""
        shared_gate, shared_up, routed_gate, routed_up = _gate_up_weight_arguments(
            shared_fc1, routed_fc1, intermediate_size
        )
        return cls(shared_gate, shared_up, shared_fc2, routed_gate, routed_up, routed_fc2)


class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to MoK's functional API."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        shared_output_gate_weight: torch.Tensor | None,
        *parameters: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run MOK forward and retain the state needed by backward."""
        from mok import functional

        # moe_mok_recompute_forward_context: release the context (the first macrobatch's routed
        # rows, (H + 3I) * 33/32 bytes per row in MXFP8) after forward and rebuild it in backward.
        # A call that an activation checkpoint replays during backward keeps it: its backward
        # follows at once. Decided and validated before MOK's first EP collective, so that an
        # unsupported layer fails on every EP rank before any of them enters MOK's barriers.
        rebuild_forward_context = module.recompute_forward_context and not _in_autograd_backward()
        if rebuild_forward_context and shared_output_gate_weight is not None:
            raise RuntimeError(
                "MOK cannot rebuild a gated forward context; disable "
                "moe_mok_recompute_forward_context for MOK-fused gated shared experts"
            )

        num_routed_parameters = len(module.autograd_routed_parameters)
        if len(parameters) != num_routed_parameters + 2:
            raise RuntimeError(
                "MOK autograd parameter count mismatch: "
                f"got {len(parameters)}, expected {num_routed_parameters + 2}"
            )
        routed_parameters = parameters[:num_routed_parameters]
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
        workspace = functional.get_workspace(
            module.mok_config,
            module.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=top_experts.shape[1],
        )
        schedule = functional.build_schedule(
            workspace, module.mok_config, top_experts, num_local_experts=module.num_local_experts
        )
        fc1_weight_view, fc2_weight_view = module.quantized_routed_weights()
        if module.use_mxfp8_weights and module.native_single_grouped_weights:
            # Single-weight MXFP8 forward consumes only rowwise data and scale.
            fc1_forward = fc1_weight_view[:2]
            fc2_forward = fc2_weight_view[:2]
        else:
            # Single-weight BF16 has no separate rowwise/columnwise physical tuple.
            # Non-single BF16/MXFP8 use SplitRoutedWeight objects consumed directly by MOK.
            fc1_forward = fc1_weight_view
            fc2_forward = fc2_weight_view
        weight_args = _MoKWeightArguments.from_combined(
            shared_fc1, shared_fc2, fc1_forward, fc2_forward, module.intermediate_size
        )
        forward_kwargs = {}
        if shared_output_gate_weight is not None:
            forward_kwargs["shared_output_gate_weight"] = shared_output_gate_weight
        if module.situ_beta is not None:
            forward_kwargs["situ_beta"] = module.situ_beta
        output, forward_context = functional.forward(
            module.mok_config,
            workspace,
            schedule,
            x,
            router_weights,
            *weight_args,
            swiglu_limit=module.swiglu_limit,
            **forward_kwargs,
        )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.rebuild_forward_context = rebuild_forward_context
        ctx.forward_context = None if rebuild_forward_context else forward_context
        if rebuild_forward_context and any(ctx.needs_input_grad):
            # Only a forward that autograd records gets a backward, and so a rebuild (the
            # no-grad first pass of an activation checkpoint has neither).
            _log_first_forward_context_release(module)
        ctx.routed_weight_views = (fc1_weight_view, fc2_weight_view)
        ctx.save_for_backward(
            x, router_weights, shared_output_gate_weight, *routed_parameters, shared_fc1, shared_fc2
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Run MOK backward and return gradients for its autograd inputs."""
        from mok import functional

        x, router_weights, shared_output_gate_weight, *parameters = ctx.saved_tensors
        num_routed_parameters = len(ctx.module.autograd_routed_parameters)
        shared_fc1, shared_fc2 = parameters[num_routed_parameters:]
        fc1_weight_view, fc2_weight_view = ctx.routed_weight_views
        if ctx.module.use_mxfp8_weights and ctx.module.native_single_grouped_weights:
            # FC1 backward may need rowwise weights for recompute and columnwise weights for
            # dgrad, while FC2 only needs columnwise weights for dgrad.
            backward_fc1 = fc1_weight_view
            backward_fc2 = fc2_weight_view[2:]
        else:
            backward_fc1 = fc1_weight_view
            backward_fc2 = fc2_weight_view
        weight_args = _MoKWeightArguments.from_combined(
            shared_fc1, shared_fc2, backward_fc1, backward_fc2, ctx.module.intermediate_size
        )
        main_grads, main_grad_storage_tables = ctx.module.main_grad_arguments()
        main_grads, main_grad_storage_tables = _gate_up_main_grad_arguments(
            main_grads, main_grad_storage_tables, ctx.module.intermediate_size
        )
        gate_kwargs = {}
        if shared_output_gate_weight is not None:
            gate_kwargs = {
                "shared_output_gate_weight": shared_output_gate_weight,
                "shared_output_gate_main_grad": _main_grad_buffer(
                    ctx.module.shared_output_gate_weight
                ),
            }
        if ctx.rebuild_forward_context:
            forward_context = _rebuild_forward_context(
                ctx.module, ctx.workspace, ctx.schedule, x, shared_fc1, fc1_weight_view
            )
        else:
            forward_context = ctx.forward_context
        d_x, d_router_weights, *_ = functional.backward(
            ctx.module.mok_config,
            ctx.workspace,
            ctx.schedule,
            forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            *weight_args,
            swiglu_limit=ctx.module.swiglu_limit,
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
            **gate_kwargs,
        )
        # MOK's backward is enqueued on this stream; the context may return to the pool.
        del forward_context

        routed_parameter_grads = ctx.module.finish_routed_weight_gradients()
        d_shared_fc1 = (
            _finish_weight_gradient(ctx.module.shared_fc1_weight)
            if shared_fc1 is not None
            else None
        )
        d_shared_fc2 = (
            _finish_weight_gradient(ctx.module.shared_fc2_weight)
            if shared_fc2 is not None
            else None
        )
        d_output_gate = (
            _finish_weight_gradient(ctx.module.shared_output_gate_weight)
            if shared_output_gate_weight is not None
            else None
        )

        ctx.module = None
        ctx.workspace = None
        ctx.schedule = None
        ctx.forward_context = None
        ctx.routed_weight_views = None
        return (
            None,
            d_x,
            d_router_weights,
            None,
            d_output_gate,
            *routed_parameter_grads,
            d_shared_fc1,
            d_shared_fc2,
        )
