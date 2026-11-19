# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situlu
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel import (
    megakernel_fuses_shared_expert,
    prepare_megakernel_shared_expert_config,
)
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version


def _mok_transformer_config(**overrides):
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
        "moe_shared_expert_intermediate_size": 256,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "add_bias_linear": False,
        "gradient_accumulation_fusion": True,
        "moe_megakernel_backend": "mok",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_mxfp8_shared_expert_config_expresses_bf16_module(monkeypatch, recwarn):
    original_recipe = object()
    config = SimpleNamespace(
        fp8="hybrid", fp8_param=True, quant_recipe=original_recipe, sentinel=object()
    )
    monkeypatch.setattr(weights, "_SHARED_EXPERT_BF16_WARNING_EMITTED", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    shared_config = weights.prepare_shared_expert_config(config)

    assert shared_config is not config
    assert config.fp8 == "hybrid" and config.fp8_param
    assert shared_config.fp8 is None and not shared_config.fp8_param
    assert shared_config.quant_recipe is original_recipe
    assert shared_config.sentinel is config.sentinel
    assert len(recwarn) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"moe_token_dispatcher_type": "alltoall"},
        {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "hybridep"},
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "hybridep",
            "sequence_packing_scheduler": "dp_balanced",
            "max_seqlen_per_dp_cp_rank": 1024,
            "calculate_per_token_loss": True,
            "variable_seq_lengths": True,
        },
        {"moe_shared_expert_gate": True},
        {"moe_router_dtype": None},
        {"moe_router_dtype": "fp32"},
        {"moe_shared_expert_gate": True, "activation_func": situlu},
        {"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True, "moe_shared_expert_gate": True},
        {
            "fp8": "hybrid",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_modules": [],
            "moe_layer_recompute": True,
        },
    ],
)
def test_mok_accepts_key_supported_configurations(overrides):
    assert _mok_transformer_config(**overrides).moe_megakernel_backend == "mok"


def test_native_moe_still_accepts_fp64_router():
    config = _mok_transformer_config(moe_megakernel_backend=None, moe_router_dtype="fp64")
    assert config.moe_router_dtype == "fp64"


@pytest.mark.skipif(not is_te_min_version("2.14.0"), reason="single grouped weights need TE>=2.14")
def test_native_single_grouped_weights_do_not_require_op_fuser():
    config = _mok_transformer_config(
        moe_megakernel_backend=None,
        moe_single_grouped_weight=True,
        moe_use_grouped_tensor=True,
        use_transformer_engine_op_fuser=False,
    )
    assert config.moe_single_grouped_weight
    assert not config.use_transformer_engine_op_fuser


def test_shared_execution_is_independent_of_activation_and_preserves_native_precision():
    for activation in (F.silu, situlu):
        for options in (
            None,
            {},
            {"fwd_num_comm_sms": 32},
            {"shared_expert_mode": "auto"},
            {"shared_expert_mode": "fused"},
        ):
            config = _mok_transformer_config(
                activation_func=activation, moe_megakernel_backend_config=options
            )
            assert megakernel_fuses_shared_expert(config)
        for overrides in (
            {"moe_megakernel_backend_config": {"shared_expert_mode": "native"}},
            {"moe_latent_size": 64},
            {"moe_shared_expert_intermediate_size": 512},
            {"moe_shared_expert_intermediate_size": None},
            {"tensor_model_parallel_size": 4, "sequence_parallel": True},
        ):
            config = _mok_transformer_config(
                activation_func=activation,
                fp8="hybrid",
                fp8_recipe="mxfp8",
                fp8_param=True,
                **overrides,
            )
            assert not megakernel_fuses_shared_expert(config)
            assert prepare_megakernel_shared_expert_config(config) is config
            assert config.fp8_param and config.fp8 == "hybrid"
    for overrides in (
        {"moe_latent_size": 64},
        {"moe_shared_expert_intermediate_size": 512},
        {"moe_shared_expert_intermediate_size": None},
        {"tensor_model_parallel_size": 4, "sequence_parallel": True},
    ):
        with pytest.raises(ValueError, match="Fused MOK shared execution requires"):
            _mok_transformer_config(
                moe_megakernel_backend_config={"shared_expert_mode": "fused"}, **overrides
            )
    for mode in ("invalid", None):
        with pytest.raises(ValueError, match="must be auto, fused, or native"):
            _mok_transformer_config(moe_megakernel_backend_config={"shared_expert_mode": mode})
    assert not megakernel_fuses_shared_expert(_mok_transformer_config(moe_megakernel_backend=None))
    for options in ({"shared_expert_mode": "auto"}, {"fwd_num_comm_sms": 32}):
        with pytest.raises(ValueError, match="requires moe_megakernel_backend to be set"):
            _mok_transformer_config(
                moe_megakernel_backend=None, moe_megakernel_backend_config=options
            )
    with pytest.raises(ValueError, match="requires sequence_parallel=True"):
        _mok_transformer_config(tensor_model_parallel_size=4)
    with pytest.raises(ValueError, match="requires expert TP=1"):
        _mok_transformer_config(expert_tensor_parallel_size=2)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"moe_shared_expert_intermediate_size": None, "add_bias_linear": True},
            "MOK requires add_bias_linear=False",
        ),
        ({"bf16": False, "fp16": True}, "FP32, FP16, and FP4 are not supported"),
        ({"fp4": "e2m1"}, "FP32, FP16, and FP4 are not supported"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"moe_router_dtype": "fp64"}, "MOK does not support moe_router_dtype='fp64'"),
        ({"moe_virtual_expert_load_balance": True}, "MOK does not support MoonEP"),
        ({"use_transformer_engine_op_fuser": True}, "use_transformer_engine_op_fuser=False"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not support overlap_moe_expert_parallel_comm",
        ),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": False}, "fp8_param=True"),
        ({"moe_shortcut_connection": True}, "does not support moe_shortcut_connection"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"moe_apply_probs_on_input": True}, "moe_apply_probs_on_input"),
        ({"glu_linear_offset": 1.0}, "glu_linear_offset"),
        ({"activation_func_tanh_clamp_scale": 1.0}, "activation_func_tanh_clamp_scale"),
        ({"gtp_weight_remat_size": 2}, "GTP"),
        ({"expert_gtp_weight_remat_size": 2}, "GTP"),
        (
            {
                "transformer_impl": "inference_optimized",
                "moe_router_dtype": "fp32",
                "inference_grouped_gemm_backend": "torch",
            },
            "does not support inference_optimized",
        ),
    ],
)
def test_mok_rejects_key_incompatible_configurations(overrides, error):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**overrides)


_NATIVE_SHARED = {"moe_megakernel_backend_config": {"shared_expert_mode": "native"}}


@pytest.mark.parametrize(
    "overrides",
    [
        # Native shared experts: MOK reads only routed weights, which expert GTP 1 never shards.
        {"gtp_weight_remat_size": 2, **_NATIVE_SHARED},
        # The user-facing knob resolves to the same dense GTP degree.
        {"tensor_parallel_num_weight_shards": 4, **_NATIVE_SHARED},
        # No shared expert: there is nothing for auto mode to fuse.
        {"gtp_weight_remat_size": 2, "moe_shared_expert_intermediate_size": None},
        # A structurally incompatible shared expert makes auto mode resolve to native.
        {"gtp_weight_remat_size": 2, "moe_shared_expert_intermediate_size": 512},
        # Kimi-K3-like routed-only MXFP8 with SiTU-GLU.
        {
            "tensor_parallel_num_weight_shards": 8,
            "fp8": "hybrid",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "activation_func": situlu,
            **_NATIVE_SHARED,
        },
    ],
)
def test_mok_accepts_dense_gtp_with_native_shared_experts(overrides):
    config = _mok_transformer_config(**overrides)
    assert config.gtp_weight_remat_size > 1
    assert config.expert_gtp_weight_remat_size == 1
    assert not megakernel_fuses_shared_expert(config)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        # Auto mode fuses the compatible default shared expert into the megakernel.
        ({"gtp_weight_remat_size": 2}, "requires shared experts to run natively"),
        (
            {
                "tensor_parallel_num_weight_shards": 2,
                "moe_megakernel_backend_config": {"shared_expert_mode": "fused"},
            },
            "requires shared experts to run natively",
        ),
        ({"expert_gtp_weight_remat_size": 2, **_NATIVE_SHARED}, "does not support expert GTP"),
        (
            {"expert_tensor_parallel_num_weight_shards": 2, **_NATIVE_SHARED},
            "does not support expert GTP",
        ),
        (
            {
                "tensor_parallel_num_weight_shards": 2,
                "expert_tensor_parallel_num_weight_shards": 2,
                **_NATIVE_SHARED,
            },
            "does not support expert GTP",
        ),
    ],
)
def test_mok_rejects_gtp_on_weights_the_megakernel_reads(overrides, error):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**overrides)


def _stub_mok(monkeypatch):
    """Stand in for mok.functional, so the adapter can be constructed without MOK."""
    package = ModuleType("mok")
    functional = ModuleType("mok.functional")
    functional.MoKConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    package.functional = functional
    monkeypatch.setitem(sys.modules, "mok", package)
    monkeypatch.setitem(sys.modules, "mok.functional", functional)


def _native_expert_modules(config, num_local_experts=2):
    """Per-expert routed weights and a gated combined shared MLP, shaped as MCore builds them."""
    hidden, ffn = config.hidden_size, config.moe_ffn_hidden_size

    def linear(**weights):
        module = torch.nn.Module()
        module.single_grouped_weight = False
        for name, shape in weights.items():
            module.register_parameter(
                name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
            )
        return module

    experts = torch.nn.Module()
    indices = range(num_local_experts)
    experts.linear_fc1 = linear(**{f"weight{i}": (2 * ffn, hidden) for i in indices})
    experts.linear_fc2 = linear(**{f"weight{i}": (hidden, ffn) for i in indices})
    shared = torch.nn.Module()
    shared.linear_fc1 = linear(weight=(2 * ffn, hidden))
    shared.linear_fc2 = linear(weight=(hidden, ffn))
    shared.register_parameter(
        "gate_weight", torch.nn.Parameter(torch.zeros((1, hidden), dtype=torch.bfloat16))
    )
    return experts, shared


def _gtp_shard(module, rows):
    """Replace ``module.weight`` with a GTP_remat shard: fewer rows, tagged like GTP tags it."""
    full = module.weight
    module.weight = torch.nn.Parameter(torch.zeros((rows, full.shape[1]), dtype=full.dtype))
    module.weight.is_gtp_weight_remat = True


@pytest.mark.parametrize(
    "sharded_input",
    [
        "routed_fc1_weight1",
        "routed_fc2_weight0",
        "shared_fc1_weight",
        "shared_fc2_weight",
        "shared_output_gate_weight",
    ],
)
def test_mok_backend_rejects_gtp_sharded_inputs(monkeypatch, sharded_input):
    """The adapter refuses GTP shards even if a process-group collection disagrees with the
    config, because the megakernel reads its registered weights without gathering them. A
    shared-weight shard also has the wrong shape, so the GTP error must come first."""
    from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel

    _stub_mok(monkeypatch)
    config = _mok_transformer_config(moe_shared_expert_gate=True)
    experts, shared = _native_expert_modules(config)
    if sharded_input.startswith("routed_"):
        fc = experts.linear_fc1 if "_fc1_" in sharded_input else experts.linear_fc2
        fc.get_parameter("weight" + sharded_input[-1]).is_gtp_weight_remat = True
    elif sharded_input == "shared_fc1_weight":
        _gtp_shard(shared.linear_fc1, config.moe_ffn_hidden_size)  # [2I/2, H] at GTP 2
    elif sharded_input == "shared_fc2_weight":
        _gtp_shard(shared.linear_fc2, config.hidden_size // 2)  # [H/2, I] at GTP 2
    else:
        shared.gate_weight.is_gtp_weight_remat = True

    with pytest.raises(ValueError, match=f"GTP-sharded MOK inputs: \\['{sharded_input}'\\]"):
        MoKMegakernel(config, object(), experts, shared, 2)


def test_mok_backend_accepts_unsharded_inputs(monkeypatch):
    """The same inputs build the adapter when no weight it reads is a GTP shard."""
    from megatron.core.transformer.moe.megakernel.mok.backend import MoKMegakernel

    _stub_mok(monkeypatch)
    config = _mok_transformer_config(moe_shared_expert_gate=True)
    experts, shared = _native_expert_modules(config)

    module = MoKMegakernel(config, object(), experts, shared, 2)

    assert module.shared_fc1_weight is shared.linear_fc1.weight
    assert module.shared_output_gate_weight is shared.gate_weight
    assert len(module.autograd_routed_parameters) == 4


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_rejects_per_layer_whole_layer_cuda_graph(cuda_graph_impl):
    with pytest.raises(ValueError, match="whole-layer CUDA Graph capture"):
        _mok_transformer_config(cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=[])


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [CudaGraphModule.moe],
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_preprocess],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ],
)
def test_mok_rejects_per_layer_cuda_graph_covering_moe(cuda_graph_impl, cuda_graph_modules):
    with pytest.raises(ValueError, match="moe/moe_router/moe_preprocess"):
        _mok_transformer_config(
            cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_accepts_per_layer_cuda_graph_outside_moe(cuda_graph_impl):
    cuda_graph_modules = [CudaGraphModule.attn]
    config = _mok_transformer_config(
        cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
    )

    assert config.cuda_graph_modules == cuda_graph_modules
