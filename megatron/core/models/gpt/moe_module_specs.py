# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.backends import (
    BackendSpecProvider,
    InferenceSpecProvider,
    LocalSpecProvider,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_layer import MlpBuilder


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> MlpBuilder:
    """Helper function to get module spec for MoE.

    Called by hybrid_layer_specs.py for standard (non-inference) MoE specs.
    The GPT layer specs call get_moe_module_spec_for_backend directly.

    Args:
        use_te: Whether to use Transformer Engine.
        num_experts: Number of experts.
        moe_grouped_gemm: Whether to use grouped GEMM.
        moe_use_legacy_grouped_gemm: Whether to use legacy grouped GEMM.
    """
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> MlpBuilder:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    experts = backend.grouped_mlp_modules(moe_grouped_gemm is not None and moe_grouped_gemm)
    # shared experts spec
    shared_experts = partial(SharedExpertMLP, submodules=mlp)

    # MoE module spec
    # PATCH(inference-router): the GPT inference_optimized path reaches here via
    # get_gpt_layer_with_inference_submodules with backend=InferenceSpecProvider, but no
    # router is set, so MoESubmodules falls back to the training TopKRouter (dense
    # [tokens, num_experts] bool output). get_inference_optimized_moe_spec below wires
    # router=InferenceTopKRouter (compact [tokens, topk] indices — what the fused-MoE
    # path and the AGV/permute kernels expect) and its docstring says it is called by
    # gpt_layer_specs.py, but only hybrid_layer_specs.py ever calls it. This missing
    # kwarg is the root cause of the dense-routing feed (permute IMA, NVLS "128 vs 8")
    # on GPT MoE models. Mirror the hybrid path's router here.
    submodule_kwargs = {}
    if isinstance(backend, InferenceSpecProvider):
        submodule_kwargs["router"] = InferenceTopKRouter
    return partial(
        MoELayer,
        submodules=MoESubmodules(
            experts=experts, shared_experts=shared_experts, **submodule_kwargs
        ),
    )


def get_inference_optimized_moe_spec() -> MlpBuilder:
    """MoE module spec for inference-optimized transformer impl.

    Uses InferenceSpecProvider to select inference-optimized modules:
    InferenceTopKRouter, InferenceGroupedMLP. MoELayer detects inference mode
    via config.transformer_impl and sets up the inference dispatcher internally.

    Called by hybrid_layer_specs.py and gpt_layer_specs.py.
    """
    backend = InferenceSpecProvider()
    activation_func = backend.activation_func()

    experts = backend.grouped_mlp_modules(True)
    shared_experts = partial(
        SharedExpertMLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
            activation_func=activation_func,
        ),
    )

    return partial(
        MoELayer,
        submodules=MoESubmodules(
            router=InferenceTopKRouter, experts=experts, shared_experts=shared_experts
        ),
    )
