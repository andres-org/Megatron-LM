# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import SonicMoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.transformer.moe.sonicmoe_util import sonicmoe_is_available

    HAVE_SONICMOE = sonicmoe_is_available()
except Exception:
    HAVE_SONICMOE = False

DEVICE_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _sonic_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_use_sonicmoe=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_sonicmoe_spec_selects_sonicmoe_layer():
    config = _sonic_config()
    submodules = get_gpt_layer_local_submodules(
        num_experts=config.num_moe_experts,
        moe_use_sonicmoe=config.moe_use_sonicmoe,
    )
    assert submodules.mlp.module is SonicMoELayer


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_SONICMOE,
    reason="CUDA or SonicMoE not available",
)
@pytest.mark.skipif(
    not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 9,
    reason="SonicMoE requires Hopper GPUs",
)
def test_sonicmoe_layer_constructor():
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        config = _sonic_config(use_cpu_initialization=False)
        layer = SonicMoELayer(config=config, layer_number=1).cuda()

        assert layer.num_experts == config.num_moe_experts
        assert layer.top_k == config.moe_router_topk
        assert layer.experts.weight1.shape == (
            2 * config.moe_ffn_hidden_size,
            config.hidden_size,
            layer.num_local_experts,
        )
        assert layer.experts.weight2.shape == (
            config.hidden_size,
            config.moe_ffn_hidden_size,
            layer.num_local_experts,
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_SONICMOE,
    reason="CUDA or SonicMoE not available",
)
@pytest.mark.skipif(
    not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 9,
    reason="SonicMoE requires Hopper GPUs",
)
def test_sonicmoe_layer_forward_backward_ep1():
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        config = _sonic_config(use_cpu_initialization=False)
        layer = SonicMoELayer(config=config, layer_number=1).cuda()
        hidden_states = torch.randn(
            16,
            2,
            config.hidden_size,
            device="cuda",
            dtype=config.params_dtype,
            requires_grad=True,
        )

        output, mlp_bias = layer(hidden_states)
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype
        assert mlp_bias is None

        output.float().sum().backward()
        assert hidden_states.grad is not None
        assert layer.router.weight.grad is not None
        assert layer.experts._weight1_storage.grad is not None
        assert layer.experts._weight2_storage.grad is not None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_SONICMOE,
    reason="CUDA or SonicMoE not available",
)
@pytest.mark.skipif(
    not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 9,
    reason="SonicMoE requires Hopper GPUs",
)
@pytest.mark.skipif(Utils.world_size < 2, reason="Requires torchrun with at least 2 ranks")
def test_sonicmoe_layer_forward_backward_flex_ep2():
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        config = _sonic_config(
            use_cpu_initialization=False,
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
        )
        layer = SonicMoELayer(config=config, layer_number=1).cuda()
        hidden_states = torch.randn(
            16,
            2,
            config.hidden_size,
            device="cuda",
            dtype=config.params_dtype,
            requires_grad=True,
        )

        output, mlp_bias = layer(hidden_states)
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype
        assert mlp_bias is None

        output.float().sum().backward()
        assert hidden_states.grad is not None
        assert layer.router.weight.grad is not None
        assert layer.experts._weight1_storage.grad is not None
        assert layer.experts._weight2_storage.grad is not None
    finally:
        Utils.destroy_model_parallel()
