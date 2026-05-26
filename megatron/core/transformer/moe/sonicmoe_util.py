# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""SonicMoE integration utilities."""

import torch.nn.functional as F

try:
    import sonicmoe
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import (
        moe_general_routing_inputs,
        moe_sorted_by_experts_input,
    )

    _SONICMOE_AVAILABLE = True
except ImportError:
    sonicmoe = None
    ActivationType = None
    moe_general_routing_inputs = None
    moe_sorted_by_experts_input = None
    _SONICMOE_AVAILABLE = False


def sonicmoe_is_available():
    """Return whether SonicMoE can be imported."""
    return _SONICMOE_AVAILABLE


def assert_sonicmoe_is_available():
    """Assert that SonicMoE can be imported."""
    assert sonicmoe_is_available(), (
        "SonicMoE is not available. Install sonic-moe and make sure it is on PYTHONPATH."
    )


def get_sonicmoe_activation(config):
    """Map Megatron activation settings to SonicMoE's activation enum."""
    assert_sonicmoe_is_available()

    if config.gated_linear_unit:
        if config.activation_func == F.silu:
            return ActivationType.SWIGLU
        if config.activation_func == F.gelu:
            return ActivationType.GEGLU
        if config.activation_func == F.relu:
            return ActivationType.REGLU
        raise ValueError(
            f"Unsupported GLU activation function for SonicMoE: {config.activation_func}"
        )

    if config.activation_func == F.relu:
        return ActivationType.RELU
    if config.activation_func == F.gelu:
        return ActivationType.GELU
    if config.activation_func == F.silu:
        return ActivationType.SILU
    raise ValueError(f"Unsupported activation function for SonicMoE: {config.activation_func}")
