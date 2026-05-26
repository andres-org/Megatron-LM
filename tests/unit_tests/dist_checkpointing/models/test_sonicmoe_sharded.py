# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.optim import Adam

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    optim_state_to_sharding_state,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.transformer.moe.sonicmoe_util import sonicmoe_is_available

    HAVE_SONICMOE = sonicmoe_is_available()
except Exception:
    HAVE_SONICMOE = False


def _default_config_kwargs(glu=True, sonic=False, **overrides):
    kwargs = dict(
        num_layers=parallel_state.get_pipeline_model_parallel_world_size(),
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        use_cpu_initialization=True,
        add_bias_linear=False,
        gated_linear_unit=glu,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_ffn_hidden_size=32,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
    )
    if sonic:
        kwargs["moe_use_sonicmoe"] = True
        if parallel_state.get_expert_model_parallel_world_size() > 1:
            kwargs.update(
                moe_token_dispatcher_type="flex",
                moe_flex_dispatcher_backend="deepep",
            )
    kwargs.update(overrides)
    return kwargs


def initialize_sonicmoe_layer(seed, glu=True, **config_kwargs):
    from megatron.core.transformer.moe.moe_layer import SonicMoELayer

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(**_default_config_kwargs(glu=glu, sonic=True, **config_kwargs))
    return SonicMoELayer(config=config, submodules=None, layer_number=0)


def initialize_sequential_moe_layer(seed, glu=True, **config_kwargs):
    from megatron.core.transformer.moe.moe_layer import MoELayer

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config_kwargs = _default_config_kwargs(glu=glu, sonic=False, **config_kwargs)
    config = TransformerConfig(**config_kwargs)
    layer_spec = get_gpt_layer_local_spec(
        num_experts=config_kwargs["num_moe_experts"], moe_grouped_gemm=False
    )
    return MoELayer(config=config, submodules=layer_spec.submodules.mlp.submodules, layer_number=0)


@pytest.mark.skipif(not HAVE_SONICMOE, reason="SonicMoE is not available.")
class TestSonicMoEShardedStateDict:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("src_ep,dest_ep", [(1, 1), (1, 2), (2, 1)])
    @pytest.mark.parametrize("singleton_local_shards", [False, True])
    @pytest.mark.parametrize("use_glu", [False, True])
    @pytest.mark.parametrize("add_bias_linear", [False, True])
    def test_sonicmoe_sharded_reconfiguration_roundtrip(
        self,
        tmp_path_dist_ckpt,
        src_ep,
        dest_ep,
        singleton_local_shards,
        use_glu,
        add_bias_linear,
    ):
        if Utils.world_size < max(src_ep, dest_ep):
            pytest.skip(f"World size {Utils.world_size} < required {max(src_ep, dest_ep)}")

        metadata = {"singleton_local_shards": singleton_local_shards}
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=src_ep)
        with (
            TempNamedDir(tmp_path_dist_ckpt / "sonicmoe_sharded_a") as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "sonicmoe_sharded_b") as ckpt_b,
        ):
            layer_prefix = f"{parallel_state.get_pipeline_model_parallel_rank()}."
            model_a = initialize_sonicmoe_layer(1, use_glu, add_bias_linear=add_bias_linear)
            save(model_a.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_a)
            Utils.destroy_model_parallel()

            metadata.pop("dp_cp_group", None)
            Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=dest_ep)
            model_b = initialize_sonicmoe_layer(1, use_glu, add_bias_linear=add_bias_linear)
            state_dict = load(
                model_b.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_a
            )
            model_b.load_state_dict({k.removeprefix(layer_prefix): v for k, v in state_dict.items()})
            save(model_b.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_b)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(1, 1)
            diffs = diff(load_plain_tensors(ckpt_a), load_plain_tensors(ckpt_b))
            assert not any(map(bool, diffs)), diffs

    @pytest.mark.parametrize("ep_size,use_glu", [(1, True), (2, True)])
    def test_sonicmoe_optimizer_state_sharded_roundtrip(
        self, tmp_path_dist_ckpt, ep_size, use_glu
    ):
        if Utils.world_size < ep_size:
            pytest.skip(f"World size {Utils.world_size} < required {ep_size}")

        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=ep_size)
        with TempNamedDir(tmp_path_dist_ckpt / "sonicmoe_optimizer_state") as ckpt_dir:
            model = initialize_sonicmoe_layer(1, use_glu)
            for param in model.parameters():
                param.grad = torch.ones_like(param.data)
            optim = Adam(model.parameters())
            optim.step()
            optim_sd_plain = deepcopy(optim.state_dict())

            layer_prefix = f"{parallel_state.get_pipeline_model_parallel_rank()}."
            model_sharded_sd = model.sharded_state_dict(prefix=layer_prefix, metadata={})
            param_map = get_param_id_to_sharded_param_map(
                model_sharded_sd, optim.param_groups[0]["params"]
            )
            optim_sd = optim.state_dict()
            optim_state_to_sharding_state(optim_sd, param_map, exclude_keys=("step",))
            save(optim_sd, ckpt_dir)

            model_b = initialize_sonicmoe_layer(1, use_glu)
            for param in model_b.parameters():
                param.grad = torch.ones_like(param.data)
            optim_b = Adam(model_b.parameters())
            optim_b.step()
            model_sharded_sd_b = model_b.sharded_state_dict(prefix=layer_prefix, metadata={})
            param_map_b = get_param_id_to_sharded_param_map(
                model_sharded_sd_b, optim_b.param_groups[0]["params"]
            )
            optim_sd_b = optim_b.state_dict()
            optim_state_to_sharding_state(optim_sd_b, param_map_b, exclude_keys=("step",))

            loaded_optim_sd = load(optim_sd_b, ckpt_dir)
            assert loaded_optim_sd["param_groups"] == optim_sd_plain["param_groups"]
            assert set(loaded_optim_sd["state"]) == set(optim_sd_plain["state"])
            for pid, src_state in optim_sd_plain["state"].items():
                assert torch.equal(loaded_optim_sd["state"][pid]["exp_avg"], src_state["exp_avg"])
                assert torch.equal(
                    loaded_optim_sd["state"][pid]["exp_avg_sq"], src_state["exp_avg_sq"]
                )

    @pytest.mark.parametrize("src_type,dest_type", [("sonic", "sequential"), ("sequential", "sonic")])
    @pytest.mark.parametrize("singleton_local_shards", [False, True])
    def test_sequential_sonicmoe_sharded_interchangeable(
        self, tmp_path_dist_ckpt, src_type, dest_type, singleton_local_shards
    ):
        metadata = {"singleton_local_shards": singleton_local_shards}

        def init_layer(layer_type):
            if layer_type == "sonic":
                return initialize_sonicmoe_layer(1, glu=True)
            return initialize_sequential_moe_layer(1, glu=True)

        Utils.initialize_model_parallel(1, 1)
        with (
            TempNamedDir(tmp_path_dist_ckpt / "sonicmoe_interchange_a") as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "sonicmoe_interchange_b") as ckpt_b,
        ):
            layer_prefix = f"{parallel_state.get_pipeline_model_parallel_rank()}."
            model_a = init_layer(src_type)
            save(model_a.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_a)
            Utils.destroy_model_parallel()

            metadata.pop("dp_cp_group", None)
            Utils.initialize_model_parallel(1, 1)
            model_b = init_layer(dest_type)
            state_dict = load(
                model_b.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_a
            )
            model_b.load_state_dict({k.removeprefix(layer_prefix): v for k, v in state_dict.items()})
            save(model_b.sharded_state_dict(prefix=layer_prefix, metadata=metadata), ckpt_b)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(1, 1)
            diffs = diff(load_plain_tensors(ckpt_a), load_plain_tensors(ckpt_b))
            assert not any(map(bool, diffs)), diffs
