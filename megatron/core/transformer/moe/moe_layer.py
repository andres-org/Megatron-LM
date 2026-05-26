# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Protocol, Tuple

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedTensorFactory,
)
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceCUDAGraphTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_object_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import internal_api

try:
    import flashinfer  # pylint: disable=unused-import

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

if HAVE_FLASHINFER:
    try:
        import flashinfer_cubin  # pylint: disable=unused-import
        import flashinfer_jit_cache  # pylint: disable=unused-import

        HAVE_FLASHINFER_CUBIN_AND_JIT_CACHE = True
    except ImportError:
        HAVE_FLASHINFER_CUBIN_AND_JIT_CACHE = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, te_checkpoint
else:
    TELinear, te_checkpoint = None, None


class _ConnectGradientWithZeros(torch.autograd.Function):
    """Connect a tensor to the graph with an explicit zero gradient."""

    @staticmethod
    def forward(ctx, output, tensor_to_connect):
        ctx.shape = tensor_to_connect.shape
        ctx.dtype = tensor_to_connect.dtype
        ctx.device = tensor_to_connect.device
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)


class ExpertsInterface(Protocol):
    """Interface for the experts used in an MoELayer."""

    def forward(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the experts layer."""
        ...

    def backward_dw(self) -> None:
        """Backward pass to compute weight gradients for the experts."""
        ...


class ExpertsBuilder(Protocol):
    """Protocol for building the experts used in an MoELayer."""

    def __call__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        /,
        *,
        pg_collection: ProcessGroupCollection | None,
    ) -> ExpertsInterface: ...


class SharedExpertsInterface(Protocol):
    """Interface for the shared experts used in an MoELayer."""

    def forward(self, hidden_states: torch.Tensor, /) -> torch.Tensor:
        """Forward pass of the shared experts."""
        ...

    def backward_dw(self) -> None:
        """Backward pass to compute weight gradients for the shared experts."""
        ...


class SharedExpertsBuilder(Protocol):
    """Protocol for building the shared experts used in an MoELayer."""

    def __call__(
        self, *, config: TransformerConfig, pg_collection: ProcessGroupCollection | None, gate: bool
    ) -> SharedExpertsInterface: ...


class RouterInterface(Protocol):
    """Interface for the router used in an MoELayer."""

    def forward(
        self,
        input: torch.Tensor,
        /,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the router.

        Returns:
            A tuple of (probabilities, routing_map).
        """
        ...

    def set_layer_number(self, layer_number: int) -> None:
        """Set the layer number for the router.

        Called from transformer_layer during initialization.
        """
        ...


class RouterBuilder(Protocol):
    """Protocol for building a Router."""

    def __call__(
        self, /, *, config: TransformerConfig, pg_collection: ProcessGroupCollection | None
    ) -> RouterInterface: ...


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: ExpertsBuilder
    shared_experts: SharedExpertsBuilder | None = None
    router: RouterBuilder = TopKRouter


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: RouterInterface = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        self.submodules = not_none(submodules)
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        # If using mcore cudagraphs, recompute is handled by transformer_layer.MoETransformerLayer
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective'
            and "moe" in config.recompute_modules
            and config.cuda_graph_impl != 'local'
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize router.
        self.router = self.submodules.router(
            config=self.config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer
        )
        self.tp_group = pg_collection.tp

        # Initialize latent projections.
        if self.config.moe_latent_size:
            assert HAVE_TE, "TransformerEngine is required for MoE latent projections."
            self.fc1_latent_proj = TELinear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )
            self.fc2_latent_proj = TELinear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = self.submodules.experts(
            self.num_local_experts, self.config, pg_collection=pg_collection
        )

        # Initialize shared experts
        if self.use_shared_expert:
            assert (
                self.submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Inference-optimized mode setup
        if config.transformer_impl == "inference_optimized":
            if config.inference_grouped_gemm_backend == 'auto':
                assert HAVE_FLASHINFER, (
                    "inference_grouped_gemm_backend='auto'"
                    "requires flashinfer-python. "
                    "Install flashinfer-python or set "
                    "inference_grouped_gemm_backend to 'torch' or 'te'."
                )

                # Verify that pre-compiled FlashInfer CUTLASS kernels are available
                # when using the FlashInfer backend. The flashinfer-jit-cache package
                # must be installed ahead of time to avoid a multi-minute JIT
                # compilation step at runtime.
                from megatron.core.inference.utils import check_flashinfer_jit_cache_installed

                check_flashinfer_jit_cache_installed()
            elif config.inference_grouped_gemm_backend == 'torch':
                assert hasattr(torch.nn.functional, 'grouped_mm'), (
                    "inference_grouped_gemm_backend='torch' requires "
                    "torch.nn.functional.grouped_mm (available since PyTorch 2.10)."
                )
            self._setup_inference_mode(pg_collection)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()
        self.fwd_execution_map = ["route", "expert_compute", "postprocess"]

    def _setup_inference_mode(self, pg_collection):
        """Set up inference-optimized token dispatcher and state.

        Called from __init__ when config.transformer_impl == "inference_optimized".
        Creates an InferenceCUDAGraphTokenDispatcher alongside the standard dispatcher,
        which is swapped in during CUDA-graphed forward passes.
        """

        assert self.config.moe_token_dispatcher_type == "alltoall", (
            f"Inference-optimized MoE requires 'alltoall' dispatcher, "
            f"got '{self.config.moe_token_dispatcher_type}'"
        )
        self.is_inference_cuda_graphed_iteration = False
        self._inference_token_dispatcher = InferenceCUDAGraphTokenDispatcher(
            self.num_local_experts,
            self.local_expert_indices,
            config=self.config,
            pg_collection=pg_collection,
        )

    def set_inference_cuda_graphed_iteration(self):
        """Enable CUDA-graphed iteration mode on this layer, its router, and its experts.

        Swaps in the inference-optimized token dispatcher and disables
        shared expert overlap.
        """
        self.is_inference_cuda_graphed_iteration = True
        if hasattr(self.router, "set_inference_cuda_graphed_iteration"):
            self.router.set_inference_cuda_graphed_iteration()
        if hasattr(self.experts, "set_inference_cuda_graphed_iteration"):
            self.experts.set_inference_cuda_graphed_iteration()

        if self._inference_token_dispatcher is not None:
            self._saved_token_dispatcher = self.token_dispatcher
            self.token_dispatcher = self._inference_token_dispatcher
            self._saved_shared_expert_overlap = self.shared_expert_overlap
            self.shared_expert_overlap = False

    def unset_inference_cuda_graphed_iteration(self):
        """Disable CUDA-graphed iteration mode on this layer, its router, and its experts.

        Restores the standard token dispatcher and shared expert overlap setting.
        """
        self.is_inference_cuda_graphed_iteration = False
        if hasattr(self.router, "unset_inference_cuda_graphed_iteration"):
            self.router.unset_inference_cuda_graphed_iteration()
        if hasattr(self.experts, "unset_inference_cuda_graphed_iteration"):
            self.experts.unset_inference_cuda_graphed_iteration()

        if hasattr(self, "_saved_token_dispatcher"):
            self.token_dispatcher = self._saved_token_dispatcher
            self.shared_expert_overlap = self._saved_shared_expert_overlap

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping.
        """
        probs, routing_map = apply_module(self.router)(hidden_states, padding_mask)
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher.
        """
        # Project the hidden_states from hidden dimension down to latent dimenion.
        if self.config.moe_latent_size:
            assert (
                not self.shared_expert_overlap
            ), "Shared expert overlap not supported when MoE latent projections are used."
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        apply_module(self.shared_experts),
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        apply_module(self.shared_experts), False, hidden_states
                    )
            else:
                shared_expert_output = apply_module(self.shared_experts)(hidden_states)

        return shared_expert_output

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        if (
            hasattr(self, "_inference_token_dispatcher")
            and self.is_inference_cuda_graphed_iteration
        ):
            routing_map = self.token_dispatcher.routing_map
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs, routing_map=routing_map
            )
        else:
            expert_output, mlp_bias = apply_module(self.experts)(
                dispatched_input, tokens_per_expert, permuted_probs
            )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication).
        """
        output = self.token_dispatcher.token_combine(output)
        return output

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Project the output back from latent dimension to hidden dimension after combine
        in latent dimension if needed. Combine expert output with shared_experts if needed."""

        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
        return hidden_states, probs, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.
        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        # Transpose from [bsz, seq_length] to [seq_length, bsz] to align with hidden_states
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states, intermediate_tensors=None, padding_mask=None):
            try:
                if "route" in self.fwd_execution_map:
                    shared_expert_output = self.shared_experts_compute(hidden_states)
                    probs, routing_map = self.route(hidden_states, padding_mask=padding_mask)
                    hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

                    if intermediate_tensors is not None:
                        return hidden_states, probs, shared_expert_output

            except MoECudaGraphPartialCaptureSignal as e:
                # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                # It means we should early-return from the MoE layer forward pass.
                # This happens when we are partially capturing the CUDA graph of the MoE layer,
                # like cuda_graph_scope=["moe_router", "moe_preprocess"].
                # We need to return the intermediate tensors as CUDA graph outputs.
                return e.get_early_return_outputs(hidden_states, shared_expert_output)

            if "expert_compute" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    hidden_states, probs = intermediate_tensors

                dispatched_input, probs = self.dispatch(hidden_states, probs)
                output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
                assert (
                    mlp_bias is None
                ), f"mlp_bias is not supported for {type(self.token_dispatcher)}"
                output = self.combine(output)

                if intermediate_tensors is not None:
                    return output, mlp_bias

            if "postprocess" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    output, shared_expert_output = intermediate_tensors

                output = self.postprocess(output, shared_expert_output)

                if intermediate_tensors is not None:
                    return output

            return output, mlp_bias

        if self.moe_layer_recompute and self.training:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    custom_forward,
                    False,
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                )
        else:
            outputs = custom_forward(hidden_states, intermediate_tensors, padding_mask)

        return outputs

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        """Compute weight gradients for experts and shared experts."""
        # TODO(Wohox): replace the "routed_experts" and "shared_experts" arguments with better
        # naming to better explain that they are actually from different fine-grained callables,
        # or use scanning to decide which backward_dw should be called.
        if routed_experts:
            self.experts.backward_dw()
            if self.config.moe_latent_size:
                # TODO(Wohox): fc2_latent_proj forward and backward are executed in comm stream,
                # so we execute its backward_dw in the comm stream too. But this may harm the
                # EP overlap performance. Better to check if there is a better way to handle this.
                from megatron.core.pipeline_parallel.utils import get_comm_stream

                comm_stream = get_comm_stream()
                with torch.cuda.stream(comm_stream):
                    self.fc2_latent_proj.backward_dw()
        if shared_experts:
            if self.use_shared_expert and not self.shared_expert_overlap:
                self.shared_experts.backward_dw()
            if self.config.moe_latent_size:
                self.fc1_latent_proj.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)


class _SonicMoEExpertCompute(MegatronModule):
    """Expert weights and SonicMoE expert compute used by SonicMoELayer."""

    def __init__(
        self,
        config: TransformerConfig,
        num_local_experts: int,
        num_global_experts: int,
        expert_parallel: bool,
        pg_collection: ProcessGroupCollection,
    ):
        super().__init__(config=config)

        from megatron.core.transformer.moe.sonicmoe_util import (
            assert_sonicmoe_is_available,
            get_sonicmoe_activation,
        )

        assert_sonicmoe_is_available()
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = not_none(config.moe_ffn_hidden_size)
        self.gated_linear_unit = config.gated_linear_unit
        self.activation_type = get_sonicmoe_activation(config)
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.dp_group = pg_collection.expt_dp
        self.num_global_experts = num_global_experts
        self._stream_id = None

        w1_out = self.ffn_hidden_size * 2 if config.gated_linear_unit else self.ffn_hidden_size
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        self._weight1_storage = torch.nn.Parameter(
            torch.empty(
                num_local_experts,
                w1_out,
                self.hidden_size,
                dtype=config.params_dtype,
                device=device,
            )
        )
        self._weight2_storage = torch.nn.Parameter(
            torch.empty(
                num_local_experts,
                self.hidden_size,
                self.ffn_hidden_size,
                dtype=config.params_dtype,
                device=device,
            )
        )
        setattr(self._weight1_storage, "allreduce", not expert_parallel)
        setattr(self._weight2_storage, "allreduce", not expert_parallel)

        self.bias1 = None
        self.bias2 = None
        if config.add_bias_linear:
            self.bias1 = torch.nn.Parameter(
                torch.empty(num_local_experts, w1_out, dtype=config.params_dtype, device=device)
            )
            self.bias2 = torch.nn.Parameter(
                torch.empty(
                    num_local_experts, self.hidden_size, dtype=config.params_dtype, device=device
                )
            )
            setattr(self.bias1, "allreduce", not expert_parallel)
            setattr(self.bias2, "allreduce", not expert_parallel)

        if config.perform_initialization:
            self.init_weights(config)

    @property
    def stream_id(self):
        if self._stream_id is None:
            self._stream_id = torch.cuda.current_stream().cuda_stream
        return self._stream_id

    @property
    def weight1(self):
        return self._weight1_storage.permute(1, 2, 0)

    @property
    def weight2(self):
        return self._weight2_storage.permute(1, 2, 0)

    def init_weights(self, config: TransformerConfig):
        config.init_method(self._weight1_storage)
        config.output_layer_init_method(self._weight2_storage)
        if self.bias1 is not None:
            self.bias1.data.zero_()
        if self.bias2 is not None:
            self.bias2.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_scores: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        from megatron.core.transformer.moe.sonicmoe_util import moe_general_routing_inputs

        if hidden_states.nelement() == 0:
            output = hidden_states
            output = _ConnectGradientWithZeros.apply(output, router_scores)
            output = _ConnectGradientWithZeros.apply(output, self._weight1_storage)
            output = _ConnectGradientWithZeros.apply(output, self._weight2_storage)
            if self.bias1 is not None:
                output = _ConnectGradientWithZeros.apply(output, self.bias1)
            if self.bias2 is not None:
                output = _ConnectGradientWithZeros.apply(output, self.bias2)
            return output

        output, _ = moe_general_routing_inputs(
            x=hidden_states,
            router_scores=router_scores.reshape(-1).float(),
            token_indices=token_indices.reshape(-1).to(torch.int32),
            expert_indices=expert_indices.reshape(-1).to(torch.int32),
            w1=self.weight1,
            b1=self.bias1,
            w2=self.weight2,
            b2=self.bias2,
            E=self.num_local_experts,
            stream_id=self.stream_id,
            activation_type=self.activation_type,
            is_inference_mode_enabled=not self.training,
        )
        return output

    def backward_dw(self):
        pass

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Build sharded state dict entries compatible with standard MoE checkpoints."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}

        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()
        assert (
            tp_size == 1
        ), f"SonicMoELayer does not support tensor parallelism, but tp_size={tp_size}"
        dp_rank = self.dp_group.rank()
        local_expert_indices_offset = ep_rank * self.num_local_experts

        prepend_axis_num = len(sharded_offsets)
        replica_id = (0, 0, dp_rank)

        def _break_into_individual_experts(
            experts_ten: torch.Tensor,
            key: str,
            tp_offset: Tuple[int, int, int],
            replica_id: ReplicaId,
        ):
            experts_state = []
            assert len(experts_ten) == self.num_local_experts, (
                experts_ten.shape,
                self.num_local_experts,
            )
            for local_expert_idx, expert_ten in enumerate(experts_ten):
                global_expert_idx = local_expert_indices_offset + local_expert_idx
                expert_key = key.replace(
                    f'{prefix}experts.', f'{prefix}experts.{global_expert_idx}.'
                )
                experts_state.append(
                    ShardedTensor.from_rank_offsets(
                        expert_key,
                        expert_ten.contiguous(),
                        *sharded_offsets,
                        tp_offset,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                )
            return experts_state

        def _split_glu_rows_for_checkpoint(t: torch.Tensor):
            return t[..., 0::2, :], t[..., 1::2, :]

        def _merge_glu_rows_from_checkpoint(w_rows: torch.Tensor, v_rows: torch.Tensor):
            merged = torch.empty(
                (*w_rows.shape[:-2], w_rows.shape[-2] + v_rows.shape[-2], w_rows.shape[-1]),
                dtype=w_rows.dtype,
                device=w_rows.device,
            )
            merged[..., 0::2, :] = w_rows
            merged[..., 1::2, :] = v_rows
            return merged

        def _split_glu_bias_for_checkpoint(t: torch.Tensor):
            return t[..., 0::2], t[..., 1::2]

        def _merge_glu_bias_from_checkpoint(w_bias: torch.Tensor, v_bias: torch.Tensor):
            merged = torch.empty(
                (*w_bias.shape[:-1], w_bias.shape[-1] + v_bias.shape[-1]),
                dtype=w_bias.dtype,
                device=w_bias.device,
            )
            merged[..., 0::2] = w_bias
            merged[..., 1::2] = v_bias
            return merged

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
            tp_axis: int,
            with_glu: bool,
        ):
            if tp_axis not in [0, 1]:
                raise ValueError("tp_axis should be 0 or 1.")

            if flattened_range is None:
                if with_glu:
                    assert tp_axis == 0, tp_axis
                    if singleton_local_shards:
                        w_tensor, v_tensor = _split_glu_rows_for_checkpoint(t)
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': {
                                'w': _break_into_individual_experts(
                                    w_tensor,
                                    f'{key}_w',
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                                'v': _break_into_individual_experts(
                                    v_tensor,
                                    f'{key}_v',
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                            },
                        }
                    else:
                        local_tensors = _split_glu_rows_for_checkpoint(t)
                        sub_states = [
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[0].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[1].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                        ]
                else:
                    if singleton_local_shards:
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': _break_into_individual_experts(
                                t, key, (prepend_axis_num + tp_axis, tp_rank, tp_size), replica_id
                            ),
                        }
                    else:
                        sub_states = ShardedTensor.from_rank_offsets(
                            key,
                            t.contiguous(),
                            *sharded_offsets,
                            (prepend_axis_num, ep_rank, ep_size),
                            (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        )
            return sub_states  # pylint: disable=possibly-used-before-assignment

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if isinstance(sub_state_dict, dict):
                assert sub_state_dict['singleton_local_shards']
                if with_glu:
                    assert isinstance(sub_state_dict['data'], dict)
                    sub_state_dict = _merge_glu_rows_from_checkpoint(
                        torch.stack(sub_state_dict['data']['w']),
                        torch.stack(sub_state_dict['data']['v']),
                    )
                else:
                    assert isinstance(sub_state_dict['data'], list)
                    sub_state_dict = torch.stack(sub_state_dict['data'])
            elif with_glu:
                sub_state_dict = _merge_glu_rows_from_checkpoint(*sub_state_dict)
            return sub_state_dict

        bias_specs = {
            'bias1': (f'{prefix}experts.linear_fc1.bias', 0, self.gated_linear_unit),
            'bias2': (f'{prefix}experts.linear_fc2.bias', None, False),
        }

        state_dict = self.state_dict(prefix='', keep_vars=True)
        for name, tensor in state_dict.items():
            if name == '_weight1_storage':
                tp_axis = 0
                with_glu = self.gated_linear_unit
                wkey = f'{prefix}experts.linear_fc1.weight'
            if name == '_weight2_storage':
                tp_axis = 1
                with_glu = False
                wkey = f'{prefix}experts.linear_fc2.weight'
            if name in ('_weight1_storage', '_weight2_storage'):
                sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                    wkey,
                    tensor,
                    partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                    partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                    tuple(copy.deepcopy(replica_id)),
                )
                continue

            if name in bias_specs:
                bias_ckpt_key, bias_tp_axis, bias_with_glu = bias_specs[name]

                def bias_build_fn(
                    key,
                    t,
                    replica_id,
                    flattened_range,
                    _bias_ckpt_key=bias_ckpt_key,
                    _bias_tp_axis=bias_tp_axis,
                    _bias_with_glu=bias_with_glu,
                ):
                    if singleton_local_shards:
                        if _bias_with_glu:
                            w_tensor, v_tensor = _split_glu_bias_for_checkpoint(t)
                            return {
                                'singleton_local_shards': LocalNonpersistentObject(True),
                                'data': {
                                    'w': _break_into_individual_experts(
                                        w_tensor,
                                        f'{_bias_ckpt_key}_w',
                                        (prepend_axis_num, tp_rank, tp_size),
                                        replica_id,
                                    ),
                                    'v': _break_into_individual_experts(
                                        v_tensor,
                                        f'{_bias_ckpt_key}_v',
                                        (prepend_axis_num, tp_rank, tp_size),
                                        replica_id,
                                    ),
                                },
                            }

                        tp_offset = (
                            (prepend_axis_num + _bias_tp_axis, tp_rank, tp_size)
                            if _bias_tp_axis is not None
                            else None
                        )
                        experts_state = []
                        for local_expert_idx in range(self.num_local_experts):
                            global_expert_idx = local_expert_indices_offset + local_expert_idx
                            expert_key = _bias_ckpt_key.replace(
                                f'{prefix}experts.',
                                f'{prefix}experts.{global_expert_idx}.',
                            )
                            rank_offsets = [*sharded_offsets]
                            if tp_offset is not None:
                                rank_offsets.append(tp_offset)
                            experts_state.append(
                                ShardedTensor.from_rank_offsets(
                                    expert_key,
                                    t[local_expert_idx].contiguous(),
                                    *rank_offsets,
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                )
                            )
                        return {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': experts_state,
                        }

                    if _bias_with_glu:
                        local_tensors = _split_glu_bias_for_checkpoint(t)
                        return [
                            ShardedTensor.from_rank_offsets(
                                _bias_ckpt_key,
                                local_tensors[0].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                            ShardedTensor.from_rank_offsets(
                                _bias_ckpt_key,
                                local_tensors[1].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                        ]

                    rank_offsets = [*sharded_offsets, (prepend_axis_num, ep_rank, ep_size)]
                    if _bias_tp_axis is not None:
                        rank_offsets.append(
                            (prepend_axis_num + 1 + _bias_tp_axis, tp_rank, tp_size)
                        )
                    return ShardedTensor.from_rank_offsets(
                        _bias_ckpt_key,
                        t,
                        *rank_offsets,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )

                def bias_merge_fn(sub_state_dict, _bias_with_glu=bias_with_glu):
                    if isinstance(sub_state_dict, dict):
                        assert sub_state_dict['singleton_local_shards']
                        if _bias_with_glu:
                            assert isinstance(sub_state_dict['data'], dict)
                            return _merge_glu_bias_from_checkpoint(
                                torch.stack(sub_state_dict['data']['w']),
                                torch.stack(sub_state_dict['data']['v']),
                            )
                        return torch.stack(sub_state_dict['data'])
                    if _bias_with_glu:
                        return _merge_glu_bias_from_checkpoint(*sub_state_dict)
                    return sub_state_dict

                sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                    bias_ckpt_key,
                    tensor,
                    bias_build_fn,
                    bias_merge_fn,
                    copy.deepcopy(replica_id),
                )

        extra_state_replica_id = (0, tp_rank, dp_rank)
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            if singleton_local_shards:
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, self.num_global_experts),
                )
            for mod in ['linear_fc1', 'linear_fc2']:
                if singleton_local_shards:
                    expert_key = f'{prefix}experts.{expert_global_idx}.{mod}._extra_state'
                else:
                    expert_key = f'{prefix}experts.{mod}._extra_state'
                sharded_state_dict[f'{prefix}expert{expert_global_idx}.{mod}._extra_state'] = (
                    make_sharded_object_for_checkpoint(
                        None, expert_key, expert_sharded_offsets, extra_state_replica_id
                    )
                )

        return sharded_state_dict


class SonicMoELayer(BaseMoELayer):
    """MoE layer using SonicMoE expert kernels with Megatron routing and flex dispatch."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        from megatron.core.transformer.moe.sonicmoe_util import assert_sonicmoe_is_available

        assert_sonicmoe_is_available()
        self.submodules = submodules
        self.tp_group = pg_collection.tp
        self.dp_group = pg_collection.expt_dp
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = not_none(config.moe_ffn_hidden_size)
        self.num_experts = not_none(config.num_moe_experts)
        self.top_k = config.moe_router_topk
        self.ep_size = utils.get_pg_size(self.ep_group)
        self.expert_parallel = self.ep_size > 1
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective'
            and "moe" in config.recompute_modules
            and config.cuda_graph_impl != 'local'
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.router = TopKRouter(
            config=self.config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer
        )
        self.experts = _SonicMoEExpertCompute(
            config=config,
            num_local_experts=self.num_local_experts,
            num_global_experts=self.num_experts,
            expert_parallel=self.expert_parallel,
            pg_collection=pg_collection,
        )

        self.token_dispatcher = None
        if self.ep_size > 1 or config.moe_token_dispatcher_type == "flex":
            if config.moe_token_dispatcher_type != "flex":
                raise ValueError("SonicMoELayer with EP>1 requires flex token dispatcher.")
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )

        if self.use_shared_expert:
            assert (
                self.submodules is not None and self.submodules.shared_experts is not None
            ), "Shared experts builder is not provided in the module spec."
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
            )

        def remove_extra_states_check(module, incompatible_keys):
            for key in list(incompatible_keys.unexpected_keys):
                if "_extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def _dense_to_token_expert_metadata(self, probs: torch.Tensor, routing_map: torch.Tensor):
        router_scores, expert_indices = torch.topk(probs, self.top_k, dim=-1)
        token_indices = torch.arange(probs.shape[0], device=probs.device).unsqueeze(1)
        token_indices = token_indices.expand_as(expert_indices)
        valid_mask = router_scores.reshape(-1) != 0
        return (
            router_scores.reshape(-1)[valid_mask],
            token_indices.reshape(-1)[valid_mask],
            expert_indices.reshape(-1)[valid_mask],
        )

    def route(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        return apply_module(self.router)(hidden_states, padding_mask)

    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        self.hidden_shape = hidden_states.shape
        self._routing_map = routing_map
        if self.token_dispatcher is None:
            return hidden_states.view(-1, self.hidden_size), probs, routing_map
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, routing_map

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        if self.token_dispatcher is None:
            return hidden_states, probs
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def routed_experts_compute(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
    ):
        if routing_map is None:
            routing_map = self._routing_map
        if self.token_dispatcher is None:
            router_scores, token_indices, expert_indices = self._dense_to_token_expert_metadata(
                probs, routing_map
            )
        else:
            hidden_states, metadata, _ = self.token_dispatcher.dispatch_postprocess(
                hidden_states, probs
            )
            router_scores = probs.reshape(-1)
            token_indices = metadata.token_indices.reshape(-1)
            expert_indices = metadata.expert_indices.reshape(-1)
            valid_mask = expert_indices >= 0
            router_scores = router_scores[valid_mask]
            token_indices = token_indices[valid_mask]
            expert_indices = expert_indices[valid_mask]

        output = apply_module(self.experts)(
            hidden_states, router_scores, token_indices, expert_indices
        )
        if self.token_dispatcher is not None:
            output = self.token_dispatcher.combine_preprocess(output)
        return output, None

    def combine(self, output: torch.Tensor):
        if self.token_dispatcher is None:
            return output
        return self.token_dispatcher.token_combine(output)

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        if self.token_dispatcher is not None:
            output = self.token_dispatcher.combine_postprocess(output)
        else:
            output = output.view(self.hidden_shape)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        if not self.use_shared_expert:
            return None
        if self.shared_experts_recompute:
            return tensor_parallel.checkpoint(apply_module(self.shared_experts), False, hidden_states)
        return apply_module(self.shared_experts)(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        def custom_forward(hidden_states, padding_mask=None):
            shared_expert_output = self.shared_experts_compute(hidden_states)
            probs, routing_map = self.route(hidden_states, padding_mask=padding_mask)
            hidden_states, probs, routing_map = self.preprocess(hidden_states, probs, routing_map)
            hidden_states, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(hidden_states, probs, routing_map)
            output = self.combine(output)
            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute and self.training:
            outputs = tensor_parallel.checkpoint(
                custom_forward, False, hidden_states, padding_mask
            )
        else:
            outputs = custom_forward(hidden_states, padding_mask)
        return outputs

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        if routed_experts:
            self.experts.backward_dw()
        if shared_experts and self.use_shared_expert:
            self.shared_experts.backward_dw()
