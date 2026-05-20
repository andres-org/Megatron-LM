# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from copy import deepcopy

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def make_test_packed_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


def make_test_packed_padded_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 18, 44, 52, 96, 118]).cuda()
    cu_seqlens_padded = torch.IntTensor([0, 20, 48, 56, 100, sequence_length]).cuda()
    seqlens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


class TestParallelAttentionWithPackedSequence:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        # use BF16 and a large enough hidden size to enable FlashAttention for thd format.
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
        )
        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward_thd_multi_batch(self):
        """[sq, b, h] input with b>1 and merged THD packed params: output shape matches input."""
        self.parallel_attention.cuda()
        config = self.parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        hidden_states = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )

        # cu_seqlens must cover the full merged stream of sq*b tokens.
        single_cu = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
        merged_cu = torch.cat([single_cu, single_cu[1:] + sequence_length])
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=merged_cu,
            cu_seqlens_kv=merged_cu,
            max_seqlen_q=13,
            max_seqlen_kv=13,
            qkv_format='thd',
        )
        output, bias = self.parallel_attention(
            hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )

        assert output.shape == (sequence_length, micro_batch_size, config.hidden_size)
        assert bias.shape == (config.hidden_size,)

    def test_gpu_forward_thd_multi_batch_matches_single_batch(self):
        """b>1 THD forward matches manually pre-transposed b=1 input with the same merged cu_seqlens."""

        deterministic_config = deepcopy(self.transformer_config)
        deterministic_config.attention_dropout = 0.0
        self.parallel_attention = SelfAttention(
            deterministic_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        self.parallel_attention.cuda()
        config = self.parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        torch.manual_seed(42)
        hidden_states = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size,
            dtype=torch.bfloat16, device='cuda',
        )

        # Build merged cu_seqlens covering sq*b tokens (same as _packed_collate_fn produces).
        single_cu = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
        merged_cu = torch.cat([single_cu, single_cu[1:] + sequence_length])
        merged_params = PackedSeqParams(
            cu_seqlens_q=merged_cu,
            cu_seqlens_kv=merged_cu,
            max_seqlen_q=torch.diff(merged_cu).max().item(),
            max_seqlen_kv=torch.diff(merged_cu).max().item(),
            qkv_format='thd',
        )

        # b>1 forward — internally transposes [sq, b, h] → [sq*b, 1, h] then inverts
        output_multi, _ = self.parallel_attention(
            hidden_states, attention_mask=None, packed_seq_params=merged_params
        )

        # Reference: manually do the same transpose+view and run with b=1
        hidden_states_thd = hidden_states.transpose(0, 1).contiguous().view(
            sequence_length * micro_batch_size, 1, config.hidden_size
        ) # (b, sq, h) -> (b*sq, 1, h)
        output_ref_thd, _ = self.parallel_attention(
            hidden_states_thd, attention_mask=None, packed_seq_params=merged_params
        )
        output_ref = output_ref_thd.view(
            micro_batch_size, sequence_length, config.hidden_size
        ).transpose(0, 1).contiguous()

        torch.testing.assert_close(output_multi, output_ref, atol=1e-3, rtol=1e-3)

    def test_gpu_forward_thd_no_cross_doc_attention(self):
        """Changing an earlier packed document must not affect later packed documents."""

        deterministic_config = deepcopy(self.transformer_config)
        deterministic_config.attention_dropout = 0.0
        self.parallel_attention = SelfAttention(
            deterministic_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        self.parallel_attention.cuda()
        self.parallel_attention.eval()
        config = self.parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2
        first_doc_end = 6

        torch.manual_seed(1234)
        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            config.hidden_size,
            dtype=torch.bfloat16,
            device='cuda',
        )

        single_cu = torch.IntTensor([0, first_doc_end, 19, 22, sequence_length]).cuda()
        merged_cu = torch.cat([single_cu, single_cu[1:] + sequence_length])
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=merged_cu,
            cu_seqlens_kv=merged_cu,
            max_seqlen_q=torch.diff(merged_cu).max().item(),
            max_seqlen_kv=torch.diff(merged_cu).max().item(),
            qkv_format='thd',
        )

        with torch.no_grad():
            output_before, _ = self.parallel_attention(
                hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
            )

            mutated_hidden_states = hidden_states.clone()
            mutated_hidden_states[:first_doc_end, 0, :] = torch.randn_like(
                mutated_hidden_states[:first_doc_end, 0, :]
            ) * 10
            output_after, _ = self.parallel_attention(
                mutated_hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
            )

        # The mutated document itself should change, otherwise the test would be vacuous.
        assert not torch.allclose(
            output_before[:first_doc_end, 0, :], output_after[:first_doc_end, 0, :]
        )

        # Later documents in the same sample must not attend to the mutated earlier document.
        torch.testing.assert_close(
            output_before[first_doc_end:, 0, :],
            output_after[first_doc_end:, 0, :],
            atol=1e-3,
            rtol=1e-3,
        )

        # Other batch elements must also remain independent after the THD flatten/unflatten path.
        torch.testing.assert_close(
            output_before[:, 1, :], output_after[:, 1, :], atol=1e-3, rtol=1e-3
        )

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.skipif(not is_te_min_version("1.4.0"), reason="Fused RoPE requires TE >= 1.4.0")
    def test_fused_rope_gpu_forward(self):
        self.parallel_attention.config.apply_rope_fusion = True
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False

    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 1

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = checkpointed_parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


# Note: this test requires TE >= 1.8 as well as cuDNN FusedAttention to run
class TestParallelAttentionWithPackedPaddedSequence(TestParallelAttentionWithPackedSequence):

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 128
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_padded_seq_params(sequence_length)
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
