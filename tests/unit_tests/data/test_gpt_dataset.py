# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import random

import numpy
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import Split
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


def sample_N(dataset, N, randomize):
    if randomize:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(N)]
    else:
        indices = list(range(N))
    samples = [dataset[index]["tokens"].numpy() for index in indices]
    return samples


class _DeterministicLowLevelDataset:
    def __len__(self):
        return 1


class _DeterministicGPTDataset(GPTDataset):
    def __init__(self, sample, config):
        self._sample = numpy.array(sample, dtype=numpy.int64)
        super().__init__(
            _DeterministicLowLevelDataset(),
            None,
            numpy.array([0], dtype=numpy.int64),
            1,
            Split.train,
            config,
        )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset):
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        raise NotImplementedError

    def _build_document_sample_shuffle_indices(self):
        return (
            numpy.array([0], dtype=numpy.int64),
            numpy.array([[0, 0], [0, 0]], dtype=numpy.int64),
            numpy.array([0], dtype=numpy.uint32),
        )

    def _query_document_sample_shuffle_indices(self, idx):
        return self._sample.copy(), numpy.array([0], dtype=numpy.int64)


def test_mock_gpt_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, config
    ).build()

    N = 10

    # Check iso-index variance by split
    subsets = [sample_N(dataset, N, randomize=False) for dataset in datasets]
    assert not numpy.allclose(subsets[0], subsets[1])
    assert not numpy.allclose(subsets[0], subsets[2])
    assert not numpy.allclose(subsets[1], subsets[2])

    # Check iso-split / iso-index identity
    subset_1A = sample_N(datasets[0], N, randomize=False)
    subset_1B = sample_N(datasets[0], N, randomize=False)
    assert numpy.allclose(subset_1A, subset_1B)

    # Check iso-split variance by index
    subset_1A = sample_N(datasets[0], N, randomize=True)
    subset_1B = sample_N(datasets[0], N, randomize=True)
    assert not numpy.allclose(subset_1A, subset_1B)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,10,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        drop_last_partial_validation_sequence=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [0, None, 0], lambda: True, config
    ).build()

    sample = datasets[1][datasets[1].shuffle_index.argmax()]
    argmax = sample['labels'].shape[0] - torch.flip(sample['labels'], [0]).argmax() - 1

    # Test add_extra_token_to_sequence
    assert sample['tokens'][argmax] != tokenizer.eod
    assert sample['labels'][argmax] == tokenizer.eod

    # Test eod_mask_loss, drop_last_partial_validation_sequence
    assert argmax < sample['labels'].shape[0] - 1
    assert torch.all(sample['labels'][argmax + 1 :] == 0)
    assert not torch.any(
        sample['loss_mask'][
            torch.logical_and(sample['labels'] == tokenizer.eod, sample['labels'] == 0)
        ]
    )

    sample = datasets[1][None]

    # Check handling of None index
    assert not torch.any(sample['loss_mask'])


@pytest.mark.parametrize("create_attention_mask", [True, False])
def test_gpt_dataset_cu_seqlens_from_eod_boundaries(create_attention_mask):
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=8,
        split="1,0,0",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=create_attention_mask,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
    )

    sample = [11, tokenizer.eod, 21, 22, tokenizer.eod, 31, 32, 33]
    dataset = _DeterministicGPTDataset(sample, config)

    item = dataset[0]

    assert torch.equal(item["cu_seqlens"], torch.tensor([0, 2, 5, 8], dtype=torch.int32))
    assert item["max_seqlen"].dtype == torch.int32
    assert item["max_seqlen"].item() == 3
    if create_attention_mask:
        assert "attention_mask" in item
    else:
        assert "attention_mask" not in item


def test_gpt_dataset_cu_seqlens_without_eod():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=6,
        split="1,0,0",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
    )

    sample = [11, 12, 13, 14, 15, 16]
    dataset = _DeterministicGPTDataset(sample, config)

    item = dataset[0]

    assert torch.equal(item["cu_seqlens"], torch.tensor([0, 6], dtype=torch.int32))
    assert item["max_seqlen"].item() == 6


if __name__ == "__main__":
    test_mock_gpt_dataset()
