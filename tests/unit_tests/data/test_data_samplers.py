# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.training.datasets.data_samplers import _packed_collate_fn, build_pretraining_data_loader


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def test_packed_collate_fn_merges_cu_seqlens_across_samples():
    batch = [
        {
            "tokens": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            "labels": torch.tensor([11, 12, 13, 14], dtype=torch.int64),
            "loss_mask": torch.ones(4, dtype=torch.float32),
            "position_ids": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            "cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32),
            "max_seqlen": torch.tensor(2, dtype=torch.int32),
        },
        {
            "tokens": torch.tensor([20, 21, 22, 23], dtype=torch.int64),
            "labels": torch.tensor([21, 22, 23, 24], dtype=torch.int64),
            "loss_mask": torch.ones(4, dtype=torch.float32),
            "position_ids": torch.tensor([0, 1, 2, 0], dtype=torch.int64),
            "cu_seqlens": torch.tensor([0, 3, 4], dtype=torch.int32),
            "max_seqlen": torch.tensor(3, dtype=torch.int32),
        },
    ]

    result = _packed_collate_fn(batch)

    assert torch.equal(
        result["tokens"], torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]], dtype=torch.int64)
    )
    assert torch.equal(
        result["labels"], torch.tensor([[11, 12, 13, 14, 21, 22, 23, 24]], dtype=torch.int64)
    )
    assert torch.equal(
        result["loss_mask"],
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32),
    )
    assert torch.equal(
        result["position_ids"], torch.tensor([[0, 1, 0, 1, 0, 1, 2, 0]], dtype=torch.int64)
    )
    assert torch.equal(
        result["cu_seqlens"], torch.tensor([[0, 2, 4, 7, 8]], dtype=torch.int32)
    )
    assert torch.equal(result["max_seqlen"], torch.tensor([3], dtype=torch.int32))
    assert torch.equal(result["seq_length"], torch.tensor([4], dtype=torch.int32))


def test_packed_collate_fn_rejects_attention_mask():
    batch = [
        {
            "tokens": torch.tensor([1, 2], dtype=torch.int64),
            "labels": torch.tensor([2, 3], dtype=torch.int64),
            "loss_mask": torch.ones(2, dtype=torch.float32),
            "position_ids": torch.tensor([0, 1], dtype=torch.int64),
            "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
            "max_seqlen": torch.tensor(2, dtype=torch.int32),
            "attention_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        }
    ]

    try:
        _packed_collate_fn(batch)
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as exc:
        assert "attention_mask" in str(exc)


def test_packed_collate_fn_falls_back_to_default_collate_without_packing_keys():
    batch = [
        {"tokens": torch.tensor([1, 2], dtype=torch.int64)},
        {"tokens": torch.tensor([3, 4], dtype=torch.int64)},
    ]

    result = _packed_collate_fn(batch)

    assert torch.equal(result["tokens"], torch.tensor([[1, 2], [3, 4]], dtype=torch.int64))


def test_build_pretraining_data_loader_uses_packed_collate_when_dataset_emits_packing_keys():
    dataset = _TinyDataset(
        [
            {
                "tokens": torch.tensor([1, 2], dtype=torch.int64),
                "labels": torch.tensor([2, 3], dtype=torch.int64),
                "loss_mask": torch.ones(2, dtype=torch.float32),
                "position_ids": torch.tensor([0, 1], dtype=torch.int64),
                "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
                "max_seqlen": torch.tensor(2, dtype=torch.int32),
            }
        ]
    )
    args = SimpleNamespace(
        full_validation=False,
        dataloader_type="single",
        hybrid_context_parallel=False,
        micro_batch_size=1,
        global_batch_size=1,
        num_workers=0,
        exit_signal_handler=False,
        sft=False,
    )

    with (
        mock.patch("megatron.training.datasets.data_samplers.get_args", new=lambda: args),
        mock.patch(
            "megatron.training.datasets.data_samplers.mpu.get_data_parallel_rank", new=lambda: 0
        ),
        mock.patch(
            "megatron.training.datasets.data_samplers.mpu.get_data_parallel_world_size",
            new=lambda: 1,
        ),
    ):
        dataloader = build_pretraining_data_loader(dataset, consumed_samples=0)

    assert dataloader.collate_fn is _packed_collate_fn


def test_build_pretraining_data_loader_uses_default_collate_without_packing_keys():
    dataset = _TinyDataset(
        [
            {
                "tokens": torch.tensor([1, 2], dtype=torch.int64),
                "labels": torch.tensor([2, 3], dtype=torch.int64),
                "loss_mask": torch.ones(2, dtype=torch.float32),
                "position_ids": torch.tensor([0, 1], dtype=torch.int64),
            }
        ]
    )
    args = SimpleNamespace(
        full_validation=False,
        dataloader_type="single",
        hybrid_context_parallel=False,
        micro_batch_size=1,
        global_batch_size=1,
        num_workers=0,
        exit_signal_handler=False,
        sft=True,
    )

    with (
        mock.patch("megatron.training.datasets.data_samplers.get_args", new=lambda: args),
        mock.patch(
            "megatron.training.datasets.data_samplers.mpu.get_data_parallel_rank", new=lambda: 0
        ),
        mock.patch(
            "megatron.training.datasets.data_samplers.mpu.get_data_parallel_world_size",
            new=lambda: 1,
        ),
    ):
        dataloader = build_pretraining_data_loader(dataset, consumed_samples=0)

    assert dataloader.collate_fn is torch.utils.data.dataloader.default_collate
