# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare GPT dataset caches ahead of training.

Unsupported configurations:
    --mock-data, --sft, --fim-data, --step-batch-size-schedule
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import torch.distributed as dist

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import get_train_valid_test_num_samples
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_args, unset_global_variables
from megatron.training.training import update_train_iters
from megatron.training.utils import get_blend_and_blend_per_split

try:
    from megatron.post_training.arguments import add_modelopt_args

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False


def add_prepare_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add cache-preparation specific arguments."""

    group = parser.add_argument_group(title="prepare cache")
    group.add_argument(
        "--prepare-cache-world-size",
        type=int,
        default=None,
        help=(
            "Optional override for the effective world size used to derive data-parallel size and "
            "dataset sample counts during cache preparation."
        ),
    )

    group.add_argument(
        "--prepare-cache-sample-map-path",
        type=str,
        default=None,
        help=(
            "Optional path to write a JSONL file with one record per global batch/step. Each "
            "record includes that batch's samples and their source dataset prefix, .bin/.idx "
            "paths, blended dataset id, local sample index, shuffled sample index, source "
            "sequence ids, token offsets, and byte offsets into the .bin file."
        ),
    )
    return parser


def _extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_prepare_cache_args(parser)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    return parser


def _normalize_prepare_cache_args(args: Any) -> None:
    """Apply cache-preparation specific argument normalization."""

    args.rank = 0

    if args.prepare_cache_world_size is not None:
        if args.prepare_cache_world_size <= 0:
            raise ValueError("--prepare-cache-world-size must be positive")
        args.world_size = args.prepare_cache_world_size


def _validate_prepare_cache_args(args: Any) -> None:
    """Validate options that are intentionally unsupported for offline cache prep."""

    if args.data_cache_path is None:
        raise ValueError("--data-cache-path must be provided for cache preparation")
    if args.mock_data:
        raise ValueError("--mock-data is not supported by tools/prepare_cache.py")
    if getattr(args, "sft", False):
        raise ValueError("--sft is not supported by tools/prepare_cache.py")
    if getattr(args, "fim_data", False):
        raise ValueError("--fim-data is not supported by tools/prepare_cache.py")
    if getattr(args, "step_batch_size_schedule", None) is not None:
        raise ValueError(
            "--step-batch-size-schedule is not supported by tools/prepare_cache.py"
        )


def _disable_cache_load_only_flags(args: Any) -> Dict[str, bool]:
    """Disable flags that only make sense when consuming an existing cache."""

    ignored = {
        "dataloader_fast_cache_load": bool(args.dataloader_fast_cache_load),
        "dataloader_defer_npy_index_mmap": bool(args.dataloader_defer_npy_index_mmap),
    }
    args.dataloader_fast_cache_load = False
    args.dataloader_defer_npy_index_mmap = False
    return ignored


def _get_dataset_length(dataset: Optional[Any]) -> Optional[Any]:
    if dataset is None:
        return None
    if isinstance(dataset, list):
        return [len(ds) if ds is not None else None for ds in dataset]
    return len(dataset)


def _get_gpt_sample_source_parts(
    dataset: GPTDataset, local_sample_idx: int
) -> Tuple[int, List[Dict[str, Any]]]:
    """Map a GPTDataset sample index to its source sequences and .bin byte offsets."""
    if dataset.shuffle_index is None:
        dataset.shuffle_index = np.load(
            dataset.path_to_shuffle_index, allow_pickle=True, mmap_mode="r"
        )
        dataset.sample_index = np.load(
            dataset.path_to_sample_index, allow_pickle=True, mmap_mode="r"
        )
        dataset.document_index = np.load(
            dataset.path_to_document_index, allow_pickle=True, mmap_mode="r"
        )

    shuffled_sample_idx = int(dataset.shuffle_index[local_sample_idx])
    doc_index_beg, doc_index_beg_offset = dataset.sample_index[shuffled_sample_idx]
    doc_index_end, doc_index_end_offset = dataset.sample_index[shuffled_sample_idx + 1]

    source_parts: List[Dict[str, Any]] = []
    low_level_dataset = dataset.dataset
    dtype_size = int(low_level_dataset.index.dtype_size)

    for doc_index in range(int(doc_index_beg), int(doc_index_end) + 1):
        sequence_id = int(dataset.document_index[doc_index])
        token_offset = int(doc_index_beg_offset) if doc_index == int(doc_index_beg) else 0
        if doc_index == int(doc_index_end):
            token_length = (
                int(doc_index_end_offset)
                - token_offset
                + int(dataset.config.add_extra_token_to_sequence)
            )
        else:
            token_length = int(low_level_dataset.sequence_lengths[sequence_id]) - token_offset
        sequence_pointer = int(low_level_dataset.index.sequence_pointers[sequence_id])
        byte_offset = sequence_pointer + token_offset * dtype_size

        source_parts.append(
            {
                "sequence_id": sequence_id,
                "document_index_position": doc_index,
                "token_offset": token_offset,
                "token_length": token_length,
                "byte_offset": byte_offset,
                "byte_length": token_length * dtype_size,
                "sequence_pointer": sequence_pointer,
                "sequence_length": int(low_level_dataset.sequence_lengths[sequence_id]),
            }
        )

    return shuffled_sample_idx, source_parts


def _sample_source_record(dataset: Any, split_name: str, sample_idx: int) -> Dict[str, Any]:
    """Build one JSON-safe source mapping record for a top-level dataset sample."""
    blended_dataset_id = None
    blended_sample_idx = None
    source_dataset = dataset
    local_sample_idx = sample_idx

    if isinstance(dataset, BlendedDataset):
        if dataset.dataset_index is None:
            dataset.dataset_index = np.load(
                dataset.path_to_dataset_index, allow_pickle=True, mmap_mode="r"
            )
            dataset.dataset_sample_index = np.load(
                dataset.path_to_dataset_sample_index, allow_pickle=True, mmap_mode="r"
            )
        blended_dataset_id = int(dataset.dataset_index[sample_idx])
        blended_sample_idx = int(dataset.dataset_sample_index[sample_idx])
        source_dataset = dataset.datasets[blended_dataset_id]
        local_sample_idx = blended_sample_idx

    shuffled_sample_idx, source_parts = _get_gpt_sample_source_parts(
        source_dataset, local_sample_idx
    )
    dataset_path = source_dataset.dataset_path

    return {
        "split": split_name,
        "sample_idx": sample_idx,
        "dataset_id": blended_dataset_id,
        "dataset_sample_idx": blended_sample_idx,
        "local_sample_idx": int(local_sample_idx),
        "shuffled_sample_idx": shuffled_sample_idx,
        "dataset_path": dataset_path,
        "bin_path": None if dataset_path is None else f"{dataset_path}.bin",
        "idx_path": None if dataset_path is None else f"{dataset_path}.idx",
        "source_parts": source_parts,
    }


def _write_dataset_batch_map_records(writer: Any, dataset: Any, split_name: str, batch_size: int) -> int:
    """Write JSONL records that map each global batch to source samples."""
    if dataset is None:
        return 0
    if isinstance(dataset, list):
        total = 0
        for i, child in enumerate(dataset):
            total += _write_dataset_batch_map_records(writer, child, f"{split_name}_{i}", batch_size)
        return total

    num_batches = 0
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        record = {
            "split": split_name,
            "batch_idx": num_batches,
            "global_batch_size": batch_size,
            "sample_idx_begin": batch_start,
            "sample_idx_end_exclusive": batch_end,
            "samples": [
                _sample_source_record(dataset, split_name, sample_idx)
                for sample_idx in range(batch_start, batch_end)
            ],
        }
        writer.write(json.dumps(record) + "\n")
        num_batches += 1
    return num_batches


def _write_prepare_cache_batch_map(
    path: str,
    args: Any,
    train_ds: Optional[Any],
    valid_ds: Optional[Any],
    test_ds: Optional[Any],
) -> None:
    """Write a JSONL file mapping each global batch to source data locations."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as writer:
        train_batches = _write_dataset_batch_map_records(
            writer, train_ds, "train", args.global_batch_size
        )
        valid_batches = _write_dataset_batch_map_records(
            writer, valid_ds, "validation", args.global_batch_size
        )
        test_batches = _write_dataset_batch_map_records(
            writer, test_ds, "test", args.global_batch_size
        )
    print(
        f"> wrote dataset batch source map to {path} "
        f"(train batches: {train_batches}, validation batches: {valid_batches}, "
        f"test batches: {test_batches})"
    )


def _print_effective_configuration(
    args: Any, train_valid_test_num_samples: Any, ignored_flags: Dict[str, bool]
) -> None:
    print("> preparing dataset cache with the following effective values:")
    print(f"  world size:         {args.world_size}")
    print(f"  data parallel size: {args.data_parallel_size}")
    print(f"  global batch size:  {args.global_batch_size}")
    print(f"  cache path:         {args.data_cache_path}")
    print(" > datasets target sizes (minimum size):")
    print(f"    train:      {train_valid_test_num_samples[0]}")
    print(f"    validation: {train_valid_test_num_samples[1]}")
    print(f"    test:       {train_valid_test_num_samples[2]}")
    if ignored_flags["dataloader_fast_cache_load"]:
        print("> ignoring --dataloader-fast-cache-load during cache preparation")
    if ignored_flags["dataloader_defer_npy_index_mmap"]:
        print("> ignoring --dataloader-defer-npy-index-mmap during cache preparation")


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    """Build the explicit GPTDatasetConfig used for offline cache preparation."""

    tokenizer = build_tokenizer(args)

    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
    )


def build_dataset_caches(args: Any) -> Dict[str, Any]:
    """Build the dataset caches for the plain GPTDataset path."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            rank=int(os.environ.get("RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
        )

    assert dist.get_world_size() == 1, "tools/prepare_cache.py only supports world size of 1"
    assert args.data_parallel_size == 1 and args.tensor_model_parallel_size == 1 and args.pipeline_model_parallel_size == 1 and args.expert_model_parallel_size == 1, "tools/prepare_cache.py only supports data_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size, and expert_model_parallel_size of 1"

    _validate_prepare_cache_args(args)
    ignored_flags = _disable_cache_load_only_flags(args)

    unset_global_variables()
    set_args(args)

    try:
        # Derive train_iters from --train-samples when needed (pretrain() does the same).
        update_train_iters(args)
        train_valid_test_num_samples = get_train_valid_test_num_samples()
        _print_effective_configuration(args, train_valid_test_num_samples, ignored_flags)

        compile_helpers()

        config = core_gpt_dataset_config_from_args(args)
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, train_valid_test_num_samples, lambda: True, config
        ).build()

        print("> finished preparing dataset cache")
        print(f"  train dataset length:      {_get_dataset_length(train_ds)}")
        print(f"  validation dataset length: {_get_dataset_length(valid_ds)}")
        print(f"  test dataset length:       {_get_dataset_length(test_ds)}")

        if args.prepare_cache_sample_map_path is not None:
            _write_prepare_cache_batch_map(
                args.prepare_cache_sample_map_path,
                args,
                train_ds,
                valid_ds,
                test_ds,
            )

        return {
            "world_size": args.world_size,
            "data_parallel_size": args.data_parallel_size,
            "global_batch_size": args.global_batch_size,
            "train_valid_test_num_samples": tuple(train_valid_test_num_samples),
            "train_dataset_length": _get_dataset_length(train_ds),
            "valid_dataset_length": _get_dataset_length(valid_ds),
            "test_dataset_length": _get_dataset_length(test_ds),
            "sample_map_path": args.prepare_cache_sample_map_path,
        }
    finally:
        unset_global_variables()


def main() -> Dict[str, Any]:
    args = parse_args(
        extra_args_provider=_extra_args_provider,
        ignore_unknown_args=False,
    )
    _normalize_prepare_cache_args(args)
    validate_args(args, defaults={"tokenizer_type": "GPT2BPETokenizer"})
    return build_dataset_caches(args)


if __name__ == "__main__":
    main()
