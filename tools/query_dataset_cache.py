"""Query a Megatron dataset cache manifest for the source locations of a training batch."""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from megatron.core.datasets.indexed_dataset import _IndexReader


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def _split_manifest(manifest: Dict[str, Any], split: str) -> Dict[str, Any]:
    split_manifest = manifest["splits"][split]
    if isinstance(split_manifest, list):
        raise ValueError(
            f"Split {split!r} has multiple datasets in the manifest. Query one concrete split "
            "entry manually or extend this tool with a validation-set selector."
        )
    if split_manifest is None:
        raise ValueError(f"Split {split!r} is not present in the manifest")
    return split_manifest


def _query_gpt_dataset(
    source_dataset: Dict[str, Any],
    local_sample_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    shuffle_index = np.load(source_dataset["cache"]["shuffle_index"], mmap_mode="r")
    sample_index = np.load(source_dataset["cache"]["sample_index"], mmap_mode="r")
    document_index = np.load(source_dataset["cache"]["document_index"], mmap_mode="r")

    shuffled_sample_indices = np.asarray(shuffle_index[local_sample_indices], dtype=np.int64)
    sample_starts = sample_index[shuffled_sample_indices]
    sample_ends = sample_index[shuffled_sample_indices + 1]

    doc_index_beg = np.asarray(sample_starts[:, 0], dtype=np.int64)
    doc_index_beg_offset = np.asarray(sample_starts[:, 1], dtype=np.int64)
    doc_index_end = np.asarray(sample_ends[:, 0], dtype=np.int64)
    doc_index_end_offset = np.asarray(sample_ends[:, 1], dtype=np.int64)

    sequence_id_beg = np.asarray(document_index[doc_index_beg], dtype=np.int64)
    sequence_id_end = np.asarray(document_index[doc_index_end], dtype=np.int64)

    # Only the .idx metadata is needed for byte offsets; avoid constructing IndexedDataset,
    # which also opens the .bin file and can emit noisy __del__ errors if initialization fails.
    idx_path = source_dataset["indexed_dataset"]["idx_path"]
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Could not find indexed dataset .idx file at {idx_path!r}")
    index = _IndexReader(idx_path, multimodal=False)
    dtype_size = int(index.dtype_size)
    sequence_pointer_beg = index.sequence_pointers[sequence_id_beg]
    byte_offset_beg = np.asarray(
        sequence_pointer_beg + doc_index_beg_offset * dtype_size, dtype=np.int64
    )

    return {
        "local_sample_idx": local_sample_indices.astype(np.int64),
        "shuffled_sample_idx": shuffled_sample_indices,
        "doc_index_beg": doc_index_beg,
        "doc_index_beg_offset": doc_index_beg_offset,
        "doc_index_end": doc_index_end,
        "doc_index_end_offset": doc_index_end_offset,
        "sequence_id_beg": sequence_id_beg,
        "sequence_id_end": sequence_id_end,
        "byte_offset_beg": byte_offset_beg,
        "num_source_sequences": doc_index_end - doc_index_beg + 1,
    }


def query_batch(manifest: Dict[str, Any], split: str, batch_idx: int) -> Dict[str, Any]:
    split_data = _split_manifest(manifest, split)
    global_batch_size = int(manifest["global_batch_size"])
    sample_start = batch_idx * global_batch_size
    sample_end = min(sample_start + global_batch_size, int(split_data["length"]))
    if sample_start >= int(split_data["length"]):
        raise ValueError(
            f"batch_idx {batch_idx} starts at sample {sample_start}, but split {split!r} has "
            f"length {split_data['length']}"
        )

    global_sample_indices = np.arange(sample_start, sample_end, dtype=np.int64)
    source_datasets: List[Dict[str, Any]]

    if split_data["type"] == "BlendedDataset":
        dataset_index = np.load(split_data["cache"]["dataset_index"], mmap_mode="r")
        dataset_sample_index = np.load(
            split_data["cache"]["dataset_sample_index"], mmap_mode="r"
        )
        dataset_ids = np.asarray(dataset_index[global_sample_indices], dtype=np.int64)
        local_sample_indices = np.asarray(dataset_sample_index[global_sample_indices], dtype=np.int64)
        source_datasets = split_data["datasets"]
    else:
        dataset_ids = np.zeros(len(global_sample_indices), dtype=np.int64)
        local_sample_indices = global_sample_indices
        source_datasets = [split_data]

    columns = {
        "sample_idx": global_sample_indices,
        "dataset_id": dataset_ids,
    }
    datasets = []
    for dataset_id in np.unique(dataset_ids).astype(np.int64).tolist():
        source_dataset = source_datasets[dataset_id]
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset_path": source_dataset["dataset_path"],
                "bin_path": source_dataset["bin_path"],
                "idx_path": source_dataset["idx_path"],
            }
        )
        mask = dataset_ids == dataset_id
        positions = np.nonzero(mask)[0]
        source_columns = _query_gpt_dataset(source_dataset, local_sample_indices[mask])
        for name, values in source_columns.items():
            if name not in columns:
                columns[name] = np.empty(len(global_sample_indices), dtype=np.int64)
            columns[name][positions] = values

    return {
        "split": split,
        "batch_idx": batch_idx,
        "global_batch_size": global_batch_size,
        "sample_idx_begin": sample_start,
        "sample_idx_end_exclusive": sample_end,
        "datasets": datasets,
        "columns": {name: values.tolist() for name, values in columns.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to cache manifest JSON")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--batch-idx", type=int, required=True)
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    result = query_batch(_load_json(args.manifest), args.split, args.batch_idx)
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        with open(args.output, "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=2, sort_keys=True)
            writer.write("\n")


if __name__ == "__main__":
    main()
