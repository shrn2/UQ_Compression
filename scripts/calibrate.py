#!/usr/bin/env python3
"""Build dense references and gate-initialization artifacts."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_DATASETS, load_config
from dualkd.data import load_jsonl
from dualkd.hf import HFCausalLMScorer
from dualkd.modeling import ffn_modules, transformer_blocks
from dualkd.scoring import entropy


def clear_cuda() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_with_activation_importance(scorer, blocks, examples):
    """Collect dense probabilities and activation-times-weight-norm importance."""

    sums = [
        np.zeros(ffn_modules(block)[2].in_features, dtype=np.float64)
        for _, block in blocks
    ]
    counts = np.zeros(len(blocks), dtype=np.int64)
    handles = []
    for layer_index, (_, block) in enumerate(blocks):
        _, _, down = ffn_modules(block)

        def hook(_module, inputs, index=layer_index):
            values = inputs[0].detach()[:, -1, :].abs().sum(dim=0).float().cpu().numpy()
            sums[index] += values
            counts[index] += int(inputs[0].shape[0])

        handles.append(down.register_forward_pre_hook(hook))
    try:
        probabilities = scorer.score(examples).astype(np.float32)
    finally:
        for handle in handles:
            handle.remove()
    importance = []
    for index, (_, block) in enumerate(blocks):
        _, _, down = ffn_modules(block)
        activation = sums[index] / max(1, int(counts[index]))
        weight_norm = down.weight.detach().float().norm(dim=0).cpu().numpy()
        importance.append(activation * weight_norm)
    return probabilities, np.stack(importance).astype(np.float32)


def save_dense(path: Path, examples, probabilities: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=probabilities.astype(np.float32),
        labels=np.asarray([example.answer for example in examples], dtype=np.int64),
        example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset-key", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model_key not in config["models"]:
        raise ValueError(f"unknown model key {args.model_key!r}")
    source_root = args.source_root or Path(config["source_root"])
    data_root = source_root / "datasets"
    output_root = source_root / args.model_key / args.dataset_key
    batch_size = args.batch_size or int(
        config["evaluation"]["batch_size"][args.dataset_key]
    )
    examples = {
        partition: load_jsonl(data_root / partition / f"{args.dataset_key}.jsonl")
        for partition in ("gate_training", "lambda_selection", "evaluation")
    }
    manifest = json.loads(
        (data_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    scorer = HFCausalLMScorer(
        config["models"][args.model_key],
        device_map="auto",
        dtype="bfloat16",
        batch_size=batch_size,
        max_batch_tokens=8192,
    )
    scorer.model.config.use_cache = False
    blocks = transformer_blocks(scorer.model)
    dense, importance = score_with_activation_importance(
        scorer, blocks, examples["gate_training"]
    )
    dense_entropy = entropy(dense, normalized=True)
    order = np.argsort(dense_entropy, kind="stable")
    strata = np.empty(len(order), dtype=np.int64)
    cells = []
    for stratum, chunk in enumerate(np.array_split(order, 3)):
        strata[chunk] = stratum
        cells.append(
            {
                "stratum": stratum,
                "examples": len(chunk),
                "entropy_min": float(dense_entropy[chunk].min()),
                "entropy_max": float(dense_entropy[chunk].max()),
            }
        )
    dataset_manifest = manifest["datasets"][args.dataset_key]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": config["study"],
        "model_key": args.model_key,
        "model": config["models"][args.model_key],
        "dataset_key": args.dataset_key,
        "dataset": dataset_manifest["label"],
        "source_splits": {
            key: value["source_split"] for key, value in dataset_manifest["splits"].items()
        },
        "evaluation_protocol": dataset_manifest["protocol"],
        "examples": len(examples["gate_training"]),
        "entropy_cells": cells,
        "stratification": "three equal-frequency dense-teacher entropy strata",
        "gate_initialization": "mean absolute FFN activation times down-projection column norm",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "calibration.npz",
        dense_probabilities=dense,
        labels=np.asarray([example.answer for example in examples["gate_training"]], dtype=np.int64),
        selected_indices=np.arange(len(examples["gate_training"]), dtype=np.int64),
        example_ids=np.asarray([example.id for example in examples["gate_training"]], dtype=np.str_),
        stratum_labels=strata,
        baseline_importance=importance,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    for partition, filename in (
        ("lambda_selection", "selection_dense.npz"),
        ("evaluation", "evaluation_dense.npz"),
    ):
        probabilities = scorer.score(examples[partition]).astype(np.float32)
        save_dense(
            output_root / filename,
            examples[partition],
            probabilities,
            {**metadata, "partition": partition, "examples": len(examples[partition])},
        )
    (output_root / "calibration_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del scorer
    clear_cuda()
    print(
        json.dumps(
            {
                "model_key": args.model_key,
                "dataset_key": args.dataset_key,
                "counts": {key: len(value) for key, value in examples.items()},
                "strata": [cell["examples"] for cell in cells],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
