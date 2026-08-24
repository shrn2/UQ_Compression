#!/usr/bin/env python3
"""Train or score the supported KL and Dual-KD structured-gate grid."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_DATASETS, SUPPORTED_METHODS, load_config, retention_tag
from dualkd.data import load_jsonl
from dualkd.hf import HFCausalLMScorer
from dualkd.modeling import register_channel_mask_hooks, transformer_blocks
from dualkd.training import train_gate_masks


PARTITIONS = ("lambda_selection", "evaluation")


def clear_cuda() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mask_checksum(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.int64).tobytes()).hexdigest()


def train_one(config, root, source_root, model_key, dataset_key, method, retention, seed):
    output_dir = root / model_key / dataset_key / method / retention_tag(retention) / f"seed_{seed}"
    training_path = output_dir / "training.json"
    masks_path = output_dir / "masks.npz"
    settings = config["training"]
    metadata = train_gate_masks(
        method=method,
        data_path=source_root / "datasets/gate_training" / f"{dataset_key}.jsonl",
        calibration_artifact_path=source_root / model_key / dataset_key / "calibration.npz",
        model_name=config["models"][model_key],
        output_path=training_path,
        masks_output_path=masks_path,
        retention=retention,
        epochs=int(settings["epochs"]),
        per_stratum=int(settings["per_stratum"]),
        learning_rate=float(settings["learning_rate"]),
        dual_learning_rate=float(settings["dual_learning_rate"]),
        start_temperature=float(settings["start_temperature"]),
        end_temperature=float(settings["end_temperature"]),
        start_noise=float(settings["start_noise"]),
        end_noise=float(settings["end_noise"]),
        choice_microbatch_size=int(settings["choice_microbatch_size"][dataset_key]),
        seed=seed,
        gradient_checkpointing=bool(settings["gradient_checkpointing"]),
        device_map="auto",
        dtype="bfloat16",
        wandb_config={
            "study": config["study"],
            "model_key": model_key,
            "dataset_key": dataset_key,
            "dataset": config["datasets"][dataset_key]["label"],
        },
    )
    metadata.update(
        {
            "study": config["study"],
            "model_key": model_key,
            "dataset_key": dataset_key,
            "dataset": config["datasets"][dataset_key]["label"],
            "method_key": method,
            "gate_training_examples": int(config["datasets"][dataset_key]["expected"]["gate_training"]),
            "sample_presentations": int(metadata["sampling_coverage"]["sample_presentations"]),
        }
    )
    training_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with np.load(masks_path, allow_pickle=False) as saved:
        retained = np.asarray(saved["retained"], dtype=np.int64)
        baseline = np.asarray(saved["baseline_retained"], dtype=np.int64)
        logits = np.asarray(saved["logits"], dtype=np.float32)
        sparsity = np.asarray(saved["sparsity"])
    np.savez_compressed(
        masks_path,
        retained=retained,
        baseline_retained=baseline,
        logits=logits,
        sparsity=sparsity,
        method=np.asarray(method, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "method": method,
        "retention_ratio": retention,
        "seed": seed,
        "steps": metadata["steps"],
        "training_path": str(training_path.resolve()),
        "mask_path": str(masks_path.resolve()),
        "mask_sha256": mask_checksum(retained),
    }


def score_model(config, root, source_root, model_key, dataset_key, seed, batch_size):
    scorer = HFCausalLMScorer(
        config["models"][model_key],
        device_map="auto",
        dtype="bfloat16",
        batch_size=batch_size,
        max_batch_tokens=8192,
    )
    blocks = transformer_blocks(scorer.model)
    rows = []
    for partition in PARTITIONS:
        examples = load_jsonl(source_root / "datasets" / partition / f"{dataset_key}.jsonl")
        dense_name = "selection_dense.npz" if partition == "lambda_selection" else "evaluation_dense.npz"
        with np.load(source_root / model_key / dataset_key / dense_name, allow_pickle=False) as saved:
            dense_ids = np.asarray(saved["example_ids"], dtype=np.str_)
        example_ids = np.asarray([example.id for example in examples], dtype=np.str_)
        if not np.array_equal(dense_ids, example_ids):
            raise ValueError(f"dense IDs do not match {dataset_key}/{partition}")
        for method in SUPPORTED_METHODS:
            for retention in config["retentions"]:
                mask_path = root / model_key / dataset_key / method / retention_tag(retention) / f"seed_{seed}/masks.npz"
                output = root / model_key / dataset_key / "predictions" / partition / f"{method}_seed_{seed}_{retention_tag(retention)}.npz"
                if output.exists():
                    with np.load(output, allow_pickle=False) as saved:
                        rows.append(json.loads(str(saved["metadata"])))
                    continue
                with np.load(mask_path, allow_pickle=False) as saved:
                    retained = np.asarray(saved["retained"], dtype=np.int64)
                    metadata = json.loads(str(saved["metadata"]))
                handles = register_channel_mask_hooks(blocks, retained)
                started = time.perf_counter()
                try:
                    probabilities = scorer.score(examples).astype(np.float32)
                finally:
                    for handle in handles:
                        handle.remove()
                output_metadata = {
                    **metadata,
                    "partition": partition,
                    "scoring_seconds": time.perf_counter() - started,
                    "mask_mode": "logical_channel_hook",
                }
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output,
                    probabilities=probabilities,
                    example_ids=example_ids,
                    metadata=np.asarray(json.dumps(output_metadata, sort_keys=True), dtype=np.str_),
                )
                rows.append({**output_metadata, "artifact": str(output.resolve())})
                print(f"[score] {model_key} {dataset_key} {partition} {method} {retention:.2f}", flush=True)
                del probabilities
                clear_cuda()
    del scorer
    clear_cuda()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stage", choices=("train", "score"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset-key", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS)
    parser.add_argument("--retention", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--manifest-path", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model_key not in config["models"]:
        raise ValueError(f"unknown model key {args.model_key!r}")
    root = args.root or Path(config["output_root"])
    source_root = args.source_root or Path(config["source_root"])
    seed = args.seed if args.seed is not None else int(config["training"]["seed"])
    if args.retention is not None and args.retention not in config["retentions"]:
        raise ValueError("retention is not present in the experiment configuration")
    if args.stage == "train":
        methods = (args.method,) if args.method else SUPPORTED_METHODS
        retentions = (args.retention,) if args.retention is not None else config["retentions"]
        rows = [
            train_one(config, root, source_root, args.model_key, args.dataset_key, method, retention, seed)
            for method in methods
            for retention in retentions
        ]
    else:
        batch_size = args.batch_size or int(config["evaluation"]["batch_size"][args.dataset_key])
        rows = score_model(
            config, root, source_root, args.model_key, args.dataset_key, seed, batch_size
        )
    manifest = {
        "study": config["study"],
        "stage": args.stage,
        "model_key": args.model_key,
        "dataset_key": args.dataset_key,
        "methods": list(SUPPORTED_METHODS),
        "rows": rows,
    }
    path = args.manifest_path or root / "manifest_parts" / f"{args.stage}_{args.model_key}_{args.dataset_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
