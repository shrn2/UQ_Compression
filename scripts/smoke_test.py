#!/usr/bin/env python3
"""Run a tiny GPU smoke test for both supported objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_DATASETS, SUPPORTED_METHODS, load_config
from dualkd.training import train_gate_masks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--model-key", default="1p5b_bf16")
    parser.add_argument("--dataset-key", choices=SUPPORTED_DATASETS, default="arc_challenge")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root = args.source_root or Path(config["source_root"])
    if args.model_key not in config["models"]:
        raise ValueError(f"unknown model key {args.model_key!r}")
    source_calibration = source_root / args.model_key / args.dataset_key / "calibration.npz"
    with np.load(source_calibration, allow_pickle=False) as saved:
        strata = np.asarray(saved["stratum_labels"], dtype=np.int64)
        selected = np.concatenate([np.flatnonzero(strata == label)[:2] for label in range(3)])
        smoke_calibration = {
            "selected_indices": np.asarray(saved["selected_indices"], dtype=np.int64)[selected],
            "dense_probabilities": np.asarray(saved["dense_probabilities"], dtype=np.float32)[selected],
            "example_ids": np.asarray(saved["example_ids"], dtype=np.str_)[selected],
            "stratum_labels": strata[selected],
            "baseline_importance": np.asarray(saved["baseline_importance"], dtype=np.float32),
        }
    calibration_path = args.output_root / "calibration.npz"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(calibration_path, **smoke_calibration)
    rows = []
    for method in SUPPORTED_METHODS:
        output = args.output_root / method
        metadata = train_gate_masks(
            method=method,
            data_path=source_root / "datasets/gate_training" / f"{args.dataset_key}.jsonl",
            calibration_artifact_path=calibration_path,
            model_name=config["models"][args.model_key],
            output_path=output / "training.json",
            masks_output_path=output / "masks.npz",
            retention=0.75,
            epochs=1,
            per_stratum=1,
            learning_rate=0.05,
            dual_learning_rate=0.05,
            start_temperature=0.5,
            end_temperature=0.1,
            start_noise=0.5,
            end_noise=0.05,
            choice_microbatch_size=1,
            seed=1,
        )
        rows.append(
            {
                "method": method,
                "steps": metadata["steps"],
                "final_loss": metadata["trace"][-1]["loss"],
                "dual_final": metadata["dual_final"],
            }
        )
    print(json.dumps({"status": "passed", "runs": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
