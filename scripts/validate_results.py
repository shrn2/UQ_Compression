#!/usr/bin/env python3
"""Validate the complete two-method, four-dataset experiment artifact set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_DATASETS, SUPPORTED_METHODS, load_config, retention_tag
from dualkd.data import load_jsonl


PARTITIONS = ("lambda_selection", "evaluation")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--seed",
        type=int,
        help="Gate-training seed to validate. Defaults to the configured seed.",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--downstream", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.root or Path(config["output_root"])
    source_root = args.source_root or Path(config["source_root"])
    models = tuple(config["models"])
    retentions = config["retentions"]
    seed = args.seed if args.seed is not None else int(config["training"]["seed"])
    epochs = int(config["training"]["epochs"])
    checks: dict[str, object] = {}
    examples = {}
    for dataset in SUPPORTED_DATASETS:
        all_ids = []
        for partition, expected in config["datasets"][dataset]["expected"].items():
            rows = load_jsonl(source_root / "datasets" / partition / f"{dataset}.jsonl")
            require(len(rows) == int(expected), f"wrong count for {dataset}/{partition}")
            ids = [row.id for row in rows]
            require(len(ids) == len(set(ids)), f"duplicate IDs in {dataset}/{partition}")
            all_ids.extend(ids)
            examples[(dataset, partition)] = rows
        require(len(all_ids) == len(set(all_ids)), f"partition overlap in {dataset}")
    checks["dataset_partitions"] = "counts_match_and_ids_are_disjoint"

    calibrations = training_conditions = prediction_files = 0
    for dataset in SUPPORTED_DATASETS:
        training_count = int(config["datasets"][dataset]["expected"]["gate_training"])
        gate_ids = np.asarray([row.id for row in examples[(dataset, "gate_training")]], dtype=np.str_)
        for model in models:
            calibration = source_root / model / dataset / "calibration.npz"
            require(calibration.exists(), f"missing {calibration}")
            with np.load(calibration, allow_pickle=False) as saved:
                dense = np.asarray(saved["dense_probabilities"])
                ids = np.asarray(saved["example_ids"], dtype=np.str_)
                strata = np.asarray(saved["stratum_labels"], dtype=np.int64)
                importance = np.asarray(saved["baseline_importance"])
            require(dense.shape[0] == training_count, f"wrong calibration rows for {model}/{dataset}")
            require(np.array_equal(ids, gate_ids), f"calibration IDs mismatch for {model}/{dataset}")
            require(np.array_equal(np.unique(strata), np.arange(3)), f"invalid strata for {model}/{dataset}")
            require(importance.ndim == 2 and np.all(np.isfinite(importance)), f"invalid importance for {model}/{dataset}")
            calibrations += 1
            for partition in PARTITIONS:
                filename = "selection_dense.npz" if partition == "lambda_selection" else "evaluation_dense.npz"
                expected_ids = np.asarray([row.id for row in examples[(dataset, partition)]], dtype=np.str_)
                with np.load(source_root / model / dataset / filename, allow_pickle=False) as saved:
                    ids = np.asarray(saved["example_ids"], dtype=np.str_)
                    probabilities = np.asarray(saved["probabilities"])
                require(np.array_equal(ids, expected_ids), f"dense IDs mismatch for {model}/{dataset}/{partition}")
                require(probabilities.shape[0] == len(expected_ids), f"dense row mismatch for {model}/{dataset}/{partition}")
            for method in SUPPORTED_METHODS:
                for retention in retentions:
                    condition = root / model / dataset / method / retention_tag(retention) / f"seed_{seed}"
                    training_path, masks_path = condition / "training.json", condition / "masks.npz"
                    require(training_path.exists() and masks_path.exists(), f"missing {condition}")
                    metadata = json.loads(training_path.read_text(encoding="utf-8"))
                    coverage = metadata["sampling_coverage"]
                    require(
                        metadata.get("method_key", metadata.get("method")) == method,
                        f"method mismatch in {condition}",
                    )
                    require(metadata["epochs_requested"] == epochs, f"epoch mismatch in {condition}")
                    require(metadata["full_choice_objective"] is True, f"full-choice disabled in {condition}")
                    require(coverage["unique_examples_presented"] == training_count, f"coverage mismatch in {condition}")
                    require(coverage["sample_presentations"] == epochs * training_count, f"presentation mismatch in {condition}")
                    require(coverage["minimum_presentations_per_example"] == epochs, f"minimum coverage mismatch in {condition}")
                    require(coverage["maximum_presentations_per_example"] == epochs, f"maximum coverage mismatch in {condition}")
                    require(math.isfinite(float(metadata["training_seconds"])), f"invalid duration in {condition}")
                    require((metadata["dual_final"] is None) == (method == "kl_only"), f"dual state mismatch in {condition}")
                    training_conditions += 1
                    for partition in PARTITIONS:
                        prediction = root / model / dataset / "predictions" / partition / f"{method}_seed_{seed}_{retention_tag(retention)}.npz"
                        require(prediction.exists(), f"missing {prediction}")
                        expected_ids = np.asarray([row.id for row in examples[(dataset, partition)]], dtype=np.str_)
                        with np.load(prediction, allow_pickle=False) as saved:
                            ids = np.asarray(saved["example_ids"], dtype=np.str_)
                            probabilities = np.asarray(saved["probabilities"])
                        require(np.array_equal(ids, expected_ids), f"prediction IDs mismatch in {prediction}")
                        require(probabilities.shape[0] == len(expected_ids), f"prediction row mismatch in {prediction}")
                        require(np.all(np.isfinite(probabilities)), f"nonfinite probabilities in {prediction}")
                        require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), f"unnormalized probabilities in {prediction}")
                        prediction_files += 1
    expected_conditions = len(SUPPORTED_DATASETS) * len(models) * len(SUPPORTED_METHODS) * len(retentions)
    require(calibrations == len(SUPPORTED_DATASETS) * len(models), "calibration count mismatch")
    require(training_conditions == expected_conditions, "training condition count mismatch")
    require(prediction_files == 2 * expected_conditions, "prediction file count mismatch")
    checks.update(
        {
            "calibrations": calibrations,
            "training_conditions": training_conditions,
            "prediction_files": prediction_files,
        }
    )

    downstream = args.downstream or root / "outputs/downstream"
    counts = {
        "evaluation": len(read_csv(downstream / "evaluation_summary.csv")),
        "validation": len(read_csv(downstream / "validation_summary.csv")),
        "training": len(read_csv(downstream / "training_audit.csv")),
        "bootstrap": len(read_csv(downstream / "bootstrap_comparisons.csv")),
        "cascade": len(read_csv(downstream / "cascade_curves.csv")),
    }
    require(counts["evaluation"] == expected_conditions, "evaluation summary count mismatch")
    require(counts["validation"] == expected_conditions, "validation summary count mismatch")
    require(counts["training"] == expected_conditions, "training audit count mismatch")
    require(counts["bootstrap"] == expected_conditions, "bootstrap count mismatch")
    require(counts["cascade"] == expected_conditions * 12, "cascade count mismatch")
    checks["downstream_rows"] = counts

    report_dir = args.report_dir or root / "report"
    artifact_path, html_path = report_dir / "artifact.json", report_dir / "report.html"
    require(artifact_path.exists() and html_path.exists(), "missing report outputs")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    require(tuple(artifact["methods"]) == SUPPORTED_METHODS, "report contains unsupported methods")
    require(len(artifact["evaluation"]) == expected_conditions, "report evaluation snapshot is incomplete")
    require(html_path.stat().st_size > 10_000, "report HTML is unexpectedly small")
    for dataset in SUPPORTED_DATASETS:
        require(config["datasets"][dataset]["label"] in html_path.read_text(encoding="utf-8"), f"report omits {dataset}")
    checks["report"] = {"artifact": str(artifact_path.resolve()), "html": str(html_path.resolve())}

    receipt = args.receipt or root / "validation/receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps({"status": "passed", "checks": checks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "receipt": str(receipt), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
