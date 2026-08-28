#!/usr/bin/env python3
"""Compute matched task, fidelity, calibration, and selective metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dualkd.config import SUPPORTED_DATASETS, SUPPORTED_METHODS, load_config, retention_tag
from dualkd.data import load_jsonl
from dualkd.evaluation import (
    DEFERRAL_RATES,
    cascade_curve,
    error_detection_metrics,
    paired_bootstrap_binary_ranking_difference,
    summarize_bootstrap,
)
from dualkd.metrics import area_under_risk_coverage, expected_calibration_error
from dualkd.scoring import entropy


METRICS = (
    "accuracy",
    "dense_accuracy",
    "entropy_mae",
    "margin_mae",
    "error_auroc",
    "error_auprc",
    "aurc",
    "ece",
    "error_rate",
)


def margin(values: np.ndarray) -> np.ndarray:
    top = np.sort(np.asarray(values, dtype=np.float64), axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty result file {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_values(dense: np.ndarray, probabilities: np.ndarray, labels: np.ndarray):
    dense_entropy = entropy(dense, normalized=True)
    compressed_entropy = entropy(probabilities, normalized=True)
    dense_margin = margin(dense)
    compressed_margin = margin(probabilities)
    correct = probabilities.argmax(axis=1) == labels
    ranking = error_detection_metrics(
        correct,
        compressed_entropy,
        probabilities.max(axis=1),
        compressed_margin,
    )
    values = {
        "accuracy": float(correct.mean()),
        "dense_accuracy": float((dense.argmax(axis=1) == labels).mean()),
        "entropy_mae": float(np.abs(compressed_entropy - dense_entropy).mean()),
        "margin_mae": float(np.abs(compressed_margin - dense_margin).mean()),
        "ece": float(expected_calibration_error(probabilities, labels)),
        "aurc": float(area_under_risk_coverage(probabilities, labels)),
        "error_auroc": ranking["entropy_auroc"],
        "error_auprc": ranking["entropy_auprc"],
        "error_rate": ranking["error_rate"],
    }
    arrays = {
        "dense_prediction": dense.argmax(axis=1),
        "compressed_prediction": probabilities.argmax(axis=1),
        "dense_correct": dense.argmax(axis=1) == labels,
        "compressed_correct": correct,
        "dense_entropy": dense_entropy,
        "compressed_entropy": compressed_entropy,
        "dense_margin": dense_margin,
        "compressed_margin": compressed_margin,
        "compressed_max_probability": probabilities.max(axis=1),
    }
    return values, arrays


def load_prediction(root, model, dataset, partition, method, retention, seed):
    path = root / model / dataset / "predictions" / partition / f"{method}_seed_{seed}_{retention_tag(retention)}.npz"
    with np.load(path, allow_pickle=False) as saved:
        return np.asarray(saved["probabilities"], dtype=np.float64), json.loads(
            str(saved["metadata"])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Gate-training seed to analyse. Defaults to the configured seed. "
            "The analysis is single-seed by design; run it once per seed and "
            "compare the outputs rather than pooling them."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.root or Path(config["output_root"])
    source_root = args.source_root or Path(config["source_root"])
    output = args.output or root / "outputs/downstream"
    output.mkdir(parents=True, exist_ok=True)
    datasets = tuple(dict.fromkeys(args.datasets or SUPPORTED_DATASETS))
    models = tuple(dict.fromkeys(args.models or config["models"]))
    if any(model not in config["models"] for model in models):
        raise ValueError("requested model is absent from the configuration")
    retentions = config["retentions"]
    seed = args.seed if args.seed is not None else int(config["training"]["seed"])
    reps = args.bootstrap_replicates or int(config["evaluation"]["bootstrap_replicates"])
    if reps < 1000:
        raise ValueError("use at least 1,000 paired bootstrap replicates")

    examples_by_partition = {}
    dense_by_key = {}
    for dataset in datasets:
        for partition in ("lambda_selection", "evaluation"):
            examples = load_jsonl(source_root / "datasets" / partition / f"{dataset}.jsonl")
            examples_by_partition[(dataset, partition)] = examples
            ids = np.asarray([example.id for example in examples], dtype=np.str_)
            for model in models:
                filename = "selection_dense.npz" if partition == "lambda_selection" else "evaluation_dense.npz"
                with np.load(source_root / model / dataset / filename, allow_pickle=False) as saved:
                    dense = np.asarray(saved["probabilities"], dtype=np.float64)
                    labels = np.asarray(saved["labels"], dtype=np.int64)
                    dense_ids = np.asarray(saved["example_ids"], dtype=np.str_)
                if not np.array_equal(ids, dense_ids):
                    raise ValueError(f"dense IDs do not match {dataset}/{model}/{partition}")
                dense_by_key[(dataset, model, partition)] = (dense, labels, ids)

    partition_rows = []
    bootstrap_rows = []
    cascade_rows = []
    conditions = {}
    per_example_path = output / "per_example.jsonl"
    with per_example_path.open("w", encoding="utf-8") as per_example:
        for dataset_index, dataset in enumerate(datasets):
            label = config["datasets"][dataset]["label"]
            for model_index, model in enumerate(models):
                for partition in ("lambda_selection", "evaluation"):
                    dense, labels, ids = dense_by_key[(dataset, model, partition)]
                    for retention in retentions:
                        for method in SUPPORTED_METHODS:
                            probabilities, metadata = load_prediction(
                                root, model, dataset, partition, method, retention, seed
                            )
                            values, arrays = metric_values(dense, probabilities, labels)
                            partition_rows.append(
                                {
                                    "model": model,
                                    "dataset": label,
                                    "partition": partition,
                                    "retention": retention,
                                    "method": method,
                                    "seed": seed,
                                    "examples": len(labels),
                                    "epochs": metadata.get("epochs_requested"),
                                    "gate_training_examples": metadata.get("gate_training_examples"),
                                    **values,
                                }
                            )
                            conditions[(dataset, model, partition, retention, method)] = {
                                "values": values,
                                "arrays": arrays,
                                "probabilities": probabilities,
                            }
                            if partition == "evaluation":
                                for index in range(len(labels)):
                                    per_example.write(
                                        json.dumps(
                                            {
                                                "model": model,
                                                "dataset": label,
                                                "retention": retention,
                                                "method": method,
                                                "seed": seed,
                                                "example_id": str(ids[index]),
                                                "gold_label": int(labels[index]),
                                                "dense_prediction": int(arrays["dense_prediction"][index]),
                                                "compressed_prediction": int(arrays["compressed_prediction"][index]),
                                                "dense_correct": bool(arrays["dense_correct"][index]),
                                                "compressed_correct": bool(arrays["compressed_correct"][index]),
                                                "dense_entropy": float(arrays["dense_entropy"][index]),
                                                "compressed_entropy": float(arrays["compressed_entropy"][index]),
                                                "dense_margin": float(arrays["dense_margin"][index]),
                                                "compressed_margin": float(arrays["compressed_margin"][index]),
                                                "compressed_max_probability": float(arrays["compressed_max_probability"][index]),
                                            },
                                            allow_nan=False,
                                        )
                                        + "\n"
                                    )
                dense, labels, _ = dense_by_key[(dataset, model, "evaluation")]
                dense_predictions = dense.argmax(axis=1)
                for retention in retentions:
                    kl = conditions[(dataset, model, "evaluation", retention, "kl_only")]
                    dual = conditions[(dataset, model, "evaluation", retention, "dual_kl")]
                    for metric in ("error_auroc", "error_auprc"):
                        samples = paired_bootstrap_binary_ranking_difference(
                            dual["arrays"]["compressed_correct"],
                            dual["arrays"]["compressed_entropy"],
                            kl["arrays"]["compressed_correct"],
                            kl["arrays"]["compressed_entropy"],
                            reps=reps,
                            seed=20261900 + dataset_index * 100000 + model_index * 10000 + int(retention * 1000),
                            metric=metric,
                        )
                        bootstrap_rows.append(
                            {
                                "model": model,
                                "dataset": label,
                                "retention": retention,
                                "comparison": "dual_kl_vs_kl_only",
                                "seed": seed,
                                "metric": metric,
                                **summarize_bootstrap(
                                    samples,
                                    point=float(dual["values"][metric] - kl["values"][metric]),
                                ),
                            }
                        )
                    for method in SUPPORTED_METHODS:
                        record = conditions[(dataset, model, "evaluation", retention, method)]
                        for row in cascade_curve(
                            record["probabilities"].argmax(axis=1),
                            dense_predictions,
                            labels,
                            record["arrays"]["compressed_entropy"],
                            DEFERRAL_RATES,
                        ):
                            cascade_rows.append(
                                {
                                    "model": model,
                                    "dataset": label,
                                    "retention": retention,
                                    "method": method,
                                    **row,
                                }
                            )

    evaluation_rows = [row for row in partition_rows if row["partition"] == "evaluation"]
    validation_rows = [row for row in partition_rows if row["partition"] == "lambda_selection"]
    training_rows = []
    for dataset in datasets:
        for model in models:
            for retention in retentions:
                for method in SUPPORTED_METHODS:
                    path = root / model / dataset / method / retention_tag(retention) / f"seed_{seed}/training.json"
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    coverage = metadata["sampling_coverage"]
                    training_rows.append(
                        {
                            "model": model,
                            "dataset": config["datasets"][dataset]["label"],
                            "retention": retention,
                            "method": method,
                            "steps": metadata["steps"],
                            "training_seconds": metadata["training_seconds"],
                            "peak_cuda_memory_gib": metadata.get("peak_cuda_memory_gib"),
                            "choice_microbatch_size": metadata.get("choice_microbatch_size"),
                            "unique_examples": coverage["unique_examples_presented"],
                            "sample_presentations": coverage["sample_presentations"],
                            "minimum_presentations": coverage["minimum_presentations_per_example"],
                            "maximum_presentations": coverage["maximum_presentations_per_example"],
                            "dual_final": metadata.get("dual_final"),
                            "final_kl": metadata["trace"][-1]["kl"],
                            "final_entropy_bias": metadata["trace"][-1].get("entropy_bias"),
                            "final_entropy_mae": metadata["trace"][-1].get("entropy_mae"),
                        }
                    )
    for name, rows in (
        ("evaluation_summary.csv", evaluation_rows),
        ("validation_summary.csv", validation_rows),
        ("training_audit.csv", training_rows),
        ("bootstrap_comparisons.csv", bootstrap_rows),
        ("cascade_curves.csv", cascade_rows),
    ):
        write_csv(output / name, rows)
    manifest = {
        "study": config["study"],
        "models": list(models),
        "datasets": {
            key: {
                "label": config["datasets"][key]["label"],
                "partition_counts": config["datasets"][key]["expected"],
                "protocol": config["datasets"][key]["protocol"],
            }
            for key in datasets
        },
        "methods": list(SUPPORTED_METHODS),
        "method_definitions": {
            "kl_only": "teacher-to-student forward KL",
            "dual_kl": "forward KL with global entropy-equality dual ascent",
        },
        "retentions": list(retentions),
        "epochs": int(config["training"]["epochs"]),
        "seed": seed,
        "choice_objective": "full differentiable length-normalized continuation likelihood",
        "sampling": "exact without-replacement epochs",
        "bootstrap_replicates": reps,
        "counts": {
            "per_example": sum(1 for _ in per_example_path.open(encoding="utf-8")),
            "evaluation_summary": len(evaluation_rows),
            "validation_summary": len(validation_rows),
            "training_audit": len(training_rows),
            "bootstrap": len(bootstrap_rows),
            "cascade": len(cascade_rows),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
