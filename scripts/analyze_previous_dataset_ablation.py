"""Analyze prior-cohort UPGD ablation artifacts with paired bootstrap intervals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.scoring import entropy
from tuac.stochastic_gate_pruning import analyze_upgd_comparison


ROOT = Path("runs/iter15_ablation_previous")
MODELS = ("1p5b", "3b_nf4")
DATASETS = ("mmlu_aux", "arc_challenge")
DATASET_LABELS = {"mmlu_aux": "MMLU auxiliary", "arc_challenge": "ARC-Challenge"}
METHODS = ("random_seed1", "random_seed2", "random_seed3", "one_shot", "kl_only", "kl_entropy_margin", "full_upgd")


def margin(probabilities: np.ndarray) -> np.ndarray:
    top = np.partition(np.asarray(probabilities, dtype=np.float64), -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def main() -> int:
    rows: list[dict] = []
    for model_index, model_key in enumerate(MODELS):
        for dataset_index, dataset_key in enumerate(DATASETS):
            model_dir = ROOT / model_key / dataset_key
            reference = model_dir / "scores" / "reference.npz"
            with np.load(reference, allow_pickle=False) as saved:
                dense = np.asarray(saved["dense"], dtype=np.float64)
                baseline = np.asarray(saved["baseline_aggressive"], dtype=np.float64)
                labels = np.asarray(saved["labels"], dtype=np.int64)
                ids = np.asarray(saved["example_ids"], dtype=np.str_)
            dense_entropy = entropy(dense, normalized=True)
            dense_margin = margin(dense)
            for method_index, method in enumerate(METHODS):
                output = model_dir / f"{method}_analysis.json"
                result = analyze_upgd_comparison(
                    reference_artifact_path=reference,
                    upgd_artifact_path=model_dir / "scores" / f"{method}.npz",
                    output_path=output,
                    strata=3,
                    conformal_alpha=0.1,
                    conformal_fraction=0.5,
                    split_seed=20262000 + model_index * 1000 + dataset_index * 100 + method_index,
                    bootstrap_replicates=5000,
                    bootstrap_seed=20262100 + model_index * 1000 + dataset_index * 100 + method_index,
                )
                with np.load(model_dir / "scores" / f"{method}.npz", allow_pickle=False) as saved:
                    probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
                    if not np.array_equal(ids, np.asarray(saved["example_ids"], dtype=np.str_)):
                        raise ValueError(f"example IDs changed for {model_key}/{dataset_key}/{method}")
                lookup = {item["condition"]: item for item in result["metric_rows"]}
                metric = lookup["upgd_aggressive"]
                baseline_metric = lookup["baseline_aggressive"]
                dense_metric = lookup["dense"]
                paired = result["paired_comparison"]
                rows.append({
                    "model_key": model_key, "dataset_key": dataset_key, "dataset": DATASET_LABELS[dataset_key],
                    "method": method, "examples": int(len(labels)), "accuracy": float(metric["accuracy"]),
                    "baseline_accuracy": float(baseline_metric["accuracy"]), "dense_accuracy": float(dense_metric["accuracy"]),
                    "difference": float(paired["difference"]), "ci_low": float(paired["bootstrap_95_ci"][0]),
                    "ci_high": float(paired["bootstrap_95_ci"][1]), "gate": bool(result["gate"]["passed"]),
                    "kl": float(metric["mean_kl_from_dense"]), "agreement": float(metric["prediction_agreement_with_dense"]),
                    "entropy_rho": float(metric["entropy_spearman_with_dense"]), "aurc": float(metric["aurc"]),
                    "ece": float(metric["ece"]), "entropy_mae": float(np.mean(np.abs(entropy(probabilities, normalized=True) - dense_entropy))),
                    "margin_mae": float(np.mean(np.abs(margin(probabilities) - dense_margin))), "analysis": str(output.resolve()),
                })
    random_summary = []
    for model_key in MODELS:
        for dataset_key in DATASETS:
            selected = [row for row in rows if row["model_key"] == model_key and row["dataset_key"] == dataset_key and row["method"].startswith("random_")]
            random_summary.append({
                "model_key": model_key, "dataset_key": dataset_key, "dataset": DATASET_LABELS[dataset_key], "seeds": len(selected),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])), "sd_accuracy": float(np.std([row["accuracy"] for row in selected], ddof=1)),
                "mean_difference": float(np.mean([row["difference"] for row in selected])), "sd_difference": float(np.std([row["difference"] for row in selected], ddof=1)),
            })
    manifest = {"models": list(MODELS), "datasets": list(DATASETS), "methods": list(METHODS), "rows": rows, "random_summary": random_summary, "bootstrap_replicates": 5000}
    (ROOT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"models": list(MODELS), "datasets": list(DATASETS), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
