"""Analyze the matched UPGD ablation artifacts on HellaSwag."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.scoring import entropy
from tuac.stochastic_gate_pruning import analyze_upgd_comparison


ROOT = Path("runs/iter14_ablation_hellaswag")
MODELS = ("1p5b", "3b_nf4")
METHODS = ("random_seed1", "random_seed2", "random_seed3", "one_shot", "kl_only", "kl_entropy_margin", "full_upgd")


def margin(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    top = np.partition(values, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def main() -> int:
    rows: list[dict] = []
    for model_index, model_key in enumerate(MODELS):
        model_dir = ROOT / model_key
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
                split_seed=20261000 + model_index * 100 + method_index,
                bootstrap_replicates=5000,
                bootstrap_seed=20261100 + model_index * 100 + method_index,
            )
            with np.load(model_dir / "scores" / f"{method}.npz", allow_pickle=False) as saved:
                probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
                if not np.array_equal(ids, np.asarray(saved["example_ids"], dtype=np.str_)):
                    raise ValueError(f"example IDs changed for {model_key}/{method}")
            row_lookup = {item["condition"]: item for item in result["metric_rows"]}
            metric = row_lookup["upgd_aggressive"]
            baseline_metric = row_lookup["baseline_aggressive"]
            dense_metric = row_lookup["dense"]
            paired = result["paired_comparison"]
            rows.append({
                "model_key": model_key,
                "method": method,
                "dataset": "HellaSwag",
                "examples": int(len(labels)),
                "accuracy": float(metric["accuracy"]),
                "baseline_accuracy": float(baseline_metric["accuracy"]),
                "dense_accuracy": float(dense_metric["accuracy"]),
                "difference": float(paired["difference"]),
                "ci_low": float(paired["bootstrap_95_ci"][0]),
                "ci_high": float(paired["bootstrap_95_ci"][1]),
                "gate": bool(result["gate"]["passed"]),
                "kl": float(metric["mean_kl_from_dense"]),
                "agreement": float(metric["prediction_agreement_with_dense"]),
                "entropy_rho": float(metric["entropy_spearman_with_dense"]),
                "aurc": float(metric["aurc"]),
                "ece": float(metric["ece"]),
                "entropy_mae": float(np.mean(np.abs(entropy(probabilities, normalized=True) - dense_entropy))),
                "margin_mae": float(np.mean(np.abs(margin(probabilities) - dense_margin))),
                "analysis": str(output.resolve()),
            })
    manifest = {"dataset": "HellaSwag", "models": list(MODELS), "methods": list(METHODS), "rows": rows,
                "bootstrap_replicates": 5000, "notes": ["Dense and all masked conditions share example IDs.", "Metrics use each model's dense probabilities as reference."]}
    (ROOT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
