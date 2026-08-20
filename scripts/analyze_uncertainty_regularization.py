"""Analyze lambda-UQ artifacts with seed summaries and paired bootstrap deltas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.cascade_compression import aurc_from_scores
from tuac.metrics import classification_metrics, spearman_correlation
from tuac.scoring import entropy, kl_divergence

ROOT = Path("runs/iter16_uncertainty_reg")
MODELS = ("1p5b", "3b_nf4")
DATASETS = ("hellaswag", "arc_challenge", "mmlu_aux")
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
SEEDS = (1, 2, 3)


def margin(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] < 2:
        return np.ones(values.shape[0])
    top = np.partition(values, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def metric_values(dense: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    row = classification_metrics(probabilities, labels)
    dense_entropy = entropy(dense, normalized=True)
    masked_entropy = entropy(probabilities, normalized=True)
    dense_margin = margin(dense)
    masked_margin = margin(probabilities)
    errors = probabilities.argmax(axis=1) != labels
    row.update({
        "kl": float(np.mean(kl_divergence(dense, probabilities))),
        "agreement": float(np.mean(dense.argmax(axis=1) == probabilities.argmax(axis=1))),
        "entropy_rho": float(spearman_correlation(dense_entropy, masked_entropy)),
        "entropy_mae": float(np.mean(np.abs(masked_entropy - dense_entropy))),
        "margin_mae": float(np.mean(np.abs(masked_margin - dense_margin))),
        "aurc": float(aurc_from_scores(masked_entropy, errors)),
    })
    return {key: float(value) for key, value in row.items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, seed: int, reps: int = 5000) -> dict[str, float]:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(reps, len(delta)))
    means = delta[indices].mean(axis=1)
    return {"difference": float(delta.mean()), "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)), "bootstrap_replicates": reps}


def main() -> int:
    manifest = json.loads((ROOT / "study_manifest.json").read_text(encoding="utf-8"))
    metric_rows: list[dict] = []
    pair_rows: list[dict] = []
    by_key: dict[tuple[str, str, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_key in MODELS:
        for dataset_key in DATASETS:
            score_dir = ROOT / model_key / "scores" / dataset_key
            with np.load(score_dir / "dense.npz", allow_pickle=False) as saved:
                dense = np.asarray(saved["probabilities"], dtype=np.float64)
                labels = np.asarray(saved["labels"], dtype=np.int64)
                ids = np.asarray(saved["example_ids"], dtype=np.str_)
            dense_metrics = metric_values(dense, dense, labels)
            for lam in LAMBDAS:
                for seed in SEEDS:
                    tag = f"lambda_{lam:g}_seed_{seed}"
                    path = score_dir / f"{tag}.npz"
                    with np.load(path, allow_pickle=False) as saved:
                        probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
                        if not np.array_equal(ids, np.asarray(saved["example_ids"], dtype=np.str_)):
                            raise ValueError(f"example IDs changed for {model_key}/{dataset_key}/{tag}")
                    values = metric_values(dense, probabilities, labels)
                    metric_rows.append({"model_key": model_key, "dataset_key": dataset_key,
                                        "lambda_uq": lam, "seed": seed, "examples": len(labels), **values,
                                        "dense_accuracy": dense_metrics["accuracy"], "delta_vs_dense": values["accuracy"] - dense_metrics["accuracy"]})
                    by_key[(model_key, dataset_key, lam, seed)] = (dense, probabilities, labels)
            for seed in SEEDS:
                dense0, baseline, labels0 = by_key[(model_key, dataset_key, 0.0, seed)]
                for lam in LAMBDAS[1:]:
                    _, current, _ = by_key[(model_key, dataset_key, lam, seed)]
                    per_metric = {
                        "accuracy": (current.argmax(axis=1) == labels0).astype(float) - (baseline.argmax(axis=1) == labels0).astype(float),
                        "kl": kl_divergence(dense0, current) - kl_divergence(dense0, baseline),
                        "entropy_mae": np.abs(entropy(current, normalized=True) - entropy(dense0, normalized=True)) - np.abs(entropy(baseline, normalized=True) - entropy(dense0, normalized=True)),
                        "margin_mae": np.abs(margin(current) - margin(dense0)) - np.abs(margin(baseline) - margin(dense0)),
                    }
                    for metric, delta in per_metric.items():
                        interval = paired_bootstrap(delta, np.zeros_like(delta), seed=20268000 + seed * 100 + int(lam * 100) + len(pair_rows))
                        pair_rows.append({"model_key": model_key, "dataset_key": dataset_key, "lambda_uq": lam,
                                          "seed": seed, "metric": metric, **interval})
    summary_rows = []
    for model_key in MODELS:
        for dataset_key in DATASETS:
            for lam in LAMBDAS:
                selected = [row for row in metric_rows if row["model_key"] == model_key and row["dataset_key"] == dataset_key and row["lambda_uq"] == lam]
                summary_rows.append({"model_key": model_key, "dataset_key": dataset_key, "lambda_uq": lam,
                                     "seeds": len(selected), "accuracy_mean": float(np.mean([r["accuracy"] for r in selected])),
                                     "accuracy_sd": float(np.std([r["accuracy"] for r in selected], ddof=1)),
                                     "delta_vs_dense_mean": float(np.mean([r["delta_vs_dense"] for r in selected])),
                                     "entropy_mae_mean": float(np.mean([r["entropy_mae"] for r in selected])),
                                     "entropy_mae_sd": float(np.std([r["entropy_mae"] for r in selected], ddof=1)),
                                     "kl_mean": float(np.mean([r["kl"] for r in selected])),
                                     "aurc_mean": float(np.mean([r["aurc"] for r in selected])),
                                     "ece_mean": float(np.mean([r["ece"] for r in selected]))})
    pareto = [{"model_key": r["model_key"], "dataset_key": r["dataset_key"], "lambda_uq": r["lambda_uq"],
               "seed": r["seed"], "accuracy_loss": -r["delta_vs_dense"], "entropy_mae": r["entropy_mae"]} for r in metric_rows]
    output = {"study": manifest.get("study"), "metric_rows": metric_rows, "summary_rows": summary_rows,
              "paired_lambda_vs_zero": pair_rows, "pareto_rows": pareto,
              "bootstrap_replicates": 5000, "claim_boundary": "paired quality evidence; no physical packed-speed claim"}
    (ROOT / "analysis_manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metric_rows": len(metric_rows), "summary_rows": len(summary_rows), "paired_rows": len(pair_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
