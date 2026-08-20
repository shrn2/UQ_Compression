"""Recompute pilot summaries and paired uncertainty from saved TUAC artifacts."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from tuac.artifacts import CalibrationArtifact
from tuac.metrics import classification_metrics

ROOT = Path("runs/pilot_arc")


def load_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def paired_comparison(left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> dict:
    left_correct = left.argmax(axis=1) == labels
    right_correct = right.argmax(axis=1) == labels
    differences = left_correct.astype(float) - right_correct.astype(float)
    rng = np.random.default_rng(20260817)
    samples = rng.integers(0, labels.size, size=(20_000, labels.size))
    bootstrap = differences[samples].mean(axis=1)
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "accuracy_difference": float(differences.mean()),
        "bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": p_value,
    }


def main() -> int:
    benchmark = load_json("benchmark/results.json")
    ablations = load_json("ablations/results.json")
    policies = load_json("policies.json")
    teacher_eval = load_json("teacher_evaluation.json")
    artifact = CalibrationArtifact.load(ROOT / "calibration.npz")
    calibration_metrics = {
        "teacher": classification_metrics(artifact.teacher_probabilities, artifact.labels),
        "student": classification_metrics(artifact.student_probabilities, artifact.labels),
    }
    records = {row["name"]: row for row in benchmark["records"]}
    records.update({row["name"]: row for row in ablations["records"]})
    main_probabilities = np.load(ROOT / "benchmark/probabilities.npz", allow_pickle=False)
    ablation_probabilities = np.load(ROOT / "ablations/probabilities.npz", allow_pickle=False)
    labels = main_probabilities["labels"]

    def probability(name: str) -> np.ndarray:
        source = ablation_probabilities if name in ablation_probabilities.files else main_probabilities
        return source[name]

    random_summaries = {}
    for budget in (4.0, 3.5, 3.0):
        rows = [row for row in benchmark["records"] if row["family"] == "random" and row["target_average_bits"] == budget]
        summary = {}
        for field in ("accuracy", "ece", "brier", "error_auroc", "aurc", "student_uncertainty_spearman"):
            values = np.asarray([row[field] for row in rows])
            summary[field] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        random_summaries[str(budget)] = summary

    comparisons = {}
    for budget in (4.0, 3.5):
        key = str(budget)
        comparisons[f"combined_vs_student_{key}"] = paired_comparison(
            probability(f"combined_{key}"), probability(f"student_only_{key}"), labels
        )
        comparisons[f"teacher_uncertainty_vs_combined_{key}"] = paired_comparison(
            probability(f"teacher_uncertainty_{key}"), probability(f"combined_{key}"), labels
        )

    allocation_diagnostics = {}
    for budget in (4.0, 3.5, 3.0):
        key = str(budget)
        mode_bits = {
            mode: policies["policies"][mode]["budgets"][key]["bits"]
            for mode in ("student_only", "combined", "teacher_uncertainty", "disagreement")
        }
        allocation_diagnostics[key] = {
            "counts": {mode: dict(sorted(Counter(bits).items())) for mode, bits in mode_bits.items()},
            "combined_vs_student_hamming": sum(a != b for a, b in zip(mode_bits["combined"], mode_bits["student_only"])),
            "uncertainty_vs_combined_hamming": sum(a != b for a, b in zip(mode_bits["teacher_uncertainty"], mode_bits["combined"])),
            "combined_equals_disagreement": mode_bits["combined"] == mode_bits["disagreement"],
        }

    selected = [
        "full_precision",
        "uniform_4.0",
        "student_only_4.0",
        "combined_4.0",
        "teacher_uncertainty_4.0",
        "student_only_3.5",
        "combined_3.5",
        "teacher_uncertainty_3.5",
        "uniform_3.0",
    ]
    table = []
    for name in selected:
        row = dict(records[name])
        correct = round(row["accuracy"] * labels.size)
        row["accuracy_wilson_95_ci"] = wilson(correct, labels.size)
        table.append(row)

    output = {
        "scope": {
            "dataset": "ARC-Challenge",
            "calibration_split": "train",
            "calibration_examples": int(artifact.labels.size),
            "evaluation_split": "validation",
            "evaluation_examples": int(labels.size),
            "teacher_model": artifact.metadata["teacher_model"],
            "student_model": artifact.metadata["student_model"],
            "probe_bits": artifact.probe_bits,
            "group_size": artifact.metadata["group_size"],
        },
        "calibration_metrics": calibration_metrics,
        "teacher_evaluation_metrics": teacher_eval,
        "main_table": table,
        "random_summaries": random_summaries,
        "paired_comparisons": comparisons,
        "allocation_diagnostics": allocation_diagnostics,
        "layer_importance_spearman": policies["diagnostics"]["student_teacher_layer_importance_spearman"],
    }
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

