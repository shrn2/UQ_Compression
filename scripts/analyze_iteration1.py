"""Create a compact, source-backed summary for Iteration 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.iteration import perturbations
from tuac.metrics import spearman_correlation
from tuac.scoring import entropy

ROOT = Path("runs/iter1_arc")
POLICY_LABELS = {
    "student_only": "Student-only",
    "teacher_confidence": "Teacher confidence",
    "teacher_uncertainty": "Teacher uncertainty",
    "disagreement": "Teacher-student disagreement",
    "oracle": "Oracle teacher-correct/student-wrong",
    "random": "Random mixed precision",
    "uniform": "Uniform quantization",
}


def main() -> int:
    artifact = MultiBitCalibrationArtifact.load(ROOT / "calibration.npz")
    policies = json.loads((ROOT / "policies.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    delta = perturbations(artifact)
    bit_index = artifact.probe_bits.index(4)
    uncertainty = entropy(artifact.teacher_probabilities, normalized=True)
    teacher_correct = artifact.teacher_probabilities.argmax(axis=1) == artifact.labels
    student_correct = artifact.student_probabilities.argmax(axis=1) == artifact.labels
    example_perturbation = delta[bit_index].mean(axis=0)

    relation_rows = [
        {
            "example_id": artifact.example_ids[index],
            "teacher_entropy": float(uncertainty[index]),
            "mean_4bit_kl": float(example_perturbation[index]),
            "teacher_correct": bool(teacher_correct[index]),
            "student_correct": bool(student_correct[index]),
            "oracle_case": bool(teacher_correct[index] and not student_correct[index]),
        }
        for index in range(artifact.labels.size)
    ]

    layer_interaction_rows = []
    importance_rows = []
    seed_rows = []
    seeds = policies["calibration_seeds"]
    for layer_index, layer in enumerate(artifact.layer_names):
        high_minus_low = [
            policies["seeds"][str(seed)]["rq1"]["4"][layer_index]["high_minus_low"]
            for seed in seeds
        ]
        correlations = [
            policies["seeds"][str(seed)]["rq1"]["4"][layer_index]["spearman"]
            for seed in seeds
        ]
        student_importance = [
            policies["seeds"][str(seed)]["policies"]["student_only"]["ranking"][layer_index]
            for seed in seeds
        ]
        uncertainty_importance = [
            policies["seeds"][str(seed)]["policies"]["teacher_uncertainty"]["ranking"][layer_index]
            for seed in seeds
        ]
        layer_interaction_rows.append(
            {
                "layer": layer.replace("model.layers.", "L"),
                "layer_index": layer_index,
                "high_minus_low_4bit_kl": float(np.mean(high_minus_low)),
                "high_minus_low_std": float(np.std(high_minus_low, ddof=1)),
                "entropy_perturbation_spearman": float(np.mean(correlations)),
            }
        )
        importance_rows.append(
            {
                "layer": layer.replace("model.layers.", "L"),
                "layer_index": layer_index,
                "student_importance": float(np.mean(student_importance)),
                "teacher_uncertainty_importance": float(np.mean(uncertainty_importance)),
            }
        )

    for seed in seeds:
        row = policies["seeds"][str(seed)]
        seed_rows.append(
            {
                "calibration_seed": seed,
                "examples": policies["subset_size"],
                "teacher_accuracy": row["teacher_metrics"]["accuracy"],
                "student_accuracy": row["student_metrics"]["accuracy"],
                "teacher_student_gap": row["teacher_student_accuracy_gap"],
                "oracle_cases": row["diagnostics"]["teacher_correct_student_wrong"],
                "uncertainty_vs_student_rho": row["policies"]["teacher_uncertainty"]["ranking_vs_student_spearman"],
                "uncertainty_changed_blocks_4bit": row["policies"]["teacher_uncertainty"]["allocations"]["4.0"]["changed_vs_student"],
                "uncertainty_changed_blocks_3_5bit": row["policies"]["teacher_uncertainty"]["allocations"]["3.5"]["changed_vs_student"],
                "uncertainty_changed_blocks_3bit": row["policies"]["teacher_uncertainty"]["allocations"]["3.0"]["changed_vs_student"],
            }
        )

    accuracy_rows = []
    metric_rows = []
    for budget in policies["budgets"]:
        for family in (*POLICY_LABELS,):
            key = f"{family}:{budget}"
            if key not in benchmark["summaries"]:
                continue
            summary = benchmark["summaries"][key]
            accuracy_rows.append(
                {
                    "method": POLICY_LABELS[family],
                    "family": family,
                    "average_bits": budget,
                    "accuracy": summary["accuracy"]["mean"],
                    "accuracy_std": summary["accuracy"]["std"],
                }
            )
            metric_rows.append(
                {
                    "method": POLICY_LABELS[family],
                    "family": family,
                    "average_bits": budget,
                    "accuracy_mean": summary["accuracy"]["mean"],
                    "accuracy_std": summary["accuracy"]["std"],
                    "ece_mean": summary["ece"]["mean"],
                    "brier_mean": summary["brier"]["mean"],
                    "error_auroc_mean": summary["error_auroc"]["mean"],
                    "aurc_mean": summary["aurc"]["mean"],
                    "uq_spearman_mean": summary["student_uncertainty_spearman"]["mean"],
                }
            )

    proposed_vs_student = {
        str(budget): benchmark["comparisons"][f"teacher_uncertainty_vs_student_only:{budget}"]
        for budget in policies["budgets"]
    }
    proposed_vs_random = {
        str(budget): benchmark["comparisons"][f"teacher_uncertainty_vs_random:{budget}"]
        for budget in policies["budgets"]
    }
    allocation_rhos = [row["uncertainty_vs_student_rho"] for row in seed_rows]
    oracle_advantage = {
        str(budget): benchmark["summaries"][f"oracle:{budget}"]["accuracy"]["mean"]
        - benchmark["summaries"][f"student_only:{budget}"]["accuracy"]["mean"]
        for budget in policies["budgets"]
    }
    go_no_go = {
        "decision": "NO-GO",
        "criteria": {
            "different_ranking": {
                "passed": float(np.mean(allocation_rhos)) < 0.9,
                "mean_teacher_uncertainty_vs_student_rho": float(np.mean(allocation_rhos)),
            },
            "stable_ranking": {
                "passed": policies["stability"]["teacher_uncertainty"]["pairwise_ranking_spearman_min"] >= 0.8,
                **policies["stability"]["teacher_uncertainty"],
            },
            "clear_accuracy_or_reliability_advantage": {
                "passed": False,
                "reason": "Paired bootstrap intervals versus student-only cross zero at both mixed-precision budgets; 4-bit allocations are identical.",
            },
            "exceeds_random_variance": {
                "passed": False,
                "reason": "Proposed-versus-random paired bootstrap intervals cross zero at 3.5 and 3.0 bits.",
            },
            "oracle_advantage": {
                "passed": all(value > 0 for value in oracle_advantage.values()),
                "accuracy_difference_vs_student": oracle_advantage,
            },
        },
        "rationale": (
            "Teacher uncertainty changes the ranking, but that change produces no 4-bit allocation difference, "
            "no clear held-out advantage at mixed budgets, and no oracle advantage. The observed gains are within random-policy variance."
        ),
        "recommended_pivot": "Quantization-induced uncertainty distortion and delegation drift in SLM-to-LLM cascades.",
    }

    bit_sensitivity = []
    for index, bit in enumerate(artifact.probe_bits):
        values = delta[index]
        bit_sensitivity.append(
            {
                "bits": bit,
                "mean_kl": float(values.mean()),
                "median_kl": float(np.median(values)),
                "p95_kl": float(np.quantile(values, 0.95)),
            }
        )

    output = {
        "scope": {
            "dataset": "ARC-Challenge",
            "calibration_pool": artifact.labels.size,
            "calibration_subset_per_seed": policies["subset_size"],
            "calibration_seeds": seeds,
            "evaluation_examples": benchmark["example_count"],
            "teacher_model": benchmark["teacher_model"],
            "student_model": benchmark["student_model"],
            "probe_bits": list(artifact.probe_bits),
            "budgets": policies["budgets"],
            "group_size": artifact.metadata["group_size"],
            "quantization": artifact.metadata["quantization"],
        },
        "reference_metrics": {
            "teacher": benchmark["teacher_metrics"],
            "full_precision_student": benchmark["full_precision_metrics"],
        },
        "teacher_uncertainty_vs_student": proposed_vs_student,
        "teacher_uncertainty_vs_random": proposed_vs_random,
        "global_teacher_entropy_vs_mean_4bit_perturbation_spearman": spearman_correlation(
            uncertainty, example_perturbation
        ),
        "seed_diagnostics": seed_rows,
        "bit_sensitivity": bit_sensitivity,
        "relation_rows": relation_rows,
        "layer_interaction_rows": layer_interaction_rows,
        "importance_rows": importance_rows,
        "accuracy_rows": accuracy_rows,
        "metric_rows": metric_rows,
        "go_no_go": go_no_go,
    }
    (ROOT / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": go_no_go["decision"], "global_rho": output["global_teacher_entropy_vs_mean_4bit_perturbation_spearman"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
