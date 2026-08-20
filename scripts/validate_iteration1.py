"""Independent integrity and reproducibility checks for Iteration 1 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.artifacts import MultiBitCalibrationArtifact
from tuac.data import load_jsonl
from tuac.iteration import POLICY_MODES, perturbations
from tuac.metrics import classification_metrics

ROOT = Path("runs/iter1_arc")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def probabilities_valid(values: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.allclose(values.sum(axis=-1), 1.0, atol=1e-5)
    )


def main() -> int:
    calibration = load_jsonl(ROOT / "calibration_pool.jsonl")
    evaluation = load_jsonl(ROOT / "evaluation.jsonl")
    artifact = MultiBitCalibrationArtifact.load(ROOT / "calibration.npz")
    policies = json.loads((ROOT / "policies.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    report_artifact = json.loads((ROOT / "report/artifact.json").read_text(encoding="utf-8"))
    calibration_ids = {item.id for item in calibration}
    evaluation_ids = {item.id for item in evaluation}
    delta = perturbations(artifact)

    checks: dict[str, bool] = {
        "calibration_pool_count_750": len(calibration) == 750,
        "evaluation_full_validation_count_299": len(evaluation) == 299,
        "split_ids_disjoint": calibration_ids.isdisjoint(evaluation_ids),
        "artifact_shape_3bits_24layers_750examples": artifact.quantized_probabilities.shape[:3] == (3, 24, 750),
        "candidate_bits_are_2_4_8": artifact.probe_bits == (2, 4, 8),
        "teacher_probabilities_valid": probabilities_valid(artifact.teacher_probabilities),
        "student_probabilities_valid": probabilities_valid(artifact.student_probabilities),
        "probe_probabilities_valid": probabilities_valid(artifact.quantized_probabilities),
        "five_calibration_seeds": len(policies["calibration_seeds"]) == 5,
        "report_has_four_charts": len(report_artifact["manifest"]["charts"]) == 4,
        "report_has_required_table": len(report_artifact["manifest"]["tables"]) == 1,
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "decision_is_no_go": summary["go_no_go"]["decision"] == "NO-GO",
    }
    for seed in policies["calibration_seeds"]:
        row = policies["seeds"][str(seed)]
        indices = row["subset_indices"]
        checks[f"seed_{seed}_has_500_unique_examples"] = len(indices) == len(set(indices)) == 500
        checks[f"seed_{seed}_has_meaningful_teacher_gap"] = row["teacher_student_accuracy_gap"] >= 0.05
        checks[f"seed_{seed}_has_oracle_cases"] = row["diagnostics"]["teacher_correct_student_wrong"] >= 50
        checks[f"seed_{seed}_has_all_policies"] = set(row["policies"]) == set(POLICY_MODES)

    mean_by_bit = delta.mean(axis=(1, 2))
    checks["empirical_error_decreases_with_precision"] = bool(
        mean_by_bit[0] > mean_by_bit[1] > mean_by_bit[2]
    )

    with np.load(ROOT / "benchmark/probabilities.npz", allow_pickle=False) as probabilities:
        labels = probabilities["labels"]
        checks["benchmark_label_count"] = labels.shape == (299,)
        checks["teacher_evaluation_probabilities_valid"] = probabilities_valid(probabilities["teacher"])
        checks["fp_evaluation_probabilities_valid"] = probabilities_valid(probabilities["full_precision"])
        recomputed_teacher = classification_metrics(probabilities["teacher"], labels)
        recomputed_fp = classification_metrics(probabilities["full_precision"], labels)
        checks["teacher_metrics_recomputed"] = all(
            np.isclose(recomputed_teacher[key], benchmark["teacher_metrics"][key], equal_nan=True)
            for key in recomputed_teacher
        )
        checks["fp_metrics_recomputed"] = all(
            np.isclose(recomputed_fp[key], benchmark["full_precision_metrics"][key], equal_nan=True)
            for key in recomputed_fp
        )
        for row in benchmark["records"]:
            values = probabilities[row["probability_key"]]
            checks[f"probabilities_valid_{row['name']}"] = probabilities_valid(values)
            recomputed = classification_metrics(values, labels)
            checks[f"accuracy_recomputed_{row['name']}"] = np.isclose(
                recomputed["accuracy"], row["accuracy"]
            )
            checks[f"budget_respected_{row['name']}"] = (
                row["actual_average_bits"] <= row["target_average_bits"] + 1e-12
            )
            if row["target_average_bits"] == 3.0:
                checks[f"three_bit_point_is_mixed_{row['name']}"] = len(set(row["bits"])) > 1

    for budget in policies["budgets"]:
        rows = [
            row
            for row in benchmark["records"]
            if row["family"] == "random" and row["target_average_bits"] == budget
        ]
        checks[f"five_random_baselines_{budget}"] = len(rows) == 5
        checks[f"student_comparison_saved_{budget}"] = (
            f"teacher_uncertainty_vs_student_only:{budget}" in benchmark["comparisons"]
        )
        checks[f"random_comparison_saved_{budget}"] = (
            f"teacher_uncertainty_vs_random:{budget}" in benchmark["comparisons"]
        )

    key_files = [
        ROOT / "calibration_pool.jsonl",
        ROOT / "evaluation.jsonl",
        ROOT / "calibration.npz",
        ROOT / "policies.json",
        ROOT / "benchmark/results.json",
        ROOT / "benchmark/probabilities.npz",
        ROOT / "summary.json",
        ROOT / "report/artifact.json",
        ROOT / "report/report.html",
    ]
    refs = {}
    for model in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"):
        ref = Path(".cache/huggingface/hub") / f"models--Qwen--{model}" / "refs/main"
        refs[model] = ref.read_text(encoding="utf-8").strip() if ref.exists() else None

    checks = {key: bool(value) for key, value in checks.items()}

    output = {
        "assessment": "share_with_caveats" if all(checks.values()) else "needs_revision",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_revisions": refs,
        "mean_probe_kl_by_bit": {
            str(bit): float(value) for bit, value in zip(artifact.probe_bits, mean_by_bit)
        },
        "report_package_verification": "structural_only",
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in key_files},
        "material_caveats": [
            "Only ARC-Challenge and one teacher/student model family were evaluated.",
            "Fake quantization does not establish packed storage, memory, latency, or throughput gains.",
            "Calibration subsets overlap within one 750-example training pool.",
            "The runtime-limited {2,4,8} candidate set has no uniform 3-bit baseline.",
            "Portable report packaging passed structurally, but browser QA did not run because Chromium is unavailable.",
        ],
    }
    (ROOT / "validation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "assessment": output["assessment"],
                "checks": len(checks),
                "passed": sum(checks.values()),
            },
            indent=2,
        )
    )
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
