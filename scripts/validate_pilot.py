"""Independent integrity checks for the saved ARC pilot artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from tuac.artifacts import CalibrationArtifact
from tuac.data import load_jsonl
from tuac.metrics import classification_metrics

ROOT = Path("runs/pilot_arc")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def probability_checks(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values >= 0) and np.allclose(values.sum(axis=-1), 1.0, atol=1e-5))


def main() -> int:
    calibration_examples = load_jsonl(ROOT / "calibration.jsonl")
    evaluation_examples = load_jsonl(ROOT / "evaluation.jsonl")
    calibration_ids = {item.id for item in calibration_examples}
    evaluation_ids = {item.id for item in evaluation_examples}
    artifact = CalibrationArtifact.load(ROOT / "calibration.npz")
    benchmark = json.loads((ROOT / "benchmark/results.json").read_text(encoding="utf-8"))
    ablations = json.loads((ROOT / "ablations/results.json").read_text(encoding="utf-8"))
    teacher_metrics = json.loads((ROOT / "teacher_evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))

    checks: dict[str, bool] = {
        "calibration_count_32": len(calibration_examples) == 32,
        "evaluation_count_64": len(evaluation_examples) == 64,
        "split_ids_disjoint": calibration_ids.isdisjoint(evaluation_ids),
        "artifact_has_24_layers": artifact.quantized_probabilities.shape == (24, 32, 4),
        "teacher_probabilities_valid": probability_checks(artifact.teacher_probabilities),
        "student_probabilities_valid": probability_checks(artifact.student_probabilities),
        "probe_probabilities_valid": probability_checks(artifact.quantized_probabilities),
        "report_html_exists": (ROOT / "report/report.html").is_file(),
        "report_artifact_exists": (ROOT / "report/artifact.json").is_file(),
    }

    for folder, result in (("benchmark", benchmark), ("ablations", ablations)):
        with np.load(ROOT / folder / "probabilities.npz", allow_pickle=False) as probabilities:
            labels = probabilities["labels"]
            checks[f"{folder}_label_count"] = labels.shape == (64,)
            records = {row["name"]: row for row in result["records"]}
            for name in probabilities.files:
                if name in ("labels", "example_ids"):
                    continue
                checks[f"{folder}_{name}_probabilities_valid"] = probability_checks(probabilities[name])
                recomputed = classification_metrics(probabilities[name], labels)
                if name in records:
                    checks[f"{folder}_{name}_accuracy_recomputed"] = abs(recomputed["accuracy"] - records[name]["accuracy"]) < 1e-12

    with np.load(ROOT / "teacher_evaluation.npz", allow_pickle=False) as teacher:
        recomputed_teacher = classification_metrics(teacher["probabilities"], teacher["labels"])
    checks["teacher_accuracy_recomputed"] = abs(recomputed_teacher["accuracy"] - teacher_metrics["accuracy"]) < 1e-12

    for row in benchmark["records"] + ablations["records"]:
        if row["family"] != "full_precision":
            key = f"budget_respected_{row['name']}"
            checks[key] = row["actual_average_bits"] <= row["target_average_bits"] + 1e-12

    with np.load(ROOT / "benchmark/probabilities.npz", allow_pickle=False) as probabilities:
        uniform_three = probabilities["uniform_3.0"]
        checks["three_bit_policies_degenerate"] = all(
            np.array_equal(uniform_three, probabilities[name])
            for name in (
                "random_3.0_seed0", "random_3.0_seed1", "random_3.0_seed2",
                "student_only_3.0", "combined_3.0",
            )
        )

    refs = {}
    for model in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"):
        ref = Path(".cache/huggingface/hub") / f"models--Qwen--{model}" / "refs/main"
        refs[model] = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    key_files = [
        ROOT / "calibration.jsonl",
        ROOT / "evaluation.jsonl",
        ROOT / "calibration.npz",
        ROOT / "policies.json",
        ROOT / "benchmark/results.json",
        ROOT / "ablations/results.json",
        ROOT / "summary.json",
        ROOT / "report/artifact.json",
        ROOT / "report/report.html",
    ]
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
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in key_files},
        "material_caveats": [
            "32 calibration and 64 evaluation examples are below the planned sample sizes.",
            "Only ARC-Challenge was evaluated.",
            "Fake quantization does not establish packed storage or runtime gains.",
            "The 3.0-bit point is degenerate because every candidate policy selects the minimum width.",
            "Browser-level report verification failed; structural packaging verification passed.",
        ],
        "summary_cross_check": {
            "teacher_eval_accuracy": summary["teacher_evaluation_metrics"]["accuracy"],
            "combined_vs_student_4bit": summary["paired_comparisons"]["combined_vs_student_4.0"],
        },
    }
    (ROOT / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assessment": output["assessment"], "checks": len(checks), "passed": sum(checks.values())}, indent=2))
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

