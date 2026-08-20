"""Independently validate Iteration 11 masks, evaluations, and reported claims."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.uncertainty_robust_pruning import _paired_accuracy_interval


ROOT = Path("runs/iter11_upgd")


def _metadata(saved: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(saved["metadata"].item()))


def _accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(probabilities, axis=1) == labels))


def main() -> int:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with np.load(ROOT / "masks.npz", allow_pickle=False) as masks:
        retained = masks["retained"]
        baseline = masks["baseline_retained"]
        logits = masks["logits"]
        metadata = _metadata(masks)
    check("28 transformer layers", retained.shape[0] == 28)
    check("exact 50% channel retention", retained.shape == (28, 4480))
    check("baseline mask shape matches", baseline.shape == retained.shape)
    check("gate logits cover every FFN channel", logits.shape == (28, 8960))
    check("each learned layer has unique valid indices", all(
        np.unique(row).size == 4480 and row.min() >= 0 and row.max() < 8960 for row in retained
    ))
    check("weights frozen", metadata["weights_frozen"] is True)
    check("single predeclared training run", metadata["steps"] == 600 and metadata["seed"] == 20260829)
    overlap = float(np.mean([np.intersect1d(a, b).size / a.size for a, b in zip(retained, baseline)]))
    check("reported mask overlap recomputes", abs(overlap - metadata["retained_overlap_with_initial_baseline"]) < 1e-12)
    check("learned mask materially differs", overlap < 0.90, f"overlap={overlap:.6f}")

    def validate_artifact(path: Path, expected_n: int, expected_choices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as saved:
            probs = saved["probabilities"].astype(np.float64)
            labels = saved["labels"].astype(np.int64)
            ids = saved["example_ids"].astype(str)
            artifact_metadata = _metadata(saved)
        check(f"{path.name} shape", probs.shape == (expected_n, expected_choices))
        check(f"{path.name} finite", np.isfinite(probs).all())
        check(f"{path.name} normalized", np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))
        check(f"{path.name} labels", labels.shape == (expected_n,) and np.all((labels >= 0) & (labels < expected_choices)))
        check(f"{path.name} unique ids", len(set(ids.tolist())) == expected_n)
        check(f"{path.name} physical pruning", artifact_metadata["physical_channel_pruning"] is True)
        return probs, labels, ids

    upgd_dev, dev_labels, dev_ids = validate_artifact(ROOT / "development_evaluation.npz", 600, 4)
    upgd_ext, ext_labels, ext_ids = validate_artifact(ROOT / "mmlu_pro_upgd.npz", 1000, 10)
    base_ext, ext_base_labels, ext_base_ids = validate_artifact(ROOT / "mmlu_pro_baseline.npz", 1000, 10)

    with np.load("runs/iter10_ursa/development_evaluation.npz", allow_pickle=False) as reference:
        dense_dev = reference["dense"].astype(np.float64)
        base_dev = reference["baseline_aggressive"].astype(np.float64)
        ref_labels = reference["labels"].astype(np.int64)
        ref_ids = reference["example_ids"].astype(str)
        compute_proxy = _metadata(reference)["compute"]["aggressive"]["normalized_linear_flops"]
    with np.load("runs/iter9_mmlu_pro/evaluation.npz", allow_pickle=False) as reference:
        dense_ext = reference["dense"].astype(np.float64)
        ref_ext_labels = reference["labels"].astype(np.int64)
        ref_ext_ids = reference["example_ids"].astype(str)

    check("development cohort alignment", np.array_equal(dev_labels, ref_labels) and np.array_equal(dev_ids, ref_ids))
    check("external baseline cohort alignment", np.array_equal(ext_base_labels, ref_ext_labels) and np.array_equal(ext_base_ids, ref_ext_ids))
    check("external UPGD cohort alignment", np.array_equal(ext_labels, ref_ext_labels) and np.array_equal(ext_ids, ref_ext_ids))
    check("analytical compute proxy", abs(compute_proxy - 0.5588235294117647) < 1e-12)

    for filename, dense, base, learned, labels, seed in (
        ("development_analysis.json", dense_dev, base_dev, upgd_dev, dev_labels, 20260830),
        ("mmlu_pro_analysis.json", dense_ext, base_ext, upgd_ext, ext_labels, 20260832),
    ):
        analysis = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        rows = {row["condition"]: row for row in analysis["metric_rows"]}
        check(f"{filename} dense accuracy", abs(_accuracy(dense, labels) - rows["dense"]["accuracy"]) < 1e-12)
        check(f"{filename} baseline accuracy", abs(_accuracy(base, labels) - rows["baseline_aggressive"]["accuracy"]) < 1e-12)
        check(f"{filename} UPGD accuracy", abs(_accuracy(learned, labels) - rows["upgd_aggressive"]["accuracy"]) < 1e-12)
        baseline_correct = np.argmax(base, axis=1) == labels
        learned_correct = np.argmax(learned, axis=1) == labels
        interval = _paired_accuracy_interval(
            learned_correct, baseline_correct, seed=seed, replicates=5000
        )
        saved = analysis["paired_comparison"]
        check(f"{filename} paired difference", abs(interval["difference"] - saved["difference"]) < 1e-12)
        check(f"{filename} paired interval", np.allclose(interval["bootstrap_95_ci"], saved["bootstrap_95_ci"], atol=1e-12))
        check(f"{filename} gate is conservative", analysis["gate"]["passed"] is False)

    report_artifact_path = ROOT / "report" / "artifact.json"
    report_html_path = ROOT / "report" / "report.html"
    check("report artifact exists", report_artifact_path.exists())
    check("portable report exists", report_html_path.exists())
    if report_artifact_path.exists() and report_html_path.exists():
        report_artifact = json.loads(report_artifact_path.read_text(encoding="utf-8"))
        report_html = report_html_path.read_text(encoding="utf-8")
        check("report surface", report_artifact["surface"] == "report")
        check("report snapshot ready", report_artifact["snapshot"]["status"] == "ready")
        check("report chart and table counts", len(report_artifact["manifest"]["charts"]) == 3 and len(report_artifact["manifest"]["tables"]) == 2)
        check("report technical sections", all(section in report_html for section in (
            "Technical summary", "Scope, data, and metric definitions", "Methodology",
            "Limitations, uncertainty, and robustness checks", "Recommended next steps", "Further questions",
        )))
        check("semantic fallback embedded", "portable-fallback" in report_html)

    result = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "passed": sum(item["passed"] for item in checks),
        "total": len(checks),
        "checks": checks,
    }
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "passed", "total")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
