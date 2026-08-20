"""Independently validate the larger-model 4-bit UPGD benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.uncertainty_robust_pruning import _paired_accuracy_interval


ROOT = Path("runs/iter12_q4_3b")


def metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return json.loads(str(saved["metadata"].item()))


def main() -> int:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    calibration = metadata(ROOT / "calibration.npz")
    training = json.loads((ROOT / "training.json").read_text(encoding="utf-8"))
    check("calibration is 3B NF4", calibration["model"] == "Qwen/Qwen2.5-3B-Instruct" and calibration["quantization"] == "4bit")
    check("calibration uses 2400 examples", calibration["examples"] == 2400)
    check("calibration strata balanced", calibration["stratum_counts"] == [800, 800, 800])
    check("training is 3B NF4", training["model"] == "Qwen/Qwen2.5-3B-Instruct" and training["quantization"] == "4bit")
    check("resource-adapted training recorded", training["steps"] == 300 and training["seed"] == 20260839)
    check("weights frozen", training["weights_frozen"] is True)
    check("exact 50 percent 3B mask", training["keep_channels_per_layer"] == 5504 and training["sparsity"] == 0.5)
    check("mask movement is material", training["retained_overlap_with_initial_baseline"] < 0.90)
    check("training completed in finite time", np.isfinite(training["training_seconds"]) and training["training_seconds"] > 0)

    with np.load(ROOT / "masks.npz", allow_pickle=False) as saved:
        retained = np.asarray(saved["retained"], dtype=np.int64)
        baseline = np.asarray(saved["baseline_retained"], dtype=np.int64)
        logits = np.asarray(saved["logits"])
    check("36 layers", retained.shape[0] == 36)
    check("3B FFN width", retained.shape == (36, 5504) and logits.shape == (36, 11008))
    check("baseline shape matches", baseline.shape == retained.shape)
    check("unique valid retained indices", all(np.unique(row).size == 5504 and row.min() >= 0 and row.max() < 11008 for row in retained))
    overlap = float(np.mean([np.intersect1d(a, b).size / a.size for a, b in zip(retained, baseline)]))
    check("mask overlap recomputes", abs(overlap - training["retained_overlap_with_initial_baseline"]) < 1e-12)

    for dataset, n, expected_gate, seed in (("arc", 299, True, 20260842), ("mmlu_pro", 1000, False, 20260844)):
        reference_path = ROOT / f"{dataset}_reference.npz"
        upgd_path = ROOT / f"{dataset}_upgd.npz"
        analysis = json.loads((ROOT / f"{dataset}_analysis.json").read_text(encoding="utf-8"))
        with np.load(reference_path, allow_pickle=False) as reference, np.load(upgd_path, allow_pickle=False) as learned:
            dense = np.asarray(reference["dense"], dtype=np.float64)
            baseline_prob = np.asarray(reference["baseline_aggressive"], dtype=np.float64)
            upgd_prob = np.asarray(learned["probabilities"], dtype=np.float64)
            labels = np.asarray(reference["labels"], dtype=np.int64)
            ref_ids = np.asarray(reference["example_ids"], dtype=str)
            learned_ids = np.asarray(learned["example_ids"], dtype=str)
        check(f"{dataset} row count", dense.shape[0] == n and upgd_prob.shape[0] == n)
        check(f"{dataset} IDs aligned", np.array_equal(ref_ids, learned_ids) and len(set(ref_ids.tolist())) == n)
        check(f"{dataset} finite probabilities", np.isfinite(dense).all() and np.isfinite(baseline_prob).all() and np.isfinite(upgd_prob).all())
        check(f"{dataset} normalized probabilities", all(np.allclose(values.sum(axis=1), 1.0, atol=1e-5) for values in (dense, baseline_prob, upgd_prob)))
        baseline_correct = baseline_prob.argmax(axis=1) == labels
        upgd_correct = upgd_prob.argmax(axis=1) == labels
        interval = _paired_accuracy_interval(upgd_correct, baseline_correct, seed=seed, replicates=5000)
        saved = analysis["paired_comparison"]
        check(f"{dataset} accuracy recomputes", abs(float(upgd_correct.mean() - baseline_correct.mean()) - saved["difference"]) < 1e-12)
        check(f"{dataset} bootstrap recomputes", np.allclose(interval["bootstrap_95_ci"], saved["bootstrap_95_ci"], atol=1e-12))
        check(f"{dataset} gate recomputes", analysis["gate"]["passed"] is expected_gate)
        ref_rows = {row["condition"]: row for row in analysis["metric_rows"]}
        check(f"{dataset} baseline metric accuracy", abs(float(baseline_correct.mean()) - ref_rows["baseline_aggressive"]["accuracy"]) < 1e-12)
        check(f"{dataset} UPGD metric accuracy", abs(float(upgd_correct.mean()) - ref_rows["upgd_aggressive"]["accuracy"]) < 1e-12)

    manifest = json.loads((ROOT / "scoring_manifest.json").read_text(encoding="utf-8"))
    check("scoring manifest covers two datasets", {row["dataset"] for row in manifest} == {"arc", "mmlu_pro"})
    check("scoring manifest quantized", all(row["quantization"] == "4bit" and row["mask_mode"] == "logical_channel_hook" for row in manifest))

    report_artifact_path = ROOT / "report" / "artifact.json"
    report_html_path = ROOT / "report" / "report.html"
    chart_map_path = ROOT / "report" / "chart_map.json"
    report_ok = False
    if report_artifact_path.exists():
        report = json.loads(report_artifact_path.read_text(encoding="utf-8"))
        manifest_obj = report.get("manifest", {})
        blocks = report.get("manifest", {}).get("blocks", [])
        block_ids = {block.get("id") for block in blocks}
        report_ok = (
            report.get("surface") == "report"
            and manifest_obj.get("surface") == "report"
            and len(manifest_obj.get("charts", [])) == 2
            and len(manifest_obj.get("tables", [])) == 1
            and {"summary", "accuracy-chart", "strata-chart", "metrics-table", "limitations"}.issubset(block_ids)
        )
    check("report artifact is complete", report_ok)
    check("portable report package exists", report_html_path.exists() and report_html_path.stat().st_size > 0)
    check("chart map exists", chart_map_path.exists() and chart_map_path.stat().st_size > 0)

    result = {"status": "passed" if all(row["passed"] for row in checks) else "failed",
              "passed": sum(row["passed"] for row in checks), "total": len(checks), "checks": checks}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "passed", "total")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
