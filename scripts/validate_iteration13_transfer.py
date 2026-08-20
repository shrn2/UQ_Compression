"""Independently validate the two-model, four-dataset UPGD transfer run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuac.data import load_jsonl
from tuac.uncertainty_robust_pruning import _paired_accuracy_interval


ROOT = Path("runs/iter13_transfer")


def metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return json.loads(str(saved["metadata"].item()))


def main() -> int:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    datasets = ("commonsenseqa", "hellaswag", "openbookqa", "winogrande")
    manifest = json.loads((ROOT / "analysis_manifest.json").read_text(encoding="utf-8"))
    check("eight analysis rows", len(manifest) == 8)
    check("all cohorts have 500 examples", all(row["examples"] == 500 for row in manifest))
    for path in (ROOT / "datasets").glob("*.jsonl"):
        examples = load_jsonl(path)
        check(f"{path.stem} input count", len(examples) == 500)
        check(f"{path.stem} input IDs unique", len({example.id for example in examples}) == 500)
        check(f"{path.stem} input labels valid", all(example.answer is not None and 0 <= example.answer < len(example.choices) for example in examples))

    for model_index, model_key in enumerate(("1p5b", "3b_nf4")):
        model_rows = [row for row in manifest if row["model_key"] == model_key]
        check(f"{model_key} covers four cohorts", {row["dataset"] for row in model_rows} == set(datasets))
        for dataset_index, dataset in enumerate(datasets):
            reference_path = ROOT / f"{model_key}_{dataset}_reference.npz"
            upgd_path = ROOT / f"{model_key}_{dataset}_upgd.npz"
            analysis = json.loads((ROOT / f"{model_key}_{dataset}_analysis.json").read_text(encoding="utf-8"))
            with np.load(reference_path, allow_pickle=False) as reference, np.load(upgd_path, allow_pickle=False) as learned:
                dense = np.asarray(reference["dense"], dtype=np.float64)
                baseline = np.asarray(reference["baseline_aggressive"], dtype=np.float64)
                upgd = np.asarray(learned["probabilities"], dtype=np.float64)
                labels = np.asarray(reference["labels"], dtype=np.int64)
                ref_ids = np.asarray(reference["example_ids"], dtype=str)
                learned_ids = np.asarray(learned["example_ids"], dtype=str)
            check(f"{model_key} {dataset} row count", dense.shape[0] == 500 and upgd.shape[0] == 500)
            check(f"{model_key} {dataset} IDs aligned", np.array_equal(ref_ids, learned_ids) and len(set(ref_ids.tolist())) == 500)
            check(f"{model_key} {dataset} finite probabilities", np.isfinite(dense).all() and np.isfinite(baseline).all() and np.isfinite(upgd).all())
            check(f"{model_key} {dataset} normalized probabilities", all(np.allclose(values.sum(axis=1), 1.0, atol=1e-5) for values in (dense, baseline, upgd)))
            meta = metadata(reference_path)
            expected_quantization = "none" if model_key == "1p5b" else "4bit"
            check(f"{model_key} {dataset} metadata", meta["model_key"] == model_key and meta["quantization"] == expected_quantization and meta["mask_mode"] == "logical_channel_hook")
            baseline_correct = baseline.argmax(axis=1) == labels
            upgd_correct = upgd.argmax(axis=1) == labels
            seed = 20260870 + model_index * 100 + dataset_index
            interval = _paired_accuracy_interval(upgd_correct, baseline_correct, seed=seed, replicates=5000)
            saved = analysis["paired_comparison"]
            check(f"{model_key} {dataset} accuracy recomputes", abs(float(upgd_correct.mean() - baseline_correct.mean()) - saved["difference"]) < 1e-12)
            check(f"{model_key} {dataset} bootstrap recomputes", np.allclose(interval["bootstrap_95_ci"], saved["bootstrap_95_ci"], atol=1e-12))
            manifest_row = next(row for row in model_rows if row["dataset"] == dataset)
            check(f"{model_key} {dataset} manifest matches", abs(manifest_row["difference"] - saved["difference"]) < 1e-12 and manifest_row["gate"] is analysis["gate"]["passed"])

    result = {"status": "passed" if all(row["passed"] for row in checks) else "failed",
              "passed": sum(row["passed"] for row in checks), "total": len(checks), "checks": checks}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "passed", "total")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
