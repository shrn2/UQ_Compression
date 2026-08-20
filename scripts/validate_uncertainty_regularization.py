"""Validate the Iteration 16 uncertainty-regularization artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("runs/iter16_uncertainty_reg")


def main() -> int:
    manifest = json.loads((ROOT / "study_manifest.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    training = manifest.get("training_runs", [])
    slices = manifest.get("score_slices", [])
    check("training_run_count", len(training) == 24, f"{len(training)}/24")
    check("score_slice_count", len(slices) == 72, f"{len(slices)}/72")
    check("lambda_grid", sorted({float(r["lambda_uq"]) for r in training}) == [0.0, 0.25, 0.5, 1.0], "four lambdas")
    check("seed_grid", sorted({int(r["seed"]) for r in training}) == [1, 2, 3], "three seeds")
    check("matched_steps", all(int(r["steps"]) == 300 for r in training), "300 steps each")
    for row in training:
        path = Path(row["masks"])
        with np.load(path, allow_pickle=False) as saved:
            retained = np.asarray(saved["retained"])
            logits = np.asarray(saved["logits"])
        check(f"mask_shape:{row['model_key']}:{row['lambda_uq']}:{row['seed']}", retained.ndim == 2 and logits.ndim == 2 and retained.shape[0] == logits.shape[0], f"retained={retained.shape} logits={logits.shape}")
        check(f"mask_budget:{row['model_key']}:{row['lambda_uq']}:{row['seed']}", retained.shape[1] * 2 == logits.shape[1], f"{retained.shape[1]}/{logits.shape[1]}")
    for row in slices:
        with np.load(row["artifact"], allow_pickle=False) as saved:
            probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
            labels = np.asarray(saved["labels"])
            ids = np.asarray(saved["example_ids"])
        sums = probabilities.sum(axis=1)
        check(f"probability_normalization:{row['model_key']}:{row['dataset_key']}:{row['lambda_uq']}:{row['seed']}", np.allclose(sums, 1.0, atol=2e-4), f"max error {float(np.max(np.abs(sums - 1))):.3g}")
        check(f"labels_ids:{row['model_key']}:{row['dataset_key']}:{row['lambda_uq']}:{row['seed']}", len(labels) == len(ids), str(len(ids)))
    passed = sum(item["passed"] for item in checks)
    result = {"passed": passed, "total": len(checks), "checks": checks}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "total": len(checks)}, indent=2))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
