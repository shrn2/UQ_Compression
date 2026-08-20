"""Validate previous-cohort ablation score and mask artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("runs/iter15_ablation_previous")
MODELS = ("1p5b", "3b_nf4")
DATASETS = {"mmlu_aux": 600, "arc_challenge": 299}
METHODS = ("random_seed1", "random_seed2", "random_seed3", "one_shot", "kl_only", "kl_entropy_margin", "full_upgd")
EXPECTED_KEEP = {"1p5b": 4480, "3b_nf4": 5504}


def check(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> int:
    checks: list[dict] = []
    manifest = json.loads((ROOT / "analysis_manifest.json").read_text(encoding="utf-8"))
    check(len(manifest["rows"]) == 28, "28 model/dataset/method analysis rows", checks)
    for model_key in MODELS:
        model_masks = {}
        for method in METHODS:
            with np.load(ROOT / model_key / "masks" / f"{method}_masks.npz", allow_pickle=False) as saved:
                retained = np.asarray(saved["retained"], dtype=np.int64)
                baseline = np.asarray(saved["baseline_retained"], dtype=np.int64)
            check(retained.shape[1] == EXPECTED_KEEP[model_key], f"{model_key}/{method}: exact 50% retained width", checks)
            check(baseline.shape[1] == EXPECTED_KEEP[model_key], f"{model_key}/{method}: exact 50% baseline width", checks)
            check(np.all(np.diff(retained, axis=1) > 0), f"{model_key}/{method}: unique sorted mask", checks)
            model_masks[method] = retained
        check(np.array_equal(model_masks["one_shot"], baseline), f"{model_key}: one-shot baseline mask", checks)
        for dataset_key, expected_examples in DATASETS.items():
            score_dir = ROOT / model_key / dataset_key / "scores"
            with np.load(score_dir / "reference.npz", allow_pickle=False) as saved:
                dense = np.asarray(saved["dense"])
                baseline_probs = np.asarray(saved["baseline_aggressive"])
                labels = np.asarray(saved["labels"])
                ids = np.asarray(saved["example_ids"])
            check(dense.shape[0] == expected_examples, f"{model_key}/{dataset_key}: expected evaluation population", checks)
            check(np.allclose(dense.sum(axis=1), 1.0, atol=1e-4), f"{model_key}/{dataset_key}: dense probabilities normalized", checks)
            check(np.allclose(baseline_probs.sum(axis=1), 1.0, atol=1e-4), f"{model_key}/{dataset_key}: baseline probabilities normalized", checks)
            for method in METHODS:
                with np.load(score_dir / f"{method}.npz", allow_pickle=False) as saved:
                    probabilities = np.asarray(saved["probabilities"])
                    check(np.array_equal(ids, np.asarray(saved["example_ids"])), f"{model_key}/{dataset_key}/{method}: IDs aligned", checks)
                    check(np.array_equal(labels, np.asarray(saved["labels"])), f"{model_key}/{dataset_key}/{method}: labels aligned", checks)
                    check(np.all(np.isfinite(probabilities)), f"{model_key}/{dataset_key}/{method}: finite probabilities", checks)
                    check(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4), f"{model_key}/{dataset_key}/{method}: probabilities normalized", checks)
    output = {"datasets": list(DATASETS), "checks": checks, "passed": len(checks), "total": len(checks), "all_passed": True}
    (ROOT / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": list(DATASETS), "passed": len(checks), "total": len(checks), "all_passed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
