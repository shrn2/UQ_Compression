"""Validate the ablation matrix's masks, score artifacts, and objective metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("runs/iter14_ablation_hellaswag")
MODELS = ("1p5b", "3b_nf4")
EXPECTED_KEEP = {"1p5b": 4480, "3b_nf4": 5504}
METHODS = ("random_seed1", "random_seed2", "random_seed3", "one_shot", "kl_only", "kl_entropy_margin", "full_upgd")


def check(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> int:
    checks: list[dict] = []
    manifest = json.loads((ROOT / "analysis_manifest.json").read_text(encoding="utf-8"))
    check(len(manifest["rows"]) == 14, "14 model/method analysis rows", checks)
    expected_weights = {
        "kl_only": (0.0, 0.0, 0.0),
        "kl_entropy_margin": (1.0, 0.5, 0.0),
    }
    random_seeds: dict[str, set[int]] = {key: set() for key in MODELS}
    for model_key in MODELS:
        model_dir = ROOT / model_key
        with np.load(model_dir / "scores" / "reference.npz", allow_pickle=False) as saved:
            dense = np.asarray(saved["dense"])
            baseline = np.asarray(saved["baseline_aggressive"])
            labels = np.asarray(saved["labels"])
            ids = np.asarray(saved["example_ids"])
        check(dense.shape == baseline.shape and dense.shape[0] == 500, f"{model_key}: 500 paired reference rows", checks)
        check(np.allclose(dense.sum(axis=1), 1.0, atol=1e-4), f"{model_key}: dense probabilities normalized", checks)
        check(np.allclose(baseline.sum(axis=1), 1.0, atol=1e-4), f"{model_key}: baseline probabilities normalized", checks)
        masks: dict[str, np.ndarray] = {}
        for method in METHODS:
            mask_path = model_dir / "masks" / f"{method}_masks.npz"
            with np.load(mask_path, allow_pickle=False) as saved:
                retained = np.asarray(saved["retained"], dtype=np.int64)
                baseline_retained = np.asarray(saved["baseline_retained"], dtype=np.int64)
                metadata = json.loads(str(saved["metadata"]))
            check(retained.ndim == 2 and retained.shape == baseline_retained.shape, f"{model_key}/{method}: mask shape", checks)
            check(np.all(np.sum(np.diff(retained, axis=1) <= 0, axis=1) == 0), f"{model_key}/{method}: retained indices sorted", checks)
            check(np.all(np.diff(retained, axis=1) > 0), f"{model_key}/{method}: no duplicate retained indices", checks)
            check(retained.shape[1] == EXPECTED_KEEP[model_key], f"{model_key}/{method}: exact 50% retained width", checks)
            check(baseline_retained.shape[1] == EXPECTED_KEEP[model_key], f"{model_key}/{method}: exact 50% baseline width", checks)
            masks[method] = retained
            if method.startswith("random_"):
                random_seeds[model_key].add(int(metadata["seed"]))
            if method in expected_weights:
                training = json.loads((model_dir / f"{method}_training.json").read_text(encoding="utf-8"))
                expected_entropy, expected_margin, expected_robust = expected_weights[method]
                check(float(training["entropy_weight"]) == expected_entropy, f"{model_key}/{method}: entropy weight", checks)
                check(float(training["margin_weight"]) == expected_margin, f"{model_key}/{method}: margin weight", checks)
                check(float(training["robust_weight"]) == expected_robust, f"{model_key}/{method}: robust weight", checks)
                check(training["objective_name"] == method, f"{model_key}/{method}: objective name", checks)
                check(bool(training.get("model_frozen", True)), f"{model_key}/{method}: model frozen metadata", checks)
            with np.load(model_dir / "scores" / f"{method}.npz", allow_pickle=False) as scored:
                probabilities = np.asarray(scored["probabilities"])
                check(np.array_equal(ids, np.asarray(scored["example_ids"])), f"{model_key}/{method}: example IDs aligned", checks)
                check(np.array_equal(labels, np.asarray(scored["labels"])), f"{model_key}/{method}: labels aligned", checks)
                check(np.all(np.isfinite(probabilities)), f"{model_key}/{method}: finite probabilities", checks)
                check(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4), f"{model_key}/{method}: probabilities normalized", checks)
        check(len(random_seeds[model_key]) == 3, f"{model_key}: three unique random seeds", checks)
        with np.load(model_dir / "masks" / "one_shot_masks.npz", allow_pickle=False) as saved:
            check(np.array_equal(masks["one_shot"], np.asarray(saved["baseline_retained"])), f"{model_key}: one-shot equals baseline mask", checks)
    output = {"dataset": "HellaSwag", "checks": checks, "passed": len(checks), "total": len(checks), "all_passed": True}
    (ROOT / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
