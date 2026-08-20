"""Materialize the JSONL logging deliverables for a completed ablation run."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_upgd_ablation import EVALUATION_DATA, MODELS, ROOT, write_evaluation_predictions, write_training_log
from tuac.data import load_jsonl


def main() -> int:
    examples = load_jsonl(EVALUATION_DATA)
    labels = np.asarray([example.answer for example in examples], dtype=np.int64)
    for model_key in MODELS:
        model_dir = ROOT / model_key
        for training_path in model_dir.glob("*_training.json"):
            write_training_log(training_path)
        with np.load(model_dir / "scores" / "reference.npz", allow_pickle=False) as saved:
            dense = np.asarray(saved["dense"], dtype=np.float32)
            baseline = np.asarray(saved["baseline_aggressive"], dtype=np.float32)
        write_evaluation_predictions(model_dir / "scores" / "dense_predictions.jsonl", examples, dense, dense, labels)
        write_evaluation_predictions(model_dir / "scores" / "one_shot_predictions.jsonl", examples, dense, baseline, labels)
        for score_path in (model_dir / "scores").glob("*.npz"):
            if score_path.name == "reference.npz":
                continue
            with np.load(score_path, allow_pickle=False) as saved:
                probabilities = np.asarray(saved["probabilities"], dtype=np.float32)
            write_evaluation_predictions(model_dir / "scores" / f"{score_path.stem}_predictions.jsonl", examples, dense, probabilities, labels)
    print(ROOT / "ablation_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
