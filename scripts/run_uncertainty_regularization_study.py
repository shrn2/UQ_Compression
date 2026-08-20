"""Run the matched multi-task lambda-UQ sweep and score every learned mask."""

from __future__ import annotations

import gc
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tuac.data import load_jsonl
from tuac.hf import HFCausalLMScorer
from tuac.quantization import transformer_blocks
from tuac.scoring import entropy
from tuac.stochastic_gate_pruning import train_upgd_masks, uncertainty_strata
from tuac.structured_sparsity import register_channel_mask_hooks

ROOT = Path("runs/iter16_uncertainty_reg")
CALIBRATION = {
    "1p5b": ROOT / "calibration_1p5b.npz",
    "3b_nf4": ROOT / "calibration_3b_nf4.npz",
}
MODELS = {
    "1p5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": "none"},
    "3b_nf4": {"model": "Qwen/Qwen2.5-3B-Instruct", "quantization": "4bit"},
}
DATASETS = {
    "hellaswag": {"path": Path("runs/iter13_transfer/datasets/hellaswag.jsonl"), "label": "HellaSwag"},
    "arc_challenge": {"path": Path("runs/iter15_ablation_previous/datasets/arc_challenge.jsonl"), "label": "ARC-Challenge"},
    "mmlu_aux": {"path": Path("runs/iter15_ablation_previous/datasets/mmlu_aux.jsonl"), "label": "MMLU auxiliary"},
}
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
SEEDS = (1, 2, 3)
STEPS = 300  # matched across models; 3B cannot complete the prior 600-step budget reliably


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def write_predictions(path: Path, examples, dense: np.ndarray, masked: np.ndarray, labels: np.ndarray) -> None:
    dense_entropy = entropy(dense, normalized=True)
    masked_entropy = entropy(masked, normalized=True)
    dense_margin = _margin(dense)
    masked_margin = _margin(masked)
    _, strata = uncertainty_strata(dense, strata=3)
    rows = []
    for i, example in enumerate(examples):
        rows.append({
            "id": example.id, "label": int(labels[i]),
            "dense_prediction": int(dense[i].argmax()), "masked_prediction": int(masked[i].argmax()),
            "dense_probs": dense[i].astype(float).tolist(), "masked_probs": masked[i].astype(float).tolist(),
            "dense_entropy": float(dense_entropy[i]), "masked_entropy": float(masked_entropy[i]),
            "dense_margin": float(dense_margin[i]), "masked_margin": float(masked_margin[i]),
            "correct": bool(masked[i].argmax() == labels[i]), "uncertainty_stratum": int(strata[i]),
        })
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _margin(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[1] < 2:
        return np.ones(values.shape[0])
    top = np.partition(values, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def train_matrix(model_filter: str | None = None, lambda_filter: float | None = None, seed_filter: int | None = None) -> list[dict]:
    rows = []
    for model_key, spec in MODELS.items():
        if model_filter is not None and model_key != model_filter:
            continue
        for lam in LAMBDAS:
            if lambda_filter is not None and float(lam) != float(lambda_filter):
                continue
            for seed in SEEDS:
                if seed_filter is not None and int(seed) != int(seed_filter):
                    continue
                tag = f"lambda_{lam:g}/seed_{seed}"
                out_dir = ROOT / model_key / tag
                training = out_dir / "training.json"
                masks = out_dir / "masks.npz"
                print(f"[train] {model_key} lambda={lam:g} seed={seed}", flush=True)
                metadata = train_upgd_masks(
                    data_path=str(ROOT / "calibration_sources" / "multitask_calibration.jsonl"),
                    calibration_artifact_path=str(CALIBRATION[model_key]), slm_model=spec["model"],
                    output_path=str(training), masks_output_path=str(masks), sparsity=0.5,
                    steps=STEPS, per_stratum=1, learning_rate=0.05, lambda_uq=lam, seed=seed,
                    groups_per_batch=1,
                    quantization=spec["quantization"], bnb_4bit_compute_dtype="float16",
                )
                (out_dir / "training_log.jsonl").write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in metadata.get("trace", [])), encoding="utf-8"
                )
                rows.append({"model_key": model_key, "lambda_uq": lam, "seed": seed,
                             "training": str(training.resolve()), "masks": str(masks.resolve()),
                             "steps": STEPS, "metadata": metadata})
    return rows


def score_model(model_key: str, spec: dict, train_rows: list[dict]) -> list[dict]:
    scorer = HFCausalLMScorer(spec["model"], device_map="auto", dtype="auto", quantization=spec["quantization"],
                              bnb_4bit_compute_dtype="float16", batch_size=8, max_batch_tokens=4096)
    blocks = transformer_blocks(scorer.model)
    output_rows = []
    selected = [row for row in train_rows if row["model_key"] == model_key]
    for dataset_key, dataset_spec in DATASETS.items():
        examples = load_jsonl(dataset_spec["path"])
        labels = np.asarray([example.answer for example in examples], dtype=np.int64)
        ids = np.asarray([example.id for example in examples], dtype=np.str_)
        dense = scorer.score(examples).astype(np.float32)
        output_dir = ROOT / model_key / "scores" / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_meta = {"model": spec["model"], "model_key": model_key, "dataset": dataset_spec["label"],
                    "dataset_key": dataset_key, "examples": len(examples), "quantization": spec["quantization"],
                    "mask_mode": "logical_channel_hook", "calibration_mixture": str((ROOT / "calibration_sources" / "multitask_calibration.jsonl").resolve()),
                    "claim_boundary": "quality validation under logical masks; no packed sparse-kernel speed claim"}
        np.savez_compressed(output_dir / "dense.npz", probabilities=dense, labels=labels, example_ids=ids,
                            metadata=np.asarray(json.dumps(ref_meta, sort_keys=True), dtype=np.str_))
        write_predictions(output_dir / "dense_predictions.jsonl", examples, dense, dense, labels)
        for row in selected:
            mask_path = Path(row["masks"])
            with np.load(mask_path, allow_pickle=False) as saved:
                retained = np.asarray(saved["retained"], dtype=np.int64)
            handles = register_channel_mask_hooks(blocks, retained)
            started = time.perf_counter()
            try:
                probabilities = scorer.score(examples).astype(np.float32)
            finally:
                for handle in handles:
                    handle.remove()
            lam = float(row["lambda_uq"])
            method = f"lambda_{lam:g}_seed_{row['seed']}"
            artifact = output_dir / f"{method}.npz"
            meta = {**ref_meta, "method": "UPGD", "lambda_uq": lam, "seed": row["seed"],
                    "mask_path": str(mask_path.resolve()), "scoring_seconds": time.perf_counter() - started}
            np.savez_compressed(artifact, probabilities=probabilities, labels=labels, example_ids=ids,
                                metadata=np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_))
            write_predictions(output_dir / f"{method}_predictions.jsonl", examples, dense, probabilities, labels)
            output_rows.append({"model_key": model_key, "dataset_key": dataset_key, "dataset": dataset_spec["label"],
                                "lambda_uq": lam, "seed": row["seed"], "artifact": str(artifact.resolve()),
                                "reference": str((output_dir / "dense.npz").resolve()), "mask": str(mask_path.resolve()),
                                "metadata": meta})
            del probabilities
            clear_cuda()
    del scorer
    clear_cuda()
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="experiment YAML (recorded by the HPC wrapper)")
    parser.add_argument("--model", choices=tuple(MODELS), help="run one model")
    parser.add_argument("--lambda-uq", type=float, choices=LAMBDAS, help="run one lambda")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="run one seed")
    parser.add_argument("--array-index", type=int, help="Slurm array index in the 24-run matrix")
    parser.add_argument("--train-only", action="store_true", help="train selected masks without scoring")
    parser.add_argument("--resume", action="store_true", help="reuse existing training artifacts")
    args = parser.parse_args()
    if args.array_index is not None:
        if not 0 <= args.array_index < len(MODELS) * len(LAMBDAS) * len(SEEDS):
            raise SystemExit("--array-index must be in [0, 23]")
        model_index, remainder = divmod(args.array_index, len(LAMBDAS) * len(SEEDS))
        lambda_index, seed_index = divmod(remainder, len(SEEDS))
        args.model = tuple(MODELS)[model_index]
        args.lambda_uq = LAMBDAS[lambda_index]
        args.seed = SEEDS[seed_index]
    ROOT.mkdir(parents=True, exist_ok=True)
    train_rows = train_matrix(args.model, args.lambda_uq, args.seed)
    if args.train_only:
        print(json.dumps({"training_runs": len(train_rows)}, indent=2))
        return 0
    score_rows = []
    models = {args.model: MODELS[args.model]} if args.model else MODELS
    for model_key, spec in models.items():
        score_rows.extend(score_model(model_key, spec, train_rows))
    manifest = {"study": "UPGD uncertainty regularization lambda sweep", "models": MODELS,
                "lambdas": list(LAMBDAS), "seeds": list(SEEDS), "steps": STEPS,
                "matched_steps": True, "training_runs": train_rows, "score_slices": score_rows,
                "expected_training_runs": 24, "expected_score_slices": 72,
                "controls": {"dense": "per-model/dataset dense reference", "random_and_one_shot": "prior Iteration 14/15 controls retained in consolidated report"}}
    (ROOT / "study_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"training_runs": len(train_rows), "score_slices": len(score_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
