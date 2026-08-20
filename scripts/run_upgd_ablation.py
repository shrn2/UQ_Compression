"""Run the minimum UPGD ablation matrix on the strongest transfer cohort."""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from tuac.ablation import ABLATION_CONFIGS
from tuac.data import load_jsonl
from tuac.hf import HFCausalLMScorer
from tuac.quantization import transformer_blocks
from tuac.scoring import entropy
from tuac.stochastic_gate_pruning import train_upgd_masks, uncertainty_strata
from tuac.structured_sparsity import register_channel_mask_hooks


ROOT = Path("runs/iter14_ablation_hellaswag")
CALIBRATION_DATA = Path("runs/iter8_arc/ccd_training.jsonl")
EVALUATION_DATA = Path("runs/iter13_transfer/datasets/hellaswag.jsonl")

MODELS = {
    "1p5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": "none", "steps": 600,
        "seed": 20260829, "calibration": Path("runs/iter10_ursa/calibration.npz"),
        "full_masks": Path("runs/iter11_upgd/masks.npz"),
    },
    "3b_nf4": {
        "model": "Qwen/Qwen2.5-3B-Instruct", "quantization": "4bit", "steps": 300,
        "seed": 20260839, "calibration": Path("runs/iter12_q4_3b/calibration.npz"),
        "full_masks": Path("runs/iter12_q4_3b/masks.npz"),
    },
}

MINIMUM_METHODS = ("dense", "random_seed1", "random_seed2", "random_seed3", "one_shot", "kl_only", "kl_entropy_margin", "full_upgd")


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def save_mask(path: Path, retained: np.ndarray, baseline: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, retained=np.asarray(retained, dtype=np.int64), baseline_retained=np.asarray(baseline, dtype=np.int64),
        sparsity=np.asarray(0.5), method=np.asarray(metadata["method"], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def load_retained(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        baseline = np.asarray(saved["baseline_retained"], dtype=np.int64)
        retained = np.asarray(saved["retained"], dtype=np.int64)
    return retained, baseline


def _margin(probabilities: np.ndarray) -> np.ndarray:
    top = np.partition(np.asarray(probabilities, dtype=np.float64), -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def write_training_log(training_json: Path) -> None:
    record = json.loads(training_json.read_text(encoding="utf-8"))
    output = training_json.with_name(training_json.stem + "_log.jsonl")
    output.write_text("".join(json.dumps(trace, sort_keys=True) + "\n" for trace in record.get("trace", [])), encoding="utf-8")


def write_evaluation_predictions(path: Path, examples, dense: np.ndarray, masked: np.ndarray, labels: np.ndarray) -> None:
    dense_entropy = entropy(dense, normalized=True)
    masked_entropy = entropy(masked, normalized=True)
    dense_margin = _margin(dense)
    masked_margin = _margin(masked)
    _, strata = uncertainty_strata(dense, strata=3)
    rows = []
    for index, example in enumerate(examples):
        rows.append({
            "id": example.id, "label": int(labels[index]),
            "dense_prediction": int(dense[index].argmax()), "masked_prediction": int(masked[index].argmax()),
            "dense_probs": dense[index].astype(float).tolist(), "masked_probs": masked[index].astype(float).tolist(),
            "dense_entropy": float(dense_entropy[index]), "masked_entropy": float(masked_entropy[index]),
            "dense_margin": float(dense_margin[index]), "masked_margin": float(masked_margin[index]),
            "correct": bool(masked[index].argmax() == labels[index]), "uncertainty_stratum": int(strata[index]),
        })
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prepare_methods(model_key: str, spec: dict) -> dict[str, Path]:
    method_dir = ROOT / model_key / "masks"
    method_dir.mkdir(parents=True, exist_ok=True)
    full_retained, baseline = load_retained(spec["full_masks"])
    paths: dict[str, Path] = {}
    full_metadata = {"method": "full_upgd", "model": spec["model"], "model_key": model_key,
                     "source_masks": str(spec["full_masks"].resolve()), "objective": ABLATION_CONFIGS["full_upgd"].__dict__}
    full_path = method_dir / "full_upgd_masks.npz"
    save_mask(full_path, full_retained, baseline, full_metadata)
    paths["full_upgd"] = full_path
    one_shot_path = method_dir / "one_shot_masks.npz"
    save_mask(one_shot_path, baseline, baseline, {"method": "one_shot", "model": spec["model"], "model_key": model_key, "source_masks": str(spec["full_masks"].resolve())})
    paths["one_shot"] = one_shot_path
    with np.load(spec["full_masks"], allow_pickle=False) as saved:
        channels = int(saved["logits"].shape[1])
    keep = full_retained.shape[1]
    for seed_index in range(1, 4):
        rng = np.random.default_rng(20260900 + seed_index + (0 if model_key == "1p5b" else 100))
        random_mask = np.stack([np.sort(rng.choice(channels, size=keep, replace=False)) for _ in range(full_retained.shape[0])])
        random_path = method_dir / f"random_seed{seed_index}_masks.npz"
        save_mask(random_path, random_mask, baseline, {"method": f"random_seed{seed_index}", "model": spec["model"], "model_key": model_key, "seed": 20260900 + seed_index + (0 if model_key == "1p5b" else 100), "random_structured_pruning": True})
        paths[f"random_seed{seed_index}"] = random_path
    for objective_name in ("kl_only", "kl_entropy_margin"):
        config = ABLATION_CONFIGS[objective_name]
        output_path = ROOT / model_key / f"{objective_name}_training.json"
        masks_path = method_dir / f"{objective_name}_masks.npz"
        train_upgd_masks(
            data_path=str(CALIBRATION_DATA), calibration_artifact_path=str(spec["calibration"]),
            slm_model=spec["model"], output_path=str(output_path), masks_output_path=str(masks_path),
            sparsity=0.5, steps=spec["steps"], per_stratum=1, learning_rate=0.05,
            start_temperature=0.5, end_temperature=0.1, start_noise=0.5, end_noise=0.05,
            entropy_weight=config.entropy_weight, margin_weight=config.margin_weight,
            robust_weight=config.robust_weight, robust_temperature=0.1,
            objective_name=objective_name, seed=spec["seed"], quantization=spec["quantization"],
            bnb_4bit_compute_dtype="float16",
        )
        write_training_log(output_path)
        paths[objective_name] = masks_path
    return paths


def score_model(model_key: str, spec: dict, methods: dict[str, Path]) -> list[dict]:
    examples = load_jsonl(EVALUATION_DATA)
    labels = np.asarray([example.answer for example in examples], dtype=np.int64)
    scorer = HFCausalLMScorer(
        spec["model"], device_map="auto", dtype="auto", quantization=spec["quantization"],
        bnb_4bit_compute_dtype="float16", batch_size=8, max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    started = time.perf_counter()
    dense = scorer.score(examples)
    _, baseline = load_retained(spec["full_masks"])
    baseline_handles = register_channel_mask_hooks(blocks, baseline)
    try:
        baseline_probabilities = scorer.score(examples)
    finally:
        for handle in baseline_handles:
            handle.remove()
    output_dir = ROOT / model_key / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_metadata = {"model": spec["model"], "model_key": model_key, "dataset": "HellaSwag", "examples": len(examples), "quantization": spec["quantization"], "mask_mode": "logical_channel_hook", "claim_boundary": "ablation quality validation; no packed sparse-kernel speed claim"}
    reference_path = output_dir / "reference.npz"
    np.savez_compressed(reference_path, dense=dense.astype(np.float32), baseline_aggressive=baseline_probabilities.astype(np.float32), labels=labels, example_ids=np.asarray([example.id for example in examples], dtype=np.str_), metadata=np.asarray(json.dumps(reference_metadata, sort_keys=True), dtype=np.str_))
    write_evaluation_predictions(output_dir / "dense_predictions.jsonl", examples, dense, dense, labels)
    write_evaluation_predictions(output_dir / "one_shot_predictions.jsonl", examples, dense, baseline_probabilities, labels)
    rows = []
    for method, mask_path in methods.items():
        if method == "dense":
            continue
        retained, _ = load_retained(mask_path)
        handles = register_channel_mask_hooks(blocks, retained)
        try:
            probabilities = scorer.score(examples)
        finally:
            for handle in handles:
                handle.remove()
        metadata = {"model": spec["model"], "model_key": model_key, "method": method, "dataset": "HellaSwag", "examples": len(examples), "quantization": spec["quantization"], "mask_mode": "logical_channel_hook", "mask_path": str(mask_path.resolve()), "scoring_seconds": time.perf_counter() - started}
        upgd_path = output_dir / f"{method}.npz"
        np.savez_compressed(upgd_path, probabilities=probabilities.astype(np.float32), labels=labels, example_ids=np.asarray([example.id for example in examples], dtype=np.str_), metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_))
        write_evaluation_predictions(output_dir / f"{method}_predictions.jsonl", examples, dense, probabilities, labels)
        rows.append({"model_key": model_key, "method": method, "reference": str(reference_path.resolve()), "artifact": str(upgd_path.resolve()), "mask": str(mask_path.resolve()), "metadata": metadata})
        del probabilities
        clear_cuda()
    del scorer
    clear_cuda()
    return rows


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model_key, spec in MODELS.items():
        methods = {"dense": None, **prepare_methods(model_key, spec)}
        (ROOT / model_key / "experiment_config.json").write_text(json.dumps({"model_key": model_key, "model": spec["model"], "quantization": spec["quantization"], "dataset": "HellaSwag", "calibration_data": str(CALIBRATION_DATA.resolve()), "evaluation_data": str(EVALUATION_DATA.resolve()), "sparsity": 0.5, "steps": spec["steps"], "seed": spec["seed"], "methods": list(MINIMUM_METHODS), "loss_configs": {name: cfg.__dict__ for name, cfg in ABLATION_CONFIGS.items()}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest.extend(score_model(model_key, spec, methods))
    (ROOT / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"models": sorted({row["model_key"] for row in manifest}), "methods": sorted({row["method"] for row in manifest}), "rows": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
