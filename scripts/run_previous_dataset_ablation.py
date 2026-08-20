"""Score the completed UPGD ablation masks on the prior MMLU-aux and ARC cohorts."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_upgd_ablation import (  # noqa: E402
    MODELS,
    clear_cuda,
    load_retained,
    write_evaluation_predictions,
)

from tuac.data import load_jsonl
from tuac.hf import HFCausalLMScorer
from tuac.quantization import transformer_blocks
from tuac.structured_sparsity import register_channel_mask_hooks


ROOT = Path("runs/iter15_ablation_previous")
SOURCE_ROOT = Path("runs/iter14_ablation_hellaswag")
CALIBRATION_DATA = Path("runs/iter8_arc/ccd_training.jsonl")
DATASETS = {
    "mmlu_aux": {"source": CALIBRATION_DATA, "indices": Path("runs/iter11_upgd/development_evaluation.npz"), "label": "MMLU auxiliary", "limit": 600},
    "arc_challenge": {"source": Path("runs/iter1_arc/evaluation.jsonl"), "indices": None, "label": "ARC-Challenge", "limit": 299},
}
METHODS = ("random_seed1", "random_seed2", "random_seed3", "one_shot", "kl_only", "kl_entropy_margin", "full_upgd")


def materialize_datasets() -> dict[str, Path]:
    dataset_dir = ROOT / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for key, spec in DATASETS.items():
        examples = load_jsonl(spec["source"])
        if spec["indices"] is not None:
            with np.load(spec["indices"], allow_pickle=False) as saved:
                indices = np.asarray(saved["selected_indices"], dtype=np.int64)
                expected_ids = np.asarray(saved["example_ids"], dtype=np.str_)
            examples = [examples[int(index)] for index in indices]
            if [example.id for example in examples] != expected_ids.tolist():
                raise ValueError(f"{key}: selected IDs do not match the prior evaluation artifact")
        if len(examples) != spec["limit"]:
            raise ValueError(f"{key}: expected {spec['limit']} examples, found {len(examples)}")
        path = dataset_dir / f"{key}.jsonl"
        path.write_text("".join(json.dumps(asdict(example), sort_keys=True) + "\n" for example in examples), encoding="utf-8")
        outputs[key] = path
    return outputs


def copy_masks(model_key: str) -> dict[str, Path]:
    source_dir = SOURCE_ROOT / model_key / "masks"
    target_dir = ROOT / model_key / "masks"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for method in METHODS:
        source = source_dir / f"{method}_masks.npz"
        target = target_dir / source.name
        shutil.copyfile(source, target)
        paths[method] = target
    return paths


def score_model(model_key: str, spec: dict, dataset_paths: dict[str, Path], mask_paths: dict[str, Path]) -> list[dict]:
    scorer = HFCausalLMScorer(
        spec["model"], device_map="auto", dtype="auto", quantization=spec["quantization"],
        bnb_4bit_compute_dtype="float16", batch_size=8, max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    rows = []
    for dataset_key, dataset_path in dataset_paths.items():
        examples = load_jsonl(dataset_path)
        labels = np.asarray([example.answer for example in examples], dtype=np.int64)
        dense = scorer.score(examples).astype(np.float32)
        one_shot_retained, _ = load_retained(mask_paths["one_shot"])
        handles = register_channel_mask_hooks(blocks, one_shot_retained)
        try:
            baseline = scorer.score(examples).astype(np.float32)
        finally:
            for handle in handles:
                handle.remove()
        output_dir = ROOT / model_key / dataset_key / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        ids = np.asarray([example.id for example in examples], dtype=np.str_)
        metadata = {
            "model": spec["model"], "model_key": model_key, "dataset": DATASETS[dataset_key]["label"],
            "dataset_key": dataset_key, "examples": len(examples), "quantization": spec["quantization"],
            "mask_mode": "logical_channel_hook", "calibration_data": str(CALIBRATION_DATA.resolve()),
            "claim_boundary": "ablation transfer quality validation; no packed sparse-kernel speed claim",
        }
        np.savez_compressed(output_dir / "reference.npz", dense=dense, baseline_aggressive=baseline, labels=labels, example_ids=ids, metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_))
        write_evaluation_predictions(output_dir / "dense_predictions.jsonl", examples, dense, dense, labels)
        write_evaluation_predictions(output_dir / "one_shot_predictions.jsonl", examples, dense, baseline, labels)
        for method in METHODS:
            retained, _ = load_retained(mask_paths[method])
            started = time.perf_counter()
            handles = register_channel_mask_hooks(blocks, retained)
            try:
                probabilities = scorer.score(examples).astype(np.float32)
            finally:
                for handle in handles:
                    handle.remove()
            method_metadata = {**metadata, "method": method, "mask_path": str(mask_paths[method].resolve()), "scoring_seconds": time.perf_counter() - started}
            artifact = output_dir / f"{method}.npz"
            np.savez_compressed(artifact, probabilities=probabilities, labels=labels, example_ids=ids, metadata=np.asarray(json.dumps(method_metadata, sort_keys=True), dtype=np.str_))
            write_evaluation_predictions(output_dir / f"{method}_predictions.jsonl", examples, dense, probabilities, labels)
            rows.append({"model_key": model_key, "dataset": dataset_key, "artifact": str(artifact.resolve()), "reference": str((output_dir / 'reference.npz').resolve()), "mask": str(mask_paths[method].resolve()), "method": method, "metadata": method_metadata})
            del probabilities
            clear_cuda()
    del scorer
    clear_cuda()
    return rows


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    dataset_paths = materialize_datasets()
    manifest = []
    for model_key, spec in MODELS.items():
        mask_paths = copy_masks(model_key)
        config = {"model_key": model_key, "model": spec["model"], "quantization": spec["quantization"], "datasets": {key: {"label": value["label"], "examples": value["limit"], "path": str(path.resolve())} for (key, value), path in zip(DATASETS.items(), dataset_paths.values())}, "masks_source": str((SOURCE_ROOT / model_key / "masks").resolve()), "methods": ["dense", *METHODS], "mask_mode": "logical_channel_hook"}
        model_dir = ROOT / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest.extend(score_model(model_key, spec, dataset_paths, mask_paths))
    (ROOT / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"models": sorted({row["model_key"] for row in manifest}), "datasets": sorted({row["dataset"] for row in manifest}), "methods": sorted({row["method"] for row in manifest}), "rows": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
