"""Score fixed UPGD and one-shot masks on transfer datasets for both models."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from tuac.data import load_jsonl
from tuac.hf import HFCausalLMScorer
from tuac.quantization import transformer_blocks
from tuac.structured_sparsity import register_channel_mask_hooks


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        baseline_key = "baseline_retained" if "baseline_retained" in saved.files else "aggressive"
        return np.asarray(saved[baseline_key], dtype=np.int64), np.asarray(saved["retained"], dtype=np.int64)


def score_model(model_name: str, model_key: str, masks_path: Path, quantization: str,
                compute_dtype: str, datasets_dir: Path, output_dir: Path, batch_size: int,
                only_dataset: set[str] | None = None) -> list[dict]:
    baseline, upgd = load_masks(masks_path)
    scorer = HFCausalLMScorer(
        model_name, device_map="auto", dtype="auto", quantization=quantization,
        bnb_4bit_compute_dtype=compute_dtype, batch_size=batch_size, max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    rows: list[dict] = []
    for data_path in sorted(datasets_dir.glob("*.jsonl")):
        name = data_path.stem
        if only_dataset is not None and name not in only_dataset:
            continue
        examples = load_jsonl(data_path)
        labels = np.asarray([example.answer if example.answer is not None else -1 for example in examples], dtype=np.int64)
        if np.any(labels < 0) or len(set(example.id for example in examples)) != len(examples):
            raise ValueError(f"{name} must be fully labeled with unique IDs")
        started = time.perf_counter()
        dense = scorer.score(examples)
        baseline_handles = register_channel_mask_hooks(blocks, baseline)
        try:
            baseline_probabilities = scorer.score(examples)
        finally:
            for handle in baseline_handles:
                handle.remove()
        upgd_handles = register_channel_mask_hooks(blocks, upgd)
        try:
            upgd_probabilities = scorer.score(examples)
        finally:
            for handle in upgd_handles:
                handle.remove()
        metadata = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": name, "data_path": str(data_path.resolve()), "examples": len(examples),
            "model": model_name, "model_key": model_key, "quantization": quantization,
            "bnb_4bit_compute_dtype": compute_dtype, "mask_mode": "logical_channel_hook",
            "baseline_mask": str(masks_path.resolve()), "scoring_seconds": time.perf_counter() - started,
            "claim_boundary": "transfer quality validation with logical channel masks; no packed sparse-kernel speed claim",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{model_key}_{name}"
        np.savez_compressed(
            output_dir / f"{stem}_reference.npz", dense=dense.astype(np.float32),
            baseline_aggressive=baseline_probabilities.astype(np.float32), labels=labels,
            selected_indices=np.arange(len(examples), dtype=np.int64),
            example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        np.savez_compressed(
            output_dir / f"{stem}_upgd.npz", probabilities=upgd_probabilities.astype(np.float32), labels=labels,
            selected_indices=np.arange(len(examples), dtype=np.int64),
            example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        rows.append(metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)
        del dense, baseline_probabilities, upgd_probabilities
        clear_cuda()
    del scorer
    clear_cuda()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-15b", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-3b", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--masks-15b", required=True)
    parser.add_argument("--masks-3b", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compute-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--only-model", choices=("1p5b", "3b_nf4"), default=None)
    parser.add_argument("--only-dataset", action="append", default=None)
    args = parser.parse_args()
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)
    only_dataset = set(args.only_dataset) if args.only_dataset else None
    rows = []
    if args.only_model in (None, "1p5b"):
        rows.extend(score_model(args.model_15b, "1p5b", Path(args.masks_15b), "none", args.compute_dtype, datasets_dir, output_dir, args.batch_size, only_dataset))
    if args.only_model in (None, "3b_nf4"):
        rows.extend(score_model(args.model_3b, "3b_nf4", Path(args.masks_3b), "4bit", args.compute_dtype, datasets_dir, output_dir, args.batch_size, only_dataset))
    (output_dir / "scoring_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": sorted({row["dataset"] for row in rows}), "models": sorted({row["model_key"] for row in rows}), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
