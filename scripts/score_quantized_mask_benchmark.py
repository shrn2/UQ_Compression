"""Score dense, matched baseline, and UPGD masks in one quantized model load."""

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


def _clear() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def score_one(name: str, data_path: str, masks_path: str, model: str, output_dir: Path,
              *, batch_size: int, quantization: str, compute_dtype: str) -> dict:
    examples = load_jsonl(data_path)
    labels = np.asarray([example.answer if example.answer is not None else -1 for example in examples], dtype=np.int64)
    if np.any(labels < 0):
        raise ValueError(f"{name} contains unlabeled examples")
    with np.load(masks_path, allow_pickle=False) as saved:
        baseline_key = "baseline_retained" if "baseline_retained" in saved.files else "aggressive"
        baseline = np.asarray(saved[baseline_key], dtype=np.int64)
        upgd = np.asarray(saved["retained"], dtype=np.int64)
    scorer = HFCausalLMScorer(
        model, device_map="auto", dtype="auto", quantization=quantization,
        bnb_4bit_compute_dtype=compute_dtype, batch_size=batch_size, max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
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
        "dataset": name, "data_path": str(Path(data_path).resolve()), "examples": len(examples),
        "model": model, "quantization": quantization, "bnb_4bit_compute_dtype": compute_dtype,
        "mask_mode": "logical_channel_hook", "baseline_mask": str(Path(masks_path).resolve()),
        "scoring_seconds": time.perf_counter() - started,
        "claim_boundary": "quantized quality validation with logical channel masks; no packed sparse-kernel speed claim",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{name}_reference.npz", dense=dense.astype(np.float32),
        baseline_aggressive=baseline_probabilities.astype(np.float32), labels=labels,
        selected_indices=np.arange(len(examples), dtype=np.int64),
        example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    np.savez_compressed(
        output_dir / f"{name}_upgd.npz", probabilities=upgd_probabilities.astype(np.float32), labels=labels,
        selected_indices=np.arange(len(examples), dtype=np.int64),
        example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    del scorer
    _clear()
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default="4bit")
    parser.add_argument("--compute-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--arc", required=True)
    parser.add_argument("--mmlu-pro", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = [
        score_one("arc", args.arc, args.masks, args.model, output_dir, batch_size=args.batch_size,
                  quantization=args.quantization, compute_dtype=args.compute_dtype),
        score_one("mmlu_pro", args.mmlu_pro, args.masks, args.model, output_dir, batch_size=args.batch_size,
                  quantization=args.quantization, compute_dtype=args.compute_dtype),
    ]
    (output_dir / "scoring_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
