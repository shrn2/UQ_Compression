"""Build a disjoint 3-task calibration mixture for the lambda-UQ study."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tuac.data import MultipleChoiceExample, load_jsonl
from tuac.hf import HFCausalLMScorer
from tuac.scoring import entropy
from tuac.stochastic_gate_pruning import uncertainty_strata


ROOT = Path("runs/iter16_uncertainty_reg")
SOURCE_DIR = ROOT / "calibration_sources"
MIXTURE_PATH = SOURCE_DIR / "multitask_calibration.jsonl"
MODEL_SPECS = {
    "1p5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": "none", "baseline": Path("runs/iter10_ursa/calibration.npz")},
    "3b_nf4": {"model": "Qwen/Qwen2.5-3B-Instruct", "quantization": "4bit", "baseline": Path("runs/iter12_q4_3b/calibration.npz")},
}


def hellaswag_examples() -> list[MultipleChoiceExample]:
    final_ids = {example.id for example in load_jsonl("runs/iter13_transfer/datasets/hellaswag.jsonl")}
    rows = []
    with (SOURCE_DIR / "hellaswag_train.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            example_id = f"hellaswag-train-{item['ind']}"
            if f"hellaswag-{item['ind']}" in final_ids:
                continue
            prompt = f"Question: {item['ctx']}\nContinuation:"
            rows.append(MultipleChoiceExample(id=example_id, dataset="HellaSwag-calibration", prompt=prompt, choices=tuple(item["endings"]), answer=int(item["label"])))
            if len(rows) >= 1024:
                break
    return rows


def disjoint_sample(source: Path, final_ids: set[str], count: int, seed: int, dataset_label: str) -> list[MultipleChoiceExample]:
    examples = [example for example in load_jsonl(source) if example.id not in final_ids]
    if len(examples) < count:
        raise ValueError(f"not enough disjoint {dataset_label} calibration examples: {len(examples)}")
    rng = random.Random(seed)
    selected = rng.sample(examples, count)
    return [MultipleChoiceExample(id=example.id, dataset=dataset_label, prompt=example.prompt, choices=example.choices, answer=example.answer) for example in selected]


def build_mixture() -> list[MultipleChoiceExample]:
    final_mmlu = {example.id for example in load_jsonl("runs/iter15_ablation_previous/datasets/mmlu_aux.jsonl")}
    final_arc = {example.id for example in load_jsonl("runs/iter15_ablation_previous/datasets/arc_challenge.jsonl")}
    rows = (
        disjoint_sample(Path("runs/iter8_arc/ccd_training.jsonl"), final_mmlu, 256, 20263001, "MMLU-auxiliary-calibration")
        + disjoint_sample(Path("runs/iter1_arc/calibration_pool.jsonl"), final_arc, 256, 20263002, "ARC-Challenge-calibration")
        + hellaswag_examples()[:256]
    )
    if len(rows) != 768:
        raise ValueError(f"expected 768 calibration rows, found {len(rows)}")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MIXTURE_PATH.write_text("".join(json.dumps(asdict(example), sort_keys=True) + "\n" for example in rows), encoding="utf-8")
    return rows


def pad_probabilities(rows: np.ndarray, width: int = 5) -> np.ndarray:
    output = np.zeros((rows.shape[0], width), dtype=np.float32)
    output[:, : rows.shape[1]] = rows
    return output


def main() -> int:
    rows = build_mixture()
    dataset_labels = np.asarray([example.dataset for example in rows], dtype=np.str_)
    dataset_keys = sorted(set(dataset_labels.tolist()))
    for model_key, spec in MODEL_SPECS.items():
        scorer = HFCausalLMScorer(spec["model"], device_map="auto", dtype="auto", quantization=spec["quantization"], bnb_4bit_compute_dtype="float16", batch_size=8, max_batch_tokens=4096)
        probabilities_by_dataset = []
        for dataset_label in dataset_keys:
            indices = np.flatnonzero(dataset_labels == dataset_label)
            probabilities_by_dataset.append((indices, scorer.score([rows[int(index)] for index in indices])))
        dense = np.zeros((len(rows), 5), dtype=np.float32)
        for indices, probabilities in probabilities_by_dataset:
            dense[indices] = pad_probabilities(np.asarray(probabilities, dtype=np.float32))
        strata = np.zeros(len(rows), dtype=np.int64)
        entropy_values = np.zeros(len(rows), dtype=np.float32)
        for dataset_label in dataset_keys:
            indices = np.flatnonzero(dataset_labels == dataset_label)
            entropy_values[indices] = entropy(dense[indices], normalized=True).astype(np.float32)
            _, strata[indices] = uncertainty_strata(dense[indices], strata=3)
        dataset_codes = np.asarray([dataset_keys.index(label) for label in dataset_labels], dtype=np.int64)
        batch_groups = dataset_codes * 3 + strata
        with np.load(spec["baseline"], allow_pickle=False) as baseline_saved:
            baseline_importance = np.asarray(baseline_saved["baseline_importance"], dtype=np.float32)
        artifact_path = ROOT / f"calibration_{model_key}.npz"
        metadata = {"model": spec["model"], "model_key": model_key, "examples": len(rows), "datasets": {label: int(np.sum(dataset_labels == label)) for label in dataset_keys}, "mixture_path": str(MIXTURE_PATH.resolve()), "disjoint_from_final_evaluation": True, "stratification": "nine balanced dataset-by-dense-entropy cells (three strata per dataset)", "source_baseline_importance": str(spec["baseline"].resolve())}
        np.savez_compressed(artifact_path, dense_probabilities=dense, uncertainty=entropy_values, stratum_labels=batch_groups, uncertainty_strata=strata, selected_indices=np.arange(len(rows), dtype=np.int64), example_ids=np.asarray([example.id for example in rows], dtype=np.str_), dataset_labels=dataset_labels, baseline_importance=baseline_importance, metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_))
        del scorer
    manifest = {"mixture_path": str(MIXTURE_PATH.resolve()), "examples": len(rows), "datasets": {label: int(np.sum(dataset_labels == label)) for label in dataset_keys}, "models": {key: str((ROOT / f"calibration_{key}.npz").resolve()) for key in MODEL_SPECS}}
    (ROOT / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
