"""End-to-end calibration collection and offline policy analysis."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .allocation import allocate_bits, average_bits, random_allocation
from .artifacts import CalibrationArtifact
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import classification_metrics, uncertainty_preservation
from .quantization import (
    parameter_count,
    temporarily_mixed_quantized,
    temporarily_quantized,
    transformer_blocks,
)
from .scoring import ScoreMode, compute_importance

SCORE_MODES: tuple[ScoreMode, ...] = (
    "student_only",
    "teacher_confidence",
    "teacher_uncertainty",
    "disagreement",
    "combined",
)


def collect_calibration(
    *,
    data_path: str,
    teacher_model: str,
    student_model: str,
    output_path: str,
    probe_bits: int = 4,
    group_size: int = 128,
    limit: int | None = None,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> CalibrationArtifact:
    """Run teacher once, then FP and one-layer-at-a-time student inference."""

    examples = load_jsonl(data_path, limit=limit)
    teacher = HFCausalLMScorer(
        teacher_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code
    )
    teacher_probabilities = teacher.score(examples)
    del teacher
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    student = HFCausalLMScorer(
        student_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code
    )
    student_probabilities = student.score(examples)
    width = max(teacher_probabilities.shape[1], student_probabilities.shape[1])
    teacher_probabilities = _right_pad(teacher_probabilities, width)
    student_probabilities = _right_pad(student_probabilities, width)

    blocks = transformer_blocks(student.model)
    layer_outputs = []
    for _, block in blocks:
        with temporarily_quantized(block, probe_bits, group_size=group_size):
            layer_outputs.append(_right_pad(student.score(examples), width))

    artifact = CalibrationArtifact(
        teacher_probabilities=teacher_probabilities,
        student_probabilities=student_probabilities,
        quantized_probabilities=np.stack(layer_outputs),
        layer_names=tuple(name for name, _ in blocks),
        parameter_counts=np.asarray([parameter_count(block) for _, block in blocks]),
        labels=np.asarray([example.answer if example.answer is not None else -1 for example in examples]),
        example_ids=tuple(example.id for example in examples),
        probe_bits=probe_bits,
        metadata={
            "teacher_model": teacher_model,
            "student_model": student_model,
            "data_path": str(Path(data_path).resolve()),
            "group_size": group_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "Sensitivity uses fake weight quantization; no packed-kernel runtime claim is implied.",
        },
    )
    artifact.save(output_path)
    return artifact


def _right_pad(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[1] == width:
        return values
    result = np.zeros((values.shape[0], width), dtype=values.dtype)
    result[:, : values.shape[1]] = values
    return result


def analyze_artifact(
    artifact: CalibrationArtifact,
    *,
    budgets: Iterable[float] = (4.0, 3.5, 3.0),
    bit_choices: Sequence[int] = (3, 4, 8),
) -> dict:
    """Produce all signal ablations and matched-budget policies."""

    budgets = tuple(float(value) for value in budgets)
    bit_choices = tuple(int(value) for value in bit_choices)
    report: dict = {
        "metadata": artifact.metadata,
        "probe_bits": artifact.probe_bits,
        "bit_choices": list(bit_choices),
        "layers": list(artifact.layer_names),
        "parameter_counts": artifact.parameter_counts.tolist(),
        "policies": {},
        "baselines": {"uniform": {}, "random_mixed_precision": {}},
    }
    for budget in budgets:
        budget = float(budget)
        if budget.is_integer() and int(budget) in bit_choices:
            widths = np.full(len(artifact.layer_names), int(budget), dtype=np.int64)
            report["baselines"]["uniform"][str(budget)] = {
                "bits": widths.tolist(),
                "average_bits": average_bits(widths, artifact.parameter_counts),
            }
        random_runs = {}
        for seed in range(5):
            allocation = random_allocation(
                artifact.parameter_counts, budget, bit_choices=bit_choices, seed=seed
            )
            random_runs[str(seed)] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
            }
        report["baselines"]["random_mixed_precision"][str(budget)] = random_runs
    importance_by_mode = {}
    for mode in SCORE_MODES:
        result = compute_importance(
            artifact.teacher_probabilities,
            artifact.student_probabilities,
            artifact.quantized_probabilities,
            mode=mode,
        )
        importance_by_mode[mode] = result.layer_importance
        policies = {}
        for budget in budgets:
            allocation = allocate_bits(
                result.layer_importance,
                artifact.parameter_counts,
                float(budget),
                bit_choices=bit_choices,
            )
            policies[str(float(budget))] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
                "estimated_objective": allocation.objective,
            }
        report["policies"][mode] = {
            "example_weights": result.example_weights.tolist(),
            "teacher_entropy": result.teacher_entropy.tolist(),
            "disagreement": result.disagreement.tolist(),
            "layer_importance": result.layer_importance.tolist(),
            "budgets": policies,
        }

    student = importance_by_mode["student_only"]
    combined = importance_by_mode["combined"]
    from .metrics import spearman_correlation

    report["diagnostics"] = {
        "student_teacher_layer_importance_spearman": spearman_correlation(student, combined)
    }
    return report


def save_report(report: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def evaluate_policy(
    *,
    data_path: str,
    student_model: str,
    artifact: CalibrationArtifact,
    report: dict,
    mode: str,
    budget: float,
    output_path: str,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Apply a derived policy with fake quantization and evaluate its probabilities."""

    key = str(float(budget))
    try:
        policy = report["policies"][mode]["budgets"][key]
    except KeyError as exc:
        raise ValueError(f"report contains no {mode!r} policy at budget {key}") from exc
    examples = load_jsonl(data_path)
    scorer = HFCausalLMScorer(
        student_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code
    )
    blocks = transformer_blocks(scorer.model)
    if [name for name, _ in blocks] != report["layers"]:
        raise ValueError("loaded model blocks do not match the calibration report")
    group_size = int(artifact.metadata.get("group_size", 128))
    with temporarily_mixed_quantized(blocks, policy["bits"], group_size=group_size):
        probabilities = scorer.score(examples)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([item.answer if item.answer is not None else -1 for item in examples])
    np.savez_compressed(
        destination,
        probabilities=probabilities.astype(np.float32),
        labels=labels,
        example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
        bits=np.asarray(policy["bits"], dtype=np.int64),
    )
    result = {
        "mode": mode,
        "target_average_bits": float(budget),
        "actual_average_bits": average_bits(policy["bits"], artifact.parameter_counts),
        "output": str(destination),
    }
    if np.all(labels >= 0):
        result["metrics"] = classification_metrics(probabilities, labels)
        if (
            probabilities.shape == artifact.student_probabilities.shape
            and tuple(item.id for item in examples) == artifact.example_ids
        ):
            result["uncertainty_preservation"] = uncertainty_preservation(
                probabilities,
                artifact.student_probabilities,
                artifact.teacher_probabilities,
            )
    return result


def benchmark_policies(
    *,
    data_path: str,
    student_model: str,
    artifact: CalibrationArtifact,
    report: dict,
    output_dir: str,
    modes: Sequence[str] = ("student_only", "combined"),
    budgets: Sequence[float] = (4.0, 3.5, 3.0),
    random_seeds: Sequence[int] = (0, 1, 2),
    include_baselines: bool = True,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Evaluate FP, uniform, random, and selected sensitivity policies in one model load."""

    examples = load_jsonl(data_path)
    labels = np.asarray([item.answer if item.answer is not None else -1 for item in examples])
    if np.any(labels < 0):
        raise ValueError("benchmark data requires an answer on every example")
    scorer = HFCausalLMScorer(
        student_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code
    )
    blocks = transformer_blocks(scorer.model)
    if [name for name, _ in blocks] != report["layers"]:
        raise ValueError("loaded model blocks do not match the calibration report")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    group_size = int(artifact.metadata.get("group_size", 128))
    full_precision = scorer.score(examples)
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {"full_precision": full_precision.astype(np.float32)}

    def measure(name: str, family: str, target: float, bits: Sequence[int], seed: int | None = None):
        with temporarily_mixed_quantized(blocks, bits, group_size=group_size):
            probabilities = scorer.score(examples)
        metrics = classification_metrics(probabilities, labels)
        preservation = uncertainty_preservation(probabilities, full_precision)
        record = {
            "name": name,
            "family": family,
            "target_average_bits": target,
            "actual_average_bits": average_bits(bits, artifact.parameter_counts),
            "seed": seed,
            **metrics,
            **preservation,
        }
        records.append(record)
        arrays[name] = probabilities.astype(np.float32)

    fp_metrics = classification_metrics(full_precision, labels)
    records.append(
        {
            "name": "full_precision",
            "family": "full_precision",
            "target_average_bits": 16.0,
            "actual_average_bits": 16.0,
            "seed": None,
            **fp_metrics,
            "mean_absolute_uncertainty_distortion": 0.0,
            "student_uncertainty_spearman": 1.0,
        }
    )
    for budget in (float(value) for value in budgets):
        key = str(budget)
        if include_baselines:
            uniform = report.get("baselines", {}).get("uniform", {}).get(key)
            if uniform is not None:
                measure(f"uniform_{key}", "uniform", budget, uniform["bits"])
            random_runs = report["baselines"]["random_mixed_precision"][key]
            for seed in random_seeds:
                policy = random_runs[str(int(seed))]
                measure(f"random_{key}_seed{seed}", "random", budget, policy["bits"], int(seed))
        for mode in modes:
            policy = report["policies"][mode]["budgets"][key]
            measure(f"{mode}_{key}", mode, budget, policy["bits"])

    np.savez_compressed(
        destination / "probabilities.npz",
        labels=labels,
        example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
        **arrays,
    )
    result = {
        "student_model": student_model,
        "data_path": str(Path(data_path).resolve()),
        "example_count": len(examples),
        "records": records,
    }
    (destination / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    return result
