"""Iteration-1 teacher/quantization interaction experiment."""

from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .allocation import allocate_bits, average_bits, random_allocation
from .artifacts import MultiBitCalibrationArtifact
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import classification_metrics, spearman_correlation, uncertainty_preservation
from .quantization import parameter_count, temporarily_mixed_quantized, temporarily_quantized, transformer_blocks
from .scoring import entropy, kl_divergence

POLICY_MODES = (
    "student_only",
    "teacher_confidence",
    "teacher_uncertainty",
    "disagreement",
    "oracle",
)
SCORING_VERSION = "leftpad-logitskeep-tokenbucket4096-v1"


def _pad(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[1] == width:
        return values
    result = np.zeros((values.shape[0], width), dtype=values.dtype)
    result[:, : values.shape[1]] = values
    return result


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def collect_multibit_calibration(
    *,
    data_path: str,
    teacher_model: str,
    student_model: str,
    output_path: str,
    probe_bits: Sequence[int] = (2, 4, 8),
    group_size: int = 128,
    batch_size: int = 32,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> MultiBitCalibrationArtifact:
    """Collect resumable teacher, FP-student, and per-bit/per-layer probabilities."""

    examples = load_jsonl(data_path)
    bits = tuple(sorted(set(int(value) for value in probe_bits)))
    if not bits or bits[0] < 2:
        raise ValueError("probe_bits must contain widths of at least 2 bits")
    output = Path(output_path)
    checkpoint = output.with_suffix("").with_name(output.stem + "_checkpoints")
    checkpoint.mkdir(parents=True, exist_ok=True)
    base_path = checkpoint / "base.npz"

    if base_path.exists():
        with np.load(base_path, allow_pickle=False) as saved:
            teacher_probabilities = saved["teacher_probabilities"]
            student_probabilities = saved["student_probabilities"]
            saved_ids = tuple(str(value) for value in saved["example_ids"])
            saved_batch_size = int(saved["student_batch_size"]) if "student_batch_size" in saved else None
            saved_scoring_version = str(saved["student_scoring_version"]) if "student_scoring_version" in saved else None
        if saved_ids != tuple(item.id for item in examples):
            raise ValueError("checkpoint example IDs do not match the requested calibration data")
        if saved_batch_size == batch_size and saved_scoring_version == SCORING_VERSION:
            print(f"resumed teacher/student probabilities from {base_path}", flush=True)
        else:
            student = HFCausalLMScorer(
                student_model,
                device_map=device_map,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                batch_size=batch_size,
            )
            student_probabilities = _pad(student.score(examples), teacher_probabilities.shape[1])
            del student
            _clear_cuda()
            np.savez_compressed(
                base_path,
                teacher_probabilities=teacher_probabilities.astype(np.float32),
                student_probabilities=student_probabilities.astype(np.float32),
                example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
                student_batch_size=np.asarray(batch_size, dtype=np.int64),
                student_scoring_version=np.asarray(SCORING_VERSION, dtype=np.str_),
            )
            print(
                f"preserved teacher cache and refreshed FP student probabilities at batch {batch_size}",
                flush=True,
            )
    else:
        teacher = HFCausalLMScorer(
            teacher_model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
        )
        teacher_probabilities = teacher.score(examples)
        del teacher
        _clear_cuda()
        student = HFCausalLMScorer(
            student_model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
        )
        student_probabilities = student.score(examples)
        del student
        _clear_cuda()
        width = max(teacher_probabilities.shape[1], student_probabilities.shape[1])
        teacher_probabilities = _pad(teacher_probabilities, width)
        student_probabilities = _pad(student_probabilities, width)
        np.savez_compressed(
            base_path,
            teacher_probabilities=teacher_probabilities.astype(np.float32),
            student_probabilities=student_probabilities.astype(np.float32),
            example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
            student_batch_size=np.asarray(batch_size, dtype=np.int64),
            student_scoring_version=np.asarray(SCORING_VERSION, dtype=np.str_),
        )
        print(f"saved teacher/student checkpoint to {base_path}", flush=True)

    student = HFCausalLMScorer(
        student_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
    )
    blocks = transformer_blocks(student.model)
    layer_names = tuple(name for name, _ in blocks)
    parameter_counts = np.asarray([parameter_count(block) for _, block in blocks], dtype=np.int64)
    width = student_probabilities.shape[1]
    all_probes: list[np.ndarray] = []
    for bit in bits:
        bit_probes: list[np.ndarray] = []
        for layer_index, (layer_name, block) in enumerate(blocks):
            probe_path = checkpoint / f"bit{bit}_{SCORING_VERSION}_batch{batch_size}_layer{layer_index:02d}.npy"
            if probe_path.exists():
                values = np.load(probe_path, allow_pickle=False)
            else:
                started = time.perf_counter()
                with temporarily_quantized(block, bit, group_size=group_size):
                    values = _pad(student.score(examples), width).astype(np.float32)
                np.save(probe_path, values, allow_pickle=False)
                elapsed = time.perf_counter() - started
                print(
                    f"probe bit={bit} layer={layer_index + 1}/{len(blocks)} "
                    f"({layer_name}) completed in {elapsed:.1f}s",
                    flush=True,
                )
            if values.shape != student_probabilities.shape:
                raise ValueError(f"invalid checkpoint shape at {probe_path}: {values.shape}")
            bit_probes.append(values)
        all_probes.append(np.stack(bit_probes))
    del student
    _clear_cuda()

    artifact = MultiBitCalibrationArtifact(
        teacher_probabilities=np.asarray(teacher_probabilities),
        student_probabilities=np.asarray(student_probabilities),
        quantized_probabilities=np.stack(all_probes),
        probe_bits=bits,
        layer_names=layer_names,
        parameter_counts=parameter_counts,
        labels=np.asarray([item.answer if item.answer is not None else -1 for item in examples]),
        example_ids=tuple(item.id for item in examples),
        metadata={
            "teacher_model": teacher_model,
            "student_model": student_model,
            "data_path": str(Path(data_path).resolve()),
            "group_size": group_size,
            "batch_size": batch_size,
            "scoring_version": SCORING_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "quantization": "symmetric per-row/per-group fake weight quantization",
            "scope_note": "Fake quantization supports sensitivity analysis, not packed-size or runtime claims.",
        },
    )
    artifact.save(output)
    print(f"saved multi-bit calibration artifact to {output}", flush=True)
    return artifact


def perturbations(artifact: MultiBitCalibrationArtifact) -> np.ndarray:
    """Return KL(student FP || one-layer quantized), shaped [bits, layers, examples]."""

    values = np.empty(artifact.quantized_probabilities.shape[:3], dtype=np.float64)
    for bit_index in range(values.shape[0]):
        for layer_index in range(values.shape[1]):
            values[bit_index, layer_index] = kl_divergence(
                artifact.student_probabilities,
                artifact.quantized_probabilities[bit_index, layer_index],
            )
    return values


def stratified_subset_indices(
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    size: int,
    seed: int,
    entropy_bins: int = 5,
) -> np.ndarray:
    """Sample without replacement within correctness x teacher-entropy strata."""

    count = labels.size
    if not 1 <= size <= count:
        raise ValueError("subset size must be between 1 and the calibration-pool size")
    teacher_correct = teacher_probabilities.argmax(axis=1) == labels
    student_correct = student_probabilities.argmax(axis=1) == labels
    uncertainty = entropy(teacher_probabilities, normalized=True)
    order = np.argsort(uncertainty, kind="stable")
    entropy_group = np.empty(count, dtype=np.int64)
    entropy_group[order] = np.minimum(np.arange(count) * entropy_bins // count, entropy_bins - 1)
    keys = [
        (int(teacher_correct[index]), int(student_correct[index]), int(entropy_group[index]))
        for index in range(count)
    ]
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    raw = {key: size * len(indices) / count for key, indices in groups.items()}
    quotas = {key: min(len(groups[key]), int(math.floor(value))) for key, value in raw.items()}
    remaining = size - sum(quotas.values())
    candidates = sorted(groups, key=lambda key: (raw[key] - quotas[key], len(groups[key]), key), reverse=True)
    cursor = 0
    while remaining:
        key = candidates[cursor % len(candidates)]
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
        cursor += 1
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for key in sorted(groups):
        selected.extend(rng.choice(groups[key], size=quotas[key], replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _mean_or_zeros(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(values.shape[:2], dtype=np.float64)
    return values[:, :, mask].mean(axis=2).T


def _error_matrices(
    artifact: MultiBitCalibrationArtifact,
    delta: np.ndarray,
    indices: np.ndarray,
    *,
    interaction_lambda: float,
) -> tuple[dict[str, np.ndarray], dict]:
    if not 0.0 <= interaction_lambda <= 1.0:
        raise ValueError("interaction_lambda must be in [0, 1] for bounded relative interactions")
    selected = delta[:, :, indices]
    teacher = artifact.teacher_probabilities[indices]
    student = artifact.student_probabilities[indices]
    labels = artifact.labels[indices]
    uncertainty = entropy(teacher, normalized=True)
    disagreement = kl_divergence(teacher, student)
    q25, q75 = np.quantile(uncertainty, [0.25, 0.75])
    d25, d75 = np.quantile(disagreement, [0.25, 0.75])
    low_unc, high_unc = uncertainty <= q25, uncertainty >= q75
    low_dis, high_dis = disagreement <= d25, disagreement >= d75
    oracle = (teacher.argmax(axis=1) == labels) & (student.argmax(axis=1) != labels)

    base = selected.mean(axis=2).T
    high_uncertainty = _mean_or_zeros(selected, high_unc)
    low_uncertainty = _mean_or_zeros(selected, low_unc)
    high_disagreement = _mean_or_zeros(selected, high_dis)
    low_disagreement = _mean_or_zeros(selected, low_dis)
    epsilon = np.finfo(np.float64).eps
    uncertainty_interaction = (high_uncertainty - low_uncertainty) / (
        high_uncertainty + low_uncertainty + epsilon
    )
    disagreement_interaction = (high_disagreement - low_disagreement) / (
        high_disagreement + low_disagreement + epsilon
    )
    oracle_errors = _mean_or_zeros(selected, oracle) if oracle.any() else base.copy()
    raw = {
        "student_only": base,
        "teacher_confidence": base * (1.0 - interaction_lambda * uncertainty_interaction),
        "teacher_uncertainty": base * (1.0 + interaction_lambda * uncertainty_interaction),
        "disagreement": base * (1.0 + interaction_lambda * disagreement_interaction),
        "oracle": oracle_errors,
    }
    matrices = {name: np.maximum(values, np.finfo(np.float64).tiny) for name, values in raw.items()}
    diagnostics = {
        "teacher_entropy_q25": float(q25),
        "teacher_entropy_q75": float(q75),
        "disagreement_q25": float(d25),
        "disagreement_q75": float(d75),
        "teacher_correct_student_wrong": int(oracle.sum()),
        "clipped_error_cells": {name: int(np.sum(values <= 0)) for name, values in raw.items()},
        "interaction_lambda": float(interaction_lambda),
        "interaction_formulation": "base * (1 + lambda * (high-low)/(high+low))",
    }
    return matrices, diagnostics


def _top_overlap(left: np.ndarray, right: np.ndarray, k: int) -> int:
    count = min(k, left.size)
    a = set(np.argsort(left)[-count:].tolist())
    b = set(np.argsort(right)[-count:].tolist())
    return len(a & b)


def analyze_iteration(
    artifact: MultiBitCalibrationArtifact,
    *,
    output_path: str,
    subset_size: int = 500,
    calibration_seeds: Iterable[int] = (11, 23, 37, 53, 71),
    budgets: Iterable[float] = (4.0, 3.5, 3.0),
    interaction_lambda: float = 1.0,
) -> dict:
    """Build five stratified policies using empirical errors at every precision."""

    if np.any(artifact.labels < 0):
        raise ValueError("iteration analysis requires labels for all calibration examples")
    seeds = tuple(int(value) for value in calibration_seeds)
    budgets = tuple(float(value) for value in budgets)
    bits = tuple(artifact.probe_bits)
    delta = perturbations(artifact)
    ranking_bit_index = bits.index(4) if 4 in bits else int(np.argmin(np.abs(np.asarray(bits) - 4)))
    report: dict = {
        "metadata": artifact.metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe_bits": list(bits),
        "budgets": list(budgets),
        "subset_size": subset_size,
        "calibration_seeds": list(seeds),
        "ranking_bit": bits[ranking_bit_index],
        "layers": list(artifact.layer_names),
        "parameter_counts": artifact.parameter_counts.tolist(),
        "seeds": {},
        "random_baselines": {},
    }
    teacher_uncertainty = entropy(artifact.teacher_probabilities, normalized=True)
    for seed in seeds:
        indices = stratified_subset_indices(
            artifact.teacher_probabilities,
            artifact.student_probabilities,
            artifact.labels,
            size=subset_size,
            seed=seed,
        )
        matrices, diagnostics = _error_matrices(
            artifact, delta, indices, interaction_lambda=interaction_lambda
        )
        teacher_metrics = classification_metrics(artifact.teacher_probabilities[indices], artifact.labels[indices])
        student_metrics = classification_metrics(artifact.student_probabilities[indices], artifact.labels[indices])
        policies: dict = {}
        student_rank = matrices["student_only"][:, ranking_bit_index]
        for mode in POLICY_MODES:
            ranking = matrices[mode][:, ranking_bit_index]
            allocations = {}
            for budget in budgets:
                allocation = allocate_bits(
                    np.ones(len(artifact.layer_names)),
                    artifact.parameter_counts,
                    budget,
                    bit_choices=bits,
                    errors=matrices[mode],
                )
                allocations[str(budget)] = {
                    "bits": allocation.bits.tolist(),
                    "average_bits": allocation.average_bits,
                    "total_bits": allocation.total_bits,
                    "budget_bits": allocation.budget_bits,
                    "objective": allocation.objective,
                }
            policies[mode] = {
                "empirical_error_matrix": matrices[mode].tolist(),
                "ranking": ranking.tolist(),
                "ranking_vs_student_spearman": spearman_correlation(student_rank, ranking),
                "top5_overlap_with_student": _top_overlap(student_rank, ranking, 5),
                "top10_overlap_with_student": _top_overlap(student_rank, ranking, 10),
                "allocations": allocations,
            }
        for mode in POLICY_MODES[1:]:
            for budget in budgets:
                left = policies["student_only"]["allocations"][str(budget)]["bits"]
                right = policies[mode]["allocations"][str(budget)]["bits"]
                policies[mode]["allocations"][str(budget)]["changed_vs_student"] = int(
                    sum(a != b for a, b in zip(left, right))
                )

        rq1 = {}
        for bit_index, bit in enumerate(bits):
            per_layer = []
            high = teacher_uncertainty[indices] >= np.quantile(teacher_uncertainty[indices], 0.75)
            low = teacher_uncertainty[indices] <= np.quantile(teacher_uncertainty[indices], 0.25)
            for layer_index, name in enumerate(artifact.layer_names):
                values = delta[bit_index, layer_index, indices]
                per_layer.append(
                    {
                        "layer": name,
                        "spearman": spearman_correlation(teacher_uncertainty[indices], values),
                        "pearson": float(np.corrcoef(teacher_uncertainty[indices], values)[0, 1])
                        if np.std(values) > 0
                        else float("nan"),
                        "high_minus_low": float(values[high].mean() - values[low].mean()),
                    }
                )
            rq1[str(bit)] = per_layer
        report["seeds"][str(seed)] = {
            "subset_indices": indices.tolist(),
            "example_ids": [artifact.example_ids[index] for index in indices],
            "teacher_metrics": teacher_metrics,
            "student_metrics": student_metrics,
            "teacher_student_accuracy_gap": teacher_metrics["accuracy"] - student_metrics["accuracy"],
            "diagnostics": diagnostics,
            "rq1": rq1,
            "policies": policies,
        }
    for budget in budgets:
        runs = {}
        for seed in range(5):
            allocation = random_allocation(
                artifact.parameter_counts,
                budget,
                bit_choices=bits,
                seed=seed,
            )
            runs[str(seed)] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
            }
        report["random_baselines"][str(budget)] = runs

    stability = {}
    for mode in POLICY_MODES:
        correlations = []
        allocations = {str(budget): [] for budget in budgets}
        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                left = np.asarray(report["seeds"][str(left_seed)]["policies"][mode]["ranking"])
                right = np.asarray(report["seeds"][str(right_seed)]["policies"][mode]["ranking"])
                correlations.append(spearman_correlation(left, right))
                for budget in budgets:
                    a = report["seeds"][str(left_seed)]["policies"][mode]["allocations"][str(budget)]["bits"]
                    b = report["seeds"][str(right_seed)]["policies"][mode]["allocations"][str(budget)]["bits"]
                    allocations[str(budget)].append(sum(x != y for x, y in zip(a, b)))
        stability[mode] = {
            "pairwise_ranking_spearman_mean": float(np.nanmean(correlations)),
            "pairwise_ranking_spearman_min": float(np.nanmin(correlations)),
            "pairwise_allocation_hamming_mean": {
                budget: float(np.mean(values)) for budget, values in allocations.items()
            },
        }
    report["stability"] = stability
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


def _mcnemar(left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> dict:
    left_correct = left.argmax(axis=1) == labels
    right_correct = right.argmax(axis=1) == labels
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    total = left_only + right_only
    if total == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(total, value) for value in range(0, min(left_only, right_only) + 1)) / (2**total)
        p_value = min(1.0, 2.0 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "mcnemar_exact_p": p_value}


def _paired_bootstrap(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    labels: np.ndarray,
    *,
    seed: int = 2026,
    replicates: int = 5000,
) -> dict:
    left_correct = np.stack([values.argmax(axis=1) == labels for values in left])
    right_correct = np.stack([values.argmax(axis=1) == labels for values in right])
    differences = left_correct.astype(np.int8) - right_correct.astype(np.int8)
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        columns = rng.integers(0, labels.size, size=labels.size)
        samples[index] = np.mean(differences[:, columns])
    return {
        "mean_accuracy_difference": observed,
        "paired_bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "replicates": replicates,
    }


def benchmark_iteration(
    *,
    data_path: str,
    teacher_model: str,
    student_model: str,
    artifact: MultiBitCalibrationArtifact,
    policies: dict,
    output_dir: str,
    batch_size: int = 32,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Evaluate every unique matched-budget allocation on fixed held-out data."""

    examples = load_jsonl(data_path)
    labels = np.asarray([item.answer if item.answer is not None else -1 for item in examples])
    if np.any(labels < 0):
        raise ValueError("benchmark data requires labels")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    teacher = HFCausalLMScorer(
        teacher_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
    )
    teacher_probabilities = teacher.score(examples)
    del teacher
    _clear_cuda()
    student = HFCausalLMScorer(
        student_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
    )
    blocks = transformer_blocks(student.model)
    if tuple(name for name, _ in blocks) != artifact.layer_names:
        raise ValueError("evaluation model blocks do not match calibration artifact")
    full_precision = student.score(examples)
    group_size = int(artifact.metadata.get("group_size", 128))
    cache: dict[tuple[int, ...], tuple[str, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {
        "teacher": teacher_probabilities.astype(np.float32),
        "full_precision": full_precision.astype(np.float32),
    }
    records: list[dict] = []

    def probability_for(bits: Sequence[int]) -> tuple[str, np.ndarray]:
        key = tuple(int(value) for value in bits)
        if key not in cache:
            name = f"allocation_{len(cache):03d}"
            started = time.perf_counter()
            with temporarily_mixed_quantized(blocks, key, group_size=group_size):
                values = student.score(examples)
            cache[key] = (name, values)
            arrays[name] = values.astype(np.float32)
            print(f"evaluated {name} ({average_bits(key, artifact.parameter_counts):.3f} bits) in {time.perf_counter() - started:.1f}s", flush=True)
        return cache[key]

    def add_record(
        *, name: str, family: str, budget: float, bits: Sequence[int], calibration_seed: int | None = None, random_seed: int | None = None
    ) -> None:
        probability_key, values = probability_for(bits)
        records.append(
            {
                "name": name,
                "family": family,
                "target_average_bits": budget,
                "actual_average_bits": average_bits(bits, artifact.parameter_counts),
                "calibration_seed": calibration_seed,
                "random_seed": random_seed,
                "probability_key": probability_key,
                "bits": list(bits),
                **classification_metrics(values, labels),
                **uncertainty_preservation(values, full_precision),
            }
        )

    bits = tuple(artifact.probe_bits)
    budgets = tuple(float(value) for value in policies["budgets"])
    for budget in budgets:
        if budget.is_integer() and int(budget) in bits:
            add_record(
                name=f"uniform_{budget}",
                family="uniform",
                budget=budget,
                bits=[int(budget)] * len(blocks),
            )
        for random_seed, row in policies["random_baselines"][str(budget)].items():
            add_record(
                name=f"random_{budget}_seed{random_seed}",
                family="random",
                budget=budget,
                bits=row["bits"],
                random_seed=int(random_seed),
            )
        for calibration_seed in policies["calibration_seeds"]:
            seed_row = policies["seeds"][str(calibration_seed)]
            for mode in POLICY_MODES:
                row = seed_row["policies"][mode]["allocations"][str(budget)]
                add_record(
                    name=f"{mode}_{budget}_cal{calibration_seed}",
                    family=mode,
                    budget=budget,
                    bits=row["bits"],
                    calibration_seed=int(calibration_seed),
                )

    del student
    _clear_cuda()
    np.savez_compressed(
        destination / "probabilities.npz",
        labels=labels,
        example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
        **arrays,
    )

    summaries = {}
    for family in (*POLICY_MODES, "random", "uniform"):
        for budget in budgets:
            rows = [row for row in records if row["family"] == family and row["target_average_bits"] == budget]
            if not rows:
                continue
            summaries[f"{family}:{budget}"] = {
                field: {
                    "mean": float(np.nanmean([row[field] for row in rows])),
                    "std": float(np.nanstd([row[field] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                }
                for field in ("accuracy", "ece", "brier", "error_auroc", "aurc", "student_uncertainty_spearman")
            }

    comparisons = {}
    by_name = {row["name"]: row for row in records}
    for budget in budgets:
        for baseline in ("student_only", "uniform", "random"):
            left_values = []
            right_values = []
            per_seed = {}
            for calibration_seed in policies["calibration_seeds"]:
                proposed_name = f"teacher_uncertainty_{budget}_cal{calibration_seed}"
                proposed = arrays[by_name[proposed_name]["probability_key"]]
                if baseline == "student_only":
                    comparison_name = f"student_only_{budget}_cal{calibration_seed}"
                elif baseline == "uniform":
                    comparison_name = f"uniform_{budget}"
                    if comparison_name not in by_name:
                        continue
                else:
                    comparison_name = f"random_{budget}_seed{policies['calibration_seeds'].index(calibration_seed)}"
                comparison = arrays[by_name[comparison_name]["probability_key"]]
                left_values.append(proposed)
                right_values.append(comparison)
                per_seed[str(calibration_seed)] = _mcnemar(proposed, comparison, labels)
            if left_values:
                comparisons[f"teacher_uncertainty_vs_{baseline}:{budget}"] = {
                    **_paired_bootstrap(left_values, right_values, labels),
                    "mcnemar_by_calibration_seed": per_seed,
                }

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_model": teacher_model,
        "student_model": student_model,
        "data_path": str(Path(data_path).resolve()),
        "example_count": len(examples),
        "teacher_metrics": classification_metrics(teacher_probabilities, labels),
        "full_precision_metrics": classification_metrics(full_precision, labels),
        "unique_allocations_evaluated": len(cache),
        "records": records,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (destination / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    return result
