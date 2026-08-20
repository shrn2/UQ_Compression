"""Cascade-aware mixed-precision quantization experiments (Iteration 4)."""

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
from .cascade import CascadeProbabilityArtifact, uncertainty_scores
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import binary_auroc, binary_average_precision, classification_metrics, spearman_correlation
from .quantization import parameter_count, temporarily_mixed_quantized, temporarily_quantized, transformer_blocks
from .scoring import kl_divergence


SCORING_VERSION = "shared-prompt-single-token-v2"


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def collect_cascade_layer_probes(
    *,
    cascade_artifact_path: str,
    calibration_data_path: str,
    slm_model: str,
    output_path: str,
    bits: Sequence[int] = (2, 4, 8),
    group_size: int = 128,
    batch_size: int = 128,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> MultiBitCalibrationArtifact:
    """Reuse cached FP/fallback outputs and score one-layer SLM probes."""

    cascade = CascadeProbabilityArtifact.load(cascade_artifact_path)
    examples = load_jsonl(calibration_data_path)
    ids = tuple(item.id for item in examples)
    if ids != cascade.calibration_ids:
        raise ValueError("calibration IDs do not match the cascade artifact")
    widths = tuple(sorted(set(int(bit) for bit in bits)))
    if not widths or widths[0] < 2:
        raise ValueError("bits must contain widths of at least two")
    destination = Path(output_path)
    checkpoint = destination.with_suffix("").with_name(destination.stem + "_checkpoints")
    checkpoint.mkdir(parents=True, exist_ok=True)

    scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    names = tuple(name for name, _ in blocks)
    counts = np.asarray([parameter_count(block) for _, block in blocks], dtype=np.int64)
    all_probes: list[np.ndarray] = []
    for bit in widths:
        bit_rows: list[np.ndarray] = []
        for layer_index, (layer_name, block) in enumerate(blocks):
            path = checkpoint / f"bit{bit}_{SCORING_VERSION}_batch{batch_size}_layer{layer_index:02d}.npy"
            if path.exists():
                values = np.load(path, allow_pickle=False)
            else:
                started = time.perf_counter()
                with temporarily_quantized(block, bit, group_size=group_size):
                    values = scorer.score(examples).astype(np.float32)
                np.save(path, values, allow_pickle=False)
                print(
                    f"probe bit={bit} layer={layer_index + 1}/{len(blocks)} "
                    f"({layer_name}) completed in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if values.shape != cascade.slm_fp_calibration.shape:
                raise ValueError(f"invalid checkpoint shape at {path}: {values.shape}")
            bit_rows.append(values)
        all_probes.append(np.stack(bit_rows))
    del scorer
    _clear_cuda()

    artifact = MultiBitCalibrationArtifact(
        teacher_probabilities=cascade.fallback_calibration,
        student_probabilities=cascade.slm_fp_calibration,
        quantized_probabilities=np.stack(all_probes),
        probe_bits=widths,
        layer_names=names,
        parameter_counts=counts,
        labels=cascade.calibration_labels,
        example_ids=cascade.calibration_ids,
        metadata={
            "student_model": slm_model,
            "fallback_model": cascade.metadata["fallback_model"],
            "cascade_artifact": str(Path(cascade_artifact_path).resolve()),
            "data_path": str(Path(calibration_data_path).resolve()),
            "group_size": group_size,
            "batch_size": batch_size,
            "scoring_version": SCORING_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "quantization": "one-transformer-block symmetric per-row/per-group fake weight quantization",
            "scope_note": "Fake quantization supports causal policy comparison, not packed size or runtime claims.",
        },
    )
    artifact.save(destination)
    return artifact


def threshold_for_rate(scores: np.ndarray, rate: float) -> float:
    """Return a deterministic FP calibration threshold for approximately ``rate``."""

    values = np.asarray(scores, dtype=np.float64)
    count = max(1, min(values.size, int(round(rate * values.size))))
    ordered = np.sort(values, kind="stable")
    if count == values.size:
        return float(np.nextafter(ordered[0], -np.inf))
    boundary = values.size - count
    return float((ordered[boundary - 1] + ordered[boundary]) / 2.0)


def routing_outcomes(
    probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    decisions = np.asarray(scores) > threshold
    slm_predictions = np.asarray(probabilities).argmax(axis=1)
    fallback_predictions = np.asarray(fallback_probabilities).argmax(axis=1)
    slm_correct = slm_predictions == labels
    fallback_correct = fallback_predictions == labels
    final_correct = np.where(decisions, fallback_correct, slm_correct)
    delegated = max(1, int(decisions.sum()))
    values = {
        "slm_accuracy": float(slm_correct.mean()),
        "cascade_accuracy": float(final_correct.mean()),
        "delegation_rate": float(decisions.mean()),
        "fallback_correction_rate": float(np.sum(decisions & ~slm_correct & fallback_correct) / delegated),
        "harmful_escalation_rate": float(np.mean(decisions & slm_correct & ~fallback_correct)),
        "missed_beneficial_delegation_rate": float(np.mean(~decisions & ~slm_correct & fallback_correct)),
    }
    return values, final_correct, decisions


def aurc_from_scores(scores: np.ndarray, errors: np.ndarray) -> float:
    """Area under risk-coverage when lower scores are accepted first."""

    order = np.argsort(np.asarray(scores), kind="stable")
    cumulative_risk = np.cumsum(np.asarray(errors, dtype=np.float64)[order]) / np.arange(1, len(order) + 1)
    return float(cumulative_risk.mean())


def analyze_cascade_compression(
    artifact: MultiBitCalibrationArtifact,
    *,
    output_path: str,
    router_signal: str = "margin",
    target_rates: Iterable[float] = (0.1, 0.2, 0.3, 0.5),
    cost_coefficients: Iterable[float] = (0.0, 0.01, 0.025, 0.05, 0.1),
    budgets: Iterable[float] = (3.5, 4.0),
    random_seeds: Iterable[int] = range(10),
) -> dict:
    """Build frozen Stage-1 KL, CUAQ, and random allocations on calibration data."""

    rates = tuple(float(value) for value in target_rates)
    costs = tuple(float(value) for value in cost_coefficients)
    budgets = tuple(float(value) for value in budgets)
    seeds = tuple(int(value) for value in random_seeds)
    if router_signal not in uncertainty_scores(artifact.student_probabilities):
        raise ValueError(f"unknown router signal: {router_signal}")
    fp_scores = uncertainty_scores(artifact.student_probabilities)[router_signal]
    thresholds = {rate: threshold_for_rate(fp_scores, rate) for rate in rates}
    labels = artifact.labels
    fallback = artifact.teacher_probabilities
    fp_outcomes = {}
    fp_utility_vectors = []
    for rate in rates:
        metrics, correct, decisions = routing_outcomes(
            artifact.student_probabilities, fallback, labels, fp_scores, thresholds[rate]
        )
        fp_outcomes[str(rate)] = metrics
        for cost in costs:
            fp_utility_vectors.append(correct.astype(np.float64) - cost * decisions.astype(np.float64))
    fp_mean_utility = float(np.mean(fp_utility_vectors))

    bit_count, layer_count = artifact.quantized_probabilities.shape[:2]
    kl = np.empty((layer_count, bit_count), dtype=np.float64)
    accuracy_drop = np.empty_like(kl)
    uq_disruption = np.empty_like(kl)
    cuaq = np.empty_like(kl)
    dbaq = np.empty_like(kl)
    dbaq_decision = np.empty_like(kl)
    fp_correct = artifact.student_probabilities.argmax(axis=1) == labels
    fallback_correct = fallback.argmax(axis=1) == labels
    delegation_value = fallback_correct.astype(np.float64) - fp_correct.astype(np.float64)
    fp_accuracy = float(fp_correct.mean())
    layer_rows = []
    for bit_index, bit in enumerate(artifact.probe_bits):
        for layer_index, layer_name in enumerate(artifact.layer_names):
            probabilities = artifact.quantized_probabilities[bit_index, layer_index]
            scores = uncertainty_scores(probabilities)[router_signal]
            kl_values = kl_divergence(artifact.student_probabilities, probabilities)
            kl[layer_index, bit_index] = float(kl_values.mean())
            accuracy_drop[layer_index, bit_index] = fp_accuracy - float(np.mean(probabilities.argmax(axis=1) == labels))
            uq_disruption[layer_index, bit_index] = 1.0 - spearman_correlation(fp_scores, scores)
            probe_utility_vectors = []
            weighted_kl = []
            weighted_flips = []
            for rate in rates:
                threshold = thresholds[rate]
                _, correct, decisions = routing_outcomes(probabilities, fallback, labels, scores, threshold)
                for cost in costs:
                    probe_utility_vectors.append(correct.astype(np.float64) - cost * decisions.astype(np.float64))
                distance = np.abs(fp_scores - threshold)
                nonzero = distance[distance > 0]
                temperature = float(np.median(nonzero)) if nonzero.size else 1.0
                boundary = np.exp(-distance / max(temperature, np.finfo(float).eps))
                consequence = np.abs(delegation_value)
                weights = consequence * boundary
                weighted_kl.append(float(np.sum(weights * kl_values) / max(np.sum(weights), np.finfo(float).eps)))
                fp_decisions = fp_scores > threshold
                weighted_flips.append(float(np.mean(consequence * (decisions != fp_decisions))))
            probe_mean_utility = float(np.mean(probe_utility_vectors))
            cuaq[layer_index, bit_index] = fp_mean_utility - probe_mean_utility
            dbaq[layer_index, bit_index] = float(np.mean(weighted_kl))
            dbaq_decision[layer_index, bit_index] = float(np.mean(weighted_flips))
            layer_rows.append(
                {
                    "bit": bit,
                    "layer": layer_name,
                    "layer_index": layer_index,
                    "kl_sensitivity": kl[layer_index, bit_index],
                    "accuracy_drop_sensitivity": accuracy_drop[layer_index, bit_index],
                    "uq_sensitivity": uq_disruption[layer_index, bit_index],
                    "cuaq_sensitivity": cuaq[layer_index, bit_index],
                    "dbaq_sensitivity": dbaq[layer_index, bit_index],
                    "dbaq_decision_sensitivity": dbaq_decision[layer_index, bit_index],
                }
            )

    matrices = {
        "kl": kl,
        "accuracy": accuracy_drop,
        "uq": uq_disruption,
        "cuaq": cuaq,
        "dbaq": dbaq,
    }
    policies: dict[str, dict] = {}
    for budget in budgets:
        budget_key = str(budget)
        policies[budget_key] = {}
        for method, matrix in matrices.items():
            allocation = allocate_bits(
                np.ones(layer_count),
                artifact.parameter_counts,
                budget,
                bit_choices=artifact.probe_bits,
                errors=matrix,
                spend_full_budget=True,
            )
            policies[budget_key][method] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
                "objective": allocation.objective,
            }
        random_rows = {}
        for seed in seeds:
            allocation = random_allocation(
                artifact.parameter_counts, budget, bit_choices=artifact.probe_bits, seed=seed
            )
            random_rows[str(seed)] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
            }
        policies[budget_key]["random"] = random_rows

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": artifact.metadata,
        "stage": "stage1_local_cuaq",
        "scope": {
            "examples": len(labels),
            "bits": list(artifact.probe_bits),
            "budgets": list(budgets),
            "target_rates": list(rates),
            "cost_coefficients": list(costs),
            "random_seeds": list(seeds),
            "router_signal": router_signal,
            "threshold_policy": "FP calibration thresholds frozen across compression methods",
        },
        "thresholds": {str(rate): value for rate, value in thresholds.items()},
        "fp_reference": {"mean_utility": fp_mean_utility, "routing": fp_outcomes},
        "delegation_value_counts": {
            "slm_correct_fallback_correct": int(np.sum(fp_correct & fallback_correct)),
            "slm_wrong_fallback_correct": int(np.sum(~fp_correct & fallback_correct)),
            "slm_correct_fallback_wrong": int(np.sum(fp_correct & ~fallback_correct)),
            "slm_wrong_fallback_wrong": int(np.sum(~fp_correct & ~fallback_correct)),
        },
        "layer_rows": layer_rows,
        "policies": policies,
        "allocation_rank_correlations": {
            method: spearman_correlation(kl[:, artifact.probe_bits.index(4)], matrix[:, artifact.probe_bits.index(4)])
            for method, matrix in matrices.items() if method != "kl"
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def _paired_interval(left: np.ndarray, right: np.ndarray, *, seed: int, replicates: int) -> dict:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        columns = rng.integers(0, differences.shape[-1], size=differences.shape[-1])
        samples[index] = differences[..., columns].mean()
    return {
        "difference": float(differences.mean()),
        "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "replicates": replicates,
    }


def _mcnemar(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    total = left_only + right_only
    if total == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(total, value) for value in range(min(left_only, right_only) + 1)) / (2**total)
        p_value = min(1.0, 2.0 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "exact_p": p_value}


def benchmark_cascade_compression(
    *,
    data_path: str,
    cascade_artifact_path: str,
    probe_artifact: MultiBitCalibrationArtifact,
    policies: dict,
    slm_model: str,
    output_dir: str,
    methods: Sequence[str] = ("kl", "cuaq"),
    batch_size: int = 128,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260819,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Evaluate frozen compression policies with cached fallback outputs."""

    cascade = CascadeProbabilityArtifact.load(cascade_artifact_path)
    examples = load_jsonl(data_path)
    labels = np.asarray([item.answer for item in examples], dtype=np.int64)
    ids = tuple(item.id for item in examples)
    if ids != cascade.evaluation_ids:
        raise ValueError("evaluation IDs do not match cascade artifact")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    if tuple(name for name, _ in blocks) != probe_artifact.layer_names:
        raise ValueError("evaluation model blocks do not match probe artifact")
    cache: dict[tuple[int, ...], tuple[str, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {
        "full_precision": cascade.slm_fp_evaluation.astype(np.float32),
        "fallback": cascade.fallback_evaluation.astype(np.float32),
    }

    def probability_for(bits: Sequence[int]) -> tuple[str, np.ndarray]:
        key = tuple(int(value) for value in bits)
        if key not in cache:
            name = f"allocation_{len(cache):03d}"
            started = time.perf_counter()
            if len(set(key)) == 1 and key[0] in cascade.bits:
                values = cascade.quantized_evaluation[cascade.bits.index(key[0])]
            else:
                with temporarily_mixed_quantized(blocks, key, group_size=int(probe_artifact.metadata["group_size"])):
                    values = scorer.score(examples)
            cache[key] = (name, values)
            arrays[name] = values.astype(np.float32)
            print(f"evaluated {name} ({average_bits(key, probe_artifact.parameter_counts):.3f} bits) in {time.perf_counter() - started:.1f}s", flush=True)
        return cache[key]

    configurations = [
        {"name": "full_precision", "family": "fp", "budget": 16.0, "bits": [16] * len(blocks), "probability_key": "full_precision"}
    ]
    for budget in policies["scope"]["budgets"]:
        budget_key = str(budget)
        if float(budget).is_integer() and int(budget) in probe_artifact.probe_bits:
            key, _ = probability_for([int(budget)] * len(blocks))
            configurations.append({"name": f"uniform_{budget_key}", "family": "uniform", "budget": budget, "bits": [int(budget)] * len(blocks), "probability_key": key})
        for method in methods:
            row = policies["policies"][budget_key][method]
            key, _ = probability_for(row["bits"])
            configurations.append({"name": f"{method}_{budget_key}", "family": method, "budget": budget, "bits": row["bits"], "probability_key": key})
        for seed, row in policies["policies"][budget_key]["random"].items():
            key, _ = probability_for(row["bits"])
            configurations.append({"name": f"random_{budget_key}_seed{seed}", "family": "random", "budget": budget, "random_seed": int(seed), "bits": row["bits"], "probability_key": key})
    del scorer
    _clear_cuda()

    fp_probabilities = cascade.slm_fp_evaluation
    fallback = cascade.fallback_evaluation
    fp_predictions = fp_probabilities.argmax(axis=1)
    router_signal = policies["scope"]["router_signal"]
    thresholds = {float(key): value for key, value in policies["thresholds"].items()}
    costs = tuple(policies["scope"]["cost_coefficients"])
    records = []
    utility_vectors: dict[tuple[str, float, float], np.ndarray] = {}
    correct_vectors: dict[tuple[str, float], np.ndarray] = {}
    for configuration in configurations:
        probabilities = arrays[configuration["probability_key"]]
        scores = uncertainty_scores(probabilities)[router_signal]
        predicted = probabilities.argmax(axis=1)
        base_metrics = classification_metrics(probabilities, labels)
        config = {
            **configuration,
            "actual_average_bits": 16.0 if configuration["family"] == "fp" else average_bits(configuration["bits"], probe_artifact.parameter_counts),
            "estimated_block_weight_gib": float(np.dot(configuration["bits"], probe_artifact.parameter_counts) / 8 / (1024**3)),
            "compression_ratio_vs_fp16": 1.0 if configuration["family"] == "fp" else 16.0 / average_bits(configuration["bits"], probe_artifact.parameter_counts),
            "standalone_accuracy": base_metrics["accuracy"],
            "prediction_agreement_with_fp": float(np.mean(predicted == fp_predictions)),
            "output_kl_from_fp": float(np.mean(kl_divergence(fp_probabilities, probabilities))),
            "error_auroc": binary_auroc(scores, predicted != labels),
            "error_auprc": binary_average_precision(scores, predicted != labels),
            "aurc": aurc_from_scores(scores, predicted != labels),
            "uncertainty_spearman_with_fp": spearman_correlation(uncertainty_scores(fp_probabilities)[router_signal], scores),
        }
        for rate, threshold in thresholds.items():
            metrics, correct, decisions = routing_outcomes(probabilities, fallback, labels, scores, threshold)
            fp_decisions = uncertainty_scores(fp_probabilities)[router_signal] > threshold
            correct_vectors[(configuration["name"], rate)] = correct
            for cost in costs:
                utility = correct.astype(np.float64) - cost * decisions.astype(np.float64)
                utility_vectors[(configuration["name"], rate, cost)] = utility
                records.append({
                    **config,
                    "target_rate": rate,
                    "cost_coefficient": cost,
                    **metrics,
                    "cascade_utility": float(utility.mean()),
                    "frozen_fp_threshold_delegation_flip_rate": float(np.mean(decisions != fp_decisions)),
                })

    comparisons = {}
    gate_rows = []
    for budget in policies["scope"]["budgets"]:
        budget_key = str(budget)
        cuaq_name = f"cuaq_{budget_key}"
        kl_name = f"kl_{budget_key}"
        random_names = [row["name"] for row in configurations if row["family"] == "random" and row["budget"] == budget]
        if cuaq_name not in {row["name"] for row in configurations}:
            continue
        cuaq_matrix = np.stack([utility_vectors[(cuaq_name, rate, cost)] for rate in thresholds for cost in costs])
        kl_matrix = np.stack([utility_vectors[(kl_name, rate, cost)] for rate in thresholds for cost in costs])
        random_matrices = [np.stack([utility_vectors[(name, rate, cost)] for rate in thresholds for cost in costs]) for name in random_names]
        random_mean_matrix = np.mean(random_matrices, axis=0)
        random_objectives = [float(matrix.mean()) for matrix in random_matrices]
        best_index = int(np.argmax(random_objectives))
        best_random_matrix = random_matrices[best_index]
        vs_kl = _paired_interval(cuaq_matrix, kl_matrix, seed=bootstrap_seed, replicates=bootstrap_replicates)
        vs_random_mean = _paired_interval(cuaq_matrix, random_mean_matrix, seed=bootstrap_seed + 1, replicates=bootstrap_replicates)
        vs_best_random = _paired_interval(cuaq_matrix, best_random_matrix, seed=bootstrap_seed + 2, replicates=bootstrap_replicates)
        comparisons[budget_key] = {
            "cuaq_vs_kl": vs_kl,
            "cuaq_vs_random_mean": vs_random_mean,
            "cuaq_vs_best_random": vs_best_random,
            "random_objective_mean": float(np.mean(random_objectives)),
            "random_objective_std": float(np.std(random_objectives, ddof=1)),
            "best_random_name": random_names[best_index],
            "best_random_objective": random_objectives[best_index],
            "mcnemar_by_rate": {
                str(rate): _mcnemar(correct_vectors[(cuaq_name, rate)], correct_vectors[(kl_name, rate)])
                for rate in thresholds
            },
        }
        passed = vs_kl["difference"] > 0 and vs_random_mean["difference"] > 0
        statistically_confirmed = vs_kl["bootstrap_95_ci"][0] > 0 and vs_random_mean["bootstrap_95_ci"][0] > 0
        gate_rows.append({"budget": budget, "passed": passed, "statistically_confirmed": statistically_confirmed})

    gate = {
        "passed": any(row["passed"] for row in gate_rows),
        "statistically_confirmed": any(row["statistically_confirmed"] for row in gate_rows),
        "rule": "Proceed when local CUAQ has higher held-out mean utility than KL-sensitive and the ten-seed random mean at either matched budget.",
        "rows": gate_rows,
        "next_stage": "DBAQ" if any(row["passed"] for row in gate_rows) else "STOP",
    }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage1_local_cuaq",
        "models": {"slm": slm_model, "fallback": cascade.metadata["fallback_model"]},
        "scope": {"evaluation_examples": len(labels), "methods": list(methods), "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed},
        "configurations": configurations,
        "records": records,
        "comparisons": comparisons,
        "gate": gate,
    }
    np.savez_compressed(destination / "probabilities.npz", labels=labels, example_ids=np.asarray(ids, dtype=np.str_), **arrays)
    (destination / "results.json").write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output
