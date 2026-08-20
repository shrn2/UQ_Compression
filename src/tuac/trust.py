"""Trust-critical layer analysis and matched-budget policy evaluation."""

from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .allocation import allocate_bits, average_bits, random_allocation
from .artifacts import MultiBitCalibrationArtifact
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import (
    classification_metrics,
    margin_uncertainty,
    pairwise_ordering_flip_rate,
    spearman_correlation,
    uncertainty_preservation,
)
from .quantization import temporarily_mixed_quantized, transformer_blocks
from .scoring import entropy, kl_divergence

TRUST_POLICY_MODES = (
    "capability_kl",
    "capability_flip",
    "capability_accuracy_drop",
    "trust_entropy",
    "joint_0.25",
    "joint_0.50",
    "joint_0.75",
)


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _top_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    count = min(k, left.size)
    a = set(np.argsort(left)[-count:].tolist())
    b = set(np.argsort(right)[-count:].tolist())
    return len(a & b) / count


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _mean_ci(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def analyze_trust_layers(
    artifact: MultiBitCalibrationArtifact,
    *,
    output_path: str,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260817,
) -> dict:
    """Compute capability and trust sensitivity maps from cached one-block probes."""

    if np.any(artifact.labels < 0):
        raise ValueError("trust-layer analysis requires labels for every probing example")
    bits = tuple(artifact.probe_bits)
    layers = len(artifact.layer_names)
    examples = artifact.labels.size
    fp = artifact.student_probabilities
    labels = artifact.labels
    fp_entropy = entropy(fp, normalized=True)
    fp_margin = margin_uncertainty(fp)
    fp_predictions = fp.argmax(axis=1)
    fp_correct = fp_predictions == labels
    fp_metrics = classification_metrics(fp, labels)

    shape = (len(bits), layers)
    capability_kl = np.zeros(shape)
    capability_flip = np.zeros(shape)
    capability_accuracy_drop = np.zeros(shape)
    trust_entropy_rank = np.zeros(shape)
    trust_margin_rank = np.zeros(shape)
    trust_auroc_drop = np.zeros(shape)
    trust_aurc_increase = np.zeros(shape)
    trust_ece_increase = np.zeros(shape)
    trust_brier_increase = np.zeros(shape)
    trust_pairwise_flip = np.zeros(shape)
    trust_correct_rank = np.zeros(shape)
    trust_incorrect_rank = np.zeros(shape)
    teacher_agreement_change = np.zeros(shape)
    per_example_kl = np.zeros((len(bits), layers, examples))
    per_example_flip = np.zeros((len(bits), layers, examples), dtype=bool)
    quantized_entropy = np.zeros((len(bits), layers, examples))
    quantized_margin = np.zeros((len(bits), layers, examples))

    teacher_base = np.mean(kl_divergence(artifact.teacher_probabilities, fp))
    for bit_index in range(len(bits)):
        for layer_index in range(layers):
            probabilities = artifact.quantized_probabilities[bit_index, layer_index]
            current_entropy = entropy(probabilities, normalized=True)
            current_margin = margin_uncertainty(probabilities)
            predictions = probabilities.argmax(axis=1)
            metrics = classification_metrics(probabilities, labels)
            per_example_kl[bit_index, layer_index] = kl_divergence(fp, probabilities)
            per_example_flip[bit_index, layer_index] = predictions != fp_predictions
            quantized_entropy[bit_index, layer_index] = current_entropy
            quantized_margin[bit_index, layer_index] = current_margin
            capability_kl[bit_index, layer_index] = per_example_kl[bit_index, layer_index].mean()
            capability_flip[bit_index, layer_index] = per_example_flip[bit_index, layer_index].mean()
            capability_accuracy_drop[bit_index, layer_index] = fp_metrics["accuracy"] - metrics["accuracy"]
            trust_entropy_rank[bit_index, layer_index] = 1.0 - spearman_correlation(fp_entropy, current_entropy)
            trust_margin_rank[bit_index, layer_index] = 1.0 - spearman_correlation(fp_margin, current_margin)
            trust_auroc_drop[bit_index, layer_index] = fp_metrics["error_auroc"] - metrics["error_auroc"]
            trust_aurc_increase[bit_index, layer_index] = metrics["aurc"] - fp_metrics["aurc"]
            trust_ece_increase[bit_index, layer_index] = metrics["ece"] - fp_metrics["ece"]
            trust_brier_increase[bit_index, layer_index] = metrics["brier"] - fp_metrics["brier"]
            trust_pairwise_flip[bit_index, layer_index] = pairwise_ordering_flip_rate(
                fp_entropy, current_entropy
            )
            if fp_correct.sum() >= 3:
                trust_correct_rank[bit_index, layer_index] = 1.0 - spearman_correlation(
                    fp_entropy[fp_correct], current_entropy[fp_correct]
                )
            if (~fp_correct).sum() >= 3:
                trust_incorrect_rank[bit_index, layer_index] = 1.0 - spearman_correlation(
                    fp_entropy[~fp_correct], current_entropy[~fp_correct]
                )
            teacher_agreement_change[bit_index, layer_index] = (
                np.mean(kl_divergence(artifact.teacher_probabilities, probabilities)) - teacher_base
            )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap: dict[str, dict] = {}
    layer_rows: list[dict] = []
    taxonomy_rows: list[dict] = []
    region_rows: list[dict] = []
    gate_reasons: list[str] = []
    gate_passed = False
    top_count = max(1, math.ceil(layers * 0.25))
    region_names = np.empty(layers, dtype=object)
    for region, indices in zip(("early", "middle", "late"), np.array_split(np.arange(layers), 3)):
        region_names[indices] = region

    for bit_index, bit in enumerate(bits):
        rho_samples = np.empty(bootstrap_replicates)
        margin_rho_samples = np.empty(bootstrap_replicates)
        overlap_samples = {k: np.empty(bootstrap_replicates) for k in (3, 5, 8)}
        capability_samples = np.empty((bootstrap_replicates, layers))
        trust_samples = np.empty((bootstrap_replicates, layers))
        margin_samples = np.empty((bootstrap_replicates, layers))
        capability_top = np.zeros(layers)
        trust_top = np.zeros(layers)
        for replicate in range(bootstrap_replicates):
            indices = rng.integers(0, examples, size=examples)
            cap = per_example_kl[bit_index][:, indices].mean(axis=1)
            rel = np.asarray(
                [
                    1.0
                    - spearman_correlation(
                        fp_entropy[indices], quantized_entropy[bit_index, layer, indices]
                    )
                    for layer in range(layers)
                ]
            )
            margin_rel = np.asarray(
                [
                    1.0
                    - spearman_correlation(
                        fp_margin[indices], quantized_margin[bit_index, layer, indices]
                    )
                    for layer in range(layers)
                ]
            )
            capability_samples[replicate] = cap
            trust_samples[replicate] = rel
            margin_samples[replicate] = margin_rel
            rho_samples[replicate] = spearman_correlation(cap, rel)
            margin_rho_samples[replicate] = spearman_correlation(cap, margin_rel)
            for k in overlap_samples:
                overlap_samples[k][replicate] = _top_overlap(cap, rel, k)
            capability_top[np.argsort(cap)[-top_count:]] += 1
            trust_top[np.argsort(rel)[-top_count:]] += 1

        observed_rho = spearman_correlation(capability_kl[bit_index], trust_entropy_rank[bit_index])
        observed_margin_rho = spearman_correlation(
            capability_kl[bit_index], trust_margin_rank[bit_index]
        )
        observed_overlap = {
            str(k): _top_overlap(capability_kl[bit_index], trust_entropy_rank[bit_index], k)
            for k in (3, 5, 8)
        }
        rho_ci = _mean_ci(rho_samples)
        bootstrap[str(bit)] = {
            "capability_trust_spearman": observed_rho,
            "capability_trust_spearman_bootstrap_95_ci": rho_ci,
            "capability_margin_spearman": observed_margin_rho,
            "capability_margin_spearman_bootstrap_95_ci": _mean_ci(margin_rho_samples),
            "top_k_overlap": observed_overlap,
            "top_k_overlap_bootstrap_95_ci": {
                str(k): _mean_ci(values) for k, values in overlap_samples.items()
            },
            "entropy_margin_trust_ranking_spearman": spearman_correlation(
                trust_entropy_rank[bit_index], trust_margin_rank[bit_index]
            ),
            "replicates": bootstrap_replicates,
        }
        if bit in (2, 4) and observed_rho < 0.7 and rho_ci[1] < 0.9:
            gate_passed = True
            gate_reasons.append(
                f"{bit}-bit capability/trust rho={observed_rho:.3f}, bootstrap upper={rho_ci[1]:.3f}"
            )

        capability_ranks = _ranks_desc(capability_kl[bit_index])
        trust_ranks = _ranks_desc(trust_entropy_rank[bit_index])
        capability_frequency = capability_top / bootstrap_replicates
        trust_frequency = trust_top / bootstrap_replicates
        observed_cap_top = set(np.argsort(capability_kl[bit_index])[-top_count:].tolist())
        observed_trust_top = set(np.argsort(trust_entropy_rank[bit_index])[-top_count:].tolist())
        for layer_index, layer_name in enumerate(artifact.layer_names):
            if layer_index in observed_cap_top and layer_index in observed_trust_top:
                group = "both"
            elif layer_index in observed_cap_top:
                group = "capability_only"
            elif layer_index in observed_trust_top:
                group = "trust_only"
            else:
                group = "neither"
            stable_trust_only = bool(
                trust_frequency[layer_index] >= 0.8 and capability_frequency[layer_index] <= 0.2
            )
            row = {
                "bits": bit,
                "layer": layer_name,
                "layer_index": layer_index,
                "region": str(region_names[layer_index]),
                "capability_kl": float(capability_kl[bit_index, layer_index]),
                "capability_kl_ci_low": float(np.quantile(capability_samples[:, layer_index], 0.025)),
                "capability_kl_ci_high": float(np.quantile(capability_samples[:, layer_index], 0.975)),
                "capability_flip": float(capability_flip[bit_index, layer_index]),
                "capability_accuracy_drop": float(capability_accuracy_drop[bit_index, layer_index]),
                "trust_entropy_rank_disruption": float(trust_entropy_rank[bit_index, layer_index]),
                "trust_entropy_ci_low": float(np.quantile(trust_samples[:, layer_index], 0.025)),
                "trust_entropy_ci_high": float(np.quantile(trust_samples[:, layer_index], 0.975)),
                "trust_margin_rank_disruption": float(trust_margin_rank[bit_index, layer_index]),
                "trust_auroc_drop": float(trust_auroc_drop[bit_index, layer_index]),
                "trust_aurc_increase": float(trust_aurc_increase[bit_index, layer_index]),
                "trust_ece_increase": float(trust_ece_increase[bit_index, layer_index]),
                "trust_brier_increase": float(trust_brier_increase[bit_index, layer_index]),
                "trust_pairwise_ordering_flip": float(trust_pairwise_flip[bit_index, layer_index]),
                "trust_correct_rank_disruption": float(trust_correct_rank[bit_index, layer_index]),
                "trust_incorrect_rank_disruption": float(trust_incorrect_rank[bit_index, layer_index]),
                "teacher_agreement_kl_change": float(teacher_agreement_change[bit_index, layer_index]),
                "capability_rank": int(capability_ranks[layer_index]),
                "trust_rank": int(trust_ranks[layer_index]),
                "rank_displacement": int(abs(capability_ranks[layer_index] - trust_ranks[layer_index])),
                "capability_top_quartile_frequency": float(capability_frequency[layer_index]),
                "trust_top_quartile_frequency": float(trust_frequency[layer_index]),
                "taxonomy": group,
                "stable_trust_only": stable_trust_only,
            }
            layer_rows.append(row)
            taxonomy_rows.append(
                {
                    "bits": bit,
                    "layer": layer_name.replace("model.layers.", "L"),
                    "layer_index": layer_index,
                    "taxonomy": group,
                    "stable_trust_only": stable_trust_only,
                    "rank_displacement": row["rank_displacement"],
                    "capability_top_frequency": row["capability_top_quartile_frequency"],
                    "trust_top_frequency": row["trust_top_quartile_frequency"],
                }
            )

        normalized_capability = _minmax(capability_kl[bit_index])
        normalized_trust = _minmax(trust_entropy_rank[bit_index])
        for region in ("early", "middle", "late"):
            mask = region_names == region
            region_rows.append(
                {
                    "bits": bit,
                    "region": region,
                    "capability_normalized_mean": float(normalized_capability[mask].mean()),
                    "trust_normalized_mean": float(normalized_trust[mask].mean()),
                    "layers": int(mask.sum()),
                }
            )

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": artifact.metadata,
        "probe_bits": list(bits),
        "layer_names": list(artifact.layer_names),
        "parameter_counts": artifact.parameter_counts.tolist(),
        "example_count": examples,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "fp_metrics": fp_metrics,
        "gate": {
            "passed": gate_passed,
            "rule": "At 2 or 4 bits: observed rho < 0.7 and bootstrap 95% CI upper bound < 0.9.",
            "reasons": gate_reasons,
        },
        "ranking_diagnostics": bootstrap,
        "layer_rows": layer_rows,
        "taxonomy_rows": taxonomy_rows,
        "region_rows": region_rows,
        "score_matrices": {
            "capability_kl": capability_kl.T.tolist(),
            "capability_flip": capability_flip.T.tolist(),
            "capability_accuracy_drop": capability_accuracy_drop.T.tolist(),
            "trust_entropy": trust_entropy_rank.T.tolist(),
            "trust_margin": trust_margin_rank.T.tolist(),
            "trust_auroc_drop": trust_auroc_drop.T.tolist(),
            "trust_aurc_increase": trust_aurc_increase.T.tolist(),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    return output


def build_trust_policies(
    artifact: MultiBitCalibrationArtifact,
    analysis: dict,
    *,
    output_path: str,
    budgets: Sequence[float] = (4.0, 3.5, 3.0),
) -> dict:
    """Build capability, trust, joint, uniform, and matched-random policies."""

    if not analysis["gate"]["passed"]:
        raise ValueError("trust-policy gate did not pass; whole-model policy validation is not warranted")
    bits = tuple(artifact.probe_bits)
    budgets = tuple(float(value) for value in budgets)
    capability = np.maximum(np.asarray(analysis["score_matrices"]["capability_kl"]), 0.0)
    flip = np.maximum(np.asarray(analysis["score_matrices"]["capability_flip"]), 0.0)
    accuracy_drop = np.maximum(
        np.asarray(analysis["score_matrices"]["capability_accuracy_drop"]), 0.0
    )
    trust = np.maximum(np.asarray(analysis["score_matrices"]["trust_entropy"]), 0.0)
    matrices = {
        "capability_kl": capability,
        "capability_flip": flip,
        "capability_accuracy_drop": accuracy_drop,
        "trust_entropy": trust,
    }
    normalized_capability = _minmax(capability)
    normalized_trust = _minmax(trust)
    for value in (0.25, 0.50, 0.75):
        matrices[f"joint_{value:.2f}"] = (
            (1.0 - value) * normalized_capability + value * normalized_trust
        )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe_bits": list(bits),
        "budgets": list(budgets),
        "layers": list(artifact.layer_names),
        "parameter_counts": artifact.parameter_counts.tolist(),
        "gate": analysis["gate"],
        "policies": {},
        "uniform": {},
        "random": {},
    }
    for mode, matrix in matrices.items():
        allocations = {}
        safe = np.maximum(matrix, np.finfo(np.float64).tiny)
        for budget in budgets:
            allocation = allocate_bits(
                np.ones(len(artifact.layer_names)),
                artifact.parameter_counts,
                budget,
                bit_choices=bits,
                errors=safe,
            )
            allocations[str(budget)] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
                "objective": allocation.objective,
            }
        report["policies"][mode] = {
            "score_matrix": matrix.tolist(),
            "allocations": allocations,
        }
    for budget in budgets:
        if budget.is_integer() and int(budget) in bits:
            widths = np.full(len(artifact.layer_names), int(budget), dtype=np.int64)
            report["uniform"][str(budget)] = {
                "bits": widths.tolist(),
                "average_bits": average_bits(widths, artifact.parameter_counts),
            }
        runs = {}
        for seed in range(5):
            allocation = random_allocation(
                artifact.parameter_counts, budget, bit_choices=bits, seed=seed
            )
            runs[str(seed)] = {
                "bits": allocation.bits.tolist(),
                "average_bits": allocation.average_bits,
                "total_bits": allocation.total_bits,
                "budget_bits": allocation.budget_bits,
            }
        report["random"][str(budget)] = runs
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _policy_metrics(
    probabilities: np.ndarray, full_precision: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    result = classification_metrics(probabilities, labels)
    result.update(uncertainty_preservation(probabilities, full_precision))
    result["prediction_agreement_with_fp"] = float(
        np.mean(probabilities.argmax(axis=1) == full_precision.argmax(axis=1))
    )
    result["mean_kl_from_fp"] = float(np.mean(kl_divergence(full_precision, probabilities)))
    result["uncertainty_pairwise_ordering_flip"] = pairwise_ordering_flip_rate(
        entropy(full_precision, normalized=True), entropy(probabilities, normalized=True)
    )
    return result


def _mcnemar(left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> dict:
    left_correct = left.argmax(axis=1) == labels
    right_correct = right.argmax(axis=1) == labels
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    total = left_only + right_only
    if total == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(total, value) for value in range(min(left_only, right_only) + 1)) / (
            2**total
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": p_value,
    }


def _bootstrap_policy_difference(
    left: np.ndarray,
    right: np.ndarray,
    full: np.ndarray,
    labels: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    fields = (
        "accuracy",
        "error_auroc",
        "error_auprc",
        "aurc",
        "ece",
        "brier",
        "student_uncertainty_spearman",
    )
    def bootstrap_metrics(
        probabilities: np.ndarray, reference: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        # Pairwise ordering is O(n^2) and is not part of this confidence interval.
        # Avoid computing it for every bootstrap draw.
        values = classification_metrics(probabilities, targets)
        values.update(uncertainty_preservation(probabilities, reference))
        return values

    observed_left = bootstrap_metrics(left, full, labels)
    observed_right = bootstrap_metrics(right, full, labels)
    samples = {field: np.empty(replicates) for field in fields}
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = rng.integers(0, labels.size, size=labels.size)
        left_metrics = bootstrap_metrics(left[indices], full[indices], labels[indices])
        right_metrics = bootstrap_metrics(right[indices], full[indices], labels[indices])
        for field in fields:
            samples[field][replicate] = left_metrics[field] - right_metrics[field]
    return {
        "left_minus_right": {
            field: {
                "observed": observed_left[field] - observed_right[field],
                "bootstrap_95_ci": _mean_ci(values),
            }
            for field, values in samples.items()
        },
        "replicates": replicates,
        "mcnemar": _mcnemar(left, right, labels),
    }


def benchmark_trust_policies(
    *,
    data_path: str,
    student_model: str,
    artifact: MultiBitCalibrationArtifact,
    policies: dict,
    output_dir: str,
    batch_size: int = 128,
    bootstrap_replicates: int = 2000,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Evaluate trust-aware and capability-aware policies on fixed held-out data."""

    examples = load_jsonl(data_path)
    labels = np.asarray([item.answer if item.answer is not None else -1 for item in examples])
    if np.any(labels < 0):
        raise ValueError("benchmark data requires labels")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scorer = HFCausalLMScorer(
        student_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
    )
    blocks = transformer_blocks(scorer.model)
    if tuple(name for name, _ in blocks) != artifact.layer_names:
        raise ValueError("evaluation model blocks do not match the probing artifact")
    full_precision = scorer.score(examples)
    group_size = int(artifact.metadata.get("group_size", 128))
    cache: dict[tuple[int, ...], tuple[str, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {"full_precision": full_precision.astype(np.float32)}
    records: list[dict] = []

    def probability_for(widths: Sequence[int]) -> tuple[str, np.ndarray]:
        key = tuple(int(value) for value in widths)
        if key not in cache:
            name = f"allocation_{len(cache):03d}"
            started = time.perf_counter()
            with temporarily_mixed_quantized(blocks, key, group_size=group_size):
                values = scorer.score(examples)
            cache[key] = (name, values)
            arrays[name] = values.astype(np.float32)
            print(
                f"evaluated {name} ({average_bits(key, artifact.parameter_counts):.3f} bits) "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        return cache[key]

    def add(name: str, family: str, budget: float, widths: Sequence[int], seed: int | None = None):
        probability_key, probabilities = probability_for(widths)
        records.append(
            {
                "name": name,
                "family": family,
                "target_average_bits": budget,
                "actual_average_bits": average_bits(widths, artifact.parameter_counts),
                "random_seed": seed,
                "probability_key": probability_key,
                "bits": list(widths),
                **_policy_metrics(probabilities, full_precision, labels),
            }
        )

    budgets = tuple(float(value) for value in policies["budgets"])
    records.append(
        {
            "name": "full_precision",
            "family": "full_precision",
            "target_average_bits": 16.0,
            "actual_average_bits": 16.0,
            "random_seed": None,
            "probability_key": "full_precision",
            "bits": [],
            **_policy_metrics(full_precision, full_precision, labels),
        }
    )
    for budget in budgets:
        uniform = policies["uniform"].get(str(budget))
        if uniform is not None:
            add(f"uniform_{budget}", "uniform", budget, uniform["bits"])
        for seed, row in policies["random"][str(budget)].items():
            add(f"random_{budget}_seed{seed}", "random", budget, row["bits"], int(seed))
        for mode in TRUST_POLICY_MODES:
            row = policies["policies"][mode]["allocations"][str(budget)]
            add(f"{mode}_{budget}", mode, budget, row["bits"])
    del scorer
    _clear_cuda()

    np.savez_compressed(
        destination / "probabilities.npz",
        labels=labels,
        example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
        **arrays,
    )
    metric_fields = (
        "accuracy",
        "prediction_agreement_with_fp",
        "mean_kl_from_fp",
        "ece",
        "brier",
        "error_auroc",
        "error_auprc",
        "aurc",
        "student_uncertainty_spearman",
        "mean_absolute_uncertainty_distortion",
        "uncertainty_pairwise_ordering_flip",
    )
    summaries = {}
    for family in ("uniform", "random", *TRUST_POLICY_MODES):
        for budget in budgets:
            rows = [
                row
                for row in records
                if row["family"] == family and row["target_average_bits"] == budget
            ]
            if not rows:
                continue
            summaries[f"{family}:{budget}"] = {
                field: {
                    "mean": float(np.nanmean([row[field] for row in rows])),
                    "std": float(np.nanstd([row[field] for row in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0,
                }
                for field in metric_fields
            }
    by_name = {row["name"]: row for row in records}
    comparisons = {}
    for index, budget in enumerate(budgets):
        capability_row = by_name[f"capability_kl_{budget}"]
        capability_probabilities = arrays[capability_row["probability_key"]]
        for mode in ("trust_entropy", "joint_0.50"):
            row = by_name[f"{mode}_{budget}"]
            values = arrays[row["probability_key"]]
            comparisons[f"{mode}_vs_capability_kl:{budget}"] = _bootstrap_policy_difference(
                values,
                capability_probabilities,
                full_precision,
                labels,
                replicates=bootstrap_replicates,
                seed=20260817 + index * 10 + (1 if mode == "trust_entropy" else 2),
            )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "student_model": student_model,
        "data_path": str(Path(data_path).resolve()),
        "example_count": len(examples),
        "bootstrap_replicates": bootstrap_replicates,
        "unique_allocations_evaluated": len(cache),
        "full_precision_metrics": _policy_metrics(full_precision, full_precision, labels),
        "records": records,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (destination / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    return result
