"""Capability-constrained cascade quantization (C3Q, Iteration 5)."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .allocation import average_bits, random_allocation
from .artifacts import MultiBitCalibrationArtifact
from .cascade import CascadeProbabilityArtifact, uncertainty_scores
from .cascade_compression import aurc_from_scores, routing_outcomes, threshold_for_rate
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import binary_auroc, binary_average_precision, spearman_correlation
from .quantization import persistent_mixed_quantizer, temporarily_mixed_quantized, transformer_blocks
from .scoring import kl_divergence


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _configuration_id(bits: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in bits).encode("ascii")
    return hashlib.sha1(payload).hexdigest()[:12]


def _weighted_total(bits: Sequence[int], counts: np.ndarray) -> int:
    return int(np.dot(np.asarray(bits, dtype=np.int64), np.asarray(counts, dtype=np.int64)))


def generate_exact_neighborhood(
    base_bits: Sequence[int],
    parameter_counts: Sequence[int],
    *,
    bit_choices: Sequence[int] = (2, 4, 8),
    local_count: int = 84,
    random_count: int = 15,
    seeds: Sequence[int] = (11, 23, 37),
) -> list[dict]:
    """Generate a deterministic exact-budget pool around ``base_bits``.

    The returned pool includes the KL baseline, local 2--4 layer mutations split
    across three generation seeds, and whole-space random controls.
    """

    base = tuple(int(value) for value in base_bits)
    counts = np.asarray(parameter_counts, dtype=np.int64)
    choices = tuple(sorted(set(int(value) for value in bit_choices)))
    if counts.shape != (len(base),):
        raise ValueError("parameter_counts must match base_bits")
    if local_count < 0 or random_count < 0 or not seeds:
        raise ValueError("candidate counts must be non-negative and seeds non-empty")
    target = _weighted_total(base, counts)
    seen = {base}
    rows = [
        {
            "candidate_id": _configuration_id(base),
            "family": "kl",
            "generation_seed": None,
            "changed_layers": 0,
            "bits": list(base),
        }
    ]

    # Enumerating the bounded mutation space makes the generator reliable for the
    # all-Q4 baseline, whose smallest exact move changes three equal-size blocks.
    mutation_pool: list[tuple[int, ...]] = []
    for radius in range(2, 5):
        for indices in combinations(range(len(base)), radius):
            alternatives = [tuple(value for value in choices if value != base[index]) for index in indices]
            for replacements in product(*alternatives):
                candidate = list(base)
                for index, value in zip(indices, replacements):
                    candidate[index] = value
                key = tuple(candidate)
                if _weighted_total(key, counts) == target:
                    mutation_pool.append(key)
    mutation_pool = sorted(set(mutation_pool))
    if len(mutation_pool) < local_count:
        raise ValueError(f"only {len(mutation_pool)} exact local candidates exist; requested {local_count}")

    # Round-robin seed ownership makes per-seed robustness summaries comparable.
    shuffled_by_seed = []
    for seed in seeds:
        indices = np.arange(len(mutation_pool))
        np.random.default_rng(int(seed)).shuffle(indices)
        shuffled_by_seed.append(iter(indices.tolist()))
    seed_cursor = 0
    while sum(row["family"] == "local" for row in rows) < local_count:
        seed = int(seeds[seed_cursor % len(seeds)])
        iterator = shuffled_by_seed[seed_cursor % len(seeds)]
        seed_cursor += 1
        while True:
            try:
                candidate = mutation_pool[next(iterator)]
            except StopIteration as exc:  # pragma: no cover - impossible for intended model
                raise RuntimeError("exhausted a local candidate stream") from exc
            if candidate not in seen:
                break
        seen.add(candidate)
        rows.append(
            {
                "candidate_id": _configuration_id(candidate),
                "family": "local",
                "generation_seed": seed,
                "changed_layers": int(sum(left != right for left, right in zip(base, candidate))),
                "bits": list(candidate),
            }
        )

    random_seed = 1000
    while sum(row["family"] == "random" for row in rows) < random_count:
        allocation = random_allocation(counts, target / counts.sum(), bit_choices=choices, seed=random_seed)
        candidate = tuple(int(value) for value in allocation.bits)
        if candidate not in seen and _weighted_total(candidate, counts) == target:
            seen.add(candidate)
            rows.append(
                {
                    "candidate_id": _configuration_id(candidate),
                    "family": "random",
                    "generation_seed": random_seed,
                    "changed_layers": int(sum(left != right for left, right in zip(base, candidate))),
                    "bits": list(candidate),
                }
            )
        random_seed += 1
        if random_seed > 100000:  # pragma: no cover - defensive
            raise RuntimeError("could not generate enough unique random exact-budget allocations")
    return rows


def evaluate_configuration(
    probabilities: np.ndarray,
    *,
    fp_probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
    labels: np.ndarray,
    router_signal: str,
    target_rates: Sequence[float],
    cost_coefficients: Sequence[float],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compute capability, routing, and cascade metrics on one split."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    fp_probabilities = np.asarray(fp_probabilities, dtype=np.float64)
    fallback_probabilities = np.asarray(fallback_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scores = uncertainty_scores(probabilities)[router_signal]
    fp_scores = uncertainty_scores(fp_probabilities)[router_signal]
    predictions = probabilities.argmax(axis=1)
    fp_predictions = fp_probabilities.argmax(axis=1)
    errors = predictions != labels
    fallback_correct = fallback_probabilities.argmax(axis=1) == labels
    beneficial = errors & fallback_correct
    thresholds = {str(rate): threshold_for_rate(scores, float(rate)) for rate in target_rates}
    routing = {}
    utility_vectors = []
    arrays: dict[str, np.ndarray] = {}
    flip_rates = []
    for rate in target_rates:
        key = str(float(rate))
        metrics, final_correct, decisions = routing_outcomes(
            probabilities, fallback_probabilities, labels, scores, thresholds[key]
        )
        fp_threshold = threshold_for_rate(fp_scores, float(rate))
        fp_decisions = fp_scores > fp_threshold
        metrics["matched_rate_delegation_flip_rate_from_fp"] = float(np.mean(decisions != fp_decisions))
        routing[key] = metrics
        arrays[f"correct_{key}"] = final_correct
        arrays[f"delegated_{key}"] = decisions
        flip_rates.append(metrics["matched_rate_delegation_flip_rate_from_fp"])
        for cost in cost_coefficients:
            utility_vectors.append(final_correct.astype(np.float64) - float(cost) * decisions.astype(np.float64))
    utility_matrix = np.stack(utility_vectors)
    result = {
        "standalone_accuracy": float(np.mean(~errors)),
        "prediction_agreement_with_fp": float(np.mean(predictions == fp_predictions)),
        "output_kl_from_fp": float(kl_divergence(fp_probabilities, probabilities).mean()),
        "error_auroc": binary_auroc(scores, errors),
        "error_auprc": binary_average_precision(scores, errors),
        "uplift_auroc": binary_auroc(scores, beneficial),
        "aurc": aurc_from_scores(scores, errors),
        "uncertainty_spearman_with_fp": spearman_correlation(fp_scores, scores),
        "mean_matched_rate_delegation_flip_rate_from_fp": float(np.mean(flip_rates)),
        "mean_utility": float(utility_matrix.mean()),
        "thresholds": thresholds,
        "routing": routing,
    }
    arrays["utility"] = utility_matrix
    return result, arrays


def _correlation(rows: list[dict], field: str) -> float | None:
    values = np.asarray([row["metrics"][field] for row in rows], dtype=np.float64)
    utilities = np.asarray([row["metrics"]["mean_utility"] for row in rows], dtype=np.float64)
    value = spearman_correlation(values, utilities)
    return float(value) if np.isfinite(value) else None


def search_c3q(
    *,
    calibration_data_path: str,
    cascade_artifact_path: str,
    probe_artifact: MultiBitCalibrationArtifact,
    policies: dict,
    slm_model: str,
    output_dir: str,
    budgets: Sequence[float] = (3.5, 4.0),
    candidates_per_budget: int = 100,
    random_candidates: int = 15,
    generation_seeds: Sequence[int] = (11, 23, 37),
    router_signal: str = "margin",
    target_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    cost_coefficients: Sequence[float] = (0.0, 0.01, 0.025, 0.05, 0.1),
    kl_tolerance: float = 0.10,
    accuracy_tolerance: float = 0.01,
    headroom_threshold: float = 0.005,
    screen_with_layer_probes: bool = False,
    exact_finalists: int = 6,
    batch_size: int = 128,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Score calibration candidates and freeze C3Q selections if headroom exists."""

    if candidates_per_budget < 2 or random_candidates >= candidates_per_budget:
        raise ValueError("candidate pool must include a KL baseline and local candidates")
    cascade = CascadeProbabilityArtifact.load(cascade_artifact_path)
    examples = load_jsonl(calibration_data_path)
    ids = tuple(item.id for item in examples)
    if ids != cascade.calibration_ids or ids != probe_artifact.example_ids:
        raise ValueError("calibration IDs do not match supplied artifacts")
    destination = Path(output_dir)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        raise ValueError("model blocks do not match the probe artifact")
    probability_cache: dict[tuple[int, ...], np.ndarray] = {}
    q4 = tuple([4] * len(blocks))
    if 4 in cascade.bits:
        probability_cache[q4] = cascade.quantized_calibration[cascade.bits.index(4)]
    quantizer_manager = persistent_mixed_quantizer(
        blocks, group_size=int(probe_artifact.metadata["group_size"])
    )
    apply_bits = quantizer_manager.__enter__()

    def probability_for(bits: Sequence[int]) -> np.ndarray:
        key = tuple(int(value) for value in bits)
        if key in probability_cache:
            return probability_cache[key]
        candidate_id = _configuration_id(key)
        path = checkpoint_dir / f"{candidate_id}.npy"
        if path.exists():
            values = np.load(path, allow_pickle=False)
            if values.shape != cascade.slm_fp_calibration.shape:
                raise ValueError(f"invalid checkpoint shape at {path}")
        else:
            started = time.perf_counter()
            apply_bits(key)
            values = scorer.score(examples).astype(np.float32)
            np.save(path, values, allow_pickle=False)
            print(f"calibration candidate {candidate_id} completed in {time.perf_counter() - started:.1f}s", flush=True)
        probability_cache[key] = values
        return values

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "c3q_calibration_search",
        "models": {"slm": slm_model, "fallback": cascade.metadata["fallback_model"]},
        "scope": {
            "calibration_examples": len(examples),
            "budgets": [float(value) for value in budgets],
            "candidate_precisions": list(probe_artifact.probe_bits),
            "candidates_per_budget": int(candidates_per_budget),
            "random_candidates_per_budget": int(random_candidates),
            "generation_seeds": [int(value) for value in generation_seeds],
            "router_signal": router_signal,
            "target_rates": [float(value) for value in target_rates],
            "cost_coefficients": [float(value) for value in cost_coefficients],
            "threshold_policy": "candidate-specific matched-rate calibration thresholds frozen for held-out evaluation",
            "quantization": "symmetric group-size-128 fake weight quantization",
            "screening": (
                "additive log-probability layer-probe surrogate with frozen exact finalists"
                if screen_with_layer_probes
                else "none; every candidate receives exact whole-model scoring"
            ),
            "exact_finalists_per_budget": int(exact_finalists) if screen_with_layer_probes else int(candidates_per_budget),
        },
        "guardrail": {
            "kl_multiplier": 1.0 + float(kl_tolerance),
            "accuracy_tolerance": float(accuracy_tolerance),
            "headroom_minimum_range": float(headroom_threshold),
        },
        "budgets": {},
    }
    fp_metrics, _ = evaluate_configuration(
        cascade.slm_fp_calibration,
        fp_probabilities=cascade.slm_fp_calibration,
        fallback_probabilities=cascade.fallback_calibration,
        labels=cascade.calibration_labels,
        router_signal=router_signal,
        target_rates=target_rates,
        cost_coefficients=cost_coefficients,
    )
    result["fp_reference"] = fp_metrics

    authorized_budgets: list[str] = []
    for budget in budgets:
        budget_key = str(float(budget))
        policy_rows = policies["policies"][budget_key]
        base_bits = policy_rows["kl"]["bits"]
        candidates = generate_exact_neighborhood(
            base_bits,
            probe_artifact.parameter_counts,
            bit_choices=probe_artifact.probe_bits,
            local_count=candidates_per_budget - random_candidates - 1,
            random_count=random_candidates,
            seeds=generation_seeds,
        )
        if screen_with_layer_probes:
            fp_log = np.log(np.clip(cascade.slm_fp_calibration.astype(np.float64), 1e-12, None))
            bit_to_index = {int(bit): index for index, bit in enumerate(probe_artifact.probe_bits)}
            for candidate in candidates:
                combined = fp_log.copy()
                for layer_index, bit in enumerate(candidate["bits"]):
                    probe = probe_artifact.quantized_probabilities[bit_to_index[int(bit)], layer_index]
                    combined += np.log(np.clip(probe, 1e-12, None)) - fp_log
                combined -= combined.max(axis=1, keepdims=True)
                surrogate = np.exp(combined)
                surrogate /= surrogate.sum(axis=1, keepdims=True)
                candidate["screening_metrics"], _ = evaluate_configuration(
                    surrogate,
                    fp_probabilities=cascade.slm_fp_calibration,
                    fallback_probabilities=cascade.fallback_calibration,
                    labels=cascade.calibration_labels,
                    router_signal=router_signal,
                    target_rates=target_rates,
                    cost_coefficients=cost_coefficients,
                )
            screen_kl = next(row for row in candidates if row["family"] == "kl")
            screen_kl_limit = screen_kl["screening_metrics"]["output_kl_from_fp"] * (1.0 + kl_tolerance)
            screen_accuracy_floor = screen_kl["screening_metrics"]["standalone_accuracy"] - accuracy_tolerance
            for candidate in candidates:
                candidate["screening_safe_joint"] = (
                    candidate["screening_metrics"]["output_kl_from_fp"] <= screen_kl_limit + 1e-12
                    and candidate["screening_metrics"]["standalone_accuracy"] >= screen_accuracy_floor - 1e-12
                )
            finalists = [screen_kl]
            for seed in generation_seeds:
                seed_rows = [
                    row for row in candidates
                    if row["family"] == "local"
                    and row["generation_seed"] == int(seed)
                    and row["screening_safe_joint"]
                ]
                if seed_rows:
                    finalists.append(max(seed_rows, key=lambda row: row["screening_metrics"]["mean_utility"]))
            safe_random_screen = [
                row for row in candidates if row["family"] == "random" and row["screening_safe_joint"]
            ]
            if safe_random_screen:
                finalists.append(max(safe_random_screen, key=lambda row: row["screening_metrics"]["mean_utility"]))
            ranked_screen = sorted(
                (row for row in candidates if row["screening_safe_joint"]),
                key=lambda row: row["screening_metrics"]["mean_utility"],
                reverse=True,
            )
            finalist_ids = {row["candidate_id"] for row in finalists}
            for row in ranked_screen:
                if len(finalist_ids) >= exact_finalists:
                    break
                finalist_ids.add(row["candidate_id"])
            # If the surrogate rejects nearly everything, fill by utility rank so
            # the exact guardrail still tests alternatives to the KL point.
            for row in sorted(candidates, key=lambda item: item["screening_metrics"]["mean_utility"], reverse=True):
                if len(finalist_ids) >= exact_finalists:
                    break
                finalist_ids.add(row["candidate_id"])
            candidates_to_score = [row for row in candidates if row["candidate_id"] in finalist_ids]
        else:
            candidates_to_score = candidates
        # Greedy Hamming ordering minimizes block rewrites while preserving the
        # pre-generated candidate set and seed ownership.
        remaining = candidates_to_score.copy()
        ordered_candidates = []
        current_bits = tuple(base_bits)
        while remaining:
            best_index = min(
                range(len(remaining)),
                key=lambda index: sum(
                    left != right for left, right in zip(current_bits, remaining[index]["bits"])
                ),
            )
            candidate = remaining.pop(best_index)
            ordered_candidates.append(candidate)
            current_bits = tuple(candidate["bits"])
        for index, candidate in enumerate(ordered_candidates, start=1):
            values = probability_for(candidate["bits"])
            metrics, _ = evaluate_configuration(
                values,
                fp_probabilities=cascade.slm_fp_calibration,
                fallback_probabilities=cascade.fallback_calibration,
                labels=cascade.calibration_labels,
                router_signal=router_signal,
                target_rates=target_rates,
                cost_coefficients=cost_coefficients,
            )
            candidate["metrics"] = metrics
            candidate["average_bits"] = average_bits(candidate["bits"], probe_artifact.parameter_counts)
            if index % 10 == 0 or index == len(candidates):
                print(f"budget {budget_key}: analyzed {index}/{len(candidates_to_score)} exact candidates", flush=True)

        kl_row = next(row for row in candidates_to_score if row["family"] == "kl")
        kl_limit = kl_row["metrics"]["output_kl_from_fp"] * (1.0 + kl_tolerance)
        accuracy_floor = kl_row["metrics"]["standalone_accuracy"] - accuracy_tolerance
        for candidate in candidates:
            if "metrics" not in candidate:
                candidate["safe_kl_only"] = None
                candidate["safe_accuracy_only"] = None
                candidate["safe_joint"] = None
                continue
            candidate["safe_kl_only"] = candidate["metrics"]["output_kl_from_fp"] <= kl_limit + 1e-12
            candidate["safe_accuracy_only"] = candidate["metrics"]["standalone_accuracy"] >= accuracy_floor - 1e-12
            candidate["safe_joint"] = candidate["safe_kl_only"] and candidate["safe_accuracy_only"]
        safe = [row for row in candidates_to_score if row["safe_joint"]]
        alternatives = [row for row in safe if row["family"] != "kl"]
        utilities = np.asarray([row["metrics"]["mean_utility"] for row in safe], dtype=np.float64)
        utility_range = float(np.ptp(utilities)) if utilities.size else 0.0
        utility_variance = float(np.var(utilities)) if utilities.size else 0.0
        headroom = len(alternatives) >= 2 and utility_range > headroom_threshold
        if headroom:
            authorized_budgets.append(budget_key)
        eligible_c3q = [row for row in safe if row["family"] in {"kl", "local"}]
        c3q = max(eligible_c3q, key=lambda row: row["metrics"]["mean_utility"])
        safe_randoms = [row for row in safe if row["family"] == "random"]
        best_random = max(safe_randoms, key=lambda row: row["metrics"]["mean_utility"]) if safe_randoms else None

        seed_winners = []
        for seed in generation_seeds:
            seed_rows = [row for row in safe if row["family"] == "local" and row["generation_seed"] == int(seed)]
            if not seed_rows:
                continue
            winner = max(seed_rows, key=lambda row: row["metrics"]["mean_utility"])
            seed_winners.append(
                {
                    "seed": int(seed),
                    "candidate_id": winner["candidate_id"],
                    "mean_utility": winner["metrics"]["mean_utility"],
                    "modified_layers": [index for index, (left, right) in enumerate(zip(base_bits, winner["bits"])) if left != right],
                }
            )
        overlaps = []
        for left, right in combinations(seed_winners, 2):
            a, b = set(left["modified_layers"]), set(right["modified_layers"])
            overlaps.append({"left_seed": left["seed"], "right_seed": right["seed"], "jaccard": len(a & b) / max(1, len(a | b))})

        required_baselines = {}
        baseline_bits = {
            "uniform_q4": list(q4),
            "accuracy": policy_rows["accuracy"]["bits"],
            "cuaq": policy_rows["cuaq"]["bits"],
        }
        for name, bits in baseline_bits.items():
            if headroom:
                values = probability_for(bits)
                metrics, _ = evaluate_configuration(
                    values,
                    fp_probabilities=cascade.slm_fp_calibration,
                    fallback_probabilities=cascade.fallback_calibration,
                    labels=cascade.calibration_labels,
                    router_signal=router_signal,
                    target_rates=target_rates,
                    cost_coefficients=cost_coefficients,
                )
                required_baselines[name] = {"bits": list(bits), "candidate_id": _configuration_id(bits), "metrics": metrics, "status": "exact"}
            else:
                required_baselines[name] = {"bits": list(bits), "candidate_id": _configuration_id(bits), "metrics": None, "status": "not_run_after_headroom_stop"}

        result["budgets"][budget_key] = {
            "candidate_count": len(candidates),
            "exact_candidate_count": len(candidates_to_score),
            "safe_counts": {
                "kl_only": int(sum(bool(row["safe_kl_only"]) for row in candidates_to_score)),
                "accuracy_only": int(sum(bool(row["safe_accuracy_only"]) for row in candidates_to_score)),
                "joint": len(safe),
                "joint_alternatives": len(alternatives),
                "joint_random": len(safe_randoms),
                "screening_joint": int(sum(bool(row.get("screening_safe_joint")) for row in candidates)) if screen_with_layer_probes else None,
            },
            "limits": {"output_kl": kl_limit, "standalone_accuracy": accuracy_floor},
            "headroom": {
                "utility_variance": utility_variance,
                "utility_range": utility_range,
                "meaningful": headroom,
                "accuracy_utility_spearman": _correlation(safe, "standalone_accuracy") if len(safe) > 1 else None,
                "kl_utility_spearman": _correlation(safe, "output_kl_from_fp") if len(safe) > 1 else None,
            },
            "selection": {
                "kl": kl_row["candidate_id"],
                "c3q": c3q["candidate_id"],
                "best_safe_random": best_random["candidate_id"] if best_random else None,
            },
            "seed_robustness": {"winners": seed_winners, "modified_layer_overlap": overlaps},
            "baselines": required_baselines,
            "candidates": candidates,
        }
        print(
            f"budget {budget_key}: {len(safe)}/{len(candidates_to_score)} exact finalists safe, "
            f"utility range={utility_range:.6f}, headroom={headroom}",
            flush=True,
        )

    result["gate"] = {
        "status": "pass" if authorized_budgets else "stop",
        "criterion": "a budget has >=2 safe alternatives and mean-utility range > headroom minimum",
        "authorized_budgets": authorized_budgets,
        "heldout_authorized": bool(authorized_budgets),
    }
    quantizer_manager.__exit__(None, None, None)
    del scorer
    _clear_cuda()
    _write_json(destination / "results.json", result)
    return result


def _paired_interval(left: np.ndarray, right: np.ndarray, *, seed: int, replicates: int) -> dict:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        columns = rng.integers(0, differences.shape[-1], size=differences.shape[-1])
        samples[index] = differences[..., columns].mean()
    return {"difference": float(differences.mean()), "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(), "replicates": replicates}


def _mcnemar(left: np.ndarray, right: np.ndarray) -> dict:
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    left_only, right_only = int(np.sum(left & ~right)), int(np.sum(~left & right))
    total = left_only + right_only
    if total == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(total, value) for value in range(min(left_only, right_only) + 1)) / (2**total)
        p_value = min(1.0, 2.0 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "exact_p": p_value}


def benchmark_c3q(
    *,
    evaluation_data_path: str,
    cascade_artifact_path: str,
    probe_artifact: MultiBitCalibrationArtifact,
    search_results: dict,
    slm_model: str,
    output_dir: str,
    batch_size: int = 128,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260820,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Run the one-shot held-out ARC comparison after the headroom gate passes."""

    if not search_results.get("gate", {}).get("heldout_authorized", False):
        raise ValueError("calibration headroom gate did not authorize held-out evaluation")
    cascade = CascadeProbabilityArtifact.load(cascade_artifact_path)
    examples = load_jsonl(evaluation_data_path)
    ids = tuple(item.id for item in examples)
    if ids != cascade.evaluation_ids:
        raise ValueError("evaluation IDs do not match the cascade artifact")
    destination = Path(output_dir)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    cache: dict[tuple[int, ...], np.ndarray] = {}
    q4 = tuple([4] * len(blocks))
    if 4 in cascade.bits:
        cache[q4] = cascade.quantized_evaluation[cascade.bits.index(4)]

    def probability_for(bits: Sequence[int]) -> np.ndarray:
        key = tuple(int(value) for value in bits)
        if key in cache:
            return cache[key]
        path = checkpoint_dir / f"{_configuration_id(key)}.npy"
        if path.exists():
            values = np.load(path, allow_pickle=False)
        else:
            started = time.perf_counter()
            with temporarily_mixed_quantized(blocks, key, group_size=int(probe_artifact.metadata["group_size"])):
                values = scorer.score(examples).astype(np.float32)
            np.save(path, values, allow_pickle=False)
            print(f"held-out candidate {_configuration_id(key)} completed in {time.perf_counter() - started:.1f}s", flush=True)
        cache[key] = values
        return values

    rates = search_results["scope"]["target_rates"]
    costs = search_results["scope"]["cost_coefficients"]
    configurations = []
    arrays = {"fp": cascade.slm_fp_evaluation.astype(np.float32), "fallback": cascade.fallback_evaluation.astype(np.float32)}
    evaluated = {}
    vector_cache = {}
    authorized_budgets = set(search_results.get("gate", {}).get("authorized_budgets", search_results["budgets"].keys()))
    for budget_key, budget_row in search_results["budgets"].items():
        if budget_key not in authorized_budgets:
            continue
        candidates = {row["candidate_id"]: row for row in budget_row["candidates"]}
        selected = {
            "kl": candidates[budget_row["selection"]["kl"]],
            "c3q": candidates[budget_row["selection"]["c3q"]],
        }
        if budget_row["selection"]["best_safe_random"]:
            selected["best_safe_random"] = candidates[budget_row["selection"]["best_safe_random"]]
        for baseline_name in ("uniform_q4", "accuracy", "cuaq"):
            selected[baseline_name] = budget_row["baselines"][baseline_name]
        for family, candidate in selected.items():
            bits = candidate["bits"]
            cache_key = (budget_key, family)
            probabilities = probability_for(bits)
            scores = uncertainty_scores(probabilities)[search_results["scope"]["router_signal"]]
            fp_scores = uncertainty_scores(cascade.slm_fp_evaluation)[search_results["scope"]["router_signal"]]
            predictions = probabilities.argmax(axis=1)
            errors = predictions != cascade.evaluation_labels
            calibration_metrics = candidate["metrics"]
            utility_vectors, rate_vectors = [], {}
            routing = {}
            for rate in rates:
                rate_key = str(float(rate))
                threshold = calibration_metrics["thresholds"][rate_key]
                metrics, correct, decisions = routing_outcomes(
                    probabilities, cascade.fallback_evaluation, cascade.evaluation_labels, scores, threshold
                )
                fp_threshold = search_results["fp_reference"]["thresholds"][rate_key]
                metrics["frozen_threshold_delegation_flip_rate_from_fp"] = float(np.mean(decisions != (fp_scores > fp_threshold)))
                routing[rate_key] = metrics
                rate_vectors[rate_key] = correct
                for cost in costs:
                    utility_vectors.append(correct.astype(np.float64) - float(cost) * decisions.astype(np.float64))
            utility = np.stack(utility_vectors)
            name = f"{family}_{budget_key}"
            metrics = {
                "standalone_accuracy": float(np.mean(~errors)),
                "prediction_agreement_with_fp": float(np.mean(predictions == cascade.slm_fp_evaluation.argmax(axis=1))),
                "output_kl_from_fp": float(kl_divergence(cascade.slm_fp_evaluation, probabilities).mean()),
                "error_auroc": binary_auroc(scores, errors),
                "error_auprc": binary_average_precision(scores, errors),
                "aurc": aurc_from_scores(scores, errors),
                "uncertainty_spearman_with_fp": spearman_correlation(fp_scores, scores),
                "mean_utility": float(utility.mean()),
                "routing": routing,
            }
            evaluated[name] = metrics
            vector_cache[name] = {"utility": utility, "rate_correct": rate_vectors}
            arrays[name] = probabilities.astype(np.float32)
            configurations.append({"name": name, "family": family, "budget": float(budget_key), "candidate_id": _configuration_id(bits), "bits": bits, "metrics": metrics})

    comparisons = {}
    arc_passes = []
    for budget_key in authorized_budgets:
        c3q_name, kl_name = f"c3q_{budget_key}", f"kl_{budget_key}"
        random_name = f"best_safe_random_{budget_key}"
        comparison = {
            "c3q_vs_kl_mean_utility": _paired_interval(vector_cache[c3q_name]["utility"], vector_cache[kl_name]["utility"], seed=bootstrap_seed, replicates=bootstrap_replicates),
            "c3q_vs_kl_by_rate": {},
        }
        for rate in rates:
            rate_key = str(float(rate))
            left = vector_cache[c3q_name]["rate_correct"][rate_key]
            right = vector_cache[kl_name]["rate_correct"][rate_key]
            comparison["c3q_vs_kl_by_rate"][rate_key] = {
                "cascade_accuracy": _paired_interval(left, right, seed=bootstrap_seed + int(float(rate) * 1000), replicates=bootstrap_replicates),
                "mcnemar": _mcnemar(left, right),
            }
        if random_name in vector_cache:
            comparison["c3q_vs_best_safe_random_mean_utility"] = _paired_interval(vector_cache[c3q_name]["utility"], vector_cache[random_name]["utility"], seed=bootstrap_seed + 1, replicates=bootstrap_replicates)
        utility_gain = comparison["c3q_vs_kl_mean_utility"]["difference"] > 0
        capability_preserved = abs(evaluated[c3q_name]["standalone_accuracy"] - evaluated[kl_name]["standalone_accuracy"]) < 0.01
        beats_random = random_name not in vector_cache or comparison["c3q_vs_best_safe_random_mean_utility"]["difference"] > 0
        significant_region = any(row["cascade_accuracy"]["bootstrap_95_ci"][0] > 0 for row in comparison["c3q_vs_kl_by_rate"].values())
        comparison["arc_positive"] = bool(utility_gain and capability_preserved and beats_random and significant_region)
        comparison["criteria"] = {"utility_gain": utility_gain, "capability_preserved_within_1pp": capability_preserved, "beats_best_safe_random": beats_random, "one_rate_ci_excludes_zero": significant_region}
        arc_passes.append(comparison["arc_positive"])
        comparisons[budget_key] = comparison

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "c3q_heldout_arc",
        "scope": {"evaluation_examples": len(examples), "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed},
        "configurations": configurations,
        "comparisons": comparisons,
        "gate": {
            "status": "pass" if any(arc_passes) else "stop",
            "mmlu_authorized": bool(any(arc_passes)),
            "criterion": "positive mean utility vs KL, <1 pp capability gap, beats safe random, and a per-rate cascade-accuracy CI excluding zero",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "probabilities.npz", **arrays)
    _write_json(destination / "results.json", result)
    del scorer
    _clear_cuda()
    return result
