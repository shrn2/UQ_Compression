"""Uncertainty-Robust Structured Allocation (URSA).

URSA uses predictive uncertainty while constructing one static compressed model.
Calibration examples are partitioned into equal-frequency entropy strata.  FFN
channel importance is measured independently in every stratum and a greedy
minimax allocator retains channels that reduce the soft worst-stratum omitted
importance.  No uncertainty estimator or routing policy is used at inference.
"""

from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import MultipleChoiceExample, load_jsonl
from .hf import HFCausalLMScorer
from .metrics import classification_metrics, spearman_correlation
from .quantization import transformer_blocks
from .scoring import entropy, kl_divergence
from .structured_sparsity import (
    apply_dense_masks,
    ffn_modules,
    linear_compute_ratios,
    nested_masks,
)


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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _write_json(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def uncertainty_strata(probabilities: np.ndarray, strata: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized entropy and stable equal-frequency stratum labels."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if strata < 2 or probabilities.ndim != 2 or probabilities.shape[0] < strata:
        raise ValueError("probabilities must be [examples, classes] with at least one row per stratum")
    uncertainty = entropy(probabilities, normalized=True)
    order = np.argsort(uncertainty, kind="stable")
    labels = np.empty(len(order), dtype=np.int64)
    for label, indices in enumerate(np.array_split(order, strata)):
        labels[indices] = label
    return uncertainty, labels


def greedy_minimax_order(
    stratum_importance: np.ndarray, *, temperature: float = 0.05, refresh_interval: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Produce a full nested channel order using a soft minimax objective.

    ``stratum_importance`` has shape ``[strata, layers, channels]``.  Each
    stratum/layer row is normalized to unit mass.  At each greedy step, the
    softmax over residual omitted mass emphasizes whichever uncertainty stratum
    is currently least protected.
    """

    values = np.asarray(stratum_importance, dtype=np.float64)
    if values.ndim != 3 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("stratum_importance must be finite non-negative [strata, layers, channels]")
    if values.shape[0] < 2 or values.shape[2] < 1 or temperature <= 0 or refresh_interval < 1:
        raise ValueError(
            "at least two strata, one channel, positive temperature, and refresh interval are required"
        )
    totals = values.sum(axis=2, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every stratum/layer importance row must have positive mass")
    shares = values / totals
    _, layers, channels = shares.shape
    orders = np.empty((layers, channels), dtype=np.int64)
    residual_history = np.empty((layers, channels + 1, shares.shape[0]), dtype=np.float64)
    for layer in range(layers):
        residual = np.ones(shares.shape[0], dtype=np.float64)
        available = np.ones(channels, dtype=bool)
        residual_history[layer, 0] = residual
        step = 0
        while step < channels:
            centered = (residual - residual.max()) / temperature
            weights = np.exp(centered)
            weights /= weights.sum()
            marginal = weights @ shares[:, layer, :]
            marginal[~available] = -np.inf
            # Refresh the adversarial stratum weights in bounded chunks.  This
            # retains the minimax feedback while avoiding O(channels^2) Python
            # work for 9k-wide LLM FFNs.
            candidates = np.argsort(-marginal, kind="stable")
            selected_batch = candidates[: min(refresh_interval, channels - step)]
            for selected in selected_batch:
                selected = int(selected)
                orders[layer, step] = selected
                available[selected] = False
                residual = np.maximum(0.0, residual - shares[:, layer, selected])
                residual_history[layer, step + 1] = residual
                step += 1
    return orders, residual_history


def masks_from_order(
    order: np.ndarray, medium_sparsity: float, aggressive_sparsity: float
) -> dict[str, np.ndarray]:
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 2 or not 0 <= medium_sparsity < aggressive_sparsity < 1:
        raise ValueError("order must be [layers, channels] with ordered sparsity levels")
    channels = order.shape[1]
    expected = np.arange(channels)
    if any(not np.array_equal(np.sort(row), expected) for row in order):
        raise ValueError("each order row must be a channel permutation")
    medium_keep = max(1, int(round(channels * (1.0 - medium_sparsity))))
    aggressive_keep = max(1, int(round(channels * (1.0 - aggressive_sparsity))))
    return {
        "medium": np.sort(order[:, :medium_keep], axis=1),
        "aggressive": np.sort(order[:, :aggressive_keep], axis=1),
    }


def _selected_examples(
    data_path: str | Path,
    index_artifact_path: str | Path | None,
    index_key: str,
) -> tuple[list[MultipleChoiceExample], np.ndarray]:
    examples = load_jsonl(data_path)
    if index_artifact_path is None:
        indices = np.arange(len(examples), dtype=np.int64)
    else:
        with np.load(index_artifact_path, allow_pickle=False) as artifact:
            if index_key not in artifact.files:
                raise ValueError(f"{index_key!r} is not present in {index_artifact_path}")
            indices = np.asarray(artifact[index_key], dtype=np.int64)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError("selected indices must be a unique vector")
        if np.any(indices < 0) or np.any(indices >= len(examples)):
            raise ValueError("selected indices are outside the data range")
    return [examples[int(index)] for index in indices], indices


def _require_single_token_choices(scorer: HFCausalLMScorer, examples: Sequence[MultipleChoiceExample]) -> None:
    if not all(
        len(scorer.tokenizer.encode(choice, add_special_tokens=False)) == 1
        for example in examples
        for choice in example.choices
    ):
        raise ValueError("URSA's prompt-final activation collector currently requires single-token choices")


def _activation_importance(
    scorer: HFCausalLMScorer,
    blocks: Sequence[tuple[str, object]],
    examples: Sequence[MultipleChoiceExample],
) -> np.ndarray:
    sums: list[np.ndarray] = []
    counts = np.zeros(len(blocks), dtype=np.int64)
    handles = []
    for layer_index, (_, block) in enumerate(blocks):
        _, _, down = ffn_modules(block)
        sums.append(np.zeros(down.in_features, dtype=np.float64))

        def hook(_module, inputs, index=layer_index):
            current = inputs[0].detach()[:, -1, :].abs().sum(dim=0).float().cpu().numpy()
            sums[index] += current
            counts[index] += int(inputs[0].shape[0])

        handles.append(down.register_forward_pre_hook(hook))
    try:
        scorer.score(examples)
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    for layer_index, (_, block) in enumerate(blocks):
        _, _, down = ffn_modules(block)
        activation = sums[layer_index] / max(1, int(counts[layer_index]))
        weight_norm = down.weight.detach().float().norm(dim=0).cpu().numpy()
        rows.append(activation * weight_norm)
    return np.stack(rows)


def _save_masks(
    path: str | Path,
    masks: dict[str, np.ndarray],
    *,
    medium_sparsity: float,
    aggressive_sparsity: float,
    method: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        medium=masks["medium"],
        aggressive=masks["aggressive"],
        medium_sparsity=np.asarray(medium_sparsity),
        aggressive_sparsity=np.asarray(aggressive_sparsity),
        method=np.asarray(method, dtype=np.str_),
    )


def collect_ursa_masks(
    *,
    data_path: str,
    slm_model: str,
    output_path: str,
    baseline_masks_path: str,
    ursa_masks_path: str,
    index_artifact_path: str | None = None,
    index_key: str = "train_indices",
    strata: int = 3,
    temperature: float = 0.05,
    refresh_interval: int = 64,
    medium_sparsity: float = 0.25,
    aggressive_sparsity: float = 0.50,
    batch_size: int = 64,
    device_map: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    bnb_4bit_compute_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict:
    """Collect matched mean-activation and URSA structured-pruning masks."""

    destination = Path(output_path)
    if destination.exists() and Path(baseline_masks_path).exists() and Path(ursa_masks_path).exists():
        with np.load(destination, allow_pickle=False) as saved:
            return json.loads(str(saved["metadata"]))
    examples, selected_indices = _selected_examples(data_path, index_artifact_path, index_key)
    scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        quantization=quantization,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=4096,
    )
    _require_single_token_choices(scorer, examples)
    blocks = transformer_blocks(scorer.model)
    started = time.perf_counter()
    dense = scorer.score(examples)
    uncertainty, stratum_labels = uncertainty_strata(dense, strata=strata)
    importance_rows = []
    stratum_counts = []
    for label in range(strata):
        subset = [example for example, current in zip(examples, stratum_labels) if current == label]
        stratum_counts.append(len(subset))
        importance_rows.append(_activation_importance(scorer, blocks, subset))
        print(f"URSA stratum {label + 1}/{strata} ({len(subset)} examples) complete", flush=True)
    stratum_importance = np.stack(importance_rows)
    frequencies = np.asarray(stratum_counts, dtype=np.float64) / len(examples)
    baseline_importance = np.sum(stratum_importance * frequencies[:, None, None], axis=0)
    baseline_masks = nested_masks(baseline_importance, medium_sparsity, aggressive_sparsity)
    robust_order, residual_history = greedy_minimax_order(
        stratum_importance, temperature=temperature, refresh_interval=refresh_interval
    )
    ursa_masks = masks_from_order(robust_order, medium_sparsity, aggressive_sparsity)
    overlap = {}
    for level in ("medium", "aggressive"):
        intersections = [
            len(set(left).intersection(right)) / len(left)
            for left, right in zip(baseline_masks[level], ursa_masks[level])
        ]
        overlap[level] = {
            "mean_retained_overlap": float(np.mean(intersections)),
            "min_retained_overlap": float(np.min(intersections)),
        }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "URSA: entropy-stratified greedy soft-minimax FFN-channel allocation",
        "baseline": "matched-data global mean activation times down-projection column norm",
        "model": slm_model,
        "quantization": quantization,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "data_path": str(Path(data_path).resolve()),
        "index_artifact": str(Path(index_artifact_path).resolve()) if index_artifact_path else None,
        "index_key": index_key if index_artifact_path else None,
        "examples": len(examples),
        "strata": strata,
        "stratum_counts": stratum_counts,
        "temperature": temperature,
        "refresh_interval": refresh_interval,
        "medium_sparsity": medium_sparsity,
        "aggressive_sparsity": aggressive_sparsity,
        "nested": True,
        "physical_channel_pruning": False,
        "logical_channel_mask": False,
        "overlap": overlap,
        "collection_seconds": time.perf_counter() - started,
        "claim_boundary": "mask construction only; no accuracy, latency, or packed-size claim",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        dense_probabilities=dense.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        stratum_labels=stratum_labels,
        selected_indices=selected_indices,
        example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
        stratum_importance=stratum_importance,
        baseline_importance=baseline_importance,
        robust_order=robust_order,
        residual_history=residual_history.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    _save_masks(
        baseline_masks_path,
        baseline_masks,
        medium_sparsity=medium_sparsity,
        aggressive_sparsity=aggressive_sparsity,
        method="matched_mean_activation",
    )
    _save_masks(
        ursa_masks_path,
        ursa_masks,
        medium_sparsity=medium_sparsity,
        aggressive_sparsity=aggressive_sparsity,
        method="ursa_soft_minimax",
    )
    del scorer
    _clear_cuda()
    return metadata


def _load_masks(path: str | Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as saved:
        return (
            np.asarray(saved["medium"], dtype=np.int64),
            np.asarray(saved["aggressive"], dtype=np.int64),
            float(saved["medium_sparsity"]),
            float(saved["aggressive_sparsity"]),
        )


def _score_nested(
    scorer: HFCausalLMScorer,
    examples: Sequence[MultipleChoiceExample],
    medium: np.ndarray,
    aggressive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = transformer_blocks(scorer.model)
    apply_dense_masks(blocks, medium)
    medium_probabilities = scorer.score(examples)
    local_aggressive = np.stack(
        [np.searchsorted(medium_row, aggressive_row) for medium_row, aggressive_row in zip(medium, aggressive)]
    )
    apply_dense_masks(blocks, local_aggressive)
    return medium_probabilities, scorer.score(examples)


def score_ursa_comparison(
    *,
    data_path: str,
    slm_model: str,
    baseline_masks_path: str,
    ursa_masks_path: str,
    output_path: str,
    index_artifact_path: str | None = None,
    index_key: str = "validation_indices",
    batch_size: int = 64,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Score matched baseline and URSA nested sparse models sequentially."""

    destination = Path(output_path)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            return json.loads(str(saved["metadata"]))
    examples, selected_indices = _selected_examples(data_path, index_artifact_path, index_key)
    labels = np.asarray([example.answer if example.answer is not None else -1 for example in examples])
    if np.any(labels < 0):
        raise ValueError("evaluation requires labels")
    baseline_medium, baseline_aggressive, medium_sparsity, aggressive_sparsity = _load_masks(
        baseline_masks_path
    )
    ursa_medium, ursa_aggressive, ursa_medium_sparsity, ursa_aggressive_sparsity = _load_masks(
        ursa_masks_path
    )
    if (medium_sparsity, aggressive_sparsity) != (ursa_medium_sparsity, ursa_aggressive_sparsity):
        raise ValueError("baseline and URSA masks must use identical sparsity levels")
    started = time.perf_counter()
    baseline_scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=4096,
    )
    _require_single_token_choices(baseline_scorer, examples)
    dense_blocks = transformer_blocks(baseline_scorer.model)
    original_width = ffn_modules(dense_blocks[0][1])[2].in_features
    compute = {
        "dense": linear_compute_ratios(baseline_scorer.model, dense_blocks, 1.0),
        "medium": linear_compute_ratios(
            baseline_scorer.model, dense_blocks, baseline_medium.shape[1] / original_width
        ),
        "aggressive": linear_compute_ratios(
            baseline_scorer.model, dense_blocks, baseline_aggressive.shape[1] / original_width
        ),
    }
    dense = baseline_scorer.score(examples)
    baseline_medium_probabilities, baseline_aggressive_probabilities = _score_nested(
        baseline_scorer, examples, baseline_medium, baseline_aggressive
    )
    del baseline_scorer
    _clear_cuda()
    ursa_scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=4096,
    )
    ursa_medium_probabilities, ursa_aggressive_probabilities = _score_nested(
        ursa_scorer, examples, ursa_medium, ursa_aggressive
    )
    del ursa_scorer
    _clear_cuda()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": slm_model,
        "data_path": str(Path(data_path).resolve()),
        "index_artifact": str(Path(index_artifact_path).resolve()) if index_artifact_path else None,
        "index_key": index_key if index_artifact_path else None,
        "examples": len(examples),
        "medium_sparsity": medium_sparsity,
        "aggressive_sparsity": aggressive_sparsity,
        "compute": compute,
        "scoring_seconds": time.perf_counter() - started,
        "physical_channel_pruning": True,
        "claim_boundary": "analytical compute proxy; no latency, packed-size, energy, or throughput claim",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        dense=dense.astype(np.float32),
        baseline_medium=baseline_medium_probabilities.astype(np.float32),
        baseline_aggressive=baseline_aggressive_probabilities.astype(np.float32),
        ursa_medium=ursa_medium_probabilities.astype(np.float32),
        ursa_aggressive=ursa_aggressive_probabilities.astype(np.float32),
        labels=labels.astype(np.int64),
        selected_indices=selected_indices,
        example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return metadata


def lac_threshold(probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample split-conformal threshold for LAC score ``1 - p_y``."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not 0 < alpha < 1 or probabilities.ndim != 2 or labels.shape != (len(probabilities),):
        raise ValueError("invalid probabilities, labels, or alpha")
    scores = 1.0 - probabilities[np.arange(len(labels)), labels]
    rank = min(len(scores), int(math.ceil((len(scores) + 1) * (1.0 - alpha))))
    return float(np.sort(scores, kind="stable")[rank - 1])


def lac_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    included = probabilities >= 1.0 - threshold
    return {
        "coverage": float(included[np.arange(len(labels)), labels].mean()),
        "mean_set_size": float(included.sum(axis=1).mean()),
        "empty_set_rate": float((included.sum(axis=1) == 0).mean()),
    }


def _paired_accuracy_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(differences), size=len(differences))
        samples[index] = differences[selected].mean()
    return {
        "difference": float(differences.mean()),
        "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "replicates": replicates,
    }


def analyze_ursa_comparison(
    artifact_path: str | Path,
    *,
    output_path: str | Path,
    strata: int = 3,
    conformal_alpha: float = 0.1,
    conformal_fraction: float = 0.5,
    split_seed: int = 20260827,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260828,
) -> dict:
    """Analyze accuracy, uncertainty preservation, and conformal efficiency."""

    with np.load(artifact_path, allow_pickle=False) as saved:
        arrays = {
            name: np.asarray(saved[name], dtype=np.float64)
            for name in (
                "dense",
                "baseline_medium",
                "baseline_aggressive",
                "ursa_medium",
                "ursa_aggressive",
            )
        }
        labels = np.asarray(saved["labels"], dtype=np.int64)
        metadata = json.loads(str(saved["metadata"]))
        example_ids = np.asarray(saved["example_ids"], dtype=np.str_)
    dense_uncertainty, dense_strata = uncertainty_strata(arrays["dense"], strata=strata)
    rows = []
    stratum_rows = []
    for name, probabilities in arrays.items():
        metrics = classification_metrics(probabilities, labels)
        metrics.update(
            {
                "condition": name,
                "mean_kl_from_dense": float(np.mean(kl_divergence(arrays["dense"], probabilities))),
                "prediction_agreement_with_dense": float(
                    np.mean(arrays["dense"].argmax(axis=1) == probabilities.argmax(axis=1))
                ),
                "entropy_spearman_with_dense": spearman_correlation(
                    dense_uncertainty, entropy(probabilities, normalized=True)
                ),
            }
        )
        rows.append(metrics)
        for label in range(strata):
            selected = dense_strata == label
            current = classification_metrics(probabilities[selected], labels[selected])
            stratum_rows.append(
                {
                    "condition": name,
                    "stratum": label,
                    "examples": int(selected.sum()),
                    "dense_uncertainty_min": float(dense_uncertainty[selected].min()),
                    "dense_uncertainty_max": float(dense_uncertainty[selected].max()),
                    **current,
                }
            )
    comparisons = []
    for level in ("medium", "aggressive"):
        left = arrays[f"ursa_{level}"].argmax(axis=1) == labels
        right = arrays[f"baseline_{level}"].argmax(axis=1) == labels
        comparisons.append(
            {
                "level": level,
                **_paired_accuracy_interval(
                    left,
                    right,
                    seed=bootstrap_seed + (0 if level == "medium" else 1),
                    replicates=bootstrap_replicates,
                ),
            }
        )
    if not 0 < conformal_fraction < 1:
        raise ValueError("conformal_fraction must be between zero and one")
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(labels))
    calibration_size = max(1, min(len(labels) - 1, int(round(len(labels) * conformal_fraction))))
    calibration_indices, evaluation_indices = order[:calibration_size], order[calibration_size:]
    conformal_rows = []
    dense_threshold = lac_threshold(
        arrays["dense"][calibration_indices], labels[calibration_indices], alpha=conformal_alpha
    )
    for name, probabilities in arrays.items():
        own_threshold = lac_threshold(
            probabilities[calibration_indices], labels[calibration_indices], alpha=conformal_alpha
        )
        conformal_rows.append(
            {
                "condition": name,
                "own_threshold": own_threshold,
                "own_calibrated": lac_metrics(
                    probabilities[evaluation_indices], labels[evaluation_indices], own_threshold
                ),
                "dense_threshold_transfer": lac_metrics(
                    probabilities[evaluation_indices], labels[evaluation_indices], dense_threshold
                ),
            }
        )
    row_by_name = {row["condition"]: row for row in rows}
    stratum_lookup = {
        (row["condition"], row["stratum"]): row for row in stratum_rows
    }
    worst_bin_advantages = {}
    for level in ("medium", "aggressive"):
        advantages = [
            stratum_lookup[(f"ursa_{level}", label)]["accuracy"]
            - stratum_lookup[(f"baseline_{level}", label)]["accuracy"]
            for label in range(strata)
        ]
        worst_bin_advantages[level] = float(min(advantages))
    medium_difference = next(row["difference"] for row in comparisons if row["level"] == "medium")
    aggressive_difference = next(row["difference"] for row in comparisons if row["level"] == "aggressive")
    gate = {
        "passed": bool(
            aggressive_difference >= 0.01
            and medium_difference >= -0.005
            and worst_bin_advantages["aggressive"] >= 0.0
            and row_by_name["ursa_aggressive"]["entropy_spearman_with_dense"]
            >= row_by_name["baseline_aggressive"]["entropy_spearman_with_dense"]
        ),
        "criterion": (
            "URSA minus matched baseline accuracy >=1 pp at aggressive sparsity, >=-0.5 pp at "
            "medium sparsity, no dense-uncertainty stratum loses at aggressive sparsity, and aggressive "
            "entropy-rank preservation does not worsen"
        ),
        "medium_accuracy_difference": medium_difference,
        "aggressive_accuracy_difference": aggressive_difference,
        "worst_stratum_accuracy_advantage": worst_bin_advantages,
    }
    output = {
        "title": "URSA uncertainty-robust structured pruning analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            **metadata,
            "artifact_path": str(Path(artifact_path).resolve()),
            "examples": len(labels),
            "strata": strata,
            "conformal_alpha": conformal_alpha,
            "conformal_calibration_examples": len(calibration_indices),
            "conformal_evaluation_examples": len(evaluation_indices),
            "bootstrap_replicates": bootstrap_replicates,
        },
        "metric_rows": rows,
        "stratum_rows": stratum_rows,
        "paired_comparisons": comparisons,
        "conformal_rows": conformal_rows,
        "gate": gate,
        "example_id_hash_input_count": len(example_ids),
        "limitations": [
            "The dense uncertainty strata are benchmark-relative equal-frequency bins.",
            "LAC conformal coverage is marginal and evaluated on one deterministic split.",
            "Transformer matrix-parameter compute is an analytical proxy, not measured runtime.",
            "This experiment tests one model family and structured FFN-channel pruning.",
        ],
    }
    _write_json(output_path, output)
    return output
