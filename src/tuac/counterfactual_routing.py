"""Counterfactual capacity routing for adaptive sparse LMs (Iteration 7)."""

from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .cascade import uncertainty_scores
from .cascade_compression import _mcnemar, threshold_for_rate
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import (
    binary_auroc,
    binary_average_precision,
    classification_metrics,
    spearman_correlation,
)
from .quantization import transformer_blocks
from .scoring import kl_divergence
from .structured_sparsity import (
    _probability_features,
    apply_dense_masks,
    cross_validated_logistic,
    js_divergence,
    load_probability_artifact,
    nested_masks,
    predict_logistic,
    stratified_folds,
)


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
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_capacity_hierarchy(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {
            "dense": values["dense"], "s25": values["s25"],
            "s40": values["s40"], "s50": values["s50"],
            "labels": values["labels"],
            "example_ids": tuple(str(value) for value in values["example_ids"]),
            "metadata": json.loads(str(values["metadata"])),
        }


def load_representation_artifact(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {
            "hidden_states": values["hidden_states"],
            "selected_probabilities": values["selected_probabilities"],
            "labels": values["labels"],
            "layer_indices": values["layer_indices"],
            "example_ids": tuple(str(value) for value in values["example_ids"]),
            "metadata": json.loads(str(values["metadata"])),
        }


def _mask_for_sparsity(importance: np.ndarray, sparsity: float) -> np.ndarray:
    if not 0.0 < sparsity < 1.0:
        raise ValueError("sparsity must lie strictly between zero and one")
    keep = max(1, int(round(importance.shape[1] * (1.0 - sparsity))))
    order = np.argsort(-np.asarray(importance), axis=1, kind="stable")
    return np.sort(order[:, :keep], axis=1)


def collect_capacity_hierarchy(
    *,
    iteration6_artifact_path: str,
    importance_path: str,
    data_path: str,
    slm_model: str,
    output_path: str,
    masks_path: str,
    batch_size: int = 128,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Reuse Iteration 6 and add a physically pruned S40 calibration point."""

    destination = Path(output_path)
    if destination.exists():
        return load_capacity_hierarchy(destination)
    prior = load_probability_artifact(iteration6_artifact_path)
    examples = load_jsonl(data_path)
    ids = tuple(item.id for item in examples)
    if ids != prior["example_ids"]:
        raise ValueError("ARC development IDs do not match the Iteration 6 artifact")
    with np.load(importance_path, allow_pickle=False) as saved:
        importance = saved["importance"]
    s25_mask = _mask_for_sparsity(importance, 0.25)
    s40_mask = _mask_for_sparsity(importance, 0.40)
    s50_mask = _mask_for_sparsity(importance, 0.50)
    if not all(
        np.isin(more_sparse, less_sparse).all()
        for less_sparse, more_sparse in ((s25_mask, s40_mask), (s40_mask, s50_mask))
    ):
        raise ValueError("capacity masks are not nested")
    masks_destination = Path(masks_path)
    masks_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        masks_destination, s25=s25_mask, s40=s40_mask, s50=s50_mask,
        sparsities=np.asarray([0.25, 0.40, 0.50]),
    )

    scorer = HFCausalLMScorer(
        slm_model, device_map=device_map, dtype=dtype,
        trust_remote_code=trust_remote_code, batch_size=batch_size,
        max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    apply_dense_masks(blocks, s40_mask)
    started = time.perf_counter()
    s40 = scorer.score(examples)
    elapsed = time.perf_counter() - started
    print(f"S40 ARC development scoring completed in {elapsed:.1f}s", flush=True)
    del scorer
    _clear_cuda()

    dense_compute = prior["metadata"]["compute"]["dense"]
    fixed = dense_compute["fixed_transformer_matrix_parameters"]
    dense_mlp = dense_compute["dense_mlp_matrix_parameters"]

    def compute(sparsity: float) -> dict:
        return {
            "normalized_linear_flops": float(
                (fixed + (1.0 - sparsity) * dense_mlp) / (fixed + dense_mlp)
            ),
            "fixed_transformer_matrix_parameters": fixed,
            "dense_mlp_matrix_parameters": dense_mlp,
            "method": dense_compute["method"],
        }

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "counterfactual_capacity_hierarchy",
        "slm_model": slm_model,
        "data_path": str(Path(data_path).resolve()),
        "iteration6_artifact": str(Path(iteration6_artifact_path).resolve()),
        "importance_path": str(Path(importance_path).resolve()),
        "importance": prior["metadata"]["importance"],
        "nested": True,
        "physical_channel_pruning": True,
        "s40_scoring_seconds": elapsed,
        "compute": {
            "dense": compute(0.0), "s25": compute(0.25),
            "s40": compute(0.40), "s50": compute(0.50),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        dense=np.asarray(prior["dense"], dtype=np.float32),
        s25=np.asarray(prior["medium"], dtype=np.float32),
        s40=np.asarray(s40, dtype=np.float32),
        s50=np.asarray(prior["aggressive"], dtype=np.float32),
        labels=np.asarray(prior["labels"], dtype=np.int64),
        example_ids=np.asarray(ids, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return load_capacity_hierarchy(destination)


def _rank_decisions(scores: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, min(len(scores), int(round(float(rate) * len(scores)))))
    order = np.argsort(-np.asarray(scores), kind="stable")
    decisions = np.zeros(len(scores), dtype=bool)
    decisions[order[:count]] = True
    return decisions


def analyze_oracle_headroom(
    hierarchy_path: str,
    *,
    output_path: str,
    target_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    minimum_gain_at_30: float = 0.10,
) -> dict:
    hierarchy = load_capacity_hierarchy(hierarchy_path)
    labels, dense = hierarchy["labels"], hierarchy["dense"]
    dense_correct = dense.argmax(axis=1) == labels
    static_rows, oracle_rows = [], []
    for variant in ("dense", "s25", "s40", "s50"):
        probabilities = hierarchy[variant]
        metrics = classification_metrics(probabilities, labels)
        static_rows.append({
            "variant": variant, **metrics,
            "prediction_agreement_with_dense": float(np.mean(
                probabilities.argmax(axis=1) == dense.argmax(axis=1)
            )),
            "output_kl_from_dense": float(np.mean(kl_divergence(dense, probabilities))),
            "normalized_linear_flops": hierarchy["metadata"]["compute"][variant]["normalized_linear_flops"],
        })
        if variant == "dense":
            continue
        sparse_correct = probabilities.argmax(axis=1) == labels
        signed = dense_correct.astype(np.int8) - sparse_correct.astype(np.int8)
        for rate in target_rates:
            decisions = _rank_decisions(signed, float(rate))
            net_advantage = float(np.mean(signed * decisions))
            sparse_accuracy = float(sparse_correct.mean())
            expected_compute = float(
                hierarchy["metadata"]["compute"][variant]["normalized_linear_flops"]
                + decisions.mean()
            )
            oracle_rows.append({
                "variant": variant, "target_rate": float(rate),
                "dense_execution_rate": float(decisions.mean()),
                "sparse_accuracy": sparse_accuracy,
                "oracle_accuracy": sparse_accuracy + net_advantage,
                "oracle_gain": net_advantage,
                "oracle_efficiency": net_advantage / decisions.mean(),
                "expected_sequential_compute": expected_compute,
                "positive_events": int(np.sum(signed == 1)),
                "harmful_events": int(np.sum(signed == -1)),
                "neutral_events": int(np.sum(signed == 0)),
            })
    chance = float(np.mean(1.0 / np.maximum(2, np.count_nonzero(dense, axis=1))))
    static_lookup = {row["variant"]: row for row in static_rows}
    row30 = {
        row["variant"]: row for row in oracle_rows if row["target_rate"] == 0.3
    }
    eligible = [
        variant for variant in ("s25", "s40", "s50")
        if static_lookup[variant]["accuracy"] > chance + 0.02
        and row30[variant]["oracle_gain"] >= minimum_gain_at_30
    ]
    selected = max(
        eligible,
        key=lambda variant: (
            row30[variant]["oracle_efficiency"],
            row30[variant]["oracle_accuracy"] / row30[variant]["expected_sequential_compute"],
        ),
    ) if eligible else None
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "counterfactual_oracle_selection",
        "scope": {
            "examples": len(labels), "target_rates": list(target_rates),
            "minimum_gain_at_30": minimum_gain_at_30,
        },
        "metadata": hierarchy["metadata"],
        "static_rows": static_rows,
        "oracle_rows": oracle_rows,
        "selection": {
            "selected_sparse_variant": selected,
            "eligible_variants": eligible,
            "rule": "among non-collapsed variants with >=10 pp oracle gain at 30%, maximize oracle gain per incremental sequential compute; break ties by oracle accuracy per total compute",
        },
        "gate_a": {
            "passed": selected is not None,
            "next_stage": "representation_extraction" if selected is not None else "STOP",
            "criterion": "at least one non-collapsed sparse variant has >=10 percentage points oracle gain at 30% dense execution",
        },
    }
    _write_json(Path(output_path), result)
    return result


def _extract_prompt_representations(
    scorer: HFCausalLMScorer,
    examples,
    layer_indices: Sequence[int],
) -> np.ndarray:
    """Extract final prompt-token residual states with one sparse forward per prompt."""

    torch = scorer.torch
    blocks = transformer_blocks(scorer.model)
    if any(index < 0 or index >= len(blocks) for index in layer_indices):
        raise ValueError("representation layer index is outside the transformer")
    sequences = [
        scorer.tokenizer.encode(item.prompt, add_special_tokens=True)[
            -int(getattr(scorer.model.config, "max_position_embeddings", 2048)):
        ]
        for item in examples
    ]
    order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
    batches, current = [], []
    for index in order:
        projected = len(sequences[index]) * (len(current) + 1)
        if current and (
            len(current) >= scorer.batch_size or projected > scorer.max_batch_tokens
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    hidden_size = int(getattr(scorer.model.config, "hidden_size"))
    result = np.empty((len(examples), len(layer_indices), hidden_size), dtype=np.float32)
    captures: dict[int, np.ndarray] = {}
    handles = []
    for output_position, layer_index in enumerate(layer_indices):
        block = blocks[layer_index][1]

        def hook(_module, _inputs, output, position=output_position):
            values = output[0] if isinstance(output, tuple) else output
            captures[position] = values[:, -1, :].detach().float().cpu().numpy()

        handles.append(block.register_forward_hook(hook))
    try:
        for batch_indices in batches:
            encoded = scorer.tokenizer.pad(
                {"input_ids": [sequences[index] for index in batch_indices]},
                padding=True, return_tensors="pt",
            )
            encoded = {key: value.to(scorer._input_device()) for key, value in encoded.items()}
            model_inputs = dict(encoded)
            if scorer._logits_keep_parameter is not None:
                model_inputs[scorer._logits_keep_parameter] = 1
            captures.clear()
            with torch.inference_mode():
                scorer.model(**model_inputs)
            if len(captures) != len(layer_indices):
                raise RuntimeError("not every requested residual layer was captured")
            for position in range(len(layer_indices)):
                result[np.asarray(batch_indices), position] = captures[position]
    finally:
        for handle in handles:
            handle.remove()
    return result


def collect_capacity_representations(
    *,
    hierarchy_path: str,
    oracle_analysis_path: str,
    masks_path: str,
    data_path: str,
    slm_model: str,
    output_path: str,
    layer_indices: Sequence[int] = (6, 13, 20, 27),
    batch_size: int = 128,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    destination = Path(output_path)
    if destination.exists():
        return load_representation_artifact(destination)
    hierarchy = load_capacity_hierarchy(hierarchy_path)
    oracle = json.loads(Path(oracle_analysis_path).read_text(encoding="utf-8"))
    if not oracle["gate_a"]["passed"]:
        raise ValueError("oracle-headroom gate did not authorize representation extraction")
    selected = oracle["selection"]["selected_sparse_variant"]
    examples = load_jsonl(data_path)
    ids = tuple(item.id for item in examples)
    if ids != hierarchy["example_ids"]:
        raise ValueError("representation data IDs do not match capacity hierarchy")
    with np.load(masks_path, allow_pickle=False) as saved:
        mask = saved[selected]
    scorer = HFCausalLMScorer(
        slm_model, device_map=device_map, dtype=dtype,
        trust_remote_code=trust_remote_code, batch_size=batch_size,
        max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    apply_dense_masks(blocks, mask)
    started = time.perf_counter()
    observed = scorer.score(examples)
    reference = hierarchy[selected]
    probability_mae = float(np.mean(np.abs(observed - reference)))
    prediction_agreement = float(np.mean(observed.argmax(axis=1) == reference.argmax(axis=1)))
    if probability_mae > 0.02 or prediction_agreement < 0.97:
        raise ValueError(
            "fresh sparse probabilities materially disagree with hierarchy reference: "
            f"MAE={probability_mae:.6f}, agreement={prediction_agreement:.4f}"
        )
    hidden = _extract_prompt_representations(scorer, examples, layer_indices)
    elapsed = time.perf_counter() - started
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "counterfactual_sparse_representations",
        "slm_model": slm_model, "selected_sparse_variant": selected,
        "layer_indices_zero_based": list(layer_indices),
        "layer_labels": [f"L{index + 1}" for index in layer_indices],
        "representation": "final prompt-token residual stream after selected sparse transformer blocks",
        "sparse_only_forward": True, "physical_channel_pruning": True,
        "scoring_and_extraction_seconds": elapsed,
        "reference_check": {
            "mean_absolute_probability_difference": probability_mae,
            "prediction_agreement": prediction_agreement,
            "acceptance": "MAE <=0.02 and prediction agreement >=0.97",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, hidden_states=hidden.astype(np.float16),
        selected_probabilities=observed.astype(np.float32),
        labels=np.asarray(hierarchy["labels"], dtype=np.int64),
        layer_indices=np.asarray(layer_indices, dtype=np.int64),
        example_ids=np.asarray(ids, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    del scorer
    _clear_cuda()
    print(f"sparse representation extraction completed in {elapsed:.1f}s", flush=True)
    return load_representation_artifact(destination)


def fit_ridge(features: np.ndarray, targets: np.ndarray, *, alpha: float = 1.0) -> dict:
    """Fit standardized multi-target ridge, using the dual when p > n."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (features - mean) / scale / math.sqrt(features.shape[1])
    target_mean = targets.mean(axis=0)
    centered = targets - target_mean
    if features.shape[1] > features.shape[0]:
        kernel = normalized @ normalized.T
        dual = np.linalg.solve(
            kernel + alpha * np.eye(len(features)), centered
        )
        return {
            "kind": "dual", "mean": mean, "scale": scale,
            "target_mean": target_mean, "train_normalized": normalized,
            "weights": dual, "feature_count": features.shape[1], "alpha": alpha,
        }
    gram = normalized.T @ normalized
    weights = np.linalg.solve(
        gram + alpha * np.eye(features.shape[1]), normalized.T @ centered
    )
    return {
        "kind": "primal", "mean": mean, "scale": scale,
        "target_mean": target_mean, "weights": weights,
        "feature_count": features.shape[1], "alpha": alpha,
    }


def predict_ridge(model: dict, features: np.ndarray) -> np.ndarray:
    normalized = (
        (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
        / math.sqrt(model["feature_count"])
    )
    if model["kind"] == "dual":
        values = (normalized @ model["train_normalized"].T) @ model["weights"]
    else:
        values = normalized @ model["weights"]
    return values + model["target_mean"]


def cross_validated_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    stratification_labels: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 20260823,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    targets = np.asarray(targets)
    if targets.ndim == 1:
        targets = targets[:, None]
    heldout_folds = stratified_folds(stratification_labels, folds=folds, seed=seed)
    predictions = np.empty_like(targets, dtype=np.float64)
    fold_ids = np.empty(len(features), dtype=np.int64)
    all_indices = np.arange(len(features))
    for fold, heldout in enumerate(heldout_folds):
        train = np.setdiff1d(all_indices, heldout, assume_unique=True)
        model = fit_ridge(features[train], targets[train], alpha=alpha)
        predictions[heldout] = predict_ridge(model, features[heldout])
        fold_ids[heldout] = fold
    return predictions, fold_ids, fit_ridge(features, targets, alpha=alpha)


def pairwise_ranking_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    left, right = np.triu_indices(len(scores), k=1)
    target_difference = np.asarray(targets)[left] - np.asarray(targets)[right]
    eligible = target_difference != 0
    if not eligible.any():
        return float("nan")
    score_difference = np.asarray(scores)[left] - np.asarray(scores)[right]
    target_sign = np.sign(target_difference[eligible])
    score_sign = np.sign(score_difference[eligible])
    return float(np.mean(np.where(score_sign == 0, 0.5, score_sign == target_sign)))


def _policy_row(
    method: str,
    scores: np.ndarray,
    rate: float,
    signed_advantage: np.ndarray,
    sparse_correct: np.ndarray,
    sparse_compute: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    decisions = _rank_decisions(scores, rate)
    final_correct = sparse_correct.astype(np.int8) + signed_advantage * decisions
    oracle_decisions = _rank_decisions(signed_advantage, rate)
    oracle_nac = float(np.mean(signed_advantage * oracle_decisions))
    nac = float(np.mean(signed_advantage * decisions))
    selected = max(1, int(decisions.sum()))
    row = {
        "method": method, "target_rate": float(rate),
        "dense_execution_rate": float(decisions.mean()),
        "accuracy": float(final_correct.mean()),
        "expected_sequential_compute": float(sparse_compute + decisions.mean()),
        "expected_selected_capacity": float(
            (1.0 - decisions.mean()) * sparse_compute + decisions.mean()
        ),
        "net_advantage_capture": nac,
        "oracle_recovery": nac / oracle_nac if oracle_nac > 0 else None,
        "useful_densification_rate": float(np.sum(decisions & (signed_advantage == 1)) / selected),
        "harmful_densification_rate": float(np.sum(decisions & (signed_advantage == -1)) / selected),
        "wasted_densification_rate": float(np.sum(decisions & (signed_advantage == 0)) / selected),
        "missed_positive_advantage_rate": float(np.mean(~decisions & (signed_advantage == 1))),
    }
    return row, decisions, final_correct.astype(bool)


def _paired_policy_interval(
    left_decisions: np.ndarray,
    right_decisions: np.ndarray,
    signed_advantage: np.ndarray,
    sparse_correct: np.ndarray,
    rate: float,
    *,
    seed: int,
    replicates: int,
) -> dict:
    rng = np.random.default_rng(seed)
    accuracy, nac, recovery = [], [], []
    for _ in range(replicates):
        indices = rng.integers(0, len(signed_advantage), size=len(signed_advantage))
        advantage = signed_advantage[indices]
        left = left_decisions[indices]
        right = right_decisions[indices]
        base = sparse_correct[indices].astype(np.int8)
        accuracy.append(float(np.mean((base + advantage * left) - (base + advantage * right))))
        nac.append(float(np.mean(advantage * left) - np.mean(advantage * right)))
        bootstrap_oracle = _rank_decisions(advantage, rate)
        oracle_nac = float(np.mean(advantage * bootstrap_oracle))
        if oracle_nac > 0:
            recovery.append(float((np.mean(advantage * left) - np.mean(advantage * right)) / oracle_nac))
    return {
        "accuracy_difference": float(np.mean(signed_advantage * left_decisions) - np.mean(signed_advantage * right_decisions)),
        "accuracy_difference_ci": np.quantile(accuracy, [0.025, 0.975]).tolist(),
        "nac_difference": float(np.mean(signed_advantage * left_decisions) - np.mean(signed_advantage * right_decisions)),
        "nac_difference_ci": np.quantile(nac, [0.025, 0.975]).tolist(),
        "oracle_recovery_difference_ci": np.quantile(recovery, [0.025, 0.975]).tolist() if recovery else [None, None],
    }


def analyze_counterfactual_routing(
    *,
    hierarchy_path: str,
    oracle_analysis_path: str,
    representations_path: str,
    output_path: str,
    probe_output_path: str,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    target_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    random_seeds: Sequence[int] = tuple(range(10)),
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260824,
) -> dict:
    hierarchy = load_capacity_hierarchy(hierarchy_path)
    oracle = json.loads(Path(oracle_analysis_path).read_text(encoding="utf-8"))
    representations = load_representation_artifact(representations_path)
    selected = oracle["selection"]["selected_sparse_variant"]
    if selected != representations["metadata"]["selected_sparse_variant"]:
        raise ValueError("representation variant does not match oracle selection")
    labels = hierarchy["labels"]
    dense = hierarchy["dense"]
    sparse = representations["selected_probabilities"]
    instability_reference = {"s50": "s25", "s40": "s25", "s25": "dense"}[selected]
    medium = hierarchy[instability_reference]
    adjacent_variant = {"s50": "s40", "s40": "s25", "s25": "dense"}[selected]
    adjacent = hierarchy[adjacent_variant]
    dense_correct = dense.argmax(axis=1) == labels
    sparse_correct = sparse.argmax(axis=1) == labels
    signed_advantage = dense_correct.astype(np.int8) - sparse_correct.astype(np.int8)
    positive_advantage = signed_advantage == 1
    error_target = ~sparse_correct
    loss_advantage = np.log(
        np.clip(dense[np.arange(len(labels)), labels], 1e-12, 1.0)
        / np.clip(sparse[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    instability = sparse.argmax(axis=1) != medium.argmax(axis=1)
    soft_instability = js_divergence(sparse, medium)
    adjacent_instability = sparse.argmax(axis=1) != adjacent.argmax(axis=1)
    adjacent_soft_instability = js_divergence(sparse, adjacent)
    output_features = _probability_features(sparse)
    hidden = representations["hidden_states"].astype(np.float64)
    layer_labels = representations["metadata"]["layer_labels"]

    error_oof, error_folds, _ = cross_validated_logistic(
        output_features, error_target, folds=folds, seed=bootstrap_seed
    )
    uplift_oof, uplift_folds, _ = cross_validated_logistic(
        output_features, positive_advantage, folds=folds, seed=bootstrap_seed + 1
    )
    ridge_targets = np.column_stack([
        signed_advantage, loss_advantage, instability.astype(float), soft_instability
    ])
    output_ridge, output_fold_ids, _ = cross_validated_ridge(
        output_features, ridge_targets, positive_advantage,
        folds=folds, seed=bootstrap_seed + 2, alpha=ridge_alpha,
    )
    sparse_uncertainty = uncertainty_scores(sparse)
    scores = {
        "margin": sparse_uncertainty["margin"],
        "entropy": sparse_uncertainty["entropy"],
        "max_probability": sparse_uncertainty["max_probability"],
        "error_probe": error_oof,
        "binary_uplift_probe": uplift_oof,
        "output_signed_ridge": output_ridge[:, 0],
        "output_loss_ridge": output_ridge[:, 1],
        "output_instability_ridge": output_ridge[:, 2],
        "actual_capacity_disagreement": instability.astype(float),
        "actual_capacity_js": soft_instability,
        "actual_adjacent_capacity_disagreement": adjacent_instability.astype(float),
        "actual_adjacent_capacity_js": adjacent_soft_instability,
        "error_oracle": error_target.astype(float),
        "positive_uplift_oracle": positive_advantage.astype(float),
        "signed_advantage_oracle": signed_advantage.astype(float),
    }
    fold_ids_by_method = {
        "error_probe": np.concatenate([
            np.full(row["evaluation_examples"], row["fold"]) for row in error_folds
        ]) if False else output_fold_ids,
        "binary_uplift_probe": output_fold_ids,
    }
    feature_sets = {}
    for position, label in enumerate(layer_labels):
        feature_sets[f"hidden_{label.lower()}"] = hidden[:, position, :]
    feature_sets["hidden_concat"] = hidden.reshape(len(hidden), -1)
    feature_sets["hidden_concat_output"] = np.column_stack([
        hidden.reshape(len(hidden), -1), output_features
    ])
    ridge_models = {}
    for offset, (feature_name, features) in enumerate(feature_sets.items()):
        predictions, fold_ids, model = cross_validated_ridge(
            features, ridge_targets, positive_advantage, folds=folds,
            seed=bootstrap_seed + 10 + offset, alpha=ridge_alpha,
        )
        scores[f"signed_advantage_{feature_name}"] = predictions[:, 0]
        scores[f"loss_advantage_{feature_name}"] = predictions[:, 1]
        scores[f"predicted_instability_{feature_name}"] = predictions[:, 2]
        scores[f"predicted_js_{feature_name}"] = predictions[:, 3]
        for prefix in ("signed_advantage", "loss_advantage", "predicted_instability", "predicted_js"):
            fold_ids_by_method[f"{prefix}_{feature_name}"] = fold_ids
        ridge_models[feature_name] = model

    estimator_rows = []
    oracle_names = {"error_oracle", "positive_uplift_oracle", "signed_advantage_oracle"}
    for name, values in scores.items():
        fold_ids = fold_ids_by_method.get(name, output_fold_ids)
        fold_spearman = []
        for fold in range(folds):
            heldout = fold_ids == fold
            fold_spearman.append(spearman_correlation(values[heldout], loss_advantage[heldout]))
        estimator_rows.append({
            "estimator": name,
            "deployability": "oracle" if name in oracle_names else "expensive_diagnostic" if name.startswith("actual_") else "sparse_only",
            "auroc_positive_advantage": binary_auroc(values, positive_advantage),
            "auprc_positive_advantage": binary_average_precision(values, positive_advantage),
            "loss_advantage_spearman": spearman_correlation(values, loss_advantage),
            "signed_pairwise_accuracy": pairwise_ranking_accuracy(values, signed_advantage),
            "fold_loss_spearman": fold_spearman,
            "positive_fold_spearman_count": int(np.sum(np.asarray(fold_spearman) > 0)),
        })

    sparse_compute = oracle["metadata"]["compute"][selected]["normalized_linear_flops"]
    policy_rows, decisions_by_key, correctness_by_key = [], {}, {}
    for name, values in scores.items():
        for rate in target_rates:
            row, decisions, correct = _policy_row(
                name, values, float(rate), signed_advantage, sparse_correct, sparse_compute
            )
            policy_rows.append(row)
            decisions_by_key[(name, float(rate))] = decisions
            correctness_by_key[(name, float(rate))] = correct
    random_rows = []
    for seed in random_seeds:
        random_scores = np.random.default_rng(int(seed)).random(len(labels))
        for rate in target_rates:
            row, _, _ = _policy_row(
                f"random_seed_{seed}", random_scores, float(rate),
                signed_advantage, sparse_correct, sparse_compute,
            )
            random_rows.append({"seed": int(seed), **row})
    random_summary = []
    for rate in target_rates:
        rows = [row for row in random_rows if row["target_rate"] == float(rate)]
        random_summary.append({
            "target_rate": float(rate),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
            "accuracy_sd": float(np.std([row["accuracy"] for row in rows], ddof=1)),
            "nac_mean": float(np.mean([row["net_advantage_capture"] for row in rows])),
            "oracle_recovery_mean": float(np.mean([row["oracle_recovery"] for row in rows])),
        })

    estimator_lookup = {row["estimator"]: row for row in estimator_rows}
    policy_lookup = {(row["method"], row["target_rate"]): row for row in policy_rows}
    generic_names = ("margin", "entropy", "max_probability")
    representation_names = [
        name for name in scores
        if name.startswith(("signed_advantage_hidden", "loss_advantage_hidden", "predicted_instability_hidden", "predicted_js_hidden"))
    ]
    best_generic_auc = max(
        estimator_lookup[name]["auroc_positive_advantage"] for name in generic_names
    )
    best_generic_nac30 = max(
        policy_lookup[(name, 0.3)]["net_advantage_capture"] for name in generic_names
    )
    best_representation = max(
        representation_names,
        key=lambda name: (
            policy_lookup[(name, 0.3)]["oracle_recovery"],
            estimator_lookup[name]["auroc_positive_advantage"],
        ),
    )
    best_rep_metrics = estimator_lookup[best_representation]
    best_rep_policy30 = policy_lookup[(best_representation, 0.3)]
    stable_ranking = best_rep_metrics["positive_fold_spearman_count"] >= folds - 1
    gate_b_passed = bool(
        best_rep_metrics["auroc_positive_advantage"] > best_generic_auc
        and best_rep_policy30["net_advantage_capture"] > best_generic_nac30
        and stable_ranking
        and (
            best_rep_metrics["auroc_positive_advantage"] >= 0.68
            or best_rep_policy30["oracle_recovery"] >= 0.40
        )
    )
    comparisons = []
    for rate in target_rates:
        for baseline in ("entropy", "binary_uplift_probe", "output_signed_ridge"):
            left_decisions = decisions_by_key[(best_representation, float(rate))]
            right_decisions = decisions_by_key[(baseline, float(rate))]
            values = _paired_policy_interval(
                left_decisions, right_decisions, signed_advantage, sparse_correct,
                float(rate), seed=bootstrap_seed + int(float(rate) * 1000),
                replicates=bootstrap_replicates,
            )
            comparisons.append({
                "target_rate": float(rate),
                "comparison": f"{best_representation}_minus_{baseline}",
                **values,
                "mcnemar": _mcnemar(
                    correctness_by_key[(best_representation, float(rate))],
                    correctness_by_key[(baseline, float(rate))],
                ),
            })
    thresholds = {
        name: {
            str(float(rate)): threshold_for_rate(values, float(rate))
            for rate in target_rates
        }
        for name, values in scores.items()
        if name not in oracle_names
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "counterfactual_arc_development",
        "scope": {
            "examples": len(labels), "folds": folds, "ridge_alpha": ridge_alpha,
            "target_rates": list(target_rates), "random_seeds": list(random_seeds),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "data_role": "development only; ARC validation was not reused",
        },
        "selected_sparse_variant": selected,
        "instability_reference_variant": instability_reference,
        "adjacent_diagnostic_variant": adjacent_variant,
        "sparse_compute": sparse_compute,
        "targets": {
            "positive_advantage_count": int(np.sum(signed_advantage == 1)),
            "neutral_advantage_count": int(np.sum(signed_advantage == 0)),
            "harmful_advantage_count": int(np.sum(signed_advantage == -1)),
            "instability_count": int(instability.sum()),
            "adjacent_instability_count": int(adjacent_instability.sum()),
            "loss_advantage_mean": float(loss_advantage.mean()),
        },
        "estimator_rows": estimator_rows,
        "policy_rows": policy_rows,
        "random_rows": random_rows,
        "random_summary": random_summary,
        "comparisons": comparisons,
        "thresholds": thresholds,
        "cross_validation": {
            "error_probe": error_folds, "binary_uplift_probe": uplift_folds,
            "ridge_stratification": "positive signed-advantage event",
        },
        "gate_b": {
            "passed": gate_b_passed,
            "next_stage": "fresh_mmlu_pro" if gate_b_passed else "STOP",
            "criterion": "a representation method exceeds all output uncertainty in AUROC and 30% NAC, has positive loss-advantage rank correlation in >=4/5 folds, and has AUROC >=0.68 or OR@30% >=0.40",
            "selected_method": best_representation,
            "selected_auroc": best_rep_metrics["auroc_positive_advantage"],
            "selected_oracle_recovery_at_30": best_rep_policy30["oracle_recovery"],
            "selected_nac_at_30": best_rep_policy30["net_advantage_capture"],
            "best_generic_auroc": best_generic_auc,
            "best_generic_nac_at_30": best_generic_nac30,
            "stable_ranking": stable_ranking,
        },
    }
    probe_destination = Path(probe_output_path)
    probe_destination.parent.mkdir(parents=True, exist_ok=True)
    ordered_names = list(scores)
    np.savez_compressed(
        probe_destination,
        score_names=np.asarray(ordered_names, dtype=np.str_),
        scores=np.stack([scores[name] for name in ordered_names]).astype(np.float32),
        signed_advantage=signed_advantage,
        loss_advantage=loss_advantage.astype(np.float32),
        instability=instability,
        soft_instability=soft_instability.astype(np.float32),
        sparse_correct=sparse_correct,
        dense_correct=dense_correct,
        metadata=np.asarray(json.dumps({
            "selected_sparse_variant": selected,
            "layer_labels": layer_labels,
            "ridge_alpha": ridge_alpha,
            "all_predictions_are_out_of_fold": True,
        }, sort_keys=True), dtype=np.str_),
    )
    _write_json(Path(output_path), result)
    return result
