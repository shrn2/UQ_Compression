"""Uncertainty-guided progressive densification (U-Dense, Iteration 6)."""

from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .cascade import CascadeProbabilityArtifact, uncertainty_scores
from .cascade_compression import _mcnemar, _paired_interval, threshold_for_rate
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import (
    binary_auroc,
    binary_average_precision,
    classification_metrics,
    spearman_correlation,
)
from .quantization import transformer_blocks
from .scoring import entropy, kl_divergence


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


def _save_probability_artifact(
    path: Path,
    *,
    dense: np.ndarray,
    medium: np.ndarray | None,
    aggressive: np.ndarray | None,
    labels: np.ndarray,
    example_ids: Sequence[str],
    metadata: dict,
) -> None:
    values = {
        "dense": np.asarray(dense, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "example_ids": np.asarray(example_ids, dtype=np.str_),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    }
    if medium is not None:
        values["medium"] = np.asarray(medium, dtype=np.float32)
    if aggressive is not None:
        values["aggressive"] = np.asarray(aggressive, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def load_probability_artifact(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {
            "dense": values["dense"],
            "medium": values["medium"] if "medium" in values.files else None,
            "aggressive": values["aggressive"] if "aggressive" in values.files else None,
            "labels": values["labels"],
            "example_ids": tuple(str(value) for value in values["example_ids"]),
            "metadata": json.loads(str(values["metadata"])),
        }


def ffn_modules(block) -> tuple[object, object, object]:
    mlp = getattr(block, "mlp", None)
    if mlp is None or not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("expected a gated MLP with gate_proj, up_proj, and down_proj")
    return mlp.gate_proj, mlp.up_proj, mlp.down_proj


def prune_ffn_channels(block, retained: Sequence[int]) -> None:
    """Physically shrink one gated FFN to the ordered retained channels."""

    import torch

    gate, up, down = ffn_modules(block)
    indices = torch.as_tensor(retained, dtype=torch.long, device=gate.weight.device)
    if indices.ndim != 1 or indices.numel() < 1:
        raise ValueError("retained channels must be a non-empty vector")
    if int(indices.min()) < 0 or int(indices.max()) >= gate.out_features:
        raise ValueError("retained channel index out of range")
    device, dtype = gate.weight.device, gate.weight.dtype

    def linear(in_features: int, out_features: int, bias: bool):
        return torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    new_gate = linear(gate.in_features, indices.numel(), gate.bias is not None)
    new_up = linear(up.in_features, indices.numel(), up.bias is not None)
    new_down = linear(indices.numel(), down.out_features, down.bias is not None)
    with torch.no_grad():
        new_gate.weight.copy_(gate.weight.index_select(0, indices))
        new_up.weight.copy_(up.weight.index_select(0, indices))
        new_down.weight.copy_(down.weight.index_select(1, indices))
        if gate.bias is not None:
            new_gate.bias.copy_(gate.bias.index_select(0, indices))
        if up.bias is not None:
            new_up.bias.copy_(up.bias.index_select(0, indices))
        if down.bias is not None:
            new_down.bias.copy_(down.bias)
    block.mlp.gate_proj = new_gate
    block.mlp.up_proj = new_up
    block.mlp.down_proj = new_down
    if hasattr(block.mlp, "intermediate_size"):
        block.mlp.intermediate_size = int(indices.numel())


def apply_dense_masks(blocks: Sequence[tuple[str, object]], dense_indices: np.ndarray) -> None:
    if dense_indices.shape[0] != len(blocks):
        raise ValueError("one retained-index row is required per block")
    for (_, block), retained in zip(blocks, dense_indices):
        prune_ffn_channels(block, retained.tolist())


def register_channel_mask_hooks(
    blocks: Sequence[tuple[str, object]], dense_indices: np.ndarray
) -> list[object]:
    """Apply logical FFN masks without replacing quantized Linear modules.

    This is the quantized-model validation path: the forward computation remains
    in the backend's quantized Linear4bit/Linear8bitLt modules, while retained
    channels are zeroed immediately before ``down_proj``. It establishes model
    quality under a quantized backend, but not packed sparse-kernel speed.
    """

    import torch

    indices = np.asarray(dense_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != len(blocks):
        raise ValueError("one retained-index row is required per block")
    handles: list[object] = []
    for (_, block), retained in zip(blocks, indices):
        _, _, down = ffn_modules(block)
        if retained.ndim != 1 or retained.size < 1:
            raise ValueError("retained channels must be non-empty vectors")
        if int(retained.min()) < 0 or int(retained.max()) >= down.in_features:
            raise ValueError("retained channel index out of range")
        mask = torch.zeros(down.in_features, device=down.weight.device, dtype=torch.float32)
        mask[torch.as_tensor(retained, dtype=torch.long, device=mask.device)] = 1.0

        def hook(_module, inputs, channel_mask=mask):
            values = inputs[0]
            return (values * channel_mask.to(device=values.device, dtype=values.dtype).view(1, 1, -1),)

        handles.append(down.register_forward_pre_hook(hook))
    return handles


def nested_masks(importance: np.ndarray, medium_sparsity: float, aggressive_sparsity: float) -> dict:
    importance = np.asarray(importance, dtype=np.float64)
    if importance.ndim != 2 or not 0 <= medium_sparsity < aggressive_sparsity < 1:
        raise ValueError("importance must be [layers, channels] with ordered sparsity levels")
    channels = importance.shape[1]
    medium_keep = max(1, int(round(channels * (1.0 - medium_sparsity))))
    aggressive_keep = max(1, int(round(channels * (1.0 - aggressive_sparsity))))
    order = np.argsort(-importance, axis=1, kind="stable")
    medium = np.sort(order[:, :medium_keep], axis=1)
    aggressive = np.sort(order[:, :aggressive_keep], axis=1)
    return {"medium": medium, "aggressive": aggressive}


def linear_compute_ratios(model, blocks: Sequence[tuple[str, object]], retained_fraction: float) -> dict:
    fixed, mlp = 0, 0
    for _, block in blocks:
        gate, up, down = ffn_modules(block)
        mlp += gate.weight.numel() + up.weight.numel() + down.weight.numel()
        for name, parameter in block.named_parameters():
            if not name.startswith("mlp.gate_proj") and not name.startswith("mlp.up_proj") and not name.startswith("mlp.down_proj") and parameter.ndim >= 2:
                fixed += parameter.numel()
    dense = fixed + mlp
    sparse = fixed + retained_fraction * mlp
    return {
        "fixed_transformer_matrix_parameters": int(fixed),
        "dense_mlp_matrix_parameters": int(mlp),
        "normalized_linear_flops": float(sparse / dense),
        "method": "transformer matrix multiply parameter-count proxy; embeddings and LM head excluded",
    }


def collect_sparse_hierarchy(
    *,
    calibration_data_path: str,
    cascade_artifact_path: str,
    slm_model: str,
    output_path: str,
    masks_path: str,
    importance_path: str,
    medium_sparsity: float = 0.25,
    aggressive_sparsity: float = 0.50,
    collapse_margin: float = 0.02,
    batch_size: int = 128,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Collect final-token activation importance and nested sparse probabilities."""

    destination = Path(output_path)
    if destination.exists():
        existing = load_probability_artifact(destination)
        if existing["medium"] is not None and existing["aggressive"] is not None:
            return existing
    cascade = CascadeProbabilityArtifact.load(cascade_artifact_path)
    examples = load_jsonl(calibration_data_path)
    ids = tuple(row.id for row in examples)
    if ids != cascade.calibration_ids:
        raise ValueError("calibration IDs do not match cascade artifact")
    scorer = HFCausalLMScorer(
        slm_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code,
        batch_size=batch_size, max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    importance_file, masks_file = Path(importance_path), Path(masks_path)
    if importance_file.exists() and masks_file.exists():
        with np.load(importance_file, allow_pickle=False) as saved_importance:
            importance = saved_importance["importance"]
            dense_probabilities = saved_importance["dense_probabilities"]
            dense_reference_mae = float(saved_importance["dense_reference_mae"])
            dense_reference_prediction_agreement = float(saved_importance["dense_reference_prediction_agreement"])
        with np.load(masks_file, allow_pickle=False) as saved:
            medium_masks = saved["medium"]
            aggressive_masks = saved["aggressive"]
            actual_aggressive = float(saved["aggressive_sparsity"])
    else:
        sums, counts, handles = [], [], []
        for _, block in blocks:
            _, _, down = ffn_modules(block)
            sums.append(np.zeros(down.in_features, dtype=np.float64))
            counts.append(0)
        for index, (_, block) in enumerate(blocks):
            _, _, down = ffn_modules(block)

            def hook(_module, inputs, layer_index=index):
                values = inputs[0].detach()[:, -1, :].abs().sum(dim=0).float().cpu().numpy()
                sums[layer_index][:] += values
                counts[layer_index] += int(inputs[0].shape[0])

            handles.append(down.register_forward_pre_hook(hook))
        started = time.perf_counter()
        observed_dense = scorer.score(examples)
        for handle in handles:
            handle.remove()
        dense_reference_mae = float(np.mean(np.abs(observed_dense - cascade.slm_fp_calibration)))
        dense_reference_prediction_agreement = float(np.mean(
            observed_dense.argmax(axis=1) == cascade.slm_fp_calibration.argmax(axis=1)
        ))
        if dense_reference_mae > 0.01 or dense_reference_prediction_agreement < 0.98:
            raise ValueError(
                "fresh dense probabilities materially disagree with cached calibration reference: "
                f"MAE={dense_reference_mae:.6f}, prediction agreement={dense_reference_prediction_agreement:.4f}"
            )
        dense_probabilities = observed_dense
        rows = []
        for index, (_, block) in enumerate(blocks):
            _, _, down = ffn_modules(block)
            activation = sums[index] / max(1, counts[index])
            weight_norm = down.weight.detach().float().norm(dim=0).cpu().numpy()
            rows.append(activation * weight_norm)
        importance = np.stack(rows)
        masks = nested_masks(importance, medium_sparsity, aggressive_sparsity)
        medium_masks, aggressive_masks = masks["medium"], masks["aggressive"]
        actual_aggressive = aggressive_sparsity
        importance_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            importance_file,
            importance=importance,
            activation_counts=np.asarray(counts),
            dense_probabilities=dense_probabilities.astype(np.float32),
            dense_reference_mae=np.asarray(dense_reference_mae),
            dense_reference_prediction_agreement=np.asarray(dense_reference_prediction_agreement),
        )
        masks_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            masks_file, medium=medium_masks, aggressive=aggressive_masks,
            medium_sparsity=np.asarray(medium_sparsity), aggressive_sparsity=np.asarray(actual_aggressive),
        )
        print(f"activation importance completed in {time.perf_counter() - started:.1f}s", flush=True)

    original_width = importance.shape[1]
    compute = {
        "dense": linear_compute_ratios(scorer.model, blocks, 1.0),
        "medium": linear_compute_ratios(scorer.model, blocks, medium_masks.shape[1] / original_width),
        "aggressive": linear_compute_ratios(scorer.model, blocks, aggressive_masks.shape[1] / original_width),
    }
    apply_dense_masks(blocks, medium_masks)
    started = time.perf_counter()
    medium = scorer.score(examples)
    print(f"medium sparse calibration completed in {time.perf_counter() - started:.1f}s", flush=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "slm_model": slm_model,
        "data_path": str(Path(calibration_data_path).resolve()), "cascade_artifact": str(Path(cascade_artifact_path).resolve()),
        "medium_sparsity": medium_sparsity, "aggressive_sparsity": actual_aggressive,
        "importance": "final-prompt-token mean absolute FFN activation times down-projection column norm",
        "nested": True, "physical_channel_pruning": True, "batch_size": batch_size,
        "dense_reference_check": {
            "mean_absolute_probability_difference": dense_reference_mae,
            "prediction_agreement": dense_reference_prediction_agreement,
            "acceptance": "MAE <= 0.01 and prediction agreement >= 0.98",
        },
        "compute": compute,
    }
    _save_probability_artifact(destination, dense=dense_probabilities, medium=medium, aggressive=None, labels=cascade.calibration_labels, example_ids=ids, metadata=metadata)

    # Aggressive indices are expressed in dense coordinates; map them into the
    # already-pruned, sorted medium coordinate system.
    local_aggressive = np.stack([np.searchsorted(medium_row, aggressive_row) for medium_row, aggressive_row in zip(medium_masks, aggressive_masks)])
    apply_dense_masks(blocks, local_aggressive)
    started = time.perf_counter()
    aggressive = scorer.score(examples)
    print(f"aggressive sparse calibration completed in {time.perf_counter() - started:.1f}s", flush=True)

    classes = dense_probabilities.shape[1]
    chance = 1.0 / classes
    aggressive_accuracy = float(np.mean(aggressive.argmax(axis=1) == cascade.calibration_labels))
    if actual_aggressive >= 0.50 and aggressive_accuracy <= chance + collapse_margin:
        # Rebuild once at 40% sparsity. This branch deliberately reloads dense
        # weights rather than attempting to restore removed channels.
        print(f"50% sparsity collapsed to {aggressive_accuracy:.3f}; retrying 40%", flush=True)
        del scorer
        _clear_cuda()
        actual_aggressive = 0.40
        masks = nested_masks(importance, medium_sparsity, actual_aggressive)
        aggressive_masks = masks["aggressive"]
        np.savez_compressed(
            masks_file, medium=medium_masks, aggressive=aggressive_masks,
            medium_sparsity=np.asarray(medium_sparsity), aggressive_sparsity=np.asarray(actual_aggressive),
        )
        scorer = HFCausalLMScorer(
            slm_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code,
            batch_size=batch_size, max_batch_tokens=4096,
        )
        blocks = transformer_blocks(scorer.model)
        apply_dense_masks(blocks, aggressive_masks)
        aggressive = scorer.score(examples)
        fixed = compute["dense"]["fixed_transformer_matrix_parameters"]
        dense_mlp = compute["dense"]["dense_mlp_matrix_parameters"]
        fraction = aggressive_masks.shape[1] / original_width
        compute["aggressive"] = {
            "fixed_transformer_matrix_parameters": fixed,
            "dense_mlp_matrix_parameters": dense_mlp,
            "normalized_linear_flops": float((fixed + fraction * dense_mlp) / (fixed + dense_mlp)),
            "method": compute["dense"]["method"],
        }
        metadata["aggressive_sparsity"] = actual_aggressive
        metadata["compute"] = compute

    _save_probability_artifact(destination, dense=dense_probabilities, medium=medium, aggressive=aggressive, labels=cascade.calibration_labels, example_ids=ids, metadata=metadata)
    del scorer
    _clear_cuda()
    return load_probability_artifact(destination)


def score_sparse_hierarchy(
    *,
    data_path: str,
    cascade_artifact_path: str,
    masks_path: str,
    slm_model: str,
    output_path: str,
    batch_size: int = 128,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    destination = Path(output_path)
    if destination.exists():
        existing = load_probability_artifact(destination)
        if existing["medium"] is not None and existing["aggressive"] is not None:
            return existing
    cascade = CascadeProbabilityArtifact.load(cascade_artifact_path)
    examples = load_jsonl(data_path)
    ids = tuple(row.id for row in examples)
    if ids != cascade.evaluation_ids:
        raise ValueError("evaluation IDs do not match cascade artifact")
    with np.load(masks_path, allow_pickle=False) as masks:
        medium_masks, aggressive_masks = masks["medium"], masks["aggressive"]
        medium_sparsity, aggressive_sparsity = float(masks["medium_sparsity"]), float(masks["aggressive_sparsity"])
    scorer = HFCausalLMScorer(
        slm_model, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code,
        batch_size=batch_size, max_batch_tokens=4096,
    )
    blocks = transformer_blocks(scorer.model)
    dense = scorer.score(examples)
    dense_reference_mae = float(np.mean(np.abs(dense - cascade.slm_fp_evaluation)))
    dense_reference_prediction_agreement = float(np.mean(
        dense.argmax(axis=1) == cascade.slm_fp_evaluation.argmax(axis=1)
    ))
    if dense_reference_mae > 0.01 or dense_reference_prediction_agreement < 0.98:
        raise ValueError(
            "fresh dense probabilities materially disagree with cached evaluation reference: "
            f"MAE={dense_reference_mae:.6f}, prediction agreement={dense_reference_prediction_agreement:.4f}"
        )
    apply_dense_masks(blocks, medium_masks)
    medium = scorer.score(examples)
    local_aggressive = np.stack([np.searchsorted(medium_row, aggressive_row) for medium_row, aggressive_row in zip(medium_masks, aggressive_masks)])
    apply_dense_masks(blocks, local_aggressive)
    aggressive = scorer.score(examples)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "slm_model": slm_model,
        "data_path": str(Path(data_path).resolve()), "medium_sparsity": medium_sparsity,
        "aggressive_sparsity": aggressive_sparsity, "nested": True, "physical_channel_pruning": True,
        "dense_reference_check": {
            "mean_absolute_probability_difference": dense_reference_mae,
            "prediction_agreement": dense_reference_prediction_agreement,
            "acceptance": "MAE <= 0.01 and prediction agreement >= 0.98",
        },
    }
    _save_probability_artifact(destination, dense=dense, medium=medium, aggressive=aggressive, labels=cascade.evaluation_labels, example_ids=ids, metadata=metadata)
    del scorer
    _clear_cuda()
    return load_probability_artifact(destination)


def _probability_features(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scores = uncertainty_scores(probabilities)
    ordered = np.sort(np.clip(probabilities, 1e-12, None), axis=1)
    logit_gap_uncertainty = -np.log(ordered[:, -1] / ordered[:, -2])
    return np.column_stack([
        scores["entropy"], scores["margin"], scores["max_probability"],
        logit_gap_uncertainty, probabilities,
    ])


def fit_logistic(features: np.ndarray, labels: np.ndarray, *, l2: float = 1e-2, iterations: int = 50) -> dict:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    if np.unique(labels).size < 2:
        prevalence = np.clip(labels.mean(), 1e-6, 1 - 1e-6)
        weights = np.zeros(design.shape[1]); weights[0] = math.log(prevalence / (1 - prevalence))
        return {"mean": mean, "scale": scale, "weights": weights, "constant": True}
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        gradient = design.T @ (probabilities - labels) / len(labels) + l2 * penalty @ weights
        hessian = (design.T * variance) @ design / len(labels) + l2 * penalty + np.eye(design.shape[1]) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "weights": weights, "constant": False}


def predict_logistic(model: dict, features: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), (np.asarray(features) - model["mean"]) / model["scale"]])
    logits = np.clip(design @ model["weights"], -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


def stratified_folds(labels: np.ndarray, folds: int = 5, seed: int = 20260821) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for value in (False, True):
        indices = np.flatnonzero(labels == value)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            buckets[offset % folds].append(int(index))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def cross_validated_logistic(features: np.ndarray, labels: np.ndarray, *, folds: int = 5, seed: int = 20260821) -> tuple[np.ndarray, list[dict], dict]:
    fold_indices = stratified_folds(labels, folds=folds, seed=seed)
    predictions = np.empty(len(labels), dtype=np.float64)
    fold_rows = []
    all_indices = np.arange(len(labels))
    for fold, heldout in enumerate(fold_indices):
        train = np.setdiff1d(all_indices, heldout, assume_unique=True)
        model = fit_logistic(features[train], labels[train])
        predictions[heldout] = predict_logistic(model, features[heldout])
        fold_rows.append({
            "fold": fold, "train_examples": len(train), "evaluation_examples": len(heldout),
            "positive_train": int(np.sum(labels[train])), "positive_evaluation": int(np.sum(labels[heldout])),
        })
    return predictions, fold_rows, fit_logistic(features, labels)


def precision_at(scores: np.ndarray, labels: np.ndarray, rate: float) -> float:
    count = max(1, int(round(len(labels) * rate)))
    order = np.argsort(-np.asarray(scores), kind="stable")[:count]
    return float(np.mean(np.asarray(labels, dtype=bool)[order]))


def _bootstrap_metric(scores: np.ndarray, labels: np.ndarray, metric, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        values.append(metric(scores[indices], sampled_labels))
    return np.quantile(values, [0.025, 0.975]).tolist() if values else [None, None]


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)
    return 0.5 * kl_divergence(left, middle) + 0.5 * kl_divergence(right, middle)


def analyze_densification(
    artifact_path: str,
    *,
    output_path: str,
    probe_path: str,
    folds: int = 5,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260821,
    target_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
) -> dict:
    artifact = load_probability_artifact(artifact_path)
    dense, medium, aggressive, labels = artifact["dense"], artifact["medium"], artifact["aggressive"], artifact["labels"]
    if medium is None or aggressive is None:
        raise ValueError("sparse hierarchy artifact is incomplete")
    dense_correct = dense.argmax(axis=1) == labels
    medium_correct = medium.argmax(axis=1) == labels
    aggressive_correct = aggressive.argmax(axis=1) == labels
    targets = {
        "aggressive_to_medium": ~aggressive_correct & medium_correct,
        "aggressive_to_dense": ~aggressive_correct & dense_correct,
        "medium_to_dense": ~medium_correct & dense_correct,
    }
    features = _probability_features(aggressive)
    error_labels = ~aggressive_correct
    uplift_labels = targets["aggressive_to_dense"]
    error_oof, error_folds, error_model = cross_validated_logistic(features, error_labels, folds=folds, seed=bootstrap_seed)
    uplift_oof, uplift_folds, uplift_model = cross_validated_logistic(features, uplift_labels, folds=folds, seed=bootstrap_seed + 1)
    sparse_scores = uncertainty_scores(aggressive)
    estimator_scores = {
        "margin": sparse_scores["margin"], "entropy": sparse_scores["entropy"],
        "max_probability": sparse_scores["max_probability"], "error_probe": error_oof,
        "uplift_probe": uplift_oof, "cross_sparsity_js": js_divergence(aggressive, medium),
        "prediction_disagreement": (aggressive.argmax(axis=1) != medium.argmax(axis=1)).astype(float),
    }
    estimator_rows = []
    for name, scores in estimator_scores.items():
        row = {
            "estimator": name, "target": "aggressive_to_dense", "auroc": binary_auroc(scores, uplift_labels),
            "auprc": binary_average_precision(scores, uplift_labels),
            "auroc_ci": _bootstrap_metric(scores, uplift_labels, binary_auroc, seed=bootstrap_seed, replicates=bootstrap_replicates),
            "auprc_ci": _bootstrap_metric(scores, uplift_labels, binary_average_precision, seed=bootstrap_seed + 10, replicates=bootstrap_replicates),
        }
        for rate in (0.1, 0.2, 0.3):
            row[f"precision_at_{rate}"] = precision_at(scores, uplift_labels, rate)
        estimator_rows.append(row)
    # The error probe's native error-detection result is reported separately from
    # its uplift-target result used in the conceptual ablation.
    error_detection = {
        "auroc": binary_auroc(error_oof, error_labels),
        "auprc": binary_average_precision(error_oof, error_labels),
        "prevalence": float(error_labels.mean()),
    }
    sparse_only_rows = [
        row for row in estimator_rows
        if row["estimator"] in {"margin", "entropy", "max_probability", "uplift_probe"}
    ]
    best_sparse_only_row = max(sparse_only_rows, key=lambda row: row["auroc"])
    error_probe_row = next(row for row in estimator_rows if row["estimator"] == "error_probe")
    generic = max(row["auroc"] for row in estimator_rows if row["estimator"] in {"margin", "entropy", "max_probability", "error_probe"})
    uplift_row = next(row for row in estimator_rows if row["estimator"] == "uplift_probe")
    sufficient_events = int(uplift_labels.sum()) >= folds
    selected_estimator = max(estimator_rows, key=lambda row: row["auroc"])["estimator"]
    thresholds = {
        name: {str(float(rate)): threshold_for_rate(scores, float(rate)) for rate in target_rates}
        for name, scores in estimator_scores.items()
    }
    static_rows = []
    for name, probabilities in (("dense", dense), ("medium", medium), ("aggressive", aggressive)):
        metrics = classification_metrics(probabilities, labels)
        static_rows.append({
            "variant": name, **metrics,
            "prediction_agreement_with_dense": float(np.mean(probabilities.argmax(axis=1) == dense.argmax(axis=1))),
            "output_kl_from_dense": float(np.mean(kl_divergence(dense, probabilities))),
            "normalized_linear_flops": artifact["metadata"]["compute"][name]["normalized_linear_flops"],
        })
    static_accuracy = {row["variant"]: row["accuracy"] for row in static_rows}
    chance_accuracy = 1.0 / dense.shape[1]
    meaningful_static_hierarchy = (
        static_accuracy["dense"] > static_accuracy["aggressive"]
        and static_accuracy["aggressive"] > chance_accuracy + 0.02
    )
    monotonic_static_hierarchy = (
        static_accuracy["dense"] >= static_accuracy["medium"] >= static_accuracy["aggressive"]
    )
    gate_passed = (
        meaningful_static_hierarchy
        and sufficient_events
        and best_sparse_only_row["auroc"] >= 0.60
        and best_sparse_only_row["auroc"] > error_probe_row["auroc"]
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "udense_uplift_gate",
        "scope": {"examples": len(labels), "folds": folds, "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed, "target_rates": list(target_rates)},
        "metadata": artifact["metadata"], "static_rows": static_rows,
        "event_counts": {name: {"count": int(values.sum()), "prevalence": float(values.mean())} for name, values in targets.items()},
        "estimator_rows": estimator_rows, "error_detection": error_detection,
        "cross_validation": {"error_probe": error_folds, "uplift_probe": uplift_folds},
        "thresholds": thresholds,
        "static_gate": {
            "passed": bool(meaningful_static_hierarchy),
            "monotonic": bool(monotonic_static_hierarchy),
            "chance_accuracy": chance_accuracy,
            "criterion": "dense accuracy exceeds aggressive accuracy and aggressive accuracy exceeds chance by >0.02",
        },
        "gate": {
            "passed": bool(gate_passed), "next_stage": "two_level_heldout_arc" if gate_passed else "STOP",
            "criterion": "meaningful static hierarchy; at least one cheap sparse-only estimator or uplift probe has AUROC >=0.60 and exceeds the generic error probe; and at least one positive per fold",
            "selected_estimator": best_sparse_only_row["estimator"],
            "best_overall_estimator": selected_estimator,
            "best_sparse_only_auroc": best_sparse_only_row["auroc"],
            "uplift_probe_auroc": uplift_row["auroc"], "best_generic_auroc": generic,
            "error_probe_uplift_auroc": error_probe_row["auroc"],
            "uplift_ablation_passed": bool(uplift_row["auroc"] > generic),
        },
    }
    probe_destination = Path(probe_path)
    probe_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probe_destination,
        feature_mean=uplift_model["mean"], feature_scale=uplift_model["scale"], uplift_weights=uplift_model["weights"],
        error_feature_mean=error_model["mean"], error_feature_scale=error_model["scale"], error_weights=error_model["weights"],
        uplift_oof=uplift_oof, error_oof=error_oof, uplift_labels=uplift_labels, error_labels=error_labels,
        metadata=np.asarray(json.dumps({"features": ["entropy", "margin", "max_probability", "negative_log_top_gap", "probability_vector"], "folds": folds}, sort_keys=True), dtype=np.str_),
    )
    _write_json(Path(output_path), result)
    return result


def _rank_decisions(scores: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, min(len(scores), int(round(rate * len(scores)))))
    order = np.argsort(-np.asarray(scores), kind="stable")
    selected = np.zeros(len(scores), dtype=bool); selected[order[:count]] = True
    return selected


def benchmark_densification(
    *,
    calibration_analysis_path: str,
    probe_path: str,
    evaluation_artifact_path: str,
    output_dir: str,
    random_seeds: Sequence[int] = tuple(range(10)),
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260822,
) -> dict:
    analysis = json.loads(Path(calibration_analysis_path).read_text(encoding="utf-8"))
    if not analysis["gate"]["passed"]:
        raise ValueError("uplift gate did not authorize held-out evaluation")
    artifact = load_probability_artifact(evaluation_artifact_path)
    dense, medium, aggressive, labels = artifact["dense"], artifact["medium"], artifact["aggressive"], artifact["labels"]
    with np.load(probe_path, allow_pickle=False) as probe:
        features = _probability_features(aggressive)
        uplift_model = {"mean": probe["feature_mean"], "scale": probe["feature_scale"], "weights": probe["uplift_weights"]}
        error_model = {"mean": probe["error_feature_mean"], "scale": probe["error_feature_scale"], "weights": probe["error_weights"]}
    sparse_scores = uncertainty_scores(aggressive)
    scores = {
        "margin": sparse_scores["margin"], "entropy": sparse_scores["entropy"],
        "max_probability": sparse_scores["max_probability"],
        "error_probe": predict_logistic(error_model, features),
        "uplift_probe": predict_logistic(uplift_model, features),
    }
    aggressive_correct = aggressive.argmax(axis=1) == labels
    dense_correct = dense.argmax(axis=1) == labels
    scores["error_oracle"] = (~aggressive_correct).astype(float)
    scores["uplift_oracle"] = (~aggressive_correct & dense_correct).astype(float)
    aggressive_prediction, dense_prediction = aggressive.argmax(axis=1), dense.argmax(axis=1)
    c_sparse = analysis["metadata"]["compute"]["aggressive"]["normalized_linear_flops"]
    records, vectors = [], {}
    oracle_methods = {"error_oracle", "uplift_oracle"}
    for method, values in scores.items():
        for rate in analysis["scope"]["target_rates"]:
            if method in oracle_methods:
                decisions = _rank_decisions(values, float(rate))
                policy_source = "heldout_oracle_top_k"
            else:
                threshold = float(analysis["thresholds"][method][str(float(rate))])
                decisions = np.asarray(values) > threshold
                policy_source = "frozen_calibration_threshold"
            final_prediction = np.where(decisions, dense_prediction, aggressive_prediction)
            correct = final_prediction == labels
            key = f"{method}:{float(rate)}"
            vectors[key] = correct
            records.append({
                "method": method, "target_rate": float(rate), "dense_execution_rate": float(decisions.mean()),
                "policy_source": policy_source,
                "accuracy": float(correct.mean()), "expected_sequential_compute": float(c_sparse + decisions.mean()),
                "expected_selected_capacity": float((1 - decisions.mean()) * c_sparse + decisions.mean()),
                "densification_correction_rate": float(np.sum(decisions & ~aggressive_correct & dense_correct) / max(1, decisions.sum())),
                "wasteful_densification_rate": float(np.sum(decisions & aggressive_correct) / max(1, decisions.sum())),
                "missed_correction_rate": float(np.mean(~decisions & ~aggressive_correct & dense_correct)),
            })
    random_rows = []
    for seed in random_seeds:
        rng = np.random.default_rng(int(seed))
        random_scores = rng.random(len(labels))
        for rate in analysis["scope"]["target_rates"]:
            decisions = _rank_decisions(random_scores, float(rate))
            correct = np.where(decisions, dense_correct, aggressive_correct)
            random_rows.append({"seed": int(seed), "target_rate": float(rate), "accuracy": float(correct.mean())})
    random_summary = []
    for rate in analysis["scope"]["target_rates"]:
        values = [row["accuracy"] for row in random_rows if row["target_rate"] == float(rate)]
        random_summary.append({"target_rate": float(rate), "accuracy_mean": float(np.mean(values)), "accuracy_sd": float(np.std(values, ddof=1))})

    comparisons = []
    record_lookup = {(row["method"], row["target_rate"]): row for row in records}
    for rate in analysis["scope"]["target_rates"]:
        for baseline in ("entropy", "margin", "error_probe"):
            left = vectors[f"uplift_probe:{float(rate)}"]
            right = vectors[f"{baseline}:{float(rate)}"]
            interval = _paired_interval(left, right, seed=bootstrap_seed + int(float(rate) * 1000), replicates=bootstrap_replicates)
            uplift_rate = record_lookup[("uplift_probe", float(rate))]["dense_execution_rate"]
            baseline_rate = record_lookup[(baseline, float(rate))]["dense_execution_rate"]
            comparisons.append({
                "target_rate": float(rate), "comparison": f"uplift_probe_minus_{baseline}",
                **interval, "mcnemar": _mcnemar(left, right),
                "uplift_dense_execution_rate": uplift_rate,
                "baseline_dense_execution_rate": baseline_rate,
                "dense_execution_rate_difference": uplift_rate - baseline_rate,
            })
    positive_rows = [
        row for row in comparisons
        if row["comparison"] == "uplift_probe_minus_entropy"
        and row["difference"] > 0
        and row["bootstrap_95_ci"][0] > 0
        and row["dense_execution_rate_difference"] <= 0
    ]
    gate_passed = bool(positive_rows)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "udense_two_level_heldout_arc",
        "scope": {"examples": len(labels), "random_seeds": list(random_seeds), "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed},
        "static": {
            "dense_accuracy": float(dense_correct.mean()), "medium_accuracy": float(np.mean(medium.argmax(axis=1) == labels)),
            "aggressive_accuracy": float(aggressive_correct.mean()), "aggressive_compute": c_sparse,
        },
        "records": records, "random_rows": random_rows, "random_summary": random_summary, "comparisons": comparisons,
        "gate": {
            "passed": gate_passed, "next_stage": "three_level" if gate_passed else "STOP",
            "criterion": "uplift probe accuracy exceeds entropy with paired 95% CI excluding zero while using no more held-out dense executions at the same calibration-targeted operating point",
            "mmlu_authorized": False,
        },
    }
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "results.json", result)
    np.savez_compressed(destination / "probabilities.npz", dense=dense, medium=medium, aggressive=aggressive, labels=labels)
    return result
