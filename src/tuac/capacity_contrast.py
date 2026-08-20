"""Capacity-Contrast Distillation for adaptive sparse inference (Iteration 8)."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .cascade import uncertainty_scores
from .counterfactual_routing import (
    _extract_prompt_representations,
    _json_safe,
    _policy_row,
    _paired_policy_interval,
    _write_json,
    load_capacity_hierarchy,
    load_representation_artifact,
)
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import binary_average_precision, binary_auroc
from .quantization import transformer_blocks
from .scoring import kl_divergence
from .structured_sparsity import apply_dense_masks, js_divergence, precision_at


ITERATION7_LINEAR_AUROC = 0.5450387905504471


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def scalar_output_features(probabilities: np.ndarray) -> np.ndarray:
    """Return the four deployment-safe scalar features specified by Iteration 8."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) == 0:
        raise ValueError("probabilities must be a non-empty matrix")
    scores = uncertainty_scores(probabilities)
    ordered = np.sort(np.clip(probabilities, 1e-12, None), axis=1)
    if ordered.shape[1] < 2:
        raise ValueError("at least two choices are required")
    logit_gap = np.log(ordered[:, -1] / ordered[:, -2])
    return np.column_stack(
        [scores["entropy"], scores["margin"], scores["max_probability"], logit_gap]
    ).astype(np.float32)


def capacity_contrast_targets(s50: np.ndarray, s25: np.ndarray) -> dict[str, np.ndarray]:
    """Construct hard, JS, KL, and confidence-shift capacity targets."""

    left, right = np.asarray(s50, dtype=np.float64), np.asarray(s25, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("S50 and S25 probabilities must be same-shaped matrices")
    left_margin = np.sort(left, axis=1)[:, -1] - np.sort(left, axis=1)[:, -2]
    right_margin = np.sort(right, axis=1)[:, -1] - np.sort(right, axis=1)[:, -2]
    return {
        "hard": (left.argmax(axis=1) != right.argmax(axis=1)).astype(np.int8),
        "js": js_divergence(left, right).astype(np.float32),
        "kl": kl_divergence(left, right).astype(np.float32),
        "confidence_shift": np.abs(right_margin - left_margin).astype(np.float32),
    }


def deterministic_stratified_split(
    labels: np.ndarray, example_ids: Sequence[str], *, validation_fraction: float = 0.2, seed: int = 20260818
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic class-stratified train/validation split."""

    labels = np.asarray(labels, dtype=np.int8)
    if len(labels) != len(example_ids) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("invalid labels, IDs, or validation fraction")
    validation: list[int] = []
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}:{example_ids[int(index)]}".encode("utf-8")
            ).digest(),
        )
        count = max(1, int(round(len(ordered) * validation_fraction)))
        validation.extend(int(index) for index in ordered[:count])
    validation_array = np.asarray(sorted(validation), dtype=np.int64)
    training_array = np.setdiff1d(np.arange(len(labels)), validation_array, assume_unique=True)
    return training_array, validation_array


def load_contrast_artifact(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as values:
        return {
            "hidden_states": values["hidden_states"],
            "s50": values["s50"],
            "s25": values["s25"],
            "scalar_features": values["scalar_features"],
            "hard": values["hard"],
            "js": values["js"],
            "kl": values["kl"],
            "confidence_shift": values["confidence_shift"],
            "labels": values["labels"],
            "layer_indices": values["layer_indices"],
            "example_ids": tuple(str(value) for value in values["example_ids"]),
            "metadata": json.loads(str(values["metadata"])),
        }


def _load_pass(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        result = {"probabilities": values["probabilities"]}
        if "hidden_states" in values.files:
            result["hidden_states"] = values["hidden_states"]
        result["example_ids"] = tuple(str(value) for value in values["example_ids"])
        return result


def collect_capacity_contrast(
    *,
    data_path: str,
    masks_path: str,
    slm_model: str,
    output_path: str,
    layer_indices: Sequence[int] = (6, 13, 20, 27),
    batch_size: int = 64,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Run sequential S50 and S25 passes and save a paired CCD dataset."""

    destination = Path(output_path)
    if destination.exists():
        return load_contrast_artifact(destination)
    examples = load_jsonl(data_path)
    ids = tuple(item.id for item in examples)
    if len(set(ids)) != len(ids):
        raise ValueError("paired-capacity data contain duplicate example IDs")
    if any((item.dataset or "").lower() == "mmlu-pro" for item in examples):
        raise ValueError("MMLU-Pro examples are forbidden in CCD training data")
    labels = np.asarray([item.answer if item.answer is not None else -1 for item in examples])
    with np.load(masks_path, allow_pickle=False) as saved:
        s50_mask, s25_mask = saved["s50"], saved["s25"]
    s50_cache = destination.with_name(destination.stem + "_s50.npz")
    s25_cache = destination.with_name(destination.stem + "_s25.npz")

    s50_pass = _load_pass(s50_cache)
    s50_seconds = 0.0
    if s50_pass is None:
        scorer = HFCausalLMScorer(
            slm_model, device_map=device_map, dtype=dtype,
            trust_remote_code=trust_remote_code, batch_size=batch_size, max_batch_tokens=4096,
        )
        apply_dense_masks(transformer_blocks(scorer.model), s50_mask)
        started = time.perf_counter()
        probabilities = scorer.score(examples)
        hidden = _extract_prompt_representations(scorer, examples, layer_indices)
        s50_seconds = time.perf_counter() - started
        s50_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            s50_cache, probabilities=probabilities.astype(np.float32),
            hidden_states=hidden.astype(np.float16), example_ids=np.asarray(ids, dtype=np.str_),
        )
        s50_pass = {"probabilities": probabilities, "hidden_states": hidden, "example_ids": ids}
        del scorer
        _clear_cuda()
        print(f"S50 feature pass completed in {s50_seconds:.1f}s", flush=True)
    if s50_pass["example_ids"] != ids:
        raise ValueError("S50 cache IDs do not match the requested data")

    s25_pass = _load_pass(s25_cache)
    s25_seconds = 0.0
    if s25_pass is None:
        scorer = HFCausalLMScorer(
            slm_model, device_map=device_map, dtype=dtype,
            trust_remote_code=trust_remote_code, batch_size=batch_size, max_batch_tokens=4096,
        )
        apply_dense_masks(transformer_blocks(scorer.model), s25_mask)
        started = time.perf_counter()
        probabilities = scorer.score(examples)
        s25_seconds = time.perf_counter() - started
        np.savez_compressed(
            s25_cache, probabilities=probabilities.astype(np.float32),
            example_ids=np.asarray(ids, dtype=np.str_),
        )
        s25_pass = {"probabilities": probabilities, "example_ids": ids}
        del scorer
        _clear_cuda()
        print(f"S25 contrast pass completed in {s25_seconds:.1f}s", flush=True)
    if s25_pass["example_ids"] != ids:
        raise ValueError("S25 cache IDs do not match the requested data")

    s50, s25 = s50_pass["probabilities"], s25_pass["probabilities"]
    targets = capacity_contrast_targets(s50, s25)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "capacity_contrast_pair_collection",
        "model": slm_model,
        "examples": len(examples),
        "dataset_counts": {
            name: sum((item.dataset or "unknown") == name for item in examples)
            for name in sorted({item.dataset or "unknown" for item in examples})
        },
        "mmlu_pro_excluded": True,
        "physical_channel_pruning": True,
        "capacity_pair": ["s50", "s25"],
        "layer_indices_zero_based": list(layer_indices),
        "layer_labels": [f"L{index + 1}" for index in layer_indices],
        "representation": "S50 final prompt-token residual stream",
        "passes": {"s50_seconds": s50_seconds, "s25_seconds": s25_seconds},
        "target_prevalence": float(np.mean(targets["hard"])),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        hidden_states=np.asarray(s50_pass["hidden_states"], dtype=np.float16),
        s50=np.asarray(s50, dtype=np.float32), s25=np.asarray(s25, dtype=np.float32),
        scalar_features=scalar_output_features(s50), labels=labels.astype(np.int64),
        hard=targets["hard"], js=targets["js"], kl=targets["kl"],
        confidence_shift=targets["confidence_shift"],
        layer_indices=np.asarray(layer_indices, dtype=np.int64),
        example_ids=np.asarray(ids, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return load_contrast_artifact(destination)


def _torch_classes():
    import torch

    class LinearRouter(torch.nn.Module):
        def __init__(self, hidden_dim: int, scalar_dim: int, hidden_mean, hidden_scale, scalar_mean, scalar_scale):
            super().__init__()
            self.register_buffer("hidden_mean", hidden_mean)
            self.register_buffer("hidden_scale", hidden_scale)
            self.register_buffer("scalar_mean", scalar_mean)
            self.register_buffer("scalar_scale", scalar_scale)
            self.output = torch.nn.Linear(hidden_dim + scalar_dim, 1)

        def forward(self, hidden, scalar):
            # Iteration-7-style single-layer baseline uses L14 (or the second cached layer).
            position = min(1, hidden.shape[1] - 1)
            values = (hidden[:, position] - self.hidden_mean[position]) / self.hidden_scale[position]
            scalar = (scalar - self.scalar_mean) / self.scalar_scale
            return self.output(torch.cat([values, scalar], dim=1)).squeeze(1)

    class CCDRouter(torch.nn.Module):
        def __init__(
            self, layers: int, hidden_dim: int, scalar_dim: int,
            hidden_mean, hidden_scale, scalar_mean, scalar_scale,
            projection_dim: int = 64, dropout: float = 0.2, multitask: bool = False,
        ):
            super().__init__()
            self.register_buffer("hidden_mean", hidden_mean)
            self.register_buffer("hidden_scale", hidden_scale)
            self.register_buffer("scalar_mean", scalar_mean)
            self.register_buffer("scalar_scale", scalar_scale)
            self.projections = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_dim, projection_dim) for _ in range(layers)]
            )
            combined = layers * projection_dim + scalar_dim
            self.body = torch.nn.Sequential(
                torch.nn.Linear(combined, 128), torch.nn.GELU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 32), torch.nn.GELU(), torch.nn.Dropout(dropout),
            )
            self.hard_head = torch.nn.Linear(32, 1)
            self.soft_head = torch.nn.Linear(32, 1)
            self.multitask = multitask

        def forward(self, hidden, scalar):
            hidden = (hidden - self.hidden_mean) / self.hidden_scale
            projected = [torch.nn.functional.gelu(layer(hidden[:, i])) for i, layer in enumerate(self.projections)]
            scalar = (scalar - self.scalar_mean) / self.scalar_scale
            body = self.body(torch.cat(projected + [scalar], dim=1))
            return self.hard_head(body).squeeze(1), self.soft_head(body).squeeze(1)

    return LinearRouter, CCDRouter


def _fit_router(
    hidden: np.ndarray,
    scalar: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    regime: str,
    seed: int,
    device: str,
    max_epochs: int = 80,
    patience: int = 10,
) -> tuple[object, dict]:
    import torch

    LinearRouter, CCDRouter = _torch_classes()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_hidden = torch.as_tensor(hidden[train_indices], dtype=torch.float32)
    train_scalar = torch.as_tensor(scalar[train_indices], dtype=torch.float32)
    hidden_mean = train_hidden.mean(dim=0)
    hidden_scale = train_hidden.std(dim=0).clamp_min(1e-4)
    scalar_mean = train_scalar.mean(dim=0)
    scalar_scale = train_scalar.std(dim=0).clamp_min(1e-4)
    if regime == "linear":
        model = LinearRouter(hidden.shape[2], scalar.shape[1], hidden_mean, hidden_scale, scalar_mean, scalar_scale)
    else:
        model = CCDRouter(
            hidden.shape[1], hidden.shape[2], scalar.shape[1], hidden_mean, hidden_scale,
            scalar_mean, scalar_scale, multitask=regime == "multitask",
        )
    model.to(device)
    tensors = {
        "hidden": torch.as_tensor(hidden, dtype=torch.float32),
        "scalar": torch.as_tensor(scalar, dtype=torch.float32),
        "hard": torch.as_tensor(hard, dtype=torch.float32),
        "soft": torch.as_tensor(soft, dtype=torch.float32),
    }
    positives = max(1, int(np.sum(hard[train_indices])))
    negatives = max(1, len(train_indices) - positives)
    pos_weight = torch.tensor([negatives / positives], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = None, math.inf, 0, 0
    batch_size = min(128, len(train_indices))
    for epoch in range(max_epochs):
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            batch_tensor = torch.as_tensor(batch, dtype=torch.long)
            h = tensors["hidden"][batch_tensor].to(device)
            s = tensors["scalar"][batch_tensor].to(device)
            target_hard = tensors["hard"][batch_tensor].to(device)
            target_soft = tensors["soft"][batch_tensor].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(h, s)
            if regime == "linear":
                loss = bce(output, target_hard)
            else:
                hard_logits, soft_logits = output
                if regime == "hard":
                    loss = bce(hard_logits, target_hard)
                elif regime == "soft":
                    loss = mse(torch.sigmoid(soft_logits) * math.log(2.0), target_soft)
                else:
                    loss = bce(hard_logits, target_hard) + 5.0 * mse(
                        torch.sigmoid(soft_logits) * math.log(2.0), target_soft
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            idx = torch.as_tensor(validation_indices, dtype=torch.long)
            h = tensors["hidden"][idx].to(device)
            s = tensors["scalar"][idx].to(device)
            target_hard = tensors["hard"][idx].to(device)
            target_soft = tensors["soft"][idx].to(device)
            output = model(h, s)
            if regime == "linear":
                validation_loss = float(bce(output, target_hard).item())
            else:
                hard_logits, soft_logits = output
                if regime == "hard":
                    validation_loss = float(bce(hard_logits, target_hard).item())
                elif regime == "soft":
                    validation_loss = float(mse(torch.sigmoid(soft_logits) * math.log(2.0), target_soft).item())
                else:
                    validation_loss = float((bce(hard_logits, target_hard) + 5.0 * mse(
                        torch.sigmoid(soft_logits) * math.log(2.0), target_soft
                    )).item())
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("router training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epoch + 1, "validation_loss": best_loss}


def _predict_router(model, hidden: np.ndarray, scalar: np.ndarray, regime: str, device: str) -> np.ndarray:
    import torch

    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(hidden), 256):
            h = torch.as_tensor(hidden[start : start + 256], dtype=torch.float32, device=device)
            s = torch.as_tensor(scalar[start : start + 256], dtype=torch.float32, device=device)
            output = model(h, s)
            if regime == "linear":
                score = torch.sigmoid(output)
            elif regime == "soft":
                score = torch.sigmoid(output[1]) * math.log(2.0)
            else:
                score = torch.sigmoid(output[0])
            rows.append(score.detach().cpu().numpy())
    return np.concatenate(rows).astype(np.float64)


def _metric_row(name: str, scores: np.ndarray, target: np.ndarray) -> dict:
    return {
        "method": name,
        "auroc": binary_auroc(scores, target),
        "auprc": binary_average_precision(scores, target),
        "precision_at_10": precision_at(scores, target, 0.10),
        "precision_at_20": precision_at(scores, target, 0.20),
        "precision_at_30": precision_at(scores, target, 0.30),
    }


def analyze_capacity_contrast(
    *,
    contrast_artifact_path: str,
    arc_hierarchy_path: str,
    arc_representations_path: str,
    iteration7_probes_path: str,
    output_path: str,
    scores_output_path: str,
    models_output_path: str,
    validation_fraction: float = 0.2,
    training_seed: int = 20260818,
    target_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    random_seeds: Sequence[int] = tuple(range(10)),
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260825,
    device: str = "auto",
) -> dict:
    """Train CCD routers, enforce gates, and conditionally evaluate ARC policies."""

    import torch

    artifact = load_contrast_artifact(contrast_artifact_path)
    hidden = artifact["hidden_states"].astype(np.float32)
    scalar = artifact["scalar_features"].astype(np.float32)
    hard, soft = artifact["hard"].astype(bool), artifact["js"].astype(np.float32)
    train, validation = deterministic_stratified_split(
        hard, artifact["example_ids"], validation_fraction=validation_fraction, seed=training_seed
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    regimes = ("linear", "hard", "soft", "multitask")
    models, training_rows, validation_scores, metric_rows = {}, [], {}, []
    started = time.perf_counter()
    for offset, regime in enumerate(regimes):
        model, diagnostics = _fit_router(
            hidden, scalar, hard, soft, train, validation, regime=regime,
            seed=training_seed + offset, device=device,
        )
        scores = _predict_router(model, hidden[validation], scalar[validation], regime, device)
        models[regime] = model
        validation_scores[regime] = scores
        row = _metric_row(f"linear_instability" if regime == "linear" else f"ccd_{regime}", scores, hard[validation])
        row.update(diagnostics)
        metric_rows.append(row)
        training_rows.append({"regime": regime, **diagnostics})
        print(f"{regime}: held-out disagreement AUROC={row['auroc']:.4f}", flush=True)
    nonlinear_names = ("hard", "soft", "multitask")
    best_regime = max(nonlinear_names, key=lambda name: metric_rows[regimes.index(name)]["auroc"])
    best_row = metric_rows[regimes.index(best_regime)]
    linear_row = metric_rows[0]
    baseline_to_beat = max(float(linear_row["auroc"]), ITERATION7_LINEAR_AUROC)
    gate1_passed = bool(best_row["auroc"] >= 0.65 and best_row["auroc"] >= baseline_to_beat + 0.01)
    gate1 = {
        "passed": gate1_passed,
        "selected_regime": f"ccd_{best_regime}",
        "selected_auroc": best_row["auroc"],
        "linear_heldout_auroc": linear_row["auroc"],
        "iteration7_linear_auroc": ITERATION7_LINEAR_AUROC,
        "criterion": "held-out AUROC >= 0.65 and >= 0.01 above both the local linear baseline and Iteration 7 linear instability AUROC",
        "next_stage": "arc_advantage_gate" if gate1_passed else "STOP",
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "capacity_contrast_distillation",
        "scope": {
            "model": artifact["metadata"]["model"], "paired_examples": len(hard),
            "training_examples": len(train), "validation_examples": len(validation),
            "validation_fraction": validation_fraction, "training_seed": training_seed,
            "capacity_pair": ["s50", "s25"], "arc_role": "development only",
            "mmlu_pro_test_used": False, "router_device": device,
            "target_rates": list(target_rates), "bootstrap_replicates": bootstrap_replicates,
        },
        "paired_data": artifact["metadata"],
        "split": {
            "train_disagreement_prevalence": float(np.mean(hard[train])),
            "validation_disagreement_prevalence": float(np.mean(hard[validation])),
            "id_overlap": int(len(set(np.asarray(artifact["example_ids"])[train]) & set(np.asarray(artifact["example_ids"])[validation]))),
        },
        "contrast_metrics": metric_rows,
        "training": training_rows,
        "gate_1": gate1,
        "gate_2": {"passed": None, "status": "not_run_gate_1_failed"},
        "gate_3": {"passed": None, "status": "not_run_gate_1_failed"},
        "fresh_benchmark": {"status": "not_run", "reason": "gated on positive ARC development results"},
        "runtime": {"status": "not_run", "reason": "gated on positive MMLU-Pro replication"},
        "training_seconds": time.perf_counter() - started,
    }

    arc_scores: dict[str, np.ndarray] = {}
    signed_advantage = np.empty(0, dtype=np.int8)
    if gate1_passed:
        hierarchy = load_capacity_hierarchy(arc_hierarchy_path)
        representations = load_representation_artifact(arc_representations_path)
        if hierarchy["example_ids"] != representations["example_ids"]:
            raise ValueError("ARC hierarchy and representations have different IDs")
        arc_hidden = representations["hidden_states"].astype(np.float32)
        arc_scalar = scalar_output_features(representations["selected_probabilities"])
        for regime in nonlinear_names:
            arc_scores[f"ccd_{regime}"] = _predict_router(
                models[regime], arc_hidden, arc_scalar, regime, device
            )
        dense_pred = hierarchy["dense"].argmax(axis=1)
        sparse_pred = representations["selected_probabilities"].argmax(axis=1)
        labels = hierarchy["labels"]
        sparse_correct, dense_correct = sparse_pred == labels, dense_pred == labels
        signed_advantage = dense_correct.astype(np.int8) - sparse_correct.astype(np.int8)
        positive_advantage = signed_advantage == 1
        with np.load(iteration7_probes_path, allow_pickle=False) as prior:
            names = [str(value) for value in prior["score_names"]]
            score_matrix = prior["scores"].astype(np.float64)
        for name in ("margin", "entropy", "max_probability", "error_probe", "binary_uplift_probe", "actual_capacity_disagreement", "signed_advantage_oracle"):
            arc_scores[name] = score_matrix[names.index(name)]
        advantage_rows = [_metric_row(name, values, positive_advantage) for name, values in arc_scores.items()]
        selected_name = f"ccd_{best_regime}"
        selected_advantage = next(row for row in advantage_rows if row["method"] == selected_name)
        uncertainty_auc = max(
            next(row["auroc"] for row in advantage_rows if row["method"] == name)
            for name in ("entropy", "max_probability")
        )
        gate2_passed = bool(selected_advantage["auroc"] > 0.65 and selected_advantage["auroc"] > uncertainty_auc)
        result["advantage_metrics"] = advantage_rows
        result["arc_targets"] = {
            "examples": len(labels), "positive": int(np.sum(signed_advantage == 1)),
            "neutral": int(np.sum(signed_advantage == 0)), "harmful": int(np.sum(signed_advantage == -1)),
        }
        result["gate_2"] = {
            "passed": gate2_passed, "selected_method": selected_name,
            "selected_auroc": selected_advantage["auroc"], "best_uncertainty_auroc": uncertainty_auc,
            "criterion": "selected CCD positive-advantage AUROC > 0.65 and greater than entropy and max-probability uncertainty",
            "next_stage": "matched_compute_policy" if gate2_passed else "STOP",
        }
        if gate2_passed:
            sparse_compute = 0.5588235294117647
            policy_rows, decisions, correctness = [], {}, {}
            policy_names = list(arc_scores)
            for name in policy_names:
                for rate in target_rates:
                    row, decision, correct = _policy_row(
                        name, arc_scores[name], float(rate), signed_advantage, sparse_correct, sparse_compute
                    )
                    policy_rows.append(row)
                    decisions[(name, float(rate))] = decision
                    correctness[(name, float(rate))] = correct
            random_rows = []
            for seed in random_seeds:
                random_score = np.random.default_rng(int(seed)).random(len(labels))
                for rate in target_rates:
                    row, _, _ = _policy_row(
                        f"random_seed_{seed}", random_score, float(rate), signed_advantage,
                        sparse_correct, sparse_compute,
                    )
                    random_rows.append({"seed": int(seed), **row})
            comparisons = []
            for rate in target_rates:
                for baseline in ("entropy", "max_probability"):
                    comparison = _paired_policy_interval(
                        decisions[(selected_name, float(rate))], decisions[(baseline, float(rate))],
                        signed_advantage, sparse_correct, float(rate),
                        seed=bootstrap_seed + int(1000 * float(rate)), replicates=bootstrap_replicates,
                    )
                    comparisons.append({"target_rate": float(rate), "comparison": f"{selected_name}_minus_{baseline}", **comparison})
            gate3_points = []
            for rate in target_rates:
                selected_accuracy = next(row["accuracy"] for row in policy_rows if row["method"] == selected_name and row["target_rate"] == float(rate))
                entropy_accuracy = next(row["accuracy"] for row in policy_rows if row["method"] == "entropy" and row["target_rate"] == float(rate))
                maxp_accuracy = next(row["accuracy"] for row in policy_rows if row["method"] == "max_probability" and row["target_rate"] == float(rate))
                gate3_points.append({
                    "target_rate": float(rate), "ccd_accuracy": selected_accuracy,
                    "entropy_accuracy": entropy_accuracy, "max_probability_accuracy": maxp_accuracy,
                    "passed": bool(selected_accuracy > entropy_accuracy and selected_accuracy > maxp_accuracy),
                })
            gate3_passed = any(row["passed"] for row in gate3_points)
            result["policy_rows"], result["random_rows"], result["comparisons"] = policy_rows, random_rows, comparisons
            result["gate_3"] = {
                "passed": gate3_passed, "operating_points": gate3_points,
                "criterion": "CCD accuracy exceeds entropy and max-probability at one or more exact matched dense rates",
                "next_stage": "fresh_mmlu_pro" if gate3_passed else "STOP",
            }
            if gate3_passed:
                result["fresh_benchmark"] = {
                    "status": "not_run", "reason": "MMLU-Pro execution requires freezing and a separate held-out evaluation run",
                }

    scores_destination = Path(scores_output_path)
    scores_destination.parent.mkdir(parents=True, exist_ok=True)
    score_names = list(validation_scores) + list(arc_scores)
    np.savez_compressed(
        scores_destination,
        validation_score_names=np.asarray(list(validation_scores), dtype=np.str_),
        validation_scores=np.stack([validation_scores[name] for name in validation_scores]),
        validation_indices=validation, train_indices=train, hard_validation=hard[validation],
        arc_score_names=np.asarray(list(arc_scores), dtype=np.str_),
        arc_scores=np.stack([arc_scores[name] for name in arc_scores]) if arc_scores else np.empty((0, 0)),
        signed_advantage=signed_advantage,
        metadata=np.asarray(json.dumps({"all_gate_1_scores_held_out": True, "arc_scores_external_or_oof": True}, sort_keys=True), dtype=np.str_),
    )
    models_destination = Path(models_output_path)
    models_destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"selected_regime": best_regime, "state_dicts": {name: model.state_dict() for name, model in models.items()},
         "layer_indices": artifact["layer_indices"], "training_seed": training_seed},
        models_destination,
    )
    _write_json(Path(output_path), result)
    return _json_safe(result)
