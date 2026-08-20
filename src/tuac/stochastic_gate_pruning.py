"""Uncertainty-Preserving Gate Distillation (UPGD).

UPGD freezes all model weights and jointly learns exact-budget structured FFN
channel gates.  Stochastic straight-through top-k masks expose the optimizer to
a local posterior over compressed subnetworks, while dense predictive
distributions supervise both capability and uncertainty preservation.
"""

from __future__ import annotations

import gc
import inspect
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
from .structured_sparsity import apply_dense_masks, ffn_modules, register_channel_mask_hooks
from .uncertainty_robust_pruning import uncertainty_strata


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def hard_topk_straight_through(
    logits,
    keep: int,
    *,
    temperature: float,
    noise_scale: float = 0.0,
    generator=None,
):
    """Return an exact top-k forward gate with a sigmoid backward relaxation."""

    import torch

    if logits.ndim != 2 or not 0 < keep <= logits.shape[1]:
        raise ValueError("logits must be [layers, channels] and keep must be in range")
    if temperature <= 0 or noise_scale < 0:
        raise ValueError("temperature must be positive and noise_scale non-negative")
    scores = logits
    if noise_scale:
        uniform = torch.rand(
            logits.shape,
            dtype=logits.dtype,
            device=logits.device,
            generator=generator,
        ).clamp_(1e-6, 1 - 1e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        scores = scores + noise_scale * logistic
    top = torch.topk(scores, keep, dim=1, sorted=False).indices
    hard = torch.zeros_like(logits).scatter_(1, top, 1.0)
    threshold = torch.topk(scores, keep, dim=1, sorted=True).values[:, -1:].detach()
    soft = torch.sigmoid((scores - threshold) / temperature)
    return hard + soft - soft.detach(), hard


def uncertainty_distillation_loss(
    predictions,
    targets,
    stratum_labels,
    *,
    entropy_weight: float = 1.0,
    margin_weight: float = 0.5,
    robust_weight: float = 0.5,
    robust_temperature: float = 0.1,
):
    """Dense-distribution KL plus entropy/margin matching and stratum CVaR."""

    import torch

    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("predictions and targets must have the same [batch, classes] shape")
    if stratum_labels.shape != (predictions.shape[0],) or robust_temperature <= 0:
        raise ValueError("invalid stratum labels or robust temperature")
    tiny = torch.finfo(predictions.dtype).tiny
    predictions = predictions.clamp_min(tiny)
    targets = targets.clamp_min(tiny)
    predictions = predictions / predictions.sum(dim=1, keepdim=True)
    targets = targets / targets.sum(dim=1, keepdim=True)
    kl = torch.sum(targets * (targets.log() - predictions.log()), dim=1)
    target_entropy = -torch.sum(targets * targets.log(), dim=1)
    prediction_entropy = -torch.sum(predictions * predictions.log(), dim=1)
    target_top = torch.topk(targets, 2, dim=1).values
    prediction_top = torch.topk(predictions, 2, dim=1).values
    target_margin = target_top[:, 0] - target_top[:, 1]
    prediction_margin = prediction_top[:, 0] - prediction_top[:, 1]
    entropy_error = (prediction_entropy - target_entropy).square()
    margin_error = (prediction_margin - target_margin).square()
    per_example = kl + entropy_weight * entropy_error + margin_weight * margin_error
    stratum_means = torch.stack(
        [per_example[stratum_labels == label].mean() for label in torch.unique(stratum_labels)]
    )
    robust = robust_temperature * (
        torch.logsumexp(stratum_means / robust_temperature, dim=0)
        - math.log(len(stratum_means))
    )
    total = per_example.mean() + robust_weight * robust
    return total, {
        "loss": total.detach(),
        "kl": kl.mean().detach(),
        "entropy_error": entropy_error.mean().detach(),
        "margin_error": margin_error.mean().detach(),
        "robust": robust.detach(),
    }


def _load_training_inputs(
    data_path: str | Path,
    calibration_artifact_path: str | Path,
) -> tuple[list[MultipleChoiceExample], np.ndarray, np.ndarray, np.ndarray]:
    all_examples = load_jsonl(data_path)
    with np.load(calibration_artifact_path, allow_pickle=False) as saved:
        selected_indices = np.asarray(saved["selected_indices"], dtype=np.int64)
        probabilities = np.asarray(saved["dense_probabilities"], dtype=np.float32)
        example_ids = np.asarray(saved["example_ids"], dtype=np.str_)
        if "stratum_labels" in saved.files:
            strata = np.asarray(saved["stratum_labels"], dtype=np.int64)
        else:
            _, strata = uncertainty_strata(probabilities, strata=3)
    examples = [all_examples[int(index)] for index in selected_indices]
    if [example.id for example in examples] != example_ids.tolist():
        raise ValueError("calibration artifact IDs do not match the source examples")
    return examples, selected_indices, probabilities, strata


def _initial_gate_logits(importance: np.ndarray) -> np.ndarray:
    importance = np.asarray(importance, dtype=np.float64)
    if importance.ndim != 2 or np.any(~np.isfinite(importance)):
        raise ValueError("importance must be finite [layers, channels]")
    order = np.argsort(np.argsort(importance, axis=1, kind="stable"), axis=1, kind="stable")
    denominator = max(1, importance.shape[1] - 1)
    return (4.0 * order / denominator - 2.0).astype(np.float32)


def _stratified_batches(labels: np.ndarray, steps: int, per_stratum: int, seed: int, groups_per_batch: int | None = None):
    labels = np.asarray(labels, dtype=np.int64)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("every uncertainty stratum must contain examples")
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(group) for group in groups]
    cursors = [0] * len(groups)
    selected_group_count = len(groups) if groups_per_batch is None else max(1, min(int(groups_per_batch), len(groups)))
    for step in range(steps):
        batch = []
        # Rotate through cells when a full multi-cell batch would exceed GPU
        # memory; over the complete budget every cell is sampled equally.
        if selected_group_count < len(groups):
            start_group = (step * selected_group_count) % len(groups)
            group_indices = [(start_group + offset) % len(groups) for offset in range(selected_group_count)]
        else:
            group_indices = range(len(groups))
        for group_index in group_indices:
            group = groups[group_index]
            if cursors[group_index] + per_stratum > len(shuffled[group_index]):
                shuffled[group_index] = rng.permutation(group)
                cursors[group_index] = 0
            start = cursors[group_index]
            batch.extend(shuffled[group_index][start : start + per_stratum].tolist())
            cursors[group_index] += per_stratum
        rng.shuffle(batch)
        yield np.asarray(batch, dtype=np.int64)


def _choice_probabilities(scorer: HFCausalLMScorer, examples: Sequence[MultipleChoiceExample]):
    """Differentiable multiple-choice probabilities for one gate-training batch.

    Unlike the original pilot implementation, this path also supports
    multi-token continuations (notably HellaSwag).  The model remains frozen,
    but the forward pass is kept attached to the gate hooks so gradients reach
    the stochastic channel logits.
    """
    torch = scorer.torch
    tokenized_choices = [
        [scorer.tokenizer.encode(choice, add_special_tokens=False) for choice in example.choices]
        for example in examples
    ]
    # Gate optimization needs a memory-bounded differentiable objective.  For
    # multi-token continuations, use the first continuation token as the
    # optimization signal; final evaluation still scores the complete choice
    # likelihood through HFCausalLMScorer.score().  This keeps HellaSwag's
    # calibration examples in the mixture without exploding activation memory.
    tokenized_choices = [[choice[:1] for choice in choices] for choices in tokenized_choices]
    max_length = int(getattr(scorer.model.config, "max_position_embeddings", 2048))
    sequences = []
    first_choice_ids = []
    choices_per_example = []
    for example, choices in zip(examples, tokenized_choices):
        prompt_ids = scorer.tokenizer.encode(example.prompt, add_special_tokens=True)
        sequences.append(prompt_ids[-max_length:])
        first_choice_ids.append([choice[0] for choice in choices])
        choices_per_example.append(len(choices))
    encoded = scorer.tokenizer.pad(
        {"input_ids": sequences}, padding=True, return_tensors="pt"
    )
    encoded = {key: value.to(scorer._input_device()) for key, value in encoded.items()}
    model_inputs = dict(encoded)
    if scorer._logits_keep_parameter is not None:
        model_inputs[scorer._logits_keep_parameter] = 1
    logits = scorer.model(**model_inputs).logits[:, -1].float()
    scores = [logits[index].gather(0, torch.tensor(choice_ids, dtype=torch.long, device=logits.device)) for index, choice_ids in enumerate(first_choice_ids)]
    # Calibration artifacts use a fixed width of five classes so mixtures of
    # four-choice and five-choice tasks can share one tensor.
    choice_count = max(5, max(choices_per_example))
    padded = torch.full((len(scores), choice_count), torch.finfo(logits.dtype).min, device=logits.device)
    for index, row in enumerate(scores):
        padded[index, : row.numel()] = row
    return torch.softmax(padded, dim=1)


def train_upgd_masks(
    *,
    data_path: str,
    calibration_artifact_path: str,
    slm_model: str,
    output_path: str,
    masks_output_path: str,
    sparsity: float = 0.50,
    steps: int = 600,
    per_stratum: int = 1,
    learning_rate: float = 0.05,
    start_temperature: float = 0.5,
    end_temperature: float = 0.1,
    start_noise: float = 0.5,
    end_noise: float = 0.05,
    entropy_weight: float = 1.0,
    margin_weight: float = 0.5,
    robust_weight: float = 0.5,
    lambda_uq: float | None = None,
    robust_temperature: float = 0.1,
    objective_name: str = "full_upgd",
    seed: int = 20260829,
    groups_per_batch: int | None = None,
    device_map: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    bnb_4bit_compute_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict:
    """Train exact-budget stochastic gates with frozen LLM weights."""

    import torch

    destination = Path(output_path)
    masks_destination = Path(masks_output_path)
    if destination.exists() and masks_destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    if not 0 < sparsity < 1 or steps < 1 or per_stratum < 1:
        raise ValueError("sparsity, steps, and per_stratum are invalid")
    if lambda_uq is not None:
        if lambda_uq < 0:
            raise ValueError("lambda_uq must be non-negative")
        entropy_weight = float(lambda_uq)
        margin_weight = 0.5 * float(lambda_uq)
        robust_weight = 0.5 * float(lambda_uq)
        objective_name = f"lambda_uq_{lambda_uq:g}"
    examples, selected_indices, dense_probabilities, stratum_labels = _load_training_inputs(
        data_path, calibration_artifact_path
    )
    with np.load(calibration_artifact_path, allow_pickle=False) as saved:
        baseline_importance = np.asarray(saved["baseline_importance"], dtype=np.float32)
    scorer = HFCausalLMScorer(
        slm_model,
        device_map=device_map,
        dtype=dtype,
        quantization=quantization,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        trust_remote_code=trust_remote_code,
        batch_size=1,
        max_batch_tokens=4096,
    )
    scorer.model.config.use_cache = False
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    blocks = transformer_blocks(scorer.model)
    channels = ffn_modules(blocks[0][1])[2].in_features
    if baseline_importance.shape != (len(blocks), channels):
        raise ValueError("baseline importance does not match the model FFN shape")
    keep = max(1, int(round(channels * (1.0 - sparsity))))
    gate_device = ffn_modules(blocks[0][1])[2].weight.device
    gate_logits = torch.nn.Parameter(
        torch.tensor(_initial_gate_logits(baseline_importance), device=gate_device)
    )
    optimizer = torch.optim.Adam([gate_logits], lr=learning_rate)
    current_gates = [torch.ones(channels, device=gate_device) for _ in blocks]
    handles = []
    for layer_index, (_, block) in enumerate(blocks):
        _, _, down = ffn_modules(block)

        def hook(_module, inputs, index=layer_index):
            gate = current_gates[index].to(dtype=inputs[0].dtype, device=inputs[0].device)
            return (inputs[0] * gate.view(1, 1, -1),)

        handles.append(down.register_forward_pre_hook(hook))
    generator = torch.Generator(device=gate_device)
    generator.manual_seed(seed)
    trace = []
    initial_mask = None
    started = time.perf_counter()
    batches = _stratified_batches(stratum_labels, steps, per_stratum, seed, groups_per_batch)
    try:
        for step, batch_indices in enumerate(batches):
            fraction = step / max(1, steps - 1)
            temperature = start_temperature * (end_temperature / start_temperature) ** fraction
            noise_scale = start_noise * (end_noise / start_noise) ** fraction
            gates, hard = hard_topk_straight_through(
                gate_logits,
                keep,
                temperature=temperature,
                noise_scale=noise_scale,
                generator=generator,
            )
            if initial_mask is None:
                initial_mask = hard.detach().cpu().numpy().astype(bool)
            current_gates[:] = [gates[layer] for layer in range(len(blocks))]
            batch_examples = [examples[int(index)] for index in batch_indices]
            targets = torch.tensor(
                dense_probabilities[batch_indices], dtype=torch.float32, device=gate_device
            )
            batch_strata = torch.tensor(
                stratum_labels[batch_indices], dtype=torch.long, device=gate_device
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = _choice_probabilities(scorer, batch_examples)
            loss, components = uncertainty_distillation_loss(
                predictions,
                targets,
                batch_strata,
                entropy_weight=entropy_weight,
                margin_weight=margin_weight,
                robust_weight=robust_weight,
                robust_temperature=robust_temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([gate_logits], 5.0)
            optimizer.step()
            with torch.no_grad():
                gate_logits.sub_(gate_logits.mean(dim=1, keepdim=True)).clamp_(-8.0, 8.0)
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
                deterministic = torch.topk(gate_logits.detach(), keep, dim=1).indices
                current_mask = torch.zeros_like(gate_logits, dtype=torch.bool).scatter_(
                    1, deterministic, True
                )
                changed = float(
                    np.mean(current_mask.cpu().numpy() != initial_mask)
                )
                row = {
                    "step": step + 1,
                    "temperature": temperature,
                    "noise_scale": noise_scale,
                    "changed_gate_fraction_from_initial": changed,
                    **{name: float(value.cpu()) for name, value in components.items()},
                }
                trace.append(row)
                print(
                    f"UPGD step {step + 1}/{steps} loss={row['loss']:.5f} changed={changed:.4f}",
                    flush=True,
                )
            # Quantized backends can retain temporary dequantized tensors across
            # autograd boundaries. Drop every graph-owned reference explicitly;
            # otherwise a long gate run can OOM even when inference fits easily.
            current_gates[:] = [None] * len(current_gates)
            del predictions, loss, components, targets, batch_strata, batch_examples, gates, hard
            if (step + 1) % 25 == 0:
                _clear_cuda()
    finally:
        for handle in handles:
            handle.remove()
    final_order = torch.argsort(gate_logits.detach(), dim=1, descending=True).cpu().numpy()
    final_masks = np.sort(final_order[:, :keep], axis=1)
    initial_order = np.argsort(-_initial_gate_logits(baseline_importance), axis=1, kind="stable")
    initial_masks = np.sort(initial_order[:, :keep], axis=1)
    overlap = np.mean(
        [len(set(left).intersection(right)) / keep for left, right in zip(initial_masks, final_masks)]
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "UPGD: stochastic exact-top-k uncertainty-preserving gate distillation",
        "objective_name": objective_name,
        "model": slm_model,
        "quantization": quantization,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "data_path": str(Path(data_path).resolve()),
        "calibration_artifact": str(Path(calibration_artifact_path).resolve()),
        "examples": len(examples),
        "selected_indices": len(selected_indices),
        "sparsity": sparsity,
        "keep_channels_per_layer": keep,
        "steps": steps,
        "batch_size": int(per_stratum * (len(np.unique(stratum_labels)) if groups_per_batch is None else groups_per_batch)),
        "groups_per_batch": groups_per_batch,
        "learning_rate": learning_rate,
        "temperature_schedule": [start_temperature, end_temperature],
        "noise_schedule": [start_noise, end_noise],
        "entropy_weight": entropy_weight,
        "margin_weight": margin_weight,
        "robust_weight": robust_weight,
        "lambda_uq": lambda_uq,
        "robust_temperature": robust_temperature,
        "seed": seed,
        "retained_overlap_with_initial_baseline": float(overlap),
        "training_seconds": time.perf_counter() - started,
        "weights_frozen": True,
        "physical_export": "retained FFN channel indices",
        "claim_boundary": "training artifact only; downstream accuracy and runtime are not established",
        "calibration_choice_objective": "first-token continuation approximation for multi-token choices; final scoring uses full choice likelihood",
        "trace": trace,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    masks_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        masks_destination,
        retained=final_masks,
        baseline_retained=initial_masks,
        logits=gate_logits.detach().cpu().numpy().astype(np.float32),
        sparsity=np.asarray(sparsity),
        method=np.asarray("upgd", dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    del scorer, gate_logits, optimizer
    _clear_cuda()
    return metadata


def score_pruned_mask(
    *,
    data_path: str,
    slm_model: str,
    masks_path: str,
    mask_key: str = "retained",
    output_path: str,
    index_artifact_path: str | None = None,
    index_key: str = "validation_indices",
    batch_size: int = 64,
    device_map: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    bnb_4bit_compute_dtype: str = "float16",
    mask_mode: str = "physical",
    trust_remote_code: bool = False,
) -> dict:
    """Physically apply a learned UPGD mask and score labeled MCQ examples."""

    destination = Path(output_path)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            return json.loads(str(saved["metadata"]))
    all_examples = load_jsonl(data_path)
    if index_artifact_path:
        with np.load(index_artifact_path, allow_pickle=False) as indices_artifact:
            selected_indices = np.asarray(indices_artifact[index_key], dtype=np.int64)
    else:
        selected_indices = np.arange(len(all_examples), dtype=np.int64)
    examples = [all_examples[int(index)] for index in selected_indices]
    labels = np.asarray([example.answer if example.answer is not None else -1 for example in examples])
    if np.any(labels < 0):
        raise ValueError("scoring requires labels")
    with np.load(masks_path, allow_pickle=False) as saved:
        if mask_key not in saved.files:
            raise ValueError(f"mask key {mask_key!r} is not present in {masks_path}")
        retained = np.asarray(saved[mask_key], dtype=np.int64)
        sparsity_key = (
            "sparsity"
            if "sparsity" in saved.files
            else "aggressive_sparsity"
            if mask_key == "aggressive" and "aggressive_sparsity" in saved.files
            else "medium_sparsity"
            if mask_key == "medium" and "medium_sparsity" in saved.files
            else None
        )
        sparsity = float(saved[sparsity_key]) if sparsity_key else float("nan")
    if mask_mode not in {"physical", "hook"}:
        raise ValueError("mask_mode must be physical or hook")
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
    blocks = transformer_blocks(scorer.model)
    started = time.perf_counter()
    if mask_mode == "physical":
        apply_dense_masks(blocks, retained)
        probabilities = scorer.score(examples)
    else:
        handles = register_channel_mask_hooks(blocks, retained)
        try:
            probabilities = scorer.score(examples)
        finally:
            for handle in handles:
                handle.remove()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": slm_model,
        "data_path": str(Path(data_path).resolve()),
        "masks_path": str(Path(masks_path).resolve()),
        "mask_key": mask_key,
        "mask_mode": mask_mode,
        "quantization": quantization,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "index_artifact": str(Path(index_artifact_path).resolve()) if index_artifact_path else None,
        "index_key": index_key if index_artifact_path else None,
        "examples": len(examples),
        "sparsity": sparsity,
        "scoring_seconds": time.perf_counter() - started,
        "physical_channel_pruning": mask_mode == "physical",
        "logical_channel_mask": mask_mode == "hook",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        probabilities=probabilities.astype(np.float32),
        labels=labels.astype(np.int64),
        selected_indices=selected_indices,
        example_ids=np.asarray([example.id for example in examples], dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    del scorer
    _clear_cuda()
    return metadata


def analyze_upgd_comparison(
    *,
    reference_artifact_path: str | Path,
    baseline_artifact_path: str | Path | None = None,
    upgd_artifact_path: str | Path,
    output_path: str | Path,
    strata: int = 3,
    conformal_alpha: float = 0.1,
    conformal_fraction: float = 0.5,
    split_seed: int = 20260827,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260830,
) -> dict:
    """Evaluate UPGD against the matched mean-activation physical mask."""

    from .uncertainty_robust_pruning import (
        _paired_accuracy_interval,
        _write_json,
        lac_metrics,
        lac_threshold,
    )

    with np.load(reference_artifact_path, allow_pickle=False) as saved:
        dense = np.asarray(saved["dense"], dtype=np.float64)
        baseline = (
            np.asarray(saved["baseline_aggressive"], dtype=np.float64)
            if "baseline_aggressive" in saved.files
            else None
        )
        labels = np.asarray(saved["labels"], dtype=np.int64)
        reference_ids = np.asarray(saved["example_ids"], dtype=np.str_)
        reference_metadata = json.loads(str(saved["metadata"]))
    if baseline is None:
        if baseline_artifact_path is None:
            raise ValueError(
                "reference artifact has no baseline_aggressive array; provide baseline_artifact_path"
            )
        with np.load(baseline_artifact_path, allow_pickle=False) as saved:
            baseline = np.asarray(saved["probabilities"], dtype=np.float64)
            baseline_labels = np.asarray(saved["labels"], dtype=np.int64)
            baseline_ids = np.asarray(saved["example_ids"], dtype=np.str_)
        if not np.array_equal(reference_ids, baseline_ids) or not np.array_equal(labels, baseline_labels):
            raise ValueError("reference and baseline evaluation populations do not match")
    with np.load(upgd_artifact_path, allow_pickle=False) as saved:
        upgd = np.asarray(saved["probabilities"], dtype=np.float64)
        upgd_labels = np.asarray(saved["labels"], dtype=np.int64)
        upgd_ids = np.asarray(saved["example_ids"], dtype=np.str_)
        upgd_metadata = json.loads(str(saved["metadata"]))
    if not np.array_equal(reference_ids, upgd_ids) or not np.array_equal(labels, upgd_labels):
        raise ValueError("reference and UPGD evaluation populations do not match")
    dense_uncertainty, stratum_labels = uncertainty_strata(dense, strata=strata)
    arrays = {"dense": dense, "baseline_aggressive": baseline, "upgd_aggressive": upgd}
    metric_rows = []
    stratum_rows = []
    for name, probabilities in arrays.items():
        row = classification_metrics(probabilities, labels)
        row.update(
            {
                "condition": name,
                "mean_kl_from_dense": float(np.mean(kl_divergence(dense, probabilities))),
                "prediction_agreement_with_dense": float(
                    np.mean(dense.argmax(axis=1) == probabilities.argmax(axis=1))
                ),
                "entropy_spearman_with_dense": spearman_correlation(
                    dense_uncertainty, entropy(probabilities, normalized=True)
                ),
            }
        )
        metric_rows.append(row)
        for label in range(strata):
            selected = stratum_labels == label
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
    baseline_correct = baseline.argmax(axis=1) == labels
    upgd_correct = upgd.argmax(axis=1) == labels
    paired = _paired_accuracy_interval(
        upgd_correct,
        baseline_correct,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(labels))
    calibration_size = max(1, min(len(labels) - 1, int(round(len(labels) * conformal_fraction))))
    calibration_indices, evaluation_indices = order[:calibration_size], order[calibration_size:]
    dense_threshold = lac_threshold(
        dense[calibration_indices], labels[calibration_indices], alpha=conformal_alpha
    )
    conformal_rows = []
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
    metric_lookup = {row["condition"]: row for row in metric_rows}
    stratum_lookup = {(row["condition"], row["stratum"]): row for row in stratum_rows}
    conformal_lookup = {row["condition"]: row for row in conformal_rows}
    stratum_advantages = [
        stratum_lookup[("upgd_aggressive", label)]["accuracy"]
        - stratum_lookup[("baseline_aggressive", label)]["accuracy"]
        for label in range(strata)
    ]
    coverage_difference = (
        conformal_lookup["upgd_aggressive"]["dense_threshold_transfer"]["coverage"]
        - conformal_lookup["baseline_aggressive"]["dense_threshold_transfer"]["coverage"]
    )
    gate = {
        "passed": bool(
            paired["difference"] >= 0.02
            and paired["bootstrap_95_ci"][0] > 0
            and min(stratum_advantages) >= 0
            and metric_lookup["upgd_aggressive"]["entropy_spearman_with_dense"]
            >= metric_lookup["baseline_aggressive"]["entropy_spearman_with_dense"]
            and coverage_difference >= -0.02
        ),
        "criterion": (
            "UPGD accuracy advantage >=2 pp with paired 95% CI above zero; no dense-uncertainty "
            "stratum loses; entropy-rank preservation does not worsen; dense-threshold conformal "
            "coverage advantage >=-2 pp"
        ),
        "accuracy_comparison": paired,
        "stratum_accuracy_advantages": stratum_advantages,
        "entropy_spearman_advantage": (
            metric_lookup["upgd_aggressive"]["entropy_spearman_with_dense"]
            - metric_lookup["baseline_aggressive"]["entropy_spearman_with_dense"]
        ),
        "dense_threshold_conformal_coverage_advantage": coverage_difference,
    }
    output = {
        "title": "UPGD held-out physical-pruning evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "reference_artifact": str(Path(reference_artifact_path).resolve()),
            "baseline_artifact": (
                str(Path(baseline_artifact_path).resolve()) if baseline_artifact_path else None
            ),
            "upgd_artifact": str(Path(upgd_artifact_path).resolve()),
            "reference_metadata": reference_metadata,
            "upgd_metadata": upgd_metadata,
            "examples": len(labels),
            "strata": strata,
            "conformal_alpha": conformal_alpha,
            "conformal_calibration_examples": len(calibration_indices),
            "conformal_evaluation_examples": len(evaluation_indices),
            "bootstrap_replicates": bootstrap_replicates,
        },
        "metric_rows": metric_rows,
        "stratum_rows": stratum_rows,
        "paired_comparison": paired,
        "conformal_rows": conformal_rows,
        "gate": gate,
        "limitations": [
            "The evaluation is development/exploratory evidence, not a fresh confirmatory benchmark.",
            "UPGD was trained once with one seed and one frozen hyperparameter configuration.",
            "Conformal results use one deterministic split and establish marginal, not conditional, coverage.",
            "Physical pruning establishes parameter/FLOP proxy reduction, not kernel speed or memory gains.",
        ],
    }
    _write_json(output_path, output)
    return output
