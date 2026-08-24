"""Exact-budget structured gate training for KL and Dual-KD."""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .config import SUPPORTED_METHODS
from .data import MultipleChoiceExample, load_jsonl
from .hf import HFCausalLMScorer
from .modeling import ffn_modules, transformer_blocks
from .objectives import (
    dual_constrained_distillation_loss,
    forward_kl_distillation_loss,
    update_dual_variables,
)


def clear_cuda() -> None:
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
        scores = scores + noise_scale * (torch.log(uniform) - torch.log1p(-uniform))
    top = torch.topk(scores, keep, dim=1, sorted=False).indices
    hard = torch.zeros_like(logits).scatter_(1, top, 1.0)
    threshold = torch.topk(scores, keep, dim=1, sorted=True).values[:, -1:].detach()
    soft = torch.sigmoid((scores - threshold) / temperature)
    return hard + soft - soft.detach(), hard


def stratified_epoch_batches(
    labels: np.ndarray,
    epochs: int,
    per_stratum: int,
    seed: int,
):
    """Yield exact without-replacement epochs from teacher-entropy strata."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or epochs < 1 or per_stratum < 1:
        raise ValueError("labels, epochs, and per_stratum are invalid")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("stratum labels must be contiguous from zero")
    groups = [np.flatnonzero(labels == label) for label in unique]
    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        shuffled = [rng.permutation(group) for group in groups]
        offsets = [0] * len(groups)
        while any(offset < len(group) for offset, group in zip(offsets, shuffled)):
            batch = []
            for group_index, group in enumerate(shuffled):
                start = offsets[group_index]
                stop = min(start + per_stratum, len(group))
                batch.extend(group[start:stop].tolist())
                offsets[group_index] = stop
            rng.shuffle(batch)
            yield np.asarray(batch, dtype=np.int64)


def _choice_probabilities(
    scorer: HFCausalLMScorer,
    examples: Sequence[MultipleChoiceExample],
    *,
    choice_count: int,
    choice_microbatch_size: int | None = None,
):
    """Differentiable full-choice continuation probabilities for gate training."""

    torch = scorer.torch
    if not examples or choice_count < 2:
        raise ValueError("examples and choice_count are invalid")
    if choice_microbatch_size is not None and choice_microbatch_size < 1:
        raise ValueError("choice_microbatch_size must be positive")
    tokenized_choices = [
        [scorer.tokenizer.encode(choice, add_special_tokens=False) for choice in example.choices]
        for example in examples
    ]
    if any(not choice for choices in tokenized_choices for choice in choices):
        raise ValueError("a choice tokenized to an empty sequence")
    choices_per_example = [len(choices) for choices in tokenized_choices]
    if choice_count < max(choices_per_example):
        raise ValueError("choice_count is smaller than the batch choice count")
    max_length = int(getattr(scorer.model.config, "max_position_embeddings", 2048))
    microbatch_size = choice_microbatch_size or len(examples)
    probability_chunks = []

    if all(len(choice) == 1 for choices in tokenized_choices for choice in choices):
        for start in range(0, len(examples), microbatch_size):
            stop = min(start + microbatch_size, len(examples))
            sequences = [
                scorer.tokenizer.encode(example.prompt, add_special_tokens=True)[-max_length:]
                for example in examples[start:stop]
            ]
            encoded = scorer.tokenizer.pad(
                {"input_ids": sequences}, padding=True, return_tensors="pt"
            )
            encoded = {key: value.to(scorer._input_device()) for key, value in encoded.items()}
            model_inputs = dict(encoded)
            if scorer._logits_keep_parameter is not None:
                model_inputs[scorer._logits_keep_parameter] = 1
            logits = scorer.model(**model_inputs).logits[:, -1].float()
            padded = torch.full(
                (stop - start, choice_count),
                torch.finfo(logits.dtype).min,
                device=logits.device,
            )
            for row, choices in enumerate(tokenized_choices[start:stop]):
                choice_ids = torch.tensor(
                    [choice[0] for choice in choices],
                    dtype=torch.long,
                    device=logits.device,
                )
                padded[row, : len(choices)] = logits[row].gather(0, choice_ids)
            probability_chunks.append(torch.softmax(padded, dim=1))
        return torch.cat(probability_chunks, dim=0)

    for start in range(0, len(examples), microbatch_size):
        stop = min(start + microbatch_size, len(examples))
        sequences = []
        choice_lengths = []
        for example, choices in zip(examples[start:stop], tokenized_choices[start:stop]):
            prompt_ids = scorer.tokenizer.encode(example.prompt, add_special_tokens=True)
            for choice in choices:
                allowed_prompt = max(1, max_length - len(choice))
                sequences.append(prompt_ids[-allowed_prompt:] + choice)
                choice_lengths.append(len(choice))
        encoded = scorer.tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_tensors="pt"
        )
        encoded = {key: value.to(scorer._input_device()) for key, value in encoded.items()}
        model_inputs = dict(encoded)
        if scorer._logits_keep_parameter is not None:
            model_inputs[scorer._logits_keep_parameter] = max(choice_lengths) + 1
        log_probs = torch.log_softmax(scorer.model(**model_inputs).logits.float(), dim=-1)
        scores = []
        for row, choice_length in enumerate(choice_lengths):
            positions = log_probs[row, -choice_length - 1 : -1]
            target_ids = encoded["input_ids"][row, -choice_length:]
            token_scores = positions.gather(1, target_ids[:, None]).squeeze(1)
            scores.append(
                token_scores.mean()
                if getattr(scorer, "length_normalize", True)
                else token_scores.sum()
            )
        padded = torch.full(
            (stop - start, choice_count),
            torch.finfo(log_probs.dtype).min,
            device=log_probs.device,
        )
        cursor = 0
        for row, count in enumerate(choices_per_example[start:stop]):
            padded[row, :count] = torch.stack(scores[cursor : cursor + count])
            cursor += count
        probability_chunks.append(torch.softmax(padded, dim=1))
    return torch.cat(probability_chunks, dim=0)


def _load_training_inputs(data_path: Path, calibration_path: Path):
    all_examples = load_jsonl(data_path)
    with np.load(calibration_path, allow_pickle=False) as saved:
        required = {
            "selected_indices",
            "dense_probabilities",
            "example_ids",
            "stratum_labels",
            "baseline_importance",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"calibration artifact is missing {sorted(missing)}")
        selected = np.asarray(saved["selected_indices"], dtype=np.int64)
        dense = np.asarray(saved["dense_probabilities"], dtype=np.float32)
        ids = np.asarray(saved["example_ids"], dtype=np.str_)
        strata = np.asarray(saved["stratum_labels"], dtype=np.int64)
        importance = np.asarray(saved["baseline_importance"], dtype=np.float32)
    examples = [all_examples[int(index)] for index in selected]
    if [example.id for example in examples] != ids.tolist():
        raise ValueError("calibration IDs do not match the source examples")
    if dense.shape[0] != len(examples) or strata.shape != (len(examples),):
        raise ValueError("calibration arrays are not aligned")
    return examples, selected, dense, strata, importance


def _initial_gate_logits(importance: np.ndarray) -> np.ndarray:
    importance = np.asarray(importance, dtype=np.float64)
    if importance.ndim != 2 or np.any(~np.isfinite(importance)):
        raise ValueError("importance must be finite [layers, channels]")
    order = np.argsort(np.argsort(importance, axis=1, kind="stable"), axis=1, kind="stable")
    return (4.0 * order / max(1, importance.shape[1] - 1) - 2.0).astype(np.float32)


def train_gate_masks(
    *,
    method: str,
    data_path: str | Path,
    calibration_artifact_path: str | Path,
    model_name: str,
    output_path: str | Path,
    masks_output_path: str | Path,
    retention: float,
    epochs: int,
    per_stratum: int,
    learning_rate: float,
    dual_learning_rate: float,
    start_temperature: float,
    end_temperature: float,
    start_noise: float,
    end_noise: float,
    choice_microbatch_size: int | None,
    seed: int,
    gradient_checkpointing: bool = True,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    wandb_config: dict | None = None,
) -> dict:
    """Train exact-budget FFN gates with frozen pretrained weights."""

    import torch

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method {method!r}; choose from {SUPPORTED_METHODS}")
    if not 0.0 < retention < 1.0:
        raise ValueError("retention must lie strictly between zero and one")
    if learning_rate <= 0 or dual_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if min(start_temperature, end_temperature, start_noise, end_noise) <= 0:
        raise ValueError("temperature and noise schedule endpoints must be positive")
    destination, masks_destination = Path(output_path), Path(masks_output_path)
    if destination.exists() and masks_destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))

    examples, selected, dense, strata, importance = _load_training_inputs(
        Path(data_path), Path(calibration_artifact_path)
    )
    batches = list(stratified_epoch_batches(strata, epochs, per_stratum, seed))
    steps = len(batches)
    presented = np.concatenate(batches)
    presentation_counts = np.bincount(presented, minlength=len(examples))
    coverage = {
        "mode": "exact_without_replacement_epochs",
        "epochs": epochs,
        "unique_examples_presented": int(np.count_nonzero(presentation_counts)),
        "sample_presentations": int(len(presented)),
        "minimum_presentations_per_example": int(presentation_counts.min()),
        "maximum_presentations_per_example": int(presentation_counts.max()),
    }

    scorer = HFCausalLMScorer(model_name, device_map=device_map, dtype=dtype, batch_size=1)
    scorer.model.config.use_cache = False
    if gradient_checkpointing:
        scorer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    blocks = transformer_blocks(scorer.model)
    channels = ffn_modules(blocks[0][1])[2].in_features
    if importance.shape != (len(blocks), channels):
        raise ValueError("baseline importance does not match the model FFN shape")
    keep = max(1, int(round(channels * retention)))
    gate_device = ffn_modules(blocks[0][1])[2].weight.device
    gate_logits = torch.nn.Parameter(
        torch.tensor(_initial_gate_logits(importance), device=gate_device)
    )
    optimizer = torch.optim.Adam([gate_logits], lr=learning_rate)
    dual_variable = (
        torch.tensor(0.0, dtype=torch.float32, device=gate_device)
        if method == "dual_kl"
        else None
    )

    current_gates = [torch.ones(channels, device=gate_device) for _ in blocks]
    handles = []
    for layer_index, (_, block) in enumerate(blocks):
        _, _, down = ffn_modules(block)

        def hook(_module, inputs, index=layer_index):
            gate = current_gates[index].to(dtype=inputs[0].dtype, device=inputs[0].device)
            return (inputs[0] * gate.view(1, 1, -1),)

        handles.append(down.register_forward_pre_hook(hook))

    wandb_run = None
    if os.environ.get("WANDB_PROJECT"):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("WANDB_PROJECT is set but wandb is not installed") from exc
        wandb_run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_RUN_NAME") or None,
            group=os.environ.get("WANDB_RUN_GROUP") or None,
            job_type=os.environ.get("WANDB_JOB_TYPE", "gate-train"),
            config={
                "method": method,
                "model": model_name,
                "retention": retention,
                "epochs": epochs,
                "steps": steps,
                "per_stratum": per_stratum,
                "learning_rate": learning_rate,
                "dual_learning_rate": dual_learning_rate if method == "dual_kl" else None,
                "seed": seed,
                **(wandb_config or {}),
            },
            reinit="finish_previous",
        )

    generator = torch.Generator(device=gate_device)
    generator.manual_seed(seed)
    trace = []
    initial_mask = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
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
            targets = torch.tensor(dense[batch_indices], dtype=torch.float32, device=gate_device)
            batch_strata = torch.tensor(strata[batch_indices], dtype=torch.long, device=gate_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = _choice_probabilities(
                scorer,
                batch_examples,
                choice_count=int(dense.shape[1]),
                choice_microbatch_size=choice_microbatch_size,
            )
            entropy_bias = None
            if method == "dual_kl":
                loss, components, entropy_bias = dual_constrained_distillation_loss(
                    predictions, targets, batch_strata, dual_variable
                )
            else:
                loss, components = forward_kl_distillation_loss(predictions, targets)
            answers = torch.tensor(
                [example.answer for example in batch_examples],
                dtype=torch.long,
                device=predictions.device,
            )
            batch_accuracy = (predictions.argmax(dim=1) == answers).float().mean().detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([gate_logits], 5.0)
            optimizer.step()
            if dual_variable is not None:
                update_dual_variables(
                    dual_variable,
                    entropy_bias,
                    learning_rate=dual_learning_rate,
                )
            with torch.no_grad():
                gate_logits.sub_(gate_logits.mean(dim=1, keepdim=True)).clamp_(-8.0, 8.0)
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
                deterministic = torch.topk(gate_logits.detach(), keep, dim=1).indices
                current_mask = torch.zeros_like(gate_logits, dtype=torch.bool).scatter_(
                    1, deterministic, True
                )
                row = {
                    "step": step + 1,
                    "temperature": temperature,
                    "noise_scale": noise_scale,
                    "accuracy": float(batch_accuracy.cpu()),
                    "changed_gate_fraction_from_initial": float(
                        np.mean(current_mask.cpu().numpy() != initial_mask)
                    ),
                    **{name: float(value.cpu()) for name, value in components.items()},
                }
                if dual_variable is not None:
                    row["dual_variable"] = float(dual_variable.cpu())
                if torch.cuda.is_available():
                    row.update(
                        {
                            "cuda_memory_allocated_gib": float(torch.cuda.memory_allocated() / 1024**3),
                            "cuda_memory_reserved_gib": float(torch.cuda.memory_reserved() / 1024**3),
                            "cuda_peak_memory_allocated_gib": float(
                                torch.cuda.max_memory_allocated() / 1024**3
                            ),
                        }
                    )
                trace.append(row)
                print(
                    f"{method} step {step + 1}/{steps} loss={row['loss']:.5f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(row, step=step + 1)
            current_gates[:] = [None] * len(current_gates)
            del predictions, loss, components, targets, batch_strata, gates, hard, answers
            if entropy_bias is not None:
                del entropy_bias
            if (step + 1) % 25 == 0:
                clear_cuda()
    finally:
        for handle in handles:
            handle.remove()

    final_order = torch.argsort(gate_logits.detach(), dim=1, descending=True).cpu().numpy()
    final_masks = np.sort(final_order[:, :keep], axis=1)
    initial_order = np.argsort(-_initial_gate_logits(importance), axis=1, kind="stable")
    initial_masks = np.sort(initial_order[:, :keep], axis=1)
    overlap = np.mean(
        [len(set(left).intersection(right)) / keep for left, right in zip(initial_masks, final_masks)]
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "dual_kd_vs_kl",
        "method": method,
        "objective_name": (
            "dual_constrained_entropy_preservation"
            if method == "dual_kl"
            else "forward_kl_only"
        ),
        "model": model_name,
        "data_path": str(Path(data_path).resolve()),
        "calibration_artifact": str(Path(calibration_artifact_path).resolve()),
        "examples": len(examples),
        "selected_indices": len(selected),
        "retention_ratio": retention,
        "sparsity": 1.0 - retention,
        "keep_channels_per_layer": keep,
        "steps": steps,
        "epochs_requested": epochs,
        "batch_size": int(per_stratum * len(np.unique(strata))),
        "per_stratum": per_stratum,
        "learning_rate": learning_rate,
        "dual_learning_rate": dual_learning_rate if method == "dual_kl" else None,
        "dual_final": float(dual_variable.detach().cpu()) if dual_variable is not None else None,
        "temperature_schedule": [start_temperature, end_temperature],
        "noise_schedule": [start_noise, end_noise],
        "full_choice_objective": True,
        "choice_microbatch_size": choice_microbatch_size,
        "gradient_checkpointing": gradient_checkpointing,
        "sampling_coverage": coverage,
        "seed": seed,
        "retained_overlap_with_initial_baseline": float(overlap),
        "training_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if torch.cuda.is_available()
            else None
        ),
        "weights_frozen": True,
        "teacher_distribution_detached": True,
        "dual_constraint_detached": method == "dual_kl",
        "entropy_constraint": (
            "mean(H_student - H_teacher) = 0" if method == "dual_kl" else None
        ),
        "dual_update": (
            "lambda += eta_lambda * detached mean(H_student - H_teacher)"
            if method == "dual_kl"
            else None
        ),
        "mask_mode": "logical exact-top-k FFN channel gates",
        "trace": trace,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    masks_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        masks_destination,
        retained=final_masks,
        baseline_retained=initial_masks,
        logits=gate_logits.detach().cpu().numpy().astype(np.float32),
        sparsity=np.asarray(1.0 - retention),
        method=np.asarray(method, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "training_seconds": metadata["training_seconds"],
                "final_loss": trace[-1]["loss"] if trace else None,
                "final_kl": trace[-1]["kl"] if trace else None,
                "final_entropy_mae": trace[-1].get("entropy_mae") if trace else None,
                "final_entropy_bias": trace[-1].get("entropy_bias") if trace else None,
                "dual_final": metadata["dual_final"],
            }
        )
        wandb_run.finish()
    del scorer, gate_logits, optimizer
    clear_cuda()
    return metadata
