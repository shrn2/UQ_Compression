"""Quantization-induced delegation drift experiments for SLM-to-LLM cascades."""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import MultiBitCalibrationArtifact
from .data import load_jsonl
from .hf import HFCausalLMScorer
from .metrics import (
    binary_auroc,
    classification_metrics,
    margin_uncertainty,
    pairwise_ordering_flip_rate,
    spearman_correlation,
)
from .quantization import temporarily_mixed_quantized, transformer_blocks
from .scoring import entropy


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
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def uncertainty_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "entropy": entropy(probabilities, normalized=True),
        "max_probability": 1.0 - probabilities.max(axis=1),
        "margin": margin_uncertainty(probabilities),
    }


def evaluate_router_gate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: str | Path,
    model: str,
    threshold: float = 0.60,
) -> dict:
    errors = np.asarray(probabilities).argmax(axis=1) != np.asarray(labels)
    signals = {
        name: {
            "error_auroc": binary_auroc(values, errors),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
        }
        for name, values in uncertainty_scores(probabilities).items()
    }
    selected = max(signals, key=lambda name: signals[name]["error_auroc"])
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "example_count": int(len(labels)),
        "accuracy": float(np.mean(~errors)),
        "signals": signals,
        "selected_signal": selected,
        "gate": {
            "passed": bool(signals[selected]["error_auroc"] > threshold),
            "threshold": threshold,
            "rule": "Best calibration error-detection AUROC must be strictly greater than 0.60.",
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def score_router_gate(
    *,
    data_path: str,
    model: str,
    output_path: str,
    probability_output: str,
    threshold: float = 0.60,
    batch_size: int = 64,
    max_batch_tokens: int = 2048,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict:
    """Score a calibration split and evaluate its router-viability gate."""

    examples = load_jsonl(data_path)
    labels = np.asarray([item.answer if item.answer is not None else -1 for item in examples])
    if np.any(labels < 0):
        raise ValueError("router gate requires labels")
    scorer = HFCausalLMScorer(
        model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    probabilities = scorer.score(examples)
    destination = Path(probability_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        probabilities=probabilities.astype(np.float32),
        labels=labels,
        example_ids=np.asarray([item.id for item in examples], dtype=np.str_),
    )
    del scorer
    _clear_cuda()
    return evaluate_router_gate(
        probabilities,
        labels,
        output_path=output_path,
        model=model,
        threshold=threshold,
    )


@dataclass(frozen=True)
class CascadeProbabilityArtifact:
    calibration_labels: np.ndarray
    evaluation_labels: np.ndarray
    calibration_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    slm_fp_calibration: np.ndarray
    slm_fp_evaluation: np.ndarray
    fallback_calibration: np.ndarray
    fallback_evaluation: np.ndarray
    quantized_calibration: np.ndarray
    quantized_evaluation: np.ndarray
    bits: tuple[int, ...]
    metadata: dict

    def validate(self) -> None:
        calibration_count, classes = self.slm_fp_calibration.shape
        evaluation_count = self.slm_fp_evaluation.shape[0]
        if self.slm_fp_evaluation.shape[1] != classes:
            raise ValueError("calibration and evaluation class dimensions differ")
        if self.fallback_calibration.shape != (calibration_count, classes):
            raise ValueError("fallback calibration shape mismatch")
        if self.fallback_evaluation.shape != (evaluation_count, classes):
            raise ValueError("fallback evaluation shape mismatch")
        if self.quantized_calibration.shape != (len(self.bits), calibration_count, classes):
            raise ValueError("quantized calibration shape mismatch")
        if self.quantized_evaluation.shape != (len(self.bits), evaluation_count, classes):
            raise ValueError("quantized evaluation shape mismatch")
        if self.calibration_labels.shape != (calibration_count,) or len(self.calibration_ids) != calibration_count:
            raise ValueError("calibration labels/IDs mismatch")
        if self.evaluation_labels.shape != (evaluation_count,) or len(self.evaluation_ids) != evaluation_count:
            raise ValueError("evaluation labels/IDs mismatch")
        for values in (
            self.slm_fp_calibration,
            self.slm_fp_evaluation,
            self.fallback_calibration,
            self.fallback_evaluation,
            self.quantized_calibration,
            self.quantized_evaluation,
        ):
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("probabilities must be finite and non-negative")
            if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-5):
                raise ValueError("probability rows must sum to one")

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            calibration_labels=self.calibration_labels.astype(np.int64),
            evaluation_labels=self.evaluation_labels.astype(np.int64),
            calibration_ids=np.asarray(self.calibration_ids, dtype=np.str_),
            evaluation_ids=np.asarray(self.evaluation_ids, dtype=np.str_),
            slm_fp_calibration=self.slm_fp_calibration.astype(np.float32),
            slm_fp_evaluation=self.slm_fp_evaluation.astype(np.float32),
            fallback_calibration=self.fallback_calibration.astype(np.float32),
            fallback_evaluation=self.fallback_evaluation.astype(np.float32),
            quantized_calibration=self.quantized_calibration.astype(np.float32),
            quantized_evaluation=self.quantized_evaluation.astype(np.float32),
            bits=np.asarray(self.bits, dtype=np.int64),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CascadeProbabilityArtifact":
        with np.load(path, allow_pickle=False) as values:
            artifact = cls(
                calibration_labels=values["calibration_labels"],
                evaluation_labels=values["evaluation_labels"],
                calibration_ids=tuple(str(value) for value in values["calibration_ids"]),
                evaluation_ids=tuple(str(value) for value in values["evaluation_ids"]),
                slm_fp_calibration=values["slm_fp_calibration"],
                slm_fp_evaluation=values["slm_fp_evaluation"],
                fallback_calibration=values["fallback_calibration"],
                fallback_evaluation=values["fallback_evaluation"],
                quantized_calibration=values["quantized_calibration"],
                quantized_evaluation=values["quantized_evaluation"],
                bits=tuple(int(value) for value in values["bits"]),
                metadata=json.loads(str(values["metadata"])),
            )
        artifact.validate()
        return artifact


def _save_collection_checkpoint(path: Path, values: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **values,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def collect_cascade_probabilities(
    *,
    calibration_data_path: str,
    evaluation_data_path: str,
    slm_model: str,
    fallback_model: str,
    output_path: str,
    bits: Sequence[int] = (8, 4, 2),
    group_size: int = 128,
    batch_size: int = 128,
    fallback_batch_size: int = 64,
    max_batch_tokens: int = 4096,
    fallback_max_batch_tokens: int = 2048,
    fallback_checkpoint_size: int = 25,
    reuse_calibration_artifact: str | None = None,
    reuse_evaluation_probabilities: str | None = None,
    reuse_key: str = "teacher",
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> CascadeProbabilityArtifact:
    """Collect resumable FP, uniformly fake-quantized, and fallback probabilities."""

    calibration = load_jsonl(calibration_data_path)
    evaluation = load_jsonl(evaluation_data_path)
    calibration_labels = np.asarray([item.answer if item.answer is not None else -1 for item in calibration])
    evaluation_labels = np.asarray([item.answer if item.answer is not None else -1 for item in evaluation])
    if np.any(calibration_labels < 0) or np.any(evaluation_labels < 0):
        raise ValueError("cascade collection requires labels on both splits")
    calibration_ids = np.asarray([item.id for item in calibration], dtype=np.str_)
    evaluation_ids = np.asarray([item.id for item in evaluation], dtype=np.str_)
    bits = tuple(int(value) for value in bits)
    destination = Path(output_path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_data_path": str(Path(calibration_data_path).resolve()),
        "evaluation_data_path": str(Path(evaluation_data_path).resolve()),
        "slm_model": slm_model,
        "fallback_model": fallback_model,
        "bits": list(bits),
        "group_size": group_size,
        "batch_size": batch_size,
        "fallback_batch_size": fallback_batch_size,
        "quantization": "uniform symmetric per-row/per-group fake weight quantization over transformer blocks",
        "scope_note": "Fake quantization supports routing-drift analysis, not packed-size or runtime claims.",
    }
    values: dict[str, np.ndarray] = {
        "calibration_labels": calibration_labels,
        "evaluation_labels": evaluation_labels,
        "calibration_ids": calibration_ids,
        "evaluation_ids": evaluation_ids,
        "bits": np.asarray(bits, dtype=np.int64),
    }
    if destination.exists():
        with np.load(destination, allow_pickle=False) as checkpoint:
            previous_metadata = json.loads(str(checkpoint["metadata"]))
            required_match = ("slm_model", "fallback_model", "bits", "group_size")
            mismatches = [
                key for key in required_match
                if previous_metadata.get(key) != metadata.get(key)
            ]
            fallback_incomplete = (
                "fallback_calibration" not in checkpoint.files
                and "fallback_partial_probabilities" not in checkpoint.files
            )
            if mismatches == ["fallback_model"] and fallback_incomplete:
                previous_metadata["fallback_model"] = fallback_model
                previous_metadata["fallback_batch_size"] = fallback_batch_size
                previous_metadata["fallback_model_changed_before_scoring"] = True
            elif mismatches:
                raise ValueError("existing cascade checkpoint is incompatible with requested run")
            for key in checkpoint.files:
                if key != "metadata":
                    values[key] = checkpoint[key]
            metadata = previous_metadata

    if "slm_fp_calibration" not in values and reuse_calibration_artifact:
        cached = MultiBitCalibrationArtifact.load(reuse_calibration_artifact)
        if tuple(cached.example_ids) != tuple(calibration_ids):
            raise ValueError("reused calibration IDs do not match cascade calibration data")
        values["slm_fp_calibration"] = np.asarray(
            cached.teacher_probabilities if reuse_key == "teacher" else cached.student_probabilities
        )
        metadata["reused_calibration_artifact"] = str(Path(reuse_calibration_artifact).resolve())
    if "slm_fp_evaluation" not in values and reuse_evaluation_probabilities:
        with np.load(reuse_evaluation_probabilities, allow_pickle=False) as cached:
            if tuple(str(value) for value in cached["example_ids"]) != tuple(evaluation_ids):
                raise ValueError("reused evaluation IDs do not match cascade evaluation data")
            values["slm_fp_evaluation"] = cached[reuse_key]
        metadata["reused_evaluation_probabilities"] = str(
            Path(reuse_evaluation_probabilities).resolve()
        )
        metadata["reuse_key"] = reuse_key
    _save_collection_checkpoint(destination, values, metadata)

    quantized_keys = [f"slm_q{bit}_{split}" for bit in bits for split in ("calibration", "evaluation")]
    needs_slm = (
        "slm_fp_calibration" not in values
        or "slm_fp_evaluation" not in values
        or any(key not in values for key in quantized_keys)
    )
    if needs_slm:
        scorer = HFCausalLMScorer(
            slm_model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        blocks = transformer_blocks(scorer.model)
        combined = [*calibration, *evaluation]
        split = len(calibration)
        if "slm_fp_calibration" not in values or "slm_fp_evaluation" not in values:
            started = time.perf_counter()
            probabilities = scorer.score(combined)
            values["slm_fp_calibration"] = probabilities[:split]
            values["slm_fp_evaluation"] = probabilities[split:]
            _save_collection_checkpoint(destination, values, metadata)
            print(f"collected SLM FP in {time.perf_counter() - started:.1f}s", flush=True)
        for bit in bits:
            calibration_key = f"slm_q{bit}_calibration"
            evaluation_key = f"slm_q{bit}_evaluation"
            if calibration_key in values and evaluation_key in values:
                continue
            started = time.perf_counter()
            with temporarily_mixed_quantized(
                blocks, np.full(len(blocks), bit), group_size=group_size
            ):
                probabilities = scorer.score(combined)
            values[calibration_key] = probabilities[:split]
            values[evaluation_key] = probabilities[split:]
            _save_collection_checkpoint(destination, values, metadata)
            print(f"collected uniform Q{bit} in {time.perf_counter() - started:.1f}s", flush=True)
        metadata["slm_layers"] = len(blocks)
        del scorer
        _clear_cuda()

    if "fallback_calibration" not in values or "fallback_evaluation" not in values:
        scorer = HFCausalLMScorer(
            fallback_model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            batch_size=fallback_batch_size,
            max_batch_tokens=fallback_max_batch_tokens,
        )
        started = time.perf_counter()
        combined_examples = [*calibration, *evaluation]
        partial = (
            np.asarray(values["fallback_partial_probabilities"])
            if "fallback_partial_probabilities" in values
            else np.empty((0, max(len(item.choices) for item in combined_examples)), dtype=np.float64)
        )
        completed = int(partial.shape[0])
        if completed > len(combined_examples):
            raise ValueError("fallback checkpoint contains too many rows")
        for start in range(completed, len(combined_examples), fallback_checkpoint_size):
            end = min(start + fallback_checkpoint_size, len(combined_examples))
            current = scorer.score(combined_examples[start:end])
            partial = np.concatenate([partial, current], axis=0)
            values["fallback_partial_probabilities"] = partial
            _save_collection_checkpoint(destination, values, metadata)
            print(
                f"checkpointed fallback rows {end}/{len(combined_examples)}",
                flush=True,
            )
        probabilities = partial
        split = len(calibration)
        values["fallback_calibration"] = probabilities[:split]
        values["fallback_evaluation"] = probabilities[split:]
        values.pop("fallback_partial_probabilities", None)
        _save_collection_checkpoint(destination, values, metadata)
        print(f"collected fallback FP in {time.perf_counter() - started:.1f}s", flush=True)
        del scorer
        _clear_cuda()

    artifact = CascadeProbabilityArtifact(
        calibration_labels=calibration_labels,
        evaluation_labels=evaluation_labels,
        calibration_ids=tuple(calibration_ids),
        evaluation_ids=tuple(evaluation_ids),
        slm_fp_calibration=values["slm_fp_calibration"],
        slm_fp_evaluation=values["slm_fp_evaluation"],
        fallback_calibration=values["fallback_calibration"],
        fallback_evaluation=values["fallback_evaluation"],
        quantized_calibration=np.asarray(
            [values[f"slm_q{bit}_calibration"] for bit in bits]
        ),
        quantized_evaluation=np.asarray(
            [values[f"slm_q{bit}_evaluation"] for bit in bits]
        ),
        bits=bits,
        metadata=metadata,
    )
    artifact.save(destination)
    return artifact


def replace_cascade_fallback(
    *,
    artifact_path: str,
    calibration_data_path: str,
    evaluation_data_path: str,
    fallback_model: str,
    batch_size: int = 16,
    max_batch_tokens: int = 512,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> CascadeProbabilityArtifact:
    """Replace only the fallback probabilities in an existing cascade artifact."""

    artifact = CascadeProbabilityArtifact.load(artifact_path)
    calibration = load_jsonl(calibration_data_path)
    evaluation = load_jsonl(evaluation_data_path)
    if tuple(item.id for item in calibration) != artifact.calibration_ids:
        raise ValueError("calibration IDs do not match the cascade artifact")
    if tuple(item.id for item in evaluation) != artifact.evaluation_ids:
        raise ValueError("evaluation IDs do not match the cascade artifact")
    scorer = HFCausalLMScorer(
        fallback_model,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    started = time.perf_counter()
    probabilities = scorer.score([*calibration, *evaluation])
    split = len(calibration)
    metadata = {
        **artifact.metadata,
        "fallback_model": fallback_model,
        "fallback_batch_size": batch_size,
        "fallback_replaced_at": datetime.now(timezone.utc).isoformat(),
    }
    updated = CascadeProbabilityArtifact(
        calibration_labels=artifact.calibration_labels,
        evaluation_labels=artifact.evaluation_labels,
        calibration_ids=artifact.calibration_ids,
        evaluation_ids=artifact.evaluation_ids,
        slm_fp_calibration=artifact.slm_fp_calibration,
        slm_fp_evaluation=artifact.slm_fp_evaluation,
        fallback_calibration=probabilities[:split],
        fallback_evaluation=probabilities[split:],
        quantized_calibration=artifact.quantized_calibration,
        quantized_evaluation=artifact.quantized_evaluation,
        bits=artifact.bits,
        metadata=metadata,
    )
    updated.save(artifact_path)
    print(f"replaced fallback FP in {time.perf_counter() - started:.1f}s", flush=True)
    del scorer
    _clear_cuda()
    return updated


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    values = np.power(np.asarray(probabilities, dtype=np.float64), 1.0 / temperature)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)

    def loss(log_temperature: float) -> float:
        current = temperature_scale(probabilities, math.exp(log_temperature))
        return float(-np.log(np.clip(current[np.arange(labels.size), labels], 1e-12, 1)).mean())

    left, right = math.log(0.05), math.log(10.0)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = loss(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = loss(d)
    return float(math.exp((left + right) / 2))


def fit_isotonic(scores: np.ndarray, targets: np.ndarray) -> dict[str, list[float]]:
    """Fit an increasing weighted PAV mapping from score to error probability."""

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=targets)
    blocks: list[dict[str, float | int]] = []
    for index, (count, total) in enumerate(zip(counts, sums)):
        blocks.append({"start": index, "end": index, "weight": float(count), "sum": float(total)})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if float(left["sum"]) / float(left["weight"]) <= float(right["sum"]) / float(right["weight"]):
                break
            blocks[-2:] = [
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": float(left["weight"]) + float(right["weight"]),
                    "sum": float(left["sum"]) + float(right["sum"]),
                }
            ]
    fitted = np.empty(unique.size)
    for block in blocks:
        fitted[int(block["start"]) : int(block["end"]) + 1] = float(block["sum"]) / float(block["weight"])
    return {"x": unique.tolist(), "y": fitted.tolist()}


def predict_isotonic(model: dict[str, list[float]], scores: np.ndarray) -> np.ndarray:
    x = np.asarray(model["x"], dtype=np.float64)
    y = np.asarray(model["y"], dtype=np.float64)
    return np.interp(np.asarray(scores, dtype=np.float64), x, y, left=y[0], right=y[-1])


def top_fraction_decisions(scores: np.ndarray, rate: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    count = int(round(rate * scores.size))
    count = min(max(count, 0), scores.size)
    result = np.zeros(scores.size, dtype=bool)
    if count:
        order = np.argsort(-scores, kind="stable")
        result[order[:count]] = True
    return result


def top_fraction_decisions_with_tiebreak(
    scores: np.ndarray, tie_breaker: np.ndarray, rate: float
) -> np.ndarray:
    """Select an exact fraction by score, resolving calibration plateaus deterministically."""

    scores = np.asarray(scores, dtype=np.float64)
    tie_breaker = np.asarray(tie_breaker, dtype=np.float64)
    if tie_breaker.shape != scores.shape:
        raise ValueError("tie breaker must match scores")
    count = min(max(int(round(rate * scores.size)), 0), scores.size)
    result = np.zeros(scores.size, dtype=bool)
    if count:
        order = np.lexsort((-tie_breaker, -scores))
        result[order[:count]] = True
    return result


def threshold_for_fraction(scores: np.ndarray, rate: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    count = int(round(rate * scores.size))
    descending = np.sort(scores)[::-1]
    if count <= 0:
        threshold = float(np.nextafter(descending[0], np.inf))
    elif count >= scores.size:
        threshold = float(np.nextafter(descending[-1], -np.inf))
    else:
        threshold = float((descending[count - 1] + descending[count]) / 2)
    return {
        "threshold": threshold,
        "target_rate": rate,
        "actual_rate": float(np.mean(scores > threshold)),
    }


def _top_overlap(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    a = top_fraction_decisions(left, fraction)
    b = top_fraction_decisions(right, fraction)
    count = int(a.sum())
    return float(np.sum(a & b) / count) if count else float("nan")


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(np.sort(left) - np.sort(right))))


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    grid = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(np.sort(left), grid, side="right") / left.size
    right_cdf = np.searchsorted(np.sort(right), grid, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def cascade_metrics(
    small_probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
    labels: np.ndarray,
    delegate: np.ndarray,
) -> dict[str, float | None]:
    small_correct = small_probabilities.argmax(axis=1) == labels
    fallback_correct = fallback_probabilities.argmax(axis=1) == labels
    final_correct = np.where(delegate, fallback_correct, small_correct)
    delegated = int(np.sum(delegate))
    return {
        "cascade_accuracy": float(final_correct.mean()),
        "delegation_rate": float(np.mean(delegate)),
        "fallback_correction_rate": (
            float(np.mean((~small_correct & fallback_correct)[delegate])) if delegated else None
        ),
        "delegated_examples": delegated,
    }


def _flip_taxonomy(
    fp_delegate: np.ndarray,
    quant_delegate: np.ndarray,
    quantized_probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    quant_correct = quantized_probabilities.argmax(axis=1) == labels
    fallback_correct = fallback_probabilities.argmax(axis=1) == labels
    false_accept = fp_delegate & ~quant_delegate
    false_escalate = ~fp_delegate & quant_delegate
    harmful = false_accept & ~quant_correct & fallback_correct
    wasteful = false_escalate & quant_correct
    beneficial = false_escalate & ~quant_correct & fallback_correct
    flip = fp_delegate != quant_delegate
    benign = flip & ~(harmful | wasteful | beneficial)
    indicators = {
        "harmful_false_accept": harmful,
        "wasteful_false_escalation": wasteful,
        "beneficial_escalation": beneficial,
        "benign_or_other_flip": benign,
    }
    count = int(flip.sum())
    output: dict[str, float | int] = {
        "delegation_flip_rate": float(flip.mean()),
        "flip_count": count,
        "false_accept_rate": float(false_accept.mean()),
        "false_escalation_rate": float(false_escalate.mean()),
    }
    for name, indicator in indicators.items():
        output[f"{name}_rate"] = float(indicator.mean())
        output[f"{name}_count"] = int(indicator.sum())
        output[f"{name}_share_of_flips"] = float(indicator.sum() / count) if count else 0.0
    return output, indicators


def _mean_ci(indicator: np.ndarray, rng: np.random.Generator, replicates: int) -> list[float]:
    samples = np.empty(replicates)
    for index in range(replicates):
        selected = rng.integers(0, indicator.size, size=indicator.size)
        samples[index] = np.mean(indicator[selected])
    return np.quantile(samples, [0.025, 0.975]).tolist()


def _matched_accuracy_bootstrap(
    fp_scores: np.ndarray,
    quant_scores: np.ndarray,
    fp_probabilities: np.ndarray,
    quant_probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
    labels: np.ndarray,
    rate: float,
    *,
    replicates: int,
    seed: int,
) -> dict:
    fp_delegate = top_fraction_decisions(fp_scores, rate)
    quant_delegate = top_fraction_decisions(quant_scores, rate)
    fp_correct = np.where(
        fp_delegate,
        fallback_probabilities.argmax(axis=1) == labels,
        fp_probabilities.argmax(axis=1) == labels,
    )
    quant_correct = np.where(
        quant_delegate,
        fallback_probabilities.argmax(axis=1) == labels,
        quant_probabilities.argmax(axis=1) == labels,
    )
    observed = float(fp_correct.mean() - quant_correct.mean())
    samples = np.empty(replicates)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = rng.integers(0, labels.size, size=labels.size)
        fp_current = top_fraction_decisions(fp_scores[indices], rate)
        quant_current = top_fraction_decisions(quant_scores[indices], rate)
        fallback_correct = fallback_probabilities[indices].argmax(axis=1) == labels[indices]
        fp_small_correct = fp_probabilities[indices].argmax(axis=1) == labels[indices]
        quant_small_correct = quant_probabilities[indices].argmax(axis=1) == labels[indices]
        samples[replicate] = np.mean(np.where(fp_current, fallback_correct, fp_small_correct)) - np.mean(
            np.where(quant_current, fallback_correct, quant_small_correct)
        )
    return {
        "fp_minus_quantized_accuracy": observed,
        "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "replicates": replicates,
    }


def _routing_only_bootstrap(
    fp_scores: np.ndarray,
    quant_scores: np.ndarray,
    quant_probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
    labels: np.ndarray,
    rate: float,
    *,
    replicates: int,
    seed: int,
) -> dict:
    """Compare FP-ranked and Q-ranked routing with identical Q/LLM predictions."""

    fp_delegate = top_fraction_decisions(fp_scores, rate)
    quant_delegate = top_fraction_decisions(quant_scores, rate)
    small_correct = quant_probabilities.argmax(axis=1) == labels
    fallback_correct = fallback_probabilities.argmax(axis=1) == labels
    fp_ranked_correct = np.where(fp_delegate, fallback_correct, small_correct)
    quant_ranked_correct = np.where(quant_delegate, fallback_correct, small_correct)
    samples = np.empty(replicates)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = rng.integers(0, labels.size, size=labels.size)
        fp_current = top_fraction_decisions(fp_scores[indices], rate)
        quant_current = top_fraction_decisions(quant_scores[indices], rate)
        small_current = small_correct[indices]
        fallback_current = fallback_correct[indices]
        samples[replicate] = np.mean(np.where(fp_current, fallback_current, small_current)) - np.mean(
            np.where(quant_current, fallback_current, small_current)
        )
    return {
        "fp_ranked_minus_quantized_ranked_accuracy": float(
            fp_ranked_correct.mean() - quant_ranked_correct.mean()
        ),
        "bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "replicates": replicates,
        "mcnemar": _mcnemar(fp_ranked_correct, quant_ranked_correct),
    }


def _mcnemar(left_correct: np.ndarray, right_correct: np.ndarray) -> dict:
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    total = left_only + right_only
    if total == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(total, value) for value in range(min(left_only, right_only) + 1)) / (2**total)
        p_value = min(1.0, 2 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "exact_p": p_value}


def analyze_cascade(
    artifact: CascadeProbabilityArtifact,
    *,
    output_path: str | Path,
    target_rates: Sequence[float] = (0.10, 0.20, 0.30, 0.50),
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260817,
) -> dict:
    """Freeze calibration policies and evaluate delegation drift on held-out examples."""

    artifact.validate()
    target_rates = tuple(float(value) for value in target_rates)
    calibration_errors = artifact.slm_fp_calibration.argmax(axis=1) != artifact.calibration_labels
    signal_values = uncertainty_scores(artifact.slm_fp_calibration)
    signal_metrics = {
        name: binary_auroc(values, calibration_errors) for name, values in signal_values.items()
    }
    selected_signal = max(signal_metrics, key=signal_metrics.get)
    gate_passed = bool(signal_metrics[selected_signal] > 0.60)
    if not gate_passed:
        raise ValueError("FP router viability gate failed; held-out cascade analysis is not warranted")
    fp_calibration_scores = signal_values[selected_signal]
    fp_evaluation_scores = uncertainty_scores(artifact.slm_fp_evaluation)[selected_signal]
    fp_thresholds = {
        str(rate): threshold_for_fraction(fp_calibration_scores, rate) for rate in target_rates
    }

    calibration_variants = {"fp": artifact.slm_fp_calibration}
    evaluation_variants = {"fp": artifact.slm_fp_evaluation}
    for index, bit in enumerate(artifact.bits):
        calibration_variants[f"q{bit}"] = artifact.quantized_calibration[index]
        evaluation_variants[f"q{bit}"] = artifact.quantized_evaluation[index]
    calibration_models: dict[str, dict] = {}
    variant_scores: dict[str, dict[str, np.ndarray]] = {}
    for precision, probabilities in calibration_variants.items():
        raw = uncertainty_scores(probabilities)[selected_signal]
        temperature = fit_temperature(probabilities, artifact.calibration_labels)
        temperature_probabilities = temperature_scale(probabilities, temperature)
        temperature_scores = uncertainty_scores(temperature_probabilities)[selected_signal]
        errors = probabilities.argmax(axis=1) != artifact.calibration_labels
        isotonic = fit_isotonic(raw, errors)
        risk = predict_isotonic(isotonic, raw)
        calibration_models[precision] = {
            "temperature": temperature,
            "isotonic": isotonic,
            "raw_thresholds": {str(rate): threshold_for_fraction(raw, rate) for rate in target_rates},
            "temperature_thresholds": {
                str(rate): threshold_for_fraction(temperature_scores, rate) for rate in target_rates
            },
            "isotonic_thresholds": {
                str(rate): threshold_for_fraction(risk, rate) for rate in target_rates
            },
        }
        evaluation_probabilities = evaluation_variants[precision]
        raw_evaluation = uncertainty_scores(evaluation_probabilities)[selected_signal]
        variant_scores[precision] = {
            "raw_calibration": raw,
            "raw_evaluation": raw_evaluation,
            "temperature_evaluation": uncertainty_scores(
                temperature_scale(evaluation_probabilities, temperature)
            )[selected_signal],
            "isotonic_evaluation": predict_isotonic(isotonic, raw_evaluation),
        }

    small_model_rows = []
    drift_rows = []
    routing_rows = []
    flip_rows = []
    proximity_rows = []
    preserved_rows = []
    comparisons = {}
    routing_only_comparisons = {}
    regret_rows = []
    labels = artifact.evaluation_labels
    fallback = artifact.fallback_evaluation
    fp_probabilities = artifact.slm_fp_evaluation
    fp_predictions = fp_probabilities.argmax(axis=1)
    fp_scores = variant_scores["fp"]["raw_evaluation"]
    rng = np.random.default_rng(bootstrap_seed)

    for precision, probabilities in evaluation_variants.items():
        metrics = classification_metrics(probabilities, labels)
        small_model_rows.append({"precision": precision, **metrics})
        scores = variant_scores[precision]["raw_evaluation"]
        drift_rows.append(
            {
                "precision": precision,
                "accuracy": metrics["accuracy"],
                "prediction_agreement_with_fp": float(np.mean(probabilities.argmax(axis=1) == fp_predictions)),
                "uncertainty_spearman": spearman_correlation(fp_scores, scores),
                "pairwise_ordering_inversion": pairwise_ordering_flip_rate(fp_scores, scores),
                "mean_uncertainty": float(scores.mean()),
                "fp_mean_uncertainty": float(fp_scores.mean()),
                "mean_shift": float(scores.mean() - fp_scores.mean()),
                "variance": float(scores.var()),
                "fp_variance": float(fp_scores.var()),
                "variance_ratio": float(scores.var() / fp_scores.var()) if fp_scores.var() else None,
                "wasserstein": _wasserstein_1d(fp_scores, scores),
                "ks_statistic": _ks_statistic(fp_scores, scores),
                "top10_overlap": _top_overlap(fp_scores, scores, 0.10),
                "top20_overlap": _top_overlap(fp_scores, scores, 0.20),
                "top30_overlap": _top_overlap(fp_scores, scores, 0.30),
            }
        )

    for rate_index, rate in enumerate(target_rates):
        fp_frozen = fp_scores > fp_thresholds[str(rate)]["threshold"]
        fp_matched = top_fraction_decisions(fp_scores, rate)
        fp_matched_metrics = cascade_metrics(fp_probabilities, fallback, labels, fp_matched)
        routing_rows.append(
            {
                "precision": "fp",
                "method": "matched_rate",
                "target_rate": rate,
                **fp_matched_metrics,
            }
        )
        fp_frozen_metrics = cascade_metrics(fp_probabilities, fallback, labels, fp_frozen)
        routing_rows.append(
            {
                "precision": "fp",
                "method": "frozen_fp_threshold",
                "target_rate": rate,
                **fp_frozen_metrics,
            }
        )
        for random_seed in range(10):
            random_scores = np.random.default_rng(random_seed).random(labels.size)
            decisions = top_fraction_decisions(random_scores, rate)
            routing_rows.append(
                {
                    "precision": "fp",
                    "method": "random",
                    "target_rate": rate,
                    "random_seed": random_seed,
                    **cascade_metrics(fp_probabilities, fallback, labels, decisions),
                }
            )
        small_correct = fp_probabilities.argmax(axis=1) == labels
        fallback_correct = fallback.argmax(axis=1) == labels
        oracle_scores = fallback_correct.astype(int) - small_correct.astype(int)
        oracle = top_fraction_decisions(oracle_scores, rate)
        routing_rows.append(
            {
                "precision": "fp",
                "method": "cascade_oracle",
                "target_rate": rate,
                **cascade_metrics(fp_probabilities, fallback, labels, oracle),
            }
        )

        for bit_index, bit in enumerate(artifact.bits):
            precision = f"q{bit}"
            probabilities = evaluation_variants[precision]
            scores = variant_scores[precision]
            methods = {
                "frozen_fp_threshold": scores["raw_evaluation"] > fp_thresholds[str(rate)]["threshold"],
                "retuned_threshold": scores["raw_evaluation"]
                > calibration_models[precision]["raw_thresholds"][str(rate)]["threshold"],
                "temperature": scores["temperature_evaluation"]
                > calibration_models[precision]["temperature_thresholds"][str(rate)]["threshold"],
                "isotonic": top_fraction_decisions_with_tiebreak(
                    scores["isotonic_evaluation"], scores["raw_evaluation"], rate
                ),
                "matched_rate": top_fraction_decisions(scores["raw_evaluation"], rate),
            }
            for method, decisions in methods.items():
                routing_rows.append(
                    {
                        "precision": precision,
                        "method": method,
                        "target_rate": rate,
                        **cascade_metrics(probabilities, fallback, labels, decisions),
                    }
                )
            for random_seed in range(10):
                random_scores = np.random.default_rng(random_seed).random(labels.size)
                decisions = top_fraction_decisions(random_scores, rate)
                routing_rows.append(
                    {
                        "precision": precision,
                        "method": "random",
                        "target_rate": rate,
                        "random_seed": random_seed,
                        **cascade_metrics(probabilities, fallback, labels, decisions),
                    }
                )
            quant_correct = probabilities.argmax(axis=1) == labels
            fallback_correct = fallback.argmax(axis=1) == labels
            oracle_scores = fallback_correct.astype(int) - quant_correct.astype(int)
            routing_rows.append(
                {
                    "precision": precision,
                    "method": "cascade_oracle",
                    "target_rate": rate,
                    **cascade_metrics(
                        probabilities,
                        fallback,
                        labels,
                        top_fraction_decisions(oracle_scores, rate),
                    ),
                }
            )
            frozen = methods["frozen_fp_threshold"]
            taxonomy, indicators = _flip_taxonomy(
                fp_frozen, frozen, probabilities, fallback, labels
            )
            flip_ci = _mean_ci(fp_frozen != frozen, rng, bootstrap_replicates)
            row = {
                "precision": precision,
                "target_rate": rate,
                "delegation_flip_ci_low": flip_ci[0],
                "delegation_flip_ci_high": flip_ci[1],
                **taxonomy,
            }
            for name, indicator in indicators.items():
                interval = _mean_ci(indicator, rng, bootstrap_replicates)
                row[f"{name}_ci_low"] = interval[0]
                row[f"{name}_ci_high"] = interval[1]
            flip_rows.append(row)
            same_prediction = probabilities.argmax(axis=1) == fp_predictions
            preserved_rows.append(
                {
                    "precision": precision,
                    "target_rate": rate,
                    "method": "frozen_fp_threshold",
                    "same_prediction_examples": int(same_prediction.sum()),
                    "delegation_flip_rate": float(np.mean((fp_frozen != frozen)[same_prediction])),
                }
            )
            matched = methods["matched_rate"]
            preserved_rows.append(
                {
                    "precision": precision,
                    "target_rate": rate,
                    "method": "matched_rate",
                    "same_prediction_examples": int(same_prediction.sum()),
                    "delegation_flip_rate": float(np.mean((fp_matched != matched)[same_prediction])),
                }
            )
            distance = np.abs(fp_scores - fp_thresholds[str(rate)]["threshold"])
            order = np.argsort(distance, kind="stable")
            for bin_index, indices in enumerate(np.array_split(order, 5), start=1):
                proximity_rows.append(
                    {
                        "precision": precision,
                        "target_rate": rate,
                        "distance_quintile": bin_index,
                        "examples": int(indices.size),
                        "mean_absolute_distance": float(distance[indices].mean()),
                        "delegation_flip_rate": float(np.mean((fp_frozen != frozen)[indices])),
                        "harmful_false_accept_rate": float(np.mean(indicators["harmful_false_accept"][indices])),
                    }
                )
            comparison_key = f"{precision}:{rate}"
            comparisons[comparison_key] = _matched_accuracy_bootstrap(
                fp_scores,
                scores["raw_evaluation"],
                fp_probabilities,
                probabilities,
                fallback,
                labels,
                rate,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + rate_index * 100 + bit_index,
            )
            fp_final = np.where(fp_matched, fallback.argmax(axis=1), fp_predictions) == labels
            quant_final = np.where(matched, fallback.argmax(axis=1), probabilities.argmax(axis=1)) == labels
            comparisons[comparison_key]["mcnemar"] = _mcnemar(fp_final, quant_final)
            routing_only_comparisons[comparison_key] = _routing_only_bootstrap(
                fp_scores,
                scores["raw_evaluation"],
                probabilities,
                fallback,
                labels,
                rate,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 500 + rate_index * 100 + bit_index,
            )
            fp_metrics = cascade_metrics(fp_probabilities, fallback, labels, fp_matched)
            quant_metrics = cascade_metrics(probabilities, fallback, labels, matched)
            for cost in (0.0, 0.02, 0.05, 0.10, 0.20):
                fp_utility = float(fp_metrics["cascade_accuracy"]) - cost * float(fp_metrics["delegation_rate"])
                quant_utility = float(quant_metrics["cascade_accuracy"]) - cost * float(quant_metrics["delegation_rate"])
                regret_rows.append(
                    {
                        "precision": precision,
                        "target_rate": rate,
                        "fallback_cost": cost,
                        "fp_utility": fp_utility,
                        "quantized_utility": quant_utility,
                        "routing_regret": fp_utility - quant_utility,
                    }
                )

    always_rows = [
        {
            "precision": precision,
            "method": "always_small",
            "target_rate": 0.0,
            **cascade_metrics(probabilities, fallback, labels, np.zeros(labels.size, dtype=bool)),
        }
        for precision, probabilities in evaluation_variants.items()
    ]
    always_rows.append(
        {
            "precision": "fallback",
            "method": "always_large",
            "target_rate": 1.0,
            **cascade_metrics(fp_probabilities, fallback, labels, np.ones(labels.size, dtype=bool)),
        }
    )
    routing_rows.extend(always_rows)

    quant_drift = {row["precision"]: row for row in drift_rows if row["precision"] != "fp"}
    nontrivial = any(row["delegation_flip_rate"] >= 0.05 for row in flip_rows)
    harmful = any(row["harmful_false_accept_rate"] >= 0.01 for row in flip_rows)
    rank_damage = any(
        quant_drift[name]["uncertainty_spearman"] < (0.90 if name == "q2" else 0.95)
        for name in quant_drift
    )
    significant_matched_loss = any(
        value["bootstrap_95_ci"][0] > 0 for value in comparisons.values()
    )
    significant_routing_only_loss = any(
        value["bootstrap_95_ci"][0] > 0 for value in routing_only_comparisons.values()
    )
    fallback_gap = (
        classification_metrics(fallback, labels)["accuracy"]
        - classification_metrics(fp_probabilities, labels)["accuracy"]
    )
    frozen_losses = []
    for row in routing_rows:
        if row["precision"].startswith("q") and row["method"] == "frozen_fp_threshold":
            fp_row = next(
                item for item in routing_rows
                if item["precision"] == "fp"
                and item["method"] == "frozen_fp_threshold"
                and item["target_rate"] == row["target_rate"]
            )
            frozen_losses.append(float(fp_row["cascade_accuracy"]) - float(row["cascade_accuracy"]))
    if (
        fallback_gap > 0
        and rank_damage
        and nontrivial
        and harmful
        and significant_routing_only_loss
    ):
        decision = "STRONG_POSITIVE"
    elif fallback_gap > 0 and nontrivial and max(frozen_losses, default=0.0) >= 0.01:
        decision = "MODERATE_POSITIVE"
    else:
        decision = "NEGATIVE"
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": artifact.metadata,
        "scope": {
            "calibration_examples": len(artifact.calibration_labels),
            "evaluation_examples": len(artifact.evaluation_labels),
            "bits": list(artifact.bits),
            "target_rates": list(target_rates),
            "random_seeds": list(range(10)),
        },
        "router": {
            "signal_aurocs": signal_metrics,
            "selected_signal": selected_signal,
            "gate_passed": gate_passed,
            "gate_threshold": 0.60,
            "fp_thresholds": fp_thresholds,
            "calibration_models": calibration_models,
        },
        "reference_metrics": {
            "slm_fp_calibration": classification_metrics(
                artifact.slm_fp_calibration, artifact.calibration_labels
            ),
            "slm_fp_evaluation": classification_metrics(fp_probabilities, labels),
            "fallback_calibration": classification_metrics(
                artifact.fallback_calibration, artifact.calibration_labels
            ),
            "fallback_evaluation": classification_metrics(fallback, labels),
        },
        "small_model_rows": small_model_rows,
        "drift_rows": drift_rows,
        "routing_rows": routing_rows,
        "flip_rows": flip_rows,
        "proximity_rows": proximity_rows,
        "accuracy_preserved_rows": preserved_rows,
        "regret_rows": regret_rows,
        "matched_rate_comparisons": comparisons,
        "routing_only_comparisons": routing_only_comparisons,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "decision": {
            "label": decision,
            "criteria": {
                "ranking_damage": rank_damage,
                "nontrivial_delegation_flip": nontrivial,
                "meaningful_harmful_false_accepts": harmful,
                "significant_matched_rate_loss": significant_matched_loss,
                "significant_routing_only_loss": significant_routing_only_loss,
                "fallback_accuracy_gap_positive": fallback_gap > 0,
                "fallback_minus_slm_fp_accuracy": fallback_gap,
            },
            "mmlu_pro_scale_up_warranted": decision in {"STRONG_POSITIVE", "MODERATE_POSITIVE"},
            "real_quantization_warranted": decision == "STRONG_POSITIVE",
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = _json_safe(output)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def analyze_cascade_drift_checkpoint(
    *,
    checkpoint_path: str,
    output_path: str,
    target_rates: Sequence[float] = (0.10, 0.20, 0.30, 0.50),
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260818,
) -> dict:
    """Analyze SLM and routing-decision drift when fallback scoring is unavailable."""

    with np.load(checkpoint_path, allow_pickle=False) as values:
        required = {
            "calibration_labels", "evaluation_labels", "bits",
            "slm_fp_calibration", "slm_fp_evaluation",
        }
        if not required.issubset(values.files):
            raise ValueError("incomplete checkpoint lacks required SLM arrays")
        calibration_labels = values["calibration_labels"]
        evaluation_labels = values["evaluation_labels"]
        bits = tuple(int(value) for value in values["bits"])
        calibration_variants = {"fp": values["slm_fp_calibration"]}
        evaluation_variants = {"fp": values["slm_fp_evaluation"]}
        for bit in bits:
            calibration_variants[f"q{bit}"] = values[f"slm_q{bit}_calibration"]
            evaluation_variants[f"q{bit}"] = values[f"slm_q{bit}_evaluation"]
        metadata = json.loads(str(values["metadata"]))
    fp_errors = calibration_variants["fp"].argmax(axis=1) != calibration_labels
    signal_aurocs = {
        name: binary_auroc(scores, fp_errors)
        for name, scores in uncertainty_scores(calibration_variants["fp"]).items()
    }
    selected = max(signal_aurocs, key=signal_aurocs.get)
    gate_passed = bool(signal_aurocs[selected] > 0.60)
    if not gate_passed:
        raise ValueError("FP router viability gate failed")
    target_rates = tuple(float(value) for value in target_rates)
    fp_cal_scores = uncertainty_scores(calibration_variants["fp"])[selected]
    fp_eval_scores = uncertainty_scores(evaluation_variants["fp"])[selected]
    fp_thresholds = {
        str(rate): threshold_for_fraction(fp_cal_scores, rate) for rate in target_rates
    }
    rng = np.random.default_rng(bootstrap_seed)
    model_rows = []
    drift_rows = []
    decision_rows = []
    preserved_rows = []
    for precision, probabilities in evaluation_variants.items():
        model_rows.append({"precision": precision, **classification_metrics(probabilities, evaluation_labels)})
        scores = uncertainty_scores(probabilities)[selected]
        rho_samples = np.empty(bootstrap_replicates)
        for replicate in range(bootstrap_replicates):
            indices = rng.integers(0, evaluation_labels.size, size=evaluation_labels.size)
            rho_samples[replicate] = spearman_correlation(fp_eval_scores[indices], scores[indices])
        drift_rows.append(
            {
                "precision": precision,
                "accuracy": float(np.mean(probabilities.argmax(axis=1) == evaluation_labels)),
                "prediction_agreement_with_fp": float(
                    np.mean(probabilities.argmax(axis=1) == evaluation_variants["fp"].argmax(axis=1))
                ),
                "uncertainty_spearman": spearman_correlation(fp_eval_scores, scores),
                "uncertainty_spearman_ci_low": float(np.quantile(rho_samples, 0.025)),
                "uncertainty_spearman_ci_high": float(np.quantile(rho_samples, 0.975)),
                "pairwise_ordering_inversion": pairwise_ordering_flip_rate(fp_eval_scores, scores),
                "mean_shift": float(scores.mean() - fp_eval_scores.mean()),
                "variance_ratio": float(scores.var() / fp_eval_scores.var()),
                "wasserstein": _wasserstein_1d(fp_eval_scores, scores),
                "ks_statistic": _ks_statistic(fp_eval_scores, scores),
                "top10_overlap": _top_overlap(fp_eval_scores, scores, 0.10),
                "top20_overlap": _top_overlap(fp_eval_scores, scores, 0.20),
                "top30_overlap": _top_overlap(fp_eval_scores, scores, 0.30),
            }
        )
        if precision == "fp":
            continue
        calibration_scores = uncertainty_scores(calibration_variants[precision])[selected]
        same_prediction = probabilities.argmax(axis=1) == evaluation_variants["fp"].argmax(axis=1)
        for rate in target_rates:
            fp_frozen = fp_eval_scores > fp_thresholds[str(rate)]["threshold"]
            quant_frozen = scores > fp_thresholds[str(rate)]["threshold"]
            fp_matched = top_fraction_decisions(fp_eval_scores, rate)
            quant_matched = top_fraction_decisions(scores, rate)
            retuned = scores > threshold_for_fraction(calibration_scores, rate)["threshold"]
            for method, left, right in (
                ("frozen_fp_threshold", fp_frozen, quant_frozen),
                ("retuned_threshold", fp_frozen, retuned),
                ("matched_rate", fp_matched, quant_matched),
            ):
                flip = left != right
                interval = _mean_ci(flip, rng, bootstrap_replicates)
                decision_rows.append(
                    {
                        "precision": precision,
                        "target_rate": rate,
                        "method": method,
                        "fp_delegation_rate": float(left.mean()),
                        "quantized_delegation_rate": float(right.mean()),
                        "delegation_rate_shift": float(right.mean() - left.mean()),
                        "delegation_flip_rate": float(flip.mean()),
                        "delegation_flip_ci_low": interval[0],
                        "delegation_flip_ci_high": interval[1],
                    }
                )
            preserved_rows.append(
                {
                    "precision": precision,
                    "target_rate": rate,
                    "same_prediction_examples": int(same_prediction.sum()),
                    "frozen_delegation_flip_rate": float(
                        np.mean((fp_frozen != quant_frozen)[same_prediction])
                    ),
                    "matched_delegation_flip_rate": float(
                        np.mean((fp_matched != quant_matched)[same_prediction])
                    ),
                }
            )
    output = _json_safe(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "partial_fallback_unavailable",
            "metadata": metadata,
            "scope": {
                "calibration_examples": len(calibration_labels),
                "evaluation_examples": len(evaluation_labels),
                "bits": list(bits),
                "target_rates": list(target_rates),
            },
            "router": {
                "signal_aurocs": signal_aurocs,
                "selected_signal": selected,
                "gate_passed": gate_passed,
                "fp_thresholds": fp_thresholds,
            },
            "small_model_rows": model_rows,
            "drift_rows": drift_rows,
            "decision_rows": decision_rows,
            "accuracy_preserved_rows": preserved_rows,
            "omitted": {
                "cascade_metrics": "7B CPU/disk-offloaded fallback did not finish within the bounded execution window; 3B retry also paged excessively.",
                "harmful_wasteful_taxonomy": "Requires fallback correctness on every held-out example.",
                "routing_regret": "Requires completed fallback outputs.",
            },
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        }
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output
