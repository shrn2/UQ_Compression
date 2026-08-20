"""Teacher-aware example weighting and layer-sensitivity calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ScoreMode = Literal[
    "student_only",
    "teacher_confidence",
    "teacher_uncertainty",
    "disagreement",
    "combined",
]


@dataclass(frozen=True)
class ImportanceResult:
    """Intermediate signals and final layer importance values."""

    example_weights: np.ndarray
    teacher_entropy: np.ndarray
    teacher_confidence: np.ndarray
    disagreement: np.ndarray
    layer_importance: np.ndarray


def _validate_probabilities(probabilities: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [examples, classes] with at least 2 classes")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"each row of {name} must have positive mass")
    return values / totals


def entropy(probabilities: np.ndarray, *, normalized: bool = False) -> np.ndarray:
    """Compute categorical entropy for each row."""

    probs = _validate_probabilities(probabilities, "probabilities")
    safe = np.clip(probs, np.finfo(np.float64).tiny, 1.0)
    result = -np.sum(probs * np.log(safe), axis=1)
    if normalized:
        # Exact zeros are padding for mixed datasets with different choice counts.
        support = np.maximum(np.count_nonzero(probs, axis=1), 2)
        result = result / np.log(support)
    return result


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute row-wise KL(p || q), accepting unnormalized positive scores."""

    left = _validate_probabilities(p, "p")
    right = _validate_probabilities(q, "q")
    if left.shape != right.shape:
        raise ValueError("p and q must have the same shape")
    tiny = np.finfo(np.float64).tiny
    return np.sum(left * (np.log(np.clip(left, tiny, 1.0)) - np.log(np.clip(right, tiny, 1.0))), axis=1)


def example_importance(
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
    *,
    mode: ScoreMode = "combined",
    normalize_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return weights, normalized teacher entropy, confidence, and disagreement.

    Mean normalization makes scores from different modes comparable without
    changing the layer ranking within a mode.
    """

    teacher = _validate_probabilities(teacher_probabilities, "teacher_probabilities")
    student = _validate_probabilities(student_probabilities, "student_probabilities")
    if teacher.shape != student.shape:
        raise ValueError("teacher and student probabilities must have the same shape")

    uncertainty = entropy(teacher, normalized=True)
    confidence = np.clip(1.0 - uncertainty, 0.0, 1.0)
    disagreement = kl_divergence(teacher, student)
    modes = {
        "student_only": np.ones(teacher.shape[0]),
        "teacher_confidence": confidence,
        "teacher_uncertainty": uncertainty,
        "disagreement": disagreement,
        "combined": confidence * disagreement,
    }
    if mode not in modes:
        raise ValueError(f"unknown score mode {mode!r}; choose from {sorted(modes)}")
    weights = modes[mode].astype(np.float64, copy=True)
    if normalize_weights:
        mean = float(weights.mean())
        if mean > 0:
            weights /= mean
    return weights, uncertainty, confidence, disagreement


def compute_importance(
    teacher_probabilities: np.ndarray,
    student_probabilities: np.ndarray,
    quantized_probabilities: np.ndarray,
    *,
    mode: ScoreMode = "combined",
) -> ImportanceResult:
    """Compute teacher-aware importance for every temporarily quantized layer.

    ``quantized_probabilities`` must have shape ``[layers, examples, classes]``.
    """

    student = _validate_probabilities(student_probabilities, "student_probabilities")
    quantized = np.asarray(quantized_probabilities, dtype=np.float64)
    if quantized.ndim != 3 or quantized.shape[1:] != student.shape:
        raise ValueError("quantized_probabilities must have shape [layers, examples, classes]")
    weights, uncertainty, confidence, disagreement = example_importance(
        teacher_probabilities, student, mode=mode
    )
    perturbations = np.stack([kl_divergence(student, layer_probs) for layer_probs in quantized])
    layer_importance = np.mean(perturbations * weights[None, :], axis=1)
    return ImportanceResult(weights, uncertainty, confidence, disagreement, layer_importance)
