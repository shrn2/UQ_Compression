"""Probability-distribution helpers used by the supported metrics."""

from __future__ import annotations

import numpy as np


def normalize_probabilities(probabilities: np.ndarray, name: str = "probabilities") -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{name} must have shape [examples, classes]")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"each row of {name} must have positive mass")
    return values / totals


def entropy(probabilities: np.ndarray, *, normalized: bool = False) -> np.ndarray:
    probs = normalize_probabilities(probabilities)
    safe = np.clip(probs, np.finfo(np.float64).tiny, 1.0)
    result = -np.sum(probs * np.log(safe), axis=1)
    if normalized:
        support = np.maximum(np.count_nonzero(probs, axis=1), 2)
        result = result / np.log(support)
    return result
