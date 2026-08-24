"""Task, calibration, error-detection, and selective-prediction metrics."""

from __future__ import annotations

import numpy as np

from .scoring import entropy, normalize_probabilities


def _inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = normalize_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (probs.shape[0],) or np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("labels must be a valid vector aligned with probabilities")
    return probs, labels


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    probs, labels = _inputs(probabilities, labels)
    if bins < 1:
        raise ValueError("bins must be positive")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            mask |= confidence == 0
        if mask.any():
            result += mask.mean() * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(result)


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape or np.any(~np.isfinite(scores)):
        raise ValueError("scores and targets must be aligned finite vectors")
    positives = int(targets.sum())
    negatives = targets.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[targets].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape or np.any(~np.isfinite(scores)):
        raise ValueError("scores and targets must be aligned finite vectors")
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_targets = scores[order], targets[order]
    true_positives = false_positives = 0
    previous_recall = result = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_targets[start:end]
        true_positives += int(group.sum())
        false_positives += int(group.size - group.sum())
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(result)


def area_under_risk_coverage(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _inputs(probabilities, labels)
    errors = (probs.argmax(axis=1) != labels).astype(np.float64)
    order = np.argsort(entropy(probs, normalized=True), kind="stable")
    return float((np.cumsum(errors[order]) / np.arange(1, errors.size + 1)).mean())
