"""Dependency-light task, calibration, and uncertainty-preservation metrics."""

from __future__ import annotations

import numpy as np

from .scoring import entropy


def _inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or labels.shape != (probs.shape[0],):
        raise ValueError("probabilities must be [examples, classes] and labels [examples]")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("labels are outside the class range")
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(probs)) or np.any(probs < 0) or np.any(totals <= 0):
        raise ValueError("probabilities must be finite/non-negative with positive row sums")
    return probs / totals, labels


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    probs, labels = _inputs(probabilities, labels)
    if bins < 1:
        raise ValueError("bins must be positive")
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            mask |= confidence == 0
        if mask.any():
            result += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _inputs(probabilities, labels)
    targets = np.zeros_like(probs)
    targets[np.arange(labels.size), labels] = 1.0
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))


def binary_auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    """AUROC with average ranks for ties; targets must contain both classes."""

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape or np.any(~np.isfinite(scores)):
        raise ValueError("scores and targets must be same-length finite vectors")
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
    """Threshold-based average precision with deterministic tie handling."""

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape or np.any(~np.isfinite(scores)):
        raise ValueError("scores and targets must be same-length finite vectors")
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    result = 0.0
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


def margin_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """Return 1 minus the top-1/top-2 probability margin."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape [examples, classes]")
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(probs)) or np.any(probs < 0) or np.any(totals <= 0):
        raise ValueError("probabilities must be finite/non-negative with positive row sums")
    probs = probs / totals
    top_two = np.partition(probs, -2, axis=1)[:, -2:]
    return 1.0 - (top_two[:, 1] - top_two[:, 0])


def pairwise_ordering_flip_rate(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Fraction of non-tied reference pairs whose uncertainty ordering changes."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate must be same-length vectors")
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(candidate)):
        raise ValueError("reference and candidate must be finite")
    left, right = np.triu_indices(reference.size, k=1)
    reference_sign = np.sign(reference[left] - reference[right])
    candidate_sign = np.sign(candidate[left] - candidate[right])
    eligible = reference_sign != 0
    if not eligible.any():
        return float("nan")
    return float(np.mean(reference_sign[eligible] != candidate_sign[eligible]))


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("inputs must be same-length vectors")

    def rank(values: np.ndarray) -> np.ndarray:
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        average_ranks = starts + (counts - 1) / 2.0
        return average_ranks[inverse]

    a, b = rank(left), rank(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def area_under_risk_coverage(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _inputs(probabilities, labels)
    errors = (probs.argmax(axis=1) != labels).astype(np.float64)
    uncertainty = entropy(probs, normalized=True)
    order = np.argsort(uncertainty, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, errors.size + 1)
    return float(cumulative_risk.mean())


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 15) -> dict[str, float]:
    probs, labels = _inputs(probabilities, labels)
    uncertainty = entropy(probs, normalized=True)
    errors = probs.argmax(axis=1) != labels
    return {
        "accuracy": float((~errors).mean()),
        "ece": expected_calibration_error(probs, labels, bins=bins),
        "brier": brier_score(probs, labels),
        "error_auroc": binary_auroc(uncertainty, errors),
        "error_auprc": binary_average_precision(uncertainty, errors),
        "aurc": area_under_risk_coverage(probs, labels),
    }


def uncertainty_preservation(
    quantized_probabilities: np.ndarray,
    full_precision_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    quantized = entropy(quantized_probabilities, normalized=True)
    full = entropy(full_precision_probabilities, normalized=True)
    result = {
        "mean_absolute_uncertainty_distortion": float(np.mean(np.abs(quantized - full))),
        "student_uncertainty_spearman": spearman_correlation(quantized, full),
    }
    if teacher_probabilities is not None:
        teacher = entropy(teacher_probabilities, normalized=True)
        result["teacher_quantized_uncertainty_spearman"] = spearman_correlation(teacher, quantized)
        result["teacher_full_precision_uncertainty_spearman"] = spearman_correlation(teacher, full)
    return result
