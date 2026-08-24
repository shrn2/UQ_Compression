"""Downstream uncertainty and paired-comparison utilities."""

from __future__ import annotations

import warnings

import numpy as np

from .metrics import binary_average_precision, binary_auroc


DEFERRAL_RATES = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    1.0,
)


def _metric_or_nan(metric, scores: np.ndarray, targets: np.ndarray, name: str) -> float:
    if np.all(targets) or not np.any(targets):
        warnings.warn(f"{name} is undefined for a one-class error target", RuntimeWarning)
        return float("nan")
    return float(metric(scores, targets))


def error_detection_metrics(
    correctness: np.ndarray,
    entropy: np.ndarray,
    max_probability: np.ndarray,
    margin: np.ndarray,
) -> dict[str, float]:
    correct = np.asarray(correctness, dtype=bool)
    entropy_values = np.asarray(entropy, dtype=np.float64)
    max_probability = np.asarray(max_probability, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    if any(
        value.ndim != 1 or value.shape != correct.shape or np.any(~np.isfinite(value))
        for value in (entropy_values, max_probability, margin)
    ):
        raise ValueError("error-detection arrays must be aligned finite vectors")
    targets = ~correct
    return {
        "error_rate": float(targets.mean()),
        "entropy_auroc": _metric_or_nan(binary_auroc, entropy_values, targets, "AUROC"),
        "entropy_auprc": _metric_or_nan(
            binary_average_precision, entropy_values, targets, "AUPRC"
        ),
        "maxprob_auroc": _metric_or_nan(
            binary_auroc, 1.0 - max_probability, targets, "max-probability AUROC"
        ),
        "maxprob_auprc": _metric_or_nan(
            binary_average_precision,
            1.0 - max_probability,
            targets,
            "max-probability AUPRC",
        ),
        "margin_auroc": _metric_or_nan(binary_auroc, -margin, targets, "margin AUROC"),
        "margin_auprc": _metric_or_nan(
            binary_average_precision, -margin, targets, "margin AUPRC"
        ),
    }


def cascade_curve(
    compressed_predictions: np.ndarray,
    dense_predictions: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    deferral_rates=DEFERRAL_RATES,
) -> list[dict[str, float]]:
    compressed = np.asarray(compressed_predictions)
    dense = np.asarray(dense_predictions)
    labels = np.asarray(labels)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if any(value.ndim != 1 or value.shape != labels.shape for value in (compressed, dense, uncertainty)):
        raise ValueError("cascade inputs must be aligned vectors")
    if labels.size == 0 or np.any(~np.isfinite(uncertainty)):
        raise ValueError("cascade inputs must be non-empty and finite")
    order = np.argsort(-uncertainty, kind="stable")
    compressed_correct = compressed == labels
    dense_correct = dense == labels
    compressed_accuracy = float(compressed_correct.mean())
    dense_accuracy = float(dense_correct.mean())
    recoverable = dense_accuracy - compressed_accuracy
    total_errors = int((~compressed_correct).sum())
    rows = []
    for rate in (float(value) for value in deferral_rates):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("deferral rates must lie in [0, 1]")
        count = int(round(rate * labels.size))
        deferred = order[:count]
        final_correct = compressed_correct.copy()
        final_correct[deferred] = dense_correct[deferred]
        accuracy = float(final_correct.mean())
        captured = int((~compressed_correct[deferred]).sum())
        rows.append(
            {
                "deferral_rate": rate,
                "dense_call_rate": rate,
                "accuracy": accuracy,
                "standalone_compressed_accuracy": compressed_accuracy,
                "dense_accuracy": dense_accuracy,
                "recovery": (
                    float((accuracy - compressed_accuracy) / recoverable)
                    if recoverable
                    else float("nan")
                ),
                "error_capture": float(captured / total_errors) if total_errors else float("nan"),
                "deferred_examples": float(count),
                "normalized_cost_proxy": 1.0 + rate,
            }
        )
    return rows


def summarize_bootstrap(values: np.ndarray, *, point: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "difference": float(point),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
        "replicates": int(values.size),
    }


def _row_auroc(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, axis=1, kind="stable")
    sorted_targets = np.take_along_axis(targets, order, axis=1).astype(np.float64)
    ranks = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    positives = sorted_targets.sum(axis=1)
    negatives = scores.shape[1] - positives
    numerator = np.sum(sorted_targets * ranks, axis=1) - positives * (positives + 1.0) / 2.0
    result = np.full(positives.shape, np.nan, dtype=np.float64)
    np.divide(numerator, positives * negatives, out=result, where=(positives > 0) & (negatives > 0))
    return result


def _row_auprc(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_targets = np.take_along_axis(targets, order, axis=1).astype(np.float64)
    cumulative = np.cumsum(sorted_targets, axis=1)
    positions = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    positives = sorted_targets.sum(axis=1)
    result = np.full(positives.shape, np.nan, dtype=np.float64)
    np.divide(
        np.sum((cumulative / positions) * sorted_targets, axis=1),
        positives,
        out=result,
        where=positives > 0,
    )
    return result


def paired_bootstrap_binary_ranking_difference(
    left_correct: np.ndarray,
    left_score: np.ndarray,
    right_correct: np.ndarray,
    right_score: np.ndarray,
    *,
    reps: int,
    seed: int,
    metric: str,
    chunk_size: int = 128,
) -> np.ndarray:
    left_correct = np.asarray(left_correct, dtype=bool)
    right_correct = np.asarray(right_correct, dtype=bool)
    left_score = np.asarray(left_score, dtype=np.float64)
    right_score = np.asarray(right_score, dtype=np.float64)
    if (
        left_correct.ndim != 1
        or left_correct.shape != right_correct.shape
        or left_score.shape != left_correct.shape
        or right_score.shape != left_correct.shape
    ):
        raise ValueError("bootstrap inputs must be aligned vectors")
    if metric not in {"error_auroc", "error_auprc"}:
        raise ValueError(metric)
    if reps < 1 or chunk_size < 1:
        raise ValueError("reps and chunk_size must be positive")
    row_metric = _row_auroc if metric == "error_auroc" else _row_auprc
    rng = np.random.default_rng(seed)
    result = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, chunk_size):
        stop = min(start + chunk_size, reps)
        samples = rng.integers(0, left_correct.size, size=(stop - start, left_correct.size))
        result[start:stop] = row_metric(
            left_score[samples], ~left_correct[samples]
        ) - row_metric(right_score[samples], ~right_correct[samples])
    return result
